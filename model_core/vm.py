"""Postfix formula VM, with opt-in validity-mask execution for research.

The original StackVM() API and unmasked crypto path are preserved.
"""
import torch
from .ops import OPS_CONFIG
from .vocab import FORMULA_VOCAB


class StackVM:
    def __init__(self, feature_names=None, ops_config=None):
        if feature_names is None:
            feature_names = FORMULA_VOCAB.feature_names
        if ops_config is None:
            ops_config = OPS_CONFIG
        self.feature_names = tuple(feature_names)
        self.ops_config = tuple(ops_config)
        self.feat_offset = len(self.feature_names)
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(self.ops_config)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(self.ops_config)}

    def execute(self, formula_tokens, feat_tensor):
        """Legacy best-effort execution (retained for the crypto strategy)."""
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if 0 <= token < self.feat_offset:
                    if token >= feat_tensor.shape[1]:
                        return None
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity:
                        return None
                    args = [stack.pop() for _ in range(arity)][::-1]
                    res = self.op_map[token](*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    return None
            return stack[0] if len(stack) == 1 else None
        except Exception:
            return None

    def execute_masked(self, formula_tokens, feat_tensor, feature_valid):
        """Strict execution: callbacks consume (values, validity) pairs.

        Returns (values, mask) of shape [stock, date]. Invalid values are
        NaN rather than silently converted to zero. Fails closed on bad tokens.
        This mode requires operator callbacks implementing the masked contract.
        """
        if feat_tensor.ndim != 3 or feature_valid.shape != feat_tensor.shape:
            raise ValueError("Expected matching [stock, factor, date] tensors")
        if feat_tensor.shape[1] != self.feat_offset:
            raise ValueError("Feature dimension does not match formula vocabulary")
        if feature_valid.dtype != torch.bool:
            raise ValueError("feature_valid must be a bool tensor")
        stack = []
        for token in formula_tokens:
            if isinstance(token, bool) or not isinstance(token, int):
                raise ValueError("Formula tokens must be integer IDs")
            if 0 <= token < self.feat_offset:
                v = feat_tensor[:, token, :]
                m = feature_valid[:, token, :] & torch.isfinite(v)
                stack.append((torch.where(m, v, torch.zeros_like(v)), m))
            elif token in self.op_map:
                arity = self.arity_map[token]
                if len(stack) < arity:
                    raise ValueError("Malformed formula: stack underflow")
                args = [stack.pop() for _ in range(arity)][::-1]
                v, m = self.op_map[token](*args)
                if v.shape != feat_tensor[:, 0, :].shape or m.shape != v.shape:
                    raise ValueError("Operator returned incorrect shape")
                good = m & torch.isfinite(v)
                stack.append((torch.where(good, v, torch.zeros_like(v)), good))
            else:
                raise ValueError(f"Unknown formula token {token}")
        if len(stack) != 1:
            raise ValueError("Malformed formula: expected exactly one result")
        v, m = stack[0]
        return v.masked_fill(~m, float("nan")), m
