"""Safe postfix formula grammar and deterministic random-baseline generator."""
from __future__ import annotations

import random

from model_core.vm import StackVM

from .daily import DAILY_COLUMNS
from .operators import ASHARE_OPS
from .tensor import DailyFactorTensor


class AShareFormulaEngine:
    def __init__(self, feature_names=DAILY_COLUMNS, ops=ASHARE_OPS, max_tokens=12):
        self.features = tuple(feature_names)
        self.ops = tuple(ops)
        self.names = self.features + tuple(op[0] for op in self.ops)
        if len(self.names) != len(set(self.names)):
            raise ValueError("Duplicate formula token names")
        self.lookup = {name: i for i, name in enumerate(self.names)}
        self.vm = StackVM(feature_names=self.features, ops_config=self.ops)
        self.max_tokens = max_tokens

    def parse(self, expression: str):
        """Parse a whitespace-separated RPN expression; e.g. 'ret_1d holding ADD'."""
        words = expression.split()
        try:
            tokens = tuple(self.lookup[word] for word in words)
        except KeyError as exc:
            raise ValueError(f"Unknown formula token: {exc.args[0]}") from exc
        self.validate(tokens)
        return tokens

    def validate(self, tokens):
        if not 1 <= len(tokens) <= self.max_tokens:
            raise ValueError("Formula length out of bounds")
        depth = 0
        for token in tokens:
            if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < len(self.names):
                raise ValueError("Invalid formula token")
            if token < len(self.features):
                depth += 1
            else:
                arity = self.ops[token-len(self.features)][2]
                if depth < arity:
                    raise ValueError("Malformed formula: stack underflow")
                depth = depth - arity + 1
        if depth != 1:
            raise ValueError("Malformed formula: leftover stack operands")
        return True

    def pretty(self, tokens):
        self.validate(tokens)
        stack = []
        for token in tokens:
            if token < len(self.features):
                stack.append(self.names[token])
            else:
                name, _, arity = self.ops[token-len(self.features)]
                args = [stack.pop() for _ in range(arity)][::-1]
                stack.append(f"{name}({', '.join(args)})")
        return stack.pop()

    def sample(self, rng: random.Random | None = None):
        """Generate only valid RPN expression trees with <= max_tokens nodes."""
        rng = rng or random.Random()
        def node(budget, depth=0):
            available = [(name, arity) for name, _, arity in self.ops
                         if arity + 1 <= budget]
            if not available or (depth and rng.random() < 0.35):
                return [self.lookup[rng.choice(self.features)]]
            name, arity = rng.choice(available)
            remaining = budget - 1
            result = []
            for k in range(arity):
                left_children = arity - k - 1
                this_budget = (remaining if not left_children
                               else rng.randint(1, remaining-left_children))
                result.extend(node(this_budget, depth+1))
                remaining -= this_budget
            result.append(self.lookup[name])
            return result
        tokens = tuple(node(self.max_tokens))
        self.validate(tokens)
        return tokens

    def execute(self, tokens, data: DailyFactorTensor):
        self.validate(tokens)
        if data.feature_names != self.features:
            raise ValueError("Tensor feature vocabulary mismatch")
        return self.vm.execute_masked(tokens, data.features, data.feature_valid)
