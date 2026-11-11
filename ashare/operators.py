"""Causal masked operators for daily A-share formulas.

Every operand is (values, valid_mask) with shape [stocks, dates].
All time-series windows require a complete sequence of observed trading dates:
no forward filling, zero padding, or borrowing data across missing sessions.
"""
from __future__ import annotations

import torch

EPS = 1e-8


def _unary(fn):
    def apply(a):
        x, m = a
        return fn(x), m
    return apply


def _binary(fn):
    def apply(a, b):
        x, ma = a
        y, mb = b
        return fn(x, y), ma & mb
    return apply


def _divide(a, b):
    x, ma = a
    y, mb = b
    valid = ma & mb & (y.abs() > EPS)
    return x / torch.where(valid, y, torch.ones_like(y)), valid


def _gate(condition, a, b):
    c, mc = condition
    x, mx = a
    y, my = b
    pick_x = c > 0
    return torch.where(pick_x, x, y), mc & torch.where(pick_x, mx, my)


def _delay(a, lag=1):
    x, m = a
    out = torch.zeros_like(x)
    valid = torch.zeros_like(m)
    if x.shape[1] > lag:
        out[:, lag:] = x[:, :-lag]
        valid[:, lag:] = m[:, :-lag]
    return out, valid


def _rolling(a, window, mode):
    x, m = a
    if x.shape[1] < window:
        return torch.zeros_like(x), torch.zeros_like(m)
    windows = x.unfold(1, window, 1)
    masks = m.unfold(1, window, 1)
    ready = masks.all(-1)
    if mode == "mean":
        values = windows.mean(-1)
    elif mode == "std":
        values = windows.std(-1, unbiased=False)
    elif mode == "max":
        values = windows.max(-1).values
    elif mode == "min":
        values = windows.min(-1).values
    elif mode == "rank":
        latest = windows[..., -1:]
        lower = (windows < latest).sum(-1)
        equal = (windows == latest).sum(-1)
        values = (lower + (equal - 1) / 2) / (window - 1)
    else:
        raise ValueError(f"Unsupported rolling mode {mode}")
    out = torch.zeros_like(x)
    valid = torch.zeros_like(m)
    out[:, window-1:] = values
    valid[:, window-1:] = ready
    return out, valid


def _delta5(a):
    x, m = a
    delayed, delayed_valid = _delay(a, 5)
    return x - delayed, m & delayed_valid


def _cs_rank(a):
    """Same-date cross-sectional percentile, with average ranks for ties."""
    x, m = a
    if x.shape[0] < 2:
        return torch.zeros_like(x), torch.zeros_like(m)
    masked = torch.where(m, x, torch.full_like(x, float("inf")))
    sorted_values, indices = masked.sort(dim=0, stable=True)
    n_stocks, n_dates = x.shape
    pos = torch.arange(1, n_stocks+1, device=x.device).unsqueeze(1).expand(-1, n_dates)
    starts = torch.ones_like(m)
    starts[1:] = sorted_values[1:] != sorted_values[:-1]
    begin = torch.cummax(torch.where(starts, pos, 0), dim=0).values
    ends = torch.ones_like(m)
    ends[:-1] = sorted_values[:-1] != sorted_values[1:]
    finish = torch.flip(torch.cummin(torch.flip(torch.where(ends, pos, n_stocks+1), [0]), dim=0).values, [0])
    count = m.sum(dim=0, keepdim=True)
    ranked_sorted = ((begin + finish) / 2 - 1) / (count - 1).clamp(min=1)
    out = torch.zeros_like(x)
    out.scatter_(0, indices, ranked_sorted.to(dtype=x.dtype))
    valid = m & (count >= 2)
    return out, valid


# Shared StackVM consumes tuples (operator_name, masked_callback, arity).
ASHARE_OPS = (
    ("ADD", _binary(torch.add), 2),
    ("SUB", _binary(torch.sub), 2),
    ("MUL", _binary(torch.mul), 2),
    ("DIV", _divide, 2),
    ("NEG", _unary(torch.neg), 1),
    ("ABS", _unary(torch.abs), 1),
    ("SIGN", _unary(torch.sign), 1),
    ("GATE", _gate, 3),
    ("DELAY1", lambda a: _delay(a, 1), 1),
    ("TS_DELTA5", _delta5, 1),
    ("TS_MEAN5", lambda a: _rolling(a, 5, "mean"), 1),
    ("TS_MEAN20", lambda a: _rolling(a, 20, "mean"), 1),
    ("TS_STD20", lambda a: _rolling(a, 20, "std"), 1),
    ("TS_MAX20", lambda a: _rolling(a, 20, "max"), 1),
    ("TS_MIN20", lambda a: _rolling(a, 20, "min"), 1),
    ("TS_RANK20", lambda a: _rolling(a, 20, "rank"), 1),
    ("CS_RANK", _cs_rank, 1),
)
