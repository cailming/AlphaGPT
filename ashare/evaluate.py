"""Read-only factor ranking diagnostic; NOT an executable trading backtest."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import random

import numpy as np
import pandas as pd

from .daily import AShareDailyLoader
from .formula import AShareFormulaEngine
from .tensor import DailyFactorTensor, build_daily_tensor


@dataclass(frozen=True)
class FactorIC:
    dates: int
    mean_ic: float
    icir: float
    positive_share: float
    observations: int


def cross_sectional_ic(values, valid, data: DailyFactorTensor, *,
                       allowed_dates=None, min_stocks=15) -> FactorIC:
    """Datewise Spearman(rank(factor), rank(T+1-open-to-T+2-open label))."""
    if values.shape != data.target.shape or valid.shape != values.shape:
        raise ValueError("Factor result/target shape mismatch")
    selection = set(pd.to_datetime(allowed_dates)) if allowed_dates is not None else None
    xs = values.detach().cpu().numpy()
    mask = (valid & data.target_valid).detach().cpu().numpy()
    ys = data.target.detach().cpu().numpy()
    daily_ic = []
    observations = 0
    for col, day in enumerate(data.dates):
        if selection is not None and day not in selection:
            continue
        keep = mask[:, col] & np.isfinite(xs[:, col]) & np.isfinite(ys[:, col])
        if keep.sum() < min_stocks:
            continue
        x = pd.Series(xs[keep, col]).rank().to_numpy()
        y = pd.Series(ys[keep, col]).rank().to_numpy()
        if x.std() == 0 or y.std() == 0:
            continue
        ic = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(ic):
            daily_ic.append(ic)
            observations += int(keep.sum())
    if not daily_ic:
        return FactorIC(0, float("nan"), float("nan"), float("nan"), 0)
    arr = np.asarray(daily_ic)
    std = arr.std(ddof=1) if len(arr) > 1 else float("nan")
    return FactorIC(len(arr), float(arr.mean()),
                    float(arr.mean() / std) if std and np.isfinite(std) else float("nan"),
                    float((arr > 0).mean()), observations)


def search_baseline(data: DailyFactorTensor, *, start: date, end: date,
                    samples=100, seed=42, min_stocks=15, min_train_days=10):
    """Select by train IC only; return holdout test diagnostics (no trade fills)."""
    days = data.dates[(data.dates >= pd.Timestamp(start)) &
                      (data.dates <= pd.Timestamp(end))]
    if len(days) < 12:
        raise ValueError("Insufficient research dates")
    cut = int(len(days) * .7)
    # Purge 2 trade dates at split: labels use the following two opens.
    train_days = days[:max(0, cut - 2)]
    test_days = days[cut:]
    engine = AShareFormulaEngine()
    rng = random.Random(seed)
    candidates = set()
    reports = []
    for _ in range(samples):
        tokens = engine.sample(rng)
        if tokens in candidates:
            continue
        candidates.add(tokens)
        values, valid = engine.execute(tokens, data)
        train = cross_sectional_ic(values, valid, data,
                                   allowed_dates=train_days, min_stocks=min_stocks)
        if train.dates >= min_train_days and np.isfinite(train.mean_ic):
            reports.append((train.mean_ic, tokens, train))
    if not reports:
        return None
    reports.sort(key=lambda x: x[0], reverse=True)
    _, best, training = reports[0]
    values, valid = engine.execute(best, data)
    testing = cross_sectional_ic(values, valid, data,
                                 allowed_dates=test_days, min_stocks=min_stocks)
    return {"tokens": best, "formula": engine.pretty(best),
            "train": training, "test": testing,
            "unique_candidates": len(candidates), "scored_candidates": len(reports)}
