"""Market/force data-frame -> aligned Torch tensors without implicit filling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .daily import DAILY_COLUMNS, LABEL_COLUMN

FORCE_FEATURES = frozenset((
    "main_force", "activity", "strength", "holding", "distribution_risk",
    "amount_ratio_5_20", "drawdown_20",
))


@dataclass(frozen=True)
class DailyFactorTensor:
    stock_codes: tuple[str, ...]
    dates: pd.DatetimeIndex
    features: torch.Tensor             # [stocks, factors, dates], invalid values zeroed
    feature_valid: torch.Tensor        # same shape, bool
    target: torch.Tensor               # [stocks, dates], invalid values zeroed
    target_valid: torch.Tensor         # same shape, bool
    feature_names: tuple[str, ...] = DAILY_COLUMNS

    def __post_init__(self):
        s, f, t = self.features.shape
        if ((s, f, t) != (len(self.stock_codes), len(self.feature_names), len(self.dates))
                or self.feature_valid.shape != self.features.shape
                or self.target.shape != (s, t)
                or self.target_valid.shape != (s, t)):
            raise ValueError("Inconsistent factor tensor dimensions")


def build_daily_tensor(frame: pd.DataFrame, *, strict_force_asof: bool = True,
                       device: str = "cpu") -> DailyFactorTensor:
    """Align days/stocks, preserving missing observations as masks.

    Strict force-as-of checks f.created_at <= T at 23:59:59 (research decision
    after T close, before T+1 open). Missing/recomputed/backfilled timestamps
    cannot establish historical availability, so force features are masked.
    """
    needed = {"stock_code", "trade_date", *DAILY_COLUMNS, LABEL_COLUMN}
    if strict_force_asof:
        needed.add("main_force_created_at")
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"Missing factor tensor fields: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot build factor tensor from empty data")
    df = frame.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="raise").dt.normalize()
    if df[["trade_date", "stock_code"]].isna().any().any():
        raise ValueError("Missing stock or trade date")
    df["stock_code"] = df["stock_code"].astype(str)
    if df.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Multiple daily records for one stock/date")
    codes = tuple(sorted(df["stock_code"].unique()))
    dates = pd.DatetimeIndex(sorted(df["trade_date"].unique()))
    idx = pd.MultiIndex.from_product((codes, dates), names=["stock_code", "trade_date"])
    aligned = df.set_index(["stock_code", "trade_date"]).reindex(idx)
    n, days = len(codes), len(dates)
    feats = np.stack([pd.to_numeric(aligned[k], errors="coerce").to_numpy(dtype=np.float64)
                      for k in DAILY_COLUMNS], axis=1).reshape(n, days, len(DAILY_COLUMNS))
    feats = feats.transpose(0, 2, 1).copy()
    feature_ok = np.isfinite(feats)
    if strict_force_asof:
        when = pd.to_datetime(aligned["main_force_created_at"], errors="coerce")
        decision_cutoff = (aligned.index.get_level_values("trade_date")
                           + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1))
        available = np.asarray(when.notna() & (when.to_numpy() <= decision_cutoff.to_numpy()))
        available = available.reshape(n, days)
        for name in FORCE_FEATURES:
            feature_ok[:, DAILY_COLUMNS.index(name), :] &= available
    label = pd.to_numeric(aligned[LABEL_COLUMN], errors="coerce").to_numpy(dtype=np.float64).reshape(n, days)
    label_ok = np.isfinite(label)
    # No NaN reaches the operator math; masks control all arithmetic.
    feat_tensor = torch.as_tensor(np.where(feature_ok, feats, 0.0), dtype=torch.float32, device=device)
    label_tensor = torch.as_tensor(np.where(label_ok, label, 0.0), dtype=torch.float32, device=device)
    return DailyFactorTensor(codes, dates, feat_tensor,
                             torch.as_tensor(feature_ok.copy(), dtype=torch.bool, device=device),
                             label_tensor,
                             torch.as_tensor(label_ok.copy(), dtype=torch.bool, device=device))
