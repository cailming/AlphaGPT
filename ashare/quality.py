"""Read-only factor coverage and historical main-force provenance diagnostics.

Timestamp eligibility matches build_daily_tensor (created before end of T).
It is NOT proof that the underlying formula was computed point-in-time.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd
from .daily import DAILY_COLUMNS
from .tensor import FORCE_FEATURES

FACTOR_BOUNDS = {
    "ret_1d": (-1.0, None), "ret_5d": (-1.0, None),
    "vol_ratio_20": (0.0, None), "ma20_dev": (-1.0, None),
    "breakout_20": (-1.0, None),
    "main_force": (0.0, 1.0), "activity": (0.0, 1.0),
    "strength": (0.0, 1.0), "holding": (0.0, 1.0),
    "distribution_risk": (0.0, 1.0),
    "amount_ratio_5_20": (0.0, None),
    "drawdown_20": (-1.0, 0.0),
}

@dataclass
class QualityReport:
    summary: pd.DataFrame
    monthly: pd.DataFrame
    backfill_batches: pd.DataFrame
    audit: dict

def _percent(a: int, b: int) -> float:
    return round(100.0 * int(a) / int(b), 2) if b else 0.0

def analyze_factor_quality(frame: pd.DataFrame, *, start: date | None = None,
                           end: date | None = None) -> QualityReport:
    """Diagnose 12 factors with both observed-market and stock/date-grid denominators.

    Tensor mask = finite + created_at cutoff (for force factors).
    Quality eligibility = tensor mask + numeric bounds. Bounds checks are
    diagnostic; the original model tensor does not enforce them yet.
    """
    needed = {"stock_code", "trade_date", "main_force_created_at", *DAILY_COLUMNS}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"Missing quality fields: {sorted(missing)}")
    df = frame.copy(deep=False)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="raise").dt.normalize()
    if start is not None:
        df = df.loc[df.trade_date >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.trade_date <= pd.Timestamp(end)]
    if df.empty:
        raise ValueError("No market rows in selected date range")
    if df[["stock_code", "trade_date"]].isna().any().any():
        raise ValueError("Null stock_code/trade_date")
    if df.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate stock/date rows; join cardinality may be wrong")

    n_rows = len(df)
    n_stocks = df.stock_code.nunique()
    n_days = df.trade_date.nunique()
    grid_rows = n_stocks * n_days
    month_groups = list(df.groupby(df.trade_date.dt.strftime("%Y-%m"), sort=True).indices.items())
    created = pd.to_datetime(df.main_force_created_at, errors="coerce")
    cutoff = df.trade_date + pd.Timedelta(days=1)
    time_ok = created.notna() & (created < cutoff)
    if "main_force_joined_code" in df:
        joined = df.main_force_joined_code.notna()
    else:
        # Fallback cannot distinguish a joined all-null score row.
        joined = created.notna() | df["main_force"].notna()

    rows, monthly_rows = [], []
    for name in DAILY_COLUMNS:
        numeric = pd.to_numeric(df[name], errors="coerce")
        finite = pd.Series(np.isfinite(numeric.to_numpy(dtype=float)), index=df.index)
        non_null = numeric.notna()
        lower, upper = FACTOR_BOUNDS[name]
        in_bounds = finite.copy()
        if lower is not None:
            in_bounds &= numeric >= lower
        if upper is not None:
            in_bounds &= numeric <= upper
        force = name in FORCE_FEATURES
        tensor_mask = finite & time_ok if force else finite
        quality_mask = in_bounds & time_ok if force else in_bounds
        row = {
            "factor": name, "source": "main_force" if force else "market",
            "market_rows": n_rows, "grid_rows": grid_rows,
            "raw_non_null": int(non_null.sum()),
            "finite_count": int(finite.sum()),
            "out_of_range": int((finite & ~in_bounds).sum()),
            "tensor_mask_count": int(tensor_mask.sum()),
            "quality_eligible_count": int(quality_mask.sum()),
            "raw_non_null_pct_market": _percent(non_null.sum(), n_rows),
            "finite_pct_market": _percent(finite.sum(), n_rows),
            "tensor_mask_pct_market": _percent(tensor_mask.sum(), n_rows),
            "tensor_mask_pct_grid": _percent(tensor_mask.sum(), grid_rows),
            "quality_eligible_pct_market": _percent(quality_mask.sum(), n_rows),
            "quality_eligible_pct_grid": _percent(quality_mask.sum(), grid_rows),
            "eligible_stocks": int(df.loc[quality_mask, "stock_code"].nunique()),
            "eligible_days": int(df.loc[quality_mask, "trade_date"].nunique()),
            "score_missing_or_null": int((joined & ~non_null).sum()) if force else 0,
            "joined_row_missing": int((~joined).sum()) if force else 0,
            "created_at_missing": int((in_bounds & created.isna()).sum()) if force else 0,
            "created_at_late": int((in_bounds & created.notna() & ~time_ok).sum()) if force else 0,
        }
        rows.append(row)
        quality_arr = quality_mask.to_numpy()
        tensor_arr = tensor_mask.to_numpy()
        finite_arr = finite.to_numpy()
        for key, positions in month_groups:
            count = len(positions)
            eligible_n = int(quality_arr[positions].sum())
            tensor_n = int(tensor_arr[positions].sum())
            monthly_rows.append({
                "month": key, "factor": name, "market_rows": count,
                "raw_finite": int(finite_arr[positions].sum()),
                "tensor_mask_count": tensor_n,
                "quality_eligible_count": eligible_n,
                "tensor_mask_pct_market": _percent(tensor_n, count),
                "quality_eligible_pct_market": _percent(eligible_n, count),
            })

    force_df = df.loc[joined, ["stock_code", "trade_date"]].copy()
    force_df["created_at"] = created.loc[joined].values
    force_df["late"] = force_df.created_at >= force_df.trade_date + pd.Timedelta(days=1)
    force_df["before_date"] = force_df.created_at < force_df.trade_date
    if force_df.empty:
        batches = pd.DataFrame(columns=["created_day", "rows", "source_min", "source_max", "late_rows", "late_pct"])
    else:
        force_df["created_day"] = force_df.created_at.dt.date
        group = force_df.dropna(subset=["created_day"]).groupby("created_day")
        batches = group.agg(rows=("stock_code", "size"),
                            source_min=("trade_date", "min"),
                            source_max=("trade_date", "max"),
                            late_rows=("late", "sum")).reset_index()
        batches["late_pct"] = [
            _percent(late, count) for late, count in zip(batches.late_rows, batches.rows)
        ]
        batches = batches.sort_values(["rows", "created_day"], ascending=[False, True])

    audit = {
        "market_rows": int(n_rows), "stock_count": int(n_stocks),
        "date_count": int(n_days), "grid_rows": int(grid_rows),
        "grid_missing_market_rows": int(grid_rows - n_rows),
        "grid_market_coverage_pct": _percent(n_rows, grid_rows),
        "force_joined_rows": int(joined.sum()),
        "force_joined_pct_market": _percent(joined.sum(), n_rows),
        "force_missing_join_rows": int((~joined).sum()),
        "force_created_at_missing_rows": int((joined & created.isna()).sum()),
        "force_created_at_late_rows": int((joined & created.notna() & ~time_ok).sum()),
        "force_created_at_before_trade_date_rows": int((joined & created.notna() & (created < df.trade_date)).sum()),
        "force_timestamp_eligible_rows": int((joined & time_ok).sum()),
        "disclaimer": (
            "created_at is creation/last generation time, not immutable lineage. "
            "Timestamp eligibility is only a necessary screening rule, not proof "
            "of point-in-time causality. Bounds checks are diagnostics; current "
            "tensor.py does not yet enforce them."
        ),
    }
    return QualityReport(pd.DataFrame(rows), pd.DataFrame(monthly_rows), batches, audit)
