"""Historical ST snapshot validation and time-safe signal screening.

Snapshots are full-market risk-warning lists BY TRADING DATE. Absence of a
stock implies NORMAL *only* when that date's complete snapshot is certified.
Never join current stock_info.st onto a historical signal date.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


@dataclass
class STFilterResult:
    signals: pd.DataFrame
    stats: dict
    per_day: pd.DataFrame


def _dates(values: pd.Series) -> pd.Series:
    data = values.astype("string").str.strip()
    # Support ISO YYYY-MM-DD and Tushare YYYYMMDD; do not guess D/M/Y.
    data = data.str.replace(
        r"^([0-9]{4})([0-9]{2})([0-9]{2})$", r"\1-\2-\3", regex=True
    )
    if data.isna().any() or not data.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("Dates must be YYYY-MM-DD or YYYYMMDD")
    result = pd.to_datetime(data, format="%Y-%m-%d", errors="raise")
    return result.dt.normalize()


def _codes(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip().str.upper()
    # Accept Tushare ts_code: 000001.SZ; normalize to stock_code=000001.
    ok = result.str.fullmatch(r"[0-9]{6}(?:\.(?:SZ|SH|BJ))?")
    if not ok.fillna(False).all():
        raise ValueError("stock_code must be six digits or six digits plus .SZ/.SH/.BJ")
    return result.str[:6]


def _complete(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not raw.isin(list(mapping)).all():
        raise ValueError("complete must be true/false or 1/0")
    return raw.map(mapping).astype(bool)


def load_st_snapshots(path: str | Path) -> pd.DataFrame:
    """Load risk-listed securities only; missing dates are not zero-risk."""
    df = pd.read_csv(path, dtype={"stock_code": "string", "ts_code": "string"})
    if "stock_code" not in df and "ts_code" in df:
        df = df.rename(columns={"ts_code": "stock_code"})
    needed = {"stock_code", "trade_date"}
    if not needed.issubset(df.columns):
        raise ValueError(f"ST snapshots missing {sorted(needed-set(df.columns))}")
    df["stock_code"] = _codes(df["stock_code"])
    df["trade_date"] = _dates(df["trade_date"])
    if df.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate ST snapshot stock_code/trade_date")
    return df


def load_st_coverage(path: str | Path) -> pd.DataFrame:
    """Completeness certificate for each trading day's full ST snapshot."""
    df = pd.read_csv(path, dtype={"source": "string"})
    needed = {"trade_date", "complete", "st_count", "source"}
    if not needed.issubset(df.columns):
        raise ValueError(f"ST coverage missing {sorted(needed-set(df.columns))}")
    df["trade_date"] = _dates(df["trade_date"])
    df["complete"] = _complete(df["complete"])
    count = pd.to_numeric(df["st_count"], errors="coerce")
    if (count.isna() | (count < 0) | (count % 1 != 0)).any():
        raise ValueError("st_count must be nonnegative integer")
    df["st_count"] = count.astype(int)
    if df["source"].isna().any() or (df["source"].str.strip() == "").any():
        raise ValueError("Every coverage row needs a nonblank source")
    if df.trade_date.duplicated().any():
        raise ValueError("Duplicate ST coverage trade_date")
    return df


def filter_signals_by_historical_st(
    signals: pd.DataFrame,
    snapshots: pd.DataFrame,
    coverage: pd.DataFrame,
) -> STFilterResult:
    """Filter on signal date T, using full ST snapshots *for T only*.

    A missing or explicitly incomplete snapshot on T is UNKNOWN and fails
    closed (all T signals blocked). This is NOT an order-day T+1 or delisting
    risk check. Historical vendor snapshots can be revised retrospectively;
    keep source/provenance and cross-check exchange announcements.
    """
    if not {"stock_code", "signal_date", "score"}.issubset(signals.columns):
        raise ValueError("Signals require stock_code, signal_date, score")
    if not {"stock_code", "trade_date"}.issubset(snapshots.columns):
        raise ValueError("ST snapshots require stock_code and trade_date")
    if not {"trade_date", "complete", "st_count", "source"}.issubset(coverage.columns):
        raise ValueError("ST coverage requires trade_date, complete, st_count, source")
    sig = signals.copy().reset_index(drop=True)
    st = snapshots.copy().reset_index(drop=True)
    cov = coverage.copy().reset_index(drop=True)
    sig["stock_code"] = _codes(sig["stock_code"])
    sig["signal_date"] = _dates(sig["signal_date"])
    st["stock_code"] = _codes(st["stock_code"])
    st["trade_date"] = _dates(st["trade_date"])
    cov["trade_date"] = _dates(cov["trade_date"])
    cov["complete"] = _complete(cov["complete"])
    st_count = pd.to_numeric(cov["st_count"], errors="coerce")
    if (st_count.isna() | (st_count < 0) | (st_count % 1 != 0)).any():
        raise ValueError("Invalid st_count in ST coverage")
    cov["st_count"] = st_count.astype(int)
    if cov.trade_date.duplicated().any() or st.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate ST date or stock/day")
    actual_counts = st.groupby("trade_date").size()
    observed = cov.trade_date.map(actual_counts).fillna(0).astype(int)
    mismatched = cov.complete & (cov.st_count != observed)
    if mismatched.any():
        bad_dates = cov.loc[mismatched, "trade_date"].dt.date.astype(str).tolist()[:8]
        raise ValueError(f"Complete ST snapshots disagree with st_count on {bad_dates}")
    if (~st.trade_date.isin(cov.trade_date)).any():
        raise ValueError("ST snapshots contain dates absent from coverage file")

    day_complete = sig.signal_date.map(cov.set_index("trade_date")["complete"])
    known = day_complete.fillna(False).astype(bool).to_numpy()
    keys = pd.MultiIndex.from_arrays([sig.stock_code, sig.signal_date])
    risk_keys = pd.MultiIndex.from_frame(st[["stock_code", "trade_date"]])
    flagged = keys.isin(risk_keys)
    eligible = known & ~flagged
    statuses = pd.Series(
        pd.Categorical.from_codes(
            (2 * (~known).astype(int) + (known & flagged).astype(int)),
            categories=["NORMAL", "ST", "UNKNOWN", "UNUSED"],
        )
    )
    # codes: known normal=0, known ST=1, unknown=2.
    sig_date = sig.signal_date
    per_day = (pd.DataFrame({"trade_date": sig_date, "status": statuses})
               .groupby(["trade_date", "status"], observed=True)
               .size().unstack(fill_value=0)
               .reindex(columns=["NORMAL", "ST", "UNKNOWN"], fill_value=0)
               .reset_index())
    matched_days = int(sig_date.nunique())
    complete_days = int(sig_date[known].nunique())
    stats = {
        "signal_rows": len(sig),
        "eligible_normal_rows": int(eligible.sum()),
        "blocked_st_rows": int((known & flagged).sum()),
        "blocked_unknown_rows": int((~known).sum()),
        "signal_days": matched_days,
        "complete_signal_days": complete_days,
        "complete_signal_day_pct": (
            round(100 * complete_days / matched_days, 2) if matched_days else 0.0),
        "coverage_rule": "UNKNOWN_FAIL_CLOSED",
        "risk_check_at": "T_SIGNAL_DATE",
        "order_day_st_check": "NOT_MODELED",
        "warning": (
            "Historical ST snapshot screening only; company actions, delisting, "
            "T+1 order-day announcements and auction availability remain unverified."
        ),
    }
    return STFilterResult(sig.loc[eligible].reset_index(drop=True), stats, per_day)
