"""Current stock_info ST blacklist for NON-PIT diagnostic comparisons ONLY.

Never interpret the stock_info snapshot as a historical ST timeline.
In strict historical research use st_history.py with T-date snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


NON_PIT_LABEL = "NON_PIT_CURRENT_ST_FILTER"


@dataclass
class CurrentSTResult:
    signals: pd.DataFrame
    audit: pd.DataFrame
    stats: dict


def load_current_stock_info(engine: Engine) -> pd.DataFrame:
    """Read a current stock_info snapshot; never changes the database."""
    query = text("""
        SELECT stock_code, stock_name, st
          FROM public.stock_info
         WHERE stock_code IS NOT NULL
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn)


def filter_signals_by_current_st(
    signals: pd.DataFrame,
    stock_info: pd.DataFrame,
    *,
    snapshot_at_utc: str | None = None,
) -> CurrentSTResult:
    """Exclude today's ST and unknown-status securities from EVERY historical day.

    IMPORTANT: this uses future information when applied to old T dates.
    It is suitable ONLY for sensitivity diagnostics; it is NOT a historical
    eligibility policy. Existing positions are not forcibly sold.
    """
    required_sig = {"stock_code", "signal_date", "score"}
    required_info = {"stock_code", "stock_name", "st"}
    if missing := required_sig - set(signals.columns):
        raise ValueError(f"Missing signals fields: {sorted(missing)}")
    if missing := required_info - set(stock_info.columns):
        raise ValueError(f"Missing stock_info fields: {sorted(missing)}")

    info = stock_info.loc[:, ["stock_code", "stock_name", "st"]].copy()
    if info["stock_code"].isna().any():
        raise ValueError("Null stock_info.stock_code")
    info["stock_code"] = info["stock_code"].astype(str).str.strip()
    if not info["stock_code"].str.fullmatch(r"\d{6}").all():
        raise ValueError("stock_info stock_code must contain six digits")
    if info.stock_code.duplicated().any():
        raise ValueError("Duplicate stock_info.stock_code")

    info["stock_name"] = info["stock_name"].astype("string").str.strip()
    numeric_st = pd.to_numeric(info["st"], errors="coerce")
    from_st = numeric_st.isin([1, 2])
    from_name = info["stock_name"].str.contains(
        "ST", case=False, regex=False, na=False
    ).astype(bool)
    flagged = from_st | from_name
    normal = (
        ~flagged & numeric_st.eq(0)
        & info["stock_name"].notna()
        & info["stock_name"].ne("")
    ).fillna(False).astype(bool)
    info["screen_status"] = np.select(
        [flagged.to_numpy(), normal.to_numpy()],
        ["CURRENT_ST", "CURRENT_NORMAL"],
        default="UNKNOWN",
    )
    info["reason"] = np.select(
        [(from_st & from_name).to_numpy(),
         (from_st & ~from_name).to_numpy(),
         (~from_st & from_name).to_numpy(),
         normal.to_numpy()],
        ["ST_FIELD_AND_NAME", "ST_FIELD", "NAME_CONTAINS_ST", "CURRENT_NORMAL"],
        default="MISSING_OR_UNRECOGNIZED_ST_OR_NAME",
    )

    sig = signals.copy()
    if sig["stock_code"].isna().any():
        raise ValueError("Null signal stock_code")
    sig["stock_code"] = sig["stock_code"].astype(str).str.strip()
    if not sig["stock_code"].str.fullmatch(r"\d{6}").all():
        raise ValueError("Signal stock_code must contain six digits")

    status_lookup = info.set_index("stock_code")["screen_status"]
    signal_status = sig.stock_code.map(status_lookup).fillna("UNKNOWN")
    allowed = signal_status.eq("CURRENT_NORMAL")
    excluded = sig.loc[~allowed, "stock_code"].value_counts()
    audit = info.copy()
    audit["blocked_signal_count"] = (
        audit.stock_code.map(excluded).fillna(0).astype(int)
    )
    missing_codes = sorted(set(sig.stock_code) - set(info.stock_code))
    if missing_codes:
        # Add unmatched codes without assuming they are normal. Reindexing
        # retains the source st column dtype even if all extras are missing.
        audit = audit.set_index("stock_code")
        audit = audit.reindex(audit.index.append(pd.Index(missing_codes)))
        audit.index.name = "stock_code"
        audit.loc[missing_codes, "screen_status"] = "UNKNOWN"
        audit.loc[missing_codes, "reason"] = "MISSING_STOCK_INFO"
        audit.loc[missing_codes, "blocked_signal_count"] = [
            int(excluded.get(code, 0)) for code in missing_codes
        ]
        audit = audit.reset_index()
        audit["blocked_signal_count"] = audit["blocked_signal_count"].astype(int)

    stamp = snapshot_at_utc or datetime.now(timezone.utc).isoformat()
    stats = {
        "status": NON_PIT_LABEL,
        "source": "public.stock_info (CURRENT SNAPSHOT)",
        "snapshot_at_utc": stamp,
        "is_point_in_time": False,
        "lookahead_risk": True,
        "signal_rows_before": int(len(sig)),
        "signal_rows_after": int(allowed.sum()),
        "blocked_current_st_signals": int(signal_status.eq("CURRENT_ST").sum()),
        "blocked_unknown_signals": int(signal_status.eq("UNKNOWN").sum()),
        "blocked_current_st_symbols": int(
            sig.loc[signal_status.eq("CURRENT_ST"), "stock_code"].nunique()
        ),
        "blocked_unknown_symbols": int(
            sig.loc[signal_status.eq("UNKNOWN"), "stock_code"].nunique()
        ),
        "current_info_symbols": int(len(info)),
        "unknown_info_symbols": int(info.screen_status.eq("UNKNOWN").sum()),
        "unmatched_signal_symbols": int(len(missing_codes)),
        "warning": (
            "NON-PIT diagnostic ONLY: today's stock_info ST status is applied "
            "retroactively to old signals; it leaks future information. "
            "This result is not historical performance and cannot train a model."
        ),
    }
    return CurrentSTResult(
        sig.loc[allowed].reset_index(drop=True),
        audit.sort_values("stock_code").reset_index(drop=True),
        stats,
    )
