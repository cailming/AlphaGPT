"""P3.3-C: dated risk-warning and suspension events, partial evidence ONLY.

A verified event may block a later prospective entry. Missing events are
NOT proof that a stock was normal. This module never certifies all-market
historical eligibility and never performs forced liquidation or valuation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .st_history import _codes, _dates


EVENT_TYPES = {
    "RISK_ST", "RISK_STAR_ST", "RISK_NORMAL",
    "HALT", "RESUME", "DELISTED",
}
RISK_TYPES = {"RISK_ST", "RISK_STAR_ST", "RISK_NORMAL"}
HALT_TYPES = {"HALT", "RESUME"}
REQUIRED = {
    "stock_code", "event_type", "effective_date",
    "published_on", "source_url", "source_id",
}
MODE = "PARTIAL_PIT_RISK_EVENT_EVIDENCE"


@dataclass
class RiskEventResult:
    signals: pd.DataFrame
    events_audit: pd.DataFrame
    blocked_by_day: pd.DataFrame
    stats: dict


def _prepare_events(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Risk events missing required columns: {sorted(missing)}")
    data = frame.copy().reset_index(drop=True)
    data["stock_code"] = _codes(data["stock_code"])
    for field in ("effective_date", "published_on"):
        data[field] = _dates(data[field])
    data["event_type"] = data["event_type"].astype("string").str.strip()
    if (~data.event_type.isin(EVENT_TYPES)).any():
        raise ValueError("Unknown event_type (RISK_ST / RISK_STAR_ST / RISK_NORMAL / HALT / RESUME / DELISTED)")
    for field in ("source_url", "source_id"):
        data[field] = data[field].astype("string").str.strip()
        if data[field].isna().any() or (data[field] == "").any():
            raise ValueError(f"Every risk event requires {field}")
    if not data.source_url.str.match(r"^https?://[A-Za-z0-9.-]+(/|$)").all():
        raise ValueError("source_url must be an absolute HTTP(S) URL")

    # A single risk/halting effective-date transition must be unambiguous.
    data["scope"] = data.event_type.map(
        lambda k: "RISK" if k in RISK_TYPES
        else "HALT" if k in HALT_TYPES else "DELISTED"
    )
    if data.duplicated(["stock_code", "scope", "effective_date"]).any():
        raise ValueError("Ambiguous/duplicate stock_code scope effective_date")
    return data.sort_values(
        ["stock_code", "effective_date", "published_on", "event_type"]
    ).reset_index(drop=True)


def load_risk_events(path: str | Path) -> pd.DataFrame:
    """Parse manually vetted public announcement events in UTF-8 CSV."""
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    return _prepare_events(frame).drop(columns="scope")


def _latest_known(items: pd.DataFrame, order_day: pd.Timestamp):
    visible = items.loc[
        (items["effective_date"] <= order_day)
        & (items["published_on"] < order_day)
    ]
    if visible.empty:
        return None
    # A later effective-date transition supersedes an earlier one.
    # A correction disclosed later cannot rewrite decisions made earlier.
    return visible.iloc[-1]


def filter_signals_by_risk_events(
    signals: pd.DataFrame,
    events: pd.DataFrame,
    market_days,
) -> RiskEventResult:
    """Block only if event is known BEFORE execution day and already in force.

    Signal generated at T close. Intended entry is T+1 open.
    A publication with known *calendar date* P is usable conservatively
    no earlier than a subsequent trading date (> P). An event effective
    in the future can be recorded but cannot block an earlier order.
    """
    if not {"stock_code", "signal_date", "score"}.issubset(signals.columns):
        raise ValueError("Signals require stock_code, signal_date, score")
    ev = _prepare_events(events)
    sig = signals.copy().reset_index(drop=True)
    sig["stock_code"] = _codes(sig["stock_code"])
    sig["signal_date"] = _dates(sig["signal_date"])

    dates = pd.DatetimeIndex(pd.to_datetime(list(market_days))).normalize()
    if dates.hasnans or len(dates) < 2 or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("market_days must contain sorted unique trading days")
    next_day = dict(zip(dates[:-1], dates[1:]))
    order_days = sig.signal_date.map(next_day)
    invalid_session = sig.signal_date.isin(dates) == False
    if invalid_session.any():
        raise ValueError("Signal date absent from market_days")
    blocked = pd.Series(False, index=sig.index, dtype=bool)
    daily_events = []
    affected_events = {}
    risk_blocks = halt_blocks = delisted_blocks = 0

    for code, items in ev.groupby("stock_code", sort=False):
        subset = sig.index[(sig.stock_code == code) & order_days.notna()]
        if not len(subset):
            continue
        risk = items.loc[items.scope == "RISK"]
        halt = items.loc[items.scope == "HALT"]
        delisted = items.loc[items.scope == "DELISTED"]
        for order_day in pd.DatetimeIndex(order_days.loc[subset].unique()).sort_values():
            selected = subset[order_days.loc[subset].eq(order_day).to_numpy()]
            if not len(selected):
                continue
            r = _latest_known(risk, order_day)
            h = _latest_known(halt, order_day)
            d = _latest_known(delisted, order_day)
            causes = []
            evidences = []
            if r is not None and r["event_type"] in ("RISK_ST", "RISK_STAR_ST"):
                causes.append("RISK_WARNING")
                risk_blocks += len(selected)
                evidences.append(r)
            if h is not None and h["event_type"] == "HALT":
                causes.append("HALT")
                halt_blocks += len(selected)
                evidences.append(h)
            if d is not None:
                causes.append("DELISTED")
                delisted_blocks += len(selected)
                evidences.append(d)
            if not causes:
                continue
            blocked.loc[selected] = True
            signal_day = sig.loc[selected[0], "signal_date"]
            daily_events.append({
                "stock_code": code,
                "signal_date": signal_day.strftime("%Y-%m-%d"),
                "intended_order_date": order_day.strftime("%Y-%m-%d"),
                "blocked_signal_rows": int(len(selected)),
                "block_reasons": "|".join(causes),
                "source_ids": "|".join(str(row.source_id) for row in evidences),
            })
            for row in evidences:
                key = (row.stock_code, row.scope, row.effective_date)
                affected_events[key] = affected_events.get(key, 0) + len(selected)

    audit = ev.drop(columns="scope").copy()
    audit["blocked_signal_rows_when_active"] = [
        affected_events.get((r.stock_code,
                             "RISK" if r.event_type in RISK_TYPES
                             else "HALT" if r.event_type in HALT_TYPES
                             else "DELISTED",
                             r.effective_date), 0)
        for r in audit.itertuples(index=False)
    ]
    for field in ("effective_date", "published_on"):
        audit[field] = audit[field].dt.strftime("%Y-%m-%d")
    unaffected_unknown = ~sig.stock_code.isin(ev.stock_code)
    stats = {
        "status": MODE,
        "evidence_scope": "PARTIAL_ANNOUNCEMENT_EVENTS_NOT_COMPLETE_ST_HISTORY",
        "is_full_market_risk_history": False,
        "all_unflagged_symbols_proven_normal": False,
        "source": "MANUALLY_VETTED_DATE_STAMPED_ANNOUNCEMENTS",
        "screen_at": "T_PLUS_1_OPEN",
        "public_date_rule": "published_on < intended_order_date",
        "effective_date_rule": "effective_date <= intended_order_date",
        "signal_rows_before": int(len(sig)),
        "signal_rows_after": int((~blocked).sum()),
        "blocked_unique_signal_rows": int(blocked.sum()),
        "risk_warning_block_hits": int(risk_blocks),
        "halt_block_hits": int(halt_blocks),
        "delisted_block_hits": int(delisted_blocks),
        "evidence_events": int(len(ev)),
        "evidence_symbols": int(ev.stock_code.nunique()),
        "signals_with_no_events_for_stock": int(unaffected_unknown.sum()),
        "signals_without_next_session_in_range": int(order_days.isna().sum()),
        "warning": (
            "Partial historical public-event evidence only. No recorded "
            "warning is NOT proof of historical NORMAL status. The filter "
            "cannot certify tradeability, delisting settlement or audited PnL."
        ),
    }
    cols = ["stock_code", "signal_date", "intended_order_date",
            "blocked_signal_rows", "block_reasons", "source_ids"]
    decisions = pd.DataFrame(daily_events, columns=cols)
    if not decisions.empty:
        decisions = decisions.sort_values(
            ["signal_date", "stock_code"]
        ).reset_index(drop=True)
    return RiskEventResult(
        sig.loc[~blocked].reset_index(drop=True),
        audit,
        decisions,
        stats,
    )
