"""P4.3-E: fail-closed corporate-action economic label quality gate.

Research ONLY. This module never writes to production SQL and never uses old
inspected test prices or labels. A known event is not proof of complete event
coverage. Only explicitly certified stock/date coverage enables train labels.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

EVENT_COLUMNS = ("stock_code", "ex_date", "published_on", "share_multiplier",
                 "cash_per_old_share", "source_url", "source_id")
COVERAGE_COLUMNS = ("stock_code", "start_date", "end_date", "verified_on",
                    "source_url", "source_id", "status")
WARNING = (
    "P4 CORPORATE-ACTION DATA-QUALITY GATE, NOT A BACKTEST. Known events without "
    "complete verified coverage yield illustrative gross economic returns ONLY, "
    "never train labels. A missing event row is NOT proof of no action. Corporate "
    "actions within the pre-signal feature window are quarantined because legacy "
    "unadjusted price/volume factors are not adjusted. Cash is counted at gross "
    "entitlement and held without reinvestment; tax, round lots, fills and fees are "
    "NOT modeled. The previously inspected old test range is excluded."
)


def _codes(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip()
    if x.isna().any() or not x.str.fullmatch(r"\d{6}").all():
        raise ValueError("Expected six-digit stock_code")
    return x


def _url(s: pd.Series, field: str) -> pd.Series:
    x = s.astype("string").str.strip()
    if x.isna().any() or not x.str.match(r"^https?://[a-zA-Z0-9.-]+(?::\d+)?(?:/|$)").all():
        raise ValueError(f"Invalid {field} source URL")
    return x


def _required(frame: pd.DataFrame, cols: tuple[str, ...], what: str) -> None:
    missing = set(cols) - set(frame)
    if missing:
        raise ValueError(f"{what} missing columns: {sorted(missing)}")


def load_actions(path: str | Path) -> pd.DataFrame:
    """One aggregated entitlement record per stock/ex-date; no inferred events."""
    x = pd.read_csv(path, dtype="string", keep_default_na=False)
    _required(x, EVENT_COLUMNS, "Corporate actions")
    x = x.loc[:, EVENT_COLUMNS].copy()
    x["stock_code"] = _codes(x.stock_code)
    for name in ("ex_date", "published_on"):
        x[name] = pd.to_datetime(x[name], format="%Y-%m-%d", errors="raise").dt.normalize()
    if x[["ex_date", "published_on"]].isna().any().any() or (x.published_on > x.ex_date).any():
        raise ValueError("Corporate action must be announced by the ex date")
    for name in ("share_multiplier", "cash_per_old_share"):
        x[name] = pd.to_numeric(x[name], errors="raise").astype(float)
        if not np.isfinite(x[name]).all():
            raise ValueError(f"Non-finite {name}")
    if x.share_multiplier.le(0).any() or x.cash_per_old_share.lt(0).any():
        raise ValueError("Nonpositive share multiplier or negative cash dividend")
    if (x.share_multiplier.eq(1) & x.cash_per_old_share.eq(0)).any():
        raise ValueError("No-op corporate action is not evidence")
    x["source_url"] = _url(x.source_url, "action")
    x["source_id"] = x.source_id.astype("string").str.strip()
    if x.source_id.isna().any() or x.source_id.eq("").any():
        raise ValueError("Action source_id required")
    if x.duplicated(["stock_code", "ex_date"]).any():
        raise ValueError("Ambiguous duplicate ex-date; aggregate multi-actions per date")
    return x.sort_values(["stock_code", "ex_date"]).reset_index(drop=True)


def load_coverage(path: str | Path) -> pd.DataFrame:
    """Complete enumeration is a separately verified assertion, not inferred."""
    x = pd.read_csv(path, dtype="string", keep_default_na=False)
    _required(x, COVERAGE_COLUMNS, "Corporate-action coverage")
    x = x.loc[:, COVERAGE_COLUMNS].copy()
    x["stock_code"] = _codes(x.stock_code)
    for name in ("start_date", "end_date", "verified_on"):
        x[name] = pd.to_datetime(x[name], format="%Y-%m-%d", errors="raise").dt.normalize()
    if x[["start_date", "end_date", "verified_on"]].isna().any().any():
        raise ValueError("Missing coverage dates")
    if x.start_date.gt(x.end_date).any() or x.verified_on.lt(x.start_date).any():
        raise ValueError("Invalid verified coverage date interval")
    if not x.status.eq("COMPLETE_VERIFIED").all():
        raise ValueError("Only explicit COMPLETE_VERIFIED coverage accepted")
    x["source_url"] = _url(x.source_url, "coverage")
    x["source_id"] = x.source_id.astype("string").str.strip()
    if x.source_id.isna().any() or x.source_id.eq("").any():
        raise ValueError("Coverage source_id required")
    if x.duplicated(["stock_code", "start_date", "end_date"]).any():
        raise ValueError("Duplicate complete coverage assertion")
    # Different overlapping certificates make provenance ambiguous. Fail closed.
    for _, group in x.sort_values("start_date").groupby("stock_code", sort=False):
        if (group.start_date.iloc[1:].to_numpy() <=
                group.end_date.iloc[:-1].to_numpy()).any():
            raise ValueError("Overlapping coverage certificates")
    return x.sort_values(["stock_code", "start_date"]).reset_index(drop=True)


def audit_corporate_actions(selected: pd.DataFrame, raw: pd.DataFrame,
                            calendar: pd.DatetimeIndex, actions: pd.DataFrame,
                            coverage: pd.DataFrame, *, hold_sessions: int = 5,
                            feature_lookback: int = 40,
                            test_start: str,
                            tolerance: float = 1e-7
                            ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Rebuild entry/exit opens and audit economic labels with strict coverage.

    Restricts both signal dates AND raw market bars to dates before test_start,
    and uses no test label values or test dates for candidate comparisons.
    The feature lookback intentionally quarantines legacy unadjusted factors
    whenever any confirmed ex-date lies in the preceding N market sessions.
    """
    if hold_sessions < 1 or feature_lookback < 21 or tolerance <= 0:
        raise ValueError("Invalid P4 economic-return audit configuration")
    cut = pd.Timestamp(test_start)
    cal = pd.DatetimeIndex(calendar).normalize()
    if cal.empty or cal.hasnans or cal.has_duplicates or not cal.is_monotonic_increasing:
        raise ValueError("Market calendar must be unique increasing")
    if cal.max() >= cut:
        raise ValueError("Audit calendar contains inspected old TEST dates")
    cols = {"stock_code", "trade_date", "session_id", "label_end_session", "target_return"}
    if cols - set(selected):
        raise ValueError(f"Pool data missing {sorted(cols-set(selected))}")
    if {"stock_code", "trade_date", "open", "not_billing"} - set(raw):
        raise ValueError("Missing formal raw open-price columns")
    s = selected.loc[:, ["stock_code", "trade_date", "session_id",
                          "label_end_session", "target_return"]].copy()
    s["stock_code"] = _codes(s.stock_code)
    s["trade_date"] = pd.to_datetime(s.trade_date).dt.normalize()
    if s.trade_date.ge(cut).any() or s.trade_date.isna().any():
        raise ValueError("Old TEST signal date must not be processed")
    if s.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate stock/date pool sample")
    sid = pd.to_numeric(s.session_id, errors="raise").to_numpy(dtype=int)
    if np.any(sid < feature_lookback - 1) or np.any(sid >= len(cal)):
        raise ValueError("Not enough historical market dates for feature safety audit")
    if not (cal[sid].to_numpy() == s.trade_date.to_numpy()).all():
        raise ValueError("Pool session_id disagrees with market calendar")
    s["feature_first_date"] = cal[sid-feature_lookback+1].to_numpy()
    entry_sid, exit_sid = sid+1, sid+hold_sessions+1
    s["entry_date"] = pd.NaT
    s["exit_date"] = pd.NaT
    within = exit_sid < len(cal)
    s.loc[within, "entry_date"] = cal[entry_sid[within]].to_numpy()
    s.loc[within, "exit_date"] = cal[exit_sid[within]].to_numpy()
    label = pd.to_numeric(s.target_return, errors="coerce")
    label_end = pd.to_numeric(s.label_end_session, errors="coerce")
    if ((label.notna() != label_end.notna()) |
            (label.notna() & (~np.isfinite(label.to_numpy(dtype=float))))).any():
        raise ValueError("Invalid original label and label_end_session")
    valid_label = label.notna().to_numpy()
    if ((valid_label & ~within) |
            (valid_label & (label_end.fillna(-1).to_numpy(dtype=int) != exit_sid))).any():
        raise ValueError("Original label extends beyond audited market calendar")
    bars = raw.loc[pd.to_numeric(raw.not_billing, errors="coerce").eq(1),
                   ["stock_code", "trade_date", "open"]].copy()
    bars["stock_code"] = _codes(bars.stock_code)
    bars["trade_date"] = pd.to_datetime(bars.trade_date).dt.normalize()
    if bars.trade_date.ge(cut).any():
        raise ValueError("Old TEST prices must not be accessed")
    if bars.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate formal price source")
    bars["open"] = pd.to_numeric(bars.open, errors="coerce")
    # Merges are anchored on selected sample rows, not a stock-level shift
    # that might jump across a suspension/missing market date.
    for date_col, price_col in (("entry_date", "entry_open"),
                                ("exit_date", "exit_open")):
        s = s.merge(bars.rename(columns={"trade_date": date_col,
                                        "open": price_col}),
                    how="left", on=["stock_code", date_col], sort=False,
                    validate="many_to_one")
    entry = s.entry_open.to_numpy(dtype=float)
    exit_ = s.exit_open.to_numpy(dtype=float)
    complete_price = (within & np.isfinite(entry) & np.isfinite(exit_) &
                      (entry > 0) & (exit_ > 0))
    raw_recomputed = np.divide(exit_, entry,
                               out=np.full(len(s), np.nan),
                               where=complete_price) - 1
    matches = valid_label & complete_price & np.isclose(
        raw_recomputed, label.to_numpy(dtype=float), rtol=0, atol=tolerance)
    s["raw_label_recomputed"] = raw_recomputed
    s["raw_label_match"] = matches
    s["event_in_label_count"] = 0
    s["event_in_feature_count"] = 0
    s["known_event_source_ids"] = ""
    shares = np.ones(len(s), dtype=float)
    cash = np.zeros(len(s), dtype=float)
    # Each event is aggregated by stock and effective ex-date. A record date
    # on entry day has already passed at the open, so only (entry, exit] applies.
    act = actions.loc[actions.ex_date.lt(cut)].sort_values(["stock_code", "ex_date"])
    for event in act.itertuples(index=False):
        same = s.stock_code.eq(event.stock_code).to_numpy()
        label_hit = (same & s.entry_date.lt(event.ex_date).to_numpy() &
                     s.exit_date.ge(event.ex_date).to_numpy())
        feature_hit = (same & s.feature_first_date.le(event.ex_date).to_numpy() &
                       s.trade_date.ge(event.ex_date).to_numpy())
        s.loc[label_hit, "event_in_label_count"] += 1
        s.loc[feature_hit, "event_in_feature_count"] += 1
        if label_hit.any():
            # Entitlements from each event scale with shares owned immediately
            # before that event. Cash is not reinvested into later share splits.
            cash[label_hit] += shares[label_hit] * float(event.cash_per_old_share)
            shares[label_hit] *= float(event.share_multiplier)
            current = s.loc[label_hit, "known_event_source_ids"].astype(str)
            s.loc[label_hit, "known_event_source_ids"] = current.map(
                lambda prior: (prior + "|" if prior else "") + str(event.source_id)
            )
    gross = np.full(len(s), np.nan)
    np.divide(shares * exit_ + cash, entry, out=gross, where=complete_price)
    gross -= 1.0
    s["illustrative_known_event_return"] = np.where(
        (s.event_in_label_count.to_numpy() > 0) & matches, gross, np.nan)
    s["coverage_verified"] = False
    s["coverage_source_id"] = ""
    # Require ONE explicit complete certificate spanning the whole feature
    # lookback AND label window; no automatic merging of partial certificates.
    for item in coverage.itertuples(index=False):
        ok = (s.stock_code.eq(item.stock_code) &
              s.feature_first_date.ge(item.start_date) &
              s.exit_date.le(item.end_date))
        if ok.any():
            s.loc[ok, "coverage_verified"] = True
            s.loc[ok, "coverage_source_id"] = str(item.source_id)
    s["feature_window_action_free"] = s.event_in_feature_count.eq(0)
    usable = (matches & s.coverage_verified.to_numpy() &
              s.feature_window_action_free.to_numpy())
    s["training_label_allowed"] = usable
    s["economic_target_return"] = np.where(usable, gross, np.nan)
    s["reason"] = np.select(
        [~valid_label, valid_label & ~complete_price,
         valid_label & complete_price & ~matches,
         ~s.coverage_verified.to_numpy(),
         ~s.feature_window_action_free.to_numpy()],
        ["MISSING_ORIGINAL_LABEL", "MISSING_PRICE_PATH",
         "RAW_LABEL_MISMATCH", "UNKNOWN_ACTION_COVERAGE",
         "UNADJUSTED_FEATURE_ACTION_WINDOW"],
        default="COMPLETE_VERIFIED_LABEL_AND_FEATURE_WINDOW"
    )
    daily = s.groupby("trade_date", sort=True).agg(
        pool_rows=("stock_code", "size"),
        original_labeled=("target_return", "count"),
        known_event_label_rows=("event_in_label_count", lambda x: int(x.gt(0).sum())),
        feature_action_rows=("event_in_feature_count", lambda x: int(x.gt(0).sum())),
        verified_coverage_rows=("coverage_verified", "sum"),
        training_label_allowed=("training_label_allowed", "sum"),
    ).reset_index()
    counts = s.reason.value_counts().to_dict()
    summary = {
        "audit_version": "P4_CORPORATE_ACTION_GATE_V1",
        "test_start_EXCLUDED": str(cut.date()),
        "label_horizon_sessions": hold_sessions,
        "feature_lookback_market_sessions": feature_lookback,
        "pool_rows_pretest": len(s),
        "original_labeled_rows": int(valid_label.sum()),
        "recomputed_original_labels_matching": int(matches.sum()),
        "known_action_label_rows": int(s.event_in_label_count.gt(0).sum()),
        "known_action_feature_rows": int(s.event_in_feature_count.gt(0).sum()),
        "verified_complete_coverage_rows": int(s.coverage_verified.sum()),
        "training_label_allowed_rows": int(usable.sum()),
        "training_is_blocked_by_incomplete_coverage": bool((~s.coverage_verified).any()),
        "reasons": {str(k): int(v) for k, v in counts.items()},
        "source_event_rows": int(len(actions)),
        "explicit_complete_coverage_assertions": int(len(coverage)),
        "all_test_prices_and_labels_excluded": True,
        "warning": WARNING,
    }
    return summary, s, daily


def main() -> None:
    from .p4_mvp import PoolConfig, build_pool, load_formal_ohlcv
    from .risk_events import load_risk_events
    p = argparse.ArgumentParser(description="P4.3-E strictly audited economic target gate, SELECT only")
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 3, 1))
    p.add_argument("--reference-metadata", type=Path, required=True)
    p.add_argument("--events", type=Path,
                   default=Path("ashare/data/corporate_actions_verified_examples.csv"))
    p.add_argument("--coverage", type=Path,
                   default=Path("ashare/data/corporate_action_coverage_examples.csv"))
    p.add_argument("--risk-events", type=Path,
                   default=Path("ashare/data/risk_events_verified_examples.csv"))
    p.add_argument("--no-risk-events", action="store_true")
    p.add_argument("--db-chunk-size", type=int, default=50000)
    p.add_argument("--max-impact-rows", type=int, default=500)
    p.add_argument("--out", type=Path, default=Path("reports/ashare_p4_corporate_actions"))
    args = p.parse_args()
    if args.max_impact_rows < 0 or args.db_chunk_size < 1:
        p.error("Invalid output row cap/chunksize")
    with args.reference_metadata.open(encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("model_version") != "P4_RIDGE_V1":
        raise ValueError("Reference model metadata not P4_RIDGE_V1")
    test_start = pd.Timestamp(metadata["test_start"]).normalize()
    historical_end = (test_start - pd.Timedelta(days=1)).date()
    if args.start > historical_end:
        p.error("Start must precede archived old test")
    config = PoolConfig(**metadata["pool_config"])
    actions = load_actions(args.events)
    coverage = load_coverage(args.coverage)
    if actions.ex_date.ge(test_start).any():
        raise ValueError("Event file includes inspected old test dates; split event files first")
    if coverage.end_date.ge(test_start).any():
        raise ValueError("Coverage file includes inspected old test dates; split coverage first")
    def log(message):
        print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)
    log("[1/4] SELECT formal market rows before archived test start ONLY")
    raw = load_formal_ohlcv(args.start, historical_end,
                            hold_sessions=config.hold_sessions,
                            include_label_tail=False, chunksize=args.db_chunk_size,
                            progress=log)
    if raw.empty:
        raise ValueError("No formal market rows")
    events = None if args.no_risk_events else load_risk_events(args.risk_events)
    log("[2/4] Rebuild P4 pool with historical-only prices and labels")
    pool = build_pool(raw, args.start, historical_end, config, events=events)
    summary, audited, daily = audit_corporate_actions(
        pool.data, raw, pool.calendar, actions, coverage,
        hold_sessions=config.hold_sessions,
        feature_lookback=config.consecutive_sessions,
        test_start=str(test_start.date())
    )
    if summary["original_labeled_rows"] != metadata["refit_samples"]:
        raise ValueError("Old Ridge refit sample count mismatch; no files saved")
    if not summary["original_labeled_rows"] == summary["recomputed_original_labels_matching"]:
        raise ValueError("Raw OHLC/old label mismatch; no files saved")
    summary.update({
        "reference_model_version": metadata["model_version"],
        "reference_refit_samples_matched": True,
        "pool_config": asdict(config),
        "known_event_source_ids": actions.source_id.astype(str).tolist(),
        "universe_coverage_complete": False if coverage.empty else
            bool(summary["training_label_allowed_rows"] == summary["original_labeled_rows"]),
        "per_row_source_snapshot_verified": False,
        "test_start_comes_from": str(args.reference_metadata),
    })
    log("[3/4] Reconcile known ex-dates; quarantine unknown/unadjusted windows")
    # Event-only affected cases and daily aggregate are safe to export without
    # exposing unverified rows as a model-ready label file.
    impacted = audited.loc[
        audited.event_in_label_count.gt(0) | audited.event_in_feature_count.gt(0)
    ].sort_values(["trade_date", "stock_code"])
    if args.max_impact_rows:
        impacted = impacted.head(args.max_impact_rows)
    else:
        impacted = impacted.head(0)
    out = args.out / f"{args.start}_{historical_end}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "corporate_action_gate.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    daily.to_csv(out / "corporate_action_daily.csv", index=False,
                 encoding="utf-8-sig")
    impacted.to_csv(out / "corporate_action_impacted_cases.csv", index=False,
                    encoding="utf-8-sig")
    log(f"[4/4] Saved research-only company action audit: {out}")
    print(json.dumps({"output": str(out),
                      "training_label_allowed_rows": summary["training_label_allowed_rows"],
                      "warning": WARNING}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
