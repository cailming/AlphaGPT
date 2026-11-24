"""Build causal sequences independently of the daily ranking cap and audit V1 refusals."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, read_json
from .p4_mvp import FEATURES, PoolConfig, build_pool
from .p4_subset_dataset import FEATURE_LOOKBACK, TRANSFORMER_SEQUENCE
from .p4_subset_inputs import RISK_FILE, hash_file


VERSION = "P5_CAUSAL_SEQUENCE_V2"
SUPPORT_FLOORS = (0, 5, 10, 20)
CHINA = ZoneInfo("Asia/Shanghai")
FEATURE_COLUMNS = ["stock_code", "trade_date", "session_id", "liquidity_rank", "close", "liquidity_20", *FEATURES]
FORWARD_COLUMNS = {"target_return", "label_end_session", "entry_date", "exit_date", "split", "sample_grade"}
VENDOR_REFERENCE = "require_vendor_reference"
PRIOR_FORMAL_REFERENCE = "prior_valid_formal_close"


@dataclass
class CausalPopulation:
    features: pd.DataFrame
    anchors: pd.DataFrame
    ranked: pd.DataFrame
    calendar: pd.DatetimeIndex
    stats: dict


def price_window_flags(raw, calendar, *, allow_auxiliary_fallback=False):
    """Validate native prices. An explicitly enabled fallback uses the observed prior close."""
    cols = ["open", "close", "high", "low", "volume", "amount", "pre_close"]
    frame = raw.loc[pd.to_numeric(raw.not_billing, errors="coerce").eq(1), ["stock_code", "trade_date", *cols]].copy()
    frame["trade_date"] = pd.to_datetime(frame.trade_date, errors="raise").dt.normalize()
    frame["stock_code"] = frame.stock_code.astype("string").str.strip()
    frame[cols] = frame[cols].apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_values(["stock_code", "trade_date"], kind="stable").reset_index(drop=True)
    frame["session_id"] = frame.trade_date.map(pd.Series(np.arange(len(calendar)), index=calendar)).astype(int)
    groups = frame.groupby("stock_code", sort=False)
    prior = groups.close.shift(1)
    reference_ok = np.isfinite(frame.pre_close) & frame.pre_close.gt(0)
    bar_ok = (np.isfinite(frame[cols[:-1]]).all(axis=1) & frame.low.gt(0) & frame.volume.gt(0) & frame.amount.gt(0)
              & frame.high.ge(frame[["open", "close"]].max(axis=1)) & frame.low.le(frame[["open", "close"]].min(axis=1)))
    continuous = frame.session_id.sub(groups.session_id.shift(1)).eq(1)
    jump = frame.open.div(prior).sub(1).abs().gt(.30) | frame.close.div(prior).sub(1).abs().gt(.32)
    prior_valid = bar_ok.groupby(frame.stock_code, sort=False).shift(1, fill_value=False)
    fallback = allow_auxiliary_fallback & ~reference_ok & bar_ok & continuous & prior_valid & ~jump
    bar_status = ["INVALID_FORMAL_BAR", "MISSING_OR_INVALID_REFERENCE", "PREVIOUS_SESSION_MISSING", "UNEXPLAINED_PRICE_JUMP"]
    frame["bar_status"] = np.select([fallback, ~bar_ok, ~reference_ok, ~continuous, jump],
                                    ["OK_AUXILIARY_REFERENCE_FALLBACK", *bar_status], default="OK")
    frame["vendor_reference_available"] = reference_ok
    frame["auxiliary_reference_fallback"] = fallback
    frame["fallback_in_price_window"] = fallback.groupby(frame.stock_code, sort=False).transform(
        lambda rows: rows.rolling(FEATURE_LOOKBACK, min_periods=FEATURE_LOOKBACK).sum().gt(0))
    frame["valid_price_window"] = frame.bar_status.str.startswith("OK").groupby(frame.stock_code, sort=False).transform(
        lambda rows: rows.rolling(FEATURE_LOOKBACK, min_periods=FEATURE_LOOKBACK).sum().eq(FEATURE_LOOKBACK))
    return frame[["stock_code", "trade_date", "session_id", "bar_status", "valid_price_window", "vendor_reference_available",
                  "auxiliary_reference_fallback", "fallback_in_price_window"]]


def build_causal_population(raw, start, end, cfg, *, events=None, reference_policy=PRIOR_FORMAL_REFERENCE):
    """Materialize all eligible feature rows, then select only the signal-day anchors."""
    if start > end:
        raise ValueError("Inverted feature dates")
    if reference_policy not in (VENDOR_REFERENCE, PRIOR_FORMAL_REFERENCE):
        raise ValueError("Unrecognized auxiliary reference policy")
    days = pd.to_datetime(raw.trade_date, errors="raise").dt.normalize()
    observed = raw.loc[days.le(pd.Timestamp(end))]
    count = observed.stock_code.nunique()
    if not count:
        raise ValueError("No stock observations through the feature cutoff")
    uncapped = build_pool(observed, start, end, replace(cfg, top_n=count), events=events)
    ranked = uncapped.data[FEATURE_COLUMNS]
    flags = price_window_flags(observed, uncapped.calendar, allow_auxiliary_fallback=reference_policy == PRIOR_FORMAL_REFERENCE)
    good = flags.loc[flags.valid_price_window, ["stock_code", "trade_date", "fallback_in_price_window"]].rename(
        columns={"fallback_in_price_window": "auxiliary_reference_fallback"})
    features = ranked.merge(good, on=["stock_code", "trade_date"], validate="one_to_one")
    features["trade_date"] = features.trade_date.dt.strftime("%Y-%m-%d")
    anchors = features.loc[features.liquidity_rank.le(cfg.top_n)].reset_index(drop=True)
    ranked = ranked.assign(trade_date=ranked.trade_date.dt.strftime("%Y-%m-%d"))
    stats = {"version": VERSION, "top_n": cfg.top_n, "uncapped_feature_rows": len(features), "anchor_rows": len(anchors),
             "auxiliary_reference_policy": reference_policy,
             "formal_bars_using_auxiliary_reference_fallback": int(flags.auxiliary_reference_fallback.sum()),
             "feature_rows_using_reference_fallback": int(features.auxiliary_reference_fallback.sum()),
             "prices_after_end_read": False, "future_data_used_for_ranking": False,
             "current_st_backfilled": False, "historical_st_complete": False,
             "price_guard_rejected_rows": len(ranked) - len(features),
             "p4_risk_vetoes": uncapped.stats["blocked_known_risk_event_rows"],
             "p4_rejected_by_reason": uncapped.stats["rejected_by_reason"]}
    return CausalPopulation(features, anchors, ranked, uncapped.calendar, stats)


def sequence_indexes(features, anchors, length=TRANSFORMER_SEQUENCE, *, known_actions=None, calendar=None):
    """Return positional indexes for contiguous same-stock causal features ending at each anchor."""
    if length < 1 or FORWARD_COLUMNS.intersection(features):
        raise ValueError("Invalid length or non-feature columns in sequence input")
    if any(frame.duplicated(["stock_code", "session_id"]).any() for frame in (features, anchors)):
        raise ValueError("Duplicate stock/session keys")
    actions = {}
    days = None
    if known_actions is not None:
        if calendar is None:
            raise ValueError("Action filtering requires the market session calendar")
        days = pd.DatetimeIndex(calendar).strftime("%Y-%m-%d")
        evidence = known_actions.loc[:, ["stock_code", "ex_date", "published_on"]].copy()
        for field in ("ex_date", "published_on"):
            parsed = pd.to_datetime(evidence[field], format="%Y-%m-%d", errors="raise")
            if parsed.isna().any():
                raise ValueError(f"Missing {field} in known actions")
            evidence[field] = parsed.dt.strftime("%Y-%m-%d")
        actions = {stock: list(group[["ex_date", "published_on"]].itertuples(index=False, name=None))
                   for stock, group in evidence.groupby("stock_code", sort=False)}
    by_stock = {stock: dict(zip(group.session_id, group.index, strict=True))
                for stock, group in features.reset_index(drop=True).groupby("stock_code", sort=False)}
    feature_days = features.trade_date.astype(str).to_numpy()
    kept, indices = [], []
    for position, (stock, session, signal) in enumerate(anchors[["stock_code", "session_id", "trade_date"]].itertuples(
            index=False, name=None)):
        tracker = by_stock.get(stock)
        if tracker is None:
            continue
        first = session + 1 - length
        rows = [tracker.get(i) for i in range(first, session + 1)]
        if any(index is None for index in rows):
            continue
        signal = str(signal)
        if feature_days[rows[-1]] != signal or np.any(feature_days[rows[:-1]] >= signal):
            raise ValueError("Sequence feature dates disagree with the signal boundary")
        if days is not None and stock in actions:
            lookback = first - FEATURE_LOOKBACK + 1
            if lookback < 0 or session >= len(days) or days[session] != signal:
                raise ValueError("Action window does not match market sessions")
            if any(days[lookback] <= ex_date <= signal and published <= signal for ex_date, published in actions[stock]):
                continue
        kept.append(position)
        indices.append(rows)
    return np.asarray(kept, dtype=np.int64), np.asarray(indices, dtype=np.int64).reshape(-1, length)


def formal_snapshot(engine, now, asof=None):
    """Read one consistent PostgreSQL snapshot, including V1's original warmup bounds."""
    first_unavailable = now.date() + timedelta(days=now.hour >= 18)
    latest = text("SELECT max(trade_date) FROM public.market WHERE not_billing=1 AND trade_date < :unavailable")
    prices = text("""
        SELECT stock_code, trade_date, open, high, low, close, volume, amount, pre_close, not_billing
          FROM public.market
         WHERE not_billing=1 AND stock_code IS NOT NULL AND trade_date BETWEEN :begin AND :end
    """)
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
        with conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            if conn.scalar(text("show transaction_read_only")) != "on":
                raise RuntimeError("Formal OHLCV requires a read-only snapshot")
            if asof is None:
                value = conn.scalar(latest, {"unavailable": first_unavailable})
                if value is None:
                    raise ValueError("No formal market session available")
                asof = pd.Timestamp(value).date()
                if (now.date() - asof).days > 10:
                    raise ValueError(f"Latest formal OHLCV is stale: {asof}")
            if asof >= first_unavailable:
                raise ValueError("Requested session is after the formal availability boundary")
            bounds = {"begin": asof - timedelta(days=140), "end": asof}
            batches = pd.read_sql_query(prices, conn.execution_options(stream_results=True), params=bounds, chunksize=75000)
            raw = pd.concat(batches, ignore_index=True)
    if raw.empty or pd.Timestamp(raw.trade_date.max()).date() != asof:
        raise ValueError(f"No observations for requested signal session: {asof}")
    return asof, raw


def refusal_diagnostics(population, flags, asof, counts, length, *, known_actions=None, v2_population=None,
                        v2_flags=None):
    anchors = population.anchors.loc[population.anchors.trade_date.eq(asof)].reset_index(drop=True)
    v1_features = population.anchors
    candidate_features = population.features if v2_population is None else v2_population.features
    v1_ok, _ = sequence_indexes(v1_features, anchors, length)
    v2_price_ok, _ = sequence_indexes(candidate_features, anchors, length)
    v2_ok, _ = sequence_indexes(candidate_features, anchors, length, known_actions=known_actions,
                                calendar=population.calendar)
    flags = flags.loc[flags.stock_code.isin(anchors.stock_code)]
    raw = {(row.stock_code, row.session_id): (row.valid_price_window, row.bar_status)
           for row in flags.itertuples(index=False)}
    substitutions = set() if v2_flags is None else set(v2_flags.loc[v2_flags.auxiliary_reference_fallback,
                                            ["stock_code", "session_id"]].itertuples(index=False, name=None))
    ranks = {(r.stock_code, r.session_id): r.liquidity_rank for r in population.ranked.loc[
        population.ranked.stock_code.isin(anchors.stock_code)].itertuples(index=False)}
    days = population.calendar.strftime("%Y-%m-%d")
    v1_ok, v2_price_ok, v2_ok = map(set, (v1_ok, v2_price_ok, v2_ok))
    details = []
    for position, row in enumerate(anchors.itertuples(index=False)):
        count = counts.get(row.stock_code, 0)
        first = row.session_id + 1 - length
        missing_raw, invalid_window, outside_top, outside_p4 = [], [], [], []
        offending_bars, reference_fallbacks = [], []
        causal_start = max(0, first - FEATURE_LOOKBACK + 1)
        for session in range(causal_start, row.session_id + 1):
            item = raw.get((row.stock_code, session))
            day = days[session]
            if item is None or item[1] != "OK":
                offending_bars.append(f"{day}:{item[1] if item else 'MISSING_FORMAL_BAR'}")
            if (row.stock_code, session) in substitutions:
                reference_fallbacks.append(day)
            if session < first:
                continue
            rank = ranks.get((row.stock_code, session))
            if item is None:
                missing_raw.append(day)
            elif not item[0]:
                invalid_window.append(day)
            elif rank is None:
                outside_p4.append(day)
            elif rank > population.stats["top_n"]:
                outside_top.append(day)
        if count < 20:
            reason = "INSUFFICIENT_TRAINING_SEQUENCES"
        elif position not in v1_ok:
            reason = "HISTORICAL_SEQUENCE_GAPS"
        else:
            reason = "SCORED"
        details.append({
            "stock_code": row.stock_code, "trade_date": row.trade_date, "liquidity_rank": row.liquidity_rank,
            "v1_training_sequences": count, "v1_outcome": reason,
            "v1_sequence_available": position in v1_ok, "v2_price_sequence_available": position in v2_price_ok,
            "v2_causal_sequence_available": position in v2_ok,
            "v2_known_action_veto": position in v2_price_ok and position not in v2_ok,
            "missing_formal_context_dates": "|".join(missing_raw), "invalid_price_window_context_dates": "|".join(invalid_window),
            "outside_top_n_context_dates": "|".join(outside_top), "outside_p4_eligibility_context_dates": "|".join(outside_p4),
            "offending_bars_in_sequence_lookback": "|".join(offending_bars),
            "v2_auxiliary_reference_fallback_dates": "|".join(reference_fallbacks),
            **{f"v2_floor_{floor}_eligible": position in v2_ok and count >= floor for floor in SUPPORT_FLOORS},
        })
    return anchors, details


def diagnose(v1_model, output, asof, risk_file, actions_file=None):
    info = read_json(v1_model / "model_metadata.json")
    model_path = v1_model / "best_model.pt"
    if hash_file(model_path) != info["weights_sha256"]:
        raise ValueError("V1 model provenance mismatch")
    now = datetime.now(CHINA)
    current, raw = formal_snapshot(AShareDailyLoader().engine, now, asof)
    length = info["architecture"]["sequence_length"]
    risk = pd.read_csv(risk_file, dtype={"stock_code": "string"})
    actions = pd.read_csv(actions_file, dtype={"stock_code": "string"}) if actions_file is not None else None
    cfg = PoolConfig(**info["pool_config"])
    start, signal = current - timedelta(days=46), current.isoformat()
    strict = build_causal_population(raw, start, current, cfg, events=risk, reference_policy=VENDOR_REFERENCE)
    v2 = build_causal_population(raw, start, current, cfg, events=risk)
    v2_flags = price_window_flags(raw, strict.calendar, allow_auxiliary_fallback=True)
    anchors, details = refusal_diagnostics(strict, price_window_flags(raw, strict.calendar), signal,
                                          info["train_stock_counts"], length, known_actions=actions,
                                          v2_population=v2, v2_flags=v2_flags)
    if not len(anchors):
        raise ValueError(f"No P4 quality-qualified anchor on {current}")
    frame = pd.DataFrame(details)
    v2_anchors = v2.anchors.loc[v2.anchors.trade_date.eq(signal)].reset_index(drop=True)
    kept, indices = sequence_indexes(v2.features, v2_anchors, length, known_actions=actions, calendar=v2.calendar)
    covered = v2_anchors.iloc[kept][["stock_code", "trade_date", "liquidity_rank"]].copy()
    covered["v1_training_sequences"] = covered.stock_code.map(info["train_stock_counts"]).fillna(0).astype(int)
    covered["auxiliary_reference_fallback"] = v2.features.auxiliary_reference_fallback.to_numpy()[indices].any(axis=1)
    covered["independent_action_evidence_required"] = True
    for floor in SUPPORT_FLOORS:
        covered[f"floor_{floor}_eligible_preliminary"] = covered.v1_training_sequences.ge(floor)
    output.mkdir(parents=True)
    csv = output / "candidate_diagnostics.csv"
    frame.to_csv(csv, index=False)
    latest_candidates = output / "preliminary_v2_coverage.csv"
    covered.to_csv(latest_candidates, index=False)
    root_causes = Counter(failure for value in frame.offending_bars_in_sequence_lookback for failure in value.split("|") if failure)
    report = {"version": VERSION, "asof": current.isoformat(), "snapshot_clock_asia_shanghai": now.isoformat(),
              "v1_model_sha256": info["weights_sha256"], "v1_candidates": len(anchors),
              "v1_outcomes": {str(k): int(v) for k, v in frame.v1_outcome.value_counts().items()},
              "v1_support_floor_before_context": {str(f): int(frame.v1_training_sequences.ge(f).sum()) for f in SUPPORT_FLOORS},
              "v1_sequence_ready_without_support_filter": int(frame.v1_sequence_available.sum()),
              "v1_offending_bars_in_sequence_lookback": dict(root_causes),
              "v2_latest_anchor_count": len(v2_anchors), "v2_sequence_ready_without_support_filter": len(covered),
              "v2_preliminary_sequences_using_auxiliary_reference_fallback": int(covered.auxiliary_reference_fallback.sum()),
              "v2_preliminary_only_not_scoring_qualification": True,
              "v2_coverage_by_v1_training_support_floor": {
                  str(f): int(covered[f"floor_{f}_eligible_preliminary"].sum()) for f in SUPPORT_FLOORS},
              "company_action_evidence": "KNOWN_POINT_IN_TIME_EVENTS" if actions_file else "PRICE_ANOMALY_GUARD_ONLY",
              "company_action_coverage_certified": False,
              "risk_source_sha256": hash_file(risk_file),
              "actions_source_sha256": hash_file(actions_file) if actions_file else None,
              "candidate_diagnostics_sha256": hash_file(csv), "read_only_database": True,
              "preliminary_v2_coverage_sha256": hash_file(latest_candidates),
              "uses_future_labels": False, "strict_v1_population": strict.stats, "population": v2.stats}
    atomic_json(output / "latest_sequence_audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--asof", type=date.fromisoformat)
    parser.add_argument("--events", type=Path, default=RISK_FILE)
    parser.add_argument("--known-actions", type=Path, help="Optional stock_code,ex_date,published_on CSV")
    args = parser.parse_args()
    report = diagnose(args.v1_model, args.out, args.asof, args.events, args.known_actions)
    print(f"{report['asof']} V1 {report['v1_outcomes']}; V2 {report['v2_coverage_by_v1_training_support_floor']}", flush=True)


if __name__ == "__main__":
    main()
