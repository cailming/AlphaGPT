"""Archive separately versioned V1 and V2 pre-auction fund factor scores."""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .p4_action_pipeline import atomic_json, exclusive, read_json
from .p4_subset_inputs import hash_file, parquet_read, parquet_write
from .p55_daily import now_shanghai
from .p55_panel import validate_existing
from .p55_reports import timestamp
from .p58_fund_snapshot import AUCTION_START, CHINA, SOURCES, verified
from .p66_calendar import next_session, planned_open

VERSION = "P66_PREOPEN_FUND_V1"
FUNDS = ("stock_main_force_daily", "stock_main_force_daily_v2")
EVENT = "stock_signal_event"
BREAKOUT = ("NECKLINE_BREAKOUT", "MAIN_FORCE_PLATFORM_BREAKOUT_V2")
PULLBACK = ("MAIN_FORCE_PULLBACK", "OPEN30_STRONG_PULLBACK")
SPECS = {
    FUNDS[0]: {"total": "main_force_score", "activity": "activity_score", "holding": "holding_score",
               "low_distribution_risk": "-distribution_risk"},
    FUNDS[1]: {"total": "selection_score", "activity": "activity_score", "holding": None,
               "low_distribution_risk": None, "pullback_support": "pullback_score", "low_overheat_risk": "-overheat_risk"},
}


def freeze(out, now):
    path = out / "policy.json"
    spec = {"version": VERSION, "source_sha256": hash_file(Path(__file__)),
            "scores": SPECS, "universe": "timely frozen equal-weight research cohort",
            "first_observation": "first run from signal day 22:00 until next scheduled session 09:15; each source independently",
            "delta": "total minus latest trusted total for the preceding scheduled signal; matching source version",
            "composite": "0.8 * average percentile rank of total in eligible cohort + 0.1 * confirmed breakout + "
                         "0.1 * confirmed pullback with stock volume ratio under 1",
            "breakout_codes": list(BREAKOUT), "pullback_codes": list(PULLBACK), "required_status": "CONFIRMED",
            "volume_ratio": {FUNDS[0]: "vol_ratio_5_20", FUNDS[1]: "vol_ratio_3_20"},
            "rank_order": "higher scores preferred; stock code ascending breaks portfolio ties",
            "labels": "formal next open to sixth scheduled open, P55 economic evidence; development research only",
            "metrics": "Top-k fixed before labels; complete k and >=80% label coverage; costs 5/10/25bp per side"}
    if path.exists():
        previous = read_json(path)
        if {k:v for k,v in previous.items() if k != "frozen_at"} != spec:
            raise ValueError("Fund definition changed; create a new research policy")
    else:
        atomic_json(path, spec | {"frozen_at": now.isoformat()})
    return hash_file(path)


def cutoff(day):
    target = next_session(day)
    return datetime.combine(target, AUCTION_START, CHINA) if target else None


def snapshot_visible(rec, day, deadline, now):
    moments = [timestamp(rec[field]) for field in ("data_read_started_at", "snapshot_at", "files_ready_at")]
    started, seen, ready = moments
    if rec["signal_date"] != day.isoformat() or not started <= seen <= ready <= now or ready >= deadline:
        return False
    if rec["formal_stocks_at_observation"] < 1 or not rec.get("read_only_repeatable_read"):
        return False
    if rec["phase"] == "preopen":
        return (rec["availability"] == "NEXT_SESSION_BEFORE_OPENING_AUCTION" and
                rec.get("target_session") == deadline.date().isoformat() and started.date() == ready.date() == deadline.date())
    allowed = {"close": "BY_CLOSE_1800", "next-open": "EVENING_OBSERVATION_FOR_FOLLOWING_SESSION"}
    return (rec["phase"] in allowed and started.date() == ready.date() == day and rec["availability"] == allowed[rec["phase"]]
            and (seen.time() >= time(18) if rec["phase"] == "next-open" else ready.time() <= time(18)))


def latest_snapshot(root, day, now):
    limit = cutoff(day)
    if not limit:
        return None
    choices = []
    for path in (root / str(day)).glob("*/r*/manifest.json"):
        rec = read_json(path)
        if snapshot_visible(rec, day, limit, now):
            # Reject files installed after the decision boundary, even when copied
            # manifests carry an earlier observation time.
            members = [path, *(path.parent/(source+".parquet") for source in SOURCES)]
            completed = datetime.fromtimestamp(max(p.stat().st_mtime for p in members), CHINA)
            if completed < limit and completed <= now:
                choices.append((timestamp(rec["files_ready_at"]), path))
    if choices:
        path = max(choices)[1]
        return path, verified(path)


def read_source(snapshot, name):
    path, rec = snapshot
    table = parquet_read(path.parent / (name + ".parquet"))
    stats = rec["sources"][name]
    if len(table) != stats["rows"] or not set(rec["source_columns"][name]).issubset(table):
        raise ValueError("Snapshot schema or row count mismatch")
    if not (table.data_source.eq(name) & table.signal_date.eq(rec["signal_date"])
            & table.snapshot_at.eq(rec["snapshot_at"]) & table.source_version.eq(stats["source_version"])).all():
        raise ValueError("Source/date lineage differs from snapshot manifest")
    observed = timestamp(rec["snapshot_at"])
    def available(value):
        if pd.isna(value):
            return False
        dt = datetime.fromisoformat(value)
        return (dt.replace(tzinfo=CHINA) if dt.tzinfo is None else dt) <= observed
    valid = table.source_created_at.map(available).astype(bool)
    if "updated_at" in rec["source_columns"][name]:
        valid &= table.source_updated_at.isna() | table.source_updated_at.map(available)
    table = table.loc[valid]
    if name != EVENT and table.stock_code.duplicated().any():
        raise ValueError("Duplicate fund score for stock and source")
    return table


def cohort(panel, legacy, day, now):
    folder = panel / "timely" / str(day)
    parent = legacy / "timely" / str(day) / "manifest.json"
    rec = validate_existing(folder, hash_file(parent), hash_file(panel / "policy.json"))
    if (rec["trade_date"] != str(day) or rec["capture_kind"] != "TIMELY" or not rec["prospective_eligible"]
            or not rec["formal_session"] or timestamp(rec["committed_at"]) > now):
        raise ValueError("Expected a timely published research universe")
    def scores(arm):
        frame = parquet_read(folder / rec["models"][arm]["predictions_file"])
        if frame.stock_code.duplicated().any() or not frame.trade_date.eq(str(day)).all():
            raise ValueError("Invalid universe or benchmark keys")
        return frame.set_index("stock_code").score
    roster = scores("equal_weight")
    return pd.DataFrame({"ma20": scores("ma20_reversal").reindex(roster.index)}, index=roster.index), rec


def previous_signal(day):
    prior = day - timedelta(days=1)
    while planned_open(prior) is False:
        prior -= timedelta(days=1)
    return prior if planned_open(prior) else None


def factor_table(source, universe, latest, previous, events):
    """Use no historical mutable source. Preserve unsupported fields and abstentions."""
    rows = latest.set_index("stock_code").reindex(universe.index)
    output = universe.copy()
    for factor, column in SPECS[source].items():
        output[factor] = np.nan if column is None else pd.to_numeric(rows[column.lstrip("-")], errors="coerce")
        if column and column.startswith("-"):
            output[factor] *= -1
    output["delta"] = np.nan
    if previous is not None:
        old = previous.set_index("stock_code").reindex(output.index)
        same = rows.source_version.eq(old.source_version)
        if source == FUNDS[1]:
            same &= rows.algorithm_version.eq(old.algorithm_version)
        old_total = pd.to_numeric(old[SPECS[source]["total"]], errors="coerce")
        output["delta"] = (output.total - old_total).where(same)
    output["breakout_pullback"] = np.nan
    if not events.empty:
        confirmed = events.loc[events.signal_status.eq("CONFIRMED")]
        breakout = output.index.isin(confirmed.loc[confirmed.signal_code.isin(BREAKOUT), "stock_code"])
        pullback = output.index.isin(confirmed.loc[confirmed.signal_code.isin(PULLBACK), "stock_code"])
        ratio = pd.to_numeric(rows["vol_ratio_5_20" if source == FUNDS[0] else "vol_ratio_3_20"], errors="coerce")
        composite = .8 * output.total.rank(pct=True) + .1 * breakout + .1 * (pullback & ratio.lt(1))
        output["breakout_pullback"] = composite.where(~pullback | ratio.notna())
    return output.replace([np.inf, -np.inf], np.nan).rename_axis("stock_code").reset_index()


def score_summary(frame, source):
    result = {}
    for name in frame.columns.difference(["ma20", "stock_code"]):
        ranked = frame.dropna(subset=[name]).sort_values([name, "stock_code"], ascending=[False, True])
        result[name] = {"scored": len(ranked), "abstained": len(frame)-len(ranked),
                        "coverage": len(ranked)/len(frame) if len(frame) else None,
                        "unavailable_field": name in SPECS[source] and SPECS[source][name] is None,
                        "top10": ranked.stock_code.head(10).tolist(), "top20": ranked.stock_code.head(20).tolist()}
    return result


def read_capture(path, policy):
    rec = read_json(path)
    if (rec["policy_sha256"] != policy or rec["source"] != path.parent.name or rec["entry"] != path.parents[1].name
            or hash_file(path.parent/"scores.parquet") != rec["scores_sha256"]
            or timestamp(rec["captured_at"]) >= timestamp(rec["deadline"])):
        raise ValueError("Fund prediction archive failed integrity verification")
    return rec


def capture(out, root, panel, legacy, now, policy, clock=now_shanghai):
    recent = sorted((panel/"timely").glob("????-??-??/manifest.json"))
    if not recent:
        return {"status": "WAITING_FOR_TIMELY_PANEL"}
    day = date.fromisoformat(recent[-1].parent.name)
    limit = cutoff(day)
    if limit is None or not planned_open(day):
        return {"status": "UNCONFIRMED_TARGET_SESSION", "signal": str(day)}
    if now < datetime.combine(day, time(22), CHINA):
        return {"status": "WAITING_FOR_2200_FUND_WINDOW", "signal": str(day)}
    if now >= limit:
        return {"status": "PREOPEN_WINDOW_CLOSED", "signal": str(day)}
    snapshot = latest_snapshot(root/"fund", day, now)
    if not snapshot:
        return {"status": "WAITING_FOR_TRUSTED_SNAPSHOT", "signal": str(day)}
    universe, _ = cohort(panel, legacy, day, now)
    prior = previous_signal(day)
    old = latest_snapshot(root/"fund", prior, now) if prior else None
    events = read_source(snapshot, EVENT)
    lanes = {}
    for source in FUNDS:
        dest = out/"predictions"/str(limit.date())/source
        existing = dest/"manifest.json"
        if existing.is_file():
            read_capture(existing, policy)
            lanes[source] = "ALREADY_CAPTURED"
            continue
        current = read_source(snapshot, source)
        if current.empty:
            lanes[source] = "WAITING_FOR_AVAILABLE_SOURCE_ROWS"
            continue
        frame = factor_table(source, universe, current, read_source(old, source) if old else None, events)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=dest.parent) as tmp:
            stage = Path(tmp)/"capture"
            stage.mkdir()
            parquet_write(frame, stage/"scores.parquet")
            meta = {"policy_sha256": policy, "signal": str(day), "entry": str(limit.date()), "deadline": limit.isoformat(),
                    "source": source, "pool_size": len(universe), "track": "PREOPEN_RESEARCH", "factors": score_summary(frame, source),
                    "snapshot": str(snapshot[0]), "snapshot_sha256": hash_file(snapshot[0]), "snapshot_at": snapshot[1]["snapshot_at"],
                    "files_ready_at": snapshot[1]["files_ready_at"], "panel_sha256": hash_file(recent[-1]),
                    "prior_snapshot_sha256": hash_file(old[0]) if old else None, "scores_sha256": hash_file(stage/"scores.parquet")}
            finished = clock()
            if finished < now or finished >= limit:
                raise ValueError("Factor publication crossed the target session cutoff")
            atomic_json(stage/"manifest.json", meta | {"captured_at": finished.isoformat()})
            stage.rename(dest)
        lanes[source] = "CAPTURED"
    return {"status": "SOURCE_OUTCOMES", "signal": str(day), "entry": str(limit.date()), "sources": lanes}


def main():
    from .p66_fund_eval import evaluate
    from .daily import AShareDailyLoader
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "panel", "legacy"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    out = args.root / "p66/fund"
    with exclusive(out):
        now = now_shanghai()
        policy = freeze(out, now)
        observed = capture(out, args.root, args.panel, args.legacy, now, policy)
        reports = evaluate(out, args.legacy, policy, now, AShareDailyLoader().engine)
        folder = out/"observations"/str(now.date())
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder/(now.strftime("%H%M%S.%f")+".json"), {"observed_at":now.isoformat(), "capture":observed,
                    "evaluation":str(reports) if reports else None})
        print(observed, reports)


if __name__ == "__main__":
    main()
