"""Capture one immutable, auditable daily P5 prospective prediction session."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, time, timedelta
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from sqlalchemy import text

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, canonical, exclusive, read_json
from .p4_mvp import FEATURES, PoolConfig
from .p4_subset_dataset import FEATURE_LOOKBACK, GUARD
from .p4_subset_inputs import RISK_FILE, hash_file
from .p5_dataset import VERSION as DATASET_VERSION
from .p5_dataset import read_prior_inference_evidence, read_recent_evidence
from .p5_evidence import RECENT_SOURCES, capture_observations
from .p5_infer import SCORE_UNITS, select_latest
from .p5_sequences import CHINA, VERSION as SEQUENCE_POLICY
from .p5_sequences import build_causal_population, formal_snapshot, sequence_indexes


VERSION = "P55_DAILY_LEDGER_V1"
OBJECTIVES = ("absolute", "excess", "rank")
FORMAL_AVAILABILITY = time(18)
FINAL_FORMAL_PROBE = time(23, 30)
LABEL_OFFSET = 6
PREDICTION_COLUMNS = {
    "prediction_id": "String", "stock_code": "String", "trade_date": "String",
    "training_sequences": "Int64", "support_group": "String", "score": "Float64",
    "liquidity_rank": "Int64", "objective": "String", "score_units": "String",
    "model_version": "String", "model_sha256": "String",
    "auxiliary_reference_fallback_feature_sessions": "Int64",
}
CANDIDATE_COLUMNS = ("stock_code", "trade_date", "liquidity_rank", "training_sequences", "outcome")


class CaptureDeadlineExceeded(RuntimeError):
    """The job finished after the immutable same-session capture window."""


def now_shanghai():
    return datetime.now(CHINA)


def parse_clock(value):
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise ValueError("Daily observation timestamp requires a timezone")
    return moment.astimezone(CHINA)


def is_timely(requested, started, completed):
    """A same-session after-close publication is before the next possible open."""
    started, completed = (moment.astimezone(CHINA) for moment in (started, completed))
    return (requested is None and started.date() == completed.date()
            and started <= completed and started.time() >= FORMAL_AVAILABILITY)


def support_group(count):
    if count < 0:
        raise ValueError("Training sequence support cannot be negative")
    return "0-4" if count < 5 else "5-19" if count < 20 else "20+"


def prediction_id(kind, trade_day, objective, weights_sha256, stock):
    return hashlib.sha256(canonical([VERSION, kind, trade_day, objective, weights_sha256, stock])).hexdigest()


def formal_row_count(engine, day):
    query = text("""
        SELECT count(DISTINCT stock_code) FROM public.market
        WHERE trade_date = :day AND stock_code IS NOT NULL AND not_billing = 1
    """)
    with engine.connect() as conn:
        with conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            if conn.scalar(text("SHOW transaction_read_only")) != "on":
                raise RuntimeError("Daily formal row probe requires a read-only transaction")
            return int(conn.scalar(query, {"day": day}) or 0)


def verify_file(path, digest):
    if hash_file(path) != digest:
        raise ValueError(f"Frozen daily dependency failed digest verification: {path}")


def frozen_metadata(source, models):
    from .p5_train import VERSION as MODEL_VERSION

    audit_path = source / "dataset/subset_audit.json"
    audit = read_json(audit_path)
    audit_hash = hash_file(audit_path)
    if audit["version"] != DATASET_VERSION or audit["v2_sequence_policy"] != SEQUENCE_POLICY:
        raise ValueError("Daily capture requires the original V2 frozen sequence policy")
    shared = ("dataset_audit_sha256", "feature_names", "mean", "scale", "architecture",
              "pool_config", "v2_sequence_policy", "selection_last_exit", "train_stock_counts", "support_floor")
    results = {}
    for objective in OBJECTIVES:
        model_dir = models / objective
        metadata_path = model_dir / "model_metadata.json"
        meta = read_json(metadata_path)
        verify_file(model_dir / "best_model.pt", meta["weights_sha256"])
        policy_path = model_dir / "evaluation_plan.json"
        verify_file(policy_path, meta["evaluation_plan_sha256"])
        policy = read_json(policy_path)
        if (meta["objective"] != objective or meta["version"] != MODEL_VERSION
                or meta["dataset_audit_sha256"] != audit_hash
                or meta["v2_sequence_policy"] != SEQUENCE_POLICY or tuple(meta["feature_names"]) != FEATURES
                or meta["prospective"] != policy["prospective"] or meta["support_floor"] != policy["support_floor"]):
            raise ValueError(f"Frozen daily model contract differs: {objective}")
        if results and any(meta[key] != results["absolute"][key] for key in shared):
            raise ValueError("Frozen daily model set does not share features, eligibility and training provenance")
        parse_clock(meta["frozen_at_utc"])
        results[objective] = meta | {"model_metadata_sha256": hash_file(metadata_path)}
    return audit_hash, results


def checked_manifest(path):
    record = read_json(path / "manifest.json")
    if record["schema_version"] != VERSION:
        raise ValueError("Unexpected daily ledger schema")
    for model in record["models"].values():
        verify_file(path / model["predictions_file"], model["predictions_sha256"])
        verify_file(path / model["candidates_file"], model["candidates_sha256"])
    return record


def pending_stocks(ledger, calendar, signal):
    """Maintain evidence for each authoritative daily cohort through its T+6 close."""
    import polars as pl

    dates = list(calendar.strftime("%Y-%m-%d"))
    signal_index = dates.index(signal)
    starts = {}
    for day in dates[max(0, signal_index - LABEL_OFFSET):signal_index]:
        path = ledger / "timely" / day
        if not path.is_dir():
            continue
        record = checked_manifest(path)
        if not record["formal_session"] or record["capture_kind"] != "TIMELY" or record["status"] != "SCORED":
            continue
        for model in record["models"].values():
            if not model["rows"]:
                continue
            predictions = pl.read_parquet(path / model["predictions_file"], columns=["stock_code"])
            for code in predictions["stock_code"].to_list():
                starts[code] = min(day, starts.get(code, day))
    return starts


def observation_intervals(population, signal, length, prior, pending, limit):
    """Compute only missing, adjacent evidence intervals for the bounded active cohort."""
    if not 1 <= limit <= 500:
        raise ValueError("Daily company-action evidence requires a stock cap between 1 and 500")
    anchors = population.anchors.loc[population.anchors.trade_date.eq(signal)].reset_index(drop=True)
    positions, _ = sequence_indexes(population.features, anchors, length)
    days = population.calendar.strftime("%Y-%m-%d")
    current = {}
    for row in anchors.iloc[positions].sort_values(["liquidity_rank", "stock_code"]).itertuples(index=False):
        first = row.session_id - length - FEATURE_LOOKBACK + 2
        if first >= 0:
            current[row.stock_code] = str(days[first])
    admitted = prior["stocks"].keys() | pending.keys()
    ranked = dict(pending)
    for code, start in current.items():
        if code in admitted:
            ranked[code] = min(start, ranked.get(code, start))
    from . import p4_action_pipeline as lineage

    grouped = defaultdict(list)
    selected = 0
    uncovered = 0
    already_covered = 0
    for code, first in ranked.items():
        provider = lineage.bao._provider_code(code)
        if provider is None:
            uncovered += 1
            continue
        previous = prior["stocks"].get(code)
        if previous is None:
            missing = [(first, signal)]
        else:
            start, end = (date.fromisoformat(previous[key]) for key in ("covered_from", "covered_through"))
            if start > end:
                raise ValueError(f"Invalid existing observation range for {code}")
            missing = []
            if first < start.isoformat():
                missing.append((first, min(signal, (start - timedelta(days=1)).isoformat())))
            if signal > end.isoformat():
                missing.append(((end + timedelta(days=1)).isoformat(), signal))
        if not missing:
            already_covered += 1
            continue
        if any((date.fromisoformat(ending) - date.fromisoformat(beginning)).days > 140
               for beginning, ending in missing):
            uncovered += 1
            continue
        if selected >= limit:
            uncovered += 1
            continue
        selected += 1
        for beginning, ending in missing:
            grouped[beginning, ending].append({"stock_code": code, "provider_code": provider})
    plans = []
    for (start, end), rows in sorted(grouped.items()):
        if start > end:
            raise ValueError("Daily evidence request would have an inverted interval")
        for begin in range(0, len(rows), 220):
            plans.append({"kind": "P55_DAILY_BOUNDED", "start": start, "end": end, "providers": list(RECENT_SOURCES),
                          "jobs": rows[begin:begin + 220]})
    return plans, {"causal_price_ready_stocks": len(current), "previous_cohort_stocks": len(pending),
                   "admitted_roster_stocks": len(admitted), "price_ready_outside_roster": len(current.keys() - admitted),
                   "already_covered_stocks": already_covered, "planned_stocks": selected,
                   "unplanned_missing_stocks": uncovered}


def merge_observed_interval(previous, start, end, actions, dual):
    """A source outage never lengthens a stock's independently observed interval."""
    if previous is None and not dual:
        return None
    active = set(map(tuple, previous["actions"])) if previous else set()
    active.update((day, day) for day in actions)
    if previous is None:
        return {"covered_from": start, "covered_through": end, "actions": sorted(active)}
    result = previous.copy()
    if dual:
        old_start = date.fromisoformat(previous["covered_from"])
        old_end = date.fromisoformat(previous["covered_through"])
        new_start, new_end = date.fromisoformat(start), date.fromisoformat(end)
        if new_start > old_end + timedelta(days=1) or old_start > new_end + timedelta(days=1):
            raise ValueError("Source coverage extension would bridge an unavailable interval")
        result["covered_from"] = min(old_start, new_start).isoformat()
        result["covered_through"] = max(old_end, new_end).isoformat()
    result["actions"] = sorted(active)
    return result


def choose_evidence(ledger, seed, at):
    """Find an immutable successful snapshot and verify its signed envelope."""
    candidates = sorted((ledger / "evidence").glob("*/dataset/subset_audit.json"), reverse=True)
    candidates += [seed / "dataset/subset_audit.json"]
    for audit in candidates:
        folder = audit.parent.parent
        evidence, digest = read_prior_inference_evidence(folder)
        if evidence["observed_at_utc"] is None:
            continue
        if parse_clock(evidence["observed_at_utc"]) <= at:
            return folder, evidence, digest
    raise ValueError("No verified company-action evidence was available by capture time")


def refresh_evidence(ledger, target, population, length, prior_folder, prior, digest, limit):
    signal = target.isoformat()
    output = ledger / "evidence" / signal
    if (output / "dataset/subset_audit.json").is_file():
        evidence, observed_digest = read_prior_inference_evidence(output)
        return output, evidence, observed_digest
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "request_plan.json"
    if plan_path.exists():
        plan_root = read_json(plan_path)
        if plan_root["parent_evidence_sha256"] != digest or plan_root["trade_date"] != signal:
            raise ValueError("Retry must use the same frozen provider request plan and prior evidence")
    else:
        pending = pending_stocks(ledger, population.calendar, signal)
        plans, stats = observation_intervals(population, signal, length, prior, pending, limit)
        plan_root = {"trade_date": signal, "parent_evidence_sha256": digest, "batches": plans, "counts": stats}
        atomic_json(plan_path, plan_root)
    if not plan_root["batches"]:
        return prior_folder, prior, digest
    stocks = {code: row.copy() for code, row in prior["stocks"].items()}
    timestamps = [prior["observed_at_utc"]]
    facts = []
    for number, plan in enumerate(plan_root["batches"], 1):
        inputs = output / f"batch-{number:04d}" / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        batch_plan = inputs / "evidence_plan.json"
        if not batch_plan.exists():
            atomic_json(batch_plan, plan)
        if read_json(batch_plan) != plan:
            raise ValueError("Daily evidence batch plan changed before retry")
        inputs_audit = inputs / "inputs.json"
        if not inputs_audit.exists():
            atomic_json(inputs_audit, {"files": {"evidence_plan.json": hash_file(batch_plan)}})
        completed = inputs.parent / "capture_done.json"
        if not completed.exists():
            capture_observations(inputs)
            atomic_json(completed, {"finished_at": now_shanghai().isoformat()})
        _, dual, action_days, report = read_recent_evidence(inputs)
        facts.append(report)
        if report["latest_observation_utc"]:
            timestamps.append(report["latest_observation_utc"])
        for job in plan["jobs"]:
            code = job["stock_code"]
            observation = merge_observed_interval(stocks.get(code), plan["start"], plan["end"],
                                                  action_days.get(code, ()), code in dual)
            if observation is not None:
                stocks[code] = observation
    result = prior | {"stocks": stocks, "max_covered_date": max(
        [prior["max_covered_date"]] + [item["covered_through"] for item in stocks.values()]),
        "observed_at_utc": max((t for t in timestamps if t), key=parse_clock)}
    folder = output / "dataset"
    # A partial export has no self-validating manifest; replace only its uncommitted generated file.
    folder.mkdir(exist_ok=True)
    evidence = folder / "inference_evidence.json"
    atomic_json(evidence, result)
    after_digest = hash_file(evidence)
    atomic_json(folder / "subset_audit.json", {"version": DATASET_VERSION, **GUARD,
                "audit_kind": "DAILY_INCREMENTAL_INFERENCE_EVIDENCE", "v2_sequence_policy": SEQUENCE_POLICY,
                "parent_inference_evidence_sha256": digest, "plan_sha256": hash_file(plan_path),
                "plan_counts": plan_root["counts"], "action_evidence_batches": facts,
                "files": {evidence.name: after_digest}})
    return output, result, after_digest


def write_probe(ledger, day, moment, status, detail):
    folder = ledger / "probes" / day.isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (moment.strftime("%H%M%S.%f") + ".json")
    record = {"schema_version": VERSION, "trade_date": day.isoformat(), "probed_at": moment.isoformat(),
              "status": status, "reason": detail, "authoritative": False}
    atomic_json(path, record)
    return record


def publish_day(destination, manifest, metadata, details, scores, clock):
    import polars as pl

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return checked_manifest(destination)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        started_at = parse_clock(manifest["capture_started_at"])
        signal = date.fromisoformat(manifest["trade_date"])
        requested = signal if manifest["requested_historical"] else None
        # Tables are built with provisional ids, then validated against the actual commit timestamp.
        for objective in OBJECTIVES:
            info = metadata[objective]
            values = scores[objective]
            prediction_file = stage / f"predictions_{objective}.parquet"
            schema = {key: getattr(pl, value) for key, value in PREDICTION_COLUMNS.items()}
            table = pl.DataFrame({column: values[column].to_list() for column in schema}, schema=schema)
            table.write_parquet(prediction_file, compression="zstd")
            candidate_file = stage / f"candidates_{objective}.csv"
            details.to_csv(candidate_file, index=False)
            manifest["models"][objective] = {
                "model_version": info["version"], "weights_sha256": info["weights_sha256"],
                "evaluation_plan_sha256": info["evaluation_plan_sha256"], "model_frozen_at": info["frozen_at_utc"],
                "model_metadata_sha256": info["model_metadata_sha256"],
                "rows": len(values), "abstentions": len(details) - len(values),
                "abstention_reasons": {str(reason): int(count) for reason, count in
                                       details.loc[details.outcome.ne("SCORED")].outcome.value_counts().items()},
                "predictions_file": prediction_file.name, "predictions_sha256": hash_file(prediction_file),
                "candidates_file": candidate_file.name, "candidates_sha256": hash_file(candidate_file),
            }
            for output in (prediction_file, candidate_file):
                with output.open("rb") as handle:
                    os.fsync(handle.fileno())
        committed = clock()
        kind = "TIMELY" if is_timely(requested, started_at, committed) else "HISTORICAL"
        missed = manifest["status"] == "MISSED_CAPTURE_WINDOW"
        if kind != manifest["capture_kind"] and not missed:
            raise CaptureDeadlineExceeded("Capture crossed the same-session boundary; record an abstention")
        observed = manifest["evidence_snapshot"].get("observed_at_utc")
        if observed is not None and parse_clock(observed) > committed:
            raise ValueError("Company-action evidence was observed after the declared publish time")
        manifest["committed_at"] = committed.isoformat()
        manifest["prospective_eligible"] = bool(kind == "TIMELY" and not missed and manifest["formal_session"] and
            all(signal > parse_clock(item["frozen_at_utc"]).date() for item in metadata.values()))
        manifest["published_in_after_close_window"] = kind == "TIMELY"
        atomic_json(stage / "manifest.json", manifest)
        if destination.exists():
            raise RuntimeError("Daily prediction already committed")
        os.rename(stage, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def fitted_scores(models, metadata, population, details, positions, indices, signal, kind):
    import torch

    from .p4_transformer_train import predict, scale_sequences
    from .p5_train import load_p5_model

    if len(positions):
        feature_matrix = population.features[list(FEATURES)].to_numpy(dtype=float)
        reference = metadata["absolute"]
        values = scale_sequences(feature_matrix, indices, reference["mean"], reference["scale"])
        rows = details.iloc[positions][["stock_code", "trade_date", "liquidity_rank", "training_sequences",
                                        "auxiliary_reference_fallback_feature_sessions"]].copy()
        rows["support_group"] = rows.training_sequences.map(support_group)
    else:
        rows = pd.DataFrame(columns=[column for column in PREDICTION_COLUMNS
                                    if column not in {"prediction_id", "score", "objective", "score_units",
                                                      "model_version", "model_sha256"}])
    result = {}
    for objective, info in metadata.items():
        if len(positions):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, checkpoint = load_p5_model(models / objective, device)
            verify_file(models / objective / "model_metadata.json", info["model_metadata_sha256"])
            if checkpoint["weights_sha256"] != info["weights_sha256"]:
                raise ValueError("Model changed after daily inference validation")
            scores = predict(model, values, device, info["architecture"]["batch_size"]).astype(float)
            if not np.isfinite(scores).all():
                raise ValueError("Frozen daily model emitted non-finite scores")
        else:
            scores = np.empty(0, dtype=float)
        result[objective] = rows.assign(score=scores, objective=objective, score_units=SCORE_UNITS[objective],
                                       model_version=info["version"], model_sha256=info["weights_sha256"],
                                       prediction_id=[prediction_id(kind, signal, objective, info["weights_sha256"], code)
                                                      for code in rows.stock_code]).sort_values(
                                           ["score", "stock_code"], ascending=[False, True])
    return result


def daily(source, models, seed, ledger, *, requested=None, refresh=True, events=RISK_FILE, limit=400,
          clock=now_shanghai, engine=None):
    started = clock()
    signal = requested or started.date()
    if signal > started.date():
        raise ValueError("Daily capture cannot read a future date")
    kind = "HISTORICAL" if requested is not None else "TIMELY"
    ledger.mkdir(parents=True, exist_ok=True)
    with exclusive(ledger):
        destination = ledger / kind.lower() / signal.isoformat()
        if destination.exists():
            return checked_manifest(destination)
        if started.date() == signal and started.time() < FORMAL_AVAILABILITY:
            return write_probe(ledger, signal, started, "DEFERRED", "BEFORE_FORMAL_CLOSE_BOUNDARY")
        engine = AShareDailyLoader().engine if engine is None else engine
        formal_rows = formal_row_count(engine, signal)
        if not formal_rows and requested is None and started.time() < FINAL_FORMAL_PROBE:
            return write_probe(ledger, signal, started, "DEFERRED", "NO_SAME_DAY_FORMAL_OHLCV_YET")

        source_sha, metadata = frozen_metadata(source, models)
        if any(parse_clock(info["frozen_at_utc"]) > started for info in metadata.values()):
            raise ValueError("Frozen model records a creation time after the beginning of capture")
        manifest = {"schema_version": VERSION, "trade_date": signal.isoformat(), "capture_kind": kind,
                    "formal_session": bool(formal_rows), "market_rows": formal_rows,
                    "capture_started_at": started.isoformat(), "requested_historical": requested is not None,
                    "feature_cutoff": None, "price_cutoff": None, "source_audit_sha256": source_sha,
                    "risk_events_sha256": hash_file(events), "uses_future_labels": False, "research_only": True,
                    "models": {}, "evidence_snapshot": {}}
        if formal_rows:
            current, prices = formal_snapshot(engine, started, asof=signal)
            price_observed_at = clock()
            if current != signal:
                raise ValueError("Daily capture must observe the exact formal trade date")
            info = metadata["absolute"]
            if current.isoformat() <= info["selection_last_exit"]:
                raise ValueError("Daily signals may not overlap the model-selection label window")
            risk = pd.read_csv(events, dtype={"stock_code": "string"})
            population = build_causal_population(prices, current - timedelta(days=46), current,
                                                PoolConfig(**info["pool_config"]), events=risk)
            location, proof, digest = choose_evidence(ledger, seed, clock())
            length = info["architecture"]["sequence_length"]
            if refresh and requested is None:
                location, proof, digest = refresh_evidence(ledger, signal, population, length, location, proof, digest, limit)
            details, positions, indices, counts = select_latest(population, signal.isoformat(), info, proof)
            scores = fitted_scores(models, metadata, population, details, positions, indices, signal.isoformat(), kind)
            manifest.update(status="SCORED" if len(positions) else "ABSTAINED",
                            feature_cutoff=current.isoformat(), price_cutoff=current.isoformat(),
                            price_snapshot_completed_at=price_observed_at.isoformat(), source_price_rows=len(prices),
                            qualified_by_training_support_floor=counts,
                            auxiliary_reference_policy=population.stats["auxiliary_reference_policy"],
                            evidence_snapshot={"root": str(location.resolve()), "sha256": digest,
                                               "max_covered_date": proof["max_covered_date"],
                                               "observed_at_utc": proof["observed_at_utc"]})
        else:
            reason = ("NO_FORMAL_OHLCV_FOR_RETROSPECTIVE_DATE" if requested is not None
                      else "NO_SAME_DAY_FORMAL_OHLCV_OBSERVED_BY_FINAL_PROBE")
            manifest.update(status="NO_FORMAL_SESSION", reason=reason)
            details = pd.DataFrame(columns=CANDIDATE_COLUMNS)
            scores = fitted_scores(models, metadata, None, details, [], None, signal.isoformat(), kind)
        try:
            return publish_day(destination, manifest, metadata, details, scores, clock)
        except CaptureDeadlineExceeded:
            manifest = manifest | {"status": "MISSED_CAPTURE_WINDOW", "reason": "COMMIT_AFTER_CAPTURE_DEADLINE",
                                   "models": {}}
            details = details.assign(outcome="COMMIT_AFTER_CAPTURE_DEADLINE")
            scores = fitted_scores(models, metadata, None, details, [], None, signal.isoformat(), kind)
            return publish_day(destination, manifest, metadata, details, scores, clock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "models", "evidence-seed", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--asof", type=date.fromisoformat, help="Publish a separate, ineligible retrospective capture")
    parser.add_argument("--events", type=Path, default=RISK_FILE)
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--evidence-limit", type=int, default=400)
    args = parser.parse_args()
    invoked = now_shanghai()
    try:
        record = daily(args.source, args.models, args.evidence_seed, args.out, requested=args.asof,
                       refresh=not args.no_refresh, events=args.events, limit=args.evidence_limit)
    except Exception as error:
        write_probe(args.out, args.asof or invoked.date(), now_shanghai(), "ERROR", type(error).__name__)
        raise
    print({"status": record["status"], "trade_date": record["trade_date"],
           "prospective_eligible": record.get("prospective_eligible", False),
           "model_rows": {key: value["rows"] for key, value in record.get("models", {}).items()}}, flush=True)


if __name__ == "__main__":
    main()
