"""Independent, immutable P5.5-B research journal paired to frozen V2 day manifests."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, canonical, exclusive, read_json
from .p4_mvp import FEATURES, PoolConfig
from .p4_subset_dataset import GUARD
from .p4_subset_inputs import RISK_FILE, hash_file, parquet_read
from .p4_transformer_train import load_model, predict, scale_sequences, verify_digest
from .p53_experiments import FACTOR_MODEL_VERSION, factor_reader
from .p54_factors import FACTOR_VERSION, P54_FEATURES, EXTRA_FEATURES, build_market_factors
from .p55_daily import (PREDICTION_COLUMNS, CaptureDeadlineExceeded, checked_manifest, is_timely,
                        now_shanghai, parse_clock, support_group)
from .p5_infer import select_latest
from .p5_sequences import VERSION as SEQUENCE_POLICY, build_causal_population, formal_snapshot

VERSION = "P55_DAILY_PANEL_V1"
ARMS = ("factors18", "ma20_reversal", "equal_weight")
DEFINITIONS = {
    "factors18": {"score_units": "five_session_economic_return", "portfolio_kind": "RANKED",
                  "version": FACTOR_MODEL_VERSION, "definition": "P54 18-factor frozen absolute-return Transformer"},
    "ma20_reversal": {"score_units": "negative_close_ma20_dev", "portfolio_kind": "RANKED",
                      "version": "P55_MA20_REVERSAL_V1", "definition": "negate canonical close_ma20_dev"},
    "equal_weight": {"score_units": "unranked_equal_weight", "portfolio_kind": "EQUAL_WEIGHT",
                     "version": "P55_EQUAL_WEIGHT_V1", "definition": "0 score, 1/n weight across eligible frozen cohort"},
}
COLUMNS = PREDICTION_COLUMNS | {"portfolio_weight": "Float64", "qualification": "String"}
KEYS = ["stock_code", "trade_date"]


class PriceContextUnavailable(ValueError):
    """An exact-session formal feature snapshot cannot qualify as scoring input."""


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def dependencies(factor_model, factor_source):
    audit_path = factor_source / "dataset/subset_audit.json"
    audit = factor_reader(audit_path.parent)
    meta_file = factor_model / "model_metadata.json"
    meta = read_json(meta_file)
    verify_digest(factor_model / "best_model.pt", meta["weights_sha256"])
    verify_digest(factor_model / "evaluation_plan.json", meta["evaluation_plan_sha256"])
    if (meta["version"] != FACTOR_MODEL_VERSION or meta["experiment_arm"] != "factors18"
            or meta["objective"] != "absolute" or meta["dataset_version"] != FACTOR_VERSION
            or tuple(meta["feature_names"]) != P54_FEATURES
            or meta["dataset_audit_sha256"] != hash_file(audit_path)
            or meta["v2_sequence_policy"] != SEQUENCE_POLICY
            or meta["architecture"]["sequence_length"] != audit["sequence_eligibility_length"]
            or meta["architecture"]["feature_count"] != len(P54_FEATURES)
            or meta["support_floor"] != 0
            or any(meta[key] != value for key, value in GUARD.items())):
        raise ValueError("P55 panel expected the frozen paired 18-factor absolute model and its data")
    if meta["training_identity"]["preflight_sha256"] != hash_file(factor_source / "transformer/transformer_preflight.json"):
        raise ValueError("P55 panel source sequence contract changed")
    factor_audit = factor_source / "factor_audit.json"
    verify_digest(factor_audit, audit["factor_audit_sha256"])
    min_stocks = read_json(factor_audit)["minimum_market_pair_stocks"]
    for key in ("mean", "scale"):
        vector = np.asarray(meta[key], dtype=float)
        if vector.shape != (len(P54_FEATURES),) or not np.isfinite(vector).all() or (key == "scale" and (vector <= 0).any()):
            raise ValueError("Malformed frozen factor normalization")
    parse_clock(meta["frozen_at_utc"])
    return meta, {
        "factor_model_metadata_sha256": hash_file(meta_file),
        "factor_model_weights_sha256": meta["weights_sha256"],
        "factor_model_policy_sha256": meta["evaluation_plan_sha256"],
        "factor_source_audit_sha256": hash_file(audit_path),
        "factor_builder_source_sha256": hash_file(Path(build_market_factors.__code__.co_filename)),
        "factor_market_minimum_stocks": min_stocks,
    }


def freeze_panel(panel, factor_model, factor_source, *, clock=now_shanghai):
    """Freeze this experiment before live recording; a changed policy needs a new ledger."""
    meta, dep = dependencies(factor_model, factor_source)
    moment = clock()
    if parse_clock(meta["frozen_at_utc"]) > moment:
        raise ValueError("The factor model freeze time has not arrived")
    record = {
        "version": VERSION, "dependencies": dep, "definitions": DEFINITIONS,
        "eligibility": "original frozen daily V2 qualified candidates rechecked from exact-session inputs; "
                       "18-factor arm requires finite paired-training factors throughout the sequence",
        "tie_rule": "score descending then stock code ascending for ranked arms; equal weight has no Top-K",
        "provenance": "day anchor is a verified original V2 manifest; same-day after-close panel commit required",
        "version_freeze": "trade date strictly after the newest model and panel policy freeze date",
        "uses_future_labels": False, "research_only": True,
    }
    panel.mkdir(parents=True, exist_ok=True)
    path = panel / "policy.json"
    with exclusive(panel):
        if path.exists():
            current = read_json(path)
            if {k: v for k, v in current.items() if k != "frozen_at"} != record:
                raise ValueError("Panel policy changed; a new independent panel ledger is required")
            parse_clock(current["frozen_at"])
            return current
        record["frozen_at"] = moment.isoformat()
        atomic_json(path, record)
    return record


def empty_details():
    return pd.DataFrame(columns=[*KEYS, "liquidity_rank", "training_sequences", "outcome",
                                 "auxiliary_reference_fallback_feature_sessions"])


def legacy_details(day, legacy):
    spec = legacy["models"]["absolute"]
    candidates = pd.read_csv(day / spec["candidates_file"], dtype={**dict.fromkeys(KEYS, "string"), "outcome": "string"},
                             keep_default_na=False)
    prediction = parquet_read(day / spec["predictions_file"])
    if (not set(empty_details().columns[:5]).issubset(candidates) or candidates.stock_code.duplicated().any()
            or not candidates.stock_code.str.fullmatch(r"\d{6}").all()
            or not candidates.trade_date.eq(legacy["trade_date"]).all() or not candidates.outcome.str.len().gt(0).all()
            or len(candidates) != spec["rows"] + spec["abstentions"]
            or set(candidates.loc[candidates.outcome.eq("SCORED"), "stock_code"]) != set(prediction.stock_code)):
        raise ValueError("Original daily candidate and prediction cohorts disagree")
    if "auxiliary_reference_fallback_feature_sessions" not in candidates:
        candidates["auxiliary_reference_fallback_feature_sessions"] = 0
    return candidates


def verified_evidence(legacy, now):
    record = legacy["evidence_snapshot"]
    path = Path(record["root"]) / "dataset/inference_evidence.json"
    verify_digest(path, record["sha256"])
    evidence = read_json(path)
    if (evidence["policy"] != "P5_DUAL_OBSERVED_RESEARCH_WINDOW_V1"
            or evidence["observed_at_utc"] != record["observed_at_utc"]
            or parse_clock(record["observed_at_utc"]) > now):
        raise ValueError("Panel evidence does not match the original chronological snapshot")
    return evidence


def fetch_panel_features(engine, now, signal, info, legacy, details, events):
    """Recheck original accepted names under a fresh exact signal-close snapshot."""
    try:
        raw_day, raw = formal_snapshot(engine, now, asof=date.fromisoformat(signal))
    except ValueError as error:
        raise PriceContextUnavailable(str(error)) from error
    if raw_day.isoformat() != signal:
        raise PriceContextUnavailable("Formal snapshot changed signal date")
    risk = pd.read_csv(events, dtype={"stock_code": "string"})
    try:
        population = build_causal_population(raw, raw_day - timedelta(days=46), raw_day,
                                             PoolConfig(**info["pool_config"]), events=risk)
    except ValueError as error:
        raise PriceContextUnavailable(str(error)) from error
    gate = info | {"support_floor": 0}
    refreshed, positions, indices, _ = select_latest(population, signal, gate, verified_evidence(legacy, now))
    outcome = refreshed.set_index("stock_code").outcome.to_dict()
    rows = details.copy()
    initially_scored = rows.outcome.eq("SCORED")
    local = rows.loc[initially_scored, "stock_code"].map(outcome).fillna("OUTSIDE_RECOMPUTED_CAUSAL_POOL")
    rows.loc[initially_scored, "outcome"] = local
    accepted = set(rows.loc[rows.outcome.eq("SCORED"), "stock_code"])
    keep = refreshed.iloc[positions].stock_code.isin(accepted).to_numpy()
    return rows, population, indices[keep], raw, refreshed.iloc[positions[keep]].stock_code.to_list()


def factor_sequences(raw, population, indices, minimum):
    if not len(indices):
        return np.empty((0, len(P54_FEATURES))), np.zeros(0, dtype=bool), indices, {}, []
    used = np.unique(indices)
    base = population.features.iloc[used].reset_index(drop=True)
    factors, audit = build_market_factors(raw, base[KEYS], min_market_stocks=minimum)
    if not (factors.stock_code.astype(str).equals(base.stock_code.astype(str))
            and factors.trade_date.astype(str).equals(base.trade_date.astype(str))):
        raise ValueError("P54 feature builder changed the live sequence key order")
    matrix = np.column_stack([base[list(FEATURES)].to_numpy(dtype=float),
                              factors[list(EXTRA_FEATURES)].to_numpy(dtype=float)])
    remapped = np.searchsorted(used, indices)
    finite = np.isfinite(matrix[remapped]).all(axis=1)
    missing = ["|".join(name for name, ok in zip(P54_FEATURES, available, strict=True) if not ok)
               for available in finite]
    return matrix, finite.all(axis=1), remapped, audit, missing


def arm_scores(rows, population, indices, raw, codes, factor_model, meta, policy):
    details = {arm: rows.copy() for arm in ARMS}
    scores = {arm: {} for arm in ARMS}
    for arm in ARMS:
        details[arm]["missing_factors"] = ""
    if not len(codes):
        return details, scores, {}
    anchor_features = population.features.iloc[indices[:, -1]]
    ma20 = -anchor_features.close_ma20_dev.to_numpy(dtype=float)
    ma20_rows = dict(zip(codes, ma20, strict=True))
    scores["ma20_reversal"] = {key: value for key, value in ma20_rows.items() if np.isfinite(value)}
    missing_ma20 = set(codes) - scores["ma20_reversal"].keys()
    details["ma20_reversal"].loc[rows.stock_code.isin(missing_ma20), "outcome"] = "MA20_NONFINITE"
    scores["equal_weight"] = dict.fromkeys(codes, 0.)
    factor_details = details["factors18"]
    factor_details["training_sequences"] = factor_details.stock_code.map(meta["train_stock_counts"]).fillna(0).astype(int)
    if rows.trade_date.iloc[0] <= meta["selection_last_exit"]:
        factor_details.loc[factor_details.outcome.eq("SCORED"), "outcome"] = "NOT_AFTER_FACTOR_MODEL_SELECTION_WINDOW"
        return details, scores, {}
    try:
        matrix, valid, remapped, audit, reasons = factor_sequences(
            raw, population, indices, policy["dependencies"]["factor_market_minimum_stocks"])
    except ValueError as error:
        factor_details.loc[factor_details.outcome.eq("SCORED"), "outcome"] = "FACTOR_INPUT_DATA_INVALID"
        return details, scores, {"issue": str(error)}
    missing = {code: reason for code, reason in zip(codes, reasons, strict=True) if reason}
    mask = factor_details.stock_code.isin(missing)
    factor_details.loc[mask, "outcome"] = "FACTOR_SEQUENCE_INCOMPLETE"
    factor_details.loc[mask, "missing_factors"] = factor_details.loc[mask, "stock_code"].map(missing)
    if valid.any():
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, check = load_model(factor_model, device, version=FACTOR_MODEL_VERSION, feature_names=P54_FEATURES)
        if check["weights_sha256"] != meta["weights_sha256"]:
            raise ValueError("Paired 18-factor dependency changed before prediction")
        scaled = scale_sequences(matrix, remapped[valid], meta["mean"], meta["scale"])
        predictions = predict(model, scaled, device, meta["architecture"]["batch_size"])
        if not np.isfinite(predictions).all():
            raise ValueError("Panel emitted nonfinite scores")
        scores["factors18"] = dict(zip(np.asarray(codes)[valid], predictions, strict=True))
    return details, scores, {k: audit[k] for k in ("version", "feature_rows", "all_extra_factors_finite_rows",
                                                    "nonfinite_by_factor", "last_source_session")}


def predictions_for(arm, details, scores, cutoff, kind, freeze_day, dependency_sha):
    definition = DEFINITIONS[arm]
    accepted = details.loc[details.outcome.eq("SCORED")].copy()
    if set(accepted.stock_code) != set(scores):
        raise ValueError("Panel prediction score set differs from eligible names")
    scored = accepted.assign(
        score=accepted.stock_code.map(scores).astype(float), objective=arm, model_version=definition["version"],
        model_sha256=dependency_sha, score_units=definition["score_units"],
        support_group=accepted.training_sequences.map(support_group),
        portfolio_weight=(1/len(accepted) if arm == "equal_weight" and len(accepted) else None),
        qualification="ORIGINAL_V2_AND_RECOMPUTED_CAUSAL_GATES",
        prediction_id=[identity([VERSION, freeze_day, kind, cutoff, arm, dependency_sha, code])
                       for code in accepted.stock_code],
    )
    return scored.sort_values(["score", "stock_code"], ascending=[False, True], kind="stable")


def validate_existing(destination, original_sha, policy_sha):
    rec = read_json(destination / "manifest.json")
    if (rec["schema_version"] != VERSION or rec["legacy_manifest_sha256"] != original_sha
            or rec["policy_sha256"] != policy_sha or set(rec["models"]) != set(ARMS)):
        raise ValueError("Existing panel entry has different origin or model policy")
    for row in rec["models"].values():
        for field in ("candidates", "predictions"):
            verify_digest(destination / row[f"{field}_file"], row[f"{field}_sha256"])
    return rec


def publish_panel(destination, manifest, policy, meta, entries, scores, clock):
    import polars as pl

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        cutoff = manifest["trade_date"]
        original_kind = manifest["legacy_capture_kind"]
        started = parse_clock(manifest["capture_started_at"])
        committed = clock()
        same_day = is_timely(None, started, committed) and committed.date().isoformat() == cutoff
        if original_kind == "TIMELY" and not same_day:
            manifest["status"] = "MISSED_CAPTURE_WINDOW"
            entries = {arm: df.assign(outcome="PANEL_AFTER_CAPTURE_WINDOW") for arm, df in entries.items()}
            scores = {arm: {} for arm in ARMS}
        manifest["committed_at"] = committed.isoformat()
        manifest["prospective_eligible"] = False
        cutoff_freeze = parse_clock(policy["frozen_at"])
        newest = max(cutoff_freeze, parse_clock(meta["frozen_at_utc"]))
        manifest["prospective_start_after"] = newest.date().isoformat()
        manifest["prospective_eligible"] = bool(original_kind == "TIMELY" and same_day
            and manifest["formal_session"] and manifest["status"] != "MISSED_CAPTURE_WINDOW"
            and cutoff > newest.date().isoformat())
        for arm in ARMS:
            definition = DEFINITIONS[arm]
            weights = meta["weights_sha256"] if arm == "factors18" else identity(definition)
            rows = entries[arm]
            scored = predictions_for(arm, rows, scores[arm], cutoff, original_kind, policy["frozen_at"], weights)
            p = stage / f"predictions_{arm}.parquet"
            schema = {key: getattr(pl, value) for key, value in COLUMNS.items()}
            pl.DataFrame({key: scored[key].to_list() for key in schema}, schema=schema).write_parquet(
                p, compression="zstd")
            q = stage / f"candidates_{arm}.csv"
            rows.assign(support_group=rows.training_sequences.map(support_group)).to_csv(q, index=False)
            ranked = definition["portfolio_kind"] == "RANKED"
            lane_freeze = newest if arm == "factors18" else cutoff_freeze
            manifest["models"][arm] = {
                "model_version": definition["version"], "weights_sha256": weights,
                "model_frozen_at": lane_freeze.isoformat(), "portfolio_kind": definition["portfolio_kind"],
                "candidate_count": len(rows), "rows": len(scored), "abstentions": len(rows) - len(scored),
                "abstention_reasons": dict(Counter(rows.loc[rows.outcome.ne("SCORED"), "outcome"])),
                "top10": scored.stock_code.head(10).to_list() if ranked else [],
                "top20": scored.stock_code.head(20).to_list() if ranked else [],
                "predictions_file": p.name, "predictions_sha256": hash_file(p),
                "candidates_file": q.name, "candidates_sha256": hash_file(q),
            }
            for file in (p, q):
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
        finished = clock()
        if original_kind == "TIMELY" and same_day and not is_timely(None, started, finished):
            raise CaptureDeadlineExceeded("Panel crossed the same-session boundary before committing")
        manifest["committed_at"] = finished.isoformat()
        if manifest["status"] != "MISSED_CAPTURE_WINDOW":
            manifest["status"] = ("SCORED" if any(m["rows"] for m in manifest["models"].values())
                                  else "ABSTAINED" if manifest["formal_session"] else "NO_FORMAL_SESSION")
        atomic_json(stage / "manifest.json", manifest)
        os.rename(stage, destination)
        fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def capture_panel(legacy_ledger, panel, factor_model, factor_source, legacy_manifest_result, *,
                  engine=None, clock=now_shanghai, events=RISK_FILE):
    if legacy_manifest_result["status"] == "DEFERRED":
        return legacy_manifest_result
    started = clock()
    signal = date.fromisoformat(legacy_manifest_result["trade_date"])
    kind = legacy_manifest_result["capture_kind"]
    if kind not in ("TIMELY", "HISTORICAL") or signal > started.date():
        raise ValueError("Panel received a future or invalid original signal")
    original = legacy_ledger / kind.lower() / signal.isoformat()
    legacy = checked_manifest(original)
    if legacy != legacy_manifest_result or legacy["trade_date"] != signal.isoformat() or legacy["capture_kind"] != kind:
        raise ValueError("P55 panel must start from the checked day journal")
    policy_path = panel / "policy.json"
    if not policy_path.is_file():
        raise ValueError("Freeze the independent panel policy before publishing")
    policy = read_json(policy_path)
    meta, source = dependencies(factor_model, factor_source)
    if (policy["version"] != VERSION or policy["definitions"] != DEFINITIONS or policy["dependencies"] != source
            or max(parse_clock(policy["frozen_at"]), parse_clock(meta["frozen_at_utc"])) > started):
        raise ValueError("Panel policy or factor dependencies changed")
    digest = hash_file(policy_path)
    original_sha = hash_file(original / "manifest.json")
    destination = panel / kind.lower() / signal.isoformat()
    with exclusive(panel):
        if destination.exists():
            return validate_existing(destination, original_sha, digest)
        manifest = {
            "schema_version": VERSION, "trade_date": signal.isoformat(), "capture_kind": kind,
            "legacy_capture_kind": kind, "legacy_manifest_sha256": original_sha,
            "formal_session": legacy["formal_session"], "status": legacy["status"],
            "policy_frozen_at": policy["frozen_at"], "policy_sha256": digest,
            "market_rows": legacy["market_rows"], "capture_started_at": started.isoformat(),
            "evidence_snapshot": legacy["evidence_snapshot"], "uses_future_labels": False, "research_only": True,
            "models": {},
        }
        rows = legacy_details(original, legacy)
        entries, values = {arm: rows.copy() for arm in ARMS}, {arm: {} for arm in ARMS}
        missed = kind == "TIMELY" and signal != started.date()
        if missed or legacy["status"] == "MISSED_CAPTURE_WINDOW":
            manifest["status"] = "MISSED_CAPTURE_WINDOW"
            entries = {arm: item.assign(outcome="PANEL_AFTER_CAPTURE_WINDOW") for arm, item in entries.items()}
        elif len(rows) and rows.outcome.eq("SCORED").any():
            verify_digest(events, legacy["risk_events_sha256"])
            engine = AShareDailyLoader().engine if engine is None else engine
            try:
                rows, population, indices, raw, codes = fetch_panel_features(
                    engine, started, signal.isoformat(), meta, legacy, rows, events)
            except PriceContextUnavailable as error:
                for frame in entries.values():
                    frame.loc[frame.outcome.eq("SCORED"), "outcome"] = "PANEL_FORMAL_PRICE_CONTEXT_UNAVAILABLE"
                manifest["formal_price_issue"] = str(error)
            else:
                observed = clock()
                entries, values, factor_audit = arm_scores(rows, population, indices, raw, codes, factor_model, meta, policy)
                manifest.update({"factor_snapshot": factor_audit, "source_price_rows": len(raw),
                                 "source_price_observed_at": observed.isoformat()})
        for arm, frame in entries.items():
            if "missing_factors" not in frame:
                frame["missing_factors"] = ""
            if arm == "factors18":
                frame["training_sequences"] = frame.stock_code.map(meta["train_stock_counts"]).fillna(0).astype(int)
        manifest["candidate_count"] = len(rows)
        try:
            return publish_panel(destination, manifest, policy, meta, entries, values, clock)
        except CaptureDeadlineExceeded:
            manifest.update(status="MISSED_CAPTURE_WINDOW", models={})
            return publish_panel(destination, manifest, policy, meta, entries, values, clock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    replay = sub.add_parser("run")
    for command in (freeze, replay):
        for flag in ("panel", "factor-model", "factor-source"):
            command.add_argument(f"--{flag}", type=Path, required=True)
    replay.add_argument("--ledger", type=Path, required=True)
    replay.add_argument("--asof", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        item = freeze_panel(args.panel, args.factor_model, args.factor_source)
        print(f"P55 panel policy frozen at {item['frozen_at']}", flush=True)
    else:
        original = checked_manifest(args.ledger / "historical" / args.asof.isoformat())
        item = capture_panel(args.ledger, args.panel, args.factor_model, args.factor_source, original)
        print({"date": item["trade_date"], "status": item["status"],
               "rows": {name: spec["rows"] for name, spec in item["models"].items()}}, flush=True)


if __name__ == "__main__":
    main()
