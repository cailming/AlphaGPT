"""Freeze and capture the trained P5.6 group C arm in an independent research journal."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, exclusive, read_json
from .p4_subset_inputs import RISK_FILE, hash_file
from .p4_transformer_train import load_model, predict, scale_sequences, verify_digest
from .p54_factors import P54_FEATURES, build_market_factors
from .p55_daily import checked_manifest, is_timely, now_shanghai, parse_clock, support_group
from .p55_nightly import publish_revision
from .p55_panel import (COLUMNS, PriceContextUnavailable, factor_sequences, fetch_panel_features,
                        identity, validate_existing)
from .p56_ablation import PLAN, VERSION as TRAIN_VERSION, active, masked_sequences

VERSION = "P58_GROUP_C_FORWARD_V1"
ARM = "group_c"


def contract(model, source, panel):
    meta = read_json(model / "model_metadata.json")
    expected_mask = [name in active(ARM) for name in P54_FEATURES]
    if (meta["version"] != TRAIN_VERSION or meta["experiment_arm"] != ARM or meta["objective"] != "absolute"
            or tuple(meta["feature_names"]) != P54_FEATURES or meta["input_mask"] != expected_mask
            or tuple(meta["active_features"]) != active(ARM) or meta["model_training_accessed_test_labels"]
            or read_json(model / "evaluation_plan.json") != PLAN):
        raise ValueError("Group C model contract mismatch")
    verify_digest(model / "best_model.pt", meta["weights_sha256"])
    verify_digest(model / "evaluation_plan.json", meta["evaluation_plan_sha256"])
    preflight = source / "transformer/transformer_preflight.json"
    verify_digest(preflight, meta["training_identity"]["preflight_sha256"])
    audit = source / "dataset/subset_audit.json"
    verify_digest(audit, meta["dataset_audit_sha256"])
    ref = read_json(panel / "policy.json")["dependencies"]
    if (ref["factor_source_audit_sha256"] != meta["dataset_audit_sha256"]
            or ref["factor_builder_source_sha256"] != hash_file(Path(build_market_factors.__code__.co_filename))):
        raise ValueError("Panel and arm use different factor definitions or source data")
    scaler = read_json(preflight)["scaler"]
    path = source / "transformer" / scaler["file"]
    verify_digest(path, scaler["sha256"])
    with np.load(path, allow_pickle=False) as data:
        if (scaler["fit_fold"] != "train" or tuple(data["feature_names"]) != P54_FEATURES
                or any(not np.array_equal(data[k], meta[k]) for k in ("mean", "scale"))
                or not np.isfinite(data["mean"]).all() or not np.isfinite(data["scale"]).all() or (data["scale"] <= 0).any()):
            raise ValueError("Frozen train-only scaling values differ")
    return meta, {"version": VERSION, "weights_sha256": meta["weights_sha256"],
                  "training_version": TRAIN_VERSION, "training_frozen_at": meta["frozen_at_utc"],
                  "metadata_sha256": hash_file(model / "model_metadata.json"),
                  "panel_policy_sha256": hash_file(panel / "policy.json"), "features": list(P54_FEATURES),
                  "input_mask": expected_mask, "mean": meta["mean"], "scale": meta["scale"],
                  "train_only_scaler_sha256": scaler["sha256"], "preflight_sha256": hash_file(preflight),
                  "factor_builder_sha256": ref["factor_builder_source_sha256"],
                  "dataset_sha256": meta["dataset_audit_sha256"], "evaluation_plan_sha256": meta["evaluation_plan_sha256"],
                  "minimum_market_stocks": ref["factor_market_minimum_stocks"], "sequence_length": meta["architecture"]["sequence_length"],
                  "eligibility": "Original V2 and panel gates; complete paired 18-feature sequence as in P5.6 training; mask after scaling",
                  "time_rule": "same-session after 18:00; signal date strictly after policy and model freeze dates",
                  "evaluation": "separate OWN and COMMON7 before labels; P55 mature-label, cost and review protocol",
                  "uses_future_labels": False, "research_only": True}


def freeze(out, panel, model, source, clock=now_shanghai):
    _, spec = contract(model, source, panel)
    with exclusive(out):
        path = out / "policy.json"
        if path.exists():
            old = read_json(path)
            if {k: v for k, v in old.items() if k != "frozen_at"} != spec:
                raise ValueError("Freeze changed research dependencies in a new journal")
            return old
        now = clock()
        if parse_clock(spec["training_frozen_at"]) > now:
            raise ValueError("Model freeze lies in the future")
        policy = spec | {"frozen_at": now.isoformat()}
        atomic_json(path, policy)
        return policy


def read_capture(folder, policy_digest):
    rec = read_json(folder / "manifest.json")
    if rec["schema_version"] != VERSION or rec["policy_sha256"] != policy_digest or set(rec["models"]) != {ARM}:
        raise ValueError("Unexpected group C capture lineage")
    lane = rec["models"][ARM]
    for k in ("candidates", "predictions"):
        if Path(lane[f"{k}_file"]).name != lane[f"{k}_file"]:
            raise ValueError("Expected local capture artifacts")
        verify_digest(folder / lane[f"{k}_file"], lane[f"{k}_sha256"])
    return rec


def get_scores(model, meta, policy, original, rows, started, day, engine):
    scores, audit = {}, {}
    if not rows.outcome.eq("SCORED").any():
        return rows, scores, audit
    verify_digest(RISK_FILE, original["risk_events_sha256"])
    try:
        rows, population, idx, raw, codes = fetch_panel_features(
            engine or AShareDailyLoader().engine, started, day, meta, original, rows, RISK_FILE)
    except PriceContextUnavailable as exc:
        rows.loc[rows.outcome.eq("SCORED"), "outcome"] = "FORMAL_PRICE_CONTEXT_UNAVAILABLE"
        return rows, scores, {"price_issue": str(exc)}
    if not codes:
        return rows, scores, audit
    matrix, good, remapped, details, missing = factor_sequences(raw, population, idx, policy["minimum_market_stocks"])
    missing_names = dict(zip(codes, missing, strict=True))
    rejected = rows.stock_code.isin(np.array(codes)[~good])
    rows.loc[rejected, "outcome"] = "FACTOR_SEQUENCE_INCOMPLETE"
    rows.loc[rejected, "missing_factors"] = rows.loc[rejected, "stock_code"].map(missing_names)
    if good.any():
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network, info = load_model(model, device, version=TRAIN_VERSION, feature_names=P54_FEATURES)
        if info["weights_sha256"] != policy["weights_sha256"]:
            raise ValueError("Group C weights changed while scoring")
        batch = masked_sequences(scale_sequences(matrix, remapped[good], policy["mean"], policy["scale"]), ARM)
        values = predict(network, batch, device, info["architecture"]["batch_size"])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite group C predictions")
        scores = dict(zip(np.array(codes)[good], values, strict=True))
    return rows, scores, {k: details[k] for k in ("feature_rows", "nonfinite_by_factor", "last_source_session")}


def track(kind, day, start, finish, policy, formal, parent_ok):
    if kind == "HISTORICAL":
        return "HISTORICAL"
    if not is_timely(None, start, finish) or finish.date().isoformat() != day:
        raise ValueError("Capture missed the timely publication boundary")
    boundary = max(parse_clock(policy[k]).date().isoformat() for k in ("frozen_at", "training_frozen_at"))
    return "PROSPECTIVE" if formal and parent_ok and day > boundary else "TIMELY_NONPROSPECTIVE"


def capture(out, panel, legacy, model, source, replay=None, *, clock=now_shanghai, engine=None):
    import polars as pl

    start = clock()
    day = replay or start.date()
    if day > start.date():
        raise ValueError("Future capture requested")
    if replay is None and start.time() < time(18):
        return {"status": "DEFERRED", "reason": "BEFORE_FORMAL_CLOSE"}
    day = day.isoformat()
    kind = "HISTORICAL" if replay is not None else "TIMELY"
    parent_kind = "historical" if replay is not None and (panel / "historical" / day).exists() else "timely"
    peer_dir = panel / parent_kind / day
    if not (peer_dir / "manifest.json").exists():
        return {"status": "DEFERRED", "reason": "WAITING_FOR_SIX_METHOD_PANEL"}
    original_dir = legacy / parent_kind / day
    parent_sha = hash_file(original_dir / "manifest.json")
    original = checked_manifest(original_dir)
    policy = read_json(out / "policy.json")
    meta, dep = contract(model, source, panel)
    if dep != {k: v for k, v in policy.items() if k != "frozen_at"} or parse_clock(policy["frozen_at"]) > start:
        raise ValueError("Group C policy or training dependency changed")
    peer = validate_existing(peer_dir, parent_sha, policy["panel_policy_sha256"])
    if any(r["trade_date"] != day or r["capture_kind"].lower() != parent_kind for r in (peer, original)):
        raise ValueError("Source journal track mismatch")
    peer_sha = hash_file(peer_dir / "manifest.json")
    digest = hash_file(out / "policy.json")
    bucket = out / kind.lower()
    with exclusive(out):
        if (bucket / day).exists():
            rec = read_capture(bucket / day, digest)
            if rec["legacy_manifest_sha256"] != parent_sha or rec["panel_manifest_sha256"] != peer_sha:
                raise ValueError("Existing group C capture lost its original six-method parents")
            return rec
        source_lane = peer["models"]["factors18"]
        rows = pd.read_csv(peer_dir / source_lane["candidates_file"], dtype={"stock_code":"string","trade_date":"string"},
                           keep_default_na=False)
        if rows.stock_code.duplicated().any() or len(rows) != source_lane["candidate_count"]:
            raise ValueError("Invalid paired model candidates")
        if day <= meta["selection_last_exit"]:
            rows.loc[rows.outcome.eq("SCORED"), "outcome"] = "NOT_AFTER_SELECTION_END"
        rows, scores, factors = get_scores(model, meta, policy, original, rows, start, day, engine)
        rows["training_sequences"] = rows.stock_code.map(meta["train_stock_counts"]).fillna(0).astype(int)
        rows["support_group"] = rows.training_sequences.map(support_group)
        accepted = rows.loc[rows.outcome.eq("SCORED")].copy()
        if set(accepted.stock_code) != scores.keys():
            raise ValueError("Group C scores and eligibility disagree")
        accepted = accepted.assign(score=accepted.stock_code.map(scores), objective=ARM, model_version=VERSION,
                    model_sha256=meta["weights_sha256"], score_units="five_session_economic_return",
                    portfolio_weight=None, qualification="P56_EXACT_PAIR_ALL18_FINITE_MASK_C",
                    prediction_id=[identity([VERSION, digest, kind, day, c]) for c in accepted.stock_code])
        accepted = accepted.sort_values(["score","stock_code"], ascending=[False,True])

        def save(stage):
            stage.mkdir()
            pp, cp = stage / "predictions_group_c.parquet", stage / "candidates_group_c.csv"
            schema = {k:getattr(pl, v) for k,v in COLUMNS.items()}
            pl.DataFrame({k:accepted[k].tolist() for k in schema}, schema=schema).write_parquet(pp)
            rows.to_csv(cp, index=False)
            finish = clock()
            capture_track = track(kind, day, start, finish, policy, peer["formal_session"], peer["prospective_eligible"])
            spec = {"model_version":VERSION, "weights_sha256":meta["weights_sha256"], "model_frozen_at":meta["frozen_at_utc"],
                    "portfolio_kind":"RANKED", "rows":len(scores), "candidate_count":len(rows), "abstentions":len(rows)-len(scores),
                    "abstention_reasons":dict(Counter(rows.loc[rows.outcome.ne("SCORED"),"outcome"])),
                    "top10":accepted.stock_code.head(10).tolist(),"top20":accepted.stock_code.head(20).tolist(),
                    "predictions_file":pp.name,"predictions_sha256":hash_file(pp),"candidates_file":cp.name,"candidates_sha256":hash_file(cp)}
            atomic_json(stage / "manifest.json", {"schema_version":VERSION, "trade_date":day, "capture_kind":kind,
                "track":capture_track, "prospective_eligible":capture_track=="PROSPECTIVE", "parent_kind":parent_kind,
                "formal_session":peer["formal_session"],"status":"SCORED" if scores else "ABSTAINED" if peer["formal_session"] else "NO_FORMAL_SESSION",
                "candidate_count":len(rows),"legacy_manifest_sha256":parent_sha, "panel_manifest_sha256":peer_sha,
                "factor_snapshot":factors, "started_at":start.isoformat(), "committed_at":finish.isoformat(),
                "policy_sha256":digest,"policy_frozen_at":policy["frozen_at"],"uses_future_labels":False,"research_only":True,
                "models":{ARM:spec}})
        published = publish_revision(bucket, day, "manifest.json", save)
        return read_capture(published, digest)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("freeze","capture","report"))
    for flag in ("out", "legacy", "panel", "model", "source"):
        cli.add_argument("--"+flag, type=Path, required=flag in ("out","panel"))
    cli.add_argument("--replay", type=date.fromisoformat)
    cli.add_argument("--asof", type=date.fromisoformat)
    args = cli.parse_args()
    if args.command != "report" and (args.model is None or args.source is None):
        cli.error("--model and --source are required to freeze or capture")
    if args.command in ("capture","report") and args.legacy is None:
        cli.error("--legacy is required for capture and report")
    if args.command == "freeze":
        print(freeze(args.out,args.panel,args.model,args.source)["frozen_at"])
    elif args.command == "capture":
        r = capture(args.out,args.panel,args.legacy,args.model,args.source,args.replay)
        print({k:r[k] for k in ("trade_date","track","status","reason") if k in r},
              {k:v["rows"] for k,v in r.get("models",{}).items()})
    else:
        from .p58_group_c_report import report
        print(report(args.out,args.panel,args.legacy,args.asof or now_shanghai().date()))

if __name__ == "__main__":
    main()
