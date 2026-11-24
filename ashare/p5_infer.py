"""P5 frozen-model inference and validation-only training-support diagnostics."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .p4_action_pipeline import atomic_json, read_json
from .p4_mvp import FEATURES, PoolConfig
from .p4_subset_inputs import RISK_FILE, hash_file, parquet_read
from .p4_subset_model import metrics

FLOORS = (0, 5, 10, 20)
SCORE_UNITS = {"absolute": "five_session_economic_return", "excess": "five_session_pool_excess_return",
               "rank": "five_session_centered_cross_section_percentile"}
VALIDATION_COLUMNS = {"stock_code", "trade_date", "target_return", "score"}


def support_values(stocks, counts):
    if any(not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError("Training sequence counts must be nonnegative integers")
    return stocks.map(counts).fillna(0).astype("int64")


def floor_summary(rows, train_counts):
    """Re-score nothing: compare support floors applied to the same frozen score vector."""
    if (not VALIDATION_COLUMNS.issubset(rows.columns) or rows.empty
            or rows[["stock_code", "trade_date"]].isna().any().any()
            or rows.duplicated(["stock_code", "trade_date"]).any()
            or not np.isfinite(rows[["target_return", "score"]].to_numpy(dtype=float)).all()):
        raise ValueError("Expected one finite frozen validation score per stock and date")
    if "split" in rows and not rows.split.eq("validation").all():
        raise ValueError("Training-support sensitivity only accepts validation scores")
    daily_base = rows.groupby("trade_date", sort=True).agg(pool_size=("stock_code", "size"),
                                                         pool_return=("target_return", "mean"))
    support = support_values(rows.stock_code, train_counts)
    reports, by_day = {}, daily_base.reset_index()
    for floor in FLOORS:
        subset = rows.loc[support.ge(floor)]
        counts = subset.groupby("trade_date").size().reindex(daily_base.index, fill_value=0)
        result = (metrics(subset, subset.score.to_numpy(dtype=float)) if len(subset) else {
            "samples": 0, "days": 0, "mae": None, "rmse": None, "direction_accuracy": None,
            "mean_daily_rank_ic": None, "rank_ic_days": 0,
        })
        ordered = subset.sort_values(["trade_date", "score", "stock_code"], ascending=[True, False, True])
        for limit in (10, 20):
            selected = ordered.groupby("trade_date", sort=False).head(limit)
            top = selected.groupby("trade_date").target_return.mean()
            result[f"top{limit}_excess_over_fixed_daily_pool"] = (
                float(top.sub(daily_base.pool_return.reindex(top.index)).mean()) if len(top) else None)
        result.update({"coverage": len(subset) / len(rows), "stocks": int(subset.stock_code.nunique()),
                       "daily_median_rows": float(counts.median()),
                       "days_with_no_predictions": int(counts.eq(0).sum())})
        reports[str(floor)] = result
        by_day[f"floor_{floor}_rows"] = counts.to_numpy()
    return reports, by_day


def publish_validation(rows, model_info, output, provenance):
    reports, by_day = floor_summary(rows, model_info["train_stock_counts"])
    output.mkdir(parents=True)
    daily_path = output / "coverage_by_day.csv"
    by_day.to_csv(daily_path, index=False)
    record = {
        "purpose": "VALIDATION_DEVELOPMENT_DIAGNOSTIC",
        "uses_historical_test_labels": False, "same_frozen_predictions_across_floors": True,
        "daily_benchmark": "mean absolute economic return of the original scored validation pool",
        "first_signal": str(rows.trade_date.min()), "last_signal": str(rows.trade_date.max()),
        "rows": len(rows), "sessions": len(by_day), "floors": reports,
        "coverage_by_day_sha256": hash_file(daily_path), **provenance,
    }
    atomic_json(output / "support_floors.json", record)
    print(f"Validation support thresholds: {len(rows):,} fixed scores across {len(by_day)} days", flush=True)
    return record


def validation(model_dir, output):
    from .p4_transformer_train import verify_digest

    path = model_dir / "validation_predictions.parquet"
    model_info = read_json(model_dir / "model_metadata.json")
    if model_info["objective"] != "absolute":
        raise ValueError("The support-floor experiment uses the absolute-return model")
    verify_digest(path, model_info["validation_predictions_sha256"])
    verify_digest(model_dir / "best_model.pt", model_info["weights_sha256"])
    rows = parquet_read(path)
    first = rows.trade_date.min()
    train_cutoff = model_info["fit_cutoff"]
    if (rows.empty or first != model_info["validation_first_signal"]
            or rows.trade_date.max() != model_info["validation_last_signal"]
            or first < train_cutoff or (first == train_cutoff and not model_info["fit_cutoff_is_exclusive"])):
        raise ValueError("Validation scores are outside the frozen split")
    return publish_validation(rows, model_info, output, {
        "model_version": model_info["version"], "weights_sha256": model_info["weights_sha256"],
        "validation_predictions_sha256": model_info["validation_predictions_sha256"],
        "sequence_policy": model_info["v2_sequence_policy"],
    })


def validation_v1(source, model_dir, output):
    import torch

    from .p4_transformer_train import FrozenData, load_model, predict

    data = FrozenData(source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info = load_model(model_dir, device)
    if (info["dataset_audit_sha256"] != data.audit_sha256
            or info["training_identity"]["preflight_sha256"] != data.preflight_sha256):
        raise ValueError("V1 model and validation source differ")
    rows, inputs, _ = data.fold("validation")
    rows = rows.assign(score=predict(model, inputs, device, info["architecture"]["batch_size"]))
    return publish_validation(rows, info, output, {
        "model_version": info["version"], "weights_sha256": info["weights_sha256"],
        "sequence_policy": "V1_CONTINUOUS_DAILY_TOP_N",
        "note": "The archived V1 validation set was used for V1 model selection.",
    })


def select_latest(population, signal, model_info, evidence):
    """Apply the same uncapped feature sequence rule and date-bounded evidence gate."""
    from .p4_subset_dataset import FEATURE_LOOKBACK
    from .p5_sequences import sequence_indexes

    length = model_info["architecture"]["sequence_length"]
    anchors = population.anchors.loc[population.anchors.trade_date.eq(signal)].reset_index(drop=True)
    calendar = population.calendar.strftime("%Y-%m-%d")
    records = evidence["stocks"]
    if date.fromisoformat(evidence["max_covered_date"]).isoformat() != evidence["max_covered_date"]:
        raise ValueError("Malformed inference evidence cutoff")
    events = []
    has_evidence = np.zeros(len(anchors), dtype=bool)
    for i, row in enumerate(anchors.itertuples(index=False)):
        witness = records.get(row.stock_code)
        start = row.session_id - length - FEATURE_LOOKBACK + 2
        if witness is None or start < 0:
            continue
        lower = date.fromisoformat(witness["covered_from"]).isoformat()
        upper = date.fromisoformat(witness["covered_through"]).isoformat()
        if lower > upper or upper > evidence["max_covered_date"]:
            raise ValueError(f"Invalid observation bounds for {row.stock_code}")
        has_evidence[i] = lower <= calendar[start] <= signal <= upper
        if has_evidence[i]:
            events.extend({"stock_code": row.stock_code, "ex_date": ex, "published_on": known}
                          for ex, known in witness["actions"])
    action_frame = pd.DataFrame(events, columns=["stock_code", "ex_date", "published_on"])
    price_positions, price_indices = sequence_indexes(population.features, anchors, length)
    positions, indices = sequence_indexes(population.features, anchors, length, known_actions=action_frame,
                                           calendar=population.calendar)

    price_ready = np.zeros(len(anchors), dtype=bool)
    action_ready = np.zeros(len(anchors), dtype=bool)
    price_ready[price_positions] = True
    action_ready[positions] = True
    support = support_values(anchors.stock_code, model_info["train_stock_counts"])
    floor = model_info["support_floor"]
    if floor not in FLOORS:
        raise ValueError("Model has an unsupported training count floor")
    available = action_ready & has_evidence
    selected = available & support.ge(floor).to_numpy()
    details = anchors[["stock_code", "trade_date", "liquidity_rank"]].copy()
    details["training_sequences"] = support
    details["has_contiguous_causal_features"] = price_ready
    details["has_bounded_dual_source_window"] = has_evidence
    details["has_causal_action_veto"] = price_ready & has_evidence & ~action_ready
    fallback_counts = np.zeros(len(anchors), dtype=np.int64)
    fallback_days = [""] * len(anchors)
    if "auxiliary_reference_fallback" in population.features:
        feature_fallback = population.features.auxiliary_reference_fallback.to_numpy(dtype=bool)[price_indices]
        dates = population.features.trade_date.astype(str).to_numpy()[price_indices]
        fallback_counts[price_positions] = feature_fallback.sum(axis=1)
        for position, flags, session_dates in zip(price_positions, feature_fallback, dates, strict=True):
            fallback_days[position] = "|".join(session_dates[flags])
    details["auxiliary_reference_fallback_feature_sessions"] = fallback_counts
    details["auxiliary_reference_fallback_feature_dates"] = fallback_days
    details["outcome"] = np.select(
        [~price_ready, ~has_evidence, ~action_ready, ~support.ge(floor)],
        ["NO_CONTIGUOUS_FEATURE_SEQUENCE", "CAUSAL_ACTION_EVIDENCE_WINDOW_UNAVAILABLE",
         "COMPANY_ACTION_INSIDE_CONTEXT", "INSUFFICIENT_TRAINING_SEQUENCES"], default="SCORED")
    counts = {str(value): int((available & support.ge(value).to_numpy()).sum()) for value in FLOORS}
    return details, positions[selected[positions]], indices[selected[positions]], counts


def publish_latest(predictions, details, output, audit):
    import polars as pl

    schema = {
        "stock_code": pl.String, "trade_date": pl.String, "liquidity_rank": pl.Int64,
        "training_sequences": pl.Int64, "score": pl.Float64, "score_units": pl.String,
        "objective": pl.String, "model_version": pl.String, "model_sha256": pl.String, "data_quality": pl.String,
        "auxiliary_reference_fallback_feature_sessions": pl.Int64,
    }
    output.mkdir(parents=True)
    prediction_path = output / "predictions.parquet"
    table = pl.DataFrame({column: predictions[column].to_list() for column in schema}, schema=schema)
    table.write_parquet(prediction_path, compression="zstd")
    details_path = output / "candidate_diagnostics.csv"
    details.to_csv(details_path, index=False)
    record = {
        **audit, "rows": len(predictions), "anchors": len(details),
        "status": "SCORED" if len(predictions) else "ABSTAINED_NO_QUALIFIED_SEQUENCE",
        "candidate_outcomes": {str(k): int(v) for k, v in details.outcome.value_counts().items()},
        "prediction_sha256": hash_file(prediction_path), "diagnostics_sha256": hash_file(details_path),
        "uses_future_labels": False, "research_only": True,
    }
    atomic_json(output / "inference_audit.json", record)
    return record


def live_population(source, data, info, asof, events_file, evidence_source):
    from .daily import AShareDailyLoader
    from .p4_transformer_train import verify_digest
    from .p5_sequences import CHINA, VERSION, build_causal_population, formal_snapshot

    if info["v2_sequence_policy"] != VERSION or tuple(info["feature_names"]) != FEATURES:
        raise ValueError("Model and live causal sequence policy differ")
    evidence_path = source / "dataset/inference_evidence.json"
    evidence_digest = data["files"][evidence_path.name]
    verify_digest(evidence_path, evidence_digest)
    evidence = read_json(evidence_path)
    evidence_manifest_digest = info["dataset_audit_sha256"]
    if evidence_source is not None:
        refreshed_manifest_path = evidence_source / "subset_audit.json"
        refreshed_manifest = read_json(refreshed_manifest_path)
        if refreshed_manifest["version"] != data["version"]:
            raise ValueError("Refreshed action evidence uses a different P5 dataset version")
        refreshed_path = evidence_source / "inference_evidence.json"
        evidence_digest = refreshed_manifest["files"][refreshed_path.name]
        verify_digest(refreshed_path, evidence_digest)
        refreshed = read_json(refreshed_path)
        if refreshed["policy"] != evidence["policy"] or refreshed["max_covered_date"] < evidence["max_covered_date"]:
            raise ValueError("Refreshed action evidence uses a different policy or is older than the model input")
        evidence = refreshed
        evidence_manifest_digest = hash_file(refreshed_manifest_path)

    capture_clock = datetime.now(CHINA)
    observed = datetime.fromisoformat(evidence["observed_at_utc"])
    if observed.tzinfo is None or observed > capture_clock:
        raise ValueError("Company-action observation is outside the live capture clock")
    current, raw = formal_snapshot(AShareDailyLoader().engine, capture_clock, asof=asof)
    signal = current.isoformat()
    if signal <= info["selection_last_exit"]:
        raise ValueError("Signal date must follow model-selection label maturity")
    events = pd.read_csv(events_file, dtype={"stock_code": "string"})
    population = build_causal_population(raw, current - timedelta(days=46), current,
                                         PoolConfig(**info["pool_config"]), events=events)
    details, positions, indices, counts = select_latest(population, signal, info, evidence)
    common = {
        "asof": signal, "snapshot_clock_asia_shanghai": capture_clock.isoformat(),
        "context": "LATEST_FORMAL_POSTGRESQL" if asof is None else "HISTORICAL_FORMAL_REPLAY",
        "observed_company_evidence_sha256": evidence_digest, "observed_company_evidence_at_utc": observed.isoformat(),
        "observed_company_evidence_manifest_sha256": evidence_manifest_digest,
        "company_evidence_last_date": evidence["max_covered_date"], "risk_source_sha256": hash_file(events_file),
        "v2_sequence_policy": VERSION, "support_floor": info["support_floor"], "qualified_by_support_floor": counts,
        "auxiliary_reference_policy": population.stats["auxiliary_reference_policy"],
        "price_only_sequences_by_support_floor": {
            str(floor): int((details.has_contiguous_causal_features & details.training_sequences.ge(floor)).sum())
            for floor in FLOORS
        },
        "pending_dual_evidence_after_price_checks": int(
            (details.has_contiguous_causal_features & ~details.has_bounded_dual_source_window).sum()),
        "qualified_with_auxiliary_reference_fallback": int(
            details.iloc[positions].auxiliary_reference_fallback_feature_sessions.gt(0).sum()),
        "full_historical_action_coverage_certified": False, "current_st_backfilled": False,
    }
    return capture_clock, current, population, details, positions, indices, common


def qualification(source, output, asof=None, events_file=RISK_FILE, evidence_source=None):
    from .p5_train import EVALUATION_PLAN, open_p5_dataset

    data = open_p5_dataset(source)
    train, _, _ = data.fold("train")
    info = {
        "architecture": data.config, "feature_names": data.audit["feature_names"],
        "pool_config": data.audit["pool"]["pool_config"], "v2_sequence_policy": data.audit["v2_sequence_policy"],
        "train_stock_counts": {str(k): int(v) for k, v in train.stock_code.value_counts().items()},
        "dataset_audit_sha256": data.audit_sha256,
        "selection_last_exit": data.audit["splits"]["validation"]["last_label_exit"],
        "support_floor": EVALUATION_PLAN["support_floor"],
    }
    _, current, _, details, positions, _, common = live_population(
        source, data.audit, info, asof, events_file, evidence_source)
    details.loc[details.outcome.eq("SCORED"), "outcome"] = "QUALIFIED_FOR_SCORING"
    output.mkdir(parents=True)
    details_path = output / "candidate_diagnostics.csv"
    details.to_csv(details_path, index=False)
    report = {
        **common, "dataset_audit_sha256": data.audit_sha256, "prediction_performed": False,
        "uses_future_labels": False, "uses_frozen_training_sequence_counts_only": True,
        "anchors": len(details), "qualified": len(positions), "within_engineering_target": 100 <= len(positions) <= 300,
        "candidate_outcomes": {str(k): int(v) for k, v in details.outcome.value_counts().items()},
        "diagnostics_sha256": hash_file(details_path),
    }
    atomic_json(output / "qualification_audit.json", report)
    print(f"{current}: {len(positions)} qualified P5 candidates, {common['qualified_by_support_floor']}", flush=True)
    return report


def latest(source, model_dir, output, asof=None, events_file=RISK_FILE, evidence_source=None):
    import torch

    from .p4_transformer_train import predict, scale_sequences, verify_digest
    from .p5_sequences import CHINA
    from .p5_train import load_p5_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info = load_p5_model(model_dir, device)
    protocol_path = model_dir / "evaluation_plan.json"
    verify_digest(protocol_path, info["evaluation_plan_sha256"])
    protocol = read_json(protocol_path)
    if protocol["support_floor"] != info["support_floor"] or protocol["prospective"] != info["prospective"]:
        raise ValueError("Model and frozen inference protocol differ")
    audit_path = source / "dataset/subset_audit.json"
    verify_digest(audit_path, info["dataset_audit_sha256"])
    capture_clock, current, population, details, positions, indices, common = live_population(
        source, read_json(audit_path), info, asof, events_file, evidence_source)
    frozen_at = datetime.fromisoformat(info["frozen_at_utc"]).astimezone(CHINA)
    predictions = details.iloc[positions][["stock_code", "trade_date", "liquidity_rank", "training_sequences",
                                           "auxiliary_reference_fallback_feature_sessions"]].copy()
    values = (predict(model, scale_sequences(population.features[list(FEATURES)].to_numpy(dtype=float), indices,
                                             info["mean"], info["scale"]),
                      device, info["architecture"]["batch_size"]).astype(float)
              if len(predictions) else np.empty(0, dtype=float))
    quality = "FORMAL_CAUSAL_SEQUENCE_WITH_BOUNDED_EVIDENCE_HISTORICAL_ST_PARTIAL"
    predictions = predictions.assign(score=values, score_units=SCORE_UNITS[info["objective"]],
                                     objective=info["objective"], model_version=info["version"],
                                     model_sha256=info["weights_sha256"], data_quality=quality)
    predictions = predictions.sort_values(["score", "stock_code"], ascending=[False, True]).reset_index(drop=True)
    timely = asof is None and current == capture_clock.date() and current > frozen_at.date()
    record = publish_latest(predictions, details, output, {
        **common,
        "prospective_eligible": timely and bool(len(predictions)),
        "prospective_session_capture": timely, "model_frozen_asia_shanghai": frozen_at.isoformat(),
        "model_version": info["version"], "model_sha256": info["weights_sha256"], "objective": info["objective"],
        "evaluation_plan_sha256": info["evaluation_plan_sha256"],
        "data_quality": quality if len(predictions) else "NO_QUALIFIED_CAUSAL_SEQUENCE",
        "scored_with_auxiliary_reference_fallback": int(
            predictions.auxiliary_reference_fallback_feature_sessions.gt(0).sum()),
    })
    print(f"{current}: {record['rows']} P5 predictions; {record['status']}; "
          f"support floors {common['qualified_by_support_floor']}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validation", "validation-v1", "latest", "qualification"):
        sub = commands.add_parser(command)
        if command != "qualification":
            sub.add_argument("--model", type=Path, required=True)
        sub.add_argument("--out", type=Path, required=True)
        if command in ("validation-v1", "latest", "qualification"):
            sub.add_argument("--source", type=Path, required=True)
        if command in ("latest", "qualification"):
            sub.add_argument("--asof", type=date.fromisoformat, help="A retrospective analysis date")
            sub.add_argument("--events", type=Path, default=RISK_FILE)
            sub.add_argument("--evidence-source", type=Path, help="Independently audited newer P5 dataset directory")
    args = parser.parse_args()
    if args.command == "validation":
        validation(args.model, args.out)
    elif args.command == "validation-v1":
        validation_v1(args.source, args.model, args.out)
    elif args.command == "qualification":
        qualification(args.source, args.out, args.asof, args.events, args.evidence_source)
    else:
        latest(args.source, args.model, args.out, args.asof, args.events, args.evidence_source)


if __name__ == "__main__":
    main()
