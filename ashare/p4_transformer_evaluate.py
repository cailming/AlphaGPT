"""One-time historical comparison on the frozen P4.4 sequence holdout."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .p4_action_pipeline import atomic_json, read_json
from .p4_mvp import RidgeModel, fit_ridge
from .p4_subset_inputs import hash_file, parquet_write
from .p4_subset_model import metrics
from .p4_transformer_train import FrozenData, load_model, predict, verify_digest


def diagnostics(frame, scores, floor, plan):
    result = metrics(frame, scores)
    data = frame[["stock_code", "trade_date", "target_return"]].assign(score=scores)
    confident = np.abs(scores) >= floor
    result["confidence"] = {
        "validation_abs_score_threshold": float(floor), "test_coverage": float(confident.mean()),
        "test_rows": int(confident.sum()),
        "test_direction_accuracy": (float((np.sign(scores[confident]) ==
                                          np.sign(data.target_return.to_numpy()[confident])).mean())
                                    if confident.any() else None),
    }
    daily = []
    lists = {k: None for k in plan["top_k"]}
    for day, group in data.groupby("trade_date", sort=True):
        ranked = group.sort_values(["score", "stock_code"], ascending=[False, True], kind="stable")
        row = {"trade_date": day, "pool_mean": float(group.target_return.mean()), "pool_size": len(group)}
        for k in plan["top_k"]:
            pick = ranked.head(k)
            chosen = set(pick.stock_code)
            prior = lists[k]
            row[f"top{k}_gross"] = float(pick.target_return.mean())
            row[f"top{k}_excess"] = row[f"top{k}_gross"] - row["pool_mean"]
            row[f"top{k}_lists_replaced"] = len(prior - chosen) / len(prior) if prior else None
            lists[k] = chosen
        daily.append(row)
    for k in plan["top_k"]:
        gross = float(np.mean([row[f"top{k}_gross"] for row in daily]))
        excess = float(np.mean([row[f"top{k}_excess"] for row in daily]))
        changes = [row[f"top{k}_lists_replaced"] for row in daily[1:]]
        result[f"top{k}"] = {
            "mean_five_session_gross": gross, "mean_excess_over_evaluable_pool": excess,
            "mean_list_replacement": float(np.mean(changes)) if changes else None,
            "mean_net_return_by_per_side_cost_bps": {str(c): gross - 2*c/10000 for c in plan["cost_bps_per_side"]},
            "mean_net_excess_over_gross_pool_by_per_side_cost_bps": {
                str(c): excess - 2*c/10000 for c in plan["cost_bps_per_side"]},
        }
    return result, daily


def compare(source, model_folder, output, device):
    data = FrozenData(source)
    transformer, info = load_model(model_folder, device)
    plan_file = model_folder / "evaluation_plan.json"
    verify_digest(plan_file, info["evaluation_plan_sha256"])
    plan = read_json(plan_file)
    if (data.audit_sha256 != info["dataset_audit_sha256"]
            or data.preflight_sha256 != info["training_identity"]["preflight_sha256"]):
        raise ValueError("Model and comparison source differ")
    original_ridge = source / "model/ridge_model.npz"
    ridge_meta = read_json(source / "model/model_metadata.json")
    verify_digest(original_ridge, ridge_meta["model_sha256"])
    if (ridge_meta["dataset_audit_sha256"] != data.audit_sha256
            or ridge_meta["model_selection_label_end"] >= data.audit["test_start"]):
        raise ValueError("Frozen Ridge is incompatible")
    ridge = RidgeModel.load(original_ridge)
    train, _, _ = data.fold("train")
    validation, vx, _ = data.fold("validation")
    variants = {}
    for alpha in plan["ridge_alphas"]:
        model = fit_ridge(train, alpha)
        variants[alpha] = model, metrics(validation, model.predict(validation))["mean_daily_rank_ic"]
    alpha = min((a for a, (_, ic) in variants.items() if ic is not None),
                key=lambda a: (-variants[a][1], a))
    matched = variants[alpha][0]
    batch = info["architecture"]["batch_size"]
    mean = info["train_mean_return"]
    validation_scores = {
        "transformer": predict(transformer, vx, device, batch),
        "frozen_ridge": ridge.predict(validation),
        "matched_train_ridge": matched.predict(validation),
        "negative_ma20": -validation.close_ma20_dev.to_numpy(),
        "train_mean": np.full(len(validation), mean),
    }
    floors = {name: float(np.quantile(np.abs(scores), plan["confidence_quantile"]))
              for name, scores in validation_scores.items()}

    # Creating the output fences a single evaluation after all tuning and thresholds are frozen.
    output.mkdir(parents=True)
    test, tx, _ = data.fold("test")
    scores = {
        "transformer": predict(transformer, tx, device, batch),
        "frozen_ridge": ridge.predict(test),
        "matched_train_ridge": matched.predict(test),
        "negative_ma20": -test.close_ma20_dev.to_numpy(),
        "train_mean": np.full(len(test), mean),
    }
    results, day_reports = {}, {}
    for name, values in scores.items():
        results[name], day_reports[name] = diagnostics(test, values, floors[name], plan)
    predictions = test[["stock_code", "trade_date", "entry_date", "exit_date", "target_return"]].assign(**scores)
    parquet_write(predictions, output / "same_sample_predictions.parquet")
    atomic_json(output / "daily_comparison.json", day_reports)
    observed = results["transformer"]
    positive_ic = (observed["mean_daily_rank_ic"] or 0.0) > 0
    positive_excess = observed["top20"]["mean_net_excess_over_gross_pool_by_per_side_cost_bps"]["10"] > 0
    report = {
        "engineering_acceptance": "PASSED", "prediction_acceptance":
            "SIGNAL_ON_THIS_HISTORICAL_HOLDOUT_REQUIRES_PROSPECTIVE_CONFIRMATION" if positive_ic and positive_excess
            else "DID_NOT_MEET_HISTORICAL_PREDICTION_CRITERIA",
        "holdout_kind": info["holdout_kind"], "same_samples": len(test),
        "test_first": str(test.trade_date.min()), "test_last": str(test.trade_date.max()),
        "test_last_exit": str(test.exit_date.max()), "test_sessions": test.trade_date.nunique(),
        "matched_ridge_alpha": alpha, "matched_ridge_fit_rows": len(train),
        "model_sha256": info["weights_sha256"], "model_manifest_sha256": hash_file(model_folder / "model_metadata.json"),
        "evaluation_plan_sha256": info["evaluation_plan_sha256"],
        "tie_break": "ascending stock code", "constant_rank_ic": "undefined",
        "metrics": results,
        "files": {path.name: hash_file(path) for path in output.iterdir()},
    }
    atomic_json(output / "historical_evaluation.json", report)
    print(f"Historical evaluation complete: {len(test)} common samples, {report['prediction_acceptance']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compare(args.source, args.model, args.out, device)


if __name__ == "__main__":
    main()
