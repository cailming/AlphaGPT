"""F3 CPU Ridge research baseline and causal, replayable inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import p4_action_pipeline as evidence
from .p4_mvp import FEATURES, RidgeModel, fit_ridge
from .p4_subset_dataset import GUARD, TRUSTED, VERSION, time_split
from .p4_subset_inputs import hash_file, parquet_read, parquet_write


MODEL_VERSION = "P4_SUBSET_RIDGE_V1"
ALPHAS = (.01, .1, 1.0, 10.0)


def dataset_summary(folder):
    audit = evidence.read_json(folder / "subset_audit.json")
    if (audit["version"] != VERSION or not audit["subset_training_allowed"]
            or tuple(audit["feature_names"]) != FEATURES or any(audit[key] != value for key, value in GUARD.items())):
        raise ValueError("Unsupported subset qualification contract")
    for filename, digest in audit["files"].items():
        if hash_file(folder / filename) != digest:
            raise ValueError(f"Dataset evidence digest mismatch: {filename}")
    return audit


def checked_folds(frame, audit):
    if (frame.duplicated(["stock_code", "trade_date"]).any() or not frame.sample_grade.eq(TRUSTED).all()
            or set(frame.split) != {"train", "validation", "test"}
            or not np.isfinite(frame.loc[:, [*FEATURES, "target_return"]].to_numpy(dtype=float)).all()):
        raise ValueError("Unqualified rows entered F3")
    if not (frame.trade_date.lt(frame.entry_date) & frame.entry_date.lt(frame.exit_date)
            & frame.exit_date.le(audit["price_end"]) & frame.exit_date.lt(audit["old_test_excluded"])).all():
        raise ValueError("Label path or old holdout boundary violated")
    expected = [time_split(row.trade_date, row.exit_date, audit["val_start"], audit["test_start"])
                for row in frame.itertuples(index=False)]
    if not np.array_equal(expected, frame.split.to_numpy()):
        raise ValueError("A forward label crosses the split assigned to its signal")
    folds = {key: part.reset_index(drop=True) for key, part in frame.groupby("split")}
    for fold, min_rows, min_days in (("train", 1000, 20), ("validation", 250, 5), ("test", 250, 5)):
        if (len(folds[fold]) < min_rows or folds[fold].trade_date.nunique() < min_days
                or len(folds[fold]) != audit["splits"][fold]["samples"]):
            raise ValueError(f"Insufficient or inconsistent {fold} set")
    return folds


def metrics(frame, values):
    truth = frame.target_return.to_numpy(dtype=float)
    error = values - truth
    daily = []
    sample = pd.DataFrame({"day": frame.trade_date, "truth": truth, "prediction": values})
    for _, group in sample.groupby("day", sort=False):
        if len(group) >= 5 and group.truth.nunique() > 1 and group.prediction.nunique() > 1:
            daily.append(float(group.truth.rank().corr(group.prediction.rank())))
    return {"samples": len(truth), "days": int(frame.trade_date.nunique()), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())), "direction_accuracy": float((np.sign(values) == np.sign(truth)).mean()),
            "mean_daily_rank_ic": float(np.mean(daily)) if daily else None, "rank_ic_days": len(daily)}


def infer_rows(model, features, asof, model_available_after):
    if asof <= model_available_after or features.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Inference requires a date after model selection and unique feature keys")
    if {"target_return", "split", "sample_grade", "exit_date"}.intersection(features.columns):
        raise ValueError("Inference requires feature-only input independent of future qualification")
    eligible = features.loc[features.trade_date.eq(asof), ["stock_code", "trade_date", "liquidity_rank", *FEATURES]].copy()
    if eligible.empty:
        raise ValueError("No causal candidates for this as-of date")
    eligible["predicted_economic_return"] = model.predict(eligible)
    return eligible[["stock_code", "trade_date", "liquidity_rank", "predicted_economic_return"]].sort_values(
        ["predicted_economic_return", "stock_code"], ascending=[False, True], kind="stable").reset_index(drop=True)


def train(dataset, output):
    audit = dataset_summary(dataset)
    folds = checked_folds(parquet_read(dataset / "training_dataset.parquet"), audit)
    nominal, validation, heldout = (folds[name] for name in ("train", "validation", "test"))
    results, models = {}, {}
    for alpha in ALPHAS:
        model = fit_ridge(nominal, alpha)
        models[alpha] = model
        results[str(alpha)] = metrics(validation, model.predict(validation))
    eligible = [alpha for alpha in ALPHAS if results[str(alpha)]["mean_daily_rank_ic"] is not None]
    if not eligible:
        raise ValueError("No valid validation rank diagnostics")
    chosen = min(eligible, key=lambda alpha: (-results[str(alpha)]["mean_daily_rank_ic"], alpha))
    model = models[chosen]
    output.mkdir(parents=True)
    model_path = output / "ridge_model.npz"
    model.save(model_path)
    restored = RidgeModel.load(model_path)
    test_scores = model.predict(heldout)
    if not np.array_equal(test_scores, restored.predict(heldout)):
        raise ValueError("Model reload does not reproduce frozen scores")
    predictions = heldout[["stock_code", "trade_date", "entry_date", "exit_date", "target_return"]].copy()
    predictions["predicted_economic_return"] = test_scores
    parquet_write(predictions, output / "holdout_predictions.parquet")
    causal = parquet_read(dataset / "causal_inference_features.parquet")
    date = str(causal.trade_date.max())
    label_end = str(nominal.exit_date.max())
    selection_end = str(validation.exit_date.max())
    replay = infer_rows(restored, causal, date, selection_end)
    parquet_write(replay, output / "causal_inference.parquet")
    diagnosis = {"version": MODEL_VERSION, **GUARD, "model_is_research_only": True, "promoted_for_trading": False,
                 "old_test_excluded": audit["old_test_excluded"], "holdout_kind": audit["test_kind"],
                 "fit_fold": "train", "hyperparameter_selection_fold": "validation", "selected_alpha": chosen,
                 "training_label_end": label_end, "model_selection_label_end": selection_end,
                 "validation": results, "test": metrics(heldout, test_scores),
                 "test_constant_zero_benchmark": metrics(heldout, np.zeros(len(heldout))),
                 "test_negative_ma20_benchmark": metrics(heldout, -heldout.close_ma20_dev.to_numpy(dtype=float)),
                 "dataset_audit_sha256": hash_file(dataset / "subset_audit.json"),
                 "model_sha256": hash_file(model_path), "inference_asof": date, "inference_rows": len(replay),
                 "inference_features_do_not_use_outcome_selection": True, "reload_scores_exact": True}
    evidence.atomic_json(output / "model_metadata.json", diagnosis)
    print(f"F3 Ridge alpha={chosen}, {len(heldout):,} holdout labels, {len(replay)} replay predictions", flush=True)


def infer(dataset, model_folder, asof, output):
    audit = dataset_summary(dataset)
    model = model_folder / "ridge_model.npz"
    check = evidence.read_json(model_folder / "model_metadata.json")
    if (check["version"] != MODEL_VERSION or check["dataset_audit_sha256"] != hash_file(dataset / "subset_audit.json")
            or hash_file(model) != check["model_sha256"] or asof >= audit["old_test_excluded"]):
        raise ValueError("Inference provenance or historical cutoff check failed")
    if output.exists():
        raise ValueError("Do not overwrite an inference artifact")
    features = parquet_read(dataset / "causal_inference_features.parquet")
    parquet_write(infer_rows(RidgeModel.load(model), features, asof, check["model_selection_label_end"]), output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    fit = subs.add_parser("train")
    score = subs.add_parser("infer")
    for sub in (fit, score):
        sub.add_argument("--dataset", required=True, type=Path)
        sub.add_argument("--out", required=True, type=Path)
    score.add_argument("--model", required=True, type=Path)
    score.add_argument("--asof", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.dataset, args.out)
    else:
        infer(args.dataset, args.model, args.asof, args.out)


if __name__ == "__main__":
    main()
