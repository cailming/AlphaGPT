"""Run the three P5 research targets on one frozen, rolling dataset."""
from __future__ import annotations

import argparse
from datetime import date
import fcntl
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .p4_action_pipeline import atomic_json, read_json
from .p4_subset_dataset import GUARD
from .p4_subset_inputs import hash_file, parquet_write
from .p4_subset_model import FEATURES
from .p4_transformer_evaluate import diagnostics
from .p4_transformer_train import (EVALUATION_PLAN as P4_PLAN, FrozenData, _fit, load_model,
                                   precision_for, predict, verify_digest)

VERSION = "P5_TRANSFORMER_V2"
DATASET_VERSION = "P5_ROLLING_GATED_ECONOMIC_V2"
OBJECTIVES = ("absolute", "excess", "rank")
EVALUATION_PLAN = {
    "historical_use": "development diagnosis only; no independent predictive claim",
    "fit": "train only, paired features and architecture for all objectives",
    "selection": "per-objective best validation Rank IC; no historical diagnostic feedback",
    "support_floors_tested_on_validation": [0, 5, 10, 20],
    "support_floor": 0,
    "pool_reference": "equal-weight available independently qualified labels on each signal date in each fold",
    "rank": "average tie rank, (rank - 0.5) / n - 0.5; score above zero means above median",
    "confidence": P4_PLAN["confidence"],
    "confidence_quantile": P4_PLAN["confidence_quantile"],
    "top_k": P4_PLAN["top_k"],
    "diagnostic_minimum_daily_breadth": 20,
    "diagnostic_minimum_sessions": 5,
    "turnover": P4_PLAN["turnover"],
    "cost_bps_per_side": P4_PLAN["cost_bps_per_side"],
    "cohort": P4_PLAN["cohort"],
    "prospective": {
        "start": "first timely official signal strictly after model and protocol freeze in Asia/Shanghai",
        "minimum_matured_sessions": 20, "minimum_scored_label_coverage": .8,
        "primary": ["daily_rank_ic", "top20_net_excess_at_10bps_per_side", "label_coverage"],
        "baseline": "frozen V2 absolute plus equal-weight evaluable names",
        "retune_on_evaluation": False,
        "labels": "independently supported T+1 open to T+6 open economic return",
    },
}


def labels_for(frame, objective):
    """Transform only already mature labels from the supplied fold."""
    absolute = frame.target_return
    if objective == "absolute":
        return absolute.to_numpy(dtype=np.float32)
    grouped = frame.groupby("trade_date", sort=False).target_return
    if objective == "excess":
        return absolute.sub(grouped.transform("mean")).to_numpy(dtype=np.float32)
    if objective == "rank":
        return ((grouped.rank(method="average") - .5) / grouped.transform("size") - .5).to_numpy(dtype=np.float32)
    raise ValueError(f"Unknown P5 objective: {objective}")


def read_p5_audit(folder):
    audit = read_json(folder / "subset_audit.json")
    boundaries = [audit[key] for key in ("val_start", "test_start", "price_end", "research_end_exclusive")]
    if (audit["version"] != DATASET_VERSION or not audit["subset_training_allowed"]
            or tuple(audit["feature_names"]) != FEATURES or any(audit[key] != value for key, value in GUARD.items())
            or boundaries != sorted(set(boundaries)) or any(date.fromisoformat(day).isoformat() != day for day in boundaries)
            or any(audit["splits"][fold]["samples"] <= 0 for fold in ("train", "validation", "test"))):
        raise ValueError("Unqualified P5 rolling artifact")
    for filename, digest in audit["files"].items():
        verify_digest(folder / filename, digest)
    return audit


def open_p5_dataset(source):
    return FrozenData(source, audit_reader=read_p5_audit)


def load_p5_model(folder, device):
    model, meta = load_model(folder, device, version=VERSION)
    if meta["objective"] not in OBJECTIVES or meta["dataset_version"] != DATASET_VERSION:
        raise ValueError("Incompatible P5 objective or dataset")
    return model, meta


def train(source, output, objective, requested, resume):
    if not torch.cuda.is_available():
        raise RuntimeError("P5 formal training requires CUDA")
    data = open_p5_dataset(source)
    if resume and not (output / "checkpoint.pt").is_file():
        raise ValueError("Resume requires a checkpoint from this run")
    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = data.audit
        specification = {
            "objective": objective, "v2_sequence_policy": audit["v2_sequence_policy"],
            "support_floor": EVALUATION_PLAN["support_floor"],
            "fit_cutoff": audit["val_start"], "fit_cutoff_is_exclusive": True,
            "price_end": audit["price_end"], "prospective": EVALUATION_PLAN["prospective"],
            "development_diagnostic_start": audit["test_start"],
            "score_units": {"absolute": "economic return", "excess": "return above same-day pool",
                            "rank": "centered percentile rank"}[objective],
        }
        _fit(data, output, precision_for(requested), resume, version=VERSION, plan=EVALUATION_PLAN,
             label_transform=partial(labels_for, objective=objective), experiment_metadata=specification)


def evaluate(source, models, output):
    data = open_p5_dataset(source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_x, _ = data.fold("validation")
    frozen = {}
    for objective in OBJECTIVES:
        path = models / objective
        model, metadata = load_p5_model(path, device)
        if (metadata["objective"] != objective or metadata["dataset_audit_sha256"] != data.audit_sha256
                or metadata["training_identity"]["preflight_sha256"] != data.preflight_sha256
                or metadata["selection_last_exit"] >= data.audit["test_start"]):
            raise ValueError("Model and rolling comparison input mismatch")
        plan_path = path / "evaluation_plan.json"
        verify_digest(plan_path, metadata["evaluation_plan_sha256"])
        if read_json(plan_path) != EVALUATION_PLAN:
            raise ValueError("Evaluation protocol changed after freezing models")
        batch = metadata["architecture"]["batch_size"]
        val_prediction = predict(model, val_x, device, batch)
        floor = float(np.quantile(np.abs(val_prediction), EVALUATION_PLAN["confidence_quantile"]))
        frozen[objective] = model, metadata, batch, floor
    output.mkdir(parents=True)
    test, tx, _ = data.fold("test")
    frozen_test_rows = len(test)
    day_size = test.groupby("trade_date").stock_code.transform("size")
    sufficient = day_size.ge(EVALUATION_PLAN["diagnostic_minimum_daily_breadth"]).to_numpy()
    test, tx = test.loc[sufficient].reset_index(drop=True), tx[sufficient]
    if test.trade_date.nunique() < EVALUATION_PLAN["diagnostic_minimum_sessions"]:
        raise ValueError("Too few common diagnostic sessions for the frozen Top-20 comparison")
    result = {}
    daily = {}
    columns = ["stock_code", "trade_date", "entry_date", "exit_date", "target_return"]
    predictions = test[columns].copy()
    for objective, (model, meta, batch, floor) in frozen.items():
        scores = predict(model, tx, device, batch)
        report, daily[objective] = diagnostics(test, scores, floor, EVALUATION_PLAN)
        strategy = labels_for(test, objective)
        report["objective_direction_accuracy"] = float((np.sign(scores) == np.sign(strategy)).mean())
        report["common_raw_return_sign_diagnostic"] = report.pop("direction_accuracy")
        report["objective_mean_absolute_error"] = float(np.abs(scores - strategy).mean())
        report["mae_raw_return_score_diagnostic"] = report.pop("mae")
        report["rmse_raw_return_score_diagnostic"] = report.pop("rmse")
        confident = np.abs(scores) >= floor
        report["confidence"]["objective_direction_accuracy"] = (
            float((np.sign(scores[confident]) == np.sign(strategy[confident])).mean()) if confident.any() else None
        )
        report["coverage"] = {"shared_evaluable_rows": len(test), "shared_evaluable_fraction": 1.0,
                              "pools_median_rows": float(test.groupby("trade_date").size().median())}
        result[objective] = report | {"weights_sha256": meta["weights_sha256"], "selected_epoch": meta["best_epoch"]}
        predictions[f"{objective}_score"] = scores
    output_file = output / "same_sample_predictions.parquet"
    parquet_write(predictions, output_file)
    atomic_json(output / "daily_comparison.json", daily)
    atomic_json(output / "development_comparison.json", {
        "engineering_status": "PASSED", "performance_status": "DEVELOPMENT_DIAGNOSTICS_ONLY",
        "data_version": DATASET_VERSION, "dataset_audit_sha256": data.audit_sha256,
        "diagnostic_rows": len(test), "first_signal": str(test.trade_date.min()),
        "frozen_diagnostic_rows_before_daily_breadth_filter": frozen_test_rows,
        "excluded_low_breadth_rows": frozen_test_rows - len(test),
        "last_signal": str(test.trade_date.max()), "last_label_exit": str(test.exit_date.max()),
        "signal_days": int(test.trade_date.nunique()), "objectives": result,
        "rank_and_excess_scores_are_not_absolute_returns": True,
        "models_selected_using_historical_diagnostic_labels": False,
        "prediction_file_sha256": hash_file(output_file),
    })
    print(f"Compared all 3 frozen objectives on {len(test):,} common development labels", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--objective", choices=OBJECTIVES, required=True)
    fit.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    fit.add_argument("--resume", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("--models", type=Path, required=True)
    for command in (fit, compare):
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.source, args.out, args.objective, args.precision, args.resume)
    else:
        evaluate(args.source, args.models, args.out)


if __name__ == "__main__":
    main()
