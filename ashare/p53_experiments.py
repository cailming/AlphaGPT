"""Train stable-pool controls and a paired market-factor experiment."""
from __future__ import annotations

import argparse
import fcntl
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .p4_action_pipeline import atomic_json, read_json
from .p4_mvp import FEATURES, fit_ridge
from .p4_subset_dataset import GUARD
from .p4_subset_inputs import hash_file, parquet_write
from .p4_subset_model import ALPHAS, metrics
from .p4_transformer_evaluate import diagnostics
from .p4_transformer_train import FrozenData, _fit, load_model, precision_for, predict, verify_digest
from .p5_train import EVALUATION_PLAN, OBJECTIVES, labels_for, open_p5_dataset

ARMS = (*OBJECTIVES, "factors18")
VERSION = "P53_STABLE_POOL_TRANSFORMER_V1"
FACTOR_MODEL_VERSION = "P54_PAIRED_MARKET_TRANSFORMER_V1"
PLAN = EVALUATION_PLAN | {
    "division": "P53 chronological stable-pool development, P54 paired covariates",
    "baselines": ["same_sample_validation_selected_ridge", "negative_ma20", "equal_weight"],
    "ridge_alphas": list(ALPHAS),
    "compare_factor_arm_to": "absolute",
    "prospective": EVALUATION_PLAN["prospective"] | {"status": "RESEARCH_CANDIDATE_OUTSIDE_DAILY_FROZEN_ROSTER"},
}


def factor_reader(folder):
    from .p54_factors import FACTOR_VERSION, P54_FEATURES

    audit = read_json(folder / "subset_audit.json")
    if (audit["version"] != FACTOR_VERSION or tuple(audit["feature_names"]) != P54_FEATURES
            or not audit["subset_training_allowed"] or any(audit[k] != v for k, v in GUARD.items())):
        raise ValueError("Not a qualified independent 18-feature research artifact")
    for filename, digest in audit["files"].items():
        verify_digest(folder / filename, digest)
    return audit


def open_experiment(source, expanded, arm):
    return FrozenData(expanded, audit_reader=factor_reader) if arm == "factors18" else open_p5_dataset(source)


def confirm_pair(control, expanded):
    for name in ("training_dataset.parquet",):
        if control.audit["files"][name] != expanded.audit["files"][name]:
            raise ValueError(f"Paired experiment changed {name}")
    for fold in ("train", "validation", "test"):
        if control.record["folds"][fold] != expanded.record["folds"][fold]:
            raise ValueError(f"Paired experiment changed {fold} sequence identity")
    if any(control.audit[key] != expanded.audit[key] for key in ("val_start", "test_start", "price_end")):
        raise ValueError("Paired experiment changed time boundaries")


def train(source, expanded, output, arm, resume):
    if not torch.cuda.is_available():
        raise RuntimeError("RTX research training requires CUDA")
    control = open_p5_dataset(source)
    data = open_experiment(source, expanded, arm) if arm == "factors18" else control
    if arm == "factors18":
        confirm_pair(control, data)
    if resume and not (output / "checkpoint.pt").is_file():
        raise ValueError("Resume requires an existing checkpoint")
    version = FACTOR_MODEL_VERSION if arm == "factors18" else VERSION
    target = "absolute" if arm == "factors18" else arm
    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = data.audit
        _fit(data, output, precision_for("auto"), resume, version=version, plan=PLAN,
             label_transform=partial(labels_for, objective=target),
             experiment_metadata={"objective": target, "experiment_arm": arm,
                                  "fit_cutoff": audit["val_start"], "v2_sequence_policy": audit["v2_sequence_policy"],
                                  "support_floor": 0, "prospective": PLAN["prospective"]})


def equal_weight(frame, train_mean, plan):
    gross = frame.groupby("trade_date").target_return.mean()
    return {
        "samples": len(frame), "days": len(gross), "mean_daily_rank_ic": None,
        "constant_train_mean_return_diagnostics": metrics(frame, np.full(len(frame), train_mean)),
        "mean_five_session_gross": float(gross.mean()),
        "net_excess_over_gross_pool_by_per_side_bps": {str(c): -2*c/10000 for c in plan["cost_bps_per_side"]},
        "selection": "all common evaluable members with equal weights; no Top-K tie breaking",
    }


def compare(source, expanded, models, output):
    control = open_p5_dataset(source)
    factor = FrozenData(expanded, audit_reader=factor_reader) if expanded else None
    if factor is not None:
        confirm_pair(control, factor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _, _ = control.fold("train")
    val, vx, _ = control.fold("validation")
    candidates = {alpha: fit_ridge(train, alpha) for alpha in ALPHAS}
    ics = {a: metrics(val, m.predict(val))["mean_daily_rank_ic"] for a, m in candidates.items()}
    chosen = min((a for a, value in ics.items() if value is not None), key=lambda a: (-ics[a], a))
    ridge = candidates[chosen]
    pool = ["absolute", "excess", "rank"] + (["factors18"] if factor else [])
    frozen = {}
    for arm in pool:
        data = factor if arm == "factors18" else control
        kind = FACTOR_MODEL_VERSION if arm == "factors18" else VERSION
        path = models / arm
        model, info = load_model(path, device, version=kind, feature_names=data.feature_names)
        if (info["experiment_arm"] != arm or info["dataset_audit_sha256"] != data.audit_sha256
                or info["training_identity"]["preflight_sha256"] != data.preflight_sha256
                or info["selection_last_exit"] >= data.audit["test_start"]):
            raise ValueError(f"Frozen model source differs for {arm}")
        verify_digest(path / "evaluation_plan.json", info["evaluation_plan_sha256"])
        if read_json(path / "evaluation_plan.json") != PLAN:
            raise ValueError("Historical scores cannot change the comparison plan")
        values = factor.fold("validation")[1] if arm == "factors18" else vx
        floor = float(np.quantile(np.abs(predict(model, values, device)), PLAN["confidence_quantile"]))
        frozen[arm] = model, floor, info
    floors = {name: float(np.quantile(np.abs(scores), PLAN["confidence_quantile"]))
              for name, scores in {"matched_ridge": ridge.predict(val),
                                   "ma20_reversal": -val.close_ma20_dev.to_numpy()}.items()}
    output.mkdir(parents=True)
    test, tx, _ = control.fold("test")
    available = test.groupby("trade_date").stock_code.transform("size").ge(PLAN["diagnostic_minimum_daily_breadth"]).to_numpy()
    counts = {"before_filter": len(test), "common": int(available.sum())}
    test, tx = test.loc[available].reset_index(drop=True), tx[available]
    if test.trade_date.nunique() < PLAN["diagnostic_minimum_sessions"]:
        raise ValueError("Too few common diagnostic sessions for Top-20")
    columns = ["stock_code", "trade_date", "entry_date", "exit_date", "target_return"]
    scores = test[columns].copy()
    arm_info = {}
    for arm, (model, floor, meta) in frozen.items():
        features = factor.fold("test")[1][available] if arm == "factors18" else tx
        scores[arm] = predict(model, features, device)
        floors[arm] = floor
        arm_info[arm] = {"weights_sha256": meta["weights_sha256"], "best_epoch": meta["best_epoch"],
                         "validation_rank_ic": meta["best_validation_rank_ic"], "epochs": len(meta["epochs"])}
    scores["matched_ridge"] = ridge.predict(test)
    scores["ma20_reversal"] = -test.close_ma20_dev.to_numpy()
    outputs, daily = {}, {}
    for arm, floor in floors.items():
        predicted = scores[arm].to_numpy(dtype=float)
        report, daily[arm] = diagnostics(test, predicted, floor, PLAN)
        target = labels_for(test, arm if arm in OBJECTIVES else "absolute")
        report["objective_direction_accuracy"] = float((np.sign(predicted) == np.sign(target)).mean())
        report["absolute_return_sign_diagnostic"] = report.pop("direction_accuracy")
        report["model"] = arm_info.get(arm)
        outputs[arm] = report
    outputs["equal_weight"] = equal_weight(test, float(train.target_return.mean()), PLAN)
    path = output / "common_predictions.parquet"
    parquet_write(scores, path)
    ridge.save(output / "matched_ridge.npz")
    atomic_json(output / "daily_comparison.json", daily)
    atomic_json(output / "stable_comparison.json", {
        "engineering_status": "PASSED", "performance_status": "DEVELOPMENT_ONLY_ALREADY_VIEWED_DATES",
        "weights_selected_using_diagnostic_labels": False, "exact_factor_pair": factor is not None,
        "training_rows": len(train), "validation_rows": len(val), "coverage": counts,
        "diagnostic_signals": int(test.trade_date.nunique()), "first_signal": str(test.trade_date.min()),
        "last_signal": str(test.trade_date.max()), "matched_ridge_alpha": chosen,
        "control_audit_sha256": control.audit_sha256, "factor_audit_sha256": factor.audit_sha256 if factor else None,
        "training_fits": arm_info, "metrics": outputs,
        "files": {file.name: hash_file(file) for file in output.iterdir()},
    })
    print(f"Compared {len(outputs)} methods on {len(test)} exact shared rows in {test.trade_date.nunique()} sessions", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--arm", choices=ARMS, required=True)
    fit.add_argument("--resume", action="store_true")
    assess = sub.add_parser("compare")
    assess.add_argument("--models", type=Path, required=True)
    for command in (fit, assess):
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--expanded", type=Path)
        command.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "train":
        if args.arm == "factors18" and args.expanded is None:
            cli.error("The 18-feature arm requires --expanded")
        train(args.source, args.expanded, args.out, args.arm, args.resume)
    else:
        compare(args.source, args.expanded, args.models, args.out)


if __name__ == "__main__":
    main()
