"""Train exact-sample market-factor ablations and inspect development diagnostics."""
from __future__ import annotations

import argparse
import fcntl
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .p4_action_pipeline import atomic_json, read_json
from .p4_mvp import FEATURES
from .p4_subset_inputs import hash_file, parquet_write
from .p4_subset_transformer import fit_sequence_scaler
from .p4_transformer_evaluate import diagnostics
from .p4_transformer_train import FrozenData, _fit, load_model, precision_for, predict, verify_digest
from .p53_experiments import PLAN as STABLE_PLAN, confirm_pair, factor_reader
from .p54_factors import EXTRA_FEATURES, P54_FEATURES
from .p5_train import labels_for, open_p5_dataset

VERSION = "P56_MARKET_FACTOR_ABLATION_V1"
GROUPS = {
    "base10": (),
    "group_a": ("market_ret_5", "market_ret_20", "market_advancer_share_5"),
    "group_b": ("market_turnover_log_change_20",),
    "group_c": ("stock_relative_ret_20", "stock_breakout_20"),
    "group_d": ("stock_volume_dryup_20", "stock_dry_volume_pullback_5"),
    "full18": EXTRA_FEATURES,
}
MARKET_FACTORS = frozenset((*GROUPS["group_a"], *GROUPS["group_b"]))
FOLDS = ("train", "validation", "test")
KEYS = ["stock_code", "trade_date"]
PLAN = {
    **{name: STABLE_PLAN[name] for name in (
        "confidence", "confidence_quantile", "top_k", "cost_bps_per_side",
        "diagnostic_minimum_daily_breadth", "diagnostic_minimum_sessions", "cohort",
    )},
    "comparison_status": "DEVELOPMENT_ONLY_ALREADY_VIEWED_DATES",
    "prospective": {"status": "RESEARCH_CANDIDATES_OUTSIDE_DAILY_FROZEN_ROSTER"},
    "objective": "absolute economic return",
    "arm_extras": {name: list(values) for name, values in GROUPS.items()},
    "architecture": "same full 18-column projection for every arm with inactive standardized inputs set to zero",
    "selection": "best validation Rank IC under the same early-stop rule; no diagnostic labels in selection",
    "budget": "existing max_epochs, early-stop patience, optimizer, batch size, learning rate and seed",
    "factor_direction": "ascending raw value, no ex-post orientation",
}


def active(arm):
    return (*FEATURES, *GROUPS[arm])


def masked_sequences(values, arm):
    if values.ndim != 3 or values.shape[2] != len(P54_FEATURES):
        raise ValueError("Ablation masks require the same 18-column input tensor")
    inactive = [pos for pos, name in enumerate(P54_FEATURES) if name not in active(arm)]
    values[:, :, inactive] = 0
    return values


def read_indexes(data, fold):
    path = data.root / "transformer" / f"{fold}_sequences.npz"
    meta = data.record["folds"][fold]
    verify_digest(path, meta["sha256"])
    with np.load(path, allow_pickle=False) as f:
        idx = f["causal_feature_rows"]
    if (idx.shape != (meta["rows"], data.config["sequence_length"]) or idx.dtype.kind not in "iu"
            or (idx < 0).any() or (idx >= len(data.features)).any()):
        raise ValueError(f"Ablation source has invalid {fold} feature indexes")
    return idx


def confirm_base_features(reference, expanded):
    columns = [*KEYS, "session_id", *FEATURES]
    if not reference.features[columns].equals(expanded.features[columns]):
        raise ValueError("Base prices, base features or sequence keys changed in paired data")


def open_pair(base, source):
    control = open_p5_dataset(base)
    expanded = FrozenData(source, audit_reader=factor_reader)
    confirm_pair(control, expanded)
    confirm_base_features(control, expanded)
    if (control.record["initial_training_config"] != expanded.record["initial_training_config"]
            or control.config["sequence_length"] != expanded.config["sequence_length"]):
        raise ValueError("Pair changed the frozen training configuration")
    train_idx = read_indexes(expanded, "train")
    mean, scale, used = fit_sequence_scaler(expanded.matrix, train_idx)
    expected = expanded.record["scaler"]
    if (used != expected["unique_feature_rows"] or not np.array_equal(mean, expanded.mean)
            or not np.array_equal(scale, expanded.scale)
            or not np.allclose(mean[:len(FEATURES)], control.mean, rtol=1e-12, atol=1e-12)
            or not np.allclose(scale[:len(FEATURES)], control.scale, rtol=1e-12, atol=1e-12)):
        raise ValueError("Ablation scaler differs from the shared training rows")
    for fold in FOLDS:
        index = read_indexes(expanded, fold)
        if not np.isfinite(expanded.matrix[index]).all():
            raise ValueError(f"{fold} is not finite on every shared 18-input sequence")
    return control, expanded


class AblationData:
    """Use the same 18-dimensional parameter initialization for every arm."""
    def __init__(self, data, arm):
        self.data = data
        self.arm = arm
        self.config = data.config
        self.audit = data.audit
        self.audit_sha256 = data.audit_sha256
        self.preflight_sha256 = data.preflight_sha256
        self.mean, self.scale = data.mean, data.scale

    def fold(self, part):
        rows, values, labels = self.data.fold(part)
        return rows, masked_sequences(values, self.arm), labels


def metadata(control, arm):
    return {
        "objective": "absolute",
        "experiment_arm": arm,
        "active_features": list(active(arm)),
        "input_mask": [name in active(arm) for name in P54_FEATURES],
        "paired_base_audit_sha256": control.audit_sha256,
        "prospective": PLAN["prospective"],
    }


def train(base, source, out, arm, resume):
    if not torch.cuda.is_available() or "3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("P5.6 training requires the RTX 3090")
    control, data = open_pair(base, source)
    if resume and not (out / "checkpoint.pt").is_file():
        raise ValueError("Resume requires an existing arm checkpoint")
    out.mkdir(parents=True, exist_ok=resume)
    with (out / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _fit(AblationData(data, arm), out, precision_for("auto"), resume, version=VERSION, plan=PLAN,
             label_transform=partial(labels_for, objective="absolute"),
             experiment_metadata=metadata(control, arm))


def rank_ic(rows, name):
    if len(rows) < 5 or rows[name].nunique() < 2 or rows.target_return.nunique() < 2:
        return None
    return float(rows[name].rank(method="average").corr(rows.target_return.rank(method="average")))


def mean_or_none(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def quantile_group(series):
    # Average ranks keep equal raw values in the same group, including zero-heavy pullbacks.
    return np.minimum(5, ((series.rank(method="average") - .5) * 5 // len(series) + 1).astype(int))


def groups_on(rows, name):
    if len(rows) < 5 or rows[name].nunique() < 2:
        return {}, None
    grouped = rows.groupby(quantile_group(rows[name])).target_return
    cells = {int(k): (float(v.mean()), len(v)) for k, v in grouped}
    spread = cells[5][0] - cells[1][0] if 1 in cells and 5 in cells else None
    return cells, spread


def summarize_groups(samples, name):
    summary = []
    for group in range(1, 6):
        groups = [cells[group] for cells, _ in samples if group in cells]
        summary.append({"factor": name, "group": group, "mean_economic_return": mean_or_none(x[0] for x in groups),
                        "rows": sum(len_group for _, len_group in groups), "cohorts": len(groups)})
    return summary


def factor_diagnostics(frame):
    """Factor IC with actual finite observations; market regimes are measured across dates."""
    summaries, monthly, daily, quintiles = {}, [], [], []
    cohort = frame.assign(month=frame.trade_date.str[:7])
    for name in EXTRA_FEATURES:
        frame = cohort.loc[np.isfinite(cohort[name])]
        market = name in MARKET_FACTORS
        if market and frame.groupby("trade_date")[name].nunique().gt(1).any():
            raise ValueError(f"Market regime factor unexpectedly differs within a day: {name}")
        mode = "TIME_SERIES_DAILY_POOL" if market else "CROSS_SECTIONAL_DAILY_STOCKS"
        month_output, scores, cohort_groups = [], [], []
        if market:
            # Every signal day gets equal weight. The target is that day's qualified pool return.
            per_day = frame.groupby(["month", "trade_date"], as_index=False).agg(
                {name: "first", "target_return": "mean"})
            score = rank_ic(per_day, name)
            cohort_groups = [groups_on(per_day, name)]
            for month, rows in per_day.groupby("month", sort=True):
                result = rank_ic(rows, name)
                month_output.append({"month": month, "rank_ic": result, "rank_ic_days": len(rows) if result is not None else 0,
                                     "signal_days": len(rows), "rows": int(frame.month.eq(month).sum()),
                                     "high_minus_low": groups_on(rows, name)[1]})
            daily.extend({"factor": name, "mode": mode, "trade_date": row.trade_date, "month": row.month,
                          "rows": int(frame.trade_date.eq(row.trade_date).sum()), "rank_ic": None,
                          "factor_value": getattr(row, name), "pool_return": row.target_return}
                         for row in per_day.itertuples(index=False))
            summaries[name] = {"rank_ic": score, "cross_sectional_rank_ic": None,
                               "defined_rank_ic_days": len(per_day) if score is not None else 0}
        else:
            raw_daily = []
            for day, rows in frame.groupby("trade_date", sort=True):
                cells, spread = groups_on(rows, name)
                score = rank_ic(rows, name)
                scores.append(score)
                cohort_groups.append((cells, spread))
                raw_daily.append({"trade_date": day, "month": day[:7], "rows": len(rows),
                                  "rank_ic": score, "high_minus_low": spread})
            for month, rows in pd.DataFrame(raw_daily, columns=("trade_date", "month", "rows", "rank_ic",
                                                                 "high_minus_low")).groupby("month", sort=True):
                month_output.append({"month": month, "rank_ic": mean_or_none(rows.rank_ic),
                                     "rank_ic_days": int(rows.rank_ic.notna().sum()), "signal_days": len(rows),
                                     "rows": int(rows.rows.sum()), "high_minus_low": mean_or_none(rows.high_minus_low)})
            summaries[name] = {"rank_ic": mean_or_none(scores), "cross_sectional_rank_ic": mean_or_none(scores),
                               "defined_rank_ic_days": sum(score is not None for score in scores)}
            daily.extend({"factor": name, "mode": mode, **row} for row in raw_daily)
        months = [row["rank_ic"] for row in month_output if row["rank_ic"] is not None]
        summaries[name] |= {"mode": mode, "rows": len(frame), "signal_days": frame.trade_date.nunique(),
                            "denominator_rows": len(cohort), "denominator_days": cohort.trade_date.nunique(),
                            "anchor_coverage": len(frame) / len(cohort),
                            "high_minus_low": mean_or_none(x[1] for x in cohort_groups),
                            "defined_months": len(months), "positive_months": sum(score > 0 for score in months)}
        monthly.extend({"factor": name, "mode": mode, **row} for row in month_output)
        quintiles.extend({"mode": mode, "sample_unit": "signal_day" if market else "stock_signal",
                          **row} for row in summarize_groups(cohort_groups, name))
    return summaries, pd.DataFrame(monthly), pd.DataFrame(daily), pd.DataFrame(quintiles)


def factor_coverage(universe, expanded):
    audit = expanded.root / "factor_audit.json"
    verify_digest(audit, expanded.audit["factor_audit_sha256"])
    provenance = read_json(audit)
    if (provenance["source_audit_sha256"] != universe.audit_sha256
            or provenance["source_preflight_sha256"] != universe.preflight_sha256):
        raise ValueError("Coverage denominator is not the P5.4 pre-pair universe")
    confirm_base_features(universe, expanded)
    rows = []
    for fold in FOLDS:
        # Apply each factor's availability mask to original stable-pool qualified sequence indexes.
        before_index, after_index = read_indexes(universe, fold), read_indexes(expanded, fold)
        for name in EXTRA_FEATURES:
            values = expanded.features[name].to_numpy()
            before, after = np.isfinite(values[before_index]), np.isfinite(values[after_index])
            for population, flags in (("PRE_PAIR_QUALIFIED_UNIVERSE", before), ("P56_EXACT_PAIR", after)):
                rows.append({"factor": name, "fold": fold, "population": population,
                             "total_sequences": len(flags), "anchor_finite_sequences": int(flags[:, -1].sum()),
                             "sequence_finite_sequences": int(flags.all(axis=1).sum()),
                             "anchor_coverage": float(flags[:, -1].mean()),
                             "full_sequence_coverage": float(flags.all(axis=1).mean())})
    return pd.DataFrame(rows)


def correlations(universe, expanded):
    index = read_indexes(universe, "test")[:, -1]
    frame = expanded.features.iloc[index][list(EXTRA_FEATURES)]
    frame = frame.where(np.isfinite(frame))
    result = frame.corr(method="spearman", min_periods=5)
    valid = frame.notna().astype("int64")
    counts = valid.T.dot(valid)
    return result, counts


def compare(base, source, universe_root, models, out):
    control, expanded = open_pair(base, source)
    universe = open_p5_dataset(universe_root)
    coverage = factor_coverage(universe, expanded)
    corr, corr_counts = correlations(universe, expanded)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_x = expanded.fold("validation")[1]
    frozen = {}
    identities = []
    for arm in GROUPS:
        folder = models / arm
        model, info = load_model(folder, device, version=VERSION, feature_names=P54_FEATURES)
        verify_digest(folder / "evaluation_plan.json", info["evaluation_plan_sha256"])
        if (any(info[key] != value for key, value in metadata(control, arm).items())
                or info["architecture"] != AblationData(expanded, arm).config
                or info["dataset_audit_sha256"] != expanded.audit_sha256
                or info["training_identity"]["preflight_sha256"] != expanded.preflight_sha256
                or info["selection_last_exit"] >= expanded.audit["test_start"]
                or info["model_training_accessed_test_labels"]
                or not 1 <= len(info["epochs"]) <= info["architecture"]["max_epochs"]
                or read_json(folder / "evaluation_plan.json") != PLAN):
            raise ValueError(f"P5.6 model/protocol does not match the paired run: {arm}")
        identities.append(tuple(info["training_identity"][key] for key in ("gpu", "precision", "torch")))
        val_score = predict(model, masked_sequences(val_x.copy(), arm), device, info["architecture"]["batch_size"])
        frozen[arm] = (model, info, float(np.quantile(np.abs(val_score), PLAN["confidence_quantile"])))
    if len(set(identities)) != 1:
        raise ValueError("Training hardware, precision or PyTorch differs between arms")

    # Holdout labels are opened only after every arm and evaluation rule are frozen.
    out.mkdir(parents=True)
    test, original_x, _ = expanded.fold("test")
    breadth_ok = test.groupby("trade_date").stock_code.transform("size").ge(PLAN["diagnostic_minimum_daily_breadth"])
    if test.loc[breadth_ok].trade_date.nunique() < PLAN["diagnostic_minimum_sessions"]:
        raise ValueError("Too few paired diagnostic sessions")
    paired_index = read_indexes(expanded, "test")[breadth_ok.to_numpy(), -1]
    rows, inputs = test.loc[breadth_ok].reset_index(drop=True), original_x[breadth_ok.to_numpy()]
    scores = rows[[*KEYS, "entry_date", "exit_date", "target_return"]].copy()
    result, daily = {}, {}
    for arm, (model, info, floor) in frozen.items():
        prediction = predict(model, masked_sequences(inputs.copy(), arm), device, info["architecture"]["batch_size"])
        scores[arm] = prediction
        result[arm], daily[arm] = diagnostics(rows, prediction, floor, PLAN)
        result[arm] |= {"active_features": info["active_features"], "best_epoch": info["best_epoch"],
                        "epochs": len(info["epochs"]), "validation_rank_ic": info["best_validation_rank_ic"],
                        "weights_sha256": info["weights_sha256"]}
    factor_frame = rows[[*KEYS, "target_return"]].join(
        expanded.features.iloc[paired_index][list(EXTRA_FEATURES)].reset_index(drop=True))
    pre_pair, _, _ = universe.fold("test")
    pre_breadth = pre_pair.groupby("trade_date").stock_code.transform("size").ge(PLAN["diagnostic_minimum_daily_breadth"])
    pre_indices = read_indexes(universe, "test")[pre_breadth.to_numpy(), -1]
    own_frame = pre_pair.loc[pre_breadth, [*KEYS, "target_return"]].reset_index(drop=True).join(
        expanded.features.iloc[pre_indices][list(EXTRA_FEATURES)].reset_index(drop=True))
    panels = [factor_diagnostics(frame) for frame in (factor_frame, own_frame)]
    populations = ("P56_EXACT_PAIR", "PRE_PAIR_BREADTH_QUALIFIED_ANCHOR_FINITE")
    factors = {name: panel[0] for name, panel in zip(populations, panels)}
    months, factor_daily, quintiles = (
        pd.concat([panel[i].assign(population=name) for name, panel in zip(populations, panels)], ignore_index=True)
        for i in (1, 2, 3)
    )
    for arm in GROUPS:
        current, base_result = result[arm], result["base10"]
        current_ic, base_ic = current["mean_daily_rank_ic"], base_result["mean_daily_rank_ic"]
        net = "mean_net_excess_over_gross_pool_by_per_side_cost_bps"
        current["delta_vs_base10"] = {
            "mean_daily_rank_ic": current_ic - base_ic if current_ic is not None and base_ic is not None else None,
            "top20_net_excess_10bps_per_side": current["top20"][net]["10"] - base_result["top20"][net]["10"],
        }
    parquet_write(scores, out / "same_sample_predictions.parquet")
    atomic_json(out / "daily_model_diagnostics.json", daily)
    for filename, frame in (("factor_coverage.csv", coverage), ("factor_monthly.csv", months),
                            ("factor_daily_ic.csv", factor_daily), ("factor_quintile_returns.csv", quintiles)):
        frame.to_csv(out / filename, index=False)
    corr.to_csv(out / "factor_correlations.csv", index_label="factor")
    corr_counts.to_csv(out / "factor_correlation_counts.csv", index_label="factor")
    atomic_json(out / "ablation_comparison.json", {
        "version": VERSION, "performance_status": PLAN["comparison_status"], "exact_pair": True,
        "training_rows": expanded.record["folds"]["train"]["rows"], "validation_rows": len(val_x),
        "diagnostic_rows_before_daily_breadth_filter": len(test), "diagnostic_rows": len(rows),
        "signal_days": rows.trade_date.nunique(), "first_signal": rows.trade_date.min(), "last_signal": rows.trade_date.max(),
        "configuration": AblationData(expanded, "base10").config,
        "gpu": identities[0][0], "precision": identities[0][1], "train_torch_version": identities[0][2],
        "paired_base_audit_sha256": control.audit_sha256, "paired_full_audit_sha256": expanded.audit_sha256,
        "pre_pair_audit_sha256": universe.audit_sha256, "daily_breadth_minimum": PLAN["diagnostic_minimum_daily_breadth"],
        "factor_diagnostics": factors, "metrics": result,
        "correlation_scope": "pooled Spearman across qualified pre-pair diagnostic anchor rows, pairwise finite",
        "weights_selected_using_diagnostic_labels": False,
        "files": {path.name: hash_file(path) for path in out.iterdir()},
    })
    print(f"P5.6 compared {len(result)} arms on {len(rows):,} rows in {rows.trade_date.nunique()} shared development days",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify")
    fit = sub.add_parser("train")
    fit.add_argument("--arm", choices=GROUPS, required=True)
    fit.add_argument("--resume", action="store_true")
    assess = sub.add_parser("compare")
    assess.add_argument("--universe", type=Path, required=True)
    assess.add_argument("--models", type=Path, required=True)
    for command in (check, fit, assess):
        command.add_argument("--base", type=Path, required=True)
        command.add_argument("--source", type=Path, required=True)
    for command in (fit, assess):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        _, data = open_pair(args.base, args.source)
        print({fold: data.record["folds"][fold]["rows"] for fold in FOLDS}, flush=True)
    elif args.command == "train":
        train(args.base, args.source, args.out, args.arm, args.resume)
    else:
        compare(args.base, args.source, args.universe, args.models, args.out)


if __name__ == "__main__":
    main()
