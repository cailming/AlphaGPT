"""P4.3-D: honest train/validation-only lightweight ranking comparisons.

No inspected test labels, production writes, execution simulation or claims of
new independent OOS. Validation was previously viewed and is exploratory.
All caps, signs and candidate definitions use training rows only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .p4_mvp import (
    FEATURES, PoolConfig, build_pool, fit_ridge, load_formal_ohlcv,
    rank_diagnostics, RESEARCH_WARNING,
)
from .p4_factor_audit import split_audit_pool
from .risk_events import load_risk_events

WARNING = (
    "TRAIN-VALIDATION EXPLORATORY COMPARISON ONLY; validation and the old "
    "2026 test period have already been inspected, and this validation is "
    "NOT fresh independent evidence. Test rows/labels are never scored. "
    "Targets are overlapping unadjusted five-session forward labels. "
    "Rank scores need not be expected returns. NO executable PnL or fills; "
    "a new untouched future period or predeclared walk-forward is required."
)


def train_factor_directions(train: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
    """Estimate signs on training dates, never from validation or test labels."""
    ic = {feature: [] for feature in FEATURES}
    for _, group in train.groupby("trade_date", sort=True):
        if len(group) < 3:
            continue
        y = group.target_return.rank()
        for feature in FEATURES:
            r = group[feature].rank().corr(y)
            if np.isfinite(r):
                ic[feature].append(float(r))
    means = {name: float(np.mean(scores)) if scores else 0.0
             for name, scores in ic.items()}
    directions = {name: int(np.sign(value)) for name, value in means.items()}
    return directions, means


def _rank_ensemble(frame: pd.DataFrame, directions: dict[str, int]) -> np.ndarray:
    if not any(directions.values()):
        raise ValueError("No train-defined factor direction")
    score = pd.Series(0.0, index=frame.index)
    active = 0
    for name in FEATURES:
        sign = directions[name]
        if sign:
            # Each signal day's ranks depend on <=T features only.
            score += sign * (frame.groupby("trade_date")[name].rank(pct=True) - 0.5)
            active += 1
    return (score / active).to_numpy(dtype=float)


def _validation_daily(frame: pd.DataFrame, *, name: str, top_k: int) -> pd.DataFrame:
    rows = []
    for day, group in frame.groupby("trade_date", sort=True):
        if len(group) < 3:
            raise ValueError("Need >=3 labelled observations per validation date")
        xr, yr = group.predicted_return.rank(), group.target_return.rank()
        corr = xr.corr(yr)
        best = group.sort_values(["predicted_return", "stock_code"],
                                 ascending=[False, True], kind="stable").head(top_k)
        rows.append({
            "candidate": name, "trade_date": day, "samples": len(group),
            "rank_ic": float(corr) if np.isfinite(corr) else np.nan,
            "topk_samples": len(best),
            "topk_positive_share": float(best.target_return.gt(0).mean()),
            "topk_mean_forward_label": float(best.target_return.mean()),
            "pool_mean_forward_label": float(group.target_return.mean()),
            "topk_excess_vs_pool_label": float(best.target_return.mean() - group.target_return.mean()),
        })
    return pd.DataFrame(rows)


def compare_train_validation(train: pd.DataFrame, val: pd.DataFrame, *,
                             top_k: int = 20, alpha: float = 1.0
                             ) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """Compare predeclared research candidates on the SAME validation rows.

    Does not select or promote a winning candidate. Any model selection using
    this known validation period would itself need a later untouched holdout.
    """
    if top_k < 1 or alpha <= 0:
        raise ValueError("Invalid comparison configuration")
    if train.empty or val.empty:
        raise ValueError("Empty train/validation split")
    if not train.trade_date.max() < val.trade_date.min():
        raise ValueError("Train and validation dates overlap")
    for frame in (train, val):
        if frame.target_return.isna().any() or not np.isfinite(
            frame[list(FEATURES) + ["target_return"]].to_numpy(dtype=float)
        ).all():
            raise ValueError("Missing or nonfinite feature/label")
    directions, ic_means = train_factor_directions(train)
    best_feature = max(FEATURES, key=lambda name: (abs(ic_means[name]), -FEATURES.index(name)))
    if directions[best_feature] == 0:
        raise ValueError("Train has no useful factor direction")
    clip95 = float(train.target_return.abs().quantile(0.95))
    clip99 = float(train.target_return.abs().quantile(0.99))
    if not np.isfinite(clip95) or clip95 <= 0 or not np.isfinite(clip99) or clip99 <= 0:
        raise ValueError("Cannot determine robust label caps from train")
    if clip95 > clip99:
        raise ValueError("Invalid cap quantiles")

    candidates = {}
    models = {}
    baseline = pd.to_numeric(val.close_ma20_dev, errors="coerce").to_numpy(dtype=float)
    candidates["neg_ma20_fixed"] = -baseline
    candidates["train_best_single_factor"] = (
        val[best_feature].to_numpy(dtype=float) * directions[best_feature]
    )
    candidates["train_signed_rank_ensemble"] = _rank_ensemble(val, directions)

    # Original Ridge is refitted on TRAIN only, not on train+validation/test.
    # alpha is fixed and reported, never chosen by looking at validation here.
    models["ridge_original"] = fit_ridge(train, alpha)
    candidates["ridge_original"] = models["ridge_original"].predict(val)
    for name, limit in (("ridge_clip_abs_train_q95", clip95),
                        ("ridge_clip_abs_train_q99", clip99)):
        clipped = train.copy()
        clipped["target_return"] = clipped.target_return.clip(-limit, limit)
        models[name] = fit_ridge(clipped, alpha)
        candidates[name] = models[name].predict(val)

    rank_target = train.copy()
    rank_target["target_return"] = (
        rank_target.groupby("trade_date").target_return.rank(pct=True) - 0.5
    )
    models["ridge_daily_rank_target"] = fit_ridge(rank_target, alpha)
    candidates["ridge_daily_rank_target"] = models["ridge_daily_rank_target"].predict(val)

    reports, daily = [], []
    for name, predictions in candidates.items():
        scored = val.loc[:, ["trade_date", "stock_code", "target_return"]].copy()
        scored["predicted_return"] = predictions
        result = rank_diagnostics(scored, top_k=top_k, roundtrip_cost_bps=0)
        reports.append({
            "candidate": name,
            "mean_rank_ic": result["mean_rank_ic"],
            "topk_positive_share": result["topk_positive_share"],
            "topk_mean_forward_label": result["topk_mean_forward_return"],
            "pool_mean_forward_label": result["universe_mean_forward_return"],
            "topk_excess_vs_pool_label": result["topk_excess_vs_universe"],
            "evaluated_days": result["evaluated_days"],
            "evaluated_samples": result["evaluated_observations"],
        })
        daily.append(_validation_daily(scored, name=name, top_k=top_k))
    summary = {
        "comparison_version": "P4_TRAIN_VAL_LIGHT_BASELINES_V1",
        "train_samples": int(len(train)),
        "validation_samples": int(len(val)),
        "train_last_signal_date": str(train.trade_date.max().date()),
        "validation_first_signal_date": str(val.trade_date.min().date()),
        "alpha_FIXED_not_tuned_in_this_run": float(alpha),
        "label_caps_abs_quantile_from_train": {"q95": clip95, "q99": clip99},
        "train_ic_means": ic_means,
        "train_factor_directions": directions,
        "train_selected_best_single_factor": best_feature,
        "no_validation_winner_promoted": True,
        "test_labels_used": False,
        "warning": WARNING,
    }
    return summary, pd.DataFrame(reports), pd.concat(daily, ignore_index=True), models


def check_frozen_ridge_validation_ic(table: pd.DataFrame, old_metadata: dict, *,
                                     alpha: float, tolerance: float = 1e-6) -> dict:
    """Ensure fixed-alpha Ridge reproduces the saved validation diagnostic.

    This is a regression guard, NOT proof of row-identical market snapshots.
    A new candidate alpha with no saved reference is explicitly not checked.
    """
    if tolerance <= 0:
        raise ValueError("Invalid reference metric tolerance")
    tuning = [entry for entry in old_metadata.get("tuning", [])
              if abs(float(entry["alpha"]) - alpha) < 1e-12]
    if not tuning:
        return {"frozen_ridge_validation_ic_reference_available": False,
                "per_row_market_snapshot_identity_verified": False}
    if len(tuning) != 1:
        raise ValueError("Ambiguous reference Ridge alpha")
    old_ic = tuning[0]["validation_diagnostics"].get("mean_rank_ic")
    scored = table.loc[table.candidate.eq("ridge_original"), "mean_rank_ic"]
    if old_ic is None or len(scored) != 1 or not np.isfinite(float(old_ic)):
        raise ValueError("Incomplete Ridge reference diagnostic")
    new_ic = float(scored.iloc[0])
    if not np.isfinite(new_ic) or abs(new_ic - float(old_ic)) > tolerance:
        raise ValueError(
            f"Rebuilt fixed Ridge validation IC mismatch: old={old_ic}, new={new_ic}"
        )
    return {"frozen_ridge_validation_ic_reference_available": True,
            "frozen_ridge_validation_ic_reference": float(old_ic),
            "frozen_ridge_validation_ic_recomputed": new_ic,
            "frozen_ridge_validation_ic_absolute_error": abs(new_ic - float(old_ic)),
            "frozen_ridge_validation_ic_matched": True,
            "per_row_market_snapshot_identity_verified": False}


def main() -> None:
    p = argparse.ArgumentParser(description="P4 train/validation lightweight baselines; no trading")
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 3, 1))
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--top-n", type=int, default=800)
    p.add_argument("--min-price", type=float, default=3)
    p.add_argument("--min-amount", type=float, default=0)
    p.add_argument("--consecutive", type=int, default=40)
    p.add_argument("--hold-sessions", type=int, default=5)
    p.add_argument("--train-fraction", type=float, default=0.6)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="frozen research alpha, NOT tuned on known validation here")
    p.add_argument("--risk-events", type=Path,
                   default=Path("ashare/data/risk_events_verified_examples.csv"))
    p.add_argument("--no-risk-events", action="store_true")
    p.add_argument("--reference-metadata", type=Path, required=True,
                   help="original Ridge metadata; fail closed if rebuilt sample differs")
    p.add_argument("--db-chunk-size", type=int, default=50000)
    p.add_argument("--out", type=Path, default=Path("reports/ashare_p4_light_baselines"))
    args = p.parse_args()
    if args.end < args.start or args.db_chunk_size < 1 or args.top_k < 1 or args.alpha <= 0:
        p.error("Invalid date range, chunk size, top-k or alpha")
    cfg = PoolConfig(top_n=args.top_n, min_price=args.min_price,
                     min_amount=args.min_amount, consecutive_sessions=args.consecutive,
                     hold_sessions=args.hold_sessions)
    def log(message):
        print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)
    with args.reference_metadata.open(encoding="utf-8") as fp:
        old = json.load(fp)
    if (old.get("model_version") != "P4_RIDGE_V1" or
            old.get("pool_config") != asdict(cfg)):
        raise ValueError("Reference Ridge model version or pool config mismatch")
    log("[1/4] SELECT only public.market, no DB writes")
    raw = load_formal_ohlcv(args.start, args.end, hold_sessions=cfg.hold_sessions,
                            chunksize=args.db_chunk_size, progress=log)
    if raw.empty:
        raise ValueError("No market OHLCV rows")
    events = None
    if not args.no_risk_events:
        if not args.risk_events.is_file():
            p.error(f"Missing partial risk event file: {args.risk_events}")
        events = load_risk_events(args.risk_events)
    log("[2/4] Rebuild identical P4 pool and apply chronological label embargo")
    pool = build_pool(raw, args.start, args.end, cfg, events=events)
    split, train, val = split_audit_pool(
        pool.data, pool.calendar, train_fraction=args.train_fraction,
        val_fraction=args.val_fraction
    )
    if (split["validation_start"] != old["validation_start"] or
            split["test_start_EXCLUDED"] != old["test_start"] or
            split["eligible_refit_pretest_samples"] != old["refit_samples"] or
            pool.stats["selected_rows"] != old["pool_stats"]["selected_rows"] or
            len(val) != old["tuning"][0]["validation_diagnostics"]["evaluated_observations"]):
        raise ValueError("Reconstructed pool/split differs from original Ridge run")
    log("[3/4] Compare seven train-fitted/frozen candidates on validation ONLY")
    result, table, daily, models = compare_train_validation(
        train, val, top_k=args.top_k, alpha=args.alpha
    )
    # Cross-check the previously saved alpha's validation Rank IC. This never
    # reads the inspected old TEST labels and fails before emitting results.
    result.update(check_frozen_ridge_validation_ic(table, old, alpha=args.alpha))
    result.update({
        "dataset_start": args.start.isoformat(),
        "dataset_end": args.end.isoformat(),
        "original_validation_start": old["validation_start"],
        "original_test_start_EXCLUDED": old["test_start"],
        "original_refit_samples": old["refit_samples"],
        "reference_split_and_counts_matched": True,
        "per_row_market_snapshot_identity_verified": False,
        "pool_config": pool.stats["pool_config"],
        "research_warning": RESEARCH_WARNING,
    })
    folder = args.out / f"{args.start}_{args.end}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "baseline_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    table.to_csv(folder / "baseline_validation_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(folder / "baseline_validation_daily.csv", index=False, encoding="utf-8-sig")
    # Not saved as deployable inference models: these include a rank-target
    # variant whose scores are not return predictions, and known validation
    # is not independent evidence for selecting a winner.
    log(f"[4/4] Saved research comparison: {folder}")
    print(json.dumps({"outputs": str(folder), "candidates": table.candidate.tolist(),
                      "test_labels_used": False, "warning": WARNING},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
