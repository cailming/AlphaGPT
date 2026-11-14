"""P4.3 post-hoc Ridge holdout diagnostics; never model selection or trade PnL.

Reads saved test_predictions.csv + model_metadata.json. No database access,
model training, portfolio compounding, or execution fills.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ("trade_date", "stock_code", "liquidity_rank", "close",
           "target_return", "predicted_return")
WARNING = (
    "POST-HOC HOLDOUT DIAGNOSTIC: this test set has already been inspected. "
    "Do NOT tune/select new models from these results. Five-session labels "
    "overlap and prices are unadjusted. No fills, NAV, or executable PnL; "
    "daily Top-K overlap is NOT realized trading turnover. "
    "Reserve later genuinely unseen dates for confirmation."
)


def load_inputs(predictions: str | Path, metadata: str | Path) -> tuple[pd.DataFrame, dict]:
    with Path(metadata).open(encoding="utf-8") as file:
        meta = json.load(file)
    if meta.get("model_version") != "P4_RIDGE_V1":
        raise ValueError("Requires P4_RIDGE_V1 metadata")
    start = pd.Timestamp(meta["test_start"])
    end = pd.Timestamp(meta["test_end"])
    last_label = pd.Timestamp(meta["train_last_label_date"])
    if not last_label < start <= end:
        raise ValueError("Invalid purged train-label / test boundaries")
    frame = pd.read_csv(predictions, dtype={"stock_code": "string"},
                        encoding="utf-8-sig")
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing predictions columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Empty test_predictions.csv")
    frame["trade_date"] = pd.to_datetime(frame.trade_date, errors="raise").dt.normalize()
    if frame.trade_date.isna().any() or not frame.trade_date.between(start, end).all():
        raise ValueError("Predictions outside metadata test date range")
    if (frame.trade_date.min() != start or frame.trade_date.max() != end
            or len(frame) != int(meta["test_samples"])):
        raise ValueError("Test predictions do not match metadata boundaries or sample count")
    if frame.stock_code.isna().any() or not frame.stock_code.str.fullmatch(r"\d{6}").all():
        raise ValueError("Stock codes must be 6 digits")
    if frame.duplicated(["trade_date", "stock_code"]).any():
        raise ValueError("Duplicate test stock/date rows")
    for col in ("liquidity_rank", "close", "target_return", "predicted_return"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {col} in test prediction data")
    if not frame.liquidity_rank.ge(1).all() or not frame.close.gt(0).all():
        raise ValueError("Invalid liquidity ranks or close prices")
    cap = int(meta["pool_config"]["top_n"])
    if not frame.liquidity_rank.le(cap).all():
        raise ValueError("Liquidity rank exceeds model pool cap")
    return frame.sort_values(["trade_date", "stock_code"], kind="stable").reset_index(drop=True), meta


def evaluate_holdout(frame: pd.DataFrame, *, top_k: int = 20
                     ) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Daily equal-weight label summaries and score buckets; never realized returns."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    daily, buckets = [], []
    previous: set[str] | None = None
    for day, group in frame.groupby("trade_date", sort=True):
        ordered = group.sort_values(["predicted_return", "stock_code"],
                                    ascending=[False, True], kind="stable")
        chosen = ordered.head(top_k)
        names = set(chosen.stock_code)
        overlap = None if previous is None else len(names & previous)
        replacements = None if overlap is None else len(names - previous)
        previous = names
        daily.append({
            "trade_date": day, "observations": len(group),
            "rank_ic": (float(group.predicted_return.rank().corr(group.target_return.rank()))
                        if len(group) >= 3 else float("nan")),
            "pool_positive_share": float(group.target_return.gt(0).mean()),
            "pool_forward_return": float(group.target_return.mean()),
            "topk_count": len(chosen),
            "topk_positive_share": float(chosen.target_return.gt(0).mean()),
            "topk_forward_return": float(chosen.target_return.mean()),
            "topk_excess_vs_pool": float(chosen.target_return.mean() - group.target_return.mean()),
            "topk_overlap_previous": overlap,
            "topk_new_names_previous": replacements,
            "topk_replacement_share": (None if overlap is None else
                                       float(replacements / max(1, len(chosen)))),
        })
        if len(ordered) >= 10:
            for bin_id, sample in ordered.groupby(
                np.floor(np.arange(len(ordered)) * 10 / len(ordered)).astype(int), sort=True
            ):
                # decile 0 = highest predicted scores; no calibrated probability.
                buckets.append({
                    "trade_date": day, "score_decile_highest_is_0": int(bin_id),
                    "rows": len(sample),
                    "positive_share": float(sample.target_return.gt(0).mean()),
                    "mean_target_return": float(sample.target_return.mean()),
                    "mean_predicted_return": float(sample.predicted_return.mean()),
                })
    daily_df = pd.DataFrame(daily)
    daily_df["month"] = daily_df.trade_date.dt.to_period("M").astype(str)
    monthly = daily_df.groupby("month", sort=True).agg(
        evaluated_days=("trade_date", "size"),
        test_samples=("observations", "sum"),
        mean_rank_ic=("rank_ic", "mean"),
        mean_topk_positive_share=("topk_positive_share", "mean"),
        mean_pool_positive_share=("pool_positive_share", "mean"),
        mean_topk_forward_return=("topk_forward_return", "mean"),
        mean_pool_forward_return=("pool_forward_return", "mean"),
        mean_topk_excess=("topk_excess_vs_pool", "mean"),
        mean_daily_topk_replacement_share=("topk_replacement_share", "mean"),
    ).reset_index()
    deciles = pd.DataFrame(buckets)
    score_buckets = deciles.groupby("score_decile_highest_is_0", sort=True).agg(
        days=("trade_date", "size"), rows=("rows", "sum"),
        mean_positive_share=("positive_share", "mean"),
        mean_forward_return=("mean_target_return", "mean"),
        mean_predicted_return=("mean_predicted_return", "mean"),
    ).reset_index() if not deciles.empty else pd.DataFrame()
    summary = {
        "evaluated_days": int(len(daily_df)),
        "evaluated_samples": int(len(frame)),
        "top_k": int(top_k),
        "mean_rank_ic": float(daily_df.rank_ic.mean()),
        "mean_topk_positive_share": float(daily_df.topk_positive_share.mean()),
        "mean_pool_positive_share": float(daily_df.pool_positive_share.mean()),
        "mean_topk_forward_return": float(daily_df.topk_forward_return.mean()),
        "mean_pool_forward_return": float(daily_df.pool_forward_return.mean()),
        "mean_topk_excess": float(daily_df.topk_excess_vs_pool.mean()),
        "mean_daily_topk_replacement_share": float(daily_df.topk_replacement_share.mean()),
        "warning": WARNING,
    }
    return summary, daily_df, monthly, score_buckets


def main() -> None:
    p = argparse.ArgumentParser(description="Post-hoc diagnostics for P4 Ridge holdout only")
    p.add_argument("--model-dir", type=Path, required=True,
                   help="directory with model_metadata.json and test_predictions.csv")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("reports/ashare_p4_diagnostics"))
    args = p.parse_args()
    def log(msg: str) -> None:
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
    log("[1/3] Validate saved holdout predictions, no DB or model training")
    frame, meta = load_inputs(args.model_dir / "test_predictions.csv",
                              args.model_dir / "model_metadata.json")
    log("[2/3] Daily / monthly / predicted-score-decile post-hoc diagnostics")
    result, daily, monthly, deciles = evaluate_holdout(frame, top_k=args.top_k)
    result["metadata_test_metrics"] = meta.get("test_metrics")
    result["metadata_baseline_metrics"] = meta.get("baseline_metrics")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "diagnostics.json").write_text(json.dumps(result, ensure_ascii=False,
                                               indent=2), encoding="utf-8")
    daily.to_csv(args.out / "daily_diagnostics.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(args.out / "monthly_diagnostics.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(args.out / "score_deciles.csv", index=False, encoding="utf-8-sig")
    log(f"[3/3] Saved diagnostics to {args.out}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
