"""P4 model inference: same past-only features and pool as training."""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path

from .p4_mvp import PoolConfig, build_pool, load_formal_ohlcv, RidgeModel
from .risk_events import load_risk_events


def main() -> None:
    p = argparse.ArgumentParser(description="P4 Ridge research ranking (NOT trade orders)")
    p.add_argument("--model-dir", type=Path, required=True,
                   help="directory containing ridge_model.npz and model_metadata.json")
    p.add_argument("--asof", type=date.fromisoformat, required=True,
                   help="must be a market session at/after last training label date")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--min-predicted-return", type=float, default=None,
                   help="Optional abstention floor; no recommendation if score is below it")
    p.add_argument("--risk-events", type=Path,
                   help="override training risk evidence; absent uses saved path")
    p.add_argument("--db-chunk-size", type=int, default=50000)
    p.add_argument("--out", type=Path, default=Path("reports/ashare_p4_predict"))
    args = p.parse_args()
    if args.top_k < 1 or args.db_chunk_size < 1:
        p.error("top-k and db-chunk-size must be positive")
    metadata = json.loads((args.model_dir / "model_metadata.json").read_text(encoding="utf8"))
    if args.asof < date.fromisoformat(metadata["train_last_label_date"]):
        p.error(
            "Requested asof predates labels used for model fitting; would leak future training labels"
        )
    cfg = PoolConfig(**metadata["pool_config"])
    model = RidgeModel.load(args.model_dir / "ridge_model.npz")
    risk_path = args.risk_events
    if risk_path is None and metadata.get("risk_event_file"):
        risk_path = Path(metadata["risk_event_file"])
    events = None
    if risk_path is not None:
        if not risk_path.is_file():
            p.error(f"Risk evidence file is missing: {risk_path}")
        events = load_risk_events(risk_path)

    def log(s):
        print(f"[{datetime.now():%H:%M:%S}] {s}", flush=True)

    log("读取所选 T 日及其历史预热行情；不读取 T 日以后的价格")
    raw = load_formal_ohlcv(args.asof, args.asof,
                            hold_sessions=cfg.hold_sessions,
                            chunksize=args.db_chunk_size, progress=log,
                            include_label_tail=False)
    if raw.empty:
        raise SystemExit("No formal market bars available on/before asof")
    pool = build_pool(raw, args.asof, args.asof, cfg, events=events)
    current = pool.data
    if current.empty or current.trade_date.max().date() != args.asof:
        raise SystemExit("No eligible pool on requested trading day")
    current = current.copy()
    current["predicted_return"] = model.predict(current)
    current = current.sort_values(
        ["predicted_return", "stock_code"], ascending=[False, True]
    ).reset_index(drop=True)
    current["prediction_rank"] = current.index + 1
    pool_size = len(current)
    if args.min_predicted_return is not None:
        current = current.loc[
            current.predicted_return >= args.min_predicted_return
        ].reset_index(drop=True)
        current["prediction_rank"] = current.index + 1

    args.out.mkdir(parents=True, exist_ok=True)
    file = args.out / f"predictions_{args.asof}.csv"
    current.head(args.top_k).loc[:, [
        "trade_date", "stock_code", "close", "liquidity_rank",
        "predicted_return", "prediction_rank"
    ]].to_csv(file, index=False, encoding="utf-8-sig")
    log(f"候选股票池 {pool_size} 只，符合预测阈值 {len(current)} 只，"
        f"输出 {min(args.top_k,len(current))} 只至 {file}")
    print(json.dumps({
        "asof": args.asof.isoformat(),
        "pool_size": pool_size,
        "above_score_floor": len(current),
        "score_floor": args.min_predicted_return,
        "returned_top_k": min(args.top_k, len(current)),
        "model_last_train_label_date": metadata["train_last_label_date"],
        "risk_evidence": str(risk_path) if risk_path else "NOT_APPLIED",
        "warning": "RESEARCH RANKINGS ONLY; do not interpret as automatic trade recommendations.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
