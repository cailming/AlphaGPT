"""Train a usable A-share universe Ridge baseline (P4 research only).

python -u -m ashare.p4_train --start 2025-03-01 --end 2026-09-30
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime
import json
from pathlib import Path

import pandas as pd

from .p4_mvp import (
    PoolConfig, build_pool, fit_splits, load_formal_ohlcv, RESEARCH_WARNING,
)
from .risk_events import load_risk_events


def main() -> None:
    p = argparse.ArgumentParser(description="P4 causal liquid-stock research baseline; no trades")
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 3, 1))
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--top-n", type=int, default=800)
    p.add_argument("--min-price", type=float, default=3.0,
                   help="minimum T close in database price units (unadjusted)")
    p.add_argument("--min-amount", type=float, default=0,
                   help="minimum trailing 20-session amount MEDIAN in DB source units")
    p.add_argument("--consecutive", type=int, default=40)
    p.add_argument("--hold-sessions", type=int, default=5,
                   help="open-to-open label horizon: T+1 open -> T+1+H open")
    p.add_argument("--train-fraction", type=float, default=0.6)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--roundtrip-cost-bps", type=float, default=25.0,
                   help="ranking diagnostic cost estimate, not executed PnL")
    p.add_argument("--db-chunk-size", type=int, default=50000)
    p.add_argument("--risk-events", type=Path,
                   default=Path("ashare/data/risk_events_verified_examples.csv"),
                   help="optional PARTIAL public risk announcements (default tracked examples)")
    p.add_argument("--no-risk-events", action="store_true",
                   help="skip even the small known public announcement overlay")
    p.add_argument("--out", type=Path, default=Path("reports/ashare_p4_mvp"))
    args = p.parse_args()
    if args.end < args.start or args.db_chunk_size < 1 or args.top_k < 1:
        p.error("Invalid end/start/chunk/top-k")
    cfg = PoolConfig(top_n=args.top_n, min_price=args.min_price,
                     min_amount=args.min_amount,
                     consecutive_sessions=args.consecutive,
                     hold_sessions=args.hold_sessions)

    def progress(s):
        print(f"[{datetime.now():%H:%M:%S}] {s}", flush=True)

    progress("[1/4] 正式日线 SELECT：只有 market OHLCV；不使用回填主力资金")
    raw = load_formal_ohlcv(args.start, args.end, hold_sessions=cfg.hold_sessions,
                            chunksize=args.db_chunk_size, progress=progress)
    if raw.empty:
        raise SystemExit("No formal market data")
    events = None
    event_path = None
    if not args.no_risk_events:
        if not args.risk_events.is_file():
            p.error(f"Risk-event file not found: {args.risk_events}")
        events = load_risk_events(args.risk_events)
        event_path = str(args.risk_events)
    progress("[2/4] 计算过去窗口因子、T日流动性股票池及严格未来交易日标签")
    pool = build_pool(raw, args.start, args.end, cfg, events=events)
    if pool.data.empty:
        raise SystemExit("No eligible liquid stocks; reduce min_price/min_amount/streak")
    progress(f"股票池：{pool.stats['selected_stock_codes']:,} 只曾入选，"
             f"{pool.stats['selected_rows']:,} 条T日样本，"
             f"{pool.stats['rows_with_valid_label']:,} 条可训练标签")

    progress("[3/4] 按日期分割、标签区间净化、训练 Ridge、验证选参并评估测试集")
    model, report, test, latest = fit_splits(
        pool, train_fraction=args.train_fraction, val_fraction=args.val_fraction,
        top_k=args.top_k, roundtrip_cost_bps=args.roundtrip_cost_bps,
    )
    report["risk_event_file"] = event_path
    report["dataset_start"] = args.start.isoformat()
    report["dataset_end"] = args.end.isoformat()
    report["feature_data_source"] = "public.market WHERE not_billing=1"
    report["model_is_tradable_strategy"] = False
    report["is_full_historical_st_history"] = False
    report["warning"] = RESEARCH_WARNING

    out = args.out / f"{args.start}_{args.end}"
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "ridge_model.npz")
    (out / "model_metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8"
    )
    test_cols = ["trade_date", "stock_code", "liquidity_rank", "close",
                 "target_return", "predicted_return"]
    test.loc[:, test_cols].sort_values(
        ["trade_date", "predicted_return", "stock_code"],
        ascending=[True, False, True]
    ).to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
    latest.loc[:, ["trade_date", "stock_code", "liquidity_rank",
                   "close", "predicted_return"]].head(50).to_csv(
        out / "latest_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pool.data.groupby("trade_date", sort=True).agg(
        pool_size=("stock_code", "size"),
        valid_labels=("target_return", "count"),
        median_liquidity=("liquidity_20", "median"),
    ).reset_index().to_csv(
        out / "pool_daily_coverage.csv", index=False, encoding="utf-8-sig"
    )
    progress("[4/4] 保存模型、训练元数据和样本外排名评估")
    progress(f"输出 {out}")
    print(json.dumps({
        "model_version": report["model_version"],
        "pool_stats": report["pool_stats"],
        "chosen_alpha": report["selection_alpha"],
        "validation_start": report["validation_start"],
        "test_start": report["test_start"],
        "test_metrics": report["test_metrics"],
        "baseline_metrics": report["baseline_metrics"],
        "warning": RESEARCH_WARNING,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
