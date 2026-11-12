"""Example A-share factor mining diagnostic (read-only; no order placement).

python -m ashare.research --start 2025-01-01 --end 2026-09-30 --samples 200
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import time
from datetime import datetime
import math

from .daily import AShareDailyLoader
from .tensor import build_daily_tensor
from .evaluate import search_baseline


def main() -> None:
    p = argparse.ArgumentParser(description="A-share formula IC research baseline (no trading)")
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--codes", help="Optional comma-separated symbols")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-stocks", type=int, default=15)
    p.add_argument("--min-train-days", type=int, default=20)
    p.add_argument("--db-chunk-size", type=int, default=50000,
                   help="PostgreSQL 每批读取行数，默认 50000")
    p.add_argument("--log-every", type=int, default=10,
                   help="每搜索多少个候选公式打印一次进度，默认 10")
    p.add_argument("--quiet", action="store_true", help="仅输出最终结果，不打印进度")
    p.add_argument("--allow-unverified-main-force", action="store_true",
                   help="Disable strict point-in-time timestamp check FOR DIAGNOSTICS ONLY")
    args = p.parse_args()
    if (args.end < args.start or args.samples < 1 or args.min_stocks < 2
            or args.db_chunk_size < 1 or args.log_every < 1):
        p.error("Invalid date range or numerical parameter")

    start_time = time.perf_counter()

    def notify(message: str) -> None:
        if not args.quiet:
            print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)

    symbols = args.codes.split(",") if args.codes else None
    notify(f"[1/4] 开始读取行情：{args.start} ~ {args.end}，"
           f"{'指定股票 ' + str(len(symbols)) + ' 只' if symbols else '全市场'}")
    fetch_start = time.perf_counter()
    raw = AShareDailyLoader().load(
        args.start, args.end, stock_codes=symbols, include_warmup=True,
        progress=notify if not args.quiet else None,
        chunksize=args.db_chunk_size,
    )
    notify(f"[1/4] 数据读取及基础因子计算完成：{len(raw):,} 行，"
           f"耗时 {time.perf_counter() - fetch_start:.1f} 秒")
    if raw.empty:
        raise SystemExit("No market records found for this range")

    notify(f"[2/4] 构建张量：股票 {raw['stock_code'].nunique():,} 只，"
           f"交易日 {raw['trade_date'].nunique():,} 天（包含预热数据）")
    tensor_start = time.perf_counter()
    tensor = build_daily_tensor(
        raw, strict_force_asof=not args.allow_unverified_main_force
    )
    availability = tensor.feature_valid.float().mean().item()
    notify(f"[2/4] 张量完成：{tuple(tensor.features.shape)} [股票, 因子, 日期]，"
           f"有效因子占比 {availability:.1%}，"
           f"耗时 {time.perf_counter() - tensor_start:.1f} 秒")

    notify(f"[3/4] 开始搜索 {args.samples} 个候选公式，"
           f"每 {args.log_every} 个打印进度")
    search_start = time.perf_counter()

    def on_candidate(done: int, total: int, unique: int,
                     scored: int, best_ic: float) -> None:
        elapsed = time.perf_counter() - search_start
        remaining = elapsed * (total - done) / done if done else float("nan")
        best_text = f"{best_ic:.6f}" if math.isfinite(best_ic) else "尚无有效候选"
        notify(f"[3/4] 公式进度 {done}/{total} ({done/total:.1%})，"
               f"不同公式 {unique}，有效候选 {scored}，"
               f"当前最佳训练 IC {best_text}，"
               f"已用 {elapsed:.1f}s，预计剩余约 {remaining:.1f}s")

    result = search_baseline(
        tensor, start=args.start, end=args.end,
        samples=args.samples, seed=args.seed,
        min_stocks=args.min_stocks, min_train_days=args.min_train_days,
        progress=on_candidate if not args.quiet else None,
        progress_every=args.log_every,
    )
    notify(f"[3/4] 搜索完成，耗时 {time.perf_counter() - search_start:.1f} 秒")
    notify(f"[4/4] 结果汇总，总耗时 {time.perf_counter() - start_time:.1f} 秒")
    if result is None:
        print("No candidate met minimum training IC observations. Check coverage/as-of filters.")
        return
    print(json.dumps({
        "formula": result["formula"],
        "tokens": result["tokens"],
        "train_ic": vars(result["train"]), "holdout_ic": vars(result["test"]),
        "unique_candidates": result["unique_candidates"],
        "scored_candidates": result["scored_candidates"],
        "warning": "Rank IC diagnostic only; not a PnL/trading backtest.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
