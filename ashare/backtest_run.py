"""P3.1 baseline exploratory daily backtest; never places real orders.

python -u -m ashare.backtest_run --start 2025-03-01 --end 2026-09-30 --exploratory
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path

from .backtest import (BacktestConfig, load_formal_market,
                       make_ma20_reversion_signals, run_backtest)


def main():
    p = argparse.ArgumentParser(description="P3.1 A-share exploratory open-fill backtest")
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 3, 1))
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--codes", help="Comma-separated stock codes")
    p.add_argument("--db-chunk-size", type=int, default=50000)
    p.add_argument("--exploratory", action="store_true",
                   help="Explicitly accept missing historical ST/limit/corporate-action checks")
    p.add_argument("--out", default="reports/ashare_backtest")
    args = p.parse_args()
    if not args.exploratory:
        p.error("Historical tradeability dataset not yet integrated; --exploratory "
                "required for a provisional research run")
    cfg = BacktestConfig(args.start, args.end, allow_unverified_tradeability=True)

    def progress(message):
        print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)

    codes = args.codes.split(",") if args.codes else None
    progress("[1/3] 读取正式日线并校验数据来源")
    market = load_formal_market(args.start, args.end, stock_codes=codes,
                                chunksize=args.db_chunk_size, progress=progress)
    if market.empty:
        raise SystemExit("No formal daily market bars returned")
    progress(f"[2/3] 生成 NEG(ma20_dev) 收盘信号：股票 {market.stock_code.nunique():,} 只")
    signals = make_ma20_reversion_signals(market)
    progress(f"[3/3] 模拟 T+1 买入、最早 T+2 卖出（探索性、非成交记录）")
    result = run_backtest(market, signals, cfg)
    out = Path(args.out) / f"{args.start}_{args.end}"
    out.mkdir(parents=True, exist_ok=True)
    result.equity.to_csv(out / "daily_equity.csv", index=False, encoding="utf-8-sig")
    result.trades.to_csv(out / "simulated_trades.csv", index=False, encoding="utf-8-sig")
    result.rejected.to_csv(out / "rejected_attempts.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(f"探索性回测完成，输出：{out}")
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
