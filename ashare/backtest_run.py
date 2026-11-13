"""P3.2 exploratory daily backtest with stale-NAV and candidate audit; no real orders.

python -u -m ashare.backtest_run --start 2025-03-01 --end 2026-09-30 --exploratory
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path

from .backtest import (BacktestConfig, load_formal_market,
                       make_ma20_reversion_signals, run_backtest)
from .st_history import (
    filter_signals_by_historical_st, load_st_snapshots, load_st_coverage,
)


def main():
    p = argparse.ArgumentParser(description="P3.2 diagnostic A-share open-fill simulation")
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 3, 1))
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--codes", help="Comma-separated stock codes")
    p.add_argument("--db-chunk-size", type=int, default=50000)
    p.add_argument("--candidate-pool", type=int, default=20,
                   help="T日收盘排序后保留前N只候选股票；不再遍历全市场买入")
    p.add_argument("--exploratory", action="store_true",
                   help="Explicitly accept missing historical ST/limit/corporate-action checks")
    p.add_argument("--log-every-days", type=int, default=20,
                   help="每多少个回测交易日打印一次进度")
    p.add_argument("--st-daily", type=Path,
                   help="CSV containing historical full-market ST-listed stocks by trade_date")
    p.add_argument("--st-coverage", type=Path,
                   help="CSV certifying dates whose ST daily list is complete")
    p.add_argument("--allow-partial-st-coverage", action="store_true",
                   help="Research-only: incomplete ST dates block ALL signals on those dates")
    p.add_argument("--out", default="reports/ashare_backtest")
    args = p.parse_args()
    if args.log_every_days < 1 or args.db_chunk_size < 1 or args.candidate_pool < 1:
        p.error("Progress and chunk sizes must be positive")
    if bool(args.st_daily) != bool(args.st_coverage):
        p.error("--st-daily and --st-coverage must be given together")
    if not args.exploratory:
        p.error("Historical tradeability dataset not yet integrated; --exploratory "
                "required for a provisional research run")
    cfg = BacktestConfig(args.start, args.end,
                         candidate_pool_size=args.candidate_pool,
                         allow_unverified_tradeability=True)

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
    # Warmup history is used ONLY for indicators, not official backtest
    # signals or ST-history coverage requirements.
    signals = signals.loc[
        (signals.signal_date.dt.date >= args.start) &
        (signals.signal_date.dt.date <= args.end)
    ].reset_index(drop=True)
    st_audit = None
    if args.st_daily is not None:
        progress("读取历史 ST 列表及每日完整性证明（当前 stock_info.st 禁止作为历史依据）")
        st_result = filter_signals_by_historical_st(
            signals,
            load_st_snapshots(args.st_daily),
            load_st_coverage(args.st_coverage),
        )
        st_audit = st_result.stats
        if (st_audit["complete_signal_days"] != st_audit["signal_days"]
                and not args.allow_partial_st_coverage):
            p.error(
                "ST snapshot coverage is incomplete: "
                f"{st_audit['complete_signal_days']}/{st_audit['signal_days']} "
                "signal days. Supply complete historical snapshots or use "
                "--allow-partial-st-coverage for a noncomparable diagnostic run."
            )
        signals = st_result.signals
        progress(
            f"历史ST信号日过滤：允许 {st_audit['eligible_normal_rows']:,}，"
            f"ST排除 {st_audit['blocked_st_rows']:,}，"
            f"状态未知排除 {st_audit['blocked_unknown_rows']:,}"
        )
    else:
        progress("警告：未提供历史 ST 列表；本次仅为无 ST 过滤的探索性回测")
    progress(f"[3/3] 模拟 T+1 买入、最早 T+2 卖出（探索性、非成交记录）")
    result = run_backtest(market, signals, cfg,
                          progress=progress, progress_every=args.log_every_days)
    result.metrics["historical_st_filter"] = (
        st_audit if st_audit is not None else {"status": "NOT_APPLIED"}
    )
    out = Path(args.out) / f"{args.start}_{args.end}"
    out.mkdir(parents=True, exist_ok=True)
    result.equity.to_csv(out / "daily_equity.csv", index=False, encoding="utf-8-sig")
    result.trades.to_csv(out / "simulated_trades.csv", index=False, encoding="utf-8-sig")
    result.rejected.to_csv(out / "rejected_attempts.csv", index=False, encoding="utf-8-sig")
    result.open_positions.to_csv(out / "open_positions.csv", index=False, encoding="utf-8-sig")
    if st_audit is not None:
        st_result.per_day.to_csv(out / "st_filter_by_day.csv",
                                 index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result.metrics["last_day_valuation_fresh"]:
        progress("警告：期末持仓中存在过期估值；最终收益率不可用于策略业绩判断")
    progress(f"探索性回测完成，输出：{out}")
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
