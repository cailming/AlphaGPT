"""P4.1 standalone read-only stock-pool audit; does not train models."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime
import json
from pathlib import Path

import pandas as pd

from .p4_mvp import PoolConfig, PoolResult, RESEARCH_WARNING, build_pool, load_formal_ohlcv
from .risk_events import load_risk_events


def build_coverage_tables(
    pool: PoolResult, start: date, end: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily and monthly pool/label coverage including zero-selection days."""
    dates = pool.calendar[
        (pool.calendar >= pd.Timestamp(start)) & (pool.calendar <= pd.Timestamp(end))
    ]
    daily = pd.DataFrame({"trade_date": dates})
    selected = pool.data
    if selected.empty:
        daily["pool_size"] = 0
        daily["valid_label_samples"] = 0
        daily["median_liquidity"] = float("nan")
    else:
        by_day = selected.groupby("trade_date").agg(
            pool_size=("stock_code", "size"),
            valid_label_samples=("target_return", "count"),
            median_liquidity=("liquidity_20", "median"),
        ).reset_index()
        daily = daily.merge(by_day, on="trade_date", how="left")
        for column in ("pool_size", "valid_label_samples"):
            daily[column] = daily[column].fillna(0).astype(int)

    daily["label_coverage"] = daily.valid_label_samples.div(
        daily.pool_size.where(daily.pool_size.gt(0))
    )
    daily["month"] = daily.trade_date.dt.to_period("M").astype(str)
    monthly = daily.groupby("month", sort=True).agg(
        trading_days=("trade_date", "size"),
        days_with_candidates=("pool_size", lambda s: int(s.gt(0).sum())),
        selected_samples=("pool_size", "sum"),
        valid_label_samples=("valid_label_samples", "sum"),
        average_daily_pool_size=("pool_size", "mean"),
        min_daily_pool_size=("pool_size", "min"),
        max_daily_pool_size=("pool_size", "max"),
    ).reset_index()
    if selected.empty:
        monthly["unique_stock_codes"] = 0
    else:
        codes = (
            selected.assign(month=selected.trade_date.dt.to_period("M").astype(str))
            .groupby("month").stock_code.nunique()
        )
        monthly["unique_stock_codes"] = monthly.month.map(codes).fillna(0).astype(int)
    monthly["valid_label_share"] = monthly.valid_label_samples.div(
        monthly.selected_samples.where(monthly.selected_samples.gt(0))
    )
    return daily, monthly


def main() -> None:
    p = argparse.ArgumentParser(description="P4.1 causal liquidity-pool audit (SELECT only)")
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 3, 1))
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--top-n", type=int, default=800)
    p.add_argument("--min-price", type=float, default=3.0)
    p.add_argument("--min-amount", type=float, default=0.0,
                   help="20-session median amount in DB native units")
    p.add_argument("--consecutive", type=int, default=40)
    p.add_argument("--hold-sessions", type=int, default=5)
    p.add_argument("--db-chunk-size", type=int, default=50000)
    p.add_argument("--risk-events", type=Path,
                   default=Path("ashare/data/risk_events_verified_examples.csv"))
    p.add_argument("--no-risk-events", action="store_true")
    p.add_argument("--out", type=Path, default=Path("reports/ashare_p4_pool"))
    args = p.parse_args()
    if args.end < args.start or args.db_chunk_size < 1:
        p.error("Invalid date range or chunk size")
    cfg = PoolConfig(
        top_n=args.top_n, min_price=args.min_price, min_amount=args.min_amount,
        consecutive_sessions=args.consecutive, hold_sessions=args.hold_sessions,
    )

    def progress(message: str) -> None:
        print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)

    progress("[1/3] Reading formal daily OHLCV (read-only)")
    raw = load_formal_ohlcv(
        args.start, args.end, hold_sessions=cfg.hold_sessions,
        chunksize=args.db_chunk_size, progress=progress,
    )
    if raw.empty:
        raise SystemExit("No formal daily bars in requested/warmup range")
    events, event_path = None, None
    if not args.no_risk_events:
        if not args.risk_events.is_file():
            p.error(f"Risk-event evidence file not found: {args.risk_events}")
        events = load_risk_events(args.risk_events)
        event_path = str(args.risk_events)

    progress("[2/3] Building dynamic pool and daily/monthly coverage")
    pool = build_pool(raw, args.start, args.end, cfg, events=events)
    daily, monthly = build_coverage_tables(pool, args.start, args.end)
    report = {
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "data_source": "public.market WHERE not_billing=1",
        "pool_config": asdict(cfg),
        "risk_event_file": event_path,
        "pool_stats": pool.stats,
        "total_observed_market_days": int(len(daily)),
        "months_reported": int(len(monthly)),
        "warning": RESEARCH_WARNING,
    }
    folder = args.out / f"{args.start}_{args.end}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pool_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    daily.to_csv(folder / "pool_daily_coverage.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(folder / "pool_monthly_coverage.csv", index=False, encoding="utf-8-sig")
    pool.data.loc[:, [
        "trade_date", "stock_code", "liquidity_rank", "close",
        "liquidity_20", "target_return",
    ]].to_csv(folder / "pool_selected.csv", index=False, encoding="utf-8-sig")
    progress(f"[3/3] Saved audit to {folder}")
    print(json.dumps({
        "selected_stock_codes": pool.stats["selected_stock_codes"],
        "selected_rows": pool.stats["selected_rows"],
        "valid_labels": pool.stats["rows_with_valid_label"],
        "rejected_by_reason": pool.stats["rejected_by_reason"],
        "months_reported": len(monthly),
        "output": str(folder),
        "warning": "Research diagnostics only; no audited trading profitability.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
