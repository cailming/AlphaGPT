"""Example A-share factor mining diagnostic (read-only; no order placement).

python -m ashare.research --start 2025-01-01 --end 2026-09-30 --samples 200
"""
from __future__ import annotations

import argparse
from datetime import date
import json

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
    p.add_argument("--allow-unverified-main-force", action="store_true",
                   help="Disable strict point-in-time timestamp check FOR DIAGNOSTICS ONLY")
    args = p.parse_args()
    if args.end < args.start or args.samples < 1 or args.min_stocks < 2:
        p.error("Invalid date range, samples, or minimum stock count")
    symbols = args.codes.split(",") if args.codes else None
    raw = AShareDailyLoader().load(args.start, args.end, stock_codes=symbols,
                                   include_warmup=True)
    if raw.empty:
        raise SystemExit("No market records found for this range")
    tensor = build_daily_tensor(raw,
        strict_force_asof=not args.allow_unverified_main_force)
    result = search_baseline(tensor, start=args.start, end=args.end,
                             samples=args.samples, seed=args.seed,
                             min_stocks=args.min_stocks,
                             min_train_days=args.min_train_days)
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
