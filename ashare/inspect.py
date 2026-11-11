"""Read-only data smoke check. Example: python -m ashare.inspect daily 2026-08-01 2026-08-31"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from .daily import AShareDailyLoader, DAILY_COLUMNS, LABEL_COLUMN
from .minute import AShareMinuteLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only A-share data adapter check")
    parser.add_argument("mode", choices=["daily", "minute"])
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--codes", help="Comma-separated stock codes; omit for all stocks")
    parser.add_argument("--include-today", action="store_true", help="Minute mode: also read billing")
    args = parser.parse_args()
    codes = args.codes.split(",") if args.codes else None
    if args.mode == "daily":
        result = AShareDailyLoader().load(args.start, args.end, codes)
    else:
        result = AShareMinuteLoader().load(args.start, args.end, codes,
                                           include_today=args.include_today)
    print("rows:", len(result), "stocks:", result["stock_code"].nunique() if not result.empty else 0)
    if result.empty:
        return
    if args.mode == "daily":
        print("missing features:")
        print(result[list(DAILY_COLUMNS) + [LABEL_COLUMN]].isna().mean().round(3).to_string())
    else:
        print("interval quality:")
        print(result["quality_flag"].value_counts(dropna=False).to_string())
    # Do not leak credentials or print full market histories.
    print(result.head(3).to_string(index=False, max_cols=12))


if __name__ == "__main__":
    main()
