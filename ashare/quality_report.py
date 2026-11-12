"""Run factor and historical main-force data-quality checks (SELECT-only)."""
from __future__ import annotations
import argparse
from datetime import date, datetime
import json
from pathlib import Path
from .daily import AShareDailyLoader
from .quality import analyze_factor_quality

def main() -> None:
    parser = argparse.ArgumentParser(description="A-share factor coverage and main-force timestamp audit")
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--codes", help="Comma-separated stock codes, omit for all")
    parser.add_argument("--db-chunk-size", type=int, default=50000)
    parser.add_argument("--out", default="reports/ashare_quality", help="Local report root directory")
    args = parser.parse_args()
    if args.start > args.end or args.db_chunk_size < 1:
        parser.error("Invalid dates/chunk size")

    def info(message: str) -> None:
        print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)

    codes = args.codes.split(",") if args.codes else None
    info("[1/3] 从 market 和 stock_main_force_daily 读取历史数据（只读）")
    df = AShareDailyLoader().load(args.start, args.end, codes,
                                  include_warmup=True, progress=info,
                                  chunksize=args.db_chunk_size)
    if df.empty:
        raise SystemExit("查询区间没有记录")
    info(f"[2/3] 计算 12 个因子覆盖率及主力评分生成时间分布；共 {len(df):,} 行含预热历史")
    report = analyze_factor_quality(df, start=args.start, end=args.end)
    out = Path(args.out) / f"{args.start}_{args.end}"
    out.mkdir(parents=True, exist_ok=True)
    report.summary.to_csv(out / "factor_summary.csv", index=False, encoding="utf-8-sig")
    report.monthly.to_csv(out / "factor_monthly.csv", index=False, encoding="utf-8-sig")
    report.backfill_batches.to_csv(out / "main_force_created_batches.csv", index=False, encoding="utf-8-sig")
    (out / "main_force_audit.json").write_text(
        json.dumps(report.audit, ensure_ascii=False, indent=2), encoding="utf-8")
    info("[3/3] 质量报告保存完成")
    print(report.summary[["factor", "raw_non_null_pct_market",
                          "tensor_mask_pct_market", "quality_eligible_pct_market",
                          "created_at_late", "out_of_range"]].to_string(index=False))
    print(json.dumps(report.audit, ensure_ascii=False, indent=2))
    print(f"报告目录：{out}")

if __name__ == "__main__":
    main()
