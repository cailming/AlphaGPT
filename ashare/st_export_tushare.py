"""Optional historical ST daily snapshot exporter from Tushare.

Read-only PostgreSQL, remote API read-only; local CSV output only.
Requires TUSHARE_TOKEN and permission for the stock_st endpoint.
"""
from __future__ import annotations
import argparse
from datetime import date, datetime
import os
from pathlib import Path
import time
import pandas as pd
from sqlalchemy import text
from .daily import AShareDailyLoader
from .st_history import _codes, _dates, load_st_coverage, load_st_snapshots


def _save(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def _formal_dates(start: date, end: date) -> list:
    engine = AShareDailyLoader().engine
    sql = text("""
        SELECT DISTINCT trade_date FROM public.market
         WHERE not_billing = 1 AND trade_date BETWEEN :start AND :end
         ORDER BY trade_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"start": start, "end": end})
    if df.empty:
        raise ValueError("未查到正式行情交易日期")
    return pd.to_datetime(df.trade_date).dt.normalize().tolist()


def export_tushare_st(start: date, end: date, out: Path,
                      *, pause_seconds: float = 0.5, retries: int = 3) -> dict:
    if end < start or pause_seconds < 0 or retries < 1:
        raise ValueError("Invalid export configuration")
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required")
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("Optional tushare dependency missing; install only after approval") from exc
    dates = _formal_dates(start, end)
    out.mkdir(parents=True, exist_ok=True)
    st_file, cov_file = out / "st_daily.csv", out / "st_coverage.csv"
    if st_file.exists() != cov_file.exists():
        raise RuntimeError("Found only one output file; retain or replace the pair")
    if st_file.exists():
        st = load_st_snapshots(st_file)
        cov = load_st_coverage(cov_file)
        actual = st.groupby("trade_date").size()
        for row in cov.loc[cov.complete].itertuples(index=False):
            if int(row.st_count) != int(actual.get(row.trade_date, 0)):
                raise ValueError("Existing output files have mismatched ST counts")
        completed = set(cov.loc[cov.complete, "trade_date"])
    else:
        st = pd.DataFrame(columns=["stock_code", "trade_date", "stock_name", "risk_type"])
        cov = pd.DataFrame(columns=["trade_date", "complete", "st_count", "source", "fetched_at"])
        completed = set()

    pro = ts.pro_api(token)
    fetched, failed, pending = 0, [], 0

    def checkpoint():
        nonlocal pending
        _save(st.sort_values(["trade_date", "stock_code"]), st_file)
        _save(cov.sort_values("trade_date"), cov_file)
        pending = 0

    for ordinal, day in enumerate(dates, 1):
        if day in completed:
            continue
        stamp = day.strftime("%Y%m%d")
        normalized = None
        last_error = ""
        for attempt in range(retries):
            try:
                frame = pro.stock_st(trade_date=stamp)
                if not isinstance(frame, pd.DataFrame):
                    raise ValueError("ST API did not return a DataFrame")
                if not {"ts_code", "trade_date"}.issubset(frame.columns):
                    raise ValueError("ST API omitted ts_code/trade_date")
                if len(frame) >= 1000:
                    raise ValueError("Potential 1000-row ST API truncation")
                if frame.empty:
                    raise ValueError("Empty full-market ST list; cannot certify without manual audit")
                if not frame.empty and not (_dates(frame.trade_date) == day).all():
                    raise ValueError("ST API returned an unexpected trade_date")
                normalized = pd.DataFrame({
                    "stock_code": _codes(frame.ts_code),
                    "trade_date": day,
                    "stock_name": (frame["name"].astype("string")
                                   if "name" in frame else ""),
                    "risk_type": (frame["type"].astype("string")
                                  if "type" in frame else "ST"),
                })
                if normalized.duplicated(["stock_code", "trade_date"]).any():
                    raise ValueError("Duplicate stock/day in ST API")
                break
            except Exception as exc:
                # Avoid accidentally logging an API credential echoed by
                # an upstream exception message.
                last_error = str(exc).replace(token, "[REDACTED]")
                if attempt + 1 < retries:
                    time.sleep(max(1, pause_seconds) * (attempt + 1))
        st = st.loc[st.trade_date != day]
        cov = cov.loc[cov.trade_date != day]
        fetched_at = datetime.now().astimezone().isoformat()
        if normalized is None:
            failed.append(stamp)
            rec = {"trade_date": day, "complete": False, "st_count": 0,
                   "source": "tushare.stock_st", "fetched_at": fetched_at}
            print(f"[{ordinal}/{len(dates)}] {stamp} 不完整：{last_error}", flush=True)
        else:
            st = pd.concat([st, normalized], ignore_index=True)
            rec = {"trade_date": day, "complete": True, "st_count": len(normalized),
                   "source": "tushare.stock_st", "fetched_at": fetched_at}
            fetched += 1
            print(f"[{ordinal}/{len(dates)}] {stamp} ST名单 {len(normalized)} 只", flush=True)
        cov = pd.concat([cov, pd.DataFrame([rec])], ignore_index=True)
        pending += 1
        if pending >= 10:
            checkpoint()
        time.sleep(pause_seconds)
    if pending or not st_file.exists():
        checkpoint()
    return {"market_days": len(dates), "newly_completed": fetched,
            "failed_days": len(failed), "failed_sample": failed[:20],
            "st_file": str(st_file), "coverage_file": str(cov_file)}


def main() -> None:
    p = argparse.ArgumentParser(description="Export dated Tushare ST lists (read-only)")
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 3, 1))
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--out", type=Path, default=Path("reports/ashare_st_history"))
    p.add_argument("--pause-seconds", type=float, default=0.5)
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args()
    result = export_tushare_st(args.start, args.end, args.out,
                               pause_seconds=args.pause_seconds, retries=args.retries)
    print(result, flush=True)
    if result["failed_days"]:
        raise SystemExit("ST data incomplete; review failed days then rerun")


if __name__ == "__main__":
    main()
