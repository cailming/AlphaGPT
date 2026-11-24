"""Freeze pre-holdout OHLCV and collect adjustment dates for a bounded causal shortlist."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from .daily import AShareDailyLoader
from . import p4_action_pipeline as ledger
from .p4_mvp import PoolConfig, build_pool


VERSION = "P4_SUBSET_INPUTS_V1"
PRICE_BEGIN = "2025-01-02"
SIGNAL_BEGIN = "2025-03-03"
VAL_START = "2026-01-05"
TEST_START = "2026-03-23"
PRICE_END = "2026-06-11"
ORIGINAL_TEST_EXCLUDED = "2026-06-12"
FACTOR_FIELDS = ("code", "dividOperateDate", "foreAdjustFactor", "backAdjustFactor", "adjustFactor")
RISK_FILE = Path("ashare/data/risk_events_verified_examples.csv")
REFERENCE = Path("reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json")


def hash_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parquet_write(frame, path):
    import polars as pl

    columns = {name: frame[name].tolist() for name in frame}
    table = pl.DataFrame(columns)
    table.write_parquet(path, compression="zstd")


def parquet_read(path):
    import polars as pl

    return pd.DataFrame(pl.read_parquet(path).to_dict(as_series=False))


def prepare(output, cfg, limit):
    if not 1 <= limit <= 1200:
        raise ValueError("Adjustment audit outreach is bounded at 1,200 securities")
    source = ledger.read_json(REFERENCE)
    if source["test_start"] != ORIGINAL_TEST_EXCLUDED:
        raise ValueError("Old test quarantine does not match reference metadata")
    stages = [PRICE_BEGIN, SIGNAL_BEGIN, VAL_START, TEST_START, PRICE_END, ORIGINAL_TEST_EXCLUDED]
    if sorted(set(stages)) != stages:
        raise ValueError("Dates must be strictly increasing")
    output.mkdir(parents=True)
    query = text("""
        SELECT stock_code, trade_date, open, high, low, close, volume, amount, pre_close, not_billing
          FROM public.market
         WHERE not_billing = 1 AND trade_date BETWEEN :start AND :finish AND stock_code IS NOT NULL
    """)
    with AShareDailyLoader().engine.connect() as connection:
        if connection.scalar(text("show transaction_read_only")) != "on":
            raise ValueError("Open a read-only database transaction using PGOPTIONS")
        batches = pd.read_sql_query(query, connection.execution_options(stream_results=True),
                                    params={"start": PRICE_BEGIN, "finish": PRICE_END}, chunksize=75000)
        raw = pd.concat(batches, ignore_index=True)
    raw["trade_date"] = pd.to_datetime(raw.trade_date).dt.strftime("%Y-%m-%d")
    raw = raw.sort_values(["stock_code", "trade_date"], kind="stable").reset_index(drop=True)
    if raw.empty or raw.trade_date.min() < PRICE_BEGIN or raw.trade_date.max() != PRICE_END:
        raise ValueError("Missing or out-of-bounds price snapshot")
    pool = build_pool(raw, date.fromisoformat(SIGNAL_BEGIN), date.fromisoformat(PRICE_END), cfg,
                      events=pd.read_csv(RISK_FILE, dtype={"stock_code": "string"}))
    features = pool.data.drop(columns=["target_return", "label_end_session"]).copy()
    features["trade_date"] = features.trade_date.dt.strftime("%Y-%m-%d")
    first = features.sort_values(["trade_date", "liquidity_rank", "stock_code"], kind="stable").drop_duplicates("stock_code")
    shortlisted = [{"stock_code": row.stock_code, "provider_code": provider, "admitted_on": row.trade_date,
                    "first_liquidity_rank": int(row.liquidity_rank)} for row in first.itertuples(index=False)
                   if (provider := ledger.bao._provider_code(row.stock_code))][:limit]
    if not shortlisted:
        raise ValueError("No causal adjustment-audit shortlist")
    parquet_write(raw, output / "formal_ohlcv.parquet")
    parquet_write(features, output / "causal_features.parquet")
    ledger.atomic_json(output / "factor_plan.json", shortlisted)
    record = {"version": VERSION, "price_begin": PRICE_BEGIN, "signal_begin": SIGNAL_BEGIN, "val_start": VAL_START,
              "test_start": TEST_START, "price_end": PRICE_END, "original_test_excluded": ORIGINAL_TEST_EXCLUDED,
              "calendar": pool.calendar.strftime("%Y-%m-%d").tolist(), "pool_config": asdict(cfg), "pool": pool.stats,
              "db_source": "public.market not_billing=1, read-only", "shortlist_candidates": len(shortlisted),
              "risk_event_source_sha256": hash_file(RISK_FILE), "old_test_metadata_sha256": hash_file(REFERENCE),
              "files": {name: hash_file(output / name) for name in
                        ("formal_ohlcv.parquet", "causal_features.parquet", "factor_plan.json")}}
    ledger.atomic_json(output / "inputs.json", record)
    print(f"Frozen {len(raw):,} bars, {len(features):,} causal candidates and {len(shortlisted):,} factor jobs", flush=True)


def validate_factor_rows(rows, provider_code, begin, end):
    dates = set()
    for row in rows:
        if len(row) != len(FACTOR_FIELDS) or row[0] != provider_code:
            raise ValueError("Factor row identity or field count mismatch")
        day = date.fromisoformat(row[1]).isoformat()
        if not begin <= day <= end or day in dates:
            raise ValueError("Duplicate or out-of-window factor date")
        dates.add(day)
        for field in row[2:]:
            number = Decimal(field)
            if not number.is_finite() or number <= 0:
                raise ValueError("Invalid factor value")
    return sorted(dates)


def factor_query(bs, provider_code, begin, end):
    with ledger.deadline(20):
        response = bs.query_adjust_factor(provider_code, start_date=begin, end_date=end)
        if str(response.error_code) != "0" or tuple(response.fields) != FACTOR_FIELDS:
            raise ValueError("Unexpected BaoStock response status or field schema")
        rows = []
        while response.next():
            rows.append(response.get_row_data())
            if len(rows) >= 1024:
                raise ValueError("Factor response may be truncated")
        if str(response.error_code) != "0":
            raise ValueError("Factor pagination did not finish")
    validate_factor_rows(rows, provider_code, begin, end)
    return rows


def read_verified_factor(path, job, manifest, plan_hash):
    signed = ledger.read_json(path)
    item = signed["factor_observation"]
    if hashlib.sha256(ledger.canonical(item)).hexdigest() != signed["sha256"]:
        raise ValueError(f"Factor evidence digest mismatch: {path}")
    if (item["scope"] != job or item["plan_sha256"] != plan_hash
            or item["start"] != manifest["price_begin"] or item["end"] != manifest["price_end"]):
        raise ValueError(f"Factor evidence does not match the bound request: {path}")
    if item["status"] == "SUCCESS":
        validate_factor_rows(item["rows"], job["provider_code"], item["start"], item["end"])
    return item


def capture(base):
    import baostock as bs

    manifest = ledger.read_json(base / "inputs.json")
    plan_hash = hash_file(base / "factor_plan.json")
    if plan_hash != manifest["files"]["factor_plan.json"]:
        raise ValueError("Factor plan changed after freezing")
    jobs = ledger.read_json(base / "factor_plan.json")
    folder = base / "factors"
    folder.mkdir(exist_ok=True)
    begin, end = manifest["price_begin"], manifest["price_end"]
    total = len(jobs)
    if total > 1200 or not jobs:
        raise ValueError("Unexpected factor scope")
    with ledger.exclusive(base):
        with ledger.deadline(20):
            if str(bs.login().error_code) != "0":
                raise ValueError("BaoStock factor login unavailable")
        consecutive_errors = 0
        try:
            for index, job in enumerate(jobs, 1):
                path = folder / (job["stock_code"] + ".json")
                if path.exists():
                    read_verified_factor(path, job, manifest, plan_hash)
                    continue
                started = time.monotonic()
                result = {"scope": job, "start": begin, "end": end, "plan_sha256": plan_hash,
                          "checked_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
                try:
                    result.update(status="SUCCESS", rows=factor_query(bs, job["provider_code"], begin, end))
                    consecutive_errors = 0
                except (ledger.RequestTimeout, InvalidOperation, ValueError, OSError) as error:
                    result.update(status="SOURCE_ERROR", rows=[], reason=type(error).__name__)
                    consecutive_errors += 1
                ledger.atomic_json(path, {"factor_observation": result,
                                          "sha256": hashlib.sha256(ledger.canonical(result)).hexdigest()})
                if index % 25 == 0 or index == total or result["status"] != "SUCCESS":
                    print(f"{index}/{total} {job['stock_code']} {result['status']}", flush=True)
                if consecutive_errors >= 3:
                    raise ValueError("Three independent factor source failures; preserve checkpoint and pause")
                time.sleep(max(0, 1.1 - (time.monotonic() - started)))
        finally:
            try:
                with ledger.deadline(5):
                    bs.logout()
            except ledger.RequestTimeout:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "capture"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--factor-limit", type=int, default=1200)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out, PoolConfig(), args.factor_limit)
    else:
        capture(args.out)


if __name__ == "__main__":
    main()
