"""Convert cumulative intraday snapshots into observed one-minute intervals.

Important: ticktime is last trade time, NOT confirmed collection timestamp.
Only consecutive observed minute buckets receive volume/amount deltas. No
OHLC minute bar is claimed; open/high/low are day-to-date values.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine

from .daily import _database_url, _numeric


SNAPSHOT_COLUMNS = ("stock_code", "trade_date", "ticktime", "trade", "volume", "amount")


def build_minute_intervals(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute interval deltas, retaining invalid rows with an explicit quality flag.

    Same-minute observations keep the latest ticktime, then prefer billing
    over billing_bak for an exact tie. Missing minutes are not interpolated.
    Deltas are attributed to the later observed minute (end timestamp).
    """
    missing = set(SNAPSHOT_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing snapshot columns: {sorted(missing)}")
    df = raw.copy()
    if df.empty:
        return df.assign(minute_volume=pd.Series(dtype=float),
                         minute_amount=pd.Series(dtype=float),
                         ret_1m=pd.Series(dtype=float),
                         quality_flag=pd.Series(dtype=str))
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="raise").dt.normalize()
    df["ticktime_td"] = pd.to_timedelta(df["ticktime"].astype(str), errors="coerce")
    if df[["stock_code", "trade_date", "ticktime_td"]].isna().any().any():
        raise ValueError("Missing/invalid stock, date, or ticktime")
    _numeric(df, ["trade", "volume", "amount", "high", "low"])
    df["minute"] = df["trade_date"] + df["ticktime_td"].dt.floor("min")
    # Never treat pre-open auctions or post-close quotes as regular minutes.
    local_time = df["minute"] - df["trade_date"]
    regular = ((local_time >= pd.Timedelta(hours=9, minutes=30)) &
               (local_time <= pd.Timedelta(hours=11, minutes=30))) | (
               (local_time >= pd.Timedelta(hours=13)) &
               (local_time <= pd.Timedelta(hours=15)))
    df = df.loc[regular].copy()
    if df.empty:
        return df.assign(minute_volume=pd.Series(dtype=float),
                         minute_amount=pd.Series(dtype=float),
                         ret_1m=pd.Series(dtype=float),
                         quality_flag=pd.Series(dtype=str))
    if "source" not in df:
        df["source"] = "archive"
    df["source_priority"] = df["source"].map({"archive": 0, "today": 1}).fillna(0)
    if "id" not in df:
        df["id"] = 0
    df = df.sort_values(["stock_code", "trade_date", "minute", "ticktime_td",
                         "source_priority", "id"], kind="stable")
    df = df.drop_duplicates(["stock_code", "trade_date", "minute"], keep="last")
    group = df.groupby(["stock_code", "trade_date"], sort=False)
    previous_minute = group["minute"].shift(1)
    prev_volume = group["volume"].shift(1)
    prev_amount = group["amount"].shift(1)
    prev_price = group["trade"].shift(1)
    df["gap_minutes"] = (df["minute"] - previous_minute).dt.total_seconds() / 60
    delta_volume = df["volume"] - prev_volume
    delta_amount = df["amount"] - prev_amount
    first = previous_minute.isna()
    rollback = (delta_volume < 0) | (delta_amount < 0)
    consecutive = df["gap_minutes"].eq(1)
    valid = consecutive & ~rollback & delta_volume.notna() & delta_amount.notna()
    df["quality_flag"] = np.select(
        [first, rollback, ~consecutive],
        ["FIRST_OBSERVATION", "CUMULATIVE_DECREASE", "NON_CONSECUTIVE"],
        default="OK",
    )
    df["minute_volume"] = delta_volume.where(valid)
    df["minute_amount"] = delta_amount.where(valid)
    df["ret_1m"] = (df["trade"].div(prev_price.where(prev_price > 0)) - 1).where(valid)
    # Rolling metrics only become available after 5 contiguous valid intervals.
    group2 = df.groupby(["stock_code", "trade_date"], sort=False)
    df["volume_5m"] = group2["minute_volume"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
    df["amount_5m"] = group2["minute_amount"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
    previous_interval_volume = group2["minute_volume"].shift(1)
    df["volume_accel"] = df["minute_volume"] - previous_interval_volume
    if {"high", "low"}.issubset(df.columns):
        spread = df["high"] - df["low"]
        df["day_price_position"] = ((df["trade"] - df["low"]) /
                                    spread.where(spread > 0))
    return df.drop(columns=["ticktime_td", "source_priority"]).reset_index(drop=True)


class AShareMinuteLoader:
    """Read archives and optionally today's table without modifying the DB."""

    def __init__(self, engine: Engine | None = None):
        self.engine = engine if engine is not None else create_engine(_database_url())

    def load(self, start: date, end: date, stock_codes: list[str] | None = None,
             include_today: bool = False, max_calendar_days: int = 10) -> pd.DataFrame:
        if end < start or (end - start).days > max_calendar_days:
            raise ValueError("Invalid or excessively large minute date range")
        if stock_codes is not None and not stock_codes:
            raise ValueError("stock_codes cannot be empty")
        branch_sql = """
            SELECT id, stock_code, trade_date, ticktime, trade, volume,
                   amount, high, low, '{source}' AS source
            FROM public.{table}
            WHERE trade_date BETWEEN :start_date AND :end_date
        """
        archive = branch_sql.format(table="billing_bak", source="archive")
        today = branch_sql.format(table="billing", source="today")
        query = archive + (" UNION ALL " + today if include_today else "")
        params: dict = {"start_date": start, "end_date": end}
        if stock_codes is not None:
            query = "SELECT * FROM (" + query + ") AS snapshots WHERE stock_code IN :stock_codes"
            params["stock_codes"] = stock_codes
        stmt = text(query)
        if stock_codes is not None:
            stmt = stmt.bindparams(bindparam("stock_codes", expanding=True))
        with self.engine.connect() as conn:
            raw = pd.read_sql_query(stmt, conn, params=params)
        return build_minute_intervals(raw)
