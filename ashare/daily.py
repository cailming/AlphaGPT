"""Point-in-time daily A-share research data from market/main-force tables.

Read-only DB adapter for public_1111.sql (2026-11-11). All factors for date T
use T or earlier data; forward open-to-open returns are LABELS ONLY.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.engine import Engine, URL


DAILY_COLUMNS = (
    "ret_1d", "ret_5d", "vol_ratio_20", "ma20_dev", "breakout_20",
    "main_force", "activity", "strength", "holding", "distribution_risk",
    "amount_ratio_5_20", "drawdown_20",
)
LABEL_COLUMN = "target_next_open_1d"


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for name in columns:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)


def make_daily_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Make causal features and separately named future label on sorted daily rows.

    Inputs must contain one row per (stock_code, trade_date). Do not train on the
    target_* column as a feature. Input prices must be consistently adjusted;
    no adjustment is performed here. The precomputed MA20 and main-force values
    are usable only if they existed at the signal's actual decision time.
    """
    required = {"stock_code", "trade_date", "open", "close", "high", "volume", "ma20"}
    absent = required - set(raw.columns)
    if absent:
        raise ValueError(f"Missing daily columns: {sorted(absent)}")
    df = raw.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="raise").dt.normalize()
    if df[["stock_code", "trade_date"]].isna().any().any():
        raise ValueError("Null stock_code/trade_date is not allowed")
    df = df.sort_values(["stock_code", "trade_date"], kind="stable").reset_index(drop=True)
    if df.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate stock/date in daily market data")
    financial = ["open", "close", "high", "volume", "amount", "ma20",
                 "main_force_score", "activity_score", "strength_score",
                 "holding_score", "distribution_risk", "amount_ratio_5_20",
                 "drawdown_20_pct"]
    _numeric(df, financial)
    for name in financial:
        if name not in df:
            df[name] = np.nan

    grouped = df.groupby("stock_code", sort=False)
    prev_close = grouped["close"].shift(1)
    five_ago = grouped["close"].shift(5)
    previous_volume_mean = grouped["volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean()
    )
    previous_high = grouped["high"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).max()
    )
    df["ret_1d"] = _safe_ratio(df["close"], prev_close) - 1
    df["ret_5d"] = _safe_ratio(df["close"], five_ago) - 1
    df["vol_ratio_20"] = _safe_ratio(df["volume"], previous_volume_mean)
    df["ma20_dev"] = _safe_ratio(df["close"], df["ma20"]) - 1
    df["breakout_20"] = _safe_ratio(df["close"], previous_high) - 1
    df["main_force"] = df["main_force_score"] / 100.0
    df["activity"] = df["activity_score"] / 25.0
    df["strength"] = df["strength_score"] / 20.0
    df["holding"] = df["holding_score"] / 20.0
    df["distribution_risk"] = df["distribution_risk"] / 100.0
    df["drawdown_20"] = df["drawdown_20_pct"] / 100.0  # 0 or negative
    # T signal -> T+1 open entry, T+2 open exit. Label is not a tradable fill.
    next_open = grouped["open"].shift(-1)
    next2_open = grouped["open"].shift(-2)
    df[LABEL_COLUMN] = _safe_ratio(next2_open, next_open) - 1
    return df


def _database_url() -> URL | str:
    explicit = os.getenv("ASHARE_DB_URL")
    if explicit:
        return explicit
    if not os.getenv("DB_PASSWORD"):
        raise ValueError("Set ASHARE_DB_URL or DB_PASSWORD and other DB_* variables")
    return URL.create(
        "postgresql+psycopg2", username=os.getenv("DB_USER", "postgres"),
        password=os.environ["DB_PASSWORD"], host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")), database=os.getenv("DB_NAME", "agu"),
    )


class AShareDailyLoader:
    """Query just market + stock_main_force_daily; no schema writes."""

    def __init__(self, engine: Engine | None = None):
        self.engine = engine if engine is not None else create_engine(_database_url())

    def load(self, start: date, end: date, stock_codes: list[str] | None = None,
             warmup_calendar_days: int = 150,
             label_lookahead_calendar_days: int = 30,
             include_warmup: bool = False) -> pd.DataFrame:
        if end < start or warmup_calendar_days < 0 or label_lookahead_calendar_days < 0:
            raise ValueError("Invalid date range / warmup days")
        if stock_codes is not None and not stock_codes:
            raise ValueError("stock_codes cannot be empty")
        # Prehistory before requested start is needed for past-only rolling windows.
        sql = """
            SELECT m.stock_code, m.trade_date, m.open, m.high, m.low,
                   m.close, m.volume, m.amount, m."MA_20" AS ma20,
                   f.main_force_score, f.activity_score, f.strength_score,
                   f.holding_score, f.distribution_risk,
                   f.amount_ratio_5_20, f.drawdown_20_pct,
                   f.created_at AS main_force_created_at
            FROM public.market AS m
            LEFT JOIN public.stock_main_force_daily AS f
              ON f.stock_code = m.stock_code AND f.trade_date = m.trade_date
            WHERE m.trade_date BETWEEN :fetch_start AND :fetch_end
              AND m.stock_code IS NOT NULL
        """
        params: dict = {"fetch_start": start - timedelta(days=warmup_calendar_days),
                        "fetch_end": end + timedelta(days=label_lookahead_calendar_days)}
        query = text(sql)
        if stock_codes is not None:
            query = text(sql + " AND m.stock_code IN :stock_codes")
            query = query.bindparams(bindparam("stock_codes", expanding=True))
            params["stock_codes"] = stock_codes
        with self.engine.connect() as connection:
            raw = pd.read_sql_query(query, connection, params=params)
        if raw.empty:
            return raw
        result = make_daily_features(raw)
        if include_warmup:
            return result.loc[result["trade_date"] <= pd.Timestamp(end)].reset_index(drop=True)
        return result.loc[result["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))].reset_index(drop=True)
