"""P3.2 diagnostic A-share T-close signal -> T+1 open entry -> later open exit.

NO live trading. NO historical corporate actions/ST/limit prices inferred.
A run without verified per-stock/day tradability requires explicit opt-in and
is labelled exploratory; results are not audited total returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable
import math

import numpy as np
import pandas as pd
from sqlalchemy import text, bindparam

from .daily import AShareDailyLoader


@dataclass(frozen=True)
class BacktestConfig:
    start: date
    end: date
    initial_cash: float = 1_000_000.0
    max_positions: int = 10
    lot_size: int = 100
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_rate_sell: float = 0.0005
    transfer_rate: float = 0.00001
    slippage_bps: float = 5.0
    allow_unverified_tradeability: bool = False
    candidate_pool_size: int = 20    # preselected at T close, not full-market order attempts

    def __post_init__(self):
        if self.end < self.start or self.initial_cash <= 0:
            raise ValueError("Invalid backtest dates or cash")
        if self.max_positions < 1 or self.lot_size < 1:
            raise ValueError("Invalid position/lot settings")
        if self.candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be positive")
        if any(x < 0 for x in (self.commission_rate, self.min_commission,
                              self.stamp_rate_sell, self.transfer_rate,
                              self.slippage_bps)):
            raise ValueError("Costs and slippage must be nonnegative")


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    rejected: pd.DataFrame
    metrics: dict
    open_positions: pd.DataFrame


def load_formal_market(start: date, end: date, *, stock_codes=None,
                       warmup_calendar_days: int = 100,
                       chunksize: int = 50000,
                       progress: Callable[[str], None] | None = None,
                       loader: AShareDailyLoader | None = None) -> pd.DataFrame:
    """SELECT only not_billing=1, with prehistory for factor warmup.

    Price OHLC is unadjusted. T+1 execution modelling cannot assume dividends
    are part of the price return. Existing DB credentials are reused.
    """
    from datetime import timedelta
    if end < start or chunksize < 1 or warmup_calendar_days < 0:
        raise ValueError("Invalid load configuration")
    engine = (loader or AShareDailyLoader()).engine
    sql = """
        SELECT stock_code, trade_date, open, high, low, close,
               volume, amount, pre_close, "MA_20" AS ma20,
               not_billing
        FROM public.market
        WHERE trade_date BETWEEN :from_date AND :end_date
          AND not_billing = 1
          AND stock_code IS NOT NULL
    """
    params = {"from_date": start - timedelta(days=warmup_calendar_days),
              "end_date": end}
    query = text(sql)
    if stock_codes is not None:
        if not stock_codes:
            raise ValueError("stock_codes cannot be empty")
        query = text(sql + " AND stock_code IN :codes").bindparams(
            bindparam("codes", expanding=True))
        params["codes"] = stock_codes
    parts = []
    rows = 0
    if progress:
        progress("开始 SELECT 正式日线（not_billing=1）")
    with engine.connect() as conn:
        streaming = conn.execution_options(stream_results=True)
        for part in pd.read_sql_query(query, streaming, params=params,
                                      chunksize=chunksize):
            parts.append(part)
            rows += len(part)
            if progress:
                progress(f"正式日线读取：{rows:,} 行")
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if progress:
        progress(f"正式日线查询完成：{len(out):,} 行")
    return out


def make_ma20_reversion_signals(market: pd.DataFrame) -> pd.DataFrame:
    """Reproduce NEG(ma20_dev) as an after-close T factor.

    No future columns; signals on days without valid formal bars are omitted.
    """
    needed = {"stock_code", "trade_date", "close", "ma20", "not_billing"}
    if not needed.issubset(market.columns):
        raise ValueError(f"Missing signal fields: {sorted(needed-set(market.columns))}")
    df = market.loc[market.not_billing == 1,
                    ["stock_code", "trade_date", "close", "ma20"]].copy()
    df["trade_date"] = pd.to_datetime(df.trade_date).dt.normalize()
    close = pd.to_numeric(df.close, errors="coerce")
    ma20 = pd.to_numeric(df.ma20, errors="coerce")
    good = np.isfinite(close) & np.isfinite(ma20) & (close > 0) & (ma20 > 0)
    df = df.loc[good, ["stock_code", "trade_date"]].copy()
    df["score"] = (1.0 - (close.loc[good] / ma20.loc[good])).to_numpy()
    return df.rename(columns={"trade_date": "signal_date"})


def _money_cost(value: float, config: BacktestConfig, side: str) -> float:
    commission = max(config.min_commission, value * config.commission_rate)
    transfer = value * config.transfer_rate
    stamp = value * config.stamp_rate_sell if side == "sell" else 0
    return commission + transfer + stamp


def run_backtest(market: pd.DataFrame, signals: pd.DataFrame,
                 config: BacktestConfig,
                 *, tradeability: pd.DataFrame | None = None,
                 progress: Callable[[str], None] | None = None,
                 progress_every: int = 20) -> BacktestResult:
    """Daily-event simulation; exit attempts start on the next session.

    A missing bar means no fill and last valid close is used as a STALE mark.
    Historical limits/ST/corporate actions are not guessed. If tradeability
    isn't provided, an explicit exploratory opt-in is required.
    """
    required = {"stock_code", "trade_date", "open", "close", "volume", "not_billing"}
    if missing := required - set(market.columns):
        raise ValueError(f"Missing market columns: {sorted(missing)}")
    if tradeability is None and not config.allow_unverified_tradeability:
        raise ValueError("Verified historical tradeability required; pass explicit exploratory opt-in")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    sig_cols = {"stock_code", "signal_date", "score"}
    if missing := sig_cols - set(signals.columns):
        raise ValueError(f"Missing signal columns: {sorted(missing)}")
    bars = market.loc[market.not_billing == 1].copy()
    bars["trade_date"] = pd.to_datetime(bars.trade_date).dt.normalize()
    bars["stock_code"] = bars.stock_code.astype(str)
    if bars.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate formal market stock/day")
    signals = signals.copy()
    signals["signal_date"] = pd.to_datetime(signals.signal_date).dt.normalize()
    signals["stock_code"] = signals.stock_code.astype(str)
    if signals.duplicated(["stock_code", "signal_date"]).any():
        raise ValueError("Duplicate signal stock/day")
    signals["score"] = pd.to_numeric(signals.score, errors="coerce")
    signals = signals.loc[np.isfinite(signals.score)]
    # Optional verified per-symbol/day conditions: only supplied rows may trade.
    eligible = None
    if tradeability is not None:
        keys = {"stock_code", "trade_date", "can_buy", "can_sell"}
        if missing := keys - set(tradeability.columns):
            raise ValueError(f"Missing tradeability columns: {sorted(missing)}")
        t = tradeability.copy()
        t["trade_date"] = pd.to_datetime(t.trade_date).dt.normalize()
        t["stock_code"] = t.stock_code.astype(str)
        if t.duplicated(["stock_code", "trade_date"]).any():
            raise ValueError("Duplicate tradeability stock/day")
        eligible = t.set_index(["trade_date", "stock_code"])

    bars = bars.loc[bars.trade_date.between(pd.Timestamp(config.start),
                                            pd.Timestamp(config.end))]
    if bars.empty:
        raise ValueError("No formal daily bars in backtest window")
    market_days = pd.DatetimeIndex(sorted(bars.trade_date.unique()))
    sig_by_day = {day: group.sort_values(["score", "stock_code"],
                                          ascending=[False, True])
                  for day, group in signals.groupby("signal_date")}
    cash = float(config.initial_cash)
    positions = {}  # stock_code -> {shares, opened, signal_date, cost_basis}
    last_close = {}
    last_mark_session = {}
    ledger, rejected, equity_rows = [], [], []
    peak_stale_sessions = 0
    total_stale_session_days = 0
    previous_day = None
    slip = config.slippage_bps / 10000.0
    unknown = tradeability is None

    def check_tradeable(day, code, side):
        if eligible is None:
            return True
        try:
            row = eligible.loc[(day, code)]
        except KeyError:
            return False
        value = row["can_buy"] if side == "buy" else row["can_sell"]
        # Reject missing, strings and other unverifiable status values.
        return isinstance(value, (bool, np.bool_)) and bool(value)

    for day_idx, (day, daybars) in enumerate(bars.groupby("trade_date", sort=True), start=1):
        current = daybars.set_index("stock_code")
        def open_price(code):
            if code not in current.index:
                return None
            row = current.loc[code]
            op = pd.to_numeric(row["open"], errors="coerce")
            if pd.isna(op) or not math.isfinite(float(op)) or float(op) <= 0:
                return None
            # Do not use same-day closing volume or high/low to decide
            # whether a T+1 opening order could have filled.
            return float(op)

        # Sell positions opened earlier than current trading session.
        for code in sorted(tuple(positions)):
            pos = positions[code]
            if day <= pos["opened"]:
                continue  # A-share T+1 sell restriction
            price = open_price(code)
            if price is None or not check_tradeable(day, code, "sell"):
                rejected.append(dict(date=day.date().isoformat(),
                                     stock_code=code, side="sell",
                                     reason="NO_BAR_OR_UNVERIFIED_SELL"))
                continue
            fill = price * (1 - slip)
            gross = fill * pos["shares"]
            fees = _money_cost(gross, config, "sell")
            cash += gross - fees
            ledger.append(dict(date=day.date().isoformat(),
                               stock_code=code, side="sell",
                               shares=pos["shares"], price=fill,
                               gross=gross, fees=fees, signal_date=pos["signal_date"]))
            del positions[code]

        # Today's signals cannot execute until NEXT session.
        prior = sig_by_day.get(previous_day)
        if prior is not None:
            # Fixed shortlist decided at T close. Do NOT sweep thousands of
            # stocks just because a limited portfolio has no buying power.
            shortlist = prior.head(config.candidate_pool_size)
            for rec in shortlist.itertuples(index=False):
                if len(positions) >= config.max_positions:
                    break
                code = rec.stock_code
                if code in positions:
                    continue
                price = open_price(code)
                if price is None or not check_tradeable(day, code, "buy"):
                    rejected.append(dict(date=day.date().isoformat(),
                                         stock_code=code, side="buy",
                                         reason="NO_BAR_OR_UNVERIFIED_BUY"))
                    continue
                fill = price * (1 + slip)
                # Allocate no more than 1/max_positions of current marked equity.
                equity_before = cash + sum(
                    pos["shares"] * last_close.get(sym, pos["entry_price"])
                    for sym, pos in positions.items()
                )
                budget = min(cash, equity_before / config.max_positions)
                shares = int(budget // (fill * config.lot_size)) * config.lot_size
                while shares >= config.lot_size:
                    value = fill * shares
                    if value + _money_cost(value, config, "buy") <= cash + 1e-8:
                        break
                    shares -= config.lot_size
                if shares < config.lot_size:
                    rejected.append(dict(date=day.date().isoformat(),
                                         stock_code=code, side="buy",
                                         reason="INSUFFICIENT_CASH"))
                    # One exhausted allocation ends today's buys; don't scan
                    # the remaining candidates for a cheaper artificial fill.
                    break
                value = fill * shares
                fees = _money_cost(value, config, "buy")
                cash -= value + fees
                positions[code] = dict(shares=shares, opened=day,
                                       entry_price=fill,
                                       signal_date=previous_day.date().isoformat())
                ledger.append(dict(date=day.date().isoformat(),
                                   stock_code=code, side="buy", shares=shares,
                                   price=fill, gross=value, fees=fees,
                                   signal_date=previous_day.date().isoformat()))

        # Update marks AFTER open fills. The closing price for D must never
        # affect the D-open share budget or whether an order gets sent.
        for code, record in current.iterrows():
            close = float(record["close"]) if pd.notna(record["close"]) else float("nan")
            if math.isfinite(close) and close > 0:
                last_close[code] = close
                last_mark_session[code] = day_idx

        stale = 0
        max_stale = 0
        stale_value = 0.0
        value = 0.0
        for code, pos in positions.items():
            mark = last_close.get(code, pos["entry_price"])
            position_value = pos["shares"] * mark
            value += position_value
            age = day_idx - last_mark_session.get(code, day_idx - 1)
            if age > 0:
                stale += 1
                stale_value += position_value
                max_stale = max(max_stale, age)
        peak_stale_sessions = max(peak_stale_sessions, max_stale)
        total_stale_session_days += int(stale > 0)
        equity_rows.append(dict(trade_date=day.date().isoformat(),
                                cash=round(cash, 2), holdings_value=round(value, 2),
                                equity=round(cash + value, 2),
                                positions=len(positions), stale_positions=stale,
                                stale_value=round(stale_value, 2),
                                max_stale_sessions=max_stale,
                                valuation_fresh=(stale == 0)))
        previous_day = day
        if progress is not None and (
            day_idx == 1 or day_idx % progress_every == 0 or day_idx == len(market_days)
        ):
            progress(f"回测交易日 {day_idx}/{len(market_days)}，"
                     f"持仓 {len(positions)} 只，过期估值 {stale} 只，"
                     f"模拟成交 {len(ledger)} 笔")

    nav = pd.DataFrame(equity_rows)
    nav["daily_return"] = nav.equity.pct_change().fillna(nav.equity / config.initial_cash - 1)
    nav["drawdown"] = nav.equity / nav.equity.cummax().clip(lower=config.initial_cash) - 1
    open_positions_rows = []
    for code, pos in sorted(positions.items()):
        recent_mark_idx = last_mark_session.get(code)
        stale_sessions = (len(market_days) - recent_mark_idx
                          if recent_mark_idx is not None else len(market_days))
        open_positions_rows.append({
            "stock_code": code, "shares": pos["shares"],
            "entry_date": pos["opened"].date().isoformat(),
            "entry_price": pos["entry_price"],
            "last_known_close": last_close.get(code),
            "last_mark_date": (market_days[recent_mark_idx - 1].date().isoformat()
                               if recent_mark_idx is not None else None),
            "stale_sessions": stale_sessions,
            "indicative_value": pos["shares"] * last_close.get(
                code, pos["entry_price"]),
        })
    last_trade_date = ledger[-1]["date"] if ledger else None
    fees_total = sum(float(rec["fees"]) for rec in ledger)
    rejection_counts = {}
    for item in rejected:
        key = item["side"] + ":" + item["reason"]
        rejection_counts[key] = rejection_counts.get(key, 0) + 1
    stale_days = int((nav["stale_positions"] > 0).sum())
    all_stale_days = int(((nav["stale_positions"] == nav["positions"])
                          & (nav["positions"] > 0)).sum())
    last_day_unfresh = bool(nav.iloc[-1]["stale_positions"] > 0)
    metrics = {
        "start": config.start.isoformat(), "end": config.end.isoformat(),
        "initial_cash": config.initial_cash,
        "ending_equity": float(nav.equity.iloc[-1]),
        "simple_return": float(nav.equity.iloc[-1] / config.initial_cash - 1),
        "max_drawdown": float(nav.drawdown.min()),
        "trade_count": len(ledger),
        "unclosed_positions": len(positions),
        "rejected_attempts": len(rejected),
        "candidate_pool_size": config.candidate_pool_size,
        "fees_assumed_total": round(fees_total, 2),
        "last_simulated_trade_date": last_trade_date,
        "days_with_stale_marks": stale_days,
        "days_all_positions_stale": all_stale_days,
        "max_stale_sessions": peak_stale_sessions,
        "end_stale_positions": int(nav.iloc[-1]["stale_positions"]),
        "end_stale_value": float(nav.iloc[-1]["stale_value"]),
        "end_stale_value_share": (
            float(nav.iloc[-1]["stale_value"] / nav.iloc[-1]["equity"])
            if float(nav.iloc[-1]["equity"]) > 0 else None
        ),
        "rejected_by_reason": rejection_counts,
        "last_day_valuation_fresh": not last_day_unfresh,
        "indicative_nav_only": True,
        "audited_total_return_available": False,
        "tradeability_mode": ("UNVERIFIED_EXPLORATORY" if unknown
                              else "EXPLICIT_HISTORICAL_TRADEABILITY"),
        "corporate_action_adjustment": "NOT_MODELED",
        "price_basis": "UNADJUSTED_OHLC",
        "warning": (
            "EXPLORATORY ONLY: daily OHLC does not prove opening auction fills. "
            "Historical ST, price limits, dividend/share adjustments and "
            "intraday liquidity are not fully modeled. This is NOT audited PnL."
        ),
    }
    return BacktestResult(nav, pd.DataFrame(ledger),
                          pd.DataFrame(rejected), metrics,
                          pd.DataFrame(open_positions_rows, columns=[
                              "stock_code", "shares", "entry_date",
                              "entry_price", "last_known_close", "last_mark_date",
                              "stale_sessions", "indicative_value",
                          ]))
