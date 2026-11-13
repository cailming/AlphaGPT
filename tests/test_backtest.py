"""P3.1 A-share baseline: pure synthetic bars, no network or real orders."""
import unittest
from datetime import date

import pandas as pd
from sqlalchemy import create_engine

from ashare.backtest import (BacktestConfig, load_formal_market,
                             make_ma20_reversion_signals, run_backtest)
from ashare.daily import AShareDailyLoader


def bars(rows):
    return pd.DataFrame([
        dict(stock_code=code, trade_date=day, open=op, close=cl,
             high=max(op, cl), low=min(op, cl),
             volume=10000, amount=100000,
             ma20=10.0, pre_close=10.0, not_billing=1)
        for code, day, op, cl in rows
    ])


def signals(*rows):
    return pd.DataFrame(rows, columns=["stock_code", "signal_date", "score"])


def config(*, allow=True):
    return BacktestConfig(date(2025, 3, 3), date(2025, 3, 7),
                          initial_cash=100000, max_positions=1,
                          allow_unverified_tradeability=allow)


class BacktestTests(unittest.TestCase):
    def test_buy_next_open_and_sell_following_open(self):
        market = bars([
            ("000001", "2025-03-03", 10, 10),
            ("000001", "2025-03-04", 10, 20),
            ("000001", "2025-03-05", 11, 11),
            ("000001", "2025-03-06", 11, 11),
        ])
        result = run_backtest(market,
                  signals(("000001", "2025-03-03", 1.0)), config())
        self.assertEqual(result.trades.side.tolist(), ["buy", "sell"])
        self.assertEqual(result.trades.date.tolist(), ["2025-03-04", "2025-03-05"])
        self.assertGreater(float(result.trades.iloc[1].fees),
                           float(result.trades.iloc[0].fees))
        self.assertEqual(result.metrics["unclosed_positions"], 0)
        self.assertIn("UNVERIFIED", result.metrics["tradeability_mode"])

    def test_same_day_close_cannot_affect_open_quantity(self):
        base = [
            ("000001", "2025-03-03", 10, 10),
            ("000001", "2025-03-04", 10, 10),
            ("000001", "2025-03-05", 10, 10),
        ]
        a = run_backtest(bars(base), signals(("000001", "2025-03-03", 1)),
                         config()).trades
        changed = list(base)
        changed[1] = ("000001", "2025-03-04", 10, 100)
        b = run_backtest(bars(changed), signals(("000001", "2025-03-03", 1)),
                         config()).trades
        self.assertEqual(int(a.iloc[0].shares), int(b.iloc[0].shares))

    def test_suspended_sell_waits_for_next_observed_session(self):
        market = bars([
            ("000001", "2025-03-03", 10, 10),
            ("000001", "2025-03-04", 10, 10),
            ("600000", "2025-03-05", 20, 20),
            ("000001", "2025-03-06", 11, 11),
            ("600000", "2025-03-06", 20, 20),
        ])
        out = run_backtest(market,
                signals(("000001", "2025-03-03", 1)), config())
        self.assertEqual(out.trades.date.tolist(), ["2025-03-04", "2025-03-06"])
        row = out.equity.set_index("trade_date").loc["2025-03-05"]
        self.assertEqual(int(row.stale_positions), 1)
        self.assertTrue(any(out.rejected.reason == "NO_BAR_OR_UNVERIFIED_SELL"))

    def test_unknown_status_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "tradeability"):
            run_backtest(bars([("000001", "2025-03-03", 10, 10)]),
                         signals(("000001", "2025-03-03", 1)), config(allow=False))

    def test_tradeability_blocks_unverified_and_missing(self):
        m = bars([
            ("000001", "2025-03-03", 10, 10),
            ("000001", "2025-03-04", 10, 10),
            ("000001", "2025-03-05", 10, 10),
        ])
        t = pd.DataFrame([
            ("000001", "2025-03-04", False, True),
            ("000001", "2025-03-05", True, True),
        ], columns=["stock_code", "trade_date", "can_buy", "can_sell"])
        o = run_backtest(m, signals(("000001", "2025-03-03", 1)),
                         config(allow=False), tradeability=t)
        self.assertEqual(len(o.trades), 0)
        self.assertEqual(len(o.rejected), 1)

    def test_ma20_ranking_uses_close_only(self):
        m = bars([
            ("000001", "2025-03-03", 10, 9),
            ("600000", "2025-03-03", 10, 11),
            ("000001", "2025-03-04", 10, 9),
        ])
        result = make_ma20_reversion_signals(m)
        idx = result.set_index(["stock_code", "signal_date"])
        self.assertAlmostEqual(float(idx.loc[("000001", pd.Timestamp("2025-03-03"))].score), .1)
        self.assertAlmostEqual(float(idx.loc[("600000", pd.Timestamp("2025-03-03"))].score), -.1)

    def test_db_loader_filters_temporary(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        with engine.begin() as conn:
            conn.exec_driver_sql("ATTACH DATABASE ':memory:' AS public")
            conn.exec_driver_sql('''CREATE TABLE public.market (
                stock_code TEXT, trade_date DATE, open REAL, high REAL,
                low REAL, close REAL, volume REAL, amount REAL,
                pre_close REAL, "MA_20" REAL, not_billing INTEGER
            )''')
            conn.exec_driver_sql('''INSERT INTO public.market VALUES (
                '000001','2025-03-03',10,11,9,10,100,1000,10,10,1
            )''')
            conn.exec_driver_sql('''INSERT INTO public.market VALUES (
                '000002','2025-03-03',10,11,9,10,100,1000,10,10,2
            )''')
        result = load_formal_market(date(2025, 3, 3), date(2025, 3, 3),
                                   loader=AShareDailyLoader(engine), chunksize=1)
        self.assertEqual(result.stock_code.tolist(), ["000001"])


if __name__ == "__main__":
    unittest.main()
