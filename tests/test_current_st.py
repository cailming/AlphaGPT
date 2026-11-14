"""P3.3-B: current-ST blacklist is explicitly NON-PIT."""
import unittest
from datetime import date

import pandas as pd
from sqlalchemy import create_engine

from ashare.current_st import (
    NON_PIT_LABEL, filter_signals_by_current_st, load_current_stock_info,
)
from ashare.backtest import BacktestConfig, run_backtest
def bars(rows):
    return pd.DataFrame([
        dict(stock_code=code, trade_date=day, open=op, close=cl,
             high=max(op, cl), low=min(op, cl),
             volume=10000, amount=100000, ma20=10,
             pre_close=10, not_billing=1)
        for code, day, op, cl in rows
    ])


def signals(*rows):
    return pd.DataFrame(rows, columns=["stock_code", "signal_date", "score"])


def info():
    return pd.DataFrame([
        ("000001", "平安银行", 0),
        ("600200", "*ST苏吴", 2),
        ("300344", "ST立方", 1),
        ("002898", "*ST赛隆", 0),  # Name wins over a stale 0 field.
        ("600000", "浦发银行", None),  # Missing st -> UNKNOWN.
        ("000638", None, 2),        # st=2 is sufficient.
    ], columns=["stock_code", "stock_name", "st"])


class CurrentSTTests(unittest.TestCase):
    def test_historical_signals_blocked_by_today_status_with_warning(self):
        raw = signals(
            ("000001", "2025-03-03", 0.1),
            ("600200", "2025-03-03", 0.9),
            ("300344", "2025-03-03", 0.8),
            ("002898", "2025-03-03", 0.7),
            ("600000", "2025-03-03", 0.6),
            ("123456", "2025-03-03", 0.5),
            ("000638", "2025-03-03", 0.4),
        )
        before = raw.copy(deep=True)
        result = filter_signals_by_current_st(
            raw, info(), snapshot_at_utc="2026-11-14T00:00:00+00:00"
        )
        self.assertEqual(result.signals.stock_code.tolist(), ["000001"])
        self.assertEqual(result.stats["signal_rows_before"], 7)
        self.assertEqual(result.stats["signal_rows_after"], 1)
        self.assertEqual(result.stats["blocked_current_st_signals"], 4)
        self.assertEqual(result.stats["blocked_unknown_signals"], 2)
        self.assertEqual(result.stats["status"], NON_PIT_LABEL)
        self.assertFalse(result.stats["is_point_in_time"])
        self.assertTrue(result.stats["lookahead_risk"])
        self.assertEqual(result.stats["unmatched_signal_symbols"], 1)
        self.assertEqual(
            result.audit.set_index("stock_code").loc["123456", "reason"],
            "MISSING_STOCK_INFO",
        )
        pd.testing.assert_frame_equal(raw, before)

    def test_st_field_and_st_name_disagreement(self):
        result = filter_signals_by_current_st(
            signals(("002898", "2025-03-03", 5),
                    ("600200", "2025-03-03", 4),
                    ("000001", "2025-03-03", 3)), info()
        )
        audit = result.audit.set_index("stock_code")
        self.assertEqual(audit.loc["002898", "reason"], "NAME_CONTAINS_ST")
        self.assertEqual(audit.loc["600200", "reason"], "ST_FIELD_AND_NAME")
        self.assertEqual(audit.loc["000001", "screen_status"], "CURRENT_NORMAL")

    def test_bad_metadata_fails_closed_and_duplicates_raise(self):
        raw = signals(("000001", "2025-03-03", 1))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            filter_signals_by_current_st(raw, pd.concat([info(), info()]))
        with self.assertRaisesRegex(ValueError, "Missing stock_info"):
            filter_signals_by_current_st(raw, info().drop(columns="stock_name"))
        bad = info()
        # Pandas 2.2+/3.x rejects assigning a string to a float64 column.
        # Cast before injecting the deliberately malformed ST value so the
        # test exercises the filter's UNKNOWN/fail-closed behavior.
        bad["st"] = bad["st"].astype(object)
        bad.loc[bad.stock_code == "000001", "st"] = "?"
        res = filter_signals_by_current_st(raw, bad)
        self.assertEqual(len(res.signals), 0)
        self.assertEqual(res.stats["blocked_unknown_signals"], 1)

    def test_loader_reads_current_stock_info_not_market(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        with engine.begin() as connection:
            connection.exec_driver_sql("ATTACH DATABASE ':memory:' AS public")
            connection.exec_driver_sql(
                "CREATE TABLE public.stock_info "
                "(stock_code TEXT, stock_name TEXT, st INTEGER)"
            )
            connection.exec_driver_sql(
                "INSERT INTO public.stock_info VALUES "
                "('000001','平安银行',0),('600200','*ST苏吴',2)"
            )
        df = load_current_stock_info(engine)
        self.assertEqual(sorted(df.stock_code.tolist()), ["000001", "600200"])

    def test_blacklist_does_not_force_sell_in_engine(self):
        raw = signals(("600200", "2025-03-03", 100),
                      ("000001", "2025-03-03", 10))
        filtered = filter_signals_by_current_st(raw, info()).signals
        market = bars([
            ("600200", "2025-03-03", 10, 10),
            ("000001", "2025-03-03", 10, 10),
            ("600200", "2025-03-04", 10, 10),
            ("000001", "2025-03-04", 10, 10),
            ("000001", "2025-03-05", 11, 11),
        ])
        cfg = BacktestConfig(
            date(2025, 3, 3), date(2025, 3, 5),
            initial_cash=100000, max_positions=1,
            allow_unverified_tradeability=True,
        )
        result = run_backtest(market, filtered, cfg)
        self.assertEqual(result.trades.stock_code.unique().tolist(), ["000001"])
        self.assertEqual(result.metrics["unclosed_positions"], 0)

    def test_only_known_st_codes_excluded_no_future_bar_filter(self):
        raw = signals(("000001", "2025-03-03", 10))
        out = filter_signals_by_current_st(raw, info())
        self.assertEqual(len(out.signals), 1)
        self.assertTrue(
            out.audit.loc[out.audit.stock_code == "000001", "screen_status"]
            .eq("CURRENT_NORMAL").all()
        )


if __name__ == "__main__":
    unittest.main()
