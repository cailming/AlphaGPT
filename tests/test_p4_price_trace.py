"""Synthetic source OHLC reconciliation tests for P4.3-D."""
from __future__ import annotations

from datetime import date
import unittest

import pandas as pd
from sqlalchemy import create_engine

from ashare.p4_price_trace import (
    load_formal_source_cases, select_cases, trace_price_paths,
)


class P4PriceTraceTests(unittest.TestCase):
    def setUp(self):
        self.cal = pd.bdate_range("2025-01-02", periods=25)
        self.expected = 18 / 13 - 1
        self.outliers = pd.DataFrame([
            {"split": "train", "trade_date": self.cal[2],
             "stock_code": "000001", "session_id": 12,
             "label_end_session": 18, "target_return": self.expected},
            {"split": "validation", "trade_date": self.cal[2],
             "stock_code": "000002", "session_id": 12,
             "label_end_session": 18, "target_return": 0.5},
        ])
        rows = []
        for i, d in enumerate(self.cal):
            for code in ("000001", "000002"):
                op = (10 + i) if code == "000001" else (20 + .1 * i)
                rows.append({"stock_code": code, "trade_date": d,
                             "open": op, "high": op + .2,
                             "low": op - .2, "close": op})
        self.bars = pd.DataFrame(rows)

    def test_exact_open_to_open_reconciliation(self):
        cases = select_cases(self.outliers.iloc[:1], test_start="2025-02-15")
        report, paths, days = trace_price_paths(cases, self.bars, self.cal)
        self.assertEqual(report["requested_cases"], 1)
        self.assertEqual(report["recomputed_equal_within_tolerance"], 1)
        self.assertEqual(paths.iloc[0]["status"], "RECOMPUTED_OK")
        self.assertAlmostEqual(paths.iloc[0]["recomputed_label"], self.expected)
        self.assertEqual(len(days), 6)
        self.assertEqual(days.offset_after_signal.tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(paths.iloc[0]["entry_trade_date"], self.cal[3])
        self.assertEqual(paths.iloc[0]["exit_trade_date"], self.cal[8])

    def test_gap_cannot_be_silently_skipped(self):
        missing = self.bars.loc[
            ~((self.bars.stock_code == "000001") &
              (self.bars.trade_date == self.cal[5]))].copy()
        cases = select_cases(self.outliers.iloc[:1], test_start="2025-02-15")
        report, paths, days = trace_price_paths(cases, missing, self.cal)
        self.assertEqual(report["missing_or_invalid_paths"], 1)
        self.assertEqual(paths.iloc[0]["status"], "MISSING_OR_INVALID_BAR")
        self.assertEqual(days.open.isna().sum(), 1)

    def test_detects_mismatch_and_large_open_jump_without_classifying_action(self):
        cases = select_cases(self.outliers, test_start="2025-02-15")
        raw = self.bars.copy()
        raw.loc[(raw.stock_code == "000002") &
                (raw.trade_date == self.cal[7]), ["open", "high", "low", "close"]] = [
                    30.0, 30.2, 29.8, 30.0
                ]
        report, path, _ = trace_price_paths(cases, raw, self.cal, jump_threshold=.15)
        self.assertEqual(report["label_mismatch_paths"], 1)
        self.assertEqual(report["large_consecutive_open_jumps_for_review"], 1)
        self.assertEqual(report["recomputed_equal_within_tolerance"], 1)
        self.assertIn("NOT proof", report["warning"])

    def test_rejects_test_period_or_fabricated_end_session(self):
        corrupt = self.outliers.copy()
        corrupt.loc[0, "trade_date"] = pd.Timestamp("2026-07-01")
        with self.assertRaisesRegex(ValueError, "test period"):
            select_cases(corrupt, test_start="2026-06-12")
        corrupt = self.outliers.copy()
        corrupt.loc[0, "label_end_session"] = 999
        with self.assertRaisesRegex(ValueError, "gap"):
            select_cases(corrupt, test_start="2025-02-15")

    def test_old_validation_label_must_finish_before_test(self):
        cases = select_cases(self.outliers.iloc[:1], test_start="2025-02-15")
        with self.assertRaisesRegex(ValueError, "excluded test boundary"):
            trace_price_paths(cases, self.bars, self.cal,
                              test_start=str(self.cal[8].date()))

    def test_select_only_formal_source_rows(self):
        db = create_engine("sqlite+pysqlite:///:memory:")
        with db.begin() as conn:
            conn.exec_driver_sql("ATTACH DATABASE ':memory:' AS public")
            conn.exec_driver_sql(
                "CREATE TABLE public.market (stock_code TEXT, trade_date DATE,"
                " open REAL, high REAL, low REAL, close REAL, not_billing INTEGER)"
            )
            for i in range(16):
                day = str(self.cal[i].date())
                conn.exec_driver_sql(
                    "INSERT INTO public.market VALUES (?, ?, ?, ?, ?, ?, 1)",
                    ("000001", day, 10+i, 10+i+.2, 10+i-.2, 10+i)
                )
            conn.exec_driver_sql(
                "INSERT INTO public.market VALUES "
                "('000001','2025-01-08',100,101,99,100,2),"
                "('000003','2025-01-08',100,101,99,100,1)"
            )
        cases = select_cases(self.outliers.iloc[:1], test_start="2025-02-15")
        raw, calendar = load_formal_source_cases(cases, engine=db)
        self.assertEqual(set(raw.stock_code), {"000001"})
        self.assertEqual(calendar[0], self.cal[0] if self.cal[0] >=
                         cases.trade_date.min() else cases.trade_date.min())
        self.assertEqual(len(raw), 14)


if __name__ == "__main__":
    unittest.main()
