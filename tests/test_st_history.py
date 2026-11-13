"""Historical ST snapshot PIT tests; no third-party feed or database required."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd

from ashare.st_history import (
    filter_signals_by_historical_st, load_st_snapshots, load_st_coverage
)


def sigs():
    return pd.DataFrame([
        ("000001", "2025-03-03", 1.2),
        ("600200", "2025-03-03", 2.2),
        ("300344", "2025-03-04", 3.3),
        ("000001", "2025-03-04", 4.4),
        ("920680", "2025-03-05", 5.5),
    ], columns=["stock_code", "signal_date", "score"])


def st():
    return pd.DataFrame([
        ("600200.SH", "2025-03-03"),
        ("300344.SZ", "2025-03-04"),
    ], columns=["stock_code", "trade_date"])


def coverage():
    return pd.DataFrame([
        ("2025-03-03", True, 1, "vendor-verified"),
        ("2025-03-04", True, 1, "vendor-verified"),
        # March 5 has no completeness certificate: UNKNOWN, not normal.
    ], columns=["trade_date", "complete", "st_count", "source"])


class STSnapshotTests(unittest.TestCase):
    def test_only_t_day_known_normal_is_allowed(self):
        r = filter_signals_by_historical_st(sigs(), st(), coverage())
        self.assertEqual(r.stats["eligible_normal_rows"], 2)
        self.assertEqual(r.stats["blocked_st_rows"], 2)
        self.assertEqual(r.stats["blocked_unknown_rows"], 1)
        self.assertEqual(r.signals.stock_code.tolist(), ["000001", "000001"])
        self.assertEqual(r.stats["risk_check_at"], "T_SIGNAL_DATE")

    def test_incomplete_date_fails_closed_even_with_no_st_rows(self):
        cov = coverage()
        cov.loc[1, "complete"] = False
        r = filter_signals_by_historical_st(sigs(), st(), cov)
        self.assertEqual(r.stats["blocked_unknown_rows"], 3)
        self.assertEqual(r.stats["eligible_normal_rows"], 1)

    def test_full_snapshot_row_count_mismatch_rejected(self):
        cov = coverage()
        cov.loc[0, "st_count"] = 0
        with self.assertRaisesRegex(ValueError, "disagree"):
            filter_signals_by_historical_st(sigs(), st(), cov)

    def test_duplicate_st_or_coverage_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            filter_signals_by_historical_st(sigs(), pd.concat([st(), st()]), coverage())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            filter_signals_by_historical_st(sigs(), st(), pd.concat([coverage(), coverage()]))

    def test_parse_csv_tushare_codes_and_dates(self):
        with TemporaryDirectory() as folder:
            file1 = Path(folder) / "st_daily.csv"
            file2 = Path(folder) / "st_coverage.csv"
            file1.write_text("ts_code,trade_date\n600200.SH,20250303\n", encoding="utf8")
            file2.write_text("trade_date,complete,st_count,source\n20250303,true,1,official\n", encoding="utf8")
            rows = load_st_snapshots(file1)
            cov = load_st_coverage(file2)
            self.assertEqual(rows.stock_code.iloc[0], "600200")
            self.assertEqual(cov.st_count.iloc[0], 1)

    def test_reject_malformed_values(self):
        with self.assertRaisesRegex(ValueError, "stock_code"):
            filter_signals_by_historical_st(
                sigs().assign(stock_code="not_code"), st(), coverage())
        with self.assertRaisesRegex(ValueError, "complete"):
            filter_signals_by_historical_st(
                sigs(), st(), coverage().assign(complete="unknown"))


if __name__ == "__main__":
    unittest.main()
