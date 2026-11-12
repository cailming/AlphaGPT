"""Pure DataFrame tests: no PostgreSQL access, no reports written."""
import unittest
import numpy as np
import pandas as pd
from ashare.daily import DAILY_COLUMNS
from ashare.quality import analyze_factor_quality

def frame():
    rows = [
        ("000001", "2025-01-02", "000001", "2025-01-02 16:00:00", .8),
        ("000001", "2025-01-03", "000001", "2025-02-10 16:00:00", .7),
        ("000001", "2025-02-03", "000001", "2025-02-03 16:00:00", 1.1),
        ("600000", "2025-01-02", "600000", None, np.nan),
        ("600000", "2025-02-03", None, None, np.nan),
    ]
    data = []
    for code, d, joined, created, value in rows:
        row = {factor: .5 for factor in DAILY_COLUMNS}
        row.update(stock_code=code, trade_date=d,
                   main_force_joined_code=joined,
                   main_force_created_at=created,
                   main_force=value,
                   holding=value if np.isfinite(value) else np.nan)
        data.append(row)
    return pd.DataFrame(data)

class QualityTests(unittest.TestCase):
    def test_force_and_market_denominators(self):
        result = analyze_factor_quality(frame())
        audit = result.audit
        self.assertEqual(audit["market_rows"], 5)
        self.assertEqual(audit["grid_rows"], 6)
        self.assertEqual(audit["grid_missing_market_rows"], 1)
        self.assertEqual(audit["force_joined_rows"], 4)
        self.assertEqual(audit["force_created_at_missing_rows"], 1)
        self.assertEqual(audit["force_created_at_late_rows"], 1)
        main = result.summary.set_index("factor").loc["main_force"]
        self.assertEqual(main["raw_non_null"], 3)
        self.assertEqual(main["out_of_range"], 1)
        self.assertEqual(main["tensor_mask_count"], 2)
        self.assertEqual(main["quality_eligible_count"], 1)
        self.assertEqual(main["created_at_late"], 1)
        self.assertAlmostEqual(main["quality_eligible_pct_market"], 20.0)
        self.assertAlmostEqual(main["quality_eligible_pct_grid"], 16.67)
        ret = result.summary.set_index("factor").loc["ret_1d"]
        self.assertEqual(ret["quality_eligible_count"], 5)

    def test_monthly_and_backfill_grouping(self):
        report = analyze_factor_quality(frame())
        groups = report.monthly[report.monthly.factor == "main_force"]
        self.assertEqual(set(groups.month), {"2025-01", "2025-02"})
        self.assertEqual(groups.quality_eligible_count.sum(), 1)
        batch = report.backfill_batches.set_index("created_day")
        self.assertIn(pd.Timestamp("2025-02-10").date(), batch.index)
        self.assertEqual(batch.loc[pd.Timestamp("2025-02-10").date(), "late_rows"], 1)

    def test_no_rows_and_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            analyze_factor_quality(frame(), start=pd.Timestamp("2024-01-01").date(),
                                   end=pd.Timestamp("2024-02-01").date())
        f = frame()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyze_factor_quality(pd.concat([f, f.iloc[[0]]], ignore_index=True))

    def test_window_filter(self):
        report = analyze_factor_quality(frame(),
                      start=pd.Timestamp("2025-02-01").date(),
                      end=pd.Timestamp("2025-02-28").date())
        self.assertEqual(report.audit["market_rows"], 2)
        self.assertEqual(report.audit["date_count"], 1)
        self.assertEqual(report.summary.shape[0], 12)

if __name__ == "__main__":
    unittest.main()
