"""No-DB tests for train/validation-only factor and label audit."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ashare.p4_factor_audit import (
    FEATURE_NAMES, audit_pool, factor_diagnostics, label_diagnostics,
    split_audit_pool,
)


def make_research_frame():
    calendar = pd.bdate_range("2025-01-02", periods=125)
    rows = []
    for i, day in enumerate(calendar[:100]):
        for j in range(5):
            # A factor that intentionally has positive daily rank IC.
            y = 0.001 * j - 0.002
            row = {
                "trade_date": day, "stock_code": f"{j:06d}",
                "session_id": i,
                "label_end_session": i + 3,
                "target_return": y,
            }
            for k, factor in enumerate(FEATURE_NAMES):
                row[factor] = float((j if k % 2 == 0 else 4 - j)
                                    + 0.00001 * i)
            rows.append(row)
    return pd.DataFrame(rows), calendar


class P4FactorAuditTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.calendar = make_research_frame()

    def test_exact_purged_date_cuts_and_holdout_excluded(self):
        split, train, val = split_audit_pool(self.frame, self.calendar)
        self.assertEqual(split["validation_start"], "2025-03-27")
        self.assertEqual(split["train_rows"], 57 * 5)
        self.assertEqual(split["validation_rows"], 17 * 5)
        self.assertEqual(split["test_rows_EXCLUDED_without_reading_labels"], 20 * 5)
        self.assertEqual(split["train_purged_or_missing_label_rows"], 3 * 5)
        self.assertEqual(split["validation_purged_or_missing_label_rows"], 3 * 5)
        self.assertLess(train.label_end_session.max(), val.session_id.min())
        self.assertLess(val.label_end_session.max(),
                        self.frame.loc[self.frame.trade_date >=
                                       self.frame.trade_date.sort_values().unique()[80],
                                       "session_id"].min())

    def test_holdout_targets_never_read(self):
        report1, daily1, factors1, labels1, outliers1 = audit_pool(
            self.frame, self.calendar, extreme_threshold=0.0025
        )
        changed = self.frame.copy()
        test_dates = sorted(changed.trade_date.unique())[80:]
        mask = changed.trade_date.isin(test_dates)
        # Impossible/non-finite labels on test should NEVER invalidate or
        # otherwise alter any train/validation output or statistics.
        changed.loc[mask, "target_return"] = float("inf")
        changed.loc[mask, "label_end_session"] = float("nan")
        report2, daily2, factors2, labels2, outliers2 = audit_pool(
            changed, self.calendar, extreme_threshold=0.0025
        )
        self.assertEqual(report1, report2)
        pd.testing.assert_frame_equal(daily1, daily2)
        pd.testing.assert_frame_equal(factors1, factors2)
        pd.testing.assert_frame_equal(labels1, labels2)
        pd.testing.assert_frame_equal(outliers1, outliers2)
        self.assertFalse(report1["test_labels_used_in_audit"])

    def test_factor_ic_direction_selected_on_train_only(self):
        split, train, val = split_audit_pool(self.frame, self.calendar)
        daily, summary = factor_diagnostics(train, val)
        self.assertEqual(len(daily), (57 + 17) * len(FEATURE_NAMES))
        self.assertEqual(len(summary), 2 * len(FEATURE_NAMES))
        positive = summary.loc[
            (summary.split == "validation") & (summary.feature == "ret_1")
        ].iloc[0]
        negative = summary.loc[
            (summary.split == "validation") & (summary.feature == "ret_5")
        ].iloc[0]
        self.assertAlmostEqual(positive.mean_ic, 1.0)
        self.assertEqual(positive.train_defined_direction, 1)
        self.assertAlmostEqual(positive.signed_mean_ic_train_direction, 1.0)
        self.assertAlmostEqual(negative.mean_ic, -1.0)
        self.assertEqual(negative.train_defined_direction, -1)
        self.assertAlmostEqual(negative.signed_mean_ic_train_direction, 1.0)

    def test_label_summary_outliers_and_source_validation(self):
        _, train, val = split_audit_pool(self.frame, self.calendar)
        stats, outliers = label_diagnostics(train, val, extreme_threshold=.0015)
        self.assertEqual(stats.sample_rows.tolist(), [285, 85])
        self.assertEqual(stats.abs_gt_threshold_rows.tolist(), [114, 34])
        self.assertTrue(outliers.target_return.abs().gt(.0015).all())
        self.assertFalse((outliers.trade_date >=
                          self.frame.trade_date.sort_values().unique()[80]).any())
        broken = self.frame.copy()
        broken.loc[0, "session_id"] = 999
        with self.assertRaisesRegex(ValueError, "session_id"):
            split_audit_pool(broken, self.calendar)
        missing = self.frame.drop(columns=["label_end_session"])
        with self.assertRaisesRegex(ValueError, "Missing required pool"):
            split_audit_pool(missing, self.calendar)

    def test_purge_blocks_cross_split_labels(self):
        broken = self.frame.copy()
        val_start = sorted(broken.trade_date.unique())[60]
        mask = broken.trade_date.eq(val_start)
        broken.loc[mask, "label_end_session"] = 100
        _, train, val = split_audit_pool(broken, self.calendar)
        self.assertFalse(val.trade_date.eq(val_start).any())
        self.assertEqual(len(train), 285)
        self.assertEqual(len(val), 16 * 5)

    def test_invalid_fraction_threshold_and_insufficient_days(self):
        with self.assertRaisesRegex(ValueError, "fractions"):
            split_audit_pool(self.frame, self.calendar, train_fraction=.95)
        with self.assertRaisesRegex(ValueError, "thresholds"):
            _, train, val = split_audit_pool(self.frame, self.calendar)
            label_diagnostics(train, val, extreme_threshold=0)
        with self.assertRaisesRegex(ValueError, "at least 80"):
            split_audit_pool(self.frame.loc[
                self.frame.trade_date < self.frame.trade_date.unique()[50]
            ], self.calendar)


if __name__ == "__main__":
    unittest.main()
