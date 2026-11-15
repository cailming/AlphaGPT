"""Synthetic train/validation tests for P4.3-D frozen lightweight baselines."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ashare.p4_factor_audit import FEATURE_NAMES, split_audit_pool
from ashare.p4_light_baselines import compare_train_validation, train_factor_directions


def bars():
    calendar = pd.bdate_range("2025-01-02", periods=120)
    rows = []
    for i, day in enumerate(calendar[:100]):
        for j in range(15):
            # Both calendar and feature names match P4's original pipeline.
            target = (j - 7) * 0.002 + (i % 7 - 3) * .0003
            row = {"trade_date": day, "stock_code": f"{j:06d}",
                   "session_id": i, "label_end_session": i+6,
                   "target_return": target}
            for k, name in enumerate(FEATURE_NAMES):
                row[name] = float((j if k % 2 == 0 else 14-j)
                                  + .001 * np.sin(i + j + k))
            rows.append(row)
    return pd.DataFrame(rows), calendar


class LightBaselineTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.calendar = bars()
        self.split, self.train, self.val = split_audit_pool(
            self.frame, self.calendar
        )

    def test_seven_candidates_on_same_validation_rows(self):
        audit, rows, daily, models = compare_train_validation(self.train, self.val, top_k=5)
        self.assertEqual(len(rows), 7)
        self.assertEqual(len(models), 4)
        self.assertEqual(len(daily), 7 * self.val.trade_date.nunique())
        self.assertEqual(set(rows.evaluated_samples), {len(self.val)})
        self.assertTrue(rows.mean_rank_ic.notna().all())
        self.assertEqual(audit["train_samples"], len(self.train))
        self.assertEqual(audit["validation_samples"], len(self.val))
        self.assertTrue(audit["no_validation_winner_promoted"])
        self.assertFalse(audit["test_labels_used"])

    def test_caps_and_signs_are_train_only(self):
        a, _, _, _ = compare_train_validation(self.train, self.val, top_k=5)
        modified = self.val.copy()
        modified["target_return"] = -modified.target_return
        b, rows, _, _ = compare_train_validation(self.train, modified, top_k=5)
        self.assertEqual(a, b)
        self.assertTrue(np.isfinite(rows.mean_rank_ic).all())
        self.assertAlmostEqual(
            a["label_caps_abs_quantile_from_train"]["q95"],
            self.train.target_return.abs().quantile(.95)
        )
        self.assertEqual(a["train_factor_directions"],
                         train_factor_directions(self.train)[0])

    def test_test_period_targets_cannot_affect_result(self):
        _, tr, val = split_audit_pool(self.frame, self.calendar)
        baseline, rows, daily, _ = compare_train_validation(tr, val, top_k=5)
        altered = self.frame.copy()
        test_start = sorted(altered.trade_date.unique())[80]
        altered.loc[altered.trade_date.ge(test_start), "target_return"] = np.inf
        altered.loc[altered.trade_date.ge(test_start), "label_end_session"] = np.nan
        _, tr2, val2 = split_audit_pool(altered, self.calendar)
        now, rows2, daily2, _ = compare_train_validation(tr2, val2, top_k=5)
        self.assertEqual(baseline, now)
        pd.testing.assert_frame_equal(rows, rows2)
        pd.testing.assert_frame_equal(daily, daily2)

    def test_invalid_configuration_and_nonfinite_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "configuration"):
            compare_train_validation(self.train, self.val, top_k=0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            compare_train_validation(self.val, self.val, top_k=5)
        corrupt = self.val.copy()
        corrupt.loc[corrupt.index[0], "ret_1"] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            compare_train_validation(self.train, corrupt, top_k=5)


if __name__ == "__main__":
    unittest.main()
