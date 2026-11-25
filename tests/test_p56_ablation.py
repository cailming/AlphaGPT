"""Regression tests for exact architecture and independent factor score summaries."""
import unittest

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from ashare.p4_mvp import FEATURES
    from ashare.p54_factors import EXTRA_FEATURES, P54_FEATURES
    from ashare.p56_ablation import GROUPS, MARKET_FACTORS, active, factor_diagnostics, groups_on, masked_sequences


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class AblationTests(unittest.TestCase):
    def test_each_group_masks_only_other_standardized_columns_at_same_dimension(self):
        self.assertEqual(tuple(sum((list(GROUPS[arm]) for arm in ("group_a", "group_b", "group_c", "group_d")), [])),
                         EXTRA_FEATURES)
        shape = (3, 16, len(P54_FEATURES))
        for arm in GROUPS:
            result = masked_sequences(np.ones(shape, dtype=np.float32), arm)
            self.assertEqual(result.shape, shape)
            for i, feature in enumerate(P54_FEATURES):
                self.assertTrue(np.all(result[:, :, i] == (1 if feature in active(arm) else 0)))
        self.assertEqual(active("base10"), FEATURES)

    @staticmethod
    def fixture():
        rows = []
        for day_index, day in enumerate(pd.bdate_range("2026-09-01", periods=6)):
            for stock in range(6):
                row = {"stock_code": f"{stock:06d}", "trade_date": day.date().isoformat(),
                       "target_return": (day_index + stock) / 100}
                row.update({name: day_index if name in MARKET_FACTORS else stock for name in EXTRA_FEATURES})
                rows.append(row)
        return pd.DataFrame(rows)

    def test_market_factors_use_temporal_daily_pool_and_stock_factors_use_cross_section(self):
        panel = self.fixture()
        summary, month, daily, quantiles = factor_diagnostics(panel)
        self.assertEqual(len(quantiles), 8 * 5)
        self.assertEqual(len(month), 8)
        self.assertEqual(len(daily), 8 * 6)
        for factor in MARKET_FACTORS:
            self.assertIsNone(summary[factor]["cross_sectional_rank_ic"])
            self.assertAlmostEqual(summary[factor]["rank_ic"], 1.)
            self.assertEqual(quantiles.loc[quantiles.factor == factor].rows.sum(), 6)
        for factor in set(EXTRA_FEATURES) - MARKET_FACTORS:
            self.assertAlmostEqual(summary[factor]["cross_sectional_rank_ic"], 1.)
            self.assertEqual(quantiles.loc[quantiles.factor == factor].rows.sum(), 36)

    def test_missing_factor_is_excluded_from_its_own_coverage_only(self):
        panel = self.fixture()
        panel[EXTRA_FEATURES[-1]] = np.nan
        summary, _, _, quantiles = factor_diagnostics(panel)
        missing = summary[EXTRA_FEATURES[-1]]
        self.assertEqual(missing["denominator_rows"], 36)
        self.assertEqual(missing["anchor_coverage"], 0)
        self.assertIsNone(missing["rank_ic"])
        self.assertEqual(quantiles.loc[quantiles.factor == EXTRA_FEATURES[-1]].rows.sum(), 0)
        self.assertEqual(summary[EXTRA_FEATURES[-2]]["anchor_coverage"], 1)

    def test_quantiles_do_not_split_tied_zero_pullbacks(self):
        rows = pd.DataFrame({"x": [0.] * 8 + [1., 2.], "target_return": np.arange(10) / 100})
        groups, spread = groups_on(rows, "x")
        self.assertIn(8, [total for _, total in groups.values()])
        self.assertNotIn(1, groups)
        self.assertIsNone(spread)
