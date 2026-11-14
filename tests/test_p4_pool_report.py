"""Unit tests for P4.1 screening audit and calendar coverage (no DB)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ashare.p4_mvp import PoolConfig, build_pool
from ashare.p4_pool_report import build_coverage_tables


def sample_bars():
    dates = pd.bdate_range("2025-01-02", periods=90)
    rows = []
    for code, level, amount in (
        ("000001", 10.0, 2_000_000.0),
        ("000002", 2.0, 4_000_000.0),
        ("000003", 18.0, 6_000_000.0),
    ):
        for i, day in enumerate(dates):
            op = level + 0.01 * i
            cl = op * (1 + 0.002 * np.sin(i / 4.0))
            rows.append({
                "trade_date": day, "stock_code": code,
                "open": op, "close": cl,
                "high": max(op, cl) * 1.01,
                "low": min(op, cl) * 0.99,
                "volume": 100_000.0, "amount": amount,
                "not_billing": 1,
            })
    return pd.DataFrame(rows), dates


class P4PoolReportTests(unittest.TestCase):
    def test_rejections_partition_requested_window_and_monthly_totals(self):
        raw, days = sample_bars()
        start, end = days[30].date(), days[70].date()
        pool = build_pool(raw, start, end, PoolConfig(
            top_n=1, min_price=3.0, consecutive_sessions=25, hold_sessions=3,
        ))
        stats = pool.stats
        reasons = stats["rejected_by_reason"]
        self.assertGreater(reasons["below_min_price"], 0)
        self.assertGreater(reasons["outside_daily_top_n"], 0)
        self.assertEqual(
            stats["requested_window_rows"],
            stats["selected_rows"] + sum(reasons.values()),
        )
        self.assertEqual(set(pool.data.stock_code), {"000003"})
        daily, monthly = build_coverage_tables(pool, start, end)
        self.assertEqual(len(daily), 41)
        self.assertEqual(int(daily.pool_size.sum()), stats["selected_rows"])
        self.assertEqual(int(monthly.selected_samples.sum()), stats["selected_rows"])
        self.assertEqual(
            int(monthly.valid_label_samples.sum()), stats["rows_with_valid_label"]
        )
        self.assertTrue(monthly.unique_stock_codes.ge(0).all())

    def test_zero_selection_days_are_not_silently_dropped(self):
        raw, days = sample_bars()
        start, end = days[30].date(), days[70].date()
        pool = build_pool(raw, start, end, PoolConfig(
            top_n=1, min_price=1000.0, consecutive_sessions=25,
        ))
        daily, monthly = build_coverage_tables(pool, start, end)
        self.assertEqual(len(pool.data), 0)
        self.assertEqual(len(daily), 41)
        self.assertEqual(int(daily.pool_size.sum()), 0)
        self.assertEqual(int(monthly.unique_stock_codes.sum()), 0)
        self.assertEqual(int(monthly.days_with_candidates.sum()), 0)


if __name__ == "__main__":
    unittest.main()
