"""Checks for optional database/search progress callbacks."""
import unittest
from datetime import date

import pandas as pd
import torch
from sqlalchemy import create_engine

from ashare.daily import AShareDailyLoader, DAILY_COLUMNS
from ashare.tensor import DailyFactorTensor
from ashare.evaluate import search_baseline


class ProgressTests(unittest.TestCase):
    def test_daily_loader_reports_chunk_rows(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        with engine.begin() as conn:
            conn.exec_driver_sql("ATTACH DATABASE ':memory:' AS public")
            conn.exec_driver_sql('''CREATE TABLE public.market (
                stock_code TEXT, trade_date DATE, open REAL, high REAL,
                low REAL, close REAL, volume REAL, amount REAL, "MA_20" REAL
            )''')
            conn.exec_driver_sql('''CREATE TABLE public.stock_main_force_daily (
                stock_code TEXT, trade_date DATE, main_force_score REAL,
                activity_score REAL, strength_score REAL, holding_score REAL,
                distribution_risk REAL, amount_ratio_5_20 REAL,
                drawdown_20_pct REAL, created_at TEXT
            )''')
            for day, price in [("2026-11-09", 10), ("2026-11-10", 11),
                               ("2026-11-11", 12)]:
                conn.exec_driver_sql(
                    'INSERT INTO public.market VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    ("000001", day, price, price + 1, price - 1,
                     price, 100, 1000, 10))
        events = []
        data = AShareDailyLoader(engine).load(
            date(2026, 11, 9), date(2026, 11, 9),
            warmup_calendar_days=0,
            label_lookahead_calendar_days=7,
            progress=events.append, chunksize=2,
        )
        self.assertEqual(len(data), 1)
        self.assertTrue(any("已读取 2 行" in msg for msg in events))
        self.assertTrue(any("已读取 3 行" in msg for msg in events))
        self.assertTrue(any("因子计算完成" in msg for msg in events))

    def test_search_reports_first_intervals_and_final(self):
        torch.manual_seed(1)
        num_stocks, days = 3, 40
        features = torch.rand(num_stocks, len(DAILY_COLUMNS), days)
        labels = torch.rand(num_stocks, days)
        data = DailyFactorTensor(
            ("000001", "000002", "600000"),
            pd.bdate_range("2026-07-01", periods=days),
            features, torch.ones_like(features, dtype=torch.bool),
            labels, torch.ones_like(labels, dtype=torch.bool),
        )
        events = []
        search_baseline(
            data, start=date(2026, 7, 1), end=date(2026, 8, 25),
            samples=12, min_stocks=3, min_train_days=3,
            progress=lambda *args: events.append(args),
            progress_every=5,
        )
        self.assertEqual([e[0] for e in events], [1, 5, 10, 12])
        self.assertTrue(all(e[1] == 12 and e[2] <= e[0] for e in events))

    def test_bad_progress_configuration(self):
        data = DailyFactorTensor(
            ("000001",), pd.bdate_range("2026-07-01", periods=20),
            torch.zeros(1, len(DAILY_COLUMNS), 20),
            torch.ones(1, len(DAILY_COLUMNS), 20, dtype=torch.bool),
            torch.zeros(1, 20), torch.ones(1, 20, dtype=torch.bool),
        )
        with self.assertRaises(ValueError):
            search_baseline(data, start=date(2026, 7, 1),
                            end=date(2026, 7, 28), progress_every=0)


if __name__ == "__main__":
    unittest.main()
