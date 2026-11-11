"""Pure-data tests; do not require a live database or network."""
import unittest

import numpy as np
import pandas as pd

from ashare.daily import make_daily_features
from ashare.minute import build_minute_intervals


class DailyTests(unittest.TestCase):
    def sample(self):
        dates = pd.bdate_range("2026-01-01", periods=26)
        records = []
        for code, base in [("000001", 10), ("600000", 100)]:
            for i, d in enumerate(dates):
                records.append({"stock_code": code, "trade_date": d, "open": base + i,
                                "close": base + i, "high": base + i + 1,
                                "volume": 100 + i, "ma20": base + i - 1,
                                "main_force_score": 80})
        return pd.DataFrame(records)

    def test_no_cross_stock_or_future_window(self):
        source = self.sample()
        out = make_daily_features(source)
        first = out[out.stock_code == "000001"].reset_index(drop=True)
        self.assertTrue(np.isnan(first.loc[0, "ret_1d"]))
        self.assertTrue(np.isnan(first.loc[19, "vol_ratio_20"]))
        self.assertAlmostEqual(first.loc[20, "vol_ratio_20"], 120 / 109.5)
        self.assertAlmostEqual(first.loc[0, "target_next_open_1d"], 1 / 11)
        source.loc[(source.stock_code == "000001") &
                   (source.trade_date == source.trade_date.max()), "volume"] = 999999
        changed = make_daily_features(source)
        first_changed = changed[changed.stock_code == "000001"].reset_index(drop=True)
        self.assertAlmostEqual(first.loc[20, "vol_ratio_20"],
                               first_changed.loc[20, "vol_ratio_20"])

    def test_duplicate_rejected(self):
        source = self.sample()
        with self.assertRaises(ValueError):
            make_daily_features(pd.concat([source, source.iloc[[0]]]))


class MinuteTests(unittest.TestCase):
    def test_delta_no_reconstruction_across_gap_or_lunch(self):
        base = {"stock_code": "000001", "trade_date": "2026-11-10",
                "high": 11, "low": 9, "trade": 10}
        records = [dict(base, ticktime=t, volume=v, amount=a)
                   for t, v, a in [("09:30:00", 100, 1000),
                                   ("09:31:00", 130, 1310),
                                   ("09:33:00", 180, 1840),
                                   ("11:30:00", 500, 5200),
                                   ("13:00:00", 520, 5420),
                                   ("13:01:00", 550, 5700)]]
        out = build_minute_intervals(pd.DataFrame(records))
        self.assertEqual(out.quality_flag.tolist(), [
            "FIRST_OBSERVATION", "OK", "NON_CONSECUTIVE",
            "NON_CONSECUTIVE", "NON_CONSECUTIVE", "OK"])
        self.assertTrue(np.isnan(out.loc[2, "minute_volume"]))
        self.assertEqual(out.loc[1, "minute_volume"], 30)
        self.assertEqual(out.loc[5, "minute_amount"], 280)

    def test_rollback_and_minute_dedup_prefer_today(self):
        raw = pd.DataFrame([
            {"stock_code": "000001", "trade_date": "2026-11-10",
             "ticktime": "09:30:00", "trade": 10, "volume": 100,
             "amount": 1000, "source": "archive"},
            {"stock_code": "000001", "trade_date": "2026-11-10",
             "ticktime": "09:31:00", "trade": 10, "volume": 130,
             "amount": 1300, "source": "archive"},
            {"stock_code": "000001", "trade_date": "2026-11-10",
             "ticktime": "09:31:00", "trade": 10, "volume": 132,
             "amount": 1330, "source": "today"},
            {"stock_code": "000001", "trade_date": "2026-11-10",
             "ticktime": "09:32:00", "trade": 10, "volume": 90,
             "amount": 900, "source": "today"},
        ])
        out = build_minute_intervals(raw)
        self.assertEqual(len(out), 3)
        self.assertEqual(out.loc[1, "minute_volume"], 32)
        self.assertEqual(out.loc[2, "quality_flag"], "CUMULATIVE_DECREASE")
        self.assertTrue(np.isnan(out.loc[2, "minute_volume"]))


if __name__ == "__main__":
    unittest.main()
