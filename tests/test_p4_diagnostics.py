"""Synthetic reproducibility and safety tests for saved holdout diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from ashare.p4_diagnostics import evaluate_holdout, load_inputs


def synthetic():
    rows = []
    for i, day in enumerate(pd.bdate_range("2026-07-01", periods=3)):
        for j in range(12):
            rows.append({
                "trade_date": day, "stock_code": f"{j:06d}",
                "liquidity_rank": j + 1, "close": 10.0 + j,
                "predicted_return": .05 - j * .01 + i * .0001,
                "target_return": j * .001 - .006,
            })
    return pd.DataFrame(rows)


def metadata():
    return {"model_version": "P4_RIDGE_V1", "test_start": "2026-07-01",
            "test_end": "2026-07-03", "train_last_label_date": "2026-06-30",
            "test_samples": 36, "pool_config": {"top_n": 800}}


class P4DiagnosticsTests(unittest.TestCase):
    def test_daily_monthly_and_deciles(self):
        frame = synthetic()
        result, daily, monthly, deciles = evaluate_holdout(frame, top_k=3)
        self.assertEqual(result["evaluated_samples"], 36)
        self.assertEqual(result["evaluated_days"], 3)
        self.assertAlmostEqual(result["mean_predicted_return_all_samples"],
                               frame.predicted_return.mean())
        self.assertAlmostEqual(result["predicted_positive_share_all_samples"],
                               (frame.predicted_return > 0).mean())
        self.assertAlmostEqual(result["mean_rank_ic"], -1.0)
        self.assertEqual(len(monthly), 1)
        self.assertEqual(int(monthly.iloc[0].test_samples), 36)
        self.assertEqual(len(deciles), 10)
        self.assertEqual(int(deciles.rows.sum()), 36)
        self.assertTrue(pd.isna(daily.iloc[0].topk_replacement_share))
        self.assertEqual(float(daily.iloc[1].topk_replacement_share), 0.0)
        self.assertGreater(float(deciles.iloc[-1].mean_forward_return),
                           float(deciles.iloc[0].mean_forward_return))

    def test_topk_replacement_is_name_churn_not_trading_pnl(self):
        frame = synthetic()
        # New high-score stock on second date replaces one prior Top-3 member.
        frame.loc[(frame.trade_date == pd.Timestamp("2026-07-02")) &
                  (frame.stock_code == "000011"), "predicted_return"] = 1.0
        result, daily, _, _ = evaluate_holdout(frame, top_k=3)
        self.assertAlmostEqual(float(daily.iloc[1].topk_replacement_share), 1/3)
        self.assertIn("NOT realized trading turnover", result["warning"])

    def test_metadata_and_duplicate_guards(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            file = parent / "test_predictions.csv"
            meta = parent / "model_metadata.json"
            data = synthetic()
            data.to_csv(file, index=False)
            meta.write_text(json.dumps(metadata()), encoding="utf-8")
            loaded, details = load_inputs(file, meta)
            self.assertEqual(len(loaded), 36)
            self.assertEqual(details["model_version"], "P4_RIDGE_V1")

            duplicated = pd.concat([data, data.iloc[[0]]], ignore_index=True)
            duplicated.to_csv(file, index=False)
            broken = metadata()
            broken["test_samples"] = len(duplicated)
            meta.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_inputs(file, meta)

            data.to_csv(file, index=False)
            broken = metadata()
            broken["train_last_label_date"] = "2026-07-02"
            meta.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid purged"):
                load_inputs(file, meta)

    def test_invalid_columns_and_nonfinite(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            file = parent / "test_predictions.csv"
            meta = parent / "model_metadata.json"
            meta.write_text(json.dumps(metadata()), encoding="utf-8")
            data = synthetic()
            data.drop(columns=["target_return"]).to_csv(file, index=False)
            with self.assertRaisesRegex(ValueError, "Missing predictions"):
                load_inputs(file, meta)
            data.loc[0, "predicted_return"] = np.nan
            data.to_csv(file, index=False)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                load_inputs(file, meta)


if __name__ == "__main__":
    unittest.main()
