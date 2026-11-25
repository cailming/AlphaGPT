"""Compare factor inputs only when the supervision and sequence keys are identical."""
from types import SimpleNamespace
import unittest

import pandas as pd

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from ashare.p53_experiments import PLAN, confirm_pair, equal_weight


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class PairedExperimentTests(unittest.TestCase):
    def test_pair_accepts_different_feature_contracts_and_rejects_changed_labels_or_keys(self):
        summary = {"val_start": "2026-06-12", "test_start": "2026-08-17", "price_end": "2026-11-23",
                   "files": {"training_dataset.parquet": "same"}}
        folds = {name: {"sha256": name, "rows": 30} for name in ("train", "validation", "test")}
        base = SimpleNamespace(audit=summary, record={"folds": folds})
        full = SimpleNamespace(audit=summary | {"version": "expanded"}, record={"folds": folds})
        confirm_pair(base, full)
        full.audit = full.audit | {"files": {"training_dataset.parquet": "different"}}
        with self.assertRaisesRegex(ValueError, "training_dataset"):
            confirm_pair(base, full)
        full.audit = summary
        full.record = {"folds": folds | {"test": {"sha256": "different", "rows": 30}}}
        with self.assertRaisesRegex(ValueError, "test sequence"):
            confirm_pair(base, full)

    def test_equal_weight_uses_whole_pool_and_net_excess_comes_only_from_cost(self):
        frame = pd.DataFrame({"trade_date": ["a", "a", "b"], "target_return": [.1, -.1, .08]})
        actual = equal_weight(frame, 0., PLAN)
        self.assertEqual(actual["days"], 2)
        self.assertEqual(actual["samples"], 3)
        self.assertIsNone(actual["mean_daily_rank_ic"])
        self.assertAlmostEqual(actual["mean_five_session_gross"], .04)
        self.assertAlmostEqual(actual["net_excess_over_gross_pool_by_per_side_bps"]["10"], -.002)
