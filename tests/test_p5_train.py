"""Regression tests for paired P5 objectives and the rolling artifact contract."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from ashare.p4_action_pipeline import atomic_json
    from ashare.p4_subset_dataset import GUARD
    from ashare.p4_subset_model import FEATURES
    from ashare.p5_train import DATASET_VERSION, labels_for, read_p5_audit


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class ObjectiveTests(unittest.TestCase):
    def test_fold_local_transforms_use_daily_equal_weight_and_average_ties(self):
        frame = pd.DataFrame({"trade_date": ["a", "b", "a", "a"], "target_return": [.2, .8, .2, -.1]})
        np.testing.assert_allclose(labels_for(frame, "absolute"), [.2, .8, .2, -.1])
        np.testing.assert_allclose(labels_for(frame, "excess"), [.1, 0, .1, -.2])
        np.testing.assert_allclose(labels_for(frame, "rank"), [1/6, 0, 1/6, -1/3])
        np.testing.assert_allclose(labels_for(frame.loc[frame.trade_date.eq("a")], "excess"),
                                   labels_for(frame, "excess")[[0, 2, 3]])
        with self.assertRaises(ValueError):
            labels_for(frame, "unsupported")

    def test_rolling_audit_has_distinct_v1_boundary_and_rejects_bad_split_order(self):
        audit = {**GUARD, "version": DATASET_VERSION, "subset_training_allowed": True,
                 "feature_names": list(FEATURES), "val_start": "2026-08-23", "test_start": "2026-10-23",
                 "price_end": "2026-11-23", "research_end_exclusive": "2026-11-24",
                 "v1_old_test_excluded": "2026-06-12", "splits": {
                     fold: {"samples": 1200} for fold in ("train", "validation", "test")},
                 "files": {}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "subset_audit.json"
            atomic_json(path, audit)
            self.assertEqual(read_p5_audit(root)["v1_old_test_excluded"], "2026-06-12")
            atomic_json(path, audit | {"test_start": audit["price_end"]})
            with self.assertRaises(ValueError):
                read_p5_audit(root)
