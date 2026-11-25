"""Independent daily panel regression checks."""
from datetime import datetime
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ashare.p54_factors import EXTRA_FEATURES, P54_FEATURES
from ashare.p55_panel import (ARMS, VERSION, empty_details, factor_sequences, freeze_panel, identity,
                              predictions_for, publish_panel, validate_existing)
from ashare.p5_sequences import CHINA


def at(day, hour):
    return datetime(2026, 11, day, hour, tzinfo=CHINA)


def details():
    return pd.DataFrame({
        "stock_code": ["000002", "000001", "000003"], "trade_date": "2026-11-26", "liquidity_rank": [1, 2, 3],
        "training_sequences": [0, 19, 22], "outcome": ["SCORED", "SCORED", "NO_PRICE"],
        "auxiliary_reference_fallback_feature_sessions": 0,
    })


class DailyPanelTests(unittest.TestCase):
    def test_policy_freeze_is_independent_immutable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = Path(tmp)
            with patch("ashare.p55_panel.dependencies", return_value=({"frozen_at_utc": at(24, 16).isoformat()},
                                                                    {"digest": "p54"})):
                first = freeze_panel(panel, panel, panel, clock=lambda: at(25, 13))
                again = freeze_panel(panel, panel, panel, clock=lambda: at(26, 19))
            self.assertEqual(first, again)
            self.assertEqual(first["frozen_at"], at(25, 13).isoformat())
            with patch("ashare.p55_panel.dependencies", return_value=({"frozen_at_utc": at(24, 16).isoformat()},
                                                                    {"digest": "changed"})):
                with self.assertRaises(ValueError):
                    freeze_panel(panel, panel, panel, clock=lambda: at(26, 19))

    def test_equal_weight_uses_all_qualified_names_and_ids_separate_replay(self):
        rows = details()
        output = predictions_for("equal_weight", rows, {"000001": 0., "000002": 0.},
                                 "2026-11-26", "TIMELY", at(25, 14).isoformat(), identity("equal"))
        self.assertEqual(output.stock_code.tolist(), ["000001", "000002"])
        self.assertEqual(output.portfolio_weight.tolist(), [.5, .5])
        self.assertEqual(output.support_group.tolist(), ["5-19", "0-4"])
        hist = predictions_for("equal_weight", rows, {"000001": 0., "000002": 0.},
                               "2026-11-26", "HISTORICAL", at(25, 14).isoformat(), identity("equal"))
        self.assertTrue(set(output.prediction_id).isdisjoint(hist.prediction_id))
        with self.assertRaises(ValueError):
            predictions_for("equal_weight", rows, {"000001": 0.}, "2026-11-26", "TIMELY", "freeze", "weights")

    def test_factor_nan_in_one_context_bar_rejects_whole_sequence(self):
        keys = pd.DataFrame({"stock_code": ["000001", "000001", "000002", "000002"],
                             "trade_date": ["2026-11-25", "2026-11-26"] * 2})
        frame = keys.assign(**{name: 1. for name in P54_FEATURES if name not in EXTRA_FEATURES})
        extras = keys.assign(**dict.fromkeys(EXTRA_FEATURES, 2.0))
        extras.loc[1, EXTRA_FEATURES[2]] = np.nan
        with patch("ashare.p55_panel.build_market_factors", return_value=(extras, {"version": "p54"})) as call:
            matrix, valid, remapped, _, missing = factor_sequences(
                pd.DataFrame(), SimpleNamespace(features=frame), np.array([[0, 1], [2, 3]]), 100)
        self.assertEqual(matrix.shape, (4, 18))
        np.testing.assert_array_equal(remapped, np.array([[0, 1], [2, 3]]))
        self.assertEqual(valid.tolist(), [False, True])
        self.assertEqual(missing, [EXTRA_FEATURES[2], ""])
        self.assertEqual(call.call_args.kwargs["min_market_stocks"], 100)
        self.assertEqual(call.call_args.args[1].columns.to_list(), list(keys.columns))


@unittest.skipUnless(importlib.util.find_spec("polars"), "Polars is provided by ubuntu92")
class PanelPublicationTests(unittest.TestCase):
    def make(self, root, started, committed, *, kind="TIMELY", policy_day=25, formal=True):
        signal = "2026-11-26"
        manifest = {"schema_version": VERSION, "status": "ABSTAINED" if formal else "NO_FORMAL_SESSION",
                    "trade_date": signal, "formal_session": formal, "legacy_capture_kind": kind,
                    "capture_kind": kind, "capture_started_at": started.isoformat(),
                    "legacy_manifest_sha256": "origin", "policy_sha256": "independent", "models": {}}
        policy = {"frozen_at": at(policy_day, 16).isoformat()}
        model = {"frozen_at_utc": at(24, 14).isoformat(), "weights_sha256": "model"}
        data = {arm: details() if formal else empty_details() for arm in ARMS}
        scores = {arm: {"000001": 0., "000002": float(arm != "equal_weight")} if formal else {} for arm in ARMS}
        return publish_panel(root / signal, manifest, policy, model, data, scores, lambda: committed)

    def test_separate_freeze_real_cutoff_and_equal_has_no_pseudo_top_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            m = self.make(folder, at(26, 18), at(26, 19))
            self.assertTrue(m["prospective_eligible"])
            self.assertEqual(m["models"]["equal_weight"]["top20"], [])
            self.assertEqual(m["models"]["ma20_reversal"]["top20"], ["000002", "000001"])
            self.assertEqual(validate_existing(folder / "2026-11-26", "origin", "independent"), m)
            with self.assertRaises(ValueError):
                validate_existing(folder / "2026-11-26", "changed", "independent")

    def test_replay_freeze_day_and_late_are_kept_out_of_prospective(self):
        for kind, freeze_day, finished, expected in (
                ("HISTORICAL", 25, at(27, 19), "SCORED"),
                ("TIMELY", 26, at(26, 19), "SCORED"),
                ("TIMELY", 25, at(27, 19), "MISSED_CAPTURE_WINDOW"),
                ("TIMELY", 25, at(26, 19), "NO_FORMAL_SESSION")):
            with self.subTest(kind=kind, freeze_day=freeze_day, finished=finished, expected=expected):
                with tempfile.TemporaryDirectory() as tmp:
                    item = self.make(Path(tmp), at(26, 18), finished, kind=kind, policy_day=freeze_day,
                                     formal=expected != "NO_FORMAL_SESSION")
                    self.assertFalse(item["prospective_eligible"])
                    self.assertEqual(item["status"], expected)
                    if expected != "SCORED":
                        self.assertEqual(item["models"]["equal_weight"]["rows"], 0)


if __name__ == "__main__":
    unittest.main()
