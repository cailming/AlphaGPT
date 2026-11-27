"""Independent policy, capture chronology, masking and joint cohort checks."""
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ashare.p54_factors import P54_FEATURES
from ashare.p55_daily import CHINA
from ashare.p55_reports import LANES, summarize
from ashare.p58_group_c import ARM, capture, freeze, get_scores, track
from ashare.p58_group_c_report import capture_gaps, compare_day, review_state


def at(day, hour=18):
    return datetime(2026, 11, day, hour, tzinfo=CHINA)


class GroupCForwardTests(unittest.TestCase):
    def test_forward_review_requires_both_session_and_label_thresholds(self):
        template = {"track": "PROSPECTIVE", "lane": ARM, "scope": "OWN", "support_group": "ALL",
                    "model_sha256": "sha", "signal_matured": True, "formal": True, "captured": True,
                    "predictions": 100, "labels": 80, "pending": 0, "rank_ic": .01, "top20_full": True,
                    "top20_net_excess": .002, "top20_replacement": None, "portfolio_net_excess": None,
                    "portfolio_replacement": None}
        rows = []
        for i, session in enumerate(pd.bdate_range("2026-01-05", periods=20)):
            rows.append(template | {"signal_session": i, "day": session.date().isoformat(),
                                     "entry_date": (session + pd.offsets.BDay()).date().isoformat(),
                                     "exit_date": (session + pd.offsets.BDay(6)).date().isoformat()})
        def status(sample):
            return review_state(summarize(sample))["OWN"]["status"]
        self.assertEqual(status(rows[:-1]), "INSUFFICIENT_PROSPECTIVE_DATA")
        self.assertEqual(status([rows[0] | {"top20_full": False}, *rows[1:]]), "INSUFFICIENT_PROSPECTIVE_DATA")
        self.assertEqual(status([r | {"labels": 79} for r in rows]), "INSUFFICIENT_PROSPECTIVE_DATA")
        self.assertEqual(status(rows), "READY_FOR_REVIEW")
        self.assertLess(review_state(summarize(rows))["OWN"]["disjoint_joint_hold_windows"], 20)

    def test_freeze_day_operational_gap_separate_from_prospective_gap(self):
        calendar = ["2026-11-25", "2026-11-26", "2026-11-27"]
        timely, forward = capture_gaps(calendar, at(26, 17).isoformat(), {"2026-11-27"}, set())
        self.assertEqual(timely, ["2026-11-26"])
        self.assertEqual(forward, ["2026-11-27"])
        timely, _ = capture_gaps(calendar, at(26, 19).isoformat(), set(), set())
        self.assertEqual(timely, ["2026-11-27"])
        historical = [{"track": "HISTORICAL", "lane": ARM, "scope": "OWN", "status": "DEVELOPMENT_DIAGNOSTIC_ONLY",
                       "jointly_valid_days": 40, "label_coverage": .95}]
        self.assertEqual(review_state(historical)["OWN"]["jointly_valid_days"], 0)
        self.assertEqual(review_state(historical)["COMMON7"]["status"], "INSUFFICIENT_PROSPECTIVE_DATA")

    def test_freeze_idempotent_and_refuses_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"training_frozen_at": at(24).isoformat(), "weights_sha256": "frozen"}
            with patch("ashare.p58_group_c.contract", return_value=({}, spec)):
                first = freeze(root, root, root, root, lambda: at(25))
                self.assertEqual(freeze(root, root, root, root, lambda: at(26)), first)
            with patch("ashare.p58_group_c.contract", return_value=({}, spec | {"weights_sha256":"changed"})):
                with self.assertRaises(ValueError):
                    freeze(root, root, root, root, lambda: at(26))

    def test_chronology_is_strict_and_replay_never_promotes(self):
        policy = {"frozen_at":at(25, 14).isoformat(), "training_frozen_at":at(24, 15).isoformat()}
        self.assertEqual(track("TIMELY", "2026-11-25", at(25), at(25), policy, True, True),
                         "TIMELY_NONPROSPECTIVE")
        self.assertEqual(track("TIMELY", "2026-11-26", at(26), at(26), policy, True, True), "PROSPECTIVE")
        self.assertEqual(track("HISTORICAL", "2026-11-26", at(27), at(27), policy, True, True), "HISTORICAL")
        self.assertEqual(track("TIMELY", "2026-11-26", at(26), at(26), policy, True, False),
                         "TIMELY_NONPROSPECTIVE")
        with self.assertRaises(ValueError):
            track("TIMELY", "2026-11-26", at(26), at(27, 0), policy, True, True)

    def test_preclose_never_captures_and_abstentions_do_not_score(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            self.assertEqual(capture(path, path, path, path, path, clock=lambda: at(25, 17))["status"], "DEFERRED")
            rows = pd.DataFrame({"stock_code":["000001"], "outcome":["COMPANY_ACTION_INSIDE_CONTEXT"]})
            with patch("ashare.p58_group_c.load_model") as call:
                _, scores, _ = get_scores(path, {}, {}, {}, rows, at(25), "2026-11-25", None)
            call.assert_not_called()
            self.assertEqual(scores, {})

    def test_standardized_inputs_mask_matches_training(self):
        n = len(P54_FEATURES)
        rows = pd.DataFrame({"stock_code":["000001","000002"], "outcome":["SCORED","SCORED"], "missing_factors":""})
        policy = {"mean":[1.] * n, "scale":[2.] * n, "minimum_market_stocks":100,"weights_sha256":"w"}
        info = {"weights_sha256":"w", "architecture":{"batch_size":64}}
        key = np.asarray([[0, 1], [2, 3]])
        features = np.full((4, n), 5.)
        with (patch("ashare.p58_group_c.verify_digest"),
              patch("ashare.p58_group_c.fetch_panel_features", return_value=(rows, None, key, None, rows.stock_code.tolist())),
              patch("ashare.p58_group_c.factor_sequences", return_value=(features, np.array([True,False]), key,
                    dict.fromkeys(("feature_rows","nonfinite_by_factor","last_source_session")), ["","missing"])),
              patch("ashare.p58_group_c.load_model", return_value=(None,info)),
              patch("ashare.p58_group_c.predict", return_value=np.array([.125])) as score):
            checked, scores, _ = get_scores(Path("."), info, policy, {"risk_events_sha256":"risk"}, rows,
                                            at(26), "2026-11-26", object())
        self.assertEqual(scores, {"000001":.125})
        self.assertEqual(checked.outcome.iloc[1], "FACTOR_SEQUENCE_INCOMPLETE")
        expected = np.asarray([2.] * 10 + [0.] * 4 + [2.,2.,0.,0.])
        np.testing.assert_array_equal(score.call_args.args[1][0], np.stack([expected, expected]))

    def test_seven_way_intersection_precedes_maturity_and_reports_own(self):
        rows = pd.DataFrame({"stock_code":["000001","000002","000003"],"score":[3.,2.,1.],
                             "maturity_status":["LABEL_PENDING","MATURE_LABEL","MATURE_LABEL"],
                             "target_return":[np.nan,.01,.03]})
        lanes = (*LANES, ARM)
        cohorts = {k:rows.copy() for k in lanes}
        cohorts[ARM] = rows.iloc[:2]
        specs = {k:{"rows":len(f), "abstentions":3-len(f), "weights_sha256":k} for k,f in cohorts.items()}
        rec = {"trade_date":"2026-11-26","track":"PROSPECTIVE","formal_session":True}
        daily = compare_day(rec, cohorts, specs, {rec["trade_date"]:0}, [rec["trade_date"]], {})
        self.assertEqual(len(daily), 14)
        paired = [d for d in daily if d["scope"] == "COMMON7"]
        self.assertTrue(all(d["predictions"] == 2 and d["labels"] == 1 for d in paired))
        self.assertTrue(all(d["top20_net_excess"] is None for d in paired))
        self.assertEqual(next(d for d in daily if d["lane"] == ARM and d["scope"] == "OWN")["prediction_coverage"],2/3)


if __name__ == "__main__":
    unittest.main()
