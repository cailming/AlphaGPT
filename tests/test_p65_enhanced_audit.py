import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from ashare.p4_transformer_train import scale_sequences
from ashare.p54_factors import P54_FEATURES
from ashare.p56_ablation import active, masked_sequences
from ashare.p4_action_pipeline import atomic_json
from ashare.p4_subset_inputs import hash_file
from ashare.p55_daily import parse_clock
from ashare.p65_enhanced_audit import (
    BASE_REPORT_FILES, QUOTE_TRACK, REPORT_FILE, SCORES_FILE, VERSION,
    active_standardized, checked_report, eligibility, patch_market_rows, report_windows,
    require_scheduled_day, run_day, score_difference,
)


class EnhancedEligibilityTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.asarray([name in active("group_c") for name in P54_FEATURES])
        self.sequences = np.array([[0, 1], [2, 3], [4, 5]])
        self.matrix = np.arange(6 * len(P54_FEATURES), dtype=float).reshape(6, -1)
        self.mean = np.linspace(0, .25, len(P54_FEATURES))
        self.scale = np.linspace(1., 2., len(P54_FEATURES))

    def test_finite_counterfactual_matches_frozen_scale_then_mask(self):
        old = masked_sequences(scale_sequences(self.matrix, self.sequences, self.mean, self.scale), "group_c")
        new = active_standardized(self.matrix, self.sequences, self.mask, self.mean, self.scale)
        np.testing.assert_array_equal(old, new)
        self.assertTrue(score_difference([.04, .09], [.04, .09])["bitwise_equal"])

    def test_active_relative_strength_still_rejects(self):
        self.matrix[0, P54_FEATURES.index("market_ret_20")] = np.nan
        self.matrix[2, P54_FEATURES.index("stock_relative_ret_20")] = np.nan
        missing, strict, active_only = eligibility(self.matrix, self.sequences, self.mask)
        np.testing.assert_array_equal(strict, [False, False, True])
        np.testing.assert_array_equal(active_only, [True, False, True])
        actual = active_standardized(self.matrix, self.sequences[active_only], self.mask, self.mean, self.scale)
        self.assertTrue(np.isfinite(actual).all())
        self.assertFalse(actual[:, :, ~self.mask].any())
        with self.assertRaises(ValueError):
            active_standardized(self.matrix, self.sequences, self.mask, self.mean, self.scale)
        dates = np.tile(["2026-11-12", "2026-11-13"], (3, 1))
        self.assertEqual(report_windows(missing, dates, self.mask), [
            {"feature": "market_ret_20", "input_date": "2026-11-12", "active_c": False, "missing_stock_days": 1},
            {"feature": "stock_relative_ret_20", "input_date": "2026-11-12", "active_c": True, "missing_stock_days": 1},
        ])

    def test_research_adapter_leaves_formal_rows_untouched_and_rejects_overlap(self):
        original = pd.DataFrame({
            "stock_code": ["600001", "600002", "600002"],
            "trade_date": [date(2026, 10, 15), date(2026, 10, 15), date(2026, 10, 16)],
            "close": [10., 11., 11.], "not_billing": [1, 1, 1],
        })
        patch = pd.DataFrame({
            "stock_code": ["600001"], "trade_date": ["2026-10-16"], "open": [10.], "high": [11.], "low": [9.],
            "close": [10.2], "volume": [100.], "amount": [1000.], "source_billing_id": [1],
            "source_quote_ticktime": ["15:02:00"], "previous_formal_close": [10.],
            "source_kind": ["OBSERVED_POST_CLOSE_QUOTE"],
        })
        scenario = patch_market_rows(original, patch)
        self.assertEqual(len(scenario), 4)
        pd.testing.assert_frame_equal(scenario.iloc[:3][original.columns], original)
        self.assertEqual(scenario.iloc[-1].stock_code, "600001")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            patch_market_rows(original, patch.assign(stock_code="600002", previous_formal_close=11.))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            patch_market_rows(original, patch.assign(source_quote_ticktime="14:58:00"))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            patch_market_rows(original, patch.assign(previous_formal_close=9.))

    def test_scheduled_day_requires_current_formal_session_after_close(self):
        day = date(2026, 11, 27)
        after = parse_clock("2026-11-27T18:25:00+08:00")
        require_scheduled_day(day, after, day)
        with self.assertRaisesRegex(ValueError, "Beijing after-close"):
            require_scheduled_day(day, parse_clock("2026-11-27T17:59:59+08:00"))
        with self.assertRaisesRegex(ValueError, "Beijing after-close"):
            require_scheduled_day(date(2026, 11, 26), after)
        with self.assertRaisesRegex(ValueError, "formal market"):
            require_scheduled_day(day, after, date(2026, 11, 26))

    def test_scheduled_duplicate_accepts_same_hashes_and_rejects_corruption(self):
        day = "2026-11-27"
        observed = "2026-11-27T18:25:00+08:00"
        parent = {"legacy": "legacy-hash", "panel": "panel-hash"}
        compare_day = "2026-09-23"
        lineage = {"version": VERSION, "track": QUOTE_TRACK,
                   "research_patch": {"artifact_sha256": "patch-hash", "manifest_sha256": "source-hash"}}
        with TemporaryDirectory() as temp:
            out = Path(temp)
            files = {}
            for name in BASE_REPORT_FILES | {SCORES_FILE}:
                (out / name).write_text(f"{name}\n", encoding="utf-8")
                files[name] = hash_file(out / name)
            manifest = out / REPORT_FILE
            report = lineage | {"forward_eligible": False, "observed_at": observed, "files": files,
                                "latest_formal": {"date": day, "parent_manifest_sha256": parent},
                                "comparison": {"date": compare_day}}
            atomic_json(manifest, report)
            now = parse_clock("2026-11-27T18:30:00+08:00")
            manifest_hash = hash_file(manifest)
            self.assertEqual(checked_report(out, day, now, lineage, parent, compare_day), manifest)
            self.assertEqual(hash_file(manifest), manifest_hash)
            with self.assertRaisesRegex(ValueError, "lineage"):
                checked_report(out, day, now, lineage, parent | {"panel": "different"}, compare_day)
            with self.assertRaisesRegex(ValueError, "lineage"):
                checked_report(out, day, now, lineage | {"research_patch": {"artifact_sha256": "other"}},
                               parent, compare_day)
            (out / SCORES_FILE).write_text("damaged\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                checked_report(out, day, now, lineage, parent, compare_day)

    def test_zero_eligible_keeps_every_abstention_and_both_parent_hashes(self):
        rows = pd.DataFrame({"outcome": ["COMPANY_ACTION_INSIDE_CONTEXT", "NO_CONTIGUOUS_FEATURE_SEQUENCE"]})
        original = {"risk_events_sha256": "risk-hash"}
        peer = {"models": {"factors18": {"rows": 0}}}
        trained = {"selection_last_exit": "2026-09-01"}
        policy = {"minimum_market_stocks": 100, "input_mask": self.mask}
        with TemporaryDirectory() as temp:
            legacy, panel = Path(temp) / "legacy", Path(temp) / "panel"
            for folder, manifest in ((legacy, original), (panel, peer)):
                folder.mkdir()
                atomic_json(folder / "manifest.json", manifest)
            pair = original, peer, legacy, panel
            with (patch("ashare.p65_enhanced_audit.verify_digest"),
                  patch("ashare.p65_enhanced_audit.legacy_details"),
                  patch("ashare.p65_enhanced_audit.fetch_panel_features",
                        return_value=(rows, None, np.empty((0, 16)), rows, [])),
                  patch("ashare.p65_enhanced_audit.factor_sequences") as build):
                result, dates, candidates, scores = run_day(None, None, "2026-11-27", None,
                                                            (trained, trained, policy), matched_parents=pair)
            build.assert_not_called()
            self.assertEqual(result["parent_manifest_sha256"],
                             {"legacy": hash_file(legacy / "manifest.json"), "panel": hash_file(panel / "manifest.json")})
            self.assertEqual(result["status"], "ABSTAINED")
            self.assertEqual((result["base_eligible"], result["full18_real_scores"], result["group_c_real_scores"]), (0, 0, 0))
            self.assertEqual(result["frozen_abstentions"], dict.fromkeys(rows.outcome, 1))
            self.assertEqual(result["active_only_diagnostic_abstentions"], result["frozen_abstentions"])
            self.assertEqual((dates, candidates, scores), ([], [], []))
