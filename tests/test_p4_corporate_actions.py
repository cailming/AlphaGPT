"""Fail-closed tests for company-action-aware P4 economic labels."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "ashare/p4_corporate_actions.py"
spec = importlib.util.spec_from_file_location("p4_corporate_actions", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixtures():
    cal = pd.bdate_range("2025-07-01", periods=70)
    sids = [25, 26, 27, 28, 29, 36, 38, 41, 42]
    rows = []
    bars = []
    for i, day in enumerate(cal):
        for code in ("002594", "000001"):
            open_price = 100.0
            if code == "002594":
                # ex-date at cal[30], one pre-event share -> three shares
                open_price = 33.0 if i >= 30 else 100.0
            bars.append({"stock_code": code, "trade_date": day,
                         "open": open_price, "not_billing": 1})
    for code in ("002594", "000001"):
        for idx in sids:
            entry, exit_ = idx+1, idx+6
            e = 33.0 if code == "002594" and entry >= 30 else 100.0
            x = 33.0 if code == "002594" and exit_ >= 30 else 100.0
            rows.append({"stock_code": code, "trade_date": cal[idx],
                         "session_id": idx, "label_end_session": exit_,
                         "target_return": x/e-1})
    actions = pd.DataFrame([{"stock_code": "002594", "ex_date": cal[30],
        "published_on": cal[20], "share_multiplier": 3.0,
        "cash_per_old_share": 3.974,
        "source_url": "https://disc.static.szse.cn/file.pdf",
        "source_id": "BYD-2025-047"}])
    coverage = pd.DataFrame([{"stock_code": "002594",
        "start_date": cal[0], "end_date": cal[55], "verified_on": cal[56],
        "source_url": "https://example.org/full_feed", "source_id": "COVERAGE-BYD",
        "status": "COMPLETE_VERIFIED"}])
    # 25 gives feature_lookback 21 and action at 30 -> feature contamination for 30..50
    return pd.DataFrame(rows), pd.DataFrame(bars), cal, actions, coverage


class CorporateActionTests(unittest.TestCase):
    def test_unknown_coverage_fails_closed_even_when_known_event_corrects_label(self):
        selected, bars, cal, events, cov = fixtures()
        summary, samples, daily = mod.audit_corporate_actions(
            selected, bars, cal, events, cov.head(0),
            feature_lookback=21, test_start="2026-01-01")
        self.assertEqual(summary["training_label_allowed_rows"], 0)
        self.assertEqual(summary["recomputed_original_labels_matching"], len(selected))
        case = samples.loc[(samples.stock_code == "002594") &
                           (samples.session_id == 25)].iloc[0]
        self.assertAlmostEqual(case.target_return, -.67)
        self.assertAlmostEqual(case.illustrative_known_event_return, (3*33+3.974)/100-1)
        self.assertTrue(np.isnan(case.economic_target_return))
        self.assertEqual(case.reason, "UNKNOWN_ACTION_COVERAGE")

    def test_explicit_coverage_only_and_feature_window_quarantine(self):
        s, bars, cal, ev, coverage = fixtures()
        info, out, _ = mod.audit_corporate_actions(
            s, bars, cal, ev, coverage, feature_lookback=21,
            test_start="2026-01-01")
        before = out.loc[(out.stock_code == "002594") & (out.session_id == 25)].iloc[0]
        self.assertTrue(before.training_label_allowed)
        self.assertAlmostEqual(before.economic_target_return, (3*33+3.974)/100-1)
        contaminated = out.loc[(out.stock_code == "002594") &
                               (out.session_id == 36)].iloc[0]
        self.assertFalse(contaminated.training_label_allowed)
        self.assertEqual(contaminated.reason, "UNADJUSTED_FEATURE_ACTION_WINDOW")
        # Uncovered stock with no reported actions is NOT safe for training.
        other = out.loc[out.stock_code == "000001"]
        self.assertTrue(other.training_label_allowed.eq(False).all())
        self.assertTrue(other.economic_target_return.isna().all())
        self.assertGreater(info["known_action_feature_rows"], 0)

    def test_entry_day_ex_date_does_not_create_entitlement(self):
        s, bars, cal, ev, coverage = fixtures()
        # T=cal[29], entry is the ex-date cal[30]; buyer is too late.
        _, result, _ = mod.audit_corporate_actions(
            s, bars, cal, ev, coverage, feature_lookback=21,
            test_start="2026-01-01")
        row = result.loc[(result.stock_code == "002594") &
                         (result.session_id == 29)].iloc[0]
        self.assertEqual(row.event_in_label_count, 0)
        self.assertEqual(row.raw_label_recomputed, 0)

    def test_cash_is_not_reinvested_before_second_action(self):
        s, bars, cal, ev, coverage = fixtures()
        # Fix prices to 100 on the first event and 20 after second for example.
        bars.loc[(bars.stock_code == "002594") &
                 (bars.trade_date >= cal[30]), "open"] = 33.0
        bars.loc[(bars.stock_code == "002594") &
                 (bars.trade_date >= cal[32]), "open"] = 20.0
        second = ev.copy().iloc[[0]]
        second.loc[:, "ex_date"] = cal[32]
        second.loc[:, "share_multiplier"] = 2.0
        second.loc[:, "cash_per_old_share"] = 1.0
        second.loc[:, "source_id"] = "SECOND"
        ev2 = pd.concat([ev, second], ignore_index=True)
        # Signal at 27 -> entry28, exit33 crosses both.
        s2 = s.loc[(s.stock_code == "002594") & (s.session_id == 27)].copy()
        s2["target_return"] = 20.0 / 100.0-1
        info, out, _ = mod.audit_corporate_actions(
            s2, bars, cal, ev2, coverage, feature_lookback=21,
            test_start="2026-01-01")
        expected = (6*20 + 3.974 + 3*1.0)/100 - 1
        self.assertAlmostEqual(out.iloc[0].economic_target_return, expected)
        self.assertEqual(out.iloc[0].event_in_label_count, 2)

    def test_recomputed_label_disagreement_blocks_training(self):
        s, bars, cal, ev, cov = fixtures()
        broken = s.copy()
        broken.loc[(broken.stock_code == "002594") &
                   (broken.session_id == 25), "target_return"] += .01
        info, out, _ = mod.audit_corporate_actions(
            broken, bars, cal, ev, cov, feature_lookback=21,
            test_start="2026-01-01")
        mismatch = out.loc[(out.stock_code == "002594") &
                           (out.session_id == 25)].iloc[0]
        self.assertEqual(mismatch.reason, "RAW_LABEL_MISMATCH")
        self.assertTrue(np.isnan(mismatch.economic_target_return))
        self.assertLess(info["recomputed_original_labels_matching"], len(s))

    def test_rejects_inspected_test_prices_and_signals(self):
        s, bars, cal, ev, cov = fixtures()
        with self.assertRaisesRegex(ValueError, "TEST dates"):
            mod.audit_corporate_actions(s, bars, cal, ev, cov,
                feature_lookback=21, test_start=str(cal[-1].date()))
        s2 = s.copy()
        s2.loc[s2.index[0], "trade_date"] = pd.Timestamp("2026-07-01")
        with self.assertRaisesRegex(ValueError, "TEST signal"):
            mod.audit_corporate_actions(s2, bars, cal, ev, cov,
                feature_lookback=21, test_start="2026-06-12")

    def test_source_loaders_and_fail_closed_manifest(self):
        with TemporaryDirectory() as d:
            act = Path(d) / "events.csv"
            cov = Path(d) / "coverage.csv"
            act.write_text("stock_code,ex_date,published_on,share_multiplier,cash_per_old_share,source_url,source_id\n"
                "002594,2025-07-29,2025-07-22,3,3.974,https://disc.static.szse.cn/x.pdf,BYD-2025-047\n")
            cov.write_text("stock_code,start_date,end_date,verified_on,source_url,source_id,status\n")
            self.assertEqual(mod.load_actions(act).iloc[0].share_multiplier, 3.0)
            self.assertTrue(mod.load_coverage(cov).empty)
            cov.write_text("stock_code,start_date,end_date,verified_on,source_url,source_id,status\n"
                "002594,2025-01-01,2025-12-31,2026-01-02,https://example.com/x,feed,UNKNOWN\n")
            with self.assertRaisesRegex(ValueError, "COMPLETE_VERIFIED"):
                mod.load_coverage(cov)
            act.write_text("stock_code,ex_date,published_on,share_multiplier,cash_per_old_share,source_url,source_id\n"
                "002594,2025-07-29,2025-07-22,0,3.974,https://disc.static.szse.cn/x.pdf,BYD-2025-047\n")
            with self.assertRaisesRegex(ValueError, "Nonpositive"):
                mod.load_actions(act)


if __name__ == "__main__":
    unittest.main()
