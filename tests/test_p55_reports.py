"""P55 comparison and maturity guards for immutable forecasting ledgers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ashare.p4_action_pipeline import read_json
from ashare.p4_subset_inputs import hash_file
from ashare.p55_reports import (CUMULATIVE_REPORT, DAILY_REPORT, LANES, PANEL_LANES, REPORT_AUDIT,
                                daily_markdown, disjoint_holds, evaluate_days, hac_mean,
                                ranked_daily, read_panel, run, summarize)


CALENDAR = [d.date().isoformat() for d in pd.bdate_range("2026-08-03", periods=32)]


def forecasts(lane, stock_count=25, track="PROSPECTIVE", day=CALENDAR[1], missing=()):
    count = np.arange(stock_count)
    scores = np.zeros(stock_count) if lane == "equal_weight" else count
    target = count / 1000.0
    target[list(missing)] = np.nan
    return pd.DataFrame({"track": track, "trade_date": day, "objective": lane,
                         "stock_code": [f"{i:06d}" for i in count], "support_group": ["20+"] * stock_count,
                         "score": scores, "target_return": target,
                         "maturity_status": ["MATURE_LABEL" if np.isfinite(i) else "NO_LABEL" for i in target]})


def capture(lane, day=CALENDAR[1], track="PROSPECTIVE", forecast_count=25, candidates=27, formal=True):
    return {"lane": lane, "day": day, "track": track, "captured": True, "formal": formal,
            "model_sha256": lane + "-sha", "version": lane + "-version", "freeze_day": CALENDAR[0],
            "candidates": candidates, "abstentions": candidates - forecast_count,
            "abstention_reasons": {"TEST_ABSTAINED": candidates - forecast_count},
            "status": "SCORED" if forecast_count else "ABSTAINED",
            "portfolio_kind": "EQUAL_WEIGHT" if lane == "equal_weight" else "RANKED"}


class ReportsTests(unittest.TestCase):
    def test_common_matches_frozen_forecasts_before_any_label_result(self):
        reads = [forecasts(lane, 24 if lane == "factors18" else 25, missing=(23,)) for lane in LANES]
        frames = pd.concat(reads, ignore_index=True)
        captures = [capture(l, forecast_count=24 if l == "factors18" else 25) for l in LANES]
        records = evaluate_days(captures, frames, CALENDAR[:9])
        day = [r for r in records if r["day"] == CALENDAR[1] and r["scope"] == "COMMON"
               and r["support_group"] == "ALL"]
        self.assertEqual(len(day), 6)
        self.assertTrue(all(r["predictions"] == 24 and r["labels"] == 23 for r in day))
        a = next(r for r in day if r["lane"] == "absolute")
        self.assertEqual(a["forecast_excluded_by_common_scope"], 1)
        self.assertEqual(a["top20_stocks"][0], "000023")
        self.assertFalse(a["top20_full"])
        self.assertIsNone(a["top20_net_excess"])

    def test_equal_weight_reports_full_pool_without_arbitrary_rank(self):
        f = forecasts("equal_weight")
        actual = ranked_daily(f, f, True, {})
        self.assertIsNone(actual["rank_ic"])
        self.assertEqual(actual["top20_stocks"], [])
        self.assertFalse(actual["top20_full"])
        self.assertAlmostEqual(actual["portfolio_net_excess"], -.002)

    def test_fewer_than_twenty_picks_cannot_claim_complete_day(self):
        f = forecasts("absolute", 19)
        actual = ranked_daily(f, f, False, {})
        self.assertEqual(actual["top20_labels"], 19)
        self.assertFalse(actual["top20_full"])
        self.assertIsNone(actual["top20_return"])
        self.assertTrue(actual["top10_full"])

    def test_only_same_track_can_form_common_and_empty_days_remain_visible(self):
        reads = [forecasts(l) for l in LANES if l not in ("rank", "factors18")]
        reads.append(forecasts("factors18", track="TIMELY_NONPROSPECTIVE"))
        records = [capture(l, track="TIMELY_NONPROSPECTIVE" if l == "factors18" else "PROSPECTIVE",
                           forecast_count=0 if l == "rank" else 25) for l in LANES]
        rows = evaluate_days(records, pd.concat(reads), CALENDAR[:9])
        self.assertFalse(any(r["scope"] == "COMMON" for r in rows))
        summary = summarize(rows)
        rank = next(r for r in summary if r["lane"] == "rank" and r["track"] == "PROSPECTIVE"
                    and r["support_group"] == "ALL")
        self.assertEqual(rank["zero_prediction_days"], [CALENDAR[1]])
        self.assertIn(CALENDAR[2], rank["missing_formal_days"])
        self.assertEqual(rank["status"], "INSUFFICIENT_PROSPECTIVE_DATA")

    def test_readiness_uses_distinct_joint_mature_dates_and_keeps_overlap_statistic(self):
        dates = CALENDAR[1:21]
        records = [capture("absolute", day=d) for d in dates]
        tables = pd.concat([forecasts("absolute", day=d) for d in dates], ignore_index=True)
        days = evaluate_days(records, tables, CALENDAR)
        result = next(r for r in summarize(days) if r["support_group"] == "ALL")
        self.assertEqual(result["jointly_valid_days"], 20)
        self.assertEqual(result["rank_ic_days"], 20)
        self.assertEqual(result["full_top20_days"], 20)
        self.assertLess(result["disjoint_joint_hold_windows"], 20)
        self.assertEqual(result["status"], "READY_FOR_REVIEW")
        days = [r for r in days if r["day"] != dates[-1]]
        later = next(r for r in summarize(days) if r["support_group"] == "ALL")
        self.assertEqual(later["status"], "INSUFFICIENT_PROSPECTIVE_DATA")

    def test_hac_uses_market_gap_instead_of_missing_index_adjacency(self):
        nearby = [(i, float(i % 2)) for i in range(20)]
        far = [(i*8, y) for i, y in nearby]
        close, separate = hac_mean(nearby), hac_mean(far)
        self.assertEqual(close["n"], separate["n"])
        self.assertEqual(close["mean"], separate["mean"])
        self.assertNotEqual(close["standard_error"], separate["standard_error"])
        self.assertIsNone(hac_mean(nearby[:4])["approximate_95_percent_interval"])
        first = {"entry_date": CALENDAR[0], "exit_date": CALENDAR[5]}
        shared_endpoint = {"entry_date": CALENDAR[5], "exit_date": CALENDAR[9]}
        independent = {"entry_date": CALENDAR[6], "exit_date": CALENDAR[10]}
        self.assertEqual(disjoint_holds([first, shared_endpoint, independent]), 2)

    def test_markdown_and_empty_run_keep_snapshot_digests(self):
        if importlib.util.find_spec("polars") is None:
            self.skipTest("Polars is provided by ubuntu92")
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            export = root / "report"
            with patch("ashare.p55_reports.AShareDailyLoader"), patch("ashare.p55_reports.formal_tail",
                    return_value=(CALENDAR[:1], pd.DataFrame(columns=["stock_code", "trade_date"]))):
                result = run(root / "ledger", export, None)
            self.assertEqual(result["label_rows"], 0)
            self.assertEqual(read_json(export / REPORT_AUDIT)["files"][DAILY_REPORT], hash_file(export / DAILY_REPORT))
            self.assertEqual(read_json(export / REPORT_AUDIT)["files"][CUMULATIVE_REPORT], hash_file(export / CUMULATIVE_REPORT))
            self.assertEqual(read_panel(root / "ledger", None)[0], [])
            self.assertIn("暂无预测记录", daily_markdown([], [], CALENDAR[:1]))
