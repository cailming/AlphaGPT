"""Tests for the daily ledger's time, coverage and immutable publication gates."""
from datetime import date, datetime
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ashare.p55_daily import (
    CHINA,
    OBJECTIVES,
    PREDICTION_COLUMNS,
    VERSION,
    CaptureDeadlineExceeded,
    checked_manifest,
    daily,
    is_timely,
    merge_observed_interval,
    observation_intervals,
    parse_clock,
    pending_stocks,
    prediction_id,
    publish_day,
    support_group,
)


def shanghai(y, m, d, h=19, minute=0):
    return datetime(y, m, d, h, minute, tzinfo=CHINA)


def metadata(frozen="2026-11-24T06:00:00Z"):
    return {name: {"version": "V2", "weights_sha256": str(i) * 64,
                   "evaluation_plan_sha256": "a" * 64, "model_metadata_sha256": "b" * 64,
                   "frozen_at_utc": frozen}
            for i, name in enumerate(OBJECTIVES, 1)}


class DailyGateTests(unittest.TestCase):
    def test_after_close_same_day_and_never_explicit_asof(self):
        start, end = shanghai(2026, 11, 25), shanghai(2026, 11, 25, 23)
        self.assertTrue(is_timely(None, start, end))
        self.assertFalse(is_timely(start.date(), start, end))
        self.assertFalse(is_timely(None, shanghai(2026, 11, 25, 17, 59), end))
        self.assertFalse(is_timely(None, start, shanghai(2026, 11, 26, 0)))
        with self.assertRaises(ValueError):
            parse_clock("2026-11-26T19:00:00")

    def test_group_cutoffs_and_ids(self):
        self.assertEqual([support_group(n) for n in (0, 4, 5, 19, 20, 99)], ["0-4", "0-4", "5-19", "5-19", "20+", "20+"])
        with self.assertRaises(ValueError):
            support_group(-1)
        key = ("TIMELY", "2026-11-25", "absolute", "a", "000001")
        self.assertEqual(prediction_id(*key), prediction_id(*key))
        self.assertNotEqual(prediction_id(*key), prediction_id("HISTORICAL", *key[1:]))
        self.assertNotEqual(prediction_id(*key), prediction_id(*key[:2], "rank", *key[3:]))

    def test_zero_formal_probe_does_not_publish_day(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with patch("ashare.p55_daily.formal_row_count", return_value=0):
                record = daily(Path("source"), Path("models"), Path("seed"), out,
                               engine=object(), clock=lambda: shanghai(2026, 11, 25, 19))
            self.assertEqual(record["reason"], "NO_SAME_DAY_FORMAL_OHLCV_YET")
            self.assertFalse((out / "timely").exists())
            self.assertEqual(len(list((out / "probes").glob("*/*.json"))), 1)

    def test_pre_close_probe_does_not_read_production(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("ashare.p55_daily.formal_row_count") as probe:
                record = daily(Path("source"), Path("models"), Path("seed"), Path(temp),
                               engine=object(), clock=lambda: shanghai(2026, 11, 25, 17, 59))
            self.assertEqual(record["status"], "DEFERRED")
            probe.assert_not_called()

    def test_merge_does_not_extend_through_partial_provider_response(self):
        old = {"covered_from": "2026-10-01", "covered_through": "2026-11-23", "actions": []}
        partial = merge_observed_interval(old, "2026-11-24", "2026-11-24", ["2026-11-24"], False)
        self.assertEqual(partial["covered_through"], "2026-11-23")
        self.assertEqual(partial["actions"], [("2026-11-24", "2026-11-24")])
        self.assertIsNone(merge_observed_interval(None, "2026-11-24", "2026-11-24", (), False))
        with self.assertRaises(ValueError):
            merge_observed_interval(old, "2026-11-25", "2026-11-25", (), True)
        dual = merge_observed_interval(partial, "2026-11-24", "2026-11-25", ["2026-11-24"], True)
        self.assertEqual(dual["covered_through"], "2026-11-25")
        self.assertEqual(len(dual["actions"]), 1)

    def test_recent_plan_chunks_more_than_220_without_dropping_cohorts(self):
        day = pd.DatetimeIndex(pd.date_range("2026-09-01", periods=56, freq="B"))
        signal = day[-1].date().isoformat()
        prior_end = day[-2].date().isoformat()
        pending = {f"{code:06d}": day[-7].date().isoformat() for code in range(1, 279)}
        old = {"stocks": {code: {"covered_from": "2026-09-01", "covered_through": prior_end,
                                 "actions": []} for code in pending}}
        population = SimpleNamespace(calendar=day,
                                     anchors=pd.DataFrame(columns=["trade_date", "stock_code", "liquidity_rank"]),
                                     features=pd.DataFrame())
        resolved = (np.empty(0, dtype=np.int64), np.empty((0, 16), dtype=np.int64))
        with patch("ashare.p55_daily.sequence_indexes", return_value=resolved):
            plans, stats = observation_intervals(population, signal, 16, old, pending, 400)
        self.assertEqual([len(x["jobs"]) for x in plans], [220, 58])
        self.assertEqual(stats["planned_stocks"], 278)
        self.assertEqual(sum(len(p["jobs"]) for p in plans), len(pending))
        self.assertEqual({p["start"] for p in plans},
                         {(date.fromisoformat(prior_end) + pd.Timedelta(days=1)).isoformat()})

    def test_nightly_does_not_expand_observations_from_the_daily_top_pool(self):
        calendar = pd.bdate_range("2026-08-01", periods=60)
        signal, previous = [day.date().isoformat() for day in calendar[-2:]][::-1]
        rows = pd.DataFrame({"stock_code": ["000001", "000002", "000003"], "trade_date": signal,
                             "liquidity_rank": [1, 2, 3], "session_id": len(calendar) - 1})
        prior = {"stocks": {"000001": {"covered_from": "2026-08-01", "covered_through": previous}}}
        pending = {"000003": calendar[-5].date().isoformat()}
        population = SimpleNamespace(calendar=calendar, anchors=rows, features=pd.DataFrame())
        with patch("ashare.p55_daily.sequence_indexes", return_value=(np.arange(3), None)):
            plans, stats = observation_intervals(population, signal, 16, prior, pending, 400)
        self.assertEqual({item["stock_code"] for p in plans for item in p["jobs"]}, {"000001", "000003"})
        self.assertEqual(stats["price_ready_outside_roster"], 1)


@unittest.skipUnless(importlib.util.find_spec("polars"), "Polars is provided by ubuntu92")
class DailyPublicationTests(unittest.TestCase):
    def test_same_freeze_date_cohort_keeps_evidence_until_t_plus_six(self):
        import polars as pl

        calendar = pd.date_range("2026-11-24", periods=9, freq="B")
        signal = calendar[-1].date().isoformat()
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder)
            past = {calendar[-7].date().isoformat(): "000001", calendar[-8].date().isoformat(): "000002"}
            for day, stock in past.items():
                path = ledger / "timely" / day
                path.mkdir(parents=True)
                pl.DataFrame({"stock_code": [stock]}).write_parquet(path / "predictions.parquet")
            earlier = ledger / "historical" / calendar[-2].date().isoformat()
            earlier.mkdir(parents=True)
            with patch("ashare.p55_daily.checked_manifest", return_value={
                "formal_session": True, "capture_kind": "TIMELY", "status": "SCORED", "prospective_eligible": False,
                "models": {"absolute": {"rows": 1, "predictions_file": "predictions.parquet"}},
            }):
                cohorts = pending_stocks(ledger, calendar, signal)
        self.assertEqual(cohorts, {"000001": calendar[-7].date().isoformat()})

    def publish(self, temp, capture, *, frozen="2026-11-24T06:00:00Z", committed=None, formal=True):
        requested = capture == "HISTORICAL"
        day = "2026-11-25"
        destination = Path(temp) / capture.lower() / day
        record = {"schema_version": VERSION, "trade_date": day, "capture_kind": capture,
                  "capture_started_at": shanghai(2026, 11, 25, 19).isoformat(), "requested_historical": requested,
                  "formal_session": formal, "status": "ABSTAINED" if formal else "NO_FORMAL_SESSION", "models": {},
                  "evidence_snapshot": {"observed_at_utc": "2026-11-25T09:00:00Z"}}
        details = pd.DataFrame({"outcome": ["CAUSE"]}) if formal else pd.DataFrame(columns=["outcome"])
        result = publish_day(destination, record, metadata(frozen), details,
                             {name: pd.DataFrame(columns=PREDICTION_COLUMNS) for name in OBJECTIVES},
                             lambda: committed or shanghai(2026, 11, 25, 21))
        return destination, result

    def test_timely_abstention_is_kept_and_day_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            archive, result = self.publish(temp, "TIMELY")
            self.assertTrue(result["prospective_eligible"])
            self.assertEqual(checked_manifest(archive), result)
            again, reread = self.publish(temp, "TIMELY", frozen="2026-11-26T14:00:00Z")
            self.assertEqual(again, archive)
            self.assertEqual(reread, result)
            self.assertEqual(result["models"]["absolute"]["abstention_reasons"], {"CAUSE": 1})

    def test_explicit_backfill_and_same_day_model_freeze_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            _, history = self.publish(temp, "HISTORICAL")
            _, same_day = self.publish(temp, "TIMELY", frozen="2026-11-25T02:00:00Z")
            self.assertFalse(history["prospective_eligible"])
            self.assertFalse(same_day["prospective_eligible"])
            self.assertEqual(same_day["capture_kind"], "TIMELY")

    def test_technical_session_missing_and_late_commits(self):
        with tempfile.TemporaryDirectory() as temp:
            _, result = self.publish(temp, "TIMELY", formal=False)
            self.assertFalse(result["prospective_eligible"])
            self.assertEqual(result["status"], "NO_FORMAL_SESSION")
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(CaptureDeadlineExceeded):
                self.publish(temp, "TIMELY", committed=shanghai(2026, 11, 26, 0))
            self.assertFalse((Path(temp) / "timely" / "2026-11-25").exists())


if __name__ == "__main__":
    unittest.main()
