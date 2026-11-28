import csv
from pathlib import Path
import tempfile
import unittest

from ashare.p66_daily_report import (METHODS, MISSING, VERSION, calendar_review, capture_models, nightly_results,
                                     publish, scheduler, scope_metrics)
from ashare.p4_action_pipeline import atomic_json


class DailyReportTests(unittest.TestCase):
    def test_missing_session_only_blocks_intersecting_mature_windows(self):
        obs = {"status": "READY_FOR_REVIEW", "model_sha256": "v"}
        rules = {"minimum_joint_prospective_sessions": 1, "minimum_mature_label_coverage": .8}
        row = {"lane": "rank", "track": "PROSPECTIVE", "scope": "OWN", "support_group": "ALL",
               "captured": "True", "formal": "True", "signal_matured": "True", "model_sha256": "v",
               "predictions": "20", "labels": "20", "rank_ic": "0.1", "top20_full": "True",
               "day": "2026-11-24", "exit_date": "2026-11-26"}
        plan = {"status": "CURRENT", "days": {"2026-11-24": {"status": "FORMAL_OBSERVED"},
                                               "2026-11-25": {"status": "SCHEDULED_OPEN_MISSING_FORMAL"},
                                               "2026-11-26": {"status": "FORMAL_OBSERVED"},
                                               "2026-11-27": {"status": "FORMAL_OBSERVED"}}}
        review = calendar_review(obs, [row], "rank", "OWN", plan, rules)
        self.assertEqual(review["status"], "CALENDAR_REVIEW_PENDING")
        self.assertEqual(review["affected_windows"][0]["unverified_days"], ["2026-11-25"])
        review = calendar_review(obs, [row, row | {"day": "2026-11-26", "exit_date": "2026-11-27"}],
                                 "rank", "OWN", plan, rules)
        self.assertEqual(review["status"], "READY_FOR_REVIEW_UNAFFECTED_WINDOWS")
        absent = calendar_review(obs, [row], "rank", "OWN", {"status": MISSING}, rules)
        self.assertEqual(absent["status"], "CALENDAR_REVIEW_PENDING")

    def test_forward_only_same_version_and_unique_days(self):
        entry = {"track": "PROSPECTIVE", "lane": "group_c", "scope": "OWN", "support_group": "ALL",
                 "model_sha256": "frozen", "rank_ic_days": 0, "mature_label_sessions": 0}
        row = entry | {"formal": "True", "captured": "True", "day": "2026-11-27"}
        rows = [row, row, row | {"track": "HISTORICAL", "day": "2026-11-26"},
                row | {"captured": "False", "day": "2026-11-25"}]
        result = scope_metrics({"summary": [entry, entry | {"model_sha256": "old"}]}, rows, "group_c", "OWN", "frozen")
        self.assertEqual(result["published_days"], 1)
        self.assertEqual(result["mature_label_sessions"], 0)
        self.assertEqual(scope_metrics(None, [], "group_c", "OWN", None)["status"], MISSING)

    def test_incomplete_or_failed_step_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "schedule-logs/2026-11-27"
            folder.mkdir(parents=True)
            (folder / "post.first.log").write_text("2026-11-27T18:10 post START\n2026-11-27T18:20 post EXIT=1\n")
            (folder / "post.last.log").write_text("2026-11-27T19:10 monitor START\n")
            post = scheduler(root, "2026-11-27")["post"]
            self.assertEqual([r["exit"] for r in post], [1, None])
            log = root / "nightly.log"
            log.write_text("2026-11-26T23:35 frozen_exit=0 research_exit=0\n"
                           "2026-11-27T18:20 frozen_exit=2 research_exit=0\n")
            self.assertEqual(nightly_results(log, "2026-11-27")[0]["frozen_exit"], 2)

    def test_historical_prediction_cannot_enter_daily_timely_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            atomic_json(path, {"trade_date": "2026-11-27", "capture_kind": "HISTORICAL"})
            with self.assertRaisesRegex(ValueError, "capture identity"):
                capture_models(path, "2026-11-27", ("group_c",), {})

    def test_revisions_preserve_zero_and_missing_and_cumulative_unique_rows(self):
        def report(day, clock):
            empty = {k: {"status": MISSING} for k in METHODS}
            return {"version": VERSION, "generated_at": clock, "generation_date": clock[:10],
                    "market": {"date": day, "stocks": 0}, "prediction_dates": [], "candidate_count": None,
                    "schedule": {}, "completed_nightly_runs": [], "issues": [], "snapshots": [], "models": empty,
                    "evaluation": {k: {"own": {"status": MISSING}, "common7": {"status": MISSING}}
                                   for k in METHODS},
                    "diagnostic": {"status": MISSING}, "accounts": {}, "source_files": {}}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = report("2026-11-26", "2026-11-26T21:00:00+08:00")
            pointer = publish(out, first)
            original = pointer.resolve()
            self.assertEqual(publish(out, first | {"generated_at": "2026-11-26T22:00:00+08:00"}).resolve(), original)
            publish(out, first | {"issues": ["MISSING_FUND"], "generated_at": "2026-11-26T23:00:00+08:00"})
            self.assertNotEqual(pointer.resolve(), original)
            self.assertTrue((original / "daily_research.json").exists())
            publish(out, report("2026-11-27", "2026-11-28T08:00:00+08:00"))
            with (pointer / "research_progress.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 28)
            self.assertEqual(len(set((r["formal_date"], r["model"], r["scope"]) for r in rows)), 28)
            (pointer / "research_progress.csv").write_text("damaged")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                publish(out, report("2026-11-27", "2026-11-28T08:00:01+08:00"))


if __name__ == "__main__":
    unittest.main()
