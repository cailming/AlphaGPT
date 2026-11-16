"""Offline regression tests for the corporate-action scheduler."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from ashare import p4_action_pipeline as p


class PipelineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.ak = self.root / "ak"
        self.bs = self.root / "bs"
        self.ak.mkdir()
        self.bs.mkdir()
        self.snapshot = self.root / "snapshot"
        self.metadata = self.root / "metadata.json"
        self.output = self.root / "output"
        self.output.mkdir()
        self.controls = self.root / "controls.csv"
        self.controls.write_text("stock_code,ex_date,share_multiplier,cash_per_old_share,source_id\n")
        self.metadata.write_text(json.dumps({"model_version": "P4_RIDGE_V1", "test_start": p.CUTOFF}))
        p.free.save_snapshot(pd.DataFrame({"stock_code": ["000001", "000002"]}),
                             start="2024-12-01", test_start=p.CUTOFF, out=self.snapshot)

    def args(self, command, *flags):
        argv = ["test", command, "--snapshot", str(self.snapshot), "--metadata", str(self.metadata),
                "--ak-root", str(self.ak), "--bao-root", str(self.bs), "--output", str(self.output),
                "--controls", str(self.controls), *flags]
        with patch("sys.argv", argv):
            return p.parse_args()

    def run_pipeline(self, command, callback, *flags):
        with patch.object(p, "Providers") as provider:
            provider.return_value.side_effect = callback
            return p.execute(self.args(command, *flags))

    def first_run(self, callback, *flags, limit="2"):
        return self.run_pipeline("run", callback, "--offset", "0", "--limit", limit,
                                 "--batch-size", "1", "--sina-interval", "0", "--cninfo-interval", "0",
                                 "--baostock-interval", "0", "--backoff", "0", *flags)

    @staticmethod
    def success(_):
        return {"status": "SUCCESS", "response_rows": 0}, []

    def test_dry_run_has_no_output_or_requests(self):
        summary, code = self.run_pipeline("run", self.fail, "--dry-run", "--offset", "0", "--limit", "1")
        self.assertEqual(code, 0)
        self.assertEqual(summary["per_endpoint"][p.BAO]["NOT_FETCHED"], 3)
        self.assertFalse(list(self.output.iterdir()))

    def test_interrupt_resume_and_repeat_never_replay_success(self):
        called = []

        def interrupt(task):
            called.append(task)
            os.kill(os.getpid(), signal.SIGTERM)
            return self.success(task)

        initial, code = self.first_run(interrupt)
        self.assertEqual(code, 2)
        self.assertEqual(initial["effective_successes"], 1)
        self.assertEqual(p.read_json(self.output / "checkpoint.json")["state"], "INTERRUPTED")

        def record(task):
            called.append(task)
            return self.success(task)

        summary, code = self.run_pipeline("resume", record)
        self.assertEqual(code, 0)
        self.assertEqual(len(called), 12)
        self.assertEqual(len(set(called)), 12)
        self.assertEqual(summary["stocks_all_requests_successful"], 2)
        self.assertTrue(summary["scope_ready_for_review"])
        self.assertEqual(self.run_pipeline("resume", self.fail)[1], 0)

    def test_retry_is_scoped_and_ambiguity_remains_unknown(self):
        calls = []
        timeout = p.Task("000001", p.BAO, 2025)
        ambiguous = p.Task("000001", "cninfo_dividend")

        def provider(task):
            calls.append(task)
            if task == timeout and calls.count(task) == 1:
                return {"status": "API_ERROR", "error": "HTTP read timeout"}, []
            if task == ambiguous:
                return {"status": "UPSTREAM_EMPTY_OR_SCHEMA"}, []
            return self.success(task)

        summary, code = self.first_run(provider, "--max-attempts", "2", limit="1")
        self.assertEqual(code, 0)
        self.assertEqual(calls.count(timeout), 2)
        self.assertEqual(calls.count(ambiguous), 2)
        self.assertEqual(summary["terminal_exceptions"], 1)
        self.assertEqual(summary["effective_successes"], 5)
        self.assertFalse(summary["full_historical_action_coverage_certified"])
        self.assertFalse(summary["can_emit_training_labels"])
        self.assertEqual(self.run_pipeline("resume", self.fail)[1], 0)

    def test_circuit_persists_and_blocks_resume(self):
        def rate_limit(_):
            return {"status": "API_ERROR", "error": "HTTP 429 Too Many Requests"}, []

        summary, code = self.first_run(rate_limit)
        self.assertEqual(code, 2)
        self.assertEqual(summary["circuit"]["reason"], "RATE_LIMIT")
        self.assertEqual(summary["new_attempts"], 1)
        summary, code = self.run_pipeline("resume", self.fail)
        self.assertEqual(code, 2)
        self.assertFalse(summary["scope_ready_for_review"])
        summary, code = self.run_pipeline("resume", self.success, "--reset-circuit")
        self.assertEqual(code, 0)
        self.assertIsNone(summary["circuit"])

    def test_sustained_errors_circuit_is_source_specific(self):
        def provider(task):
            if task.endpoint.startswith("sina"):
                return {"status": "API_ERROR", "error": "connection failed"}, []
            return self.success(task)

        summary, code = self.first_run(provider, "--max-attempts", "1", "--circuit-window", "2")
        self.assertEqual(code, 2)
        self.assertEqual(summary["circuit"]["source"], "sina")
        self.assertEqual(summary["circuit"]["observed"], 2)
        self.assertEqual(summary["new_attempts"], 3)

    def test_sha256_tampering_and_highest_attempt_deletion_rejected(self):
        self.first_run(self.success, limit="1")
        path = next((self.output / "attempts").glob("*.json"))
        original = path.read_bytes()
        doc = p.read_json(path)
        doc["attempt"]["entry"]["response_rows"] = 42
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(p.EvidenceError, "SHA256"):
            self.run_pipeline("resume", self.fail)
        path.write_bytes(original)
        path.unlink()
        with self.assertRaisesRegex(p.EvidenceError, "immutable attempt missing"):
            self.run_pipeline("resume", self.fail)

    def test_test_window_rejected_before_attempt_is_persisted(self):
        def future(task):
            if task.endpoint == "sina_dividend":
                return {"status": "SUCCESS", "response_rows": 1}, [{
                    "request_code": task.stock_code, "公告日期": "2026-06-01",
                    "除权除息日": p.CUTOFF, "股权登记日": "2026-06-11",
                    "进度": "实施", "送股": "0", "转增": "0", "派息": "5",
                }]
            return self.success(task)

        with self.assertRaisesRegex(p.EvidenceError, "future"):
            self.first_run(future, limit="1")
        self.assertFalse(list((self.output / "attempts").glob("*.json")))

    def test_reuse_legacy_copies_once_and_reject_damaged_copy(self):
        class Ak:
            def stock_history_dividend_detail(self, symbol, indicator):
                return pd.DataFrame()

            def stock_dividend_cninfo(self, symbol):
                return pd.DataFrame([dict(zip(p.free.CNINFO_COLS,
                                               ["2025-03-01", 0, 0, 2, "2025-03-05", "2025-03-04"],
                                               strict=True))])

        first = self.ak / "batches_first" / "batch_000000"
        second = self.ak / "batches_second" / "batch_000000"
        p.free.fetch_batch(self.snapshot, ak=Ak(), offset=0, limit=1, out=first, wait_seconds=0)
        shutil.copytree(first, second)
        calls = []

        def record(task):
            calls.append(task)
            return self.success(task)

        summary, code = self.first_run(record, limit="1")
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(task.endpoint == p.BAO for task in calls))
        self.assertEqual(summary["historical_attempts"], 3)
        self.assertEqual(summary["unique_candidate_events"], 1)
        with (second / "request_ledger.csv").open("a") as handle:
            handle.write("changed")
        with self.assertRaisesRegex(p.EvidenceError, "SHA256"):
            self.run_pipeline("resume", self.fail)

    def test_missing_akshare_amount_is_not_a_confirmed_zero(self):
        day = {"request_code": "000001", "公告日期": "2025-03-01",
               "除权除息日": "2025-03-05", "股权登记日": "2025-03-04",
               "进度": "实施", "送股": "", "转增": "0", "派息": "5"}
        candidates, issues = p.free._normalize(pd.DataFrame([day]), "sina_dividend",
                                              start="2024-12-01", cutoff=p.CUTOFF)
        self.assertTrue(candidates.empty)
        self.assertIn("Missing entitlement amount", issues.iloc[0].reason)
        self.assertEqual(issues.iloc[0].ex_date, "2025-03-05")
