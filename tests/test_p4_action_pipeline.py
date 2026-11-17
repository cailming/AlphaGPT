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
        self.assertEqual(len(candidates), 1)
        self.assertTrue(pd.isna(candidates.iloc[0].bonus_per_old_share))
        self.assertTrue(pd.isna(candidates.iloc[0].share_multiplier))
        self.assertIn("Missing entitlement amount", issues.iloc[0].reason)
        self.assertEqual(issues.iloc[0].ex_date, "2025-03-05")

    def test_redistributed_bonus_and_reserve_do_not_match(self):
        def provider(task):
            code = task.stock_code
            if task.endpoint == "sina_dividend":
                return {"status": "SUCCESS", "response_rows": 1}, [{
                    "request_code": code, "公告日期": "2025-03-01", "除权除息日": "2025-03-05",
                    "股权登记日": "2025-03-04", "进度": "实施", "送股": "1", "转增": "2", "派息": "5",
                }]
            if task.endpoint == p.BAO and task.year == 2025:
                return {"status": "SUCCESS", "response_rows": 1}, [{
                    "request_code": code, "query_year": 2025, "code": "sz." + code,
                    "dividPlanDate": "2025-03-01", "dividOperateDate": "2025-03-05",
                    "dividRegistDate": "2025-03-04", "dividStocksPs": "0.2",
                    "dividReserveToStockPs": "0.1", "dividCashPsBeforeTax": "0.5",
                }]
            return self.success(task)

        summary, code = self.first_run(provider, limit="1")
        self.assertEqual(code, 0)
        self.assertEqual(summary["baostock_full_field_matches"], 0)
        self.assertEqual(summary["date_or_amount_conflicts"], 1)

    def test_old_known_ex_date_does_not_cause_request_failure(self):
        class Ak:
            def stock_dividend_cninfo(self, symbol):
                return pd.DataFrame([dict(zip(p.free.CNINFO_COLS,
                                               ["", 0, 0, 2, "1993-05-17", "1993-05-14"],
                                               strict=True))])

        result, raw = p.free.fetch_request(Ak(), "000001", "cninfo_dividend",
                                           cutoff=p.CUTOFF, row_cap=1000, audit_start="2024-12-01")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertTrue(raw.empty)

    def test_incomplete_request_retains_independently_valid_rows(self):
        def provider(task):
            if task.endpoint == "sina_dividend":
                return {"status": "UNKNOWN_EX_DATE", "response_rows": 2}, [{
                    "request_code": task.stock_code, "公告日期": "2025-03-01", "除权除息日": "2025-03-05",
                    "股权登记日": "2025-03-04", "进度": "实施", "送股": "0", "转增": "0", "派息": "5",
                }]
            return self.success(task)

        summary, code = self.first_run(provider, limit="1")
        self.assertEqual(code, 0)
        self.assertEqual(summary["terminal_exceptions"], 1)
        self.assertEqual(summary["unique_candidate_events"], 1)

    def test_ak_date_diagnostics_exclude_known_out_of_period_rows(self):
        def record(ex, announced):
            return dict(zip(p.free.CNINFO_COLS,
                            [announced, 0, 0, .5, ex, "2025-05-01"], strict=True))

        class Ak:
            def stock_dividend_cninfo(self, symbol):
                return pd.DataFrame([record("", "2025-05-01"), record("bad", "2025-05-01"),
                                     record("2025-05-03", ""), record("1993-05-17", ""),
                                     record("", "2026-06-30"), record("2025-05-06", "2025-05-01")])

        entry, raw = p.free.fetch_request(Ak(), "000001", "cninfo_dividend",
                                          cutoff=p.CUTOFF, row_cap=1000, audit_start="2024-12-01")
        self.assertEqual(entry["status"], "UNKNOWN_EX_DATE")
        self.assertEqual(entry["missing_ex_date_rows"], 1)
        self.assertEqual(entry["invalid_ex_date_rows"], 1)
        self.assertEqual(entry["missing_announcement_date_rows"], 1)
        self.assertEqual(raw["除权日"].tolist(), ["2025-05-06"])

    def test_wrong_returned_security_opens_p0_circuit(self):
        def provider(task):
            if task.endpoint == p.BAO:
                return {"status": "WRONG_STOCK_CODE"}, []
            return self.success(task)

        summary, code = self.first_run(provider, limit="1")
        self.assertEqual(code, 2)
        self.assertEqual(summary["p0_exceptions"], 1)
        self.assertEqual(summary["circuit"]["reason"], "WRONG_STOCK_CODE")
        self.assertEqual(summary["training_gate"], "CLOSED")

    def test_governance_recomputes_and_refuses_incomplete_final(self):
        from ashare import p4_action_phase3_audit as governance

        def provider(task):
            if task.endpoint == "sina_dividend":
                return {"status": "UNKNOWN_EX_DATE", "response_rows": 2,
                        "missing_ex_date_rows": 2, "missing_announcement_date_rows": 1}, []
            return self.success(task)

        self.first_run(provider, limit="1")
        report_path = governance.generate(self.output, final=False)
        self.assertIn("缺除权日 2 行", report_path.read_text())
        self.assertIn("training_gate=CLOSED", report_path.read_text())
        pointer = p.read_json(self.output / "latest_governance.json")
        profile = pd.read_csv(Path(pointer["governance_dir"]) / "anomaly_profile.csv")
        missing = profile.loc[profile.category.eq("MISSING_EX_DATE")]
        self.assertEqual(missing.occurrences.iloc[0], 2)
        with self.assertRaisesRegex(p.EvidenceError, "5,449-stock"):
            governance.generate(self.output, final=True)
        report_dir = Path(p.read_json(self.output / "latest_report.json")["report_dir"])
        request_csv = report_dir / "request_status.csv"
        request_csv.write_text(request_csv.read_text().replace("TERMINAL_EXCEPTION", "NOT_FETCHED", 1))
        with self.assertRaisesRegex(p.EvidenceError, "Request CSV and summary disagree"):
            governance.generate(self.output, final=False)

    def test_paused_source_stays_pending_while_other_sources_finish(self):
        def provider(task):
            if task.endpoint == p.BAO:
                return {"status": "API_ERROR", "error": "HTTP 429 Too Many Requests"}, []
            return self.success(task)

        summary, code = self.first_run(provider)
        self.assertEqual(code, 2)
        self.assertEqual(summary["circuit"]["source"], p.BAO)

        def healthy_only(task):
            self.assertNotEqual(task.endpoint, p.BAO)
            return self.success(task)

        summary, code = self.run_pipeline("resume", healthy_only, "--isolate-circuit")
        self.assertEqual(code, 2)
        self.assertEqual(summary["effective_successes"], 6)
        self.assertEqual(summary["paused_sources"][p.BAO]["reason"], "RATE_LIMIT")
        self.assertEqual(summary["pending_retries"], 1)
        self.assertFalse(summary["scope_ready_for_review"])
        self.assertEqual(p.read_json(self.output / "checkpoint.json")["state"], "WAITING_FOR_SOURCES")

        summary, code = self.run_pipeline("resume", self.success, "--restore-source", p.BAO)
        self.assertEqual(code, 0)
        self.assertEqual(summary["paused_sources"], {})
        self.assertTrue(summary["scope_ready_for_review"])

    def test_phase3_recovery_requires_provider_login_and_preserves_other_circuits(self):
        from ashare import p4_action_phase3_recovery as recovery

        cp = {"state": "WAITING_FOR_SOURCES", "offset": 0, "limit": 5449,
              "paused_sources": {p.BAO: {"reason": "TIMEOUT"}}}
        p.atomic_json(self.output / "checkpoint.json", cp)
        with patch.object(recovery, "probe_login", return_value=False) as probe, patch.object(
                recovery.subprocess, "run") as resume:
            self.assertEqual(recovery.resume_after_login(self.output, 2700, 1), 2)
            probe.assert_called_once()
            resume.assert_not_called()
        with patch.object(recovery, "probe_login", return_value=True), patch.object(
                recovery.subprocess, "run") as resume:
            resume.return_value.returncode = 0
            self.assertEqual(recovery.resume_after_login(self.output, 2700, 1), 0)
            self.assertIn("--restore-source", resume.call_args.args[0])
        cp["paused_sources"]["cninfo"] = {"reason": "RATE_LIMIT"}
        p.atomic_json(self.output / "checkpoint.json", cp)
        with patch.object(recovery, "probe_login") as probe:
            self.assertEqual(recovery.resume_after_login(self.output, 2700, 1), 2)
            probe.assert_not_called()

    def test_baostock_only_restoration_requires_query_and_preserves_cninfo_circuit(self):
        def schema_change(task):
            if task.endpoint == "cninfo_dividend":
                return {"status": "API_ERROR", "error": "schema changed"}, []
            return self.success(task)

        initial, code = self.first_run(schema_change, limit="1")
        self.assertEqual(code, 2)
        path = self.output / "checkpoint.json"
        checkpoint = p.read_json(path)
        checkpoint["paused_sources"] = {p.BAO: {"reason": "TIMEOUT"}}
        p.atomic_json(path, checkpoint)
        flags = ("--only-source", p.BAO, "--restore-source", p.BAO)
        with self.assertRaisesRegex(ValueError, "successful query"):
            self.run_pipeline("resume", self.fail, *flags)

        args = self.args("audit")
        evidence = p.Evidence(args)
        _, prepared = p.configure(evidence, args)
        task = p.Task("000001", p.BAO, 2026)
        evidence.persist(task, {"stock_code": "000001", "endpoint": p.BAO, "year": 2026,
                                "status": p.SUCCESS, "response_rows": 0, "error": ""}, [], p.SUCCESS)
        p.track_attempt(prepared, args, evidence.effective(task))
        p.atomic_json(path, prepared)

        calls = []

        def bao_only(query):
            calls.append(query)
            return self.success(query)

        result, code = self.run_pipeline("resume", bao_only, *flags)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [p.Task("000001", p.BAO, year) for year in (2024, 2025)])
        self.assertEqual(result["circuit"], initial["circuit"])
        self.assertFalse(result["scope_ready_for_review"])
        self.assertEqual(result["training_gate"], "CLOSED")
        self.assertIsNone(p.read_json(path).get("active_source"))

    def test_second_circuit_does_not_replace_first_during_scoped_run(self):
        def cninfo_failure(task):
            if task.endpoint == "cninfo_dividend":
                return {"status": "API_ERROR", "error": "schema drift"}, []
            return self.success(task)

        initial, _ = self.first_run(cninfo_failure, limit="1")
        calls = []

        def bao_failure(task):
            calls.append(task)
            return {"status": "API_ERROR", "error": "HTTP 429 Too Many Requests"}, []

        result, code = self.run_pipeline("resume", bao_failure, "--only-source", p.BAO)
        self.assertEqual(code, 2)
        self.assertEqual(calls, [p.Task("000001", p.BAO, 2024)])
        self.assertEqual(result["circuit"], initial["circuit"])
        self.assertEqual(result["paused_sources"][p.BAO]["reason"], "RATE_LIMIT")
