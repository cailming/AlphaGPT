"""Contract tests: P4.3-E2 provider ingestion stays partial and fail-closed."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from sqlalchemy import create_engine

MODULE = Path(__file__).resolve().parents[1] / "ashare" / "p4_action_feed.py"
spec = importlib.util.spec_from_file_location("p4_action_feed", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def dividend(ts="002594.SZ", ex="20250729", *, div_proc="实施", cash="3.974",
             impl="20250722", bonus="0.8", capital="1.2", stated="2"):
    return dict(ts_code=ts, end_date="20241231", ann_date="20250329",
                imp_ann_date=impl, div_proc=div_proc, stk_div=stated,
                stk_bo_rate=bonus, stk_co_rate=capital,
                cash_div_tax=cash, record_date="20250728", ex_date=ex,
                pay_date="20250729")


def basic_fixture():
    return pd.DataFrame([
        {"ts_code":"002594.SZ", "list_status":"L", "list_date":"20110630", "delist_date":""},
        {"ts_code":"000001.SZ", "list_status":"L", "list_date":"19910403", "delist_date":""},
        {"ts_code":"600001.SH", "list_status":"D", "list_date":"19910101", "delist_date":"20250501"},
        {"ts_code":"300001.SZ", "list_status":"L", "list_date":"20261201", "delist_date":""},
        {"ts_code":"600002.SH", "list_status":"D", "list_date":"19910101", "delist_date":"20230101"},
    ])


class DummyProvider:
    def __init__(self):
        self.requested = []

    def dividend(self, *, ts_code, fields):
        self.requested.append(ts_code)
        assert "cash_div_tax" in fields
        if ts_code == "000001.SZ":
            return pd.DataFrame(columns=mod.FIELDS)
        if ts_code == "002594.SZ":
            return pd.DataFrame([dividend(), dividend(ex="20260803")])
        if ts_code == "600001.SH":
            raise RuntimeError("rate limited")
        raise AssertionError(f"Unexpected fetch: {ts_code}")

    def stock_basic(self, *, exchange, list_status, fields):
        assert exchange == ""
        assert "list_date" in fields
        return basic_fixture().query("list_status == @list_status").copy()


class FeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.market = pd.DataFrame({"stock_code": ["000001", "002594", "000999"]})

    def tearDown(self):
        self.temp.cleanup()

    def snapshot(self):
        folder = self.root / "snapshot"
        meta = mod.create_snapshot(market=self.market, basics=basic_fixture(),
                                   start="2025-01-01", test_start="2026-06-12",
                                   output=folder)
        return folder, meta

    def test_universe_union_delisted_and_unmapped_not_hidden(self):
        uni = mod.normalize_snapshot(self.market, basic_fixture(),
                                     start="2025-01-01", test_start="2026-06-12")
        self.assertEqual(set(uni.stock_code), {"000001", "000999", "002594", "600001"})
        self.assertEqual(uni.loc[uni.stock_code.eq("600001"), "list_status"].item(), "D")
        self.assertEqual(uni.loc[uni.stock_code.eq("000999"), "ts_code"].item(), "")
        self.assertEqual(int((~uni.present_in_formal_market).sum()), 1)
        p = DummyProvider()
        listing = mod.fetch_stock_basic(p)
        self.assertEqual(len(listing), len(basic_fixture()))

    def test_fetch_snapshot_excludes_old_test_event_and_keeps_errors(self):
        snapshot, _ = self.snapshot()
        pro = DummyProvider()
        batch = self.root / "batches" / "batch_000000"
        manifest = mod.fetch_batch(snapshot, pro=pro, offset=0, limit=4, out=batch)
        ledger = pd.read_csv(batch / "request_ledger.csv", dtype=str, keep_default_na=False)
        raw = pd.read_csv(batch / "raw_dividends.csv", dtype=str, keep_default_na=False)
        self.assertEqual(manifest["requested_symbols"], 4)
        self.assertEqual(manifest["success_symbols"], 2)
        self.assertEqual(set(ledger.status), {"SUCCESS", "NO_TS_MAPPING", "API_ERROR"})
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw.iloc[0].ex_date, "20250729")
        self.assertNotIn("20260803", raw.ex_date.tolist())
        self.assertEqual(int(ledger.removed_old_test_ex_date_rows.astype(int).sum()), 1)
        with self.assertRaises(FileExistsError):
            mod.fetch_batch(snapshot, pro=pro, offset=0, limit=4, out=batch)

    def test_audit_still_never_issues_complete_certificates(self):
        snapshot, _ = self.snapshot()
        batch = self.root / "batches" / "batch_000000"
        mod.fetch_batch(snapshot, pro=DummyProvider(), offset=0, limit=4, out=batch)
        control = self.root / "control.csv"
        control.write_text("stock_code,ex_date,published_on,share_multiplier,cash_per_old_share,source_url,source_id\n"
                           "002594,2025-07-29,2025-07-22,3,3.974,https://example.com/byd,BYD-2025-047\n")
        summary, events, coverage, errors, controls = mod.audit_batches(
            snapshot, self.root / "batches", official_events=control)
        self.assertEqual(summary["expected_symbols"], 4)
        self.assertEqual(summary["normalized_vendor_candidate_events"], 1)
        self.assertEqual(summary["fetched_success"], 2)
        self.assertEqual(summary["known_issuer_reference_mismatches"], 0)
        self.assertFalse(summary["full_market_actions_certified"])
        self.assertFalse(summary["rights_issue_coverage_certified"])
        self.assertEqual(summary["verified_complete_coverage_assertions_created"], 0)
        self.assertEqual(int(coverage.complete_verified.sum()), 0)
        self.assertEqual(events.iloc[0].review_status, "UNVERIFIED_VENDOR")
        self.assertEqual(events.iloc[0].share_multiplier, 3)
        self.assertNotIn("COMPLETE_VERIFIED", set(coverage.coverage_status))

    def test_conflicting_same_ex_date_quarantined_not_favored(self):
        rows = pd.DataFrame([
            dict(request_code="002594", **dividend()),
            dict(request_code="002594", **dividend(cash="3.500")),
        ])
        events, bad = mod.normalize_dividends(rows, start="2025-01-01",
                                              test_start="2026-06-12")
        self.assertTrue(events.empty)
        self.assertEqual(sum(bad.reason.eq("CONFLICTING_SAME_EX_DATE")), 2)

    def test_invalid_fields_and_proposals_cannot_be_events(self):
        rows = pd.DataFrame([
            dict(request_code="002594", **dividend(div_proc="预案")),
            dict(request_code="002594", **dividend(impl="")),
            dict(request_code="002594", **dividend(bonus="1.1")),
            dict(request_code="002594", **dividend(ex="20260803")),
        ])
        events, bad = mod.normalize_dividends(rows, start="2025-01-01",
                                              test_start="2026-06-12")
        self.assertTrue(events.empty)
        self.assertEqual(len(bad), 4)
        self.assertIn("NOT_IMPLEMENTED", bad.reason.tolist())
        self.assertIn("OLD_TEST_EXCLUDED", bad.reason.tolist())

    def test_checksum_tampering_and_duplicate_batch_fail(self):
        snapshot, _ = self.snapshot()
        batch = self.root / "batches" / "batch_000000"
        mod.fetch_batch(snapshot, pro=DummyProvider(), offset=0, limit=4, out=batch)
        (batch / "request_ledger.csv").write_text((batch / "request_ledger.csv").read_text() + "garbage\n")
        with self.assertRaisesRegex(ValueError, "Modified provider batch"):
            mod.audit_batches(snapshot, self.root / "batches")
        (snapshot / "universe.csv").write_text((snapshot / "universe.csv").read_text() + "garbage\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            mod.audit_batches(snapshot, self.root / "batches")

    def test_proposal_announced_in_old_test_is_not_saved(self):
        raw = pd.DataFrame([
            dividend(ex="", div_proc="预案", impl="", stated="0", bonus="0", capital="0", cash="0"),
            dividend(),
        ])
        raw.loc[0, "ann_date"] = "20260701"
        kept, excluded = mod._prune_future(raw, cutoff="2026-06-12")
        self.assertEqual(excluded, 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept.iloc[0].ex_date, "20250729")

    def test_duplicate_batches_are_rejected_not_implicitly_merged(self):
        snapshot, _ = self.snapshot()
        root = self.root / "batches"
        mod.fetch_batch(snapshot, pro=DummyProvider(), offset=0, limit=2,
                        out=root / "batch_000000")
        mod.fetch_batch(snapshot, pro=DummyProvider(), offset=1, limit=2,
                        out=root / "batch_000001")
        with self.assertRaisesRegex(ValueError, "Multiple batch claims"):
            mod.audit_batches(snapshot, root)

    def test_sql_market_universe_read_only_and_cutoff(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        with engine.begin() as c:
            c.exec_driver_sql("ATTACH DATABASE ':memory:' AS public")
            c.exec_driver_sql("CREATE TABLE public.market (stock_code TEXT, trade_date DATE, not_billing INTEGER)")
            c.exec_driver_sql("INSERT INTO public.market VALUES "
                              "('002594','2026-06-10',1),"
                              "('000001','2026-06-12',1),"
                              "('600001','2026-05-01',0)")
        got = mod.load_formal_market_codes("2025-01-01", "2026-06-12", engine=engine)
        self.assertEqual(got.stock_code.tolist(), ["002594"])

    def test_full_response_boundary_not_certified(self):
        snapshot, _ = self.snapshot()
        class Limited(DummyProvider):
            def dividend(self, *, ts_code, fields):
                return pd.DataFrame([dividend(ts=ts_code)] * 2)
        batch = self.root / "batches" / "batch_000000"
        mod.fetch_batch(snapshot, pro=Limited(), offset=0, limit=2,
                        out=batch, response_row_cap=2)
        led = pd.read_csv(batch / "request_ledger.csv")
        self.assertTrue(led.loc[led.ts_code.notna(), "status"].eq("ROW_LIMIT").all())
        summary, events, _, _, _ = mod.audit_batches(snapshot, self.root / "batches")
        self.assertEqual(summary["fetched_success"], 0)
        self.assertTrue(events.empty)

    def test_reference_mismatch_cannot_silent_pass(self):
        snapshot, _ = self.snapshot()
        mod.fetch_batch(snapshot, pro=DummyProvider(), offset=0, limit=4,
                        out=self.root / "batches" / "batch_000000")
        control = self.root / "control.csv"
        control.write_text("stock_code,ex_date,published_on,share_multiplier,cash_per_old_share,source_url,source_id\n"
                           "002594,2025-07-29,2025-07-22,3,1.0,https://example.com/byd,BYD-2025-047\n")
        summary, _, _, _, control_rows = mod.audit_batches(
            snapshot, self.root / "batches", official_events=control)
        self.assertEqual(summary["known_issuer_reference_mismatches"], 1)
        self.assertEqual(control_rows.status.tolist(), ["MISSING_OR_CONFLICTING_VENDOR_CANDIDATE"])


if __name__ == "__main__":
    unittest.main()
