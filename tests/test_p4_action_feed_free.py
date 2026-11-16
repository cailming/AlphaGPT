"""Regression tests for no-token historical action ingestion and audit."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import json
import unittest

import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "ashare/p4_action_feed_free.py"
spec = importlib.util.spec_from_file_location("p4_action_feed_free", MODULE)
free = importlib.util.module_from_spec(spec)
spec.loader.exec_module(free)


SINA = {"公告日期":"2025-07-22", "送股":8, "转增":12, "派息":39.74,
        "进度":"实施", "除权除息日":"2025-07-29", "股权登记日":"2025-07-28"}
CNINFO = {"实施方案公告日期":"2025-07-22", "送股比例":8,
          "转增比例":12, "派息比例":39.74, "除权日":"2025-07-29",
          "股权登记日":"2025-07-28"}
RIGHTS = {"公告日期":"2025-02-01", "配股方案":2.5, "配股价格":8.8,
          "除权日":"2025-02-05", "股权登记日":"2025-02-04"}


class FakeAk:
    def __init__(self, *, conflict=False, future=False, failing=False, rights=False):
        self.conflict,self.future,self.failing,self.rights=conflict,future,failing,rights
    def stock_history_dividend_detail(self, symbol, indicator):
        if symbol == "000001":
            return pd.DataFrame()
        if self.failing and indicator == "分红":
            raise ConnectionError("simulated rate limit")
        if indicator == "配股":
            return pd.DataFrame([RIGHTS]) if self.rights else pd.DataFrame()
        rows=[SINA.copy()]
        if self.future:
            later=SINA.copy();later["公告日期"]="2026-06-12";later["除权除息日"]="2026-06-20"
            rows.append(later)
        return pd.DataFrame(rows)
    def stock_dividents_cninfo(self, symbol):
        if symbol == "000001":
            return pd.DataFrame()
        x=CNINFO.copy()
        if self.conflict:x["派息比例"]=39.0
        return pd.DataFrame([x])


class FreeFeedTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.snap=self.root/"snapshot"
        self.batchroot=self.root/"batches"
        free.save_snapshot(pd.DataFrame({"stock_code":["002594","000001","002594"]}),
                           start="2024-12-01",test_start="2026-06-12",out=self.snap)
    def tearDown(self):self.tmp.cleanup()
    def batch(self, *, ak=None):
        d=self.batchroot/"batch_000000"
        free.fetch_batch(self.snap, ak=ak or FakeAk(), offset=0,limit=2,out=d,wait_seconds=0)
        return d

    def test_snapshot_is_historical_formal_market_and_immutable(self):
        meta,univ=free._snapshot(self.snap)
        self.assertEqual(univ.stock_code.tolist(),["000001","002594"])
        self.assertFalse(meta["full_exchange_listed_universe_certified"])
        with self.assertRaisesRegex(ValueError,"immutable"):
            free.save_snapshot(pd.DataFrame({"stock_code":["000001"]}),start="2025-01-01",test_start="2026-06-12",out=self.snap)
        with self.assertRaisesRegex(ValueError,"test boundary"):
            free._snapshot(self.snap,"2026-06-11")

    def test_byd_sina_cninfo_per_ten_dividends_and_issuer_control(self):
        self.batch()
        official=self.root/"official.csv"
        official.write_text("stock_code,ex_date,share_multiplier,cash_per_old_share,source_id\n"
            "002594,2025-07-29,3,3.974,BYD-2025-047\n")
        report,events,stocks,issues,rights,ctrl=free.audit_batches(self.snap,self.batchroot,official_events=official)
        self.assertEqual(len(events),1)
        self.assertEqual(events.iloc[0].review_status,"TWO_PUBLIC_SOURCES_MATCH_UNVERIFIED")
        self.assertAlmostEqual(events.iloc[0].share_multiplier,3)
        self.assertAlmostEqual(events.iloc[0].cash_per_old_share,3.974)
        self.assertEqual(ctrl.iloc[0].status,"MATCHED_ISSUER_CONTROL")
        self.assertEqual(report["complete_verified_assertions_created"],0)
        self.assertTrue(stocks.complete_verified.eq(False).all())
        self.assertEqual(report["fetched_not_verified"],2)
        self.assertEqual(len(issues),0)

    def test_conflicting_sources_quarantine_event(self):
        self.batch(ak=FakeAk(conflict=True))
        result,events,_,issues,_,_=free.audit_batches(self.snap,self.batchroot)
        self.assertEqual(len(events),0)
        self.assertTrue(issues.reason.str.contains("SINA_CNINFO_TERMS_MISMATCH").any())
        self.assertTrue(result["training_gate_must_remain_closed"])

    def test_future_records_never_persist(self):
        self.batch(ak=FakeAk(future=True))
        raw=free._read(self.batchroot/"batch_000000"/"raw_sina_dividend.csv")
        ledger=free._read(self.batchroot/"batch_000000"/"request_ledger.csv")
        self.assertFalse(raw["除权除息日"].astype(str).str.contains("2026-06-20").any())
        self.assertEqual(int(ledger.loc[(ledger.stock_code=="002594") & (ledger.endpoint=="sina_dividend"),"dropped_future_or_invalid"].iloc[0]),1)
        report,events,_,issues,_,_=free.audit_batches(self.snap,self.batchroot)
        self.assertEqual(len(events),1)

    def test_provider_failure_and_rights_are_not_certification(self):
        self.batch(ak=FakeAk(failing=True,rights=True))
        report,events,stocks,issues,rights,_=free.audit_batches(self.snap,self.batchroot)
        self.assertEqual(report["failed_or_truncated"],1)
        self.assertEqual(len(rights),1)
        self.assertEqual(report["rights_records_requiring_manual_review"],1)
        self.assertEqual(stocks.loc[stocks.stock_code=="002594","coverage_status"].iloc[0],"FETCH_FAILED_OR_TRUNCATED")
        self.assertTrue(report["training_gate_must_remain_closed"])

    def test_prefer_canonical_akshare_cninfo_method(self):
        class ModernAk(FakeAk):
            def stock_dividend_cninfo(self, symbol):
                self.canonical_called = True
                return super().stock_dividents_cninfo(symbol)
            def stock_dividents_cninfo(self, symbol):
                # Overridden below via parent method explicitly; ensure the
                # live adapter never calls this misspelled method directly.
                raise AssertionError("legacy CNINFO name should not be invoked")
        # Implement modern method without touching the legacy callable.
        provider=ModernAk()
        provider.stock_dividend_cninfo=lambda symbol: FakeAk.stock_dividents_cninfo(provider,symbol)
        self.batch(ak=provider)
        report,events,stocks,issues,rights,ctrl=free.audit_batches(self.snap,self.batchroot)
        self.assertEqual(report["fetched_not_verified"],2)
        self.assertEqual(len(events),1)
        self.assertEqual(report["endpoint_results"]["cninfo_dividend"]["SUCCESS"],2)

    def test_unfetched_issuer_control_is_not_called_mismatch(self):
        free.fetch_batch(self.snap, ak=FakeAk(), offset=0, limit=1,
                         out=self.batchroot/"batch_000000", wait_seconds=0)
        # BYD is present in the snapshot, but this batch fetched only 000001.
        official=self.root/"official.csv"
        official.write_text("stock_code,ex_date,share_multiplier,cash_per_old_share,source_id\n"
            "002594,2025-07-29,3,3.974,BYD-2025-047\n")
        report, _, _, _, _, ctrl=free.audit_batches(
            self.snap,self.batchroot,official_events=official)
        self.assertEqual(report["issuer_controls_total_in_reference"], 1)
        self.assertEqual(report["issuer_controls_not_fetched"], 1)
        self.assertEqual(report["issuer_controls_checked"], 0)
        self.assertEqual(report["issuer_controls_mismatch"], 0)
        self.assertEqual(ctrl.iloc[0].status, "NOT_FETCHED_FOR_CONTROL")

    def test_issuer_in_scope_with_failed_dividend_provider_is_pending(self):
        self.batch(ak=FakeAk(failing=True))
        official=self.root/"official.csv"
        official.write_text("stock_code,ex_date,share_multiplier,cash_per_old_share,source_id\n"
            "002594,2025-07-29,3,3.974,BYD-2025-047\n")
        report,_,_,_,_,controls=free.audit_batches(
            self.snap,self.batchroot,official_events=official)
        self.assertEqual(report["issuer_controls_sources_incomplete"],1)
        self.assertEqual(report["issuer_controls_mismatch"],0)
        self.assertEqual(report["endpoint_status_counts"]["sina_dividend"]["API_ERROR"],1)
        self.assertIn("simulated rate limit",report["endpoint_failure_examples"]["sina_dividend"][0])
        self.assertEqual(controls.iloc[0].status,"INCOMPLETE_FETCH_FOR_CONTROL")

    def test_rights_only_counted_in_audit_interval(self):
        class HistoricalRights(FakeAk):
            def stock_history_dividend_detail(self, symbol, indicator):
                if indicator == "配股":
                    old = RIGHTS.copy()
                    old.update({"公告日期":"2018-02-01","除权日":"2018-02-05",
                                "股权登记日":"2018-02-04"})
                    return pd.DataFrame([RIGHTS,old]) if symbol == "002594" else pd.DataFrame()
                return super().stock_history_dividend_detail(symbol,indicator)
        self.batch(ak=HistoricalRights())
        report,_,_,_,rights,_=free.audit_batches(self.snap,self.batchroot)
        self.assertEqual(report["rights_records_raw_all_history"],2)
        self.assertEqual(report["rights_records_requiring_manual_review"],1)
        self.assertEqual(len(rights),1)
        self.assertEqual(rights.iloc[0]["除权日"],"2025-02-05")

    def test_cninfo_keyerror_is_ambiguous_not_proof_of_empty_actions(self):
        class EmptyOrSchemaAk(FakeAk):
            def stock_dividend_cninfo(self, symbol):
                if symbol == "002594":
                    raise KeyError("实施方案公告日期")
                return FakeAk.stock_dividents_cninfo(self, symbol)

        self.batch(ak=EmptyOrSchemaAk())
        ledger = free._read(self.batchroot/"batch_000000"/"request_ledger.csv")
        bad = ledger.loc[(ledger.stock_code.eq("002594")) &
                         (ledger.endpoint.eq("cninfo_dividend"))].iloc[0]
        self.assertEqual(bad.status, "UPSTREAM_EMPTY_OR_SCHEMA")
        self.assertIn("not a verified empty", bad.error)
        report, events, stocks, _, _, _ = free.audit_batches(
            self.snap, self.batchroot
        )
        self.assertEqual(report["cninfo_upstream_ambiguous_calls"], 1)
        self.assertEqual(report["failed_or_truncated"], 1)
        self.assertEqual(report["fetched_not_verified"], 1)
        self.assertEqual(report["endpoint_status_counts"]["cninfo_dividend"][
            "UPSTREAM_EMPTY_OR_SCHEMA"], 1)
        self.assertEqual(report["endpoint_failure_stocks"]["cninfo_dividend"][0][
            "stock_code"], "002594")
        self.assertEqual(stocks.loc[stocks.stock_code.eq("002594"),
                                    "coverage_status"].iloc[0],
                         "FETCH_FAILED_OR_TRUNCATED")
        self.assertFalse(stocks.complete_verified.any())
        self.assertTrue(report["training_gate_must_remain_closed"])
        # Other independent provider observations remain only candidates.
        self.assertTrue(events.review_status.str.contains("UNVERIFIED").all())

    def test_other_cninfo_keyerror_remains_api_error(self):
        class BrokenAk(FakeAk):
            def stock_dividend_cninfo(self, symbol):
                if symbol == "002594":
                    raise KeyError("unexpected field")
                return FakeAk.stock_dividents_cninfo(self, symbol)

        self.batch(ak=BrokenAk())
        ledger=free._read(self.batchroot/"batch_000000"/"request_ledger.csv")
        issue=ledger.loc[(ledger.stock_code.eq("002594")) &
                         (ledger.endpoint.eq("cninfo_dividend"))].iloc[0]
        self.assertEqual(issue.status, "API_ERROR")
        report,_,_,_,_,_=free.audit_batches(self.snap,self.batchroot)
        self.assertEqual(report["cninfo_upstream_ambiguous_calls"],0)

    def test_tampered_raw_files_fail_sha256_audit(self):
        path=self.batch()
        raw=path/"raw_sina_dividend.csv"
        raw.write_text(raw.read_text()+"corruption",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"Modified batch"):
            free.audit_batches(self.snap,self.batchroot)

    def test_snapshot_missing_codes_have_not_fetched_status(self):
        free.fetch_batch(self.snap,ak=FakeAk(),offset=0,limit=1,out=self.batchroot/"batch_000000",wait_seconds=0)
        _,_,stocks,_,_,_=free.audit_batches(self.snap,self.batchroot)
        self.assertEqual(stocks.loc[stocks.stock_code=="002594","coverage_status"].iloc[0],"NOT_FETCHED")


if __name__=="__main__":unittest.main()
