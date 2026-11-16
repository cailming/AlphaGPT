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


class ModernAk(FakeAk):
    """Current AkShare spelling; legacy method must never be selected."""
    def stock_dividend_cninfo(self, symbol):
        return FakeAk.stock_dividents_cninfo(self, symbol)
    def stock_dividents_cninfo(self, symbol):
        raise AssertionError("Legacy misspelled CNINFO method was selected")


class PartialModernAk(ModernAk):
    def stock_dividend_cninfo(self, symbol):
        raise ConnectionError("simulated CNINFO request failure")


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

    def test_current_cninfo_method_name_selected_and_legacy_fallback_supported(self):
        endpoint = free._endpoint(ModernAk(), "002594", "cninfo_dividend")
        self.assertEqual(len(endpoint), 1)
        self.assertEqual(endpoint.iloc[0]["除权日"], "2025-07-29")
        old_endpoint = free._endpoint(FakeAk(), "002594", "cninfo_dividend")
        self.assertEqual(len(old_endpoint), 1)
        self.batch(ak=ModernAk())
        report, events, stocks, _, _, _ = free.audit_batches(
            self.snap, self.batchroot
        )
        self.assertEqual(report["failed_or_truncated"], 0)
        self.assertEqual(report["fetched_not_verified"], 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0].review_status,
                         "TWO_PUBLIC_SOURCES_MATCH_UNVERIFIED")

    def test_official_control_not_fetched_is_not_false_mismatch(self):
        free.fetch_batch(self.snap, ak=ModernAk(), offset=0, limit=1,
                         out=self.batchroot/"batch_000000", wait_seconds=0)
        official=self.root/"official.csv"
        official.write_text(
            "stock_code,ex_date,share_multiplier,cash_per_old_share,source_id\n"
            "002594,2025-07-29,3,3.974,BYD-2025-047\n"
        )
        report, _, stocks, _, _, controls = free.audit_batches(
            self.snap, self.batchroot, official_events=official
        )
        self.assertEqual(controls.iloc[0].status, "NOT_FETCHED_FOR_CONTROL")
        self.assertEqual(report["issuer_controls_not_fetched"], 1)
        self.assertEqual(report["issuer_controls_mismatch"], 0)
        self.assertEqual(report["issuer_controls_matched"], 0)
        self.assertEqual(report["issuer_controls_incomplete_fetch"], 0)

    def test_failed_control_source_not_misreported_as_data_conflict(self):
        self.batch(ak=PartialModernAk())
        official=self.root/"official.csv"
        official.write_text(
            "stock_code,ex_date,share_multiplier,cash_per_old_share,source_id\n"
            "002594,2025-07-29,3,3.974,BYD-2025-047\n"
        )
        report, _, _, _, _, controls = free.audit_batches(
            self.snap, self.batchroot, official_events=official
        )
        self.assertEqual(controls.iloc[0].status, "INCOMPLETE_FETCH_FOR_CONTROL")
        self.assertEqual(report["issuer_controls_incomplete_fetch"], 1)
        self.assertEqual(report["issuer_controls_mismatch"], 0)

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
