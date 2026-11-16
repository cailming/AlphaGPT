"""No-network tests for fail-closed BaoStock third-source corporate actions."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "ashare/p4_action_feed_baostock.py"
spec = importlib.util.spec_from_file_location("p4_action_feed_baostock", MODULE)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

FIELDS = list(b.RAW_FIELDS)
BYD = {
    "code": "sz.002594", "dividPlanDate": "2025-07-22",
    "dividRegistDate": "2025-07-28", "dividOperateDate": "2025-07-29",
    "dividCashPsBeforeTax": "3.974", "dividStocksPs": "0.8",
    "dividReserveToStockPs": "1.2",
}


class FakeResult:
    error_code = "0"
    error_msg = ""
    fields = FIELDS
    def __init__(self, records, *, error_code="0", fields=FIELDS):
        self.rows = records
        self.error_code = error_code
        self.fields = fields
        self.pos = 0
    def next(self):
        if self.pos >= len(self.rows):
            return False
        self.pos += 1
        return True
    def get_row_data(self):
        return [self.rows[self.pos-1][field] for field in self.fields]


class FakeBao:
    def __init__(self, records=None, *, fail=None):
        self.records = records or {}
        self.fail = fail or set()
        self.queries = []
        self.logged_out = False
    def login(self):
        return type("Login", (), {"error_code": "0", "error_msg": ""})()
    def logout(self):
        self.logged_out = True
    def query_dividend_data(self, *, code, year, yearType):
        self.queries.append((code, year, yearType))
        if (code, year) in self.fail:
            return FakeResult([], error_code="100")
        return FakeResult(self.records.get((code, year), []))


class BaoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.snap = self.root / "snapshot"
        self.snap.mkdir()
        codes = ["000004", "002594", "600000", "830001"]
        pd.DataFrame({"stock_code": codes,
                      "scope": ["FORMAL_MARKET_DB_ONLY"]*len(codes)}).to_csv(
                          self.snap / "universe.csv", index=False)
        self.meta = self.root / "metadata.json"
        self.meta.write_text(json.dumps({"model_version": "P4_RIDGE_V1",
                                         "test_start": "2026-06-12"}))
        (self.snap / "snapshot.json").write_text(json.dumps({
            "snapshot_version": "P4_AKSHARE_FORMAL_UNIVERSE_V1",
            "universe_sha256": b._sha(self.snap / "universe.csv"),
            "start": "2024-12-01", "test_start_EXCLUDED": "2026-06-12",
            "symbols": 4,
        }))
        self.rootbatch = self.root / "batches"
        self.out = self.rootbatch / "batch_000000"

    def tearDown(self):
        self.tmp.cleanup()

    def collect(self, bs, codes=("000004", "002594"), years=(2024, 2025, 2026)):
        return b.collect(self.snap, self.meta, bs=bs, out=self.out,
                         codes=list(codes), years=list(years), wait_seconds=0)

    def test_byd_per_share_match_and_no_certification(self):
        bs = FakeBao({("sz.002594", "2025"): [BYD]})
        self.collect(bs)
        self.assertTrue(bs.logged_out)
        self.assertEqual(len(bs.queries), 6)
        self.assertTrue(all(x[2] == "operate" for x in bs.queries))
        summ, events, stocks, issues, _ = b.audit(self.snap, self.meta, self.rootbatch)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0].stock_code, "002594")
        self.assertAlmostEqual(events.iloc[0].cash_per_old_share, 3.974)
        self.assertAlmostEqual(events.iloc[0].share_multiplier, 3)
        self.assertEqual(summ["complete_verified_assertions_created"], 0)
        self.assertTrue(summ["can_emit_training_labels"] is False)
        self.assertTrue(stocks.complete_verified.eq(False).all())
        self.assertEqual(stocks.loc[stocks.stock_code.eq("000004"),
                                    "provider_status"].iloc[0], "FETCHED_NOT_VERIFIED")
        self.assertEqual(int(stocks.loc[stocks.stock_code.eq("000004"),
                                        "candidate_events"].iloc[0]), 0)

    def test_zero_events_are_not_complete_coverage(self):
        self.collect(FakeBao(), codes=("000004",))
        summary, events, stocks, issues, _ = b.audit(self.snap, self.meta, self.rootbatch)
        self.assertEqual(len(events), 0)
        self.assertEqual(summary["bao_successful_queries"], 3)
        self.assertTrue(stocks.loc[stocks.stock_code.eq("000004"),
                                   "no_event_rows_observed"].iloc[0])
        self.assertFalse(summary["full_historical_action_coverage_certified"])

    def test_future_and_unresolved_ex_dates_not_persisted(self):
        future = BYD | {"dividOperateDate": "2026-06-20"}
        unknown = BYD | {"dividOperateDate": ""}
        bs = FakeBao({("sz.002594", "2026"): [future, unknown]})
        self.collect(bs, codes=("002594",), years=(2026,))
        raw = pd.read_csv(self.out / "raw_baostock_dividends.csv")
        ledger = pd.read_csv(self.out / "request_ledger.csv")
        self.assertEqual(len(raw), 0)
        self.assertEqual(ledger.iloc[0].status, "UNKNOWN_EX_DATE")
        self.assertEqual(ledger.iloc[0].excluded_out_of_window, 1)
        self.assertEqual(ledger.iloc[0].unknown_ex_date_rows, 1)
        summary, _, stocks, _, _ = b.audit(self.snap, self.meta, self.rootbatch)
        self.assertEqual(summary["bao_failed_or_unsupported_queries"], 1)
        self.assertEqual(stocks.loc[stocks.stock_code.eq("002594"),
                                    "provider_status"].iloc[0], "FAILED_OR_UNSUPPORTED")

    def test_bad_publication_date_quarantined_not_normalized(self):
        row = BYD | {"dividPlanDate": ""}
        self.collect(FakeBao({("sz.002594", "2025"): [row]}),
                     codes=("002594",), years=(2025,))
        _, events, _, issue, _ = b.audit(self.snap, self.meta, self.rootbatch)
        self.assertEqual(len(events), 0)
        self.assertEqual(len(issue), 1)
        self.assertIn("INVALID_BAOSTOCK_ROW", issue.iloc[0].reason)

    def test_unsupported_exchange_never_guessed(self):
        bs = FakeBao()
        self.collect(bs, codes=("830001",), years=(2025,))
        self.assertEqual(len(bs.queries), 0)
        report, events, stocks, _, _ = b.audit(self.snap, self.meta, self.rootbatch)
        self.assertEqual(report["bao_failed_or_unsupported_queries"], 1)
        self.assertEqual(stocks.loc[stocks.stock_code.eq("830001"),
                                    "provider_status"].iloc[0], "FAILED_OR_UNSUPPORTED")

    def test_api_error_is_not_empty_success(self):
        bs = FakeBao(fail={("sz.000004", "2025")})
        self.collect(bs, codes=("000004",), years=(2024, 2025, 2026))
        r, _, stocks, _, _ = b.audit(self.snap, self.meta, self.rootbatch)
        self.assertEqual(r["bao_successful_queries"], 2)
        self.assertEqual(stocks.loc[stocks.stock_code.eq("000004"),
                                    "provider_status"].iloc[0], "FAILED_OR_UNSUPPORTED")

    def test_duplicate_ex_date_rejected_entire_date(self):
        r2 = BYD | {"dividPlanDate": "2025-07-23"}
        self.collect(FakeBao({("sz.002594", "2025"): [BYD, r2]}),
                     codes=("002594",), years=(2025,))
        _, events, _, issue, _ = b.audit(self.snap, self.meta, self.rootbatch)
        self.assertEqual(len(events), 0)
        self.assertTrue(issue.reason.eq("DUPLICATE_EX_DATE_UNRESOLVED").all())

    def test_cross_source_match_and_terms_conflict(self):
        self.collect(FakeBao({("sz.002594", "2025"): [BYD]}),
                     codes=("002594",), years=(2025,))
        ak = self.root / "ak.csv"
        pd.DataFrame([{"stock_code": "002594", "ex_date": "2025-07-29",
                       "share_multiplier": 3.0, "cash_per_old_share": 3.974}]).to_csv(ak,index=False)
        r, events, stocks, issues, _ = b.audit(self.snap,self.meta,self.rootbatch,akshare_events=ak)
        self.assertEqual(r["cross_source_disagreements_or_missing"], 0)
        self.assertTrue(issues.result.eq("BAOSTOCK_AKSHARE_MATCH_UNVERIFIED").all())
        df = pd.read_csv(ak, dtype="string")
        df["cash_per_old_share"] = "3.5"
        df.to_csv(ak, index=False)
        r2, _, _, issues2, _ = b.audit(self.snap,self.meta,self.rootbatch,akshare_events=ak)
        self.assertEqual(r2["cross_source_disagreements_or_missing"], 1)
        self.assertTrue(issues2.result.eq("BAOSTOCK_AKSHARE_TERMS_CONFLICT").all())

    def test_batch_tamper_and_outside_universe_rejected(self):
        self.collect(FakeBao(), codes=("000004",), years=(2025,))
        p = self.out / "raw_baostock_dividends.csv"
        p.write_text(p.read_text()+"tampered")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            b.audit(self.snap,self.meta,self.rootbatch)
        with self.assertRaisesRegex(ValueError, "outside frozen"):
            b.collect(self.snap,self.meta,bs=FakeBao(),out=self.root/"new",codes=["123456"],years=[2025],wait_seconds=0)

    def test_cash_only_blank_bonuses_remain_unverified_not_dropped(self):
        row = BYD | {"dividStocksPs": "", "dividReserveToStockPs": "", "dividCashPsBeforeTax": "0.25"}
        self.collect(FakeBao({("sz.002594", "2025"): [row]}),
                     codes=("002594",), years=(2025,))
        _, events, _, issues, _ = b.audit(self.snap,self.meta,self.rootbatch)
        self.assertEqual(len(events),1)
        self.assertAlmostEqual(events.iloc[0].share_multiplier, 1)
        self.assertAlmostEqual(events.iloc[0].cash_per_old_share, .25)
        self.assertEqual(events.iloc[0].review_status,
                         "BAOSTOCK_PARTIAL_AMOUNT_FIELDS_UNVERIFIED")
        self.assertIn("dividStocksPs", events.iloc[0].blank_amount_fields)

    def test_partial_fields_are_not_fully_corrob_even_when_placeholder_matches(self):
        # The blank capitalization column is UNKNOWN, not a confirmed zero.
        row = BYD | {"dividStocksPs": "0", "dividReserveToStockPs": "",
                     "dividCashPsBeforeTax": "0.25"}
        self.collect(FakeBao({("sz.002594", "2025"): [row]}),
                     codes=("002594",), years=(2024, 2025, 2026))
        ak = self.root/"ak_partial.csv"
        pd.DataFrame([{"stock_code": "002594", "ex_date": "2025-07-29",
                       "share_multiplier": 1.0, "cash_per_old_share": .25}]).to_csv(
                           ak,index=False)
        report, _, _, issues, _ = b.audit(self.snap,self.meta,self.rootbatch,akshare_events=ak)
        self.assertEqual(report["cross_source_fully_observed_matches"],0)
        self.assertEqual(report["cross_source_partial_amount_fields_unresolved"],1)
        self.assertEqual(report["cross_source_disagreements_or_missing"],0)
        self.assertEqual(issues.iloc[0]["result"],
                         "BAOSTOCK_AKSHARE_PARTIAL_AMOUNT_FIELDS_UNRESOLVED")
        # If a missing capitalization value could explain a differing
        # share multiplier, it is NOT proof of a provider contradiction.
        pd.DataFrame([{"stock_code": "002594", "ex_date": "2025-07-29",
                       "share_multiplier": 3.0, "cash_per_old_share": .25}]).to_csv(
                           ak,index=False)
        report2, _, _, issues2, _ = b.audit(self.snap,self.meta,self.rootbatch,akshare_events=ak)
        self.assertEqual(report2["cross_source_partial_amount_fields_unresolved"],1)
        self.assertEqual(issues2.iloc[0]["result"],
                         "BAOSTOCK_AKSHARE_PARTIAL_AMOUNT_FIELDS_UNRESOLVED")
        # A disagreement in the *observed* cash amount is a real conflict.
        pd.DataFrame([{"stock_code": "002594", "ex_date": "2025-07-29",
                       "share_multiplier": 1.0, "cash_per_old_share": .30}]).to_csv(
                           ak,index=False)
        report3, _, _, issues3, _ = b.audit(self.snap,self.meta,self.rootbatch,akshare_events=ak)
        self.assertEqual(report3["cross_source_disagreements_or_missing"],1)
        self.assertEqual(issues3.iloc[0]["result"],"BAOSTOCK_AKSHARE_TERMS_CONFLICT")

    def test_partial_fields_do_not_fully_match_issuer_control(self):
        row = BYD | {"dividStocksPs": "0", "dividReserveToStockPs": "",
                     "dividCashPsBeforeTax": "0.25"}
        self.collect(FakeBao({("sz.002594", "2025"): [row]}),
                     codes=("002594",), years=(2024, 2025, 2026))
        official=self.root/"issuer_partial.csv"
        pd.DataFrame([{"stock_code":"002594","ex_date":"2025-07-29",
                       "share_multiplier":"1.0","cash_per_old_share":"0.25",
                       "source_id":"ISSUER-SAMPLE"}]).to_csv(official,index=False)
        report,_,_,_,controls=b.audit(self.snap,self.meta,self.rootbatch,
                                      issuer_controls=official)
        self.assertEqual(report["issuer_controls_terms_matched"],0)
        self.assertEqual(report["issuer_controls_partial_amount_fields"],1)
        self.assertEqual(controls.iloc[0]["result"],
                         "PARTIAL_AMOUNTS_MATCH_OBSERVED_TERMS_ONLY")
        self.assertFalse(report["can_emit_training_labels"])

    def test_all_blank_entitlements_are_rejected_not_zero_events(self):
        row = BYD | {"dividStocksPs": "", "dividReserveToStockPs": "", "dividCashPsBeforeTax": ""}
        self.collect(FakeBao({("sz.002594", "2025"): [row]}),
                     codes=("002594",), years=(2025,))
        _, events, _, issues, _ = b.audit(self.snap,self.meta,self.rootbatch)
        self.assertEqual(len(events),0)
        self.assertEqual(len(issues),1)

    def test_wrong_operative_year_marked_incomplete(self):
        self.collect(FakeBao({("sz.002594", "2026"): [BYD]}),
                     codes=("002594",), years=(2026,))
        ledger = pd.read_csv(self.out/"request_ledger.csv")
        self.assertEqual(ledger.iloc[0].status,"UNKNOWN_EX_DATE")
        self.assertEqual(len(pd.read_csv(self.out/"raw_baostock_dividends.csv")),0)

    def test_issuer_byd_control_correctly_identifies_baostock_sample(self):
        self.collect(FakeBao({("sz.002594", "2025"): [BYD]}),
                     codes=("002594",), years=(2024,2025,2026))
        official=self.root/"issuer.csv"
        pd.DataFrame([{"stock_code":"002594", "ex_date":"2025-07-29",
                       "share_multiplier":"3", "cash_per_old_share":"3.974",
                       "source_id":"BYD-2025-047"}]).to_csv(official,index=False)
        summary, _, _, _, controls = b.audit(self.snap, self.meta, self.rootbatch,
                                             issuer_controls=official)
        self.assertEqual(summary["issuer_controls_terms_matched"],1)
        self.assertEqual(controls.iloc[0].result,
            "TERMS_MATCH_ISSUER_SAMPLE_UNVERIFIED_COVERAGE")
        self.assertFalse(summary["full_historical_action_coverage_certified"])

    def test_unfetched_akshare_control_not_a_false_disagreement(self):
        self.collect(FakeBao({("sz.002594", "2025"): [BYD]}),
                     codes=("002594",), years=(2024,2025,2026))
        ak = self.root/"ak.csv"
        pd.DataFrame(columns=["stock_code", "ex_date", "share_multiplier",
                              "cash_per_old_share"]).to_csv(ak,index=False)
        cov=self.root/"coverage.csv"
        pd.DataFrame([{"stock_code":"002594", "coverage_status":"NOT_FETCHED"}]).to_csv(cov,index=False)
        report, _, _, issues, _ = b.audit(self.snap,self.meta,self.rootbatch,
                                          akshare_events=ak, akshare_stock_coverage=cov)
        self.assertEqual(report["cross_source_disagreements_or_missing"],0)
        self.assertEqual(report["cross_source_not_comparable_yet"],1)
        self.assertTrue(issues.result.eq("AKSHARE_NOT_COMPARABLE_YET").all())

    def test_future_proposal_not_saved(self):
        row = BYD | {"dividPlanDate": "2026-07-01"}
        self.collect(FakeBao({("sz.002594", "2025"): [row]}),
                     codes=("002594",), years=(2025,))
        raw = pd.read_csv(self.out / "raw_baostock_dividends.csv")
        self.assertEqual(len(raw),0)


if __name__ == "__main__":
    unittest.main()
