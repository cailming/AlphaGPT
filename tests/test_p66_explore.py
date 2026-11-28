import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ashare import p66_explore as explore
from ashare.p6_portfolio import initial_state
from ashare.p4_subset_inputs import hash_file


class ExploratoryTests(unittest.TestCase):
    a,b = "600001","600002"
    market = {c:dict(open=10.,close=10.,pre_close=10.,prior_close=10.) for c in (a,b)}
    first,second,third = "2026-11-23","2026-11-24","2026-11-25"

    def initial(self):
        return initial_state(explore.frozen.CONFIG)

    def day(self, state, day, prior, rank, bars=None, opening=(), closing=None):
        return explore.advance(state,day,prior,rank,self.market if bars is None else bars,
                               "B",set(opening),closing or {},None)

    def test_uncertain_right_is_symbol_scoped_across_checkpoint_and_cash_is_settled(self):
        orig = self.day(self.initial(),self.first,"2026-11-20",[self.a])[0]
        cash = orig["cash"]
        second,daily,*_ = self.day(orig,self.second,self.first,[self.a],closing={
            self.a:{"status":"UNKNOWN","available_at_open":False}})
        self.assertIsNone(daily["nav"])
        self.assertEqual(second["cash"],cash)
        restored = json.loads(json.dumps(second))
        res,summary,pos,trades,rejects,risk = self.day(restored,self.third,self.second,[self.b])
        self.assertEqual([(t["stock_code"],t["side"]) for t in trades],[(self.b,"buy")])
        self.assertEqual(rejects[0]["reason"],"SYMBOL_ENTITLEMENT_QUARANTINE")
        self.assertEqual(res["holdings"][self.a]["shares"],orig["holdings"][self.a]["shares"])
        self.assertGreater(risk["asset_share"],0)
        self.assertIsNone(summary["nav"])
        self.assertEqual(summary["indicative_nav"],res["equity"]/1000000)

    def test_post_open_new_buy_checked_without_earned_record_date_right(self):
        flag = {self.a:{"status":"MISSING_COVERAGE","available_at_open":False}}
        result = self.day(self.initial(),self.first,"2026-11-20",[self.a],closing=flag)
        state,row,_,trades,_,risk = result
        self.assertEqual(len(trades),1)
        self.assertEqual(risk["new_buyer_valuation_watch"],{self.a:"MISSING_COVERAGE"})
        self.assertFalse(state["unknown_claims"])
        self.assertIsNone(row["nav"])
        self.assertEqual(row["action_cash_paid"],0)
        _,next_day,*_ = self.day(state,self.second,self.first,[self.a])
        self.assertIsNotNone(next_day["nav"])
        self.assertIsNone(next_day["daily_net_return"])

    def test_known_open_risk_blocks_but_late_terms_cannot_cancel_open_sale(self):
        st = self.day(self.initial(),self.first,"2026-11-20",[self.a])[0]
        event = {self.a:{"status":"VERIFIED","available_at_open":False,"record_date":self.first,
                         "cash_per_share":0.15,"share_multiplier":1,"cash_pay_date":self.third}}
        s,d,_,fills,_,_ = self.day(st,self.second,self.first,[self.b],closing=event)
        self.assertIn((self.a,"sell"),[(t["stock_code"],t["side"]) for t in fills])
        self.assertEqual(d["action_cash_paid"],0)
        expected = explore.money(st["holdings"][self.a]["shares"]*.15)
        self.assertEqual(d["dividend_receivable"],expected)
        _,end,*_ = self.day(s,self.third,self.second,[self.b])
        self.assertEqual(end["action_cash_paid"],expected)
        _,_,_,fills,_,_ = self.day(st,self.second,self.first,[self.b],opening=[self.a],closing=event)
        self.assertNotIn(self.a,[t["stock_code"] for t in fills])

    def test_missing_or_suspended_does_not_drop_position(self):
        s = self.day(self.initial(),self.first,"2026-11-20",[self.a])[0]
        for quote in ({}, {self.a:self.market[self.a] | {"suspended":True}}):
            state,daily,_,fills,rejects,_ = self.day(s,self.second,self.first,[self.b],bars=quote)
            self.assertIn(self.a,state["holdings"])
            self.assertFalse(fills)
            self.assertTrue(rejects)
            if not quote:
                self.assertIsNone(daily["nav"])

    def test_verified_bonus_not_sellable_before_listing(self):
        st = self.day(self.initial(),self.first,"2026-11-20",[self.a])[0]
        event = {self.a:{"status":"VERIFIED","available_at_open":False,"record_date":self.first,
                         "cash_per_share":0,"share_multiplier":2,"share_list_date":self.third}}
        st2,_,_,_,_,_ = self.day(st,self.second,self.first,[self.a],closing=event)
        self.assertEqual(st2["holdings"][self.a]["pending_shares"],st["holdings"][self.a]["shares"])
        _,_,_,fills,_,_ = self.day(st2,self.third,self.second,[self.b])
        self.assertEqual([t for t in fills if t["stock_code"]==self.a][0]["shares"],2*st["holdings"][self.a]["shares"])

    def test_witness_rejects_archive_from_after_historical_decision(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            loc = root/"historical"/self.first
            loc.mkdir(parents=True)
            (loc/"manifest.json").write_text(json.dumps({"trade_date":self.first,
                "committed_at":"2026-11-28T01:00:00+08:00","evidence_snapshot":{"root":"future"}}))
            self.assertEqual(explore.witness(root,self.first,datetime.fromisoformat(self.second+"T09:15:00+08:00"),
                                             "diagnostic"),({},None))

    def test_missing_whole_session_stops_progress(self):
        self.assertEqual(explore.check_schedule(self.first,self.third,{self.first,self.third}),
                         (self.second,"AWAITING_FORMAL_ACCOUNT_DAY"))

    def test_coverage_recovery_releases_only_a_documented_no_event(self):
        c = {"stock_code":self.a,"day":self.first,"reason":"MISSING_COVERAGE"}
        event = c | {"reason":"UNKNOWN"}
        source = {self.a:{"covered_from":self.first,"covered_through":self.second,"actions":[]}}
        self.assertEqual(explore.opening_release([c,event],source),([event],[c]))
        source[self.a]["actions"] = [[self.first,self.first]]
        self.assertEqual(explore.opening_release([c],source),([c],[]))

    def test_drifted_or_missing_archived_mark_identifies_stock_and_carries_last_value(self):
        st = self.day(self.initial(),self.first,"2026-11-20",[self.a])[0]
        bars,audit = explore.day_marks({},self.second,{self.a},"diagnostic",
                                       datetime.fromisoformat(self.third+"T23:30:00+08:00"))
        self.assertEqual(audit["unavailable_closes"],[self.a])
        _,row,_,_,_,risk = self.day(st,self.second,self.first,[self.a],bars=bars)
        self.assertEqual(risk["stale_close_marks"][self.a]["carried_value"],row["holdings_value"])
        self.assertIsNone(row["nav"])

    def test_verified_rate_cannot_credit_a_previous_unknown_quantity(self):
        st = self.day(self.initial(),self.first,"2026-11-20",[self.a])[0]
        st = self.day(st,self.second,self.first,[self.a],closing={
            self.a:{"status":"UNKNOWN","available_at_open":False}})[0]
        state,row,*_ = self.day(st,self.third,self.second,[self.a],closing={self.a:{
            "status":"VERIFIED","available_at_open":False,"record_date":self.second,"cash_per_share":1.,
            "share_multiplier":1,"cash_pay_date":self.third}})
        self.assertEqual(row["action_cash_paid"],0)
        self.assertTrue(state["unknown_claims"][-1]["record_quantity_unconfirmed"])

    def test_revision_is_separate_and_leaves_cash_and_original_bytes_intact(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/"ledger"
            path = root/"forward"/self.second
            path.mkdir(parents=True)
            (root/"policy.json").write_text("{}")
            state = self.day(self.initial(),self.first,"2026-11-20",[self.a])[0]
            state = self.day(state,self.second,self.first,[self.a],closing={
                self.a:{"status":"UNKNOWN","available_at_open":False}})[0]
            ck = path/"checkpoint.json"
            ck.write_text(json.dumps({"accounts":{"ma20_reversal/B":state}}))
            digest = hash_file(ck)
            inputs = path/"inputs.json"
            inputs.write_text(json.dumps({"actions":{self.a:{"status":"UNKNOWN"}}}))
            (path/"portfolio_summary.json").write_text(json.dumps({"day":self.second,
                "policy_sha256":hash_file(root/"policy.json"),"previous_sha256":None,
                "files":{"checkpoint.json":digest,"inputs.json":hash_file(inputs)}}))
            receipt = {"record_date":self.first, "cash_per_share":.15,"cash_basis":"VERIFIED_ACCOUNT_CREDIT_PER_OLD_SHARE",
                       "share_multiplier":1}
            target = Path(tmp)/"revisions/first.json"
            with patch.object(explore,"receipt_evidence",return_value=({(1,)},[receipt])):
                explore.revision(root,"forward",self.second,Path(tmp)/"receipts",target)
                with self.assertRaises(FileExistsError):
                    explore.revision(root,"forward",self.second,Path(tmp)/"receipts",target)
            review = json.loads(target.read_text())["accounts"]["ma20_reversal/B"]
            self.assertGreater(review["revised_indicative_equity"],state["equity"])
            self.assertFalse(review["changes_spendable_cash"])
            self.assertEqual(digest,hash_file(ck))
            # Independent fixture with evidence that cash was booked under VERIFIED terms.
            inputs.write_text(json.dumps({"actions":{self.a:{"status":"VERIFIED"}}}))
            meta = json.loads((path/"portfolio_summary.json").read_text())
            meta["files"]["inputs.json"] = hash_file(inputs)
            (path/"portfolio_summary.json").write_text(json.dumps(meta))
            with patch.object(explore,"receipt_evidence",return_value=({(1,)},[receipt])):
                target2 = target.with_name("second.json")
                explore.revision(root,"forward",self.second,Path(tmp)/"receipts",target2)
                self.assertEqual(json.loads(target2.read_text())["accounts"]["ma20_reversal/B"]["verified_cash_scenario_total"],0)


if __name__ == "__main__":
    unittest.main()
