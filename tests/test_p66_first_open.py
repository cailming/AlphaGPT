import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from ashare.p66_first_open import SIGNAL, account_plan, audit_close, audit_fills, write_once
from ashare.p6_portfolio import initial_state, PortfolioConfig, trading_unit


class FirstOpenTest(unittest.TestCase):
    def setUp(self):
        self.cfg = PortfolioConfig()
        self.state = initial_state(self.cfg)
        self.state.update(session=1,last_day=SIGNAL)
        self.ranks = [f"688{i:03d}" for i in range(24)]

    def test_first_decision_slots_with_fallback_candidates(self):
        for strategy,limit in (("A",4),("B",20),("C",20)):
            plan = account_plan("m/"+strategy,self.state,self.ranks,self.cfg)
            self.assertEqual((plan["ordered_candidate_codes"],plan["max_new_positions"]), (self.ranks[:20],limit))
            self.assertEqual(plan["per_name_budget_including_fees"],50000)

    def test_empty_rank_and_no_projected_fills(self):
        plan=account_plan("m/B",self.state,[],self.cfg)
        self.assertEqual(plan["max_new_positions"],0)
        self.assertNotIn("buys",plan)
        with self.assertRaises(ValueError):
            audit_fills(plan,[{}],{},self.cfg)

    def test_unit_limits_and_quote_required(self):
        for code,expected in (("688256",(200,1)),("300001",(100,100)),("000001",(100,100)),("920001",(100,1))):
            self.assertEqual(trading_unit(code)[:2],expected)
        plan=account_plan("m/A",self.state,self.ranks,self.cfg)
        trade=dict(stock_code=self.ranks[0],side="buy",signal_date=SIGNAL,shares="200")
        with self.assertRaises(ValueError):
            audit_fills(plan,[trade],{},self.cfg)

    def test_repeated_plan_keeps_bytes(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"plan.json"
            write_once(path,{"config":"a"})
            first=path.read_bytes()
            write_once(path,{"config":"a"})
            self.assertEqual(path.read_bytes(),first)
            with self.assertRaises(ValueError):
                write_once(path,{"config":"b"})

    def test_closing_marks_match_or_are_explicitly_stale(self):
        pos = {"stock_code":"688001","mark":"9.9","stale":"True"}
        audit_close([pos],{},None)
        with self.assertRaises(ValueError):
            audit_close([pos],{},1.0)
        with self.assertRaises(ValueError):
            audit_close([pos | {"stale":"False"}],{"688001":{"close":10}},1.0)
