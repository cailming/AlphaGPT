import unittest

from ashare.p65_valuation_audit import claim_keys, marked_value, missing_check, revision_status
from ashare.p6_portfolio import PortfolioConfig, initial_state, step


class ValuationAuditTest(unittest.TestCase):
    def test_roles_preserve_unchecked_distinction_and_timed_action(self):
        day = "2026-11-26"
        stocks = {
            "688001":{"covered_from":"2026-11-23", "covered_through":day, "actions":[[day,"DIV"]]},
            "688002":{"covered_from":"2026-11-23", "covered_through":day, "actions":[]},
        }
        row = missing_check(stocks,day,{"688001","688002","688003"},{"688003"})
        self.assertEqual(row["outside_original_scan"],2)
        self.assertEqual(row["witness_gap"],["688003"])
        self.assertEqual(row["unverified_event"],["688001"])

    def test_receipt_does_not_establish_daily_coverage_or_position_credit(self):
        self.assertEqual(revision_status("MISSING_COVERAGE", {(1,)}, [{"id":"A"}]),
                         "RECEIPT_PRESENT_WHOLE_DAY_COVERAGE_STILL_UNVERIFIED")
        self.assertEqual(revision_status("UNKNOWN", {(1,)}, [{"id":"A"}]),
                         "RECEIPT_PRESENT_REVIEW_ECONOMIC_ENTITLEMENT")
        self.assertEqual(revision_status("UNKNOWN", {(1,),(2,)}, [{"id":"A"},{"id":"B"}]),
                         "CONFLICTING_RECEIPTS_REVIEW_NEEDED")
        self.assertEqual(revision_status("UNKNOWN",set(),[]),"NO_VERIFIED_RECEIPT")

    def test_carried_claims_are_not_counted_as_new_and_keep_mark_basis(self):
        gap = {"day":"2026-11-26","stock_code":"688001","reason":"MISSING_COVERAGE"}
        old = claim_keys({"unknown_claims":[gap]})
        today = claim_keys({"unknown_claims":[gap,gap|{"day":"2026-11-27"}]})
        self.assertEqual(sum((today-old).values()),1)
        self.assertEqual(marked_value({"mark":7.5,"shares":200,"pending_shares":5}),1537.5)
        self.assertIsNone(marked_value(None))

    def test_frozen_account_locks_buy_with_carried_unresolved_action(self):
        config = PortfolioConfig()
        stock, other = "688001", "688002"
        prices = {code:{"open":10., "close":10., "pre_close":10.} for code in (stock,other)}
        state, *_ = step(initial_state(config), day="2026-11-23", prior_day=None,
                         ranked=None, bars=prices, strategy="B", config=config)
        state, *_ = step(state, day="2026-11-24", prior_day="2026-11-23", ranked=[stock],
                         bars=prices, strategy="B", config=config)
        state, day, *_ = step(state, day="2026-11-25", prior_day="2026-11-24", ranked=[stock],
                              bars=prices, strategy="B", config=config,
                              actions={stock:{"status":"UNKNOWN", "available_at_open":False}})
        self.assertIsNone(day["nav"])
        state, day, positions, fills, rejections = step(
            state,day="2026-11-26",prior_day="2026-11-25",ranked=[other],
            bars=prices,strategy="B",config=config)
        self.assertEqual({r["reason"] for r in rejections},
                         {"UNCONFIRMED_ACCOUNT_ECONOMICS","UNRESOLVED_ENTITLEMENT"})
        self.assertEqual([r["stock_code"] for r in positions],[stock])
        self.assertEqual(fills,[])
        self.assertIsNone(day["nav"])


if __name__ == "__main__":
    unittest.main()
