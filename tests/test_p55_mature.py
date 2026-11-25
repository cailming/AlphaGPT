"""Regression tests for outcome maturity and honest daily evaluation."""
import unittest

import pandas as pd

from ashare.p55_mature import BAR_FIELDS, daily_report, evaluate, one_label, support_group


class MaturityTests(unittest.TestCase):
    def setUp(self):
        self.days = pd.bdate_range("2026-09-23", periods=10).strftime("%Y-%m-%d").tolist()
        self.ordinals = {day: i for i, day in enumerate(self.days)}
        self.stock = "000001"
        self.raw = pd.DataFrame({"stock_code": self.stock, "trade_date": self.days,
                                 "open": range(10,20), "close": range(10,20), "low": range(10,20),
                                 "high": range(10,20), "volume": 100, "amount": 1000})
        self.evidence = {"max_covered_date": self.days[-1], "stocks": {
            self.stock: {"covered_from": self.days[0], "covered_through": self.days[-1], "actions": []}}}

    def label(self, days=None, official=None):
        return one_label(self.stock, self.days[0], days or self.days, self.ordinals,
                         {self.stock: self.raw.set_index("trade_date")}, self.evidence, official or {})

    def test_exact_session_maturity_and_price(self):
        self.assertEqual(self.label(self.days[:6])[0], "LABEL_PENDING")
        status, buy, sell, result = self.label(self.days[:7])
        self.assertEqual((status, buy, sell), ("MATURE_LABEL", self.days[1], self.days[6]))
        self.assertAlmostEqual(result, 16/11 - 1)

    def test_evidence_holes_remain_missing_and_payoff_requires_typed_notice(self):
        stock = self.evidence["stocks"][self.stock]
        stock["covered_through"] = self.days[5]
        self.assertEqual(self.label()[0], "MISSING_DUAL_ACTION_EVIDENCE")
        stock["covered_through"] = self.days[-1]
        stock["actions"] = [[self.days[3], self.days[2]]]
        self.assertEqual(self.label()[0], "UNVERIFIED_ECONOMIC_ACTION")
        output = self.label(official={(self.stock, self.days[3]): (1., .5, "verified")})
        self.assertEqual(output[0], "MATURE_LABEL")
        self.assertAlmostEqual(output[-1], 16.5/11 - 1)

    def test_top_list_is_chosen_even_when_highest_scored_labels_are_missing(self):
        rows = pd.DataFrame({"stock_code": [f"{i:06d}" for i in range(20)], "score": range(20),
                             "target_return": [.1]*17 + [None]*3, "maturity_status": ["MATURE_LABEL"]*17+["MISSING"]*3})
        state = {}
        first = daily_report(rows, rows, state)
        self.assertEqual(first["top10"]["observed"], 7)
        self.assertIsNone(first["top10"]["gross_economic_return"])
        self.assertAlmostEqual(first["top20"]["net_excess_by_per_side_bps"]["10"], -.002)
        second = daily_report(rows.iloc[0:0], rows.iloc[0:0], state)
        self.assertEqual(second["top20"]["list_replacement"], 1.0)

    def test_historical_and_zero_signal_dates_are_kept_out_of_prospective_return(self):
        items = [
            {"track": "HISTORICAL", "trade_date": self.days[0], "status": "SCORED", "formal_session": True,
             "models": {"absolute": {"rows": 1}}},
            {"track": "PROSPECTIVE", "trade_date": self.days[1], "status": "ABSTAINED", "formal_session": True,
             "models": {"absolute": {"rows": 0}}},
        ]
        prediction = pd.DataFrame([{"prediction_id": "1", "stock_code": self.stock, "trade_date": self.days[0],
            "score": .03, "training_sequences": 3, "support_group": support_group(3),
            "objective": "absolute", "model_version": "fixture", "model_sha256": "fixture", "track": "HISTORICAL"}])
        labels, _, out = evaluate(items, prediction, self.days, self.raw, self.evidence, {})
        self.assertEqual(len(labels), 1)
        self.assertEqual(out["HISTORICAL"]["models"]["absolute"]["groups"]["0-4"]["mature_labels"], 1)
        self.assertEqual(out["PROSPECTIVE"]["zero_prediction_formal_dates"], [self.days[1]])
        self.assertFalse(out["PROSPECTIVE"]["models"]["absolute"]["time_and_label_gate_met"])

    def test_missing_formal_date_stays_visible_before_first_prospective_manifest(self):
        item = {"track": "TIMELY_NONPROSPECTIVE", "trade_date": self.days[0], "status": "ABSTAINED",
                "formal_session": True, "models": {"absolute": {"rows": 0}},
                "prospective_start_after": self.days[0]}
        columns = ["track", "objective", "support_group", "stock_code", "trade_date", "score", "prediction_id",
                   "model_version", "model_sha256", "training_sequences"]
        _, _, output = evaluate([item], pd.DataFrame(columns=columns), self.days[:2], self.raw, self.evidence, {})
        self.assertEqual(output["PROSPECTIVE"]["unrecorded_formal_signal_dates"], [self.days[1]])
