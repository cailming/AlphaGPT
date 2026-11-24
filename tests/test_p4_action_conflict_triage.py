"""Offline tests for audit-only conflict routing."""
from decimal import Decimal
import unittest

from ashare import p4_action_conflict_triage as triage


def terms(cash, capitalization="0", bonus=None, record="2025-09-02"):
    return {"record_date": record, "published_on": "2025-08-27", "cash_pre_tax": Decimal(cash),
            "capitalization": Decimal(capitalization), "bonus": bonus}


class ConflictTriageTests(unittest.TestCase):
    def test_precision_only_routes_small_positive_differences_with_same_date(self):
        self.assertEqual(triage.amount("12.700900", "sina_dividend"), Decimal("1.2700900"))
        self.assertIsNone(triage.amount("", "cninfo_dividend"))
        sources = {"cninfo_dividend": [terms("1.2700919")], "sina_dividend": [terms("1.27009")]}
        self.assertEqual(triage.classify(sources), "SMALL_DIFFERENCE_COMPATIBLE_WITH_PRECISION")
        sources["sina_dividend"][0]["record_date"] = "2025-09-01"
        self.assertEqual(triage.classify(sources), "RECORD_DATE_DIFFERENCE")
        self.assertFalse(triage.precision_compatible(Decimal("0"), Decimal("0.000001")))
        self.assertTrue(triage.precision_compatible(Decimal("0.132371"), Decimal("0.132372")))
        self.assertFalse(triage.precision_compatible(Decimal("0.12"), Decimal("0.1201")))

    def test_zero_cash_and_same_day_components_are_not_precision_differences(self):
        sources = {"cninfo_dividend": [terms("0.1", "0.4")], "sina_dividend": [terms("0", "0.4")]}
        self.assertEqual(triage.classify(sources), "SINA_CASH_ZERO_WITH_MATCHING_SHARE_TERMS")
        sources["cninfo_dividend"].append(terms("0.6"))
        self.assertEqual(triage.classify(sources), "SAME_DAY_MULTIPLE_ROWS")

    def test_missing_values_do_not_substitute_for_zeros(self):
        sources = {"cninfo_dividend": [terms("0.1", "0.4")], "sina_dividend": [terms("0", "0.4")]}
        sources["sina_dividend"][0]["cash_pre_tax"] = None
        self.assertEqual(triage.classify(sources), "UNRESOLVED_TERMS_OR_EVIDENCE")
