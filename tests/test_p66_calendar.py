import unittest
from datetime import date
from ashare.p66_calendar import coverage, next_session, planned_open, recovery_check
from ashare.p58_fund_snapshot import previous_scheduled_session
from pathlib import Path
from tempfile import TemporaryDirectory
import json


class CalendarTest(unittest.TestCase):
    def test_exchange_rules_and_prior_compatibility(self):
        for d in (date(2026,10,1),date(2026,10,4),date(2026,10,10)):
            self.assertFalse(planned_open(d))
        day = date(2026,9,1)
        while day < date(2026,12,31):
            following = next_session(day)
            self.assertEqual(previous_scheduled_session(following), day)
            day = following

    def test_open_gap_preserves_holding_period_and_unknown_cross_year(self):
        start = date(2026,11,27)
        entry = next_session(start)
        self.assertEqual(entry, date(2026,11,30))
        self.assertEqual(next_session(entry,5),date(2026,12,7))
        self.assertEqual(coverage(start,entry,{start:1})[str(entry)]["status"], "SCHEDULED_OPEN_MISSING_FORMAL")
        self.assertIsNone(next_session(date(2026,12,31)))
        self.assertIsNone(planned_open(date(2027,1,4)))

    def test_missing_pair_delays_recovery(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root/"monitor/2026-11-27/1/status.json"
            file.parent.mkdir(parents=True)
            file.write_text(json.dumps({"formal_through":"2026-11-27", "models":{},
                "formal":{"pair_support":[{"trade_date":"2026-10-19","market_session_usable":False}],
                          "missing_20d_feature_dates":[], "minimum_pair_share":.70}}))
            r = recovery_check(root,{},date(2026,11,27))
            self.assertEqual(r["earliest_clean_signal_if_future_pairs_pass"], "2026-12-07")
            r = recovery_check(root,{"2026-11-30":{"status":"SCHEDULED_OPEN_MISSING_FORMAL"}},date(2026,11,30))
            self.assertIn("2026-12-01",r["failed_or_unobservable_pair_days"])
            self.assertIsNone(r["earliest_clean_signal_if_future_pairs_pass"])
