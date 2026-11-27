import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ashare.p65_market_audit import INCIDENT, patch_scenario, publish_output, recovery


class RecoveryTests(unittest.TestCase):
    def test_prediction_waits_for_twenty_valid_pairs_and_sixteen_finite_inputs(self):
        days = [d.date() for d in pd.bdate_range("2026-10-08", "2026-11-26")]
        rows = [{"trade_date":str(d), "market_session_usable": d not in INCIDENT and d != days[0]} for d in days]
        forecast = recovery(rows, days[-1], set(INCIDENT))
        self.assertEqual(forecast["market_20d_first_recovered_for_incident"], "2026-11-16")
        self.assertEqual(forecast["sequence_16d_first_recovered_for_incident"], "2026-12-07")
        self.assertEqual(len(forecast["currently_affected_input_sessions"]), 7)
        self.assertEqual(len(forecast["scheduled_sessions_still_required"]), 7)
        self.assertEqual(forecast["observed_other_invalid_pair_sessions"],[])

    def test_known_additional_low_breadth_is_reported(self):
        days = [d.date() for d in pd.bdate_range("2026-10-08", "2026-11-26")]
        extra = date(2026,11,26)
        rows = [{"trade_date":str(d), "market_session_usable": d not in (*INCIDENT, extra)} for d in days]
        forecast = recovery(rows, days[-1], set(INCIDENT))
        self.assertEqual(forecast["observed_other_invalid_pair_sessions"],[str(extra)])

    def test_patch_never_replaces_a_formal_key(self):
        formal = pd.DataFrame({"stock_code": ["688256"], "trade_date": [INCIDENT[0]]})
        with self.assertRaisesRegex(ValueError, "original formal-market keys"):
            patch_scenario(formal, pd.DataFrame(), formal.copy())
        outside = pd.DataFrame({"stock_code": ["000001"], "trade_date": [INCIDENT[1]]})
        with self.assertRaisesRegex(ValueError, "scoped to the missing session"):
            patch_scenario(formal, pd.DataFrame(), outside)

    def test_output_is_published_only_after_serialization_succeeds(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit"
            with self.assertRaises(TypeError):
                publish_output(out, {"bad": object()}, None)
            self.assertFalse(out.exists())
            path = publish_output(out, {"day": INCIDENT[0]}, None)
            self.assertTrue(path.is_file())
            with self.assertRaises(FileExistsError):
                publish_output(out, {"replace": True}, None)


if __name__ == "__main__":
    unittest.main()
