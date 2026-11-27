import unittest
import pandas as pd
from ashare.p58_monitor import breadth


class BreadthAuditTest(unittest.TestCase):
    def test_partial_session_propagates_through_market_and_sequence(self):
        days = pd.bdate_range("2025-01-02", periods=42)
        rows = [(str(code), d, 10., 11., 9., 10., 100., 1000., 1)
                for d in days for code in range(120 if d != days[15] else 30)]
        raw = pd.DataFrame(rows, columns="stock_code trade_date open high low close volume amount not_billing".split())
        audit = breadth(raw)
        self.assertEqual(audit["failed_market_days"],[str(d.date()) for d in days[15:17]])
        self.assertEqual(audit["missing_20d_feature_dates"],[str(d.date()) for d in days[26:36]])
        self.assertEqual(audit["valid_20d_sessions_in_last_sequence"],6)
