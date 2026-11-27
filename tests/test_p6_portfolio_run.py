import tempfile
from datetime import datetime, timezone
from pathlib import Path
import unittest
from ashare.p4_action_pipeline import atomic_json
from ashare.p6_portfolio_run import benchmark, checked_day, action_flags, usable_signal
from ashare.p55_daily import parse_clock
from ashare.p6_portfolio_report import publish_day


class AccountSourceTest(unittest.TestCase):
    def test_freeze_boundary_uses_real_clock_before_unseen_ranking(self):
        early = parse_clock("2026-11-27T08:40:00+08:00")
        after = parse_clock("2026-11-27T18:08:00+08:00")
        self.assertTrue(usable_signal("2026-11-27","2026-11-27T18:26:00+08:00",early))
        self.assertFalse(usable_signal("2026-11-27","2026-11-27T18:26:00+08:00",after))
        self.assertFalse(usable_signal("2026-11-27","2026-11-28T09:00:00+08:00",early))

    def test_benchmark_uses_old_weights_for_gap_and_new_for_intraday(self):
        quotes={"old":{"open":9.,"prior_close":10.},"new":{"open":4.,"close":5.}}
        weights,ret,valid=benchmark({"old":1.},["new"],quotes,{})
        self.assertTrue(valid)
        self.assertEqual(weights,{"new":1.})
        self.assertAlmostEqual(ret,.125)
        self.assertIsNone(benchmark({"old":1.},["new"],quotes,{"old":{"status":"UNKNOWN"}})[1])

    def test_late_evidence_and_missing_held_mark(self):
        flags = action_flags({"x":{"covered_from":"2025-02-01","covered_through":"2025-03-10",
                                  "actions":[["2025-03-05"]]}}, "2025-03-05", {"x","y"})
        self.assertFalse(flags["x"]["available_at_open"])
        self.assertEqual(flags["y"]["status"],"MISSING_COVERAGE")
        self.assertEqual(benchmark({"old":1.},None,{},{}),({},None,False))

    def test_archive_hash_checks_state_and_original_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"2025-01-02"
            path.mkdir()
            atomic_json(path/"checkpoint.json",{})
            atomic_json(path/"inputs.json",{})
            publish_day(path,path.name,"diagnostic","HISTORICAL_DIAGNOSTIC",
                    dict(equity=[],positions=[],trades=[],rejections=[]),"policy",None,datetime.now(timezone.utc))
            self.assertEqual(checked_day(path,None,"policy")[0],{})
            (path/"inputs.json").write_text("different")
            with self.assertRaises(ValueError):
                checked_day(path,None,"policy")
