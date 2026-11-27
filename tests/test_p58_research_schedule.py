"""Check post-close scheduling without a source connection, model, or production cron."""
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(shutil.which("flock"), "Ubuntu schedule uses flock")
class SchedulerTests(unittest.TestCase):
    def execute(self, hour, minute, current_panel=False, patch_enabled=False):
        source = Path(__file__).resolve().parents[1]
        script = next(p for p in (source/"ashare/p58_research_schedule.sh", source/"code/p58_research_schedule.sh")
                      if p.exists())
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            root = folder/"reports/p58/run"
            root.mkdir(parents=True)
            if patch_enabled:
                (root/"p65/quote-patch-daily").mkdir(parents=True)
            for lane in ("portfolio","group_c"):
                (root/lane).mkdir()
                (root/lane/"policy.json").touch()
            fake_bin = folder/"bin"
            fake_bin.mkdir()
            clock = fake_bin/"date"
            clock.write_text(f'#!/bin/bash\ncase "$1" in\n+%F) echo 2026-11-27;;\n+%H) echo {hour};;\n'
                             f'+%H%M) echo {hour}{minute};;\n*) echo 2026-11-27T{hour}:{minute}:00+08:00;;\nesac\n')
            clock.chmod(0o700)
            calls = folder/"calls.log"
            (root/"mode.sh").write_text('echo "$*" >> "$MOCK_CALLS"\n[[ $1 != ashare.p58_fund_snapshot ]]\n')
            if current_panel:
                for child in ("old/ledger/timely/2026-11-27/manifest.json","follow/panel/timely/2026-11-27/manifest.json",
                              "follow/panel/reports/2026-11-27/panel_report_audit.json"):
                    path = folder/child
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
            env = dict(os.environ, PATH=str(fake_bin)+":"+os.environ["PATH"], MOCK_CALLS=str(calls),
                       ALPHAGPT_P5_NEXT_ROOT=str(folder/"old"), ALPHAGPT_P5_FOLLOW_ROOT=str(folder/"follow"))
            job = subprocess.run(["bash",str(script),"post",str(root)],env=env,capture_output=True,text=True)
            return job.returncode, calls.read_text() if calls.exists() else ""

    def test_preclose_does_not_touch_any_lane(self):
        self.assertEqual(self.execute("08","40"), (0, ""))

    def test_late_missing_reports_and_failing_fund_allow_independent_accounts(self):
        status, calls = self.execute("23","35")
        self.assertEqual(status, 1)
        self.assertIn("ashare.p58_monitor", calls)
        self.assertIn("ashare.p58_group_c report", calls)
        self.assertIn("ashare.p6_portfolio_run run", calls)
        self.assertNotIn("ashare.p58_group_c capture", calls)
        self.assertNotIn("p55_nightly", calls)

    def test_accounts_proceed_even_without_current_c_manifest(self):
        _, calls = self.execute("18","10",current_panel=True)
        self.assertIn("ashare.p58_group_c capture", calls)
        self.assertIn("ashare.p6_portfolio_run run", calls)

    def test_quote_patch_runs_last_and_requires_todays_frozen_manifests(self):
        _, calls = self.execute("18", "10", current_panel=True, patch_enabled=True)
        self.assertIn("ashare.p65_enhanced_audit --scheduled-day 2026-11-27", calls.splitlines()[-1])
        self.assertIn("/quote-patch-daily/runs/2026-11-27", calls.splitlines()[-1])
        _, missing_parents = self.execute("18", "10", patch_enabled=True)
        self.assertNotIn("ashare.p65_enhanced_audit", missing_parents)


if __name__ == "__main__":
    unittest.main()
