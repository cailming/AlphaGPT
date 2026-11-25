"""Regression coverage for preserving the published V2 portion of the nightly job."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from ashare.p55_nightly import nightly, publish_revision


class NightlyTests(TestCase):
    def test_incomplete_build_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(RuntimeError):
                publish_revision(path, "2026-11-25", "complete.json", lambda out: out.mkdir())
            self.assertFalse((path / "2026-11-25").exists())

    def test_existing_legacy_revision_does_not_skip_a_new_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            seed, ledger, panel = path / "seed", path / "legacy", path / "panel"
            previous = ledger / "maturity" / "2026-11-25"
            previous.mkdir(parents=True)
            existing = previous / "maturity_audit.json"
            existing.write_text("kept")
            calls = []

            def capture(*args):
                calls.append(args)

            def report(root, out, seed, legacy_ledger, focus):
                self.assertEqual((root, legacy_ledger), (panel, ledger))
                self.assertEqual(focus, "2026-11-25")
                out.mkdir()
                (out / "panel_report_audit.json").write_text("complete")

            modules = {"ashare.p55_panel": SimpleNamespace(capture_panel=capture),
                       "ashare.p55_reports": SimpleNamespace(REPORT_AUDIT="panel_report_audit.json", run=report)}
            frozen_day = {"trade_date": "2026-11-25", "status": "SCORED"}
            with patch("ashare.p55_nightly.daily", return_value=frozen_day), patch.dict(sys.modules, modules):
                nightly(path, path, seed, ledger, panel_ledger=panel, factor_model=path, factor_source=path)
            self.assertEqual(len(calls), 1)
            self.assertEqual(existing.read_text(), "kept")
            self.assertTrue((panel / "reports/2026-11-25/panel_report_audit.json").is_file())
