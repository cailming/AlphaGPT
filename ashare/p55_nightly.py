"""Run the daily research journal and archive one complete maturity revision."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import tempfile
from functools import partial

from .p55_daily import daily
from .p55_mature import OFFICIAL, run as mature
from .p5_sequences import CHINA


def publish_revision(reports, trade_date, marker, build):
    reports.mkdir(parents=True, exist_ok=True)
    destination = reports / trade_date
    if destination.exists():
        if not (destination / marker).is_file():
            raise RuntimeError(f"Existing report path has no completed audit: {destination}")
        return destination
    with tempfile.TemporaryDirectory(prefix=".revision-", dir=reports) as tmp:
        staged = Path(tmp) / "result"
        build(staged)
        if not (staged / marker).is_file():
            raise RuntimeError(f"A staged report is missing its completed audit: {staged}")
        os.rename(staged, destination)
        fd = os.open(reports, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return destination


def nightly(source, models, seed, ledger, *, panel_ledger=None, factor_model=None, factor_source=None):
    if any(v is not None for v in (panel_ledger, factor_model, factor_source)) and not all(
            v is not None for v in (panel_ledger, factor_model, factor_source)):
        raise ValueError("The six-method panel requires a journal, factor model, and factor source")
    ledger.mkdir(parents=True, exist_ok=True)
    with (ledger / ".nightly.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("P55 research journal already running", flush=True)
            return
        result = daily(source, models, seed, ledger)
        if result["status"] == "DEFERRED":
            print(f"{result['trade_date']} {result['reason']}", flush=True)
            return
        trade_date = result["trade_date"]
        destination = publish_revision(ledger / "maturity", trade_date, "maturity_audit.json",
                                       partial(mature, ledger, seed=seed / "dataset", evidence_source=None,
                                               cutoff=None, official_path=OFFICIAL))
        print(f"{datetime.now(CHINA).isoformat()} {result['status']} maturity revision {destination}", flush=True)
        if panel_ledger is not None:
            from .p55_panel import capture_panel
            from .p55_reports import REPORT_AUDIT, run as panel_reports

            capture_panel(ledger, panel_ledger, factor_model, factor_source, result)
            panel_report = publish_revision(panel_ledger / "reports", trade_date, REPORT_AUDIT,
                                            partial(panel_reports, panel_ledger, seed=seed / "dataset",
                                                    legacy_ledger=ledger, focus=trade_date))
            print(f"{datetime.now(CHINA).isoformat()} six-method report {panel_report}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "models", "evidence-seed", "ledger"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("panel-ledger", "factor-model", "factor-source"):
        parser.add_argument(f"--{flag}", type=Path)
    args = parser.parse_args()
    nightly(args.source, args.models, args.evidence_seed, args.ledger, panel_ledger=args.panel_ledger,
            factor_model=args.factor_model, factor_source=args.factor_source)


if __name__ == "__main__":
    main()
