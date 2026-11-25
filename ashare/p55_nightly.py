"""Run the daily research journal and archive one complete maturity revision."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import tempfile

from .p55_daily import daily
from .p55_mature import OFFICIAL, run as mature
from .p5_sequences import CHINA


def nightly(source, models, seed, ledger):
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
        reports = ledger / "maturity"
        reports.mkdir(exist_ok=True)
        destination = reports / result["trade_date"]
        if destination.exists():
            if not (destination / "maturity_audit.json").is_file():
                raise RuntimeError(f"Existing maturity path has no completed audit: {destination}")
            return
        with tempfile.TemporaryDirectory(prefix=".maturity-", dir=reports) as tmp:
            staged = Path(tmp) / "result"
            mature(ledger, staged, seed / "dataset", None, None, OFFICIAL)
            os.rename(staged, destination)
            fd = os.open(reports, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        print(f"{datetime.now(CHINA).isoformat()} {result['status']} maturity revision {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "models", "evidence-seed", "ledger"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    nightly(args.source, args.models, args.evidence_seed, args.ledger)


if __name__ == "__main__":
    main()
