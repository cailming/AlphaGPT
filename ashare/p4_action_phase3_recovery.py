"""Recheck a paused BaoStock login and resume only after a successful response."""
from __future__ import annotations

from argparse import ArgumentParser
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import time

from . import p4_action_pipeline as pipeline


def can_probe(checkpoint):
    paused = checkpoint.get("paused_sources", {})
    return (checkpoint.get("state") == "WAITING_FOR_SOURCES"
            and checkpoint.get("offset") == 0 and checkpoint.get("limit") == 5449
            and not checkpoint.get("circuit")
            and set(paused) == {pipeline.BAO} and paused[pipeline.BAO]["reason"] == "TIMEOUT")


def probe_login(timeout=12):
    import baostock

    try:
        with pipeline.deadline(timeout), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            logged_in = str(baostock.login().error_code) == "0"
    except (Exception, pipeline.RequestTimeout):
        return False
    if not logged_in:
        return False
    try:
        with pipeline.deadline(3), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            baostock.logout()
    except (Exception, pipeline.RequestTimeout):
        pass
    return True


def resume_after_login(output, interval, attempts):
    checkpoint = output / "checkpoint.json"
    for attempt in range(attempts):
        state = pipeline.read_json(checkpoint)
        if (output / "p0_blocker.json").exists():
            print("Recovery stopped at a P0 evidence blocker", flush=True)
            return 2
        if state["state"] == "DONE" and not state.get("paused_sources") and not state.get("circuit"):
            return 0
        if not can_probe(state):
            print("Checkpoint requires diagnosis before source recovery", flush=True)
            return 2
        success = probe_login()
        print(f"BaoStock anonymous login check {attempt + 1}/{attempts}: "
              f"{'SUCCESS' if success else 'UNAVAILABLE'}", flush=True)
        if success:
            return subprocess.run([sys.executable, "-u", "-m", "ashare.p4_action_pipeline",
                                   "resume", "--restore-source", pipeline.BAO,
                                   "--output", str(output)], check=False).returncode
        if attempt + 1 < attempts:
            time.sleep(interval)
    return 2


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=pipeline.OUTPUT)
    parser.add_argument("--interval", type=int, default=2700)
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()
    if args.interval < 1800 or not 1 <= args.attempts <= 8:
        parser.error("Use intervals of at least 30 minutes and at most 8 checks")
    raise SystemExit(resume_after_login(args.output, args.interval, args.attempts))


if __name__ == "__main__":
    main()
