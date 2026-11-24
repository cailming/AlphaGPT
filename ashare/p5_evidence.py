"""Bounded, signed dual-provider event observations for P5 research windows."""
from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from decimal import InvalidOperation
import hashlib
from pathlib import Path
import time

from . import p4_action_pipeline as ledger
from .p4_subset_inputs import factor_query, hash_file, validate_factor_rows

DAILY_INTERVAL = 1.15
RECENT_SOURCES = ("bao_adjust", "sina_dividend")


def observation_path(root, stock, provider):
    return root / "observations" / f"{stock}.{provider}.json"


def verified_observation(path, job, provider, plan, plan_hash):
    envelope = ledger.read_json(path)
    item = envelope["observation"]
    if (envelope["sha256"] != hashlib.sha256(ledger.canonical(item)).hexdigest()
            or item["job"] != job or item["provider"] != provider or item["plan_sha256"] != plan_hash
            or item["start"] != plan["start"] or item["end"] != plan["end"]):
        raise ValueError(f"Observation does not match frozen P5 capture plan: {path}")
    if item["status"] == "SUCCESS":
        if provider == "bao_adjust":
            validate_factor_rows(item["rows"], job["provider_code"], item["start"], item["end"])
        else:
            for row in item["rows"]:
                if (row["request_code"] != job["stock_code"] or row["进度"] != "实施"
                        or not item["start"] <= row["除权除息日"] <= item["end"]
                        or row["公告日期"] > item["end"]):
                    raise ValueError(f"Invalid P5 dividend observation: {path}")
    return item


def capture_observations(root: Path):
    import baostock as bs

    plan_path = root / "evidence_plan.json"
    plan = ledger.read_json(plan_path)
    plan_hash = hash_file(plan_path)
    manifest = ledger.read_json(root / "inputs.json")
    if manifest["files"]["evidence_plan.json"] != plan_hash:
        raise ValueError("P5 capture plan digest mismatch")
    if plan["providers"] != list(RECENT_SOURCES) or not 1 <= len(plan["jobs"]) <= 220:
        raise ValueError("P5 recent observations must use the bounded dual-provider plan")
    ak = ledger.free._import_akshare()
    (root / "observations").mkdir(exist_ok=True)
    first, last = plan["start"], plan["end"]
    exclusive = (date.fromisoformat(last) + timedelta(days=1)).isoformat()
    summary = Counter()
    failures = Counter()
    paused = set()
    logged_in = False
    checked = 0
    with ledger.exclusive(root):
        try:
            try:
                with ledger.deadline(12):
                    logged_in = str(bs.login().error_code) == "0"
            except (OSError, ledger.RequestTimeout):
                pass
            if not logged_in:
                paused.add("bao_adjust")
            for job in plan["jobs"]:
                for provider in RECENT_SOURCES:
                    path = observation_path(root, job["stock_code"], provider)
                    if path.is_file():
                        item = verified_observation(path, job, provider, plan, plan_hash)
                        summary[provider + "/" + item["status"]] += 1
                        continue
                    if provider in paused:
                        continue
                    started = time.monotonic()
                    details, rows = {}, []
                    try:
                        if provider == "bao_adjust":
                            rows = factor_query(bs, job["provider_code"], first, last)
                            status = "SUCCESS"
                        else:
                            with ledger.deadline(10):
                                details, frame = ledger.free.fetch_request(
                                    ak, job["stock_code"], "sina_dividend", cutoff=exclusive, row_cap=1000,
                                    audit_start=first)
                            frame = frame.astype("string").fillna("")
                            keep = frame["进度"].eq("实施") & frame["除权除息日"].between(first, last)
                            rows = frame.loc[keep].to_dict("records")
                            status = details["status"]
                        if status in ("SUCCESS", "UNKNOWN_EX_DATE"):
                            failures[provider] = 0
                        else:
                            failures[provider] += 1
                    except (ledger.RequestTimeout, InvalidOperation, ValueError, OSError) as error:
                        status = "SOURCE_ERROR"
                        details = {"error": type(error).__name__}
                        failures[provider] += 1
                    item = {"job": job, "provider": provider, "status": status, "start": first, "end": last,
                            "observed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "plan_sha256": plan_hash, "details": details, "rows": rows}
                    ledger.atomic_json(path, {"observation": item,
                                             "sha256": hashlib.sha256(ledger.canonical(item)).hexdigest()})
                    checked += 1
                    summary[provider + "/" + status] += 1
                    if checked % 20 == 0 or status not in ("SUCCESS", "UNKNOWN_EX_DATE"):
                        print(f"Recent evidence {checked}/{2*len(plan['jobs'])}: {provider} {job['stock_code']} {status}",
                              flush=True)
                    if failures[provider] >= 4:
                        paused.add(provider)
                    time.sleep(max(0, DAILY_INTERVAL - (time.monotonic() - started)))
        finally:
            if logged_in:
                try:
                    with ledger.deadline(5):
                        bs.logout()
                except (OSError, ledger.RequestTimeout):
                    pass
    print({"counts": dict(summary), "paused": sorted(paused)}, flush=True)
