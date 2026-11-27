"""Inspect installed research scheduling and verify frozen output integrity."""
from __future__ import annotations

import argparse
from datetime import time
from pathlib import Path
import subprocess
from importlib import import_module

from .p4_action_pipeline import atomic_json, read_json
from .p4_subset_inputs import hash_file
from .p58_fund_snapshot import verified as verify_fund
from .p55_daily import now_shanghai
from .p55_reports import artifact, read_panel


def verify_files(folder, expected):
    for name, digest in expected.items():
        artifact(folder, name, digest)


def scheduler(root, log, moment):
    cron = subprocess.run(["crontab", "-l"], check=True, capture_output=True).stdout
    active = subprocess.run(["systemctl", "is-active", "cron"], capture_output=True, text=True)
    today = root / "schedule-logs" / moment.date().isoformat()
    invocations = {phase: sorted(p.name for p in today.glob(f"{phase}.*.log")) for phase in ("close", "post")}
    phases = {}
    for name, due in (("close", time(17, 55)), ("post", time(18, 10))):
        phases[name] = ("OBSERVED_INVOCATION" if invocations[name] else
                        "NON_WEEKDAY" if moment.weekday() >= 5 else
                        "NOT_YET_DUE" if moment.time() < due else "NO_RUN_LOG_OBSERVED")
    return {"cron_active": active.returncode == 0 and active.stdout.strip() == "active",
            "installed_crontab_matches_archive": cron == (root / "schedule-backup/cron.after").read_bytes(),
            "schedule_sha256": hash_file(root / "code/p58_research_schedule.sh"),
            "invocations_today": invocations, "today_states": phases,
            "latest_nightly_log": log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]}


def inspect(root, legacy, panel, factor_model, group_model, moment):
    paths = {name: directory / "policy.json" for name, directory in {
        "panel": panel, "group_c": root / "group_c", "portfolio": root / "portfolio"}.items()}
    policies = {name: read_json(path) for name, path in paths.items()}
    weights = {name: hash_file(directory / "best_model.pt") for name, directory in {
        "factors18": factor_model, "group_c": group_model}.items()}
    if (weights["factors18"] != policies["panel"]["dependencies"]["factor_model_weights_sha256"]
            or weights["group_c"] != policies["group_c"]["weights_sha256"]):
        raise ValueError("Archived model policy and actual weight digest differ")
    if policies["group_c"]["panel_policy_sha256"] != hash_file(paths["panel"]):
        raise ValueError("C group dependency no longer matches its original panel")
    implementation = policies["portfolio"]["implementation_sha256"]
    for name, expected in implementation.items():
        module = import_module(f".{Path(name).stem}", package=__package__)
        if hash_file(Path(module.__file__)) != expected:
            raise ValueError(f"Portfolio policy and running implementation differ for {name}")
    captures, rows, manifests, _ = read_panel(panel, legacy)
    latest = max((r["day"] for r in captures if r["track"] != "HISTORICAL" and r["formal"]), default=None)
    recent = {r["lane"]: r for r in captures if r["day"] == latest and r["track"] != "HISTORICAL"}
    audit_path = panel / "reports" / latest / "panel_report_audit.json" if latest else None
    report = {}
    if audit_path and audit_path.is_file():
        report = read_json(audit_path)
        verify_files(audit_path.parent, report["files"])
    forward = root / "portfolio/forward"
    groups = root / "group_c/timely"
    snapshots = []
    for path in sorted((root / "fund").glob("????-??-??/*/r*/manifest.json")):
        snapshot = verify_fund(path)
        snapshots.append({"path": str(path.relative_to(root)), "sha256": hash_file(path),
                          "verified_parquet_files": len(snapshot["sources"])})
    return {
        "observed_at": moment.isoformat(),
        "scheduler": scheduler(root, legacy.parent / "nightly.log", moment),
        "policy_sha256": {name: hash_file(path) for name, path in paths.items()},
        "portfolio_running_source_sha256": implementation,
        "frozen_model_sha256": weights, "capture_manifests_verified": len(manifests),
        "frozen_prediction_rows_verified": len(rows), "latest_timely_formal_day": latest,
        "latest_models": {name: {"status": r["status"], "candidates": r["candidates"],
                                 "abstentions": r["abstention_reasons"],
                                 "rows": int(((rows.trade_date == latest) & (rows.track == r["track"])
                                              & (rows.objective == name)).sum())}
                          for name, r in recent.items()},
        "maturity_revision_sha256": hash_file(audit_path) if report else None,
        "maturity_through": report.get("formal_through"),
        "forward_account_days": sorted(p.parent.name for p in forward.glob("????-??-??/portfolio_summary.json")),
        "group_c_timely_days": sorted(p.parent.name for p in groups.glob("????-??-??/manifest.json")),
        "fund_snapshots": snapshots,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "legacy", "panel", "factor-model", "group-model", "out"):
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    audit = inspect(args.root, args.legacy, args.panel, args.factor_model, args.group_model, now_shanghai())
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(args.out / "runtime_audit.json", audit)
    print(args.out / "runtime_audit.json")


if __name__ == "__main__":
    main()
