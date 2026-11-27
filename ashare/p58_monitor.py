"""Read-only daily operations audit for the six frozen P5 methods."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import os

import pandas as pd

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, exclusive, read_json
from .p4_subset_inputs import hash_file
from .p54_factors import build_market_factors
from .p55_daily import now_shanghai, parse_clock
from .p55_reports import LANES, artifact, read_panel
from .p5_sequences import formal_snapshot


def breadth(raw, length=16):
    days = sorted(pd.to_datetime(raw.trade_date).unique())
    stock = str(raw.stock_code.iloc[0])
    window = pd.DataFrame({"stock_code": stock, "trade_date": days[-length:]})
    factors, audit = build_market_factors(raw, window)
    observed = pd.DataFrame(audit["market_session_support"]).tail(length + 20)
    invalid = observed.loc[~observed.market_session_usable]
    cols = ["market_ret_20", "market_turnover_log_change_20"]
    unavailable = factors.loc[factors[cols].isna().any(axis=1), "trade_date"].tolist()
    return {"formal_counts": raw.groupby("trade_date").stock_code.nunique().tail(3).rename(
        index=lambda d: str(d)[:10]).to_dict(), "pair_support": observed.to_dict("records"),
        "failed_market_days": invalid.trade_date.tolist(), "missing_20d_feature_dates": unavailable,
        "valid_20d_sessions_in_last_sequence": len(window) - len(unavailable),
        "current_market_pair_usable": bool(observed.market_session_usable.iloc[-1]),
        "minimum_pair_share": audit["minimum_market_pair_share"],
        "minimum_pair_stocks": audit["minimum_market_pair_stocks"]}


def collect(legacy, panel, moment):
    signal, raw = formal_snapshot(AShareDailyLoader().engine, moment)
    day = signal.isoformat()
    report = {"observed_at": moment.isoformat(), "formal_through": day, "issues": [],
              "formal": breadth(raw), "models": {}, "protocol_evaluation": [],
              "formal_freshness": "CURRENT_SESSION" if signal == moment.date() else "LAST_AVAILABLE",
              "calendar_scope": "observed formal sessions; absences of an entire exchange session need independent confirmation"}
    if signal < moment.date() and moment.weekday() < 5 and moment.hour >= 18:
        report["issues"].append("NO_CURRENT_FORMAL_BARS_CHECK_EXCHANGE_CALENDAR")
    if not report["formal"]["current_market_pair_usable"]:
        report["issues"].append("LOW_MARKET_PAIR_COVERAGE")
    records, rows, _, _ = read_panel(panel, legacy)
    report["archived_prediction_ids"] = len(rows)
    report["duplicate_prediction_ids"] = int(rows.prediction_id.duplicated().sum())
    sessions = sorted(set(str(d)[:10] for d in raw.trade_date))
    observed_keys = {(r["lane"],r["day"]) for r in records if r["track"] != "HISTORICAL"}
    expected = {r["lane"]: r["freeze_day"] for r in records if r["track"] != "HISTORICAL"}
    # Even a never-published new lane has an expected first day after policy freeze.
    for lane in ("factors18", "ma20_reversal", "equal_weight"):
        expected[lane] = max(expected.get(lane, ""), read_json(panel/"policy.json")["frozen_at"][:10])
    report["missing_timely"] = {lane: [d for d in sessions if d > frozen and
        (lane, d) not in observed_keys]
        for lane, frozen in expected.items()}
    if any(report["missing_timely"].values()):
        report["issues"].append("MISSING_EXPECTED_CAPTURE")
    for r in records:
        if r["day"] != day or r["track"] == "HISTORICAL":
            continue
        path = Path(r["source"])
        manifest = read_json(path)
        spec = manifest["models"][r["lane"]]
        report["models"][r["lane"]] = {"track": r["track"], "committed_at": manifest["committed_at"],
            "scored": spec["rows"], "candidates": r["candidates"], "abstentions": r["abstention_reasons"],
            "status": r["status"], "weights_sha256": r["model_sha256"]}
        committed = parse_clock(manifest["committed_at"])
        if committed.date().isoformat() != day or committed.hour < 18:
            report["issues"].append("LATE_OR_OUT_OF_WINDOW_" + r["lane"])
        if manifest["market_rows"] != report["formal"]["formal_counts"][day]:
            report["issues"].append("FORMAL_COUNT_CHANGED_SINCE_CAPTURE_" + r["lane"])
        if spec["rows"] == 0:
            report["issues"].append("ZERO_PREDICTIONS_" + r["lane"])
        evidence = manifest.get("evidence_snapshot") or {}
        if evidence.get("root"):
            evidence_file = Path(evidence["root"])/"dataset/inference_evidence.json"
            report["evidence"] = {k:evidence.get(k) for k in ("sha256","observed_at_utc","max_covered_date")}
            if hash_file(evidence_file) != evidence["sha256"]:
                raise ValueError("Company evidence file failed digest verification")
            if (evidence.get("max_covered_date") or "") < day:
                report["issues"].append("STALE_EVIDENCE")
    if set(report["models"]) != set(LANES):
        report["issues"].append("SIX_METHOD_CAPTURE_INCOMPLETE")
    source = panel/"reports"/day/"panel_report_audit.json"
    if source.exists():
        audit = read_json(source)
        for name, digest in audit["files"].items():
            artifact(source.parent, name, digest)
        report["maturity_asof"] = audit["formal_through"]
        report["label_statuses_all_tracks"] = audit["label_statuses"]
        report["protocol_evaluation"] = [r for r in audit["summary"] if
            r["track"] == "PROSPECTIVE" and r["scope"] == "OWN" and r["support_group"] == "ALL"]
        if audit["formal_through"] != day:
            report["issues"].append("STALE_MATURITY_REPORT")
    else:
        report["issues"].append("MISSING_MATURITY_REPORT")
    return report


def publish(out, report):
    directory = out/report["observed_at"][:10]
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive(directory):
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            stage = Path(tmp)/"report"
            stage.mkdir()
            atomic_json(stage/"status.json", report)
            models = report.get("models", {})
            lines = [f"# P5.8 运行监控 · {report['observed_at']}", "",
                     f"正式行情 {report.get('formal_through')}。异常或待查项 {', '.join(report['issues']) or '无'}。", "",
                     "| 方法 | 候选 | 预测 | 弃权原因 |", "|---|---:|---:|---|"]
            lines.extend(f"| {lane} | {r['candidates']} | {r['scored']} | {r['abstentions']} |"
                         for lane, r in models.items())
            formal = report.get("formal", {})
            lines += ["", f"最近正式行情股票数 {formal.get('formal_counts')}。完整性按配对和已归档计数核对。",
                      f"市场配对不足的日期 {formal.get('failed_market_days')}。",
                      f"16 日序列中 20 日市场因子缺值的日期 {formal.get('missing_20d_feature_dates')}。",
                      f"累计重复 ID {report.get('duplicate_prediction_ids')}。公司行为观察 {report.get('evidence')}。",
                      f"前瞻漏记日期 {report.get('missing_timely')}。各来源标签状态 {report.get('label_statuses_all_tracks')}。"]
            for r in report.get("protocol_evaluation", []):
                lines.append(f"- {r['lane']} 已成熟标签 {r['mature_labels']}，待成熟 {r['pending_predictions']}，有效联合评价日 {r['jointly_valid_days']}。")
            if "error" in report:
                lines.append(report["error"])
            (stage/"status.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
            revision = directory/report["observed_at"].replace(":", "-")
            os.rename(stage,revision)
            return revision


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for param in ("legacy","panel","out"):
        cli.add_argument("--"+param, type=Path, required=True)
    args = cli.parse_args()
    now = now_shanghai()
    try:
        report = collect(args.legacy,args.panel,now)
    except Exception as exc:
        publish(args.out, {"observed_at": now.isoformat(), "issues":["AUDIT_FAILED"], "error":str(exc) if isinstance(exc,ValueError) else type(exc).__name__})
        raise
    print(publish(args.out,report))


if __name__ == "__main__":
    main()
