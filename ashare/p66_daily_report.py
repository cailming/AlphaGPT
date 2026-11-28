"""Publish a read-only index of frozen research records and forward paper accounts."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, time, timedelta
import hashlib
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import pandas as pd
from sqlalchemy import text

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, canonical, exclusive, read_json
from .p4_subset_inputs import hash_file, parquet_read
from .p55_daily import CHINA, SCORE_UNITS, now_shanghai
from .p55_panel import DEFINITIONS
from .p55_reports import LANES, artifact, model_capture
from .p58_fund_snapshot import verified as verified_fund
from .p6_portfolio_run import checked_day

VERSION = "P66_RESEARCH_DAILY_V1.1"
METHODS = (*LANES, "group_c")
UNITS = dict(SCORE_UNITS) | {k: v["score_units"] for k, v in DEFINITIONS.items()} | {
    "group_c": "five_session_economic_return"}
MISSING = "MISSING_ARCHIVE"


def formal_session(engine, moment):
    cutoff = moment.date() + timedelta(days=moment.time() >= time(18))
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn, conn.begin():
        conn.execute(text("SET TRANSACTION READ ONLY"))
        day = conn.scalar(text("SELECT max(trade_date) FROM public.market WHERE not_billing=1 AND trade_date < :cut"),
                          {"cut": cutoff})
        if day is None:
            raise ValueError("No observed formal market session")
        codes = set(conn.scalars(text("SELECT DISTINCT stock_code FROM public.market WHERE not_billing=1 "
                                      "AND trade_date=:day AND stock_code IS NOT NULL"), {"day": day}))
    return day.isoformat(), codes


def source_note(path):
    return {"path": str(path), "sha256": hash_file(path),
            "file_updated_at": datetime.fromtimestamp(path.stat().st_mtime, CHINA).isoformat()}


def json_source(path, sources):
    data = read_json(path)
    sources[str(path)] = source_note(path)
    return data


def capture_models(path, day, expected, sources):
    if not path.is_file():
        return {k: {"status": MISSING} for k in expected}, None
    rec = json_source(path, sources)
    if (rec["trade_date"] != day or rec["capture_kind"] != "TIMELY" or
            rec["uses_future_labels"] or set(rec["models"]) != set(expected)):
        raise ValueError(f"Unexpected capture identity: {path}")
    track = "PROSPECTIVE" if rec["prospective_eligible"] else "TIMELY_NONPROSPECTIVE"
    result = {}
    for name, spec in rec["models"].items():
        model = model_capture(rec, track, name, spec, path)
        pred = artifact(path.parent, spec["predictions_file"], spec["predictions_sha256"])
        artifact(path.parent, spec["candidates_file"], spec["candidates_sha256"])
        rows = parquet_read(pred)
        if (len(rows) != spec["rows"] or rows.stock_code.duplicated().any()
            or not rows.trade_date.eq(day).all() or not rows.objective.eq(name).all()
            or not rows.model_sha256.eq(spec["weights_sha256"]).all() or not rows.score_units.eq(UNITS[name]).all()):
            raise ValueError(f"Archived predictions disagree with manifest: {pred}")
        ranked = name != "equal_weight"
        names = rows.sort_values(["score", "stock_code"], ascending=(False, True)).stock_code.tolist() if ranked else []
        for count in (10, 20):
            key = f"top{count}"
            if key in spec and spec[key] != names[:count]:
                raise ValueError(f"Published {key} differs from archived scores: {pred}")
            model[key] = names[:count] if ranked else None
        result[name] = model | {"predictions": len(rows), "committed_at": rec["committed_at"],
                                "predictions_sha256": spec["predictions_sha256"], "units": UNITS[name]}
    return result, rec


def scope_metrics(report, table, lane, scope, model_sha):
    if not report:
        return {"status": MISSING}
    matches = [s for s in report["summary"] if s["track"] == "PROSPECTIVE" and s["lane"] == lane
               and s["scope"] == scope and s["support_group"] == "ALL"
               and (model_sha is None or s["model_sha256"] == model_sha)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous model version for {lane}/{scope}")
    if not matches:
        return {"status": "NO_PROSPECTIVE_RECORDS", "published_days": 0, "mature_label_sessions": 0,
                "rank_ic_days": 0, "full_top20_days": 0, "label_coverage": None}
    entry = matches[0]
    dates = {r["day"] for r in table if r["lane"] == lane and r["scope"] == scope and r["track"] == "PROSPECTIVE"
             and r["support_group"] == "ALL" and r["model_sha256"] == entry["model_sha256"]
             and r["captured"] == "True" and r["formal"] == "True"}
    return entry | {"published_days": len(dates)}


def evaluation(path, table_name, sources, day):
    if not path.is_file():
        return None, []
    data = json_source(path, sources)
    if data["formal_through"] != day:
        raise ValueError(f"Evaluation date differs from formal session: {path}")
    digest = data["files"][table_name] if "files" in data else data["daily_sha256"]
    with artifact(path.parent, table_name, digest).open(encoding="utf-8", newline="") as handle:
        return data, list(csv.DictReader(handle))


def funds(root, day, codes, sources):
    result = []
    for path in sorted((root / "fund" / day).glob("*/r*/manifest.json")):
        snapshot = verified_fund(path)
        sources[str(path)] = source_note(path)
        if snapshot["signal_date"] != day:
            raise ValueError("Incorrect fund signal date")
        counts = {}
        for name, stats in snapshot["sources"].items():
            frame = parquet_read(path.parent / (name + ".parquet"))
            complete = frame.dropna(subset=["stock_code", *stats["required_fields"]])
            valid = set(complete.stock_code)
            if len(valid) != stats["valid_stocks"]:
                raise ValueError("Fund archive validity differs from source manifest")
            covered = len(valid & codes)
            counts[name] = {"rows": stats["rows"], "valid_stocks": len(valid), "formal_stocks_covered": covered,
                            "formal_universe_coverage": covered / len(codes) if codes else None}
        result.append({k: snapshot[k] for k in ("phase", "availability", "revision", "snapshot_at", "files_ready_at")}
                      | {"sources": counts})
    return result


def accounts(root, day, sources):
    policy_path = root / "policy.json"
    if not policy_path.is_file():
        return {"status": "AWAITING_POLICY_FREEZE"}
    policy = json_source(policy_path, sources)
    policy_sha = sources[str(policy_path)]["sha256"]
    files = sorted(p for p in (root / "forward").glob("????-??-??/portfolio_summary.json") if p.parent.name <= day)
    if not files:
        return {"status": "AWAITING_FIRST_POST_FREEZE_SIGNAL", "frozen_at": policy["frozen_at"]}
    path = files[-1]
    previous_sha = hash_file(files[-2]) if len(files) > 1 else None
    checkpoint, _ = checked_day(path.parent, previous_sha, policy_sha)
    summary = json_source(path, sources)
    details = {}
    for row in summary["accounts"]:
        key = row["model"] + "/" + row["strategy"]
        details[key] = row | {"unknown_claims": checkpoint["accounts"][key]["unknown_claims"]}
    risk_file = path.parent / "exploratory_risks.json"
    extra = None
    if risk_file.exists():
        artifact(path.parent, risk_file.name, summary["files"][risk_file.name])
        extra = json_source(risk_file, sources)
    return {"status": "CURRENT" if path.parent.name == day else "STALE_ACCOUNT_DAY", "date": path.parent.name,
            "frozen_at": policy["frozen_at"], "initial_cash": policy["config"]["initial_cash"],
            "observed_at": summary["observed_at"], "timing": summary["timing"], "accounts": details,
            "trades": summary["trade_count"], "rejection_reasons": summary["rejection_reasons"],
            "current_risk": extra, "ledger": str(path.parent)}


def scheduler(root, day):
    folder = root / "schedule-logs" / day
    phases = {}
    for phase in ("close", "post"):
        runs = []
        for path in folder.glob(phase + ".*.log"):
            lines = path.read_text().splitlines()
            matched = [m for line in lines if (m := re.search(rf"^(\S+) {phase} EXIT=(\d+)$", line))]
            runs.append({"log": str(path), "started_at": lines[0].split()[0] if lines else None,
                         "file_updated_at": datetime.fromtimestamp(path.stat().st_mtime, CHINA).isoformat(),
                         "exit": int(matched[-1][2]) if matched else None,
                         "failed_steps": [line for line in lines if re.search(r" EXIT=[1-9]\d*$", line)]})
        phases[phase] = sorted(runs, key=lambda r: r["started_at"] or "")
    return phases


def nightly_results(path, day):
    if not path.is_file():
        return []
    results = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(day):
                match = re.fullmatch(r"(\S+) frozen_exit=(\d+) research_exit=(\d+)", line.rstrip())
                if match:
                    results.append({"finished_at": match[1], "frozen_exit": int(match[2]),
                                    "research_exit": int(match[3])})
    return results


def patch_diagnostic(root, day, sources):
    path = root / "p65/quote-patch-daily/runs" / day / "enhanced_audit.json"
    if not path.is_file():
        return {"status": "NO_DIAGNOSTIC_FOR_SESSION"}
    report = json_source(path, sources)
    latest = report["latest_formal"]
    if report["track"] != "RESEARCH_QUOTE_PATCH_DIAGNOSTIC" or report["forward_eligible"] or latest["date"] != day:
        raise ValueError("Invalid quote patch track or date")
    name = "research_scenario_scores.csv"
    frame = pd.read_csv(artifact(path.parent, name, report["files"][name]), dtype={"stock_code": str})
    if not frame.signal_date.eq(day).all() or frame.stock_code.duplicated().any() or not frame.track.eq(report["track"]).all():
        raise ValueError("Malformed diagnostic scores")
    models = {}
    for lane, column in (("factors18", "frozen_full18_score"), ("group_c", "frozen_group_c_score")):
        ranked = frame.dropna(subset=[column]).sort_values([column, "stock_code"], ascending=(False, True))
        models[lane] = {"n": len(ranked), "top10": ranked.stock_code.head(10).tolist(),
                        "top20": ranked.stock_code.head(20).tolist()}
    return {"status": latest["status"], "track": report["track"], "forward_eligible": False, "models": models,
            "observed_at": report["observed_at"], "score_sha256": report["files"][name]}


def calendar_status(root, day, now, sources):
    files = sorted((root / "p66/calendar").glob("????-??-??/*.json"), reverse=True)
    for path in files:
        if path.parent.name <= now.date().isoformat():
            data = json_source(path, sources)
            if data["through"] != path.parent.name:
                raise ValueError("Calendar coverage boundary differs from dated archive")
            return data | {"status": "CURRENT" if data["through"] >= day else "STALE"}
    return {"status": MISSING}


def calendar_review(obs, rows, lane, scope, calendar, rules):
    """Apply the calendar check only to published, mature, version-matched holds."""
    impacted, usable = [], []
    coverage = calendar.get("days", {})
    for row in rows:
        if (row["track"] != "PROSPECTIVE" or row["lane"] != lane or row["scope"] != scope
                or row["support_group"] != "ALL" or row["model_sha256"] != obs.get("model_sha256")
                or row["captured"] != "True" or row["formal"] != "True" or row["signal_matured"] != "True"
                or not int(row["predictions"])):
            continue
        start = datetime.fromisoformat(row["day"]).date()
        end = datetime.fromisoformat(row["exit_date"]).date()
        window = [(start + timedelta(days=i)).isoformat() for i in range((end-start).days+1)]
        invalid = [d for d in window if coverage.get(d, {}).get("status")
                   not in ("FORMAL_OBSERVED", "SCHEDULED_CLOSED")]
        if invalid:
            impacted.append({"signal": row["day"], "exit": row["exit_date"], "unverified_days": invalid})
        else:
            usable.append(row)
    prediction_count = sum(int(r["predictions"]) for r in usable)
    label_count = sum(int(r["labels"]) for r in usable)
    share = label_count / prediction_count if prediction_count else None
    joint = sum(bool(r["rank_ic"]) and r["top20_full"] == "True" for r in usable)
    status = obs["status"]
    if impacted and lane != "equal_weight":
        sufficient = (rules and joint >= rules["minimum_joint_prospective_sessions"]
                      and (share or 0) >= rules["minimum_mature_label_coverage"])
        status = "READY_FOR_REVIEW_UNAFFECTED_WINDOWS" if sufficient else "CALENDAR_REVIEW_PENDING"
    if calendar["status"] == MISSING and status == "READY_FOR_REVIEW":
        status = "CALENDAR_REVIEW_PENDING"
    return {"status": status, "affected_windows": impacted, "verified_joint_days": joint,
            "verified_window_label_coverage": share}


def fund_factors(root, day, sources):
    folder = root / "p66/fund"
    paths = sorted((folder / "observations").glob("????-??-??/*.json"))
    if not paths:
        return {"status": MISSING}
    obs = json_source(paths[-1], sources)
    capture = obs["capture"]
    result = {"observed_at": obs["observed_at"], **capture, "evaluation": obs.get("evaluation"), "scores": {}}
    if capture.get("entry") and capture.get("signal") == day:
        from .p66_fund_research import read_capture
        policy_sha = hash_file(folder / "policy.json")
        for path in sorted((folder / "predictions" / capture["entry"]).glob("*/manifest.json")):
            data = read_capture(path, policy_sha)
            if data["signal"] != day:
                raise ValueError("Fund signal date differs from declared source observation")
            sources[str(path)] = source_note(path)
            result["scores"][data["source"]] = {k: data[k] for k in
                ("signal", "entry", "captured_at", "deadline", "files_ready_at", "pool_size", "factors", "scores_sha256")}
    return result


def collect(root, legacy, panel, day, codes, now):
    sources = {}
    issues = []
    formal = {"date": day, "stocks": len(codes)}
    monitor_root = root / "monitor"
    monitors = sorted(monitor_root.glob("????-??-??/*/status.json"))
    if monitors:
        monitor = json_source(monitors[-1], sources)
        formal["monitor_formal_through"] = monitor["formal_through"]
        formal["monitor_observed_at"] = monitor["observed_at"]
        if monitor["formal_through"] == day:
            issues.extend(monitor["issues"])
            formal["missing_factor_input_dates"] = monitor["formal"]["missing_20d_feature_dates"]
        else:
            issues.append("MONITOR_STALE")
    else:
        issues.append("MONITOR_MISSING")
    group = root / "group_c"
    models, parents = {}, []
    for base, lanes in ((legacy, LANES[:3]), (panel, LANES[3:]), (group, ("group_c",))):
        path = base / "timely" / day / "manifest.json"
        pieces, rec = capture_models(path, day, lanes, sources)
        if rec and base != legacy:
            required = parents if base == group else parents[:1]
            for key, sha in zip(("legacy_manifest_sha256", "panel_manifest_sha256"), required, strict=False):
                if sha is None:
                    issues.append("MISSING_PARENT_FOR_" + base.name)
                    pieces = {lane: {"status": "PARENT_UNAVAILABLE"} for lane in lanes}
                    break
                if rec[key] != sha:
                    raise ValueError("Timely forecast dependency mismatch")
        models.update(pieces)
        parents.append(sources.get(str(path), {}).get("sha256"))
    six_path = panel / "reports" / day / "panel_report_audit.json"
    six, six_rows = evaluation(six_path, "cumulative.csv", sources, day)
    seven, seven_rows = evaluation(group / "reports" / day / "comparison.json", "daily_comparison.csv", sources, day)
    if seven and not six:
        issues.append("SEVEN_METHOD_EVALUATION_PARENT_UNAVAILABLE")
        seven, seven_rows = None, []
    if seven and seven["six_method_audit_sha256"] != sources[str(six_path)]["sha256"]:
        raise ValueError("Seven-method evaluation revision differs from six-method source")
    progress = {}
    plan = calendar_status(root, day, now, sources)
    if plan["status"] != "CURRENT":
        issues.append("CALENDAR_" + plan["status"])
    if plan.get("missing_formal"):
        issues.append("SCHEDULED_OPEN_MISSING_FORMAL_SESSIONS")
    for lane, model in models.items():
        sha = model.get("model_sha256")
        source, rows = (seven, seven_rows) if lane == "group_c" else (six, six_rows)
        progress[lane] = {"own": scope_metrics(source, rows, lane, "OWN", sha),
                          "common7": scope_metrics(seven, seven_rows, lane, "COMMON7", sha)}
        for name, table in (("own", rows), ("common7", seven_rows)):
            metric = progress[lane][name]
            metric["calendar_review"] = calendar_review(metric, table, lane, name.upper(), plan,
                                                        six.get("forward_rules", {}) if six else {})
        if model["status"] == MISSING:
            issues.append("MISSING_TIMELY_" + lane)
        elif model.get("predictions") == 0 and "ZERO_PREDICTIONS_" + lane not in issues:
            issues.append("ZERO_PREDICTIONS_" + lane)
    if not six or not seven:
        issues.append("EVALUATION_NOT_PUBLISHED")
    accounts_by_track = {"strict": accounts(root / "portfolio", day, sources),
                         "exploratory": accounts(root / "p66/exploratory", day, sources)}
    snapshots = funds(root, day, codes, sources)
    if not snapshots:
        issues.append("NO_FUND_SNAPSHOT")
    logs = scheduler(root, day)
    wrapper = nightly_results(legacy.parent / "nightly.log", day)
    if not wrapper or any(wrapper[-1][key] for key in ("frozen_exit", "research_exit")):
        issues.append("NIGHTLY_SUCCESS_NOT_CONFIRMED")
    for phase, runs in logs.items():
        if not runs or runs[-1]["exit"] != 0:
            issues.append("SCHEDULE_NOT_CONFIRMED_" + phase)
    candidates = sorted({r["candidates"] for r in models.values() if "candidates" in r})
    if len(candidates) > 1:
        issues.append("MODEL_CANDIDATE_COUNTS_DIFFER")
    return {"version": VERSION, "generated_at": now.isoformat(), "generation_date": now.date().isoformat(),
            "market": formal, "prediction_dates": sorted({r["day"] for r in models.values() if "day" in r}),
            "candidate_count": candidates[0] if len(candidates) == 1 else None, "snapshots": snapshots,
            "schedule": logs, "completed_nightly_runs": wrapper, "issues": issues,
            "models": models, "evaluation": progress,
            "evaluation_asof": {k: r["evaluated_at"] if r else None for k, r in (("six", six), ("seven", seven))},
            "rules": six.get("forward_rules") if six else None, "accounts": accounts_by_track,
            "calendar": {k: v for k, v in plan.items() if k != "days"}, "fund_factor_research": fund_factors(root, day, sources),
            "diagnostic": patch_diagnostic(root, day, sources), "source_files": sources}


def display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown(report):
    market, models = report["market"], report["models"]
    lines = ["# A 股研究日报", "", f"生成于北京时间 {report['generated_at']}。正式行情 {market['date']}，"
             f"共 {market['stocks']} 股。预测日期 {', '.join(report['prediction_dates']) or '尚无归档'}，"
             f"候选 {display(report['candidate_count'])} 股。", "",
             "## 系统与资金覆盖", "", "监控事项 " + ("、".join(report["issues"]) or "无已记录异常") + "。", ""]
    for phase, runs in report["schedule"].items():
        last = runs[-1] if runs else {}
        state = (f"退出码 {last['exit']}" if last.get("exit") is not None else "尚无完成记录")
        lines.append(f"- {phase} 共 {len(runs)} 轮，最近开始于 {display(last.get('started_at'))}，{state}。")
    completed = report["completed_nightly_runs"]
    if completed:
        end = completed[-1]
        lines.append(f"- 最近完整夜间任务结束于 {end['finished_at']}，冻结预测退出码 {end['frozen_exit']}，"
                     f"辅助研究退出码 {end['research_exit']}。")
    lines += ["", "| 资金用途及版本 | 实际文件就绪时间 | 来源 | 行数 | 有效覆盖正式行情股数 / 总数 |",
              "|---|---|---|---:|---:|"]
    for snap in report["snapshots"]:
        for name, stats in snap["sources"].items():
            lines.append(f"| {snap['phase']} r{snap['revision']} | {snap['files_ready_at']} | {name} | "
                         f"{stats['rows']} | {stats['formal_stocks_covered']} / {market['stocks']} |")
    calendar = report.get("calendar", {})
    recovery = calendar.get("enhanced_recovery", {})
    missing = calendar.get("missing_formal")
    supported = calendar.get("supported_years", [])
    lines += ["", "## 交易日历与增强模型恢复", "",
              f"独立日历 {calendar.get('status', MISSING)}，截至 {display(calendar.get('through'))}；"
              f"应开未见正式行情日期 {('、'.join(missing) or '无') if missing is not None else '未知'}。"
              f"2027 年计划{'已载入' if 2027 in supported else '待正式安排核实'}。",
              f"监控截至 {display(recovery.get('formal_through'))}，按后续覆盖均合格的预计恢复日 "
              f"{display(recovery.get('earliest_clean_signal_if_future_pairs_pass'))}。"
              f"正式 18 因子 / C 组实际数量 {display(recovery.get('factors18_archived_scores'))} / "
              f"{display((recovery.get('current_c_capture') or {}).get('scores'))}。"]
    factor = report.get("fund_factor_research", {})
    lines += ["", "## 次开资金因子研究", "",
              f"观察 {display(factor.get('observed_at'))}。信号日 {display(factor.get('signal'))}，"
              f"目标开盘日 {display(factor.get('entry'))}，状态 {factor.get('status', MISSING)}。"]
    for source, item in factor.get("scores", {}).items():
        stats = "、".join(f"{name} {values['scored']}" for name, values in item["factors"].items())
        lines.append(f"- {source} 池 {item['pool_size']} 股，归档于 {item['captured_at']}。各因子评分股数 {stats}。")
    if factor.get("evaluation"):
        lines.append("评价归档 `" + factor["evaluation"] + "`。")
    lines += ["", "## 冻结模型预测", "", "| 方法 | 模型版本 | 单位 | 预测 / 候选 | 弃权原因 |",
              "|---|---|---|---:|---|"]
    for lane, rec in models.items():
        reasons = "、".join(f"{k} {v}" for k, v in rec.get("abstention_reasons", {}).items()) or rec["status"]
        lines.append(f"| {lane} | {display(rec.get('version'))} | {display(rec.get('units'))} | "
                     f"{display(rec.get('predictions'))} / {display(rec.get('candidates'))} | {reasons} |")
    lines.append("")
    for lane, rec in models.items():
        if lane == "equal_weight":
            lines.append("- equal_weight 对整个合格股票池等权，排名不适用。")
            continue
        for k in ("top10", "top20"):
            roster = rec.get(k)
            value = ("、".join(roster) or "零预测") if roster is not None else "尚无记录"
            lines.append(f"- {lane} {k}　{value}")
    diagnostic = report["diagnostic"]
    lines += ["", "## 补丁诊断", "", f"状态 {diagnostic['status']}。仅供诊断。"]
    for lane, info in diagnostic.get("models", {}).items():
        lines.append(f"- {lane} 共 {info['n']} 股。Top-10 {'、'.join(info['top10']) or '无'}。"
                     f"Top-20 {'、'.join(info['top20']) or '无'}。")
    lines += ["", "## 模拟账户", "",
              "| 账户 / 策略 | 初始资金 | 当前现金 | 持仓数 | 成交 / 拒单 | 累计费用 | 严格 / 指示净值 | "
              "累计净收益 / 最大回撤 | 未决权益 / 估值状态 |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for track, account in report["accounts"].items():
        for key, row in account.get("accounts", {}).items():
            lines.append(f"| {track} {key} | {display(account['initial_cash'])} | {display(row['closing_cash'])} | "
                         f"{row['positions']} | {row['trade_count']} / {row['rejections']} | "
                         f"{display(row['cumulative_fees'])} | {display(row['nav'])} / {display(row['indicative_nav'])} | "
                         f"{display(row['net_return'])} / {display(row['max_drawdown'])} | "
                         f"{row['unresolved_claims']} / {row['valuation_status']} |")
    for track, rec in report["accounts"].items():
        lines += ["", f"{track} 状态 {rec['status']}，账本日期 {display(rec.get('date'))}。"]
        if rec.get("ledger"):
            lines.append(f"成交与拒单明细位于 `{rec['ledger']}`。")
    lines += ["", "## 前瞻评价", "", "覆盖率分母为已到持有期终点的预测。收益按冻结协议计费，"
              "五日持有窗口可重叠。等权基准使用全池净超额；独立和七方法共同股票各自评价。", "",
              "| 方法 | 范围 | 前瞻记录日 | 有成熟标签日 | 标签覆盖率 | IC 有效日 | 完整 Top-20 日 | "
              "成本后超额 | 人工复核状态 |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for lane, scopes in report["evaluation"].items():
        for scope, rec in scopes.items():
            net = "portfolio_net_excess" if lane == "equal_weight" else "top20_net_excess"
            values = [lane, scope, rec.get("published_days"), rec.get("mature_label_sessions"),
                      rec.get("label_coverage"), rec.get("rank_ic_days"), rec.get("full_top20_days"),
                      rec.get(net, {}).get("mean"), rec.get("calendar_review", {}).get("status", rec["status"])]
            lines.append("| " + " | ".join(map(display, values)) + " |")
    lines += ["", "“—”表示未知、尚未成熟或不适用。JSON 保留 HAC 区间、缺失日及独立持有窗口数。", "",
              "## 报告与账本更新时间", "", "| 来源 | 文件时间（北京时间） |", "|---|---|"]
    for path, note in report["source_files"].items():
        lines.append(f"| `{path}` | {note['file_updated_at']} |")
    return "\n".join(lines) + "\n"


def progress_rows(report):
    for lane in METHODS:
        model = report["models"][lane]
        for scope, obs in report["evaluation"][lane].items():
            key = "portfolio_net_excess" if lane == "equal_weight" else "top20_net_excess"
            yield {"formal_date": report["market"]["date"], "report_generated_at": report["generated_at"],
                   "prediction_date": model.get("day"), "model": lane, "scope": scope,
                   "model_version": model.get("version"), "candidates": model.get("candidates"),
                   "predictions": model.get("predictions"), "capture_status": model["status"],
                   "published_days": obs.get("published_days"), "mature_label_days": obs.get("mature_label_sessions"),
                   "label_coverage": obs.get("label_coverage"), "valid_ic_days": obs.get("rank_ic_days"),
                   "complete_top20_days": obs.get("full_top20_days"), "net_excess": obs.get(key, {}).get("mean"),
                   "protocol_status": obs["status"],
                   "review_status": obs.get("calendar_review", {}).get("status", obs["status"])}


def publish(out, report):
    """Publish one atomic revision. latest/ holds the three corresponding outputs."""
    out.mkdir(parents=True, exist_ok=True)
    with exclusive(out):
        latest = out / "latest"
        previous = latest / "daily_research.json"
        day = report["market"]["date"]
        fingerprint = hashlib.sha256(canonical({k: v for k, v in report.items() if k != "generated_at"})).hexdigest()
        if previous.exists():
            old = read_json(previous)
            if old["version"] == VERSION:
                manifest = read_json(latest / "report_files.json")
                for name in ("daily_research.json", "daily_research.md", "research_progress.csv"):
                    artifact(latest, name, manifest[name])
            if old["market"]["date"] > day:
                raise ValueError("Cannot replace a newer daily research entry with an older session")
            if old["generated_at"] > report["generated_at"]:
                return latest
            if old["input_sha256"] == fingerprint:
                return latest
        report = report | {"input_sha256": fingerprint}
        rows = []
        prior_progress = latest / "research_progress.csv"
        if prior_progress.exists():
            with prior_progress.open(encoding="utf-8", newline="") as inp:
                rows = [r for r in csv.DictReader(inp) if r["formal_date"] != day]
        rows.extend(progress_rows(report))
        with TemporaryDirectory(dir=out, prefix=".building-") as tmp:
            stage = Path(tmp) / "report"
            stage.mkdir()
            atomic_json(stage / "daily_research.json", report)
            (stage / "daily_research.md").write_text(markdown(report), encoding="utf-8")
            with (stage / "research_progress.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[-1]))
                writer.writeheader()
                writer.writerows(rows)
            atomic_json(stage / "report_files.json", {p.name: hash_file(p) for p in stage.iterdir()})
            base = out / "revisions" / day
            base.mkdir(parents=True, exist_ok=True)
            revision = base / report["generated_at"].replace(":", "-")
            os.rename(stage, revision)
            link = Path(tmp) / "latest"
            link.symlink_to(revision.relative_to(out), target_is_directory=True)
            os.replace(link, latest)
        return latest


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "legacy", "panel", "out"):
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    now = now_shanghai()
    day, codes = formal_session(AShareDailyLoader().engine, now)
    report = collect(args.root.resolve(), args.legacy.resolve(), args.panel.resolve(), day, codes, now)
    print(publish(args.out.resolve(), report) / "daily_research.md")


if __name__ == "__main__":
    main()
