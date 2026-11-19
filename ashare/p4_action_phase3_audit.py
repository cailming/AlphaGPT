"""Recompute Phase 3 coverage and group outstanding corporate-action evidence needs."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile

import pandas as pd

from . import p4_action_pipeline as pipeline


SOURCE_NAMES = {"sina_dividend": "新浪分红", "cninfo_dividend": "巨潮分红",
                "sina_rights": "新浪配股", "baostock": "BaoStock"}
SOURCE_PROBLEMS = {
    "API_ERROR", "CONNECTION_ERROR", "TIMEOUT", "RATE_LIMIT", "SCHEMA_CHANGE", "ROW_LIMIT",
    "UPSTREAM_EMPTY_OR_SCHEMA", "UNSUPPORTED_EXCHANGE",
}
INTEGRITY_PROBLEMS = {"WRONG_STOCK_CODE", "DUPLICATE_EX_DATE_UNRESOLVED", "AKSHARE_DUPLICATE_UNRESOLVED"}
MISSING_FIELDS = {
    "dividStocksPs": "MISSING_BONUS_AMOUNT", "bonus_per_old_share": "MISSING_BONUS_AMOUNT",
    "dividReserveToStockPs": "MISSING_RESERVE_AMOUNT", "reserve_per_old_share": "MISSING_RESERVE_AMOUNT",
    "dividCashPsBeforeTax": "MISSING_CASH_AMOUNT", "cash_per_old_share": "MISSING_CASH_AMOUNT",
}


def read_csv(folder, name):
    return pd.read_csv(folder / (name + ".csv"), dtype="string", keep_default_na=False)


def integers(values):
    return pd.to_numeric(values, errors="raise").fillna(0).astype(int)


def count_rows(data, field):
    if field not in data:
        return 0
    return int(integers(data[field].replace("", pd.NA)).sum())


def audit_counts(folder, summary):
    requests = read_csv(folder, "request_status")
    coverage = read_csv(folder, "coverage_summary")
    health = read_csv(folder, "source_health")
    reconciliation = read_csv(folder, "cross_source_reconciliation")
    exceptions = read_csv(folder, "exceptions")
    key = ["stock_code", "endpoint", "year"]
    if requests.duplicated(key).any() or coverage.stock_code.duplicated().any():
        raise pipeline.EvidenceError("Duplicate request or stock key in Phase 3 report")
    totals = requests.state.value_counts()
    expected = {"effective_successes": "SUCCESS", "terminal_exceptions": "TERMINAL_EXCEPTION",
                "not_fetched_tasks": "NOT_FETCHED", "pending_retries": "RETRY_PENDING"}
    if (len(requests) != summary["task_count"] or len(coverage) != summary["stocks_in_scope"]
            or set(requests.stock_code) != set(coverage.stock_code)
            or health.endpoint.duplicated().any() or set(health.endpoint) != set(requests.endpoint)):
        raise pipeline.EvidenceError("CSV and JSON scope totals disagree")
    if any(int(totals.get(state, 0)) != summary[field] for field, state in expected.items()):
        raise pipeline.EvidenceError("Request CSV and summary disagree")
    if count_rows(requests, "new_attempts") != summary["new_attempts"]:
        raise pipeline.EvidenceError("Request CSV and new-attempt count disagree")
    if len(reconciliation) != summary["unique_candidate_events"] or reconciliation.duplicated(
            ["stock_code", "ex_date"]).any():
        raise pipeline.EvidenceError("Candidate counts or event keys disagree")
    outcomes = requests.assign(
        successful=requests.state.eq("SUCCESS"),
        terminal=requests.state.eq("TERMINAL_EXCEPTION"),
        pending=requests.state.eq("RETRY_PENDING"),
        unattempted=requests.state.eq("NOT_FETCHED"))
    by_stock = outcomes.groupby("stock_code")
    by_source = outcomes.groupby("endpoint")
    for field, row_flag in (("successful", "successful"), ("terminal_exceptions", "terminal"),
                            ("retry_pending", "pending"), ("not_fetched", "unattempted")):
        if not integers(coverage[field]).eq(coverage.stock_code.map(by_stock[row_flag].sum())).all():
            raise pipeline.EvidenceError(f"Stock-level CSV disagrees for {field}")
    for row in health.itertuples(index=False):
        group = by_source.get_group(row.endpoint)
        empty = group.successful & integers(group.response_rows).eq(0)
        if (int(row.planned) != len(group) or int(row.successful) != group.successful.sum()
                or int(row.unsuccessful) != (group.terminal | group.pending).sum()
                or int(row.not_fetched) != group.unattempted.sum()
                or int(row.successful_empty_responses) != empty.sum()):
            raise pipeline.EvidenceError(f"Source totals disagree for {row.endpoint}")
    priorities = {name: int(exceptions.priority.eq(name).sum()) for name in ("P0", "P1", "P2")}
    if priorities["P0"] != summary["p0_exceptions"] or sum(
            priorities[name] for name in ("P0", "P1")) != summary["high_priority_exceptions"]:
        raise pipeline.EvidenceError("Exception priority totals disagree")
    return requests, coverage, health, reconciliation, exceptions, priorities


def classify(category):
    if category in INTEGRITY_PROBLEMS or "DUPLICATE" in category:
        return "evidence_integrity", "Inspect the immutable ledger and raw rows for identity or duplicate keys", False
    if category in SOURCE_PROBLEMS:
        return "source_response", "Check endpoint health and authenticated response metadata", False
    if category in {"SUCCESS_EMPTY_RESPONSE", pipeline.free.CNINFO_EMPTY_UNVERIFIED}:
        return "source_response", "Seek corroborating dated events; historical coverage remains unknown", False
    if category == "UNKNOWN_EX_DATE" or "DATE" in category or "date" in category.lower():
        return "event_date", "Obtain the dated issuer implementation notice with record and ex dates", True
    if category.startswith("MISSING_") or "Missing entitlement" in category:
        return "event_terms", "Obtain issuer implementation notice with split bonus, capitalization and pre-tax cash", True
    if category == "RIGHTS_ISSUE":
        return "complex_action", "Obtain rights ratio, subscription price, dates and implementation notice", True
    if "ONLY_UNVERIFIED" in category or "NOT_COMPARABLE" in category:
        return "cross_source_gap", "Check same stock/date in other free evidence and issuer implementation notice", True
    return "event_terms", "Compare source records with the issuer implementation notice", True


def value_breakdown(exceptions, requests):
    details = []
    keys = ["stock_code", "date_or_year", "source", "category", "evidence", "priority",
            "unit", "count", "group", "required_evidence", "issuer_notice_required"]
    for row in exceptions.to_dict("records"):
        group, needed, issuer = classify(row["category"])
        details.append(row | {"unit": "exception_record", "count": 1, "group": group,
                              "required_evidence": needed, "issuer_notice_required": issuer})
        if row["category"] == "MISSING_ENTITLEMENT_FIELDS" or row["category"].startswith(
                "Missing entitlement amount"):
            for field, category in MISSING_FIELDS.items():
                if field in (row["category"] + " " + row["action"]):
                    details.append(row | {"category": category, "unit": "provider_event_field",
                                          "count": 1, "group": group, "required_evidence": needed,
                                          "issuer_notice_required": issuer})
    for row in requests.to_dict("records"):
        if row["state"] == "SUCCESS" and row["response_rows"] == "0":
            group, needed, issuer = classify("SUCCESS_EMPTY_RESPONSE")
            details.append({"stock_code": row["stock_code"], "date_or_year": row["year"],
                            "source": row["endpoint"], "category": "SUCCESS_EMPTY_RESPONSE",
                            "evidence": row["evidence"], "priority": "P2", "unit": "request",
                            "count": 1, "group": group, "required_evidence": needed,
                            "issuer_notice_required": issuer})
        for field in pipeline.DATE_DIAGNOSTICS:
            if str(row.get(field) or "") in {"", "0", "0.0"}:
                continue
            count = int(float(row[field]))
            details.append({"stock_code": row["stock_code"], "date_or_year": row["year"],
                            "source": row["endpoint"], "category": field.removesuffix("_rows").upper(),
                            "evidence": row["evidence"], "priority": "P1", "unit": "provider_row",
                            "count": count, "group": "event_date",
                            "required_evidence": "Check issuer implementation date and provider request year",
                            "issuer_notice_required": True})
    return pd.DataFrame(details, columns=keys)


def summarize_profiles(detail):
    index = ["group", "source", "category", "unit"]
    profile = detail.groupby(index, sort=True).agg(
        occurrences=("count", "sum"), affected_stocks=("stock_code", "nunique"),
        stocks=("stock_code", lambda codes: "|".join(sorted(set(codes)))),
        highest_priority=("priority", "min"), required_evidence=("required_evidence", "first"),
        issuer_notice_required=("issuer_notice_required", "any")).reset_index()
    profile["observed_distribution"] = profile.affected_stocks.map(
        lambda count: "REPEATED_ACROSS_STOCKS" if count > 1 else "OBSERVED_ON_ONE_STOCK")
    return profile.sort_values(["highest_priority", "occurrences", "source"], ascending=[True, False, True])


def count_attempts(output, baseline):
    frozen = SimpleNamespace(snapshot=pipeline.SNAPSHOT, metadata=pipeline.METADATA, ak_root=pipeline.AK_ROOT,
                             bao_root=pipeline.BAO_ROOT, output=output)
    evidence = pipeline.Evidence(frozen)
    if evidence.identity != baseline["identity"]:
        raise pipeline.EvidenceError("Phase 3 attempt baseline belongs to a different snapshot")
    start = pd.Timestamp(baseline["start_utc"]).timestamp()
    attempts = retries = local_only = 0
    for history in evidence.attempts.values():
        for index, attempt in enumerate(history):
            if attempt.number and attempt.finished >= start:
                attempts += 1
                retries += int(index > 0)
                local_only += int(attempt.category == "UNSUPPORTED_EXCHANGE")
    return {"attempts": attempts, "repeat_attempts": retries, "local_unsupported": local_only}


def report_text(summary, coverage, health, requests, reconciliation, priorities, baseline,
                source_folder, detail, phase_attempts):
    counts = {key: count_rows(requests, key) for key in pipeline.DATE_DIAGNOSTICS}
    old = requests.legacy_unclassified_date_error.str.lower().eq("true").sum()
    baseline_success = baseline.get("reused_success_tasks", "未知")
    earlier_new = baseline.get("preexisting_new_attempts")
    new_attempts = summary["new_attempts"] - earlier_new if earlier_new is not None else "未知"
    unsuccessful = integers(coverage.terminal_exceptions).gt(0).sum()
    market_like = summary["stocks_in_scope"] == summary["snapshot_stocks"]
    missing_amount_events = int(detail.loc[detail.unit.eq("provider_event_field")].drop_duplicates(
        ["stock_code", "date_or_year", "source"]).shape[0])
    same = reconciliation.result.value_counts()
    start_utc = baseline.get("start_utc", "未知")
    title = "P4.3-E2 全量审计报告" if summary["scope_ready_for_review"] and market_like else "P4.3-E2 采集与治理进度"
    lines = [
        f"# {title}", "",
        f"- 采集起点 UTC {start_utc}，审计时间 UTC {summary['as_of_utc']}。",
        f"- 冻结快照 SHA256 `{summary['snapshot_sha256']}`。历史区间自 {summary['start']} 起，排除 {summary['test_start_EXCLUDED']} 及之后的旧测试期。",
        f"- 冻结范围 {summary['snapshot_stocks']:,} 股，当前报告范围 {summary['stocks_in_scope']:,} 股。",
        f"- 已扫描 {summary['stocks_scanned']:,} 股，所有请求成功 {summary['stocks_all_requests_successful']:,} 股，至少一个终态异常 {unsuccessful:,} 股。",
        f"- {summary['task_count']:,} 个任务中成功 {summary['effective_successes']:,}，终态异常 {summary['terminal_exceptions']:,}，未抓取 {summary['not_fetched_tasks']:,}，待重试 {summary['pending_retries']:,}。",
        f"- 进入全量时可复用有效成功任务 {baseline_success}。本轮新增尝试记录 {new_attempts}，含本地判定不支持及实际重试。",
        f"- 当前熔断 `{summary['circuit']}`，已隔离来源 `{summary.get('paused_sources', {})}`。旧测试隔离 `{summary['old_test_isolation_passed']}`。P0/P1/P2 分别为 {priorities['P0']}/{priorities['P1']}/{priorities['P2']}。",
        "", "## 来源", "",
        "| 接口 | 应查 | 成功 | 未成功 | 未抓取 | 成功空表 | 有效成功率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in health.itertuples(index=False):
        rate = f"{float(row.success_rate_observed):.2%}" if row.success_rate_observed else "未知"
        lines.append(f"| {SOURCE_NAMES.get(row.endpoint, row.endpoint)} | {row.planned} | {row.successful} | {row.unsuccessful} | {row.not_fetched} | {row.successful_empty_responses} | {rate} |")
    lines.extend([
        "", "成功率按有结论的请求任务计算，分母包括无法确认日期的响应及本地判定不支持。空表保留为覆盖未知。",
        "", "## 事件对账与异常", "",
        f"- 去重候选 {summary['unique_candidate_events']:,} 条，单源候选 {summary['single_source_events']:,} 条。",
        f"- 新浪与巨潮完整匹配 {summary['sina_cninfo_matches']:,} 条、部分字段未决 {summary['sina_cninfo_partial_field_matches_unresolved']:,} 条。BaoStock 与 AkShare 完整匹配 {summary['baostock_full_field_matches']:,} 条、部分字段未决 {summary['baostock_partial_field_matches_unresolved']:,} 条。",
        f"- 日期或金额冲突类异常 {summary['date_or_amount_conflicts']:,} 条。单独的 BaoStock 条款冲突 {same.get('BAOSTOCK_AKSHARE_TERMS_CONFLICT', 0)} 条，登记日期冲突 {same.get('BAOSTOCK_AKSHARE_DATE_CONFLICT', 0)} 条。",
        f"- 可拆分诊断记录中缺除权日 {counts['missing_ex_date_rows']:,} 行、无效除权日 {counts['invalid_ex_date_rows']:,} 行、缺公告日 {counts['missing_announcement_date_rows']:,} 行、无效公告日 {counts['invalid_announcement_date_rows']:,} 行。另有 {old:,} 个历史任务只有合并日期异常状态，字段级数量未知。",
        f"- {missing_amount_events:,} 个来源事件记录存在未知权益字段，包括 BaoStock 的 {summary['baostock_candidates_with_unknown_fields']:,} 条候选。按接口和字段分项见异常画像。",
        f"- 期内配股风险 {summary['rights_records_in_scope']:,} 条。当前数据源对其他复杂权益类型的总覆盖仍未知。",
        "", "## 证据与训练闸门", "",
        "- 异常画像按来源、类型和计量单位统计，并列出涉及股票。跨股票重复表示一种分布，需要结合源端返回与发行人证据确认成因。",
        "- 优先处理证据一致性、来源接口问题和可能影响经济收益的权益差异。发行人实施公告需求见报告目录中的人工核验表。",
        "- `complete_verified_assertions_created=0`、`can_emit_training_labels=false`、`full_historical_action_coverage_certified=false`、`training_gate=CLOSED`。",
        f"- 本次原始审计 CSV/JSON 目录为 `{source_folder}`。",
    ])
    if phase_attempts is not None:
        lines.insert(8, f"- 已逐条复算新增尝试，其中已有证据的同键补采或重试 "
                     f"{phase_attempts['repeat_attempts']:,} 次，本地判定不支持 "
                     f"{phase_attempts['local_unsupported']:,} 次。")
    return "\n".join(lines) + "\n"


def generate(output: Path, *, final: bool):
    checkpoint = pipeline.read_json(output / "checkpoint.json")
    latest = pipeline.read_json(output / "latest_report.json")
    summary = latest["summary"]
    folder = Path(latest["report_dir"])
    if (checkpoint["identity"]["snapshot_sha256"] != summary["snapshot_sha256"]
            or pipeline.read_json(folder / "pipeline_summary.json") != summary):
        raise pipeline.EvidenceError("Report pointer does not match the frozen audit")
    if final and (checkpoint["state"] != "DONE" or checkpoint["offset"] != 0
                  or checkpoint["limit"] != 5449 or summary["stocks_in_scope"] != 5449
                  or summary["snapshot_stocks"] != 5449 or not summary["scope_ready_for_review"]
                  or summary["pending_retries"] or summary["not_fetched_tasks"] or summary.get("paused_sources")
                  or not summary["old_test_isolation_passed"] or (output / "p0_blocker.json").exists()):
        raise pipeline.EvidenceError("Final report requires a completed, integrity-checked 5,449-stock scope")
    if (summary["training_gate"] != "CLOSED" or summary["can_emit_training_labels"]
            or summary["full_historical_action_coverage_certified"]
            or summary["complete_verified_assertions_created"]):
        raise pipeline.EvidenceError("Training and certification gates must be closed")
    requests, coverage, health, reconciliation, exceptions, priorities = audit_counts(folder, summary)
    detail = value_breakdown(exceptions, requests)
    profile = summarize_profiles(detail)
    governance = output / "governance"
    governance.mkdir(exist_ok=True)
    dest = Path(tempfile.mkdtemp(prefix="phase3-", dir=governance))
    profile.to_csv(dest / "anomaly_profile.csv", index=False)
    detail.loc[detail.issuer_notice_required.eq(True)].sort_values(
        ["priority", "stock_code", "source", "date_or_year"]).to_csv(dest / "issuer_review_queue.csv", index=False)
    detail.loc[detail.issuer_notice_required.eq(False)].sort_values(
        ["priority", "source", "category", "stock_code"]).to_csv(dest / "source_investigation_queue.csv", index=False)
    baseline_file = output / "phase3_baseline.json"
    baseline = pipeline.read_json(baseline_file) if baseline_file.exists() else {}
    phase_attempts = count_attempts(output, baseline) if final and baseline else None
    if phase_attempts is not None and phase_attempts["attempts"] != (
            summary["new_attempts"] - baseline["preexisting_new_attempts"]):
        raise pipeline.EvidenceError("Phase 3 ledger does not match reported new attempts")
    text = report_text(summary, coverage, health, requests, reconciliation, priorities,
                       baseline, folder, detail, phase_attempts)
    path = output / ("FINAL_FULL_AUDIT_REPORT.md" if final else "PHASE3_PROGRESS.md")
    with tempfile.NamedTemporaryFile(dir=output, encoding="utf-8", mode="w", delete=False) as handle:
        handle.write(text)
        handle.write(f"- 本次异常画像与分流队列目录为 `{dest}`。\n")
        temp = Path(handle.name)
    temp.replace(path)
    pipeline.atomic_json(output / "latest_governance.json", {
        "audit_dir": str(folder), "governance_dir": str(dest), "report": str(path),
        "priorities": priorities, "phase_attempts": phase_attempts, "final": final})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=pipeline.OUTPUT)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    with pipeline.exclusive(args.output) if args.final else nullcontext():
        print(generate(args.output, final=args.final))


if __name__ == "__main__":
    main()
