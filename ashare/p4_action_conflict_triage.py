"""Triage finalized P4.3-E2 conflicts using the existing immutable evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
from decimal import Decimal
from itertools import combinations
import json
from pathlib import Path
import re
import tempfile
from unittest.mock import patch

from ashare import p4_action_pipeline as pipeline


FIELDS = {
    "sina_dividend": ("公告日期", "股权登记日", "除权除息日", "送股", "转增", "派息"),
    "cninfo_dividend": ("实施方案公告日期", "股权登记日", "除权日", "送股比例", "转增比例", "派息比例"),
    pipeline.BAO: ("dividPlanDate", "dividRegistDate", "dividOperateDate", *pipeline.bao.AMOUNT_COLUMNS),
}
TERMS = ("bonus", "capitalization", "cash_pre_tax")
CONFLICT = re.compile(r"CONFLICT|MISMATCH|date order|DUPLICATE_EX_DATE", re.IGNORECASE)
PRECISION_CAP = Decimal("0.00001")


def amount(value, endpoint):
    literal = str(value).strip()
    if literal.lower() in {"", "none", "na", "nan", "nat", "--", "-"}:
        return None
    number = Decimal(literal.replace(",", ""))
    if not number.is_finite() or number < 0:
        raise ValueError("Invalid nonnegative entitlement amount")
    return number if endpoint == pipeline.BAO else number.scaleb(-1)


def precision_compatible(left, right):
    if not left or not right:
        return False
    delta = abs(left - right)
    joint_radius = sum(Decimal("0.5").scaleb(value.as_tuple().exponent) for value in (left, right))
    return delta <= min(PRECISION_CAP, joint_radius)


def classify(sources):
    if any(len(rows) > 1 for rows in sources.values()):
        return "SAME_DAY_MULTIPLE_ROWS"
    pairs = [(endpoint, rows[0]) for endpoint, rows in sorted(sources.items()) if rows]
    if len(pairs) < 2:
        return "INSUFFICIENT_COMPARABLE_ROWS"
    if len({row["record_date"] for _, row in pairs}) != 1:
        return "RECORD_DATE_DIFFERENCE"
    differences = [(left_name, right_name, term, left[term], right[term])
                   for (left_name, left), (right_name, right) in combinations(pairs, 2)
                   for term in TERMS if left[term] is not None and right[term] is not None
                   and left[term] != right[term]]
    if differences and all(precision_compatible(left, right) for _, _, _, left, right in differences):
        return "SMALL_DIFFERENCE_COMPATIBLE_WITH_PRECISION"
    if len(differences) == 1:
        first, second, term, left, right = differences[0]
        rows = dict(pairs)
        if ((first, second, term) == ("cninfo_dividend", "sina_dividend", "cash_pre_tax")
                and left > 0 and right == 0 and rows[first]["published_on"] == rows[second]["published_on"]
                and any(rows[first][name] is not None and rows[first][name] == rows[second][name]
                        and rows[first][name] > 0 for name in TERMS[:2])):
            return "SINA_CASH_ZERO_WITH_MATCHING_SHARE_TERMS"
    return "UNRESOLVED_TERMS_OR_EVIDENCE"


def event_rows(evidence, stock, ex_date):
    sources = defaultdict(list)
    for task in evidence.tasks(stock):
        if task.endpoint not in FIELDS:
            continue
        attempt = evidence.effective(task)
        if attempt is None:
            continue
        published, record, ex, *amount_fields = FIELDS[task.endpoint]
        for row in attempt.rows:
            if row[ex] != ex_date or (task.endpoint == "sina_dividend" and row["进度"].strip() != "实施"):
                continue
            if not row[published] <= row[record] < row[ex]:
                raise pipeline.EvidenceError(f"Unexpected date order for conflict {stock}/{ex_date}")
            sources[task.endpoint].append({
                "published_on": row[published], "record_date": row[record],
                **{field: amount(row[raw], task.endpoint) for field, raw in zip(TERMS, amount_fields, strict=True)},
                "raw_terms": {field: row[field] for field in amount_fields}, "evidence": str(attempt.evidence),
            })
    return dict(sources)


def generate(notices_path):
    with patch("sys.argv", ["p4_action_pipeline", "status"]):
        args = pipeline.parse_args()
    root = args.output
    evidence = pipeline.Evidence(args)
    codes, checkpoint = pipeline.configure(evidence, args)
    summary_pointer = pipeline.read_json(root / "latest_report.json")
    summary = summary_pointer["summary"]
    audit = pipeline.read_json(root / "latest_governance.json")
    report_dir = Path(summary_pointer["report_dir"])
    if (checkpoint["state"] != "DONE" or checkpoint.get("circuit") or checkpoint.get("paused_sources")
            or not audit["final"] or Path(audit["audit_dir"]) != report_dir
            or summary != pipeline.read_json(report_dir / "pipeline_summary.json")
            or not summary["scope_ready_for_review"] or summary["training_gate"] != "CLOSED"
            or any(summary[key] for key in ("not_fetched_tasks", "pending_retries", "p0_exceptions",
                    "complete_verified_assertions_created", "can_emit_training_labels",
                    "full_historical_action_coverage_certified")) or (root / "p0_blocker.json").exists()
            or len(codes) != summary["stocks_in_scope"]
            or any(summary[key] != value for key, value in evidence.identity.items())):
        raise pipeline.EvidenceError("Conflict triage requires the finalized frozen audit with the training gate closed")

    exceptions = report_dir / "exceptions.csv"
    findings = defaultdict(Counter)
    with exceptions.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["priority"] == "P1" and CONFLICT.search(row["category"]):
                key = row["stock_code"], date.fromisoformat(row["date_or_year"]).isoformat()
                findings[key][row["category"]] += 1
    rows = sum(sum(reasons.values()) for reasons in findings.values())
    if rows != summary["date_or_amount_conflicts"]:
        raise pipeline.EvidenceError("Conflict rows do not reconcile to the frozen report")

    citations = pipeline.read_json(notices_path) if notices_path else []
    reviewed = {}
    for notice in citations:
        key = notice["stock_code"], notice["ex_date"]
        if (key not in findings or key in reviewed or not notice["source_url"].startswith("https://")
                or not notice["observation"] or not notice["location"]
                or not evidence.identity["start"] <= notice["published_on"] < key[1]
                or not evidence.identity["start"] <= notice["record_date"] < key[1]):
            raise pipeline.EvidenceError("Issuer sample does not identify a unique pre-test conflict and implementation notice")
        reviewed[key] = notice

    cases = []
    for (stock, ex_date), reasons in sorted(findings.items()):
        sources = event_rows(evidence, stock, ex_date)
        route = classify(sources)
        notice = reviewed.get((stock, ex_date))
        cases.append({"stock_code": stock, "ex_date": ex_date, "original_findings": dict(reasons),
                      "route": route, "sources": sources, "issuer_notice_observation": notice,
                      "follow_up": ("ISSUER_SAMPLE_REVIEWED" if notice else
                                    "PRECISION_TRIAGE_ONLY" if route == "SMALL_DIFFERENCE_COMPATIBLE_WITH_PRECISION"
                                    else "ISSUER_NOTICE_REQUIRED")})
    by_route = Counter(case["route"] for case in cases)
    by_follow_up = Counter(case["follow_up"] for case in cases)
    dest = root / "followup"
    dest.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="p1-conflicts-", dir=dest))
    with (output / "conflict_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, default=str, sort_keys=True) + "\n")
    ranks = {"SAME_DAY_MULTIPLE_ROWS": 0, "RECORD_DATE_DIFFERENCE": 0,
             "UNRESOLVED_TERMS_OR_EVIDENCE": 1, "INSUFFICIENT_COMPARABLE_ROWS": 1,
             "SINA_CASH_ZERO_WITH_MATCHING_SHARE_TERMS": 2}
    fields = ("stock_code", "ex_date", "route", "original_categories", "request")
    with (output / "issuer_review_remaining.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in sorted(cases, key=lambda item: (ranks.get(item["route"], 9), item["stock_code"], item["ex_date"])):
            if case["follow_up"] == "ISSUER_NOTICE_REQUIRED":
                writer.writerow({**{key: case[key] for key in fields[:3]},
                                 "original_categories": "|".join(sorted(case["original_findings"])),
                                 "request": "实施公告：核对普通A股权益登记日、除权日、税前现金、送股和转增；多笔同日须核对组成"})
    synopsis = {"source_audit": str(report_dir), "exceptions_sha256": pipeline.free.digest(exceptions),
                "triage_script_sha256": pipeline.free.digest(Path(__file__)),
                "reviewed_notices_sha256": pipeline.free.digest(notices_path) if notices_path else None,
                "artifact_sha256": {filename: pipeline.free.digest(output / filename)
                                    for filename in ("conflict_cases.jsonl", "issuer_review_remaining.csv")},
                "original_conflict_rows": rows, "stock_date_cases": len(cases),
                "stocks": len({case["stock_code"] for case in cases}),
                "by_route": dict(by_route), "by_follow_up": dict(by_follow_up),
                "training_gate": "CLOSED", "can_emit_training_labels": False,
                "full_historical_action_coverage_certified": False, "complete_verified_assertions_created": 0}
    pipeline.atomic_json(output / "manifest.json", synopsis)
    return output, synopsis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-notices", type=Path, help="Optional traced issuer notice observations (JSON array)")
    args = parser.parse_args()
    path, summary = generate(args.reviewed_notices)
    print(str(path))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
