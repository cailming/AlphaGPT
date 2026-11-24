"""Verify archived BSE dividend notices without changing the evidence ledger or training gate."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date, datetime
from decimal import Decimal
import json
from pathlib import Path
import re
import subprocess
import unicodedata
from zoneinfo import ZoneInfo

from ashare import p4_action_pipeline as pipe


NUMBER = r"(\d+(?:\.\d+)?)"
DATES = r"(\d{4})年(\d{1,2})月(\d{1,2})日"
REG = re.compile(r"(?:权益登记日|股权登记日)(?:确定)?(?:为)?[:：]?" + DATES)
EX = re.compile(r"(?:除权除息|除权\(息\)|除权及除息|除息|除权)日(?:确定)?(?:为)?[:：]?" + DATES)
TERMS = {
    "cash_pre_tax": re.compile(r"每10股(?:派发|派送|分派|派)(?:现金红利|现金股利|现金|人民币)*" + NUMBER + r"元"),
    "capitalization": re.compile(r"每10股(?:转增(?:股本)?|以资本公积(?:金)?转增)" + NUMBER + r"股"),
    "bonus": re.compile(r"每10股(?:送红股|送股|送)" + NUMBER + r"股"),
}


def compact(text):
    return re.sub(r"[\s\u200b\ufeff]+", "", unicodedata.normalize("NFKC", text))


def dates_for(pattern, source):
    return sorted({date(int(year), int(month), int(day)).isoformat()
                   for year, month, day in pattern.findall(source)})


def sections(text):
    normalized = compact(text)
    header = normalized[:260]
    extract = re.search(r"(?:^|[。:：])一[、.]\s*(?:本次)?(?:权益分派|利润分配|分红派息)方案", normalized[:3500])
    if not extract:
        raise ValueError("Named distribution section not found")
    tail = normalized[extract.end():]
    end = re.search(r"(?:二[、.](?:权益|股权)登记日|二[、.]相关日期)", tail[:4200])
    if not end:
        raise ValueError("Named dates section not found after distribution terms")
    terms_text = re.split(r"(?:\d[、.]扣税(?:说明|情况)|扣税后|按\d+%的税率|税后实际)", tail[:end.start()], maxsplit=1)[0]
    terms_text = re.sub(r"\([^()]*\)", "", terms_text)
    dates_text = tail[end.start():end.start()+370]
    return header, terms_text, dates_text


def extract(text, notice, target):
    header, terms_text, dates_text = sections(text)
    printed_codes = re.findall(r"(?:证券|股票)代码[:：]?(\d{6})(?!\d)", header)
    printed_name = compact(notice["notice_short_name"])
    if printed_codes != [notice["notice_stock_code"]] or not re.search(
            r"(?:证券|股票)简称[:：]?" + re.escape(printed_name), header):
        raise ValueError("PDF issuer code or short name does not match official index")
    if not re.search(r"(?:权益分派|利润分配|分红派息).*实施公告", header):
        raise ValueError("PDF heading is not an implementation notice")
    record = dates_for(REG, dates_text)
    ex = dates_for(EX, dates_text)
    if record != [target["record_date"]] or ex != [target["ex_date"]]:
        raise ValueError(f"Implementation dates mismatch or ambiguous: record={record}, ex={ex}")
    values = {field: {Decimal(value) for value in rule.findall(terms_text)} for field, rule in TERMS.items()}
    selected = {}
    for field, reported in values.items():
        expected = target.get(field)
        if expected is None:
            if reported:
                raise ValueError(f"Official document adds a term absent from provider evidence: {field}")
            continue
        if reported != {Decimal(expected).scaleb(1)}:
            raise ValueError(f"Official pre-tax amount/ratio missing or mismatched for {field}: {sorted(reported)}")
        selected[field] = format(next(iter(reported)), "f")
    if "cash_pre_tax" not in selected or Decimal(selected["cash_pre_tax"]) <= 0:
        raise ValueError("No positive gross cash term in issuer evidence")
    return selected, terms_text[:410], dates_text[:170]


def read_notice(folder, notice, target, output, code, ex):
    try:
        parsed = extract((folder / notice["text"]).read_text(encoding="utf-8"), notice, target)
        return parsed, {"mode": "layout"}
    except ValueError as layout_error:
        raw = subprocess.run(["pdftotext", "-raw", "-enc", "UTF-8", str(folder / notice["pdf"]), "-"],
                             check=True, capture_output=True, timeout=15).stdout
        try:
            parsed = extract(raw.decode("utf-8"), notice, target)
        except ValueError as raw_error:
            raise ValueError(f"Layout: {layout_error}; PDF text order: {raw_error}") from raw_error
        filename = f"pdf-text-order-{code}-{ex}-{notice['id']}.txt"
        (output / filename).write_bytes(raw)
        return parsed, {"mode": "pdf_text_order", "text_file": filename,
                        "text_sha256": pipe.free.digest(output / filename), "layout_failure": str(layout_error)}


def verify(archive, output, allow_partial):
    source = pipe.read_json(archive / "acquisition.json")
    jobs = source["jobs"]
    if not allow_partial and len(jobs) != source["expected"]:
        raise ValueError("Acquisition still running or stopped before the final case")
    map_path = archive / "issuer_identity_source.json"
    if pipe.free.digest(map_path) != source["issuer_identity_sha256"]:
        raise ValueError("Official issuer map changed")
    by_code = {row["code"]: row for row in pipe.read_json(map_path)["stockList"] if row.get("category") == "A股"}
    output.mkdir(parents=True)
    results, observations = [], []
    for job in jobs:
        code, ex = job["stock_code"], job["ex_date"]
        target = job["target"] | {"ex_date": ex}
        folder = archive / f"{code}-{ex}"
        failures, matching = [], []
        for notice in job["notices"]:
            try:
                if (len({notice["issuer_org_id"], job["issuer_org_id"], by_code[code]["orgId"]}) != 1
                        or not (notice["published_on"] < ex < pipe.CUTOFF)):
                    raise ValueError("Wrong official issuer identity or publication date")
                for part in ("metadata", "pdf", "text"):
                    filename = notice["metadata_file" if part == "metadata" else part]
                    if Path(filename).name != filename or pipe.free.digest(folder / filename) != notice[part + "_sha256"]:
                        raise ValueError(f"Invalid {part} artifact digest or path")
                catalog = pipe.read_json(folder / notice["metadata_file"])["announcements"]
                matches = [record for record in catalog if str(record["announcementId"]) == notice["id"]]
                if (len(matches) != 1 or matches[0]["orgId"] != notice["issuer_org_id"]
                        or matches[0]["secCode"] != notice["notice_stock_code"]
                        or matches[0]["secName"] != notice["notice_short_name"]
                        or "https://static.cninfo.com.cn/" + matches[0]["adjunctUrl"] != notice["url"]
                        or datetime.fromtimestamp(matches[0]["announcementTime"] / 1000, ZoneInfo("Asia/Shanghai"))
                        .date().isoformat() != notice["published_on"]):
                    raise ValueError("Official index fields do not match PDF provenance")
                (parsed, terms, dates), extraction = read_notice(folder, notice, target, output, code, ex)
                matching.append((notice, parsed, terms, dates, extraction))
            except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
                failures.append({"id": notice["id"], "detail": str(error)})
        state = ("ISSUER_NOTICE_VERIFIED" if job["status"] == "ACQUIRED"
                 and len(matching) == len(job["notices"]) == 1 else "UNRESOLVED")
        result = {"stock_code": code, "ex_date": ex, "status": state, "acquisition": job["status"],
                  "issues": failures, "notices": [{"id": item["id"], "url": item["url"]} for item in job["notices"]]}
        if matching:
            notice, parsed, terms, relevant_dates, extraction = matching[0]
            result.update(document=notice, terms_per_10=parsed, terms_excerpt=terms, dates_excerpt=relevant_dates,
                          extraction=extraction)
        if state == "ISSUER_NOTICE_VERIFIED":
            descriptions = {"cash_pre_tax": "税前现金（元）", "capitalization": "转增（股）", "bonus": "送股（股）"}
            amounts = "、".join(f"{descriptions[field]}{value}" for field, value in parsed.items())
            observations.append({"stock_code": code, "ex_date": ex, "record_date": target["record_date"],
                                 "published_on": notice["published_on"], "source_url": notice["url"],
                                 "location": "权益分派方案及登记、除权除息日小节",
                                 "observation": "实施公告每10股" + amounts + "；本项新浪现金字段为零，与公告不符。",
                                 "official_notice_id": notice["id"], "file_sha256": notice["pdf_sha256"],
                                 "archival_stock_code": notice["notice_stock_code"]})
        results.append(result)
    pipe.atomic_json(output / "cases.json", results)
    pipe.atomic_json(output / "reviewed_notices.json", observations)
    with (output / "unresolved.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stock_code", "ex_date", "acquisition", "issues"))
        writer.writeheader()
        writer.writerows({k: case[k] for k in writer.fieldnames} for case in results if case["status"] != "ISSUER_NOTICE_VERIFIED")
    summary = {"expected": source["expected"], "verified": len(observations), "cases_seen": len(results),
               "acquisition": dict(Counter(row["acquisition"] for row in results)),
               "extraction_modes": dict(Counter(row["extraction"]["mode"] for row in results
                                                if row["status"] == "ISSUER_NOTICE_VERIFIED")),
               "historical_security_codes_validated_by_issuer": sum(row["stock_code"] != row["archival_stock_code"] for row in observations),
               "training_gate": "CLOSED", "can_emit_training_labels": False,
               "full_historical_action_coverage_certified": False, "complete_verified_assertions_created": 0,
               "artifact_sha256": {path.name: pipe.free.digest(path) for path in output.iterdir() if path.is_file()}}
    pipe.atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    verify(args.archive, args.out, args.allow_partial)


if __name__ == "__main__":
    main()
