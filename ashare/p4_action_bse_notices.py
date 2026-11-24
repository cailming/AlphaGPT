"""Acquire exact official implementation notices for the pending Phase 3 BSE conflicts."""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import requests

from ashare import p4_action_pipeline as pipe


CNINFO = "https://www.cninfo.com.cn"
ATTACHMENTS = "https://static.cninfo.com.cn/"
BEIJING = ZoneInfo("Asia/Shanghai")
TITLE = re.compile(r"(?:权益分派|利润分配|分红派息).*实施公告")
INTERVAL = 1.0


def digest_bytes(content):
    return hashlib.sha256(content).hexdigest()


def clean_title(value):
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def attachment_url(value):
    if not re.fullmatch(r"finalpage/\d{4}-\d{2}-\d{2}/[0-9]+\.[Pp][Dd][Ff]", value):
        raise ValueError("Unexpected attachment path")
    return ATTACHMENTS + value


class SourcePause(Exception):
    pass


class Source:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AlphaGPT historical issuer notice audit/1.0",
                                     "Referer": CNINFO + "/new/disclosure", "Accept": "application/json,application/pdf"})
        self.last_started = 0

    def fetch(self, method, url, limit, data=None):
        for attempt in range(2):
            time.sleep(max(0, self.last_started + INTERVAL - time.monotonic()))
            self.last_started = time.monotonic()
            try:
                with self.session.request(method, url, data=data, timeout=(6, 12),
                                          allow_redirects=False, stream=True) as response:
                    if response.status_code in {403, 429}:
                        raise SourcePause(f"HTTP {response.status_code} at {urlsplit(url).hostname}")
                    if 500 <= response.status_code < 600 and not attempt:
                        time.sleep(2)
                        continue
                    if response.status_code != 200:
                        raise ValueError(f"HTTP {response.status_code}")
                    body = bytearray()
                    for chunk in response.iter_content(65536):
                        body.extend(chunk)
                        if len(body) > limit:
                            raise ValueError("Response exceeds audit size limit")
                    return bytes(body)
            except (requests.Timeout, requests.ConnectionError):
                if attempt:
                    raise
                time.sleep(2)
        raise RuntimeError("Source request exhausted")


def load_cases(folder):
    manifest = pipe.read_json(folder / "manifest.json")
    for filename, expected in manifest["artifact_sha256"].items():
        if pipe.free.digest(folder / filename) != expected:
            raise ValueError(f"Changed triage artifact: {filename}")
    with (folder / "issuer_review_remaining.csv").open(encoding="utf-8", newline="") as handle:
        keys = {(row["stock_code"], row["ex_date"]) for row in csv.DictReader(handle)}
    with (folder / "conflict_cases.jsonl").open(encoding="utf-8") as handle:
        cases = [case for case in map(json.loads, handle) if (case["stock_code"], case["ex_date"]) in keys]
    if (len(keys) != len(cases) or len(cases) != manifest["by_follow_up"]["ISSUER_NOTICE_REQUIRED"]
            or manifest["training_gate"] != "CLOSED" or not all(
                case["route"] == "SINA_CASH_ZERO_WITH_MATCHING_SHARE_TERMS"
                and re.fullmatch(r"920\d{3}", case["stock_code"]) and case["ex_date"] < pipe.CUTOFF for case in cases)):
        raise ValueError("Unexpected pending BSE conflict scope")
    return cases


def search(source, issuer, day_range, case_folder, label):
    begin, end = day_range
    metadata = {"stock": issuer["code"] + "," + issuer["orgId"], "tabName": "fulltext", "pageNum": 1,
                "pageSize": 30, "column": "szse", "plate": "", "category": "", "searchkey": "",
                "secid": "", "trade": "", "seDate": f"{begin}~{end}", "isHLtitle": "false"}
    result = []
    for page in range(1, 4):
        parameters = metadata | {"pageNum": page}
        raw = source.fetch("POST", CNINFO + "/new/hisAnnouncement/query", 2_000_000, parameters)
        name = f"metadata-{label}-{page}.json"
        (case_folder / name).write_bytes(raw)
        payload = json.loads(raw)
        total = payload.get("totalAnnouncement")
        if type(total) is not int or not 0 <= total <= 90:
            raise ValueError("Unsupported announcement listing size")
        for row in payload.get("announcements") or []:
            title = clean_title(row["announcementTitle"])
            if row.get("orgId") != issuer["orgId"] or not TITLE.search(title):
                continue
            published = datetime.fromtimestamp(row["announcementTime"] / 1000, BEIJING).date()
            identifier = str(row["announcementId"])
            if not re.fullmatch(r"[0-9]+", identifier) or not begin <= published <= end:
                raise ValueError("Announcement identifier or publication date is inconsistent")
            result.append({"id": identifier, "issuer_org_id": row["orgId"], "notice_stock_code": row["secCode"],
                           "notice_short_name": row["secName"], "title": title, "published_on": published.isoformat(),
                           "url": attachment_url(row["adjunctUrl"]), "metadata_file": name,
                           "metadata_sha256": digest_bytes(raw)})
        if total <= page * 30:
            break
    return list({row["id"]: row for row in result}.values())


def download(source, notice, folder):
    prefix = "notice-" + notice["id"]
    content = source.fetch("GET", notice["url"], 3_000_000)
    if not content.startswith(b"%PDF-"):
        raise ValueError("Attachment is not a PDF")
    pdf_path = folder / (prefix + ".pdf")
    text_path = folder / (prefix + ".txt")
    pdf_path.write_bytes(content)
    subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), str(text_path)],
                   check=True, capture_output=True, timeout=15)
    return notice | {"pdf": pdf_path.name, "pdf_sha256": digest_bytes(content),
                     "text": text_path.name, "text_sha256": pipe.free.digest(text_path)}


def acquire(triage_folder, output):
    cases = load_cases(triage_folder)
    output.mkdir(parents=True)
    source = Source()
    raw_map = source.fetch("GET", CNINFO + "/new/data/szse_stock.json", 1_200_000)
    (output / "issuer_identity_source.json").write_bytes(raw_map)
    by_code = {item["code"]: item for item in json.loads(raw_map)["stockList"] if item.get("category") == "A股"}
    jobs = []
    status = {"source_queue": str(triage_folder), "expected": len(cases), "jobs": jobs,
              "issuer_identity_sha256": digest_bytes(raw_map), "training_gate": "CLOSED"}
    for index, case in enumerate(cases, 1):
        code, ex = case["stock_code"], case["ex_date"]
        target = case["sources"]["cninfo_dividend"][0]
        expected = date.fromisoformat(target["published_on"])
        issuer = by_code.get(code) or {}
        job = {"stock_code": code, "ex_date": ex, "target": target,
               "issuer_org_id": issuer.get("orgId"), "current_short_name": issuer.get("zwjc"), "notices": []}
        folder = output / f"{code}-{ex}"
        folder.mkdir()
        paused = False
        try:
            if not issuer.get("orgId") or not issuer.get("zwjc"):
                raise ValueError("Issuer not in official current issuer map")
            candidates = search(source, issuer, (expected, expected), folder, "exact")
            if not candidates:
                adjacent = timedelta(days=2)
                end = min(date.fromisoformat(ex) - timedelta(days=1), expected + adjacent)
                candidates = search(source, issuer, (expected - adjacent, end), folder, "nearby")
            if len(candidates) > 3:
                raise ValueError("Too many implementation notice candidates")
            for notice in candidates:
                if notice["published_on"] < ex and notice["published_on"] < pipe.CUTOFF:
                    job["notices"].append(download(source, notice, folder))
            job["status"] = "ACQUIRED" if job["notices"] else "NO_CANDIDATE"
        except SourcePause as error:
            job.update(status="SOURCE_PAUSED", detail=str(error))
            paused = True
        except (ValueError, KeyError, OSError, requests.RequestException, subprocess.SubprocessError) as error:
            job.update(status="ACQUISITION_UNRESOLVED", detail=f"{type(error).__name__}: {str(error)[:150]}")
        jobs.append(job)
        pipe.atomic_json(output / "acquisition.json", status)
        print(index, code, ex, job["status"], len(job["notices"]), flush=True)
        if paused:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    acquire(args.triage.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
