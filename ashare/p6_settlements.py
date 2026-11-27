"""Archive pre-reviewed ordinary A-share settlement terms and the exact official PDF.

A custodian can import a JSON review and its local PDF using `capture --review
... --pdf ... --source-index ... --root ...`. The real local receipt clock is assigned after copying
the bytes. Historical ex-dates cannot regain timely eligibility through a later
import. A provider observation or an ex-date plus gross yield is insufficient.
The cash field denotes the reviewed per-old-share amount credited under the
stated research account tax basis. The reader does not contact the source site.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, time
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from .p4_action_pipeline import atomic_json, exclusive, read_json
from .p4_subset_inputs import hash_file
from .p6_portfolio import iso, trading_unit

CHINA = ZoneInfo("Asia/Shanghai")
VERSION = "P6_VERIFIED_SETTLEMENT_RECEIPT_V1"
HOSTS = ("sse.com.cn", "szse.cn", "bse.cn", "cninfo.com.cn")
FIELDS = ("stock_code", "source_id", "source_url", "source_sha256", "published_at",
          "reviewed_at", "reviewer", "pdf_page", "evidence_excerpt", "record_date", "ex_date",
          "cash_per_share", "share_multiplier", "cash_pay_date", "share_list_date",
          "cash_basis", "scope", "review_status")
def timestamp(value):
    t = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(t, datetime) or t.tzinfo is None:
        raise ValueError("Settlement timestamps must include an explicit timezone")
    return t.astimezone(CHINA)


def validate(review, observed):
    if not isinstance(review, dict) or set(review) != set(FIELDS):
        raise ValueError("Settlement review needs exactly the documented fields")
    if (review["scope"] != "COMPLETE_ORDINARY_A_SHARE_CASH_AND_BONUS_ONLY"
            or review["review_status"] != "ISSUER_IMPLEMENTATION_NOTICE_VERIFIED"):
        raise ValueError("Only fully reviewed ordinary A-share implementation terms are admissible")
    if (not isinstance(review["stock_code"], str) or not trading_unit(review["stock_code"])
            or any(not isinstance(review[k], str) or not review[k].strip()
                   for k in ("source_id", "reviewer", "evidence_excerpt"))
            or type(review["pdf_page"]) is not int or review["pdf_page"] < 1
            or not isinstance(review["source_sha256"], str)
            or not re.fullmatch("[a-f0-9]{64}", review["source_sha256"])):
        raise ValueError("Missing stock, immutable source reference or reviewer attestation")
    url = urlsplit(review["source_url"])
    host = url.hostname or ""
    if (url.scheme != "https" or url.username or url.password or url.port not in (None,443)
            or not any(host == h or host.endswith("." + h) for h in HOSTS) or not url.path.strip("/")):
        raise ValueError("Source must identify the official disclosure on an exchange or CNINFO")
    if not timestamp(review["published_at"]) <= timestamp(review["reviewed_at"]) <= observed:
        raise ValueError("Publication and review must precede actual archive observation")
    ex = iso(review["ex_date"])
    if iso(review["record_date"]) >= ex:
        raise ValueError("Need a record date strictly before the ex-date")
    try:
        cash, multi = (Decimal(str(review[k])) for k in ("cash_per_share", "share_multiplier"))
        if (not cash.is_finite() or not multi.is_finite() or cash < 0 or multi < 1
                or not all(math.isfinite(float(x)) for x in (cash,multi)) or (cash == 0 and multi == 1)):
            raise ValueError("Invalid per-old-share settlement terms")
    except (ValueError, InvalidOperation, OverflowError) as error:
        raise ValueError("Invalid per-old-share settlement terms") from error
    if ((cash > 0 and (review["cash_basis"] != "VERIFIED_ACCOUNT_CREDIT_PER_OLD_SHARE"
                       or iso(review["cash_pay_date"]) < ex))
            or (cash == 0 and (review["cash_pay_date"] is not None or review["cash_basis"] != "NO_CASH"))
            or (multi > 1 and iso(review["share_list_date"]) < ex)
            or (multi == 1 and review["share_list_date"] is not None)):
        raise ValueError("Need exact credit/listing dates and a resolved cash basis")
    return review


def review_id(review):
    blob = json.dumps(review, sort_keys=True, allow_nan=False, ensure_ascii=False, separators=(",",":")).encode()
    return hashlib.sha256(blob).hexdigest()


def official_index_match(review, path):
    index = read_json(path)
    if not isinstance(index, list):
        raise ValueError("Expected an archived official-notice index")
    proof = {"stock_code": review["stock_code"], "ex_date": review["ex_date"],
             "file_sha256": review["source_sha256"], "source_url": review["source_url"],
             "official_notice_id": review["source_id"],
             "published_on": timestamp(review["published_at"]).date().isoformat()}
    if not any(isinstance(r, dict) and all(r.get(k) == v for k,v in proof.items()) for r in index):
        raise ValueError("PDF, notice, issuer, publication date and ex-date must match the source index")


def read_receipt(folder):
    record = read_json(folder/"receipt.json")
    observed = timestamp(record["observed_at"])
    r = validate(record["review"], observed)
    official_index_match(r,folder/"source_index.json")
    if (hash_file(folder/"source_index.json") != record["source_index_sha256"] or record["version"] != VERSION or folder.name != review_id(r) or folder.parent.name != r["stock_code"]
            or folder.parent.parent.name != r["ex_date"] or folder.is_symlink()
            or (folder/"source.pdf").is_symlink() or hash_file(folder/"source.pdf") != r["source_sha256"]):
        raise ValueError("Settlement receipt or official evidence digest mismatch")
    return r, observed


def capture(root, review_path, pdf, source_index):
    """Publish a complete, manually reviewed bundle. Existing receipts are immutable."""
    root = Path(root)
    r = validate(read_json(review_path), datetime.now(CHINA))
    official_index_match(r, source_index)
    rid = review_id(r)
    with exclusive(root):
        parent = root/r["ex_date"]/r["stock_code"]
        parent.mkdir(parents=True, exist_ok=True)
        final = parent/rid
        if hash_file(pdf) != r["source_sha256"]:
            raise ValueError("Provided PDF differs from the reviewed source digest")
        if final.exists():
            read_receipt(final)
            return final
        with TemporaryDirectory(dir=parent) as tmp:
            stage = Path(tmp)/"staging"
            stage.mkdir()
            saved = stage/"source.pdf"
            shutil.copyfile(pdf, saved)
            with saved.open("rb") as f:
                if f.read(5) != b"%PDF-":
                    raise ValueError("Original official PDF required")
            if hash_file(saved) != r["source_sha256"]:
                raise ValueError("Original PDF changed while archiving")
            index = stage/"source_index.json"
            shutil.copyfile(source_index,index)
            official_index_match(r,index)
            observed = datetime.now(CHINA)
            atomic_json(stage/"receipt.json", {"version":VERSION,"observed_at":observed.isoformat(),
                                             "source_index_sha256":hash_file(index), "review":r})
            os.rename(stage,final)
    return final


def load_settlements(root, day, known_at):
    """Return stock events plus visible receipt hashes. Never backdate observations.

    Conflicting reviewed interpretations require manual resolution. If an
    already-known morning plan conflicts with a later receipt, keep its actual
    pre-open terms and set late_conflict for the closing valuation audit.
    """
    day = iso(day)
    now = timestamp(known_at)
    if now.date().isoformat() < day:
        return {}, []
    cutoff = min(now, datetime.combine(datetime.fromisoformat(day).date(),time.max,CHINA))
    opening = datetime.combine(cutoff.date(),time(9,15),CHINA)
    records = defaultdict(list)
    digests = []
    if root is not None:
        for manifest in sorted((Path(root)/day).glob("*/*/receipt.json")):
            r, observed = read_receipt(manifest.parent)
            if observed > cutoff:
                continue
            records[r["stock_code"]].append((r,observed))
            digests.append(hash_file(manifest))

    def consolidate(items, at_open):
        keys = {(r["record_date"], Decimal(str(r["cash_per_share"])),Decimal(str(r["share_multiplier"])),
                 r["cash_pay_date"],r["share_list_date"],r["cash_basis"]) for r,_ in items}
        if len(keys) != 1:
            return {"status":"UNKNOWN","available_at_open":at_open,"reason":"CONFLICTING_SETTLEMENT_RECEIPTS"}
        r = items[0][0]
        return {"status":"VERIFIED", "available_at_open":at_open, "record_date":r["record_date"],
                "cash_per_share":float(r["cash_per_share"]), "share_multiplier":float(r["share_multiplier"]),
                "cash_pay_date":r["cash_pay_date"],"share_list_date":r["share_list_date"]}

    events = {}
    for code, items in records.items():
        early = [entry for entry in items if entry[1] <= opening]
        latest = consolidate(items, bool(early))
        if early and latest["status"] == "UNKNOWN":
            event = consolidate(early, True)
            if event["status"] == "VERIFIED":
                event["late_conflict"] = True
        else:
            event = latest
        events[code] = event
    return events, digests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    ingest = sub.add_parser("capture")
    ingest.add_argument("--review",type=Path,required=True)
    ingest.add_argument("--pdf",type=Path,required=True)
    ingest.add_argument("--source-index",type=Path,required=True)
    ingest.add_argument("--root",type=Path,required=True)
    status = sub.add_parser("inspect")
    status.add_argument("--root",type=Path,required=True)
    status.add_argument("--day",required=True)
    args = parser.parse_args()
    if args.command == "capture":
        print(capture(args.root,args.review,args.pdf,args.source_index))
    else:
        events, digests = load_settlements(args.root,args.day,datetime.now(CHINA))
        print(json.dumps({"ex_date":args.day,"visible_events":events,"receipt_sha256":digests},ensure_ascii=False))


if __name__ == "__main__":
    main()
