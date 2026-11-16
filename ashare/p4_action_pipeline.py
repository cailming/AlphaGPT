"""Resume and audit immutable, pre-test A-share corporate-action requests."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import random
import shutil
import signal
import tempfile
import time

import pandas as pd

from . import p4_action_feed_baostock as bao
from . import p4_action_feed_free as free

AK_ROOT = Path("reports/ashare_p4_action_feed_free")
BAO_ROOT = Path("reports/ashare_p4_action_feed_baostock")
SNAPSHOT = AK_ROOT / "snapshot_2024-12-01_2026-06-11"
METADATA = Path("reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json")
CONTROLS = Path("ashare/data/corporate_actions_verified_examples.csv")
OUTPUT = Path("reports/ashare_p4_action_pipeline")
CUTOFF = "2026-06-12"
BAO = "baostock"
SUCCESS = "SUCCESS"
SOURCES = {**{endpoint: "sina" if endpoint.startswith("sina") else "cninfo"
              for endpoint in free.ENDPOINTS}, BAO: BAO}
RETRYABLE = {"CONNECTION_ERROR", "TIMEOUT", "RATE_LIMIT", "API_ERROR",
             "UPSTREAM_EMPTY_OR_SCHEMA"}
CIRCUIT_ERRORS = RETRYABLE | {"SCHEMA_CHANGE", "ROW_LIMIT"}
DATE_DIAGNOSTICS = ("missing_ex_date_rows", "invalid_ex_date_rows", "missing_announcement_date_rows",
                    "invalid_announcement_date_rows", "wrong_query_year_rows")
AK_COLUMNS = ["stock_code", "endpoint", "status", "response_rows",
              "dropped_future_or_invalid", "error"]
BAO_COLUMNS = ["stock_code", "bs_code", "year", "status", "response_rows",
               "saved_rows", "excluded_out_of_window", "unknown_ex_date_rows", "error"]
EXCEPTION_COLUMNS = ["stock_code", "date_or_year", "source", "category", "evidence",
                     "retry_count", "state", "automatic", "action", "priority"]
LOG = logging.getLogger(__name__)
CONFIG_DEFAULTS = {
    "batch_size": 20, "max_attempts": 3, "backoff": 2.0, "backoff_cap": 60.0,
    "sina_interval": 1.1, "cninfo_interval": 1.5, "baostock_interval": 1.1,
    "timeout": 35, "circuit_window": 5, "error_rate": 0.6, "min_free_mb": 100,
}


class EvidenceError(ValueError):
    """A batch failed integrity, provenance, or isolation checks."""


class RequestTimeout(BaseException):
    """Escape provider exception handlers when a hard request deadline expires."""


@dataclass(frozen=True, order=True)
class Task:
    stock_code: str
    endpoint: str
    year: int = 0


@dataclass
class Attempt:
    task: Task
    entry: dict
    rows: list[dict]
    evidence: Path
    category: str
    number: int = 0
    finished: float = 0
    retry_at: float = 0


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def atomic_json(path, value):
    """Publish a fully flushed file in one rename on the same filesystem."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temporary.unlink(missing_ok=True)


def category(entry):
    status = entry["status"]
    error = str(entry.get("error", "")).lower()
    if status != "API_ERROR":
        return status
    if "429" in error or "too many requests" in error or "rate limit" in error:
        return "RATE_LIMIT"
    if any(word in error for word in ("timed out", "timeout", "deadline")):
        return "TIMEOUT"
    if any(word in error for word in ("schema", "column metadata", "dataframe", "field length",
                                      "keyerror", "attributeerror", "decodeerror")):
        return "SCHEMA_CHANGE"
    if any(word in error for word in ("connection", "network", "socket", "connect", "reset by peer")):
        return "CONNECTION_ERROR"
    return status


def check_rows(task, rows, meta):
    if not rows:
        return
    if task.endpoint == BAO:
        for row in rows:
            ex = bao._nullable_day(row["dividOperateDate"])
            if (str(row["query_year"]) != str(task.year) or row["request_code"] != task.stock_code
                    or row["code"] != bao._provider_code(task.stock_code) or pd.isna(ex)
                    or ex.year != task.year or not meta["start"] <= str(ex.date()) < CUTOFF):
                raise EvidenceError("BaoStock row is outside its request or frozen interval")
            for column in ("dividPlanDate", "dividRegistDate"):
                if bao._nullable_day(row[column]) >= pd.Timestamp(CUTOFF):
                    raise EvidenceError("BaoStock date enters old test period")
    else:
        if any(row["request_code"] != task.stock_code for row in rows):
            raise EvidenceError("AkShare row belongs to a different request")
        _, removed = free._prune(pd.DataFrame(rows), task.endpoint, CUTOFF)
        if removed:
            raise EvidenceError("AkShare row has a future or invalid action date")
        if any(bao._nullable_day(row["股权登记日"]) >= pd.Timestamp(CUTOFF) for row in rows):
            raise EvidenceError("AkShare record date enters old test period")


def legacy_attempts(paths, is_bao, meta, codes, years):
    seen_batches = set()
    for manifest_path in sorted(paths):
        try:
            batch = read_json(manifest_path)
            version = batch.get("version") if is_bao else batch.get("batch_version")
            expected_version = bao.VERSION if is_bao else "P4_AKSHARE_NO_TOKEN_BATCH_V1"
            if (version != expected_version or batch["snapshot_sha256"] != meta["universe_sha256"]
                    or batch["test_start_EXCLUDED"] != CUTOFF
                    or (is_bao and batch["start"] != meta["start"])):
                raise EvidenceError("Wrong manifest snapshot or frozen interval")
            endpoints = (BAO,) if is_bao else free.ENDPOINTS
            raw_files = {ep: f"raw_{'baostock_dividends' if is_bao else ep}.csv"
                         for ep in endpoints}
            checksums = batch["files_sha256"]
            if set(checksums) != {"request_ledger.csv", *raw_files.values()}:
                raise EvidenceError("Incomplete SHA256 manifest")
            for filename, expected in checksums.items():
                if free.digest(manifest_path.parent / filename) != expected:
                    raise EvidenceError(f"SHA256 mismatch in {filename}")
            fingerprint = free.digest(manifest_path)
            if fingerprint in seen_batches:
                continue
            seen_batches.add(fingerprint)

            ledger = free._read(manifest_path.parent / "request_ledger.csv")
            expected_columns = BAO_COLUMNS if is_bao else AK_COLUMNS
            if not set(expected_columns).issubset(ledger):
                raise EvidenceError("Missing request ledger fields")
            expected_calls = batch["requested_queries" if is_bao else "api_calls"]
            if len(ledger) != expected_calls or len(ledger) == 0:
                raise EvidenceError("Wrong request ledger length")
            raw_by_task = {}
            for ep, filename in raw_files.items():
                table = free._read(manifest_path.parent / filename)
                columns = (("request_code", "query_year") + bao.RAW_FIELDS
                           if is_bao else ("request_code",) + free._cols(ep))
                if not set(columns).issubset(table):
                    raise EvidenceError(f"Missing raw columns in {filename}")
                for row in table[list(columns)].to_dict("records"):
                    key = Task(row["request_code"], ep, int(row["query_year"]) if is_bao else 0)
                    raw_by_task.setdefault(key, []).append(row)
            keys = set()
            for entry in ledger[expected_columns].to_dict("records"):
                ep = BAO if is_bao else entry["endpoint"]
                year = int(entry["year"]) if is_bao else 0
                task = Task(entry["stock_code"], ep, year)
                if (task in keys or task.stock_code not in codes or ep not in endpoints
                        or (is_bao and year not in years)):
                    raise EvidenceError("Duplicate or out-of-scope request in one batch")
                keys.add(task)
                rows = raw_by_task.pop(task, [])
                if rows and entry["status"] not in {SUCCESS, "UNKNOWN_EX_DATE"}:
                    raise EvidenceError("Raw rows have no compatible request result")
                check_rows(task, rows, meta)
                result = category(entry)
                if ep == "cninfo_dividend" and result == SUCCESS and int(entry["response_rows"]) == 0:
                    result = "UPSTREAM_EMPTY_OR_SCHEMA"
                yield Attempt(task, entry, rows, manifest_path, result)
            if raw_by_task:
                raise EvidenceError("Raw rows have no request ledger entry")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise EvidenceError(f"{manifest_path}: {exc}") from exc


class Evidence:
    def __init__(self, args):
        try:
            self.meta, self.universe = bao._snapshot(args.snapshot, args.metadata)
            free._snapshot(args.snapshot, CUTOFF)
        except (ValueError, OSError, KeyError) as exc:
            raise EvidenceError(f"Frozen snapshot or Ridge metadata invalid: {exc}") from exc
        self.args = args
        self.identity = {"snapshot_sha256": self.meta["universe_sha256"],
                         "metadata_sha256": free.digest(args.metadata),
                         "start": self.meta["start"], "test_start_EXCLUDED": CUTOFF}
        self.codes = tuple(self.universe.stock_code)
        self.years = tuple(range(int(self.meta["start"][:4]), int(CUTOFF[:4]) + 1))
        self.attempts = {}
        self.legacy_manifests = {}
        codes = set(self.codes)
        for root, is_bao in [(args.ak_root, False), (args.bao_root, True)]:
            if not root.is_dir():
                raise EvidenceError(f"Provider evidence root is missing: {root}")
            paths = sorted(root.glob("batches*/batch_*/batch.json"))
            self.legacy_manifests.update({str(path.resolve()): free.digest(path) for path in paths})
            for attempt in legacy_attempts(paths, is_bao, self.meta, codes, self.years):
                self.add(attempt)
        for path in sorted((args.output / "attempts").glob("*.json")):
            try:
                signed = read_json(path)
                data = signed["attempt"]
                if hashlib.sha256(canonical(data)).hexdigest() != signed["sha256"]:
                    raise EvidenceError("Attempt SHA256 mismatch")
                if data["identity"] != self.identity:
                    raise EvidenceError("Attempt uses different frozen inputs")
                task = Task(**data["task"])
                number = data["number"]
                if (task.stock_code not in codes or task.endpoint not in SOURCES
                        or task.year not in (self.years if task.endpoint == BAO else (0,))
                        or number != self.new_count(task) + 1
                        or path.name != self.filename(task, number)):
                    raise EvidenceError("Attempt key or sequence mismatch")
                entry = data["entry"]
                if (entry["stock_code"] != task.stock_code or entry["endpoint"] != task.endpoint
                        or entry["year"] != task.year or category(entry) != data["category"]
                        or (data["rows"] and data["category"] not in {SUCCESS, "UNKNOWN_EX_DATE"})):
                    raise EvidenceError("Attempt result or raw rows do not match its request")
                check_rows(task, data["rows"], self.meta)
                self.add(Attempt(task, data["entry"], data["rows"], path, data["category"],
                                 number, data["finished"], data["retry_at"]))
            except (ValueError, OSError, KeyError, TypeError) as exc:
                raise EvidenceError(f"{path}: {exc}") from exc

    def add(self, attempt):
        self.attempts.setdefault(attempt.task, []).append(attempt)

    def new_count(self, task):
        return sum(attempt.number > 0 for attempt in self.attempts.get(task, ()))

    @staticmethod
    def filename(task, number):
        return f"{task.stock_code}.{task.endpoint}.{task.year}.{number:08}.json"

    def persist(self, task, entry, rows, result):
        check_rows(task, rows, self.meta)
        number = self.new_count(task) + 1
        finished = time.time()
        delay = min(self.args.backoff_cap, self.args.backoff * 2 ** min(number - 1, 30))
        retry_at = finished + random.uniform(delay / 2, delay) if result in RETRYABLE else 0
        path = self.args.output / "attempts" / self.filename(task, number)
        if path.exists():
            raise EvidenceError(f"Attempt already exists: {path}")
        data = {"identity": self.identity, "task": vars(task), "entry": entry, "rows": rows,
                "category": result, "number": number, "finished": finished, "retry_at": retry_at}
        atomic_json(path, {"attempt": data, "sha256": hashlib.sha256(canonical(data)).hexdigest()})
        self.add(Attempt(task, entry, rows, path, result, number, finished, retry_at))

    def effective(self, task):
        attempts = self.attempts.get(task, ())
        # Legacy evidence is sorted by manifest path and copies are excluded.
        # Preserve the first success. Only fall back to the last failure.
        return next((item for item in attempts if item.category == SUCCESS),
                    attempts[-1] if attempts else None)

    def state(self, task):
        result = self.effective(task)
        if result is None:
            return "NOT_FETCHED"
        if result.category == SUCCESS:
            return SUCCESS
        if result.category in RETRYABLE and self.new_count(task) < self.args.max_attempts:
            return "RETRY_PENDING"
        return "TERMINAL_EXCEPTION"

    def journal(self):
        return sorted((attempt for entries in self.attempts.values()
                       for attempt in entries if attempt.number), key=lambda attempt: attempt.finished)

    def tasks(self, code):
        return [*(Task(code, ep) for ep in free.ENDPOINTS),
                *(Task(code, BAO, year) for year in self.years)]


@contextmanager
def exclusive(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A pipeline writer is already running for this output") from exc
        yield


@contextmanager
def deadline(seconds):
    def expire(*_):
        raise RequestTimeout()

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class Providers:
    def __init__(self, meta, timeout):
        import baostock
        self.ak = free._import_akshare()
        self.bs = baostock
        self.logged_in = False
        self.begin = pd.Timestamp(meta["start"])
        self.timeout = timeout

    def close(self):
        if self.logged_in:
            self.logged_in = False
            try:
                with deadline(min(self.timeout, 5)), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.bs.logout()
            except (Exception, RequestTimeout):
                LOG.warning("BaoStock logout failed or timed out")

    def __call__(self, task):
        try:
            with deadline(self.timeout), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                if task.endpoint != BAO:
                    entry, rows = free.fetch_request(self.ak, task.stock_code, task.endpoint,
                                                     cutoff=CUTOFF, row_cap=1000,
                                                     audit_start=str(self.begin.date()))
                    return entry, rows.astype("string").fillna("").to_dict("records")
                if bao._provider_code(task.stock_code) and not self.logged_in:
                    login = self.bs.login()
                    if str(login.error_code) != "0":
                        raise ConnectionError("BaoStock login failed")
                    self.logged_in = True
                entry, rows = bao.fetch_query(self.bs, task.stock_code, task.year,
                                              begin=self.begin, cut=pd.Timestamp(CUTOFF), cap=1000)
            if category(entry) in {"CONNECTION_ERROR", "TIMEOUT"}:
                self.close()
            return entry, rows
        except RequestTimeout:
            if task.endpoint == BAO:
                self.close()
            return {"status": "API_ERROR", "error": "request deadline timeout"}, []
        except ConnectionError:
            return {"status": "API_ERROR", "error": "connection login failure"}, []


def frame(records, columns):
    return pd.DataFrame(records, columns=columns).astype("string").fillna("")


def disposition(evidence, tasks):
    rows = []
    for task in tasks:
        attempt = evidence.effective(task)
        history = evidence.attempts.get(task, ())
        rows.append({"stock_code": task.stock_code, "endpoint": task.endpoint, "year": task.year,
                     "state": evidence.state(task), "result": attempt.category if attempt else "",
                     "evidence": str(attempt.evidence) if attempt else "",
                     "attempts": len(history), "new_attempts": evidence.new_count(task),
                     "response_rows": int(attempt.entry["response_rows"] or 0) if attempt else 0,
                     "legacy_unclassified_date_error": bool(attempt and attempt.category == "UNKNOWN_EX_DATE"
                                                            and "missing_ex_date_rows" not in attempt.entry),
                     **{field: attempt.entry.get(field) if attempt else None
                        for field in DATE_DIAGNOSTICS}})
    return pd.DataFrame(rows)


def tables(evidence, codes, tasks):
    status = disposition(evidence, tasks)
    scope = evidence.universe.loc[evidence.universe.stock_code.isin(codes)]
    coverage = status.groupby("stock_code", sort=False).state.agg(
        scanned=lambda x: not x.eq("NOT_FETCHED").all(),
        successful=lambda x: int(x.eq(SUCCESS).sum()),
        not_fetched=lambda x: int(x.eq("NOT_FETCHED").sum()),
        retry_pending=lambda x: int(x.eq("RETRY_PENDING").sum()),
        terminal_exceptions=lambda x: int(x.eq("TERMINAL_EXCEPTION").sum())).reset_index()
    coverage["effective_full_request_coverage"] = coverage.successful.eq(len(evidence.tasks(codes[0])))
    coverage["historical_coverage_certified"] = False
    health = status.groupby("endpoint", sort=False).state.agg(
        planned="size", successful=lambda x: int(x.eq(SUCCESS).sum()),
        unsuccessful=lambda x: int(x.isin(["RETRY_PENDING", "TERMINAL_EXCEPTION"]).sum()),
        not_fetched=lambda x: int(x.eq("NOT_FETCHED").sum())).reset_index()
    empty = status.state.eq(SUCCESS) & status.response_rows.eq(0)
    health["successful_empty_responses"] = health.endpoint.map(
        status.loc[empty].groupby("endpoint").size()).fillna(0).astype(int)
    health["success_rate_observed"] = health.successful.div(
        (health.successful + health.unsuccessful).replace(0, pd.NA))
    ak_ledger, bao_ledger, bao_raw = [], [], []
    ak_raw = {endpoint: [] for endpoint in free.ENDPOINTS}
    exceptions = []
    for task, row in zip(tasks, status.to_dict("records"), strict=True):
        result = evidence.effective(task)
        if result is None:
            continue
        if row["state"] != SUCCESS:
            pending = row["state"] == "RETRY_PENDING"
            exceptions.append({
                "stock_code": task.stock_code, "date_or_year": task.year or "",
                "source": task.endpoint, "category": row["result"],
                "evidence": row["evidence"], "retry_count": max(row["attempts"] - 1, 0),
                "state": row["state"], "automatic": pending,
                "action": "Retry this provider request" if pending else "Review upstream evidence",
                "priority": {"WRONG_STOCK_CODE": "P0", "UNKNOWN_EX_DATE": "P1"}.get(row["result"], "P2"),
            })
        entry = result.entry | {"status": result.category}
        if task.endpoint == BAO:
            bao_ledger.append(entry)
            bao_raw.extend(result.rows)
        else:
            ak_ledger.append(entry)
            ak_raw[task.endpoint].extend(result.rows)
    ak_report, ak_actions, ak_stocks, ak_issues, rights, ak_controls = free.audit_evidence(
        evidence.meta, scope, frame(ak_ledger, AK_COLUMNS),
        {ep: frame(rows, ("request_code",) + free._cols(ep)) for ep, rows in ak_raw.items()},
        official_events=evidence.args.controls)
    bao_report, bao_actions, _, compare, bao_controls = bao.audit_evidence(
        evidence.meta, scope, frame(bao_ledger, BAO_COLUMNS),
        frame(bao_raw, ("request_code", "query_year") + bao.RAW_FIELDS),
        akshare_events=ak_actions, akshare_stock_coverage=ak_stocks,
        issuer_controls=evidence.args.controls)
    return (status, coverage, health, exceptions, ak_report, ak_actions, ak_issues,
            rights, ak_controls, bao_report, bao_actions, compare, bao_controls)


def write_report(evidence, codes, checkpoint):
    tasks = [task for code in codes for task in evidence.tasks(code)]
    (status, coverage, health, exceptions, ak_report, ak, ak_issues, rights,
     ak_controls, bao_report, bs, compare, bao_controls) = tables(evidence, codes, tasks)

    sources_by_code = status.groupby("stock_code").evidence.agg(
        lambda paths: "|".join(sorted(set(paths) - {""}))).to_dict()
    retries_by_code = status.groupby("stock_code").attempts.agg(
        lambda counts: sum(max(n - 1, 0) for n in counts)).to_dict()

    def audit_issue(code, date, source, reason, priority, action):
        exceptions.append({
            "stock_code": code, "date_or_year": date, "source": source, "category": reason,
            "evidence": sources_by_code.get(code, ""), "retry_count": retries_by_code.get(code, 0),
            "state": "MANUAL_REVIEW", "automatic": False, "action": action, "priority": priority,
        })

    for item in ak_issues.itertuples(index=False):
        audit_issue(item.stock_code, item.ex_date, item.endpoint, item.reason,
                    "P2" if "Missing entitlement" in item.reason else "P1", "Check issuer implementation notice")
    for item in rights.to_dict("records"):
        audit_issue(item["request_code"], item["除权日"], "sina_rights", "RIGHTS_ISSUE",
                    "P1", "Check rights terms and treatment before computing returns")
    for item in bs.itertuples(index=False):
        if item.blank_amount_fields:
            audit_issue(item.stock_code, item.ex_date, BAO, "MISSING_ENTITLEMENT_FIELDS",
                        "P2", "Check " + item.blank_amount_fields + " against issuer notice")
    issues = compare[compare.result.isin(["BAOSTOCK_AKSHARE_TERMS_CONFLICT",
                                         "BAOSTOCK_AKSHARE_DATE_CONFLICT", "BAOSTOCK_ONLY_UNVERIFIED",
                                         "AKSHARE_ONLY_UNVERIFIED", "AKSHARE_DUPLICATE_UNRESOLVED",
                                         "INVALID_BAOSTOCK_ROW"])]
    for item in issues.fillna("").itertuples(index=False):
        audit_issue(item.stock_code, item.ex_date, "cross_source",
                    getattr(item, "reason", "") or item.result, "P1", "Check issuer implementation and date")
    for item in ak_controls.itertuples(index=False):
        if item.status == "MISSING_OR_CONFLICTING":
            audit_issue(item.stock_code, item.ex_date, "issuer_control", item.status, "P1",
                        "Resolve AkShare and issuer control difference")
    for item in bao_controls.itertuples(index=False):
        if item.result == "MISSING_OR_CONFLICTING_BAOSTOCK_CANDIDATE":
            audit_issue(item.stock_code, item.ex_date, BAO, item.result, "P1",
                        "Resolve BaoStock and issuer control difference")

    dates = ak[["stock_code", "ex_date", "record_date"]].merge(
        bs[["stock_code", "ex_date", "record_date"]], on=["stock_code", "record_date"],
        suffixes=("_akshare", "_baostock"))
    for row in dates.loc[dates.ex_date_akshare.ne(dates.ex_date_baostock)].itertuples(index=False):
        audit_issue(row.stock_code, row.ex_date_akshare, "cross_source", "EX_DATE_MISMATCH",
                    "P1", "Check same record date with BaoStock ex-date " + row.ex_date_baostock)
    key_columns = ["stock_code", "ex_date"]
    ak = ak[key_columns + ["record_date", "review_status", "sources_seen", *free.COMPONENTS]].rename(
        columns={"review_status": "akshare_result", "record_date": "akshare_record_date",
                 **{field: "akshare_" + field for field in free.COMPONENTS}})
    bs = bs[key_columns + ["record_date", "blank_amount_fields", *free.COMPONENTS]].copy()
    for raw_field, column in bao.AMOUNT_COLUMNS.items():
        bs[column] = bs[column].where(~bs.blank_amount_fields.str.contains(raw_field, regex=False), None)
    bs = bs.rename(columns={"record_date": "baostock_record_date",
                            **{field: "baostock_" + field for field in free.COMPONENTS}})
    reconciliation = ak.merge(bs, on=["stock_code", "ex_date"], how="outer", validate="one_to_one")
    comparisons = compare[compare.result.ne("INVALID_BAOSTOCK_ROW")]
    reconciliation = reconciliation.merge(comparisons[["stock_code", "ex_date", "result"]],
                                          on=["stock_code", "ex_date"], how="left", validate="one_to_one")
    observed = reconciliation.sources_seen.fillna("").map(lambda x: len(x.split("|")) if x else 0)
    reconciliation["sources_observed"] = (
        observed + reconciliation.baostock_record_date.notna().astype(int))
    reconciliation["evidence"] = reconciliation.stock_code.map(sources_by_code)

    exceptions = pd.DataFrame(exceptions, columns=EXCEPTION_COLUMNS).drop_duplicates(
        ["stock_code", "date_or_year", "source", "category"]).sort_values(
        ["priority", "stock_code", "date_or_year", "source"], key=lambda x: x.astype(str))
    complete_requests = int(status.state.eq(SUCCESS).sum())
    terminal = int(status.state.eq("TERMINAL_EXCEPTION").sum())
    offered = len(status)
    p0 = int(exceptions.priority.eq("P0").sum())
    controls = pd.concat([
        ak_controls.rename(columns={"reference_id": "source_id", "status": "result"}).assign(source="akshare"),
        bao_controls.assign(source=BAO)
    ], ignore_index=True)
    out_of_scope_controls = controls.loc[~controls.stock_code.isin(codes), "stock_code"].unique()
    # Controls such as BYD have a small, pre-existing independent evidence set
    # outside the pilot. Audit it without expanding collection.
    if len(out_of_scope_controls):
        old_codes = [code for code in out_of_scope_controls if code in evidence.codes
                     and any(evidence.effective(task) for task in evidence.tasks(code))]
        if old_codes:
            control_tables = tables(evidence, old_codes,
                                    [task for code in old_codes for task in evidence.tasks(code)])
            historical_controls = pd.concat([
                control_tables[8].rename(columns={"reference_id": "source_id", "status": "result"}).assign(source="akshare"),
                control_tables[12].assign(source=BAO)
            ], ignore_index=True)
            historical_controls = historical_controls.loc[historical_controls.stock_code.isin(old_codes)]
            controls = pd.concat([controls.loc[~controls.stock_code.isin(old_codes)], historical_controls],
                                 ignore_index=True)
    circuit = checkpoint.get("circuit")
    paused_sources = checkpoint.get("paused_sources", {})
    summary = {
        **evidence.identity, "as_of_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "snapshot_stocks": len(evidence.codes), "stocks_in_scope": len(codes),
        "first_code": codes[0], "last_code": codes[-1],
        "stocks_scanned": int(coverage.scanned.sum()),
        "stocks_all_requests_successful": int(coverage.effective_full_request_coverage.sum()),
        "stocks_not_fetched": int((~coverage.scanned).sum()),
        "stocks_with_unattempted_requests": int(coverage.not_fetched.gt(0).sum()),
        "stocks_with_failures": int((coverage.terminal_exceptions + coverage.retry_pending).gt(0).sum()),
        "stocks_with_retry_pending": int(coverage.retry_pending.gt(0).sum()),
        "task_count": offered, "effective_successes": complete_requests,
        "effective_failures": int(status.state.isin(["TERMINAL_EXCEPTION", "RETRY_PENDING"]).sum()),
        "not_fetched_tasks": int(status.state.eq("NOT_FETCHED").sum()),
        "pending_retries": int(status.state.eq("RETRY_PENDING").sum()),
        "terminal_exceptions": terminal, "new_attempts": int(status.new_attempts.sum()),
        "historical_attempts": int((status.attempts - status.new_attempts).sum()),
        "unique_candidate_events": len(reconciliation),
        "akshare_candidate_events": len(ak), "baostock_candidate_events": len(bs),
        "sina_cninfo_matches": int(ak.akshare_result.eq("TWO_PUBLIC_SOURCES_MATCH_UNVERIFIED").sum()),
        "sina_cninfo_partial_field_matches_unresolved": int(
            ak.akshare_result.eq("TWO_PUBLIC_SOURCES_PARTIAL_FIELDS_UNRESOLVED").sum()),
        "baostock_full_field_matches": bao_report["cross_source_fully_observed_matches"],
        "baostock_partial_field_matches_unresolved": bao_report["cross_source_partial_amount_fields_unresolved"],
        "baostock_candidates_with_unknown_fields": int(bs.blank_amount_fields.ne("").sum()),
        "single_source_events": int(reconciliation.sources_observed.eq(1).sum()),
        "date_or_amount_conflicts": int(exceptions.category.str.contains(
            "CONFLICT|MISMATCH|date order|DUPLICATE_EX_DATE", case=False, na=False).sum()),
        "rights_records_in_scope": len(rights),
        "historical_rights_rows_in_evidence": ak_report["rights_records_raw_all_history"],
        "high_priority_exceptions": int(exceptions.priority.isin(["P0", "P1"]).sum()),
        "p0_exceptions": p0, "old_test_isolation_passed": True, "circuit": circuit,
        "paused_sources": paused_sources,
        "scope_ready_for_review": bool(complete_requests + terminal == offered
                                       and not p0 and not circuit and not paused_sources),
        "complete_verified_assertions_created": 0, "can_emit_training_labels": False,
        "training_gate": "CLOSED",
        "full_historical_action_coverage_certified": False,
        "issuer_control_results": {str(k): int(v) for k, v in controls.result.value_counts().items()},
    }
    output = evidence.args.output
    audits = output / "audits"
    audits.mkdir(exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="audit-", dir=audits))
    files = {
        "coverage_summary.csv": coverage, "source_health.csv": health,
        "cross_source_reconciliation.csv": reconciliation, "exceptions.csv": exceptions,
        "retry_queue.csv": exceptions.loc[exceptions.state.eq("RETRY_PENDING")],
        "manual_review_queue.csv": exceptions.loc[~exceptions.automatic.astype(bool)],
        "request_status.csv": status, "issuer_controls.csv": controls,
    }
    for filename, table in files.items():
        table.to_csv(target / filename, index=False)
    atomic_json(target / "pipeline_summary.json", summary)
    atomic_json(output / "latest_report.json", {"report_dir": str(target), "summary": summary})
    return summary


def track_attempt(checkpoint, args, attempt):
    task = attempt.task
    source = SOURCES[task.endpoint]
    key = f"{task.stock_code}/{task.endpoint}/{task.year}"
    checkpoint["attempt_numbers"][key] = attempt.number
    checkpoint["last_request"][source] = attempt.finished
    window = checkpoint["health_window"].setdefault(source, [])
    window[:] = [[old, status] for old, status in window if old != key]
    window.append([key, attempt.category])
    del window[:-args.circuit_window]
    failed = sum(status in CIRCUIT_ERRORS for _, status in window)
    if (attempt.category in {"RATE_LIMIT", "SCHEMA_CHANGE", "WRONG_STOCK_CODE"}
            or (len(window) == args.circuit_window and failed / len(window) >= args.error_rate)):
        checkpoint["circuit"] = {"source": source, "reason": attempt.category, "failed": failed,
                                 "observed": len(window), "last_task": key}


def run(evidence, checkpoint, codes, provider):
    args = evidence.args
    checkpoint_path = args.output / "checkpoint.json"
    stop = False

    def interrupt(*_):
        nonlocal stop
        stop = True

    def save():
        atomic_json(checkpoint_path, checkpoint)

    def available(task):
        source = SOURCES[task.endpoint]
        if source not in checkpoint.get("paused_sources", {}):
            return True
        return task.endpoint == BAO and bao._provider_code(task.stock_code) is None

    def collect_batch(tasks):
        for task in tasks:
            if stop or checkpoint.get("circuit"):
                break
            state = evidence.state(task)
            while state in {"NOT_FETCHED", "RETRY_PENDING"} and not stop:
                source = SOURCES[task.endpoint]
                last = checkpoint["last_request"].get(source, 0)
                prior = evidence.effective(task)
                ready_at = max(last + args.intervals[source], prior.retry_at if prior else 0)
                while not stop and ready_at > time.time():
                    time.sleep(max(0, min(1, ready_at - time.time())))
                if stop:
                    break
                entry, rows = provider(task)
                result = category(entry)
                # Avoid storing HTTP URLs, headers, or credentials in diagnostics.
                entry = {"response_rows": 0, **entry, "status": result, "error": result if result != SUCCESS else ""}
                entry.update(stock_code=task.stock_code, endpoint=task.endpoint, year=task.year)
                evidence.persist(task, entry, rows, result)
                track_attempt(checkpoint, args, evidence.effective(task))
                save()
                if result != SUCCESS:
                    LOG.warning("%s %s %s: %s", task.stock_code, task.endpoint, task.year or "-", result)
                if checkpoint.get("circuit"):
                    return
                state = evidence.state(task)

    old_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for start in range(0, len(codes), args.batch_size):
            batch = codes[start:start + args.batch_size]
            pending = [task for code in batch for task in evidence.tasks(code)
                       if available(task) and evidence.state(task) in {"NOT_FETCHED", "RETRY_PENDING"}]
            if not pending:
                continue
            if stop or checkpoint.get("circuit"):
                break
            preflight_space(args)
            LOG.info("Starting scope positions %s–%s, %s pending requests",
                     checkpoint["offset"] + start,
                     checkpoint["offset"] + start + len(batch) - 1, len(pending))
            collect_batch(pending)
            checkpoint["last_batch"] = {"first": batch[0], "last": batch[-1]}
            save()
            write_report(evidence, codes, checkpoint)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    summary = write_report(evidence, codes, checkpoint)
    checkpoint["state"] = ("BLOCKED" if checkpoint.get("circuit") else "INTERRUPTED" if stop else
                           "WAITING_FOR_SOURCES" if checkpoint.get("paused_sources") else
                           "DONE" if summary["scope_ready_for_review"] else "INCOMPLETE")
    save()
    return summary, checkpoint["state"]


def preflight_space(args):
    if shutil.disk_usage(args.output).free < args.min_free_mb * 1024 ** 2:
        raise RuntimeError(f"Free disk space below {args.min_free_mb} MiB")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "plan", "run", "resume", "audit", "report"))
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--ak-root", type=Path, default=AK_ROOT)
    parser.add_argument("--bao-root", type=Path, default=BAO_ROOT)
    parser.add_argument("--controls", type=Path, default=CONTROLS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--offset", type=int)
    parser.add_argument("--limit", "--max-stocks", type=int)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--backoff", type=float)
    parser.add_argument("--backoff-cap", type=float)
    parser.add_argument("--sina-interval", type=float)
    parser.add_argument("--cninfo-interval", type=float)
    parser.add_argument("--baostock-interval", type=float)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--circuit-window", type=int)
    parser.add_argument("--error-rate", type=float)
    parser.add_argument("--min-free-mb", type=int)
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--reset-circuit", action="store_true")
    recovery.add_argument("--isolate-circuit", action="store_true",
                          help="Keep the tripped source paused and run unaffected sources")
    recovery.add_argument("--restore-source", choices=tuple(sorted(set(SOURCES.values()))),
                          help="Re-enable a paused source after independently verifying recovery")
    args = parser.parse_args()
    if (args.offset is not None and args.offset < 0
            or args.limit is not None and args.limit < 1
            or args.full and (args.offset is not None or args.limit is not None)
            or args.command == "resume" and (args.full or args.offset is not None or args.limit is not None)):
        parser.error("Invalid scope")
    if (args.isolate_circuit or args.restore_source) and (args.command != "resume" or args.dry_run):
        parser.error("Source isolation and recovery require resume")
    return args


def validate_settings(args):
    args.intervals = {"sina": args.sina_interval, "cninfo": args.cninfo_interval,
                      BAO: args.baostock_interval}
    if (args.batch_size < 1 or args.max_attempts < 1 or args.circuit_window < 2
            or not 0 < args.error_rate <= 1 or args.timeout <= 0 or args.min_free_mb < 1
            or not 0 <= args.backoff <= args.backoff_cap or min(args.intervals.values()) < 0):
        raise ValueError("Invalid resource, retry or circuit settings")


def configure(evidence, args):
    path = args.output / "checkpoint.json"
    prior = read_json(path) if path.exists() else None
    if prior and prior["identity"] != evidence.identity:
        raise EvidenceError("Checkpoint frozen inputs differ")
    previous_config = prior.get("settings", {}) if prior else {}
    for key, default in CONFIG_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, previous_config.get(key, default))
    validate_settings(args)
    if args.command == "resume" and prior is None:
        raise ValueError("No checkpoint to resume")
    if args.full:
        offset, limit = 0, len(evidence.codes)
    elif args.offset is None and args.limit is None and prior:
        offset, limit = prior["offset"], prior["limit"]
    elif args.offset is None and args.limit is None:
        offset, limit = 10, 100
    else:
        offset = args.offset if args.offset is not None else 10
        limit = args.limit if args.limit is not None else 100
    codes = evidence.codes[offset:offset + limit]
    if len(codes) != limit:
        raise ValueError("Scope reaches beyond the frozen snapshot; specify an exact range")
    checkpoint = (prior or {"identity": evidence.identity, "last_request": {}, "health_window": {},
                            "state": "NOT_STARTED", "attempt_numbers": {}}) | {
                                "offset": offset, "limit": limit,
                                "settings": {key: getattr(args, key) for key in CONFIG_DEFAULTS}}
    for path, checksum in checkpoint.get("legacy_manifests", {}).items():
        if evidence.legacy_manifests.get(path) != checksum:
            raise EvidenceError(f"Previously audited legacy manifest missing or changed: {path}")
    checkpoint["legacy_manifests"] = evidence.legacy_manifests
    journal = evidence.journal()
    counts = {f"{a.task.stock_code}/{a.task.endpoint}/{a.task.year}": a.number for a in journal}
    for key, number in checkpoint["attempt_numbers"].items():
        if counts.get(key, 0) < number:
            raise EvidenceError(f"Previously recorded immutable attempt missing: {key}")
    for attempt in journal:
        task = attempt.task
        key = f"{task.stock_code}/{task.endpoint}/{task.year}"
        if attempt.number > checkpoint["attempt_numbers"].get(key, 0):
            track_attempt(checkpoint, args, attempt)
    circuit = checkpoint.get("circuit")
    if args.reset_circuit and args.command in {"run", "resume"} and not args.dry_run:
        if circuit and circuit["reason"] == "WRONG_STOCK_CODE":
            raise EvidenceError("A security identity mismatch requires P0 investigation")
        checkpoint.pop("circuit", None)
        checkpoint["health_window"] = {}
    if args.isolate_circuit:
        if circuit is None:
            raise ValueError("No active circuit to isolate")
        if circuit["reason"] == "WRONG_STOCK_CODE" or (args.output / "p0_blocker.json").exists():
            raise EvidenceError("P0 evidence must be resolved before continuing")
        checkpoint.setdefault("paused_sources", {})[circuit["source"]] = circuit
        del checkpoint["circuit"]
    if args.restore_source:
        paused = checkpoint.get("paused_sources", {})
        if args.restore_source not in paused:
            raise ValueError(f"Source is not paused: {args.restore_source}")
        del paused[args.restore_source]
        checkpoint["health_window"].pop(args.restore_source, None)
    return codes, checkpoint


def execute(args):
    evidence = Evidence(args)
    codes, checkpoint = configure(evidence, args)
    if args.command in {"status", "plan"} or args.dry_run:
        status = disposition(evidence, [task for code in codes for task in evidence.tasks(code)])
        return {"offset": checkpoint["offset"], "limit": len(codes),
                "first_stock": codes[0], "last_stock": codes[-1],
                "state": checkpoint["state"], "circuit": checkpoint.get("circuit"),
                "paused_sources": checkpoint.get("paused_sources", {}),
                "per_endpoint": status.groupby(["endpoint", "state"]).size().unstack(
                    fill_value=0).to_dict(orient="index")}, 0
    if checkpoint.get("circuit") and args.command in {"run", "resume"}:
        atomic_json(args.output / "checkpoint.json", checkpoint | {"state": "BLOCKED"})
    if args.command in {"audit", "report"} or checkpoint.get("circuit"):
        return write_report(evidence, codes, checkpoint), 2 if checkpoint.get("circuit") else 0
    preflight_space(args)
    if not args.controls.is_file():
        raise ValueError(f"Issuer controls missing: {args.controls}")
    (args.output / "attempts").mkdir(exist_ok=True)
    provider = Providers(evidence.meta, args.timeout)
    atomic_json(args.output / "checkpoint.json", checkpoint | {"state": "RUNNING"})
    try:
        summary, state = run(evidence, checkpoint | {"state": "RUNNING"}, codes, provider)
    finally:
        provider.close()
    return summary, 0 if state == "DONE" else 2


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        if args.command in {"status", "plan"} or args.dry_run:
            summary, exit_code = execute(args)
        else:
            with exclusive(args.output):
                try:
                    summary, exit_code = execute(args)
                except EvidenceError as exc:
                    atomic_json(args.output / "p0_blocker.json", {
                        "priority": "P0", "status": "BLOCKED", "detail": str(exc),
                        "as_of_utc": pd.Timestamp.now(tz="UTC").isoformat()})
                    raise
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        LOG.error("%s", exc)
        exit_code = 2
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
