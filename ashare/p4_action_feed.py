"""P4.3-E/2: Tushare candidate event ingestion and *non-certifying* coverage audit.

Three reproducible stages: snapshot -> fetch -> audit. Vendor data is never
promoted to COMPLETE_VERIFIED by this program. Requires independently verified
company actions and coverage for ashare.p4_corporate_actions training gate.
No production database writes. No old inspected test-period events are saved.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

SOURCE_URL = "https://tushare.pro/document/2?doc_id=103"
STOCK_BASIC_URL = "https://tushare.pro/document/2?doc_id=25"
FIELDS = (
    "ts_code", "end_date", "ann_date", "imp_ann_date", "div_proc", "stk_div",
    "stk_bo_rate", "stk_co_rate", "cash_div_tax", "record_date", "ex_date", "pay_date",
)
EVENT_FIELDS = (
    "stock_code", "ex_date", "published_on", "share_multiplier",
    "cash_per_old_share", "source_url", "source_id",
)
STATUS = ("SUCCESS", "API_ERROR", "NO_TS_MAPPING", "ROW_LIMIT")
WARNING = (
    "VENDOR CANDIDATES ONLY: successfully fetching every requested stock is NOT "
    "verified event coverage. Tushare dividend excludes rights issues and other "
    "complex actions; announcements/record dates/quantities require independent "
    "primary-source review. No COMPLETE_VERIFIED coverage certificates are issued; "
    "candidate events must not be used as cleared training labels. Existing old "
    "test period remains excluded."
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _date(x: str, name: str) -> pd.Timestamp:
    if not isinstance(x, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", x):
        raise ValueError(f"{name} must use YYYY-MM-DD")
    return pd.Timestamp(x)


def _ts_date(x, field: str, *, nullable: bool = False) -> pd.Timestamp:
    if pd.isna(x) or str(x).strip().lower() in ("", "nan", "none", "nat"):
        if nullable:
            return pd.NaT
        raise ValueError(f"Missing {field}")
    value = str(x).strip()
    if not re.fullmatch(r"\d{8}", value):
        raise ValueError(f"Invalid Tushare date {field}: {value}")
    try:
        return pd.to_datetime(value, format="%Y%m%d", errors="raise").normalize()
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date {field}: {value}") from exc


def _code(value: str) -> str:
    if pd.isna(value) or not re.fullmatch(r"\d{6}", str(value)):
        raise ValueError(f"Invalid six-digit code: {value}")
    return str(value)


def _ts_code(value: str) -> str:
    value = str(value)
    if not re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", value):
        raise ValueError(f"Invalid Tushare code: {value}")
    return value


def _ensure_cutoff(reference_metadata: Path) -> tuple[str, pd.Timestamp]:
    data = json.loads(reference_metadata.read_text(encoding="utf-8"))
    if data.get("model_version") != "P4_RIDGE_V1":
        raise ValueError("Reference metadata must be P4_RIDGE_V1")
    cut = _date(data["test_start"], "model test_start")
    return data["model_version"], cut


def normalize_snapshot(market: pd.DataFrame, basics: pd.DataFrame, *,
                       start: str, test_start: str) -> pd.DataFrame:
    """Union priced-market and L/D/P provider universe; do not infer exchange."""
    lo, cut = _date(start, "start"), _date(test_start, "test_start")
    if lo >= cut:
        raise ValueError("Invalid stock universe time interval")
    if "stock_code" not in market:
        raise ValueError("Market input requires stock_code")
    if not {"ts_code", "list_status", "list_date", "delist_date"} <= set(basics):
        raise ValueError("Provider stock_basic lacks required status/date fields")
    codes = {_code(code) for code in market.stock_code}
    found = {}
    for row in basics.itertuples(index=False):
        ts = _ts_code(row.ts_code)
        code = ts[:6]
        status = str(row.list_status)
        if status not in {"L", "D", "P"}:
            raise ValueError(f"Unexpected provider list_status: {status}")
        listed = _ts_date(row.list_date, "list_date")
        delisted = _ts_date(row.delist_date, "delist_date", nullable=True)
        if pd.notna(delisted) and delisted < listed:
            raise ValueError("Delist date predates list date")
        if listed >= cut or (pd.notna(delisted) and delisted < lo):
            continue
        if code in found:
            previous = found[code]
            if previous[0] != ts or previous[1] != status:
                raise ValueError(f"Ambiguous stock_basic mapping/status: {code}")
        found[code] = (ts, status, listed.date().isoformat(),
                       "" if pd.isna(delisted) else delisted.date().isoformat())
    results = []
    for code in sorted(codes | set(found)):
        info = found.get(code)
        results.append({
            "stock_code": code,
            "ts_code": info[0] if info else "",
            "list_status": info[1] if info else "UNKNOWN",
            "listed_on": info[2] if info else "",
            "delisted_on": info[3] if info else "",
            "present_in_formal_market": code in codes,
            "present_in_provider_basic": info is not None,
        })
    if not results:
        raise ValueError("No stock universe")
    return pd.DataFrame(results)


def load_formal_market_codes(start: str, test_start: str, *, engine=None) -> pd.DataFrame:
    """Read-only PostgreSQL universe, inclusive start and EXCLUSIVE test cut."""
    from sqlalchemy import text
    if engine is None:
        from .daily import AShareDailyLoader
        engine = AShareDailyLoader().engine
    begin, cut = _date(start, "start").date(), _date(test_start, "test_start").date()
    if begin >= cut:
        raise ValueError("Invalid market selection interval")
    sql = text("""SELECT DISTINCT stock_code FROM public.market
                  WHERE not_billing=1 AND trade_date >= :start
                    AND trade_date < :test_start
                    AND stock_code IS NOT NULL ORDER BY stock_code""")
    with engine.connect() as conn:
        return pd.read_sql_query(sql, conn, params={"start": begin, "test_start": cut})


def fetch_stock_basic(pro) -> pd.DataFrame:
    frames = []
    for status in ("L", "D", "P"):
        result = pro.stock_basic(exchange="", list_status=status,
                                 fields="ts_code,list_status,list_date,delist_date")
        if not isinstance(result, pd.DataFrame):
            raise ValueError("stock_basic failed to return a DataFrame")
        if len(result) >= 6000:
            raise ValueError("stock_basic reached provider limit; universe potentially incomplete")
        if "list_status" not in result:
            result = result.assign(list_status=status)
        if not {"ts_code", "list_status", "list_date", "delist_date"} <= set(result):
            raise ValueError("Provider stock_basic response lacks required fields")
        frames.append(result.loc[:, ["ts_code", "list_status", "list_date", "delist_date"]])
    return pd.concat(frames, ignore_index=True)


def _read_tushare_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype="string", keep_default_na=False, encoding="utf-8-sig")


def create_snapshot(*, market: pd.DataFrame, basics: pd.DataFrame,
                    start: str, test_start: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Snapshot output already exists (immutable snapshots)")
    universe = normalize_snapshot(market, basics, start=start, test_start=test_start)
    output.mkdir(parents=True)
    universe.to_csv(output / "universe.csv", index=False, encoding="utf-8")
    basics.to_csv(output / "stock_basic.csv", index=False, encoding="utf-8")
    stats = {
        "snapshot_version": "P4_ACTION_UNIVERSE_V1",
        "start": start, "test_start_EXCLUDED": test_start,
        "universe_sha256": sha256_file(output / "universe.csv"),
        "expected_symbols": len(universe),
        "priced_market_symbols": int(universe.present_in_formal_market.sum()),
        "provider_listing_symbols": int(universe.present_in_provider_basic.sum()),
        "unmapped_market_symbols": int((~universe.present_in_provider_basic).sum()),
        "known_delisted_symbols": int(universe.list_status.eq("D").sum()),
        "full_historical_market_certified": False,
        "warning": WARNING,
    }
    (output / "snapshot.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def _prune_future(raw: pd.DataFrame, *, cutoff: str) -> tuple[pd.DataFrame, int]:
    """Exclude future announced/implemented/ex-dated rows before writing raw.

    Past implementation rows with a pay_date later than the cutoff may be
    retained (entitlement was established on ex-date), but proposal rows
    published in the inspected old-test period are never persisted.
    """
    if not set(FIELDS) <= set(raw):
        raise ValueError(f"Dividend response missing fields: {sorted(set(FIELDS) - set(raw))}")
    cut = _date(cutoff, "cutoff")
    future = pd.Series(False, index=raw.index, dtype=bool)
    for field in ("ex_date", "ann_date", "imp_ann_date"):
        series = raw[field].astype("string").fillna("").str.strip()
        nonblank = ~series.str.lower().isin(["", "nan", "none", "nat"])
        dates = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
        for idx in raw.index[nonblank]:
            dates.at[idx] = _ts_date(series.at[idx], field)
        future |= nonblank & dates.ge(cut)
    return raw.loc[~future, list(FIELDS)].copy(), int(future.sum())


def fetch_batch(snapshot: Path, *, pro, offset: int, limit: int,
                out: Path, wait_seconds: float = 0.0, response_row_cap: int = 2000
                ) -> dict:
    """Bounded provider collection; errors explicitly logged, no auto certification."""
    if offset < 0 or limit < 1 or wait_seconds < 0 or response_row_cap < 1:
        raise ValueError("Invalid fetch offset/limit/wait/row cap")
    if out.exists():
        raise FileExistsError("Immutable batch output already exists; use a new path")
    meta = json.loads((snapshot / "snapshot.json").read_text(encoding="utf-8"))
    univ_path = snapshot / "universe.csv"
    if sha256_file(univ_path) != meta["universe_sha256"]:
        raise ValueError("Universe snapshot was modified")
    universe = _read_tushare_csv(univ_path)
    selected = universe.iloc[offset:offset + limit].copy()
    if selected.empty:
        raise ValueError("Offset past end of universe")
    raw_rows, log_rows = [], []
    for _, row in selected.iterrows():
        code, ts = _code(row.stock_code), str(row.ts_code or "")
        status, error, response_count, removed = "SUCCESS", "", 0, 0
        if not re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", ts):
            status = "NO_TS_MAPPING"
        else:
            try:
                result = pro.dividend(ts_code=ts, fields=",".join(FIELDS))
                if not isinstance(result, pd.DataFrame):
                    raise ValueError("Dividend response was not DataFrame")
                response_count = len(result)
                if response_count >= response_row_cap:
                    status = "ROW_LIMIT"
                else:
                    kept, removed = _prune_future(result, cutoff=meta["test_start_EXCLUDED"])
                    if len(kept):
                        # A returned row must match the queried symbol; no substitutions.
                        if not kept.ts_code.astype(str).eq(ts).all():
                            raise ValueError("Dividend response contains other ts_code")
                        kept.insert(0, "request_code", code)
                        raw_rows.append(kept)
            except Exception as exc:
                status, error = "API_ERROR", f"{type(exc).__name__}: {str(exc)[:220]}"
        log_rows.append({"stock_code": code, "ts_code": ts,
                         "status": status, "response_rows": response_count,
                         "removed_old_test_ex_date_rows": removed,
                         "error": error})
        if wait_seconds and status != "NO_TS_MAPPING":
            time.sleep(wait_seconds)
    raw = (pd.concat(raw_rows, ignore_index=True) if raw_rows else
           pd.DataFrame(columns=("request_code",) + FIELDS))
    ledger = pd.DataFrame(log_rows)
    if ledger.duplicated("stock_code").any():
        raise ValueError("Duplicate stock batch codes")
    out.mkdir(parents=True)
    raw.to_csv(out / "raw_dividends.csv", index=False, encoding="utf-8")
    ledger.to_csv(out / "request_ledger.csv", index=False, encoding="utf-8")
    manifest = {
        "batch_version": "P4_TUSHARE_DIVIDEND_BATCH_V1", "snapshot_sha256": meta["universe_sha256"],
        "test_start_EXCLUDED": meta["test_start_EXCLUDED"], "offset": offset,
        "limit_requested": limit, "requested_symbols": len(ledger),
        "success_symbols": int(ledger.status.eq("SUCCESS").sum()),
        "incomplete_symbols": int(ledger.status.ne("SUCCESS").sum()),
        "saved_candidate_vendor_rows": len(raw),
        "raw_sha256": sha256_file(out / "raw_dividends.csv"),
        "ledger_sha256": sha256_file(out / "request_ledger.csv"),
        "provider": "TUSHARE_DIVIDEND", "coverage_certified": False,
        "warning": WARNING,
    }
    (out / "batch.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def _to_float(value, field: str) -> float:
    if pd.isna(value) or str(value).strip().lower() in ("", "none", "nan", "nat"):
        raise ValueError(f"Missing {field}")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Nonfinite {field}")
    return number


def normalize_dividends(raw: pd.DataFrame, *, start: str, test_start: str
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only implemented, complete, non-conflicting rows become unverified candidates."""
    if not set(FIELDS) | {"request_code"} <= set(raw):
        raise ValueError("Raw Tushare export lacks required provider fields")
    begin, cutoff = _date(start, "start"), _date(test_start, "test_start")
    candidates, rejects = [], []
    for rownum, row in enumerate(raw.itertuples(index=False), 1):
        data = row._asdict()
        code = str(data["request_code"])
        reason = None
        try:
            _code(code)
            ts = _ts_code(data["ts_code"])
            if ts[:6] != code:
                raise ValueError("Provider stock code disagrees with request_code")
            if str(data["div_proc"]).strip() != "实施":
                reason = "NOT_IMPLEMENTED"
            else:
                ex = _ts_date(data["ex_date"], "ex_date")
                if ex >= cutoff:
                    reason = "OLD_TEST_EXCLUDED"
                elif ex < begin:
                    reason = "BEFORE_WINDOW"
                else:
                    published = _ts_date(data["imp_ann_date"], "imp_ann_date")
                    recorded = _ts_date(data["record_date"], "record_date")
                    if published > ex or recorded >= ex:
                        raise ValueError("Invalid implementation/record/ex date order")
                    bonus = _to_float(data["stk_bo_rate"], "stk_bo_rate")
                    capital = _to_float(data["stk_co_rate"], "stk_co_rate")
                    cash = _to_float(data["cash_div_tax"], "cash_div_tax")
                    if min(bonus, capital, cash) < 0:
                        raise ValueError("Negative action entitlement")
                    total = bonus + capital
                    if str(data["stk_div"]).strip().lower() not in ("", "nan", "none"):
                        if not np.isclose(_to_float(data["stk_div"], "stk_div"), total,
                                          atol=1e-7, rtol=0):
                            raise ValueError("stk_div disagrees with bonus+capital")
                    if total == 0 and cash == 0:
                        reason = "NOOP_IMPLEMENTATION"
                    else:
                        candidates.append({
                            "stock_code": code, "ex_date": ex.date().isoformat(),
                            "published_on": published.date().isoformat(),
                            "share_multiplier": 1 + total,
                            "cash_per_old_share": cash,
                            "source_url": SOURCE_URL,
                            "source_id": f"UNVERIFIED-TUSHARE:{ts}:{ex.date().isoformat()}",
                            "vendor": "TUSHARE_DIVIDEND", "review_status": "UNVERIFIED_VENDOR",
                            "record_date": recorded.date().isoformat(),
                            "ann_date_proposal": str(data["ann_date"]),
                        })
        except (ValueError, TypeError, KeyError) as exc:
            reason = f"INVALID_FIELDS:{str(exc)[:180]}"
        if reason:
            rejects.append({"request_code": code, "provider_row": rownum,
                            "ex_date_raw": str(data.get("ex_date", "")),
                            "reason": reason})
    events = pd.DataFrame(candidates, columns=EVENT_FIELDS + (
        "vendor", "review_status", "record_date", "ann_date_proposal"))
    problems = pd.DataFrame(rejects, columns=("request_code", "provider_row",
                                                   "ex_date_raw", "reason"))
    if events.empty:
        return events, problems
    # An identical repeated row may reflect same distribution phase re-sent.
    # Conflicting terms/dates must be quarantined in FULL; no arbitrary winner.
    retained = []
    for (code, ex), group in events.groupby(["stock_code", "ex_date"], sort=True):
        terms = group.loc[:, ["published_on", "share_multiplier", "cash_per_old_share",
                              "record_date"]].drop_duplicates()
        if len(terms) > 1:
            for _ in range(len(group)):
                problems.loc[len(problems)] = [code, -1, ex, "CONFLICTING_SAME_EX_DATE"]
        else:
            retained.append(group.iloc[0])
    clean = pd.DataFrame(retained, columns=events.columns).reset_index(drop=True)
    return clean, problems


def audit_batches(snapshot: Path, batch_root: Path, *,
                  official_events: Path | None = None
                  ) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = json.loads((snapshot / "snapshot.json").read_text(encoding="utf-8"))
    if sha256_file(snapshot / "universe.csv") != meta["universe_sha256"]:
        raise ValueError("Universe snapshot checksum mismatch")
    universe = _read_tushare_csv(snapshot / "universe.csv")
    ledgers, raws = [], []
    files = sorted(batch_root.glob("batch_*/batch.json"))
    if not files:
        raise ValueError("No batch_*/batch.json found")
    for file in files:
        manifest = json.loads(file.read_text(encoding="utf-8"))
        if (manifest.get("snapshot_sha256") != meta["universe_sha256"] or
                manifest.get("test_start_EXCLUDED") != meta["test_start_EXCLUDED"]):
            raise ValueError("Batch belongs to another universe/cutoff")
        folder = file.parent
        if (sha256_file(folder / "request_ledger.csv") != manifest["ledger_sha256"] or
                sha256_file(folder / "raw_dividends.csv") != manifest["raw_sha256"]):
            raise ValueError(f"Modified provider batch: {folder}")
        ledgers.append(_read_tushare_csv(folder / "request_ledger.csv"))
        raws.append(_read_tushare_csv(folder / "raw_dividends.csv"))
    ledger = pd.concat(ledgers, ignore_index=True)
    if ledger.duplicated("stock_code").any():
        raise ValueError("Multiple batch claims for same stock; resolve explicit retry")
    if not ledger.stock_code.isin(universe.stock_code).all():
        raise ValueError("Provider batch includes stock outside snapshot")
    raw = pd.concat(raws, ignore_index=True)
    if len(raw) and not raw.request_code.isin(ledger.loc[ledger.status.eq("SUCCESS"), "stock_code"]).all():
        raise ValueError("Raw rows not backed by successful request ledger")
    # Distinguish an absent fetch, a successful empty response, and a verified
    # *complete* event history. None of them imply COMPLETE_VERIFIED here.
    events, rejected = normalize_dividends(raw, start=meta["start"],
                                             test_start=meta["test_start_EXCLUDED"])
    by_stock = universe.merge(ledger, how="left", on="stock_code", validate="one_to_one",
                              suffixes=("", "_request"))
    event_counts = events.groupby("stock_code").size().rename("candidate_events") if len(events) else pd.Series(dtype=int, name="candidate_events")
    by_stock = by_stock.merge(event_counts, how="left", left_on="stock_code", right_index=True)
    by_stock["candidate_events"] = by_stock.candidate_events.fillna(0).astype(int)
    by_stock["coverage_status"] = np.select(
        [by_stock.status.isna().to_numpy(dtype=bool),
         by_stock.status.eq("SUCCESS").fillna(False).to_numpy(dtype=bool)],
        ["NOT_FETCHED", "FETCHED_NOT_VERIFIED"], default="FETCH_FAILED_OR_TRUNCATED")
    by_stock["complete_verified"] = False
    reference_mismatch = []
    if official_events is not None:
        known = _read_tushare_csv(official_events)
        required = {"stock_code", "ex_date", "share_multiplier", "cash_per_old_share"}
        if not required <= set(known):
            raise ValueError("Official controls missing fields")
        begin = _date(meta["start"], "start")
        cut = _date(meta["test_start_EXCLUDED"], "cutoff")
        for r in known.itertuples(index=False):
            day = _date(str(r.ex_date), "known ex_date")
            if not begin <= day < cut:
                continue
            # Treat missing, absent fetch, malformed or mismatched control as
            # unresolved; never auto-reconcile an issuer source to vendor data.
            matched = events.loc[(events.stock_code.eq(str(r.stock_code))) &
                                 (events.ex_date.eq(str(r.ex_date)))]
            same = (len(matched) == 1 and
                    np.isclose(float(matched.iloc[0].share_multiplier), float(r.share_multiplier), atol=1e-6) and
                    np.isclose(float(matched.iloc[0].cash_per_old_share), float(r.cash_per_old_share), atol=1e-6))
            reference_mismatch.append({"stock_code": r.stock_code, "ex_date": str(r.ex_date),
                                        "source_id": r.source_id, "status": "MATCH_VENDOR_CANDIDATE" if same else "MISSING_OR_CONFLICTING_VENDOR_CANDIDATE"})
    controls = pd.DataFrame(reference_mismatch, columns=("stock_code", "ex_date", "source_id", "status"))
    result = {
        "audit_version": "P4_VENDOR_ACTION_COVERAGE_AUDIT_V1",
        "snapshot_sha256": meta["universe_sha256"],
        "test_start_EXCLUDED": meta["test_start_EXCLUDED"],
        "expected_symbols": len(universe),
        "provider_only_symbols": int((~universe.present_in_formal_market.astype("string").str.lower().eq("true")).sum()),
        "market_not_mapped_to_provider": int(universe.ts_code.fillna("").eq("").sum()),
        "fetched_success": int(by_stock.status.eq("SUCCESS").sum()),
        "fetched_failure_or_truncated": int(by_stock.status.isin(["API_ERROR", "ROW_LIMIT", "NO_TS_MAPPING"]).sum()),
        "not_fetched": int(by_stock.status.isna().sum()),
        "vendor_raw_rows": len(raw),
        "normalized_vendor_candidate_events": len(events),
        "rejected_or_unimplemented_rows": len(rejected),
        "known_issuer_reference_events_checked": len(controls),
        "known_issuer_reference_mismatches": int(controls.status.ne("MATCH_VENDOR_CANDIDATE").sum()),
        "verified_complete_coverage_assertions_created": 0,
        "full_market_actions_certified": False,
        "rights_issue_coverage_certified": False,
        "training_gate_must_remain_closed": True,
        "warning": WARNING,
    }
    return result, events, by_stock, rejected, controls


def _make_pro():
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        raise RuntimeError("Set TUSHARE_TOKEN in Ubuntu environment; never commit credentials")
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install tushare") from exc
    return ts.pro_api(token)


def main() -> None:
    parser = argparse.ArgumentParser(description="P4.3-E2 provider ingestion and UNVERIFIED coverage reporting")
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser("snapshot", help="build a fixed market+Tushare universe snapshot")
    snapshot.add_argument("--reference-metadata", type=Path, required=True)
    snapshot.add_argument("--start", default="2025-01-01")
    snapshot.add_argument("--market-codes-csv", type=Path, help="offline stock_code CSV; else SELECT formal DB")
    snapshot.add_argument("--stock-basic-csv", type=Path, help="offline Tushare basic export; else fetch L/D/P")
    snapshot.add_argument("--out", type=Path, required=True)
    fetch = sub.add_parser("fetch", help="bounded Tushare dividend fetch with immutable batch outputs")
    fetch.add_argument("--snapshot", type=Path, required=True)
    fetch.add_argument("--reference-metadata", type=Path, required=True)
    fetch.add_argument("--offset", type=int, default=0)
    fetch.add_argument("--limit", type=int, default=50)
    fetch.add_argument("--wait-seconds", type=float, default=1.5,
                       help="conservative delay between provider requests; adapt to quota")
    fetch.add_argument("--out", type=Path, required=True)
    audit = sub.add_parser("audit", help="normalize vendor events and audit stock coverage")
    audit.add_argument("--snapshot", type=Path, required=True)
    audit.add_argument("--reference-metadata", type=Path, required=True)
    audit.add_argument("--batch-root", type=Path, required=True)
    audit.add_argument("--official-controls", type=Path,
                       default=Path("ashare/data/corporate_actions_verified_examples.csv"))
    audit.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        _, cut = _ensure_cutoff(args.reference_metadata)
        if args.market_codes_csv is None:
            print("[snapshot] SELECT only formal market stock codes; zero DB writes", flush=True)
        market = (_read_tushare_csv(args.market_codes_csv) if args.market_codes_csv
                  else load_formal_market_codes(args.start, str(cut.date())))
        pro = None if args.stock_basic_csv else _make_pro()
        basic = (_read_tushare_csv(args.stock_basic_csv) if args.stock_basic_csv
                 else fetch_stock_basic(pro))
        result = create_snapshot(market=market, basics=basic, start=args.start,
                                 test_start=str(cut.date()), output=args.out)
    elif args.command == "fetch":
        _, cutoff = _ensure_cutoff(args.reference_metadata)
        snap = json.loads((args.snapshot / "snapshot.json").read_text(encoding="utf-8"))
        if snap.get("test_start_EXCLUDED") != str(cutoff.date()):
            raise ValueError("Snapshot cutoff differs from archived Ridge metadata")
        result = fetch_batch(args.snapshot, pro=_make_pro(), offset=args.offset,
                             limit=args.limit, out=args.out, wait_seconds=args.wait_seconds)
    else:
        _, cutoff = _ensure_cutoff(args.reference_metadata)
        snap = json.loads((args.snapshot / "snapshot.json").read_text(encoding="utf-8"))
        if snap.get("test_start_EXCLUDED") != str(cutoff.date()):
            raise ValueError("Snapshot cutoff differs from archived Ridge metadata")
        if args.out.exists():
            raise FileExistsError("Audit output already exists; keep old audits immutable")
        result, events, stocks, rejected, controls = audit_batches(
            args.snapshot, args.batch_root, official_events=args.official_controls)
        args.out.mkdir(parents=True)
        (args.out / "feed_coverage_audit.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        events.to_csv(args.out / "UNVERIFIED_corporate_actions_candidates.csv",
                      index=False, encoding="utf-8-sig")
        stocks.to_csv(args.out / "coverage_by_stock.csv", index=False, encoding="utf-8-sig")
        rejected.to_csv(args.out / "rejected_vendor_rows.csv", index=False, encoding="utf-8-sig")
        controls.to_csv(args.out / "issuer_control_reconciliation.csv",
                        index=False, encoding="utf-8-sig")
        (args.out / "COMPLETE_VERIFIED_NOT_ISSUED.txt").write_text(
            "No stock/date interval can be certified from dividend API downloads alone.\n" +
            WARNING + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()