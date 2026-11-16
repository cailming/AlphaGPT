"""P4.3-E/2: no-token AkShare candidate corporate-action feed.

Read-only formal-market stock universe, bounded AkShare Sina+CNINFO+rights
collection and fail-closed comparison. All results are UNVERIFIED evidence;
no COMPLETE_VERIFIED coverage assertion, model promotion or test-period label.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

SINA_URL = "https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/"
CNINFO_URL = "http://webapi.cninfo.com.cn/"
ENDPOINTS = ("sina_dividend", "cninfo_dividend", "sina_rights")
SINA_COLS = ("公告日期", "送股", "转增", "派息", "进度", "除权除息日", "股权登记日")
CNINFO_COLS = ("实施方案公告日期", "送股比例", "转增比例", "派息比例", "除权日", "股权登记日")
RIGHTS_COLS = ("公告日期", "配股方案", "配股价格", "除权日", "股权登记日")
NOTE = (
    "AKSHARE SINA/CNINFO FREE PUBLIC DATA ONLY. Upstream may be unstable, "
    "licensed or incomplete; a successful EMPTY response is NOT proof that "
    "a stock had no corporate actions. Per-ten-share units are normalized "
    "but must be checked against issuer implementation announcements. "
    "Rights issues are recorded as unresolved hazards, NOT converted into "
    "cash/bonus factors. No COMPLETE_VERIFIED coverage certificates, approved "
    "training labels, or old test-period examples are produced."
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code6(s: object) -> str:
    value = str(s).strip()
    if not re.fullmatch(r"\d{6}", value):
        raise ValueError(f"Invalid six-digit stock code: {value!r}")
    return value


def valid_day(value, *, nullable: bool = False) -> pd.Timestamp:
    if value is None or pd.isna(value) or str(value).strip().lower() in ("", "none", "nan", "nat", "--"):
        if nullable:
            return pd.NaT
        raise ValueError("Missing required date")
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        raise ValueError(f"Invalid date {value!r}")
    return pd.Timestamp(date).normalize()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype="string", keep_default_na=False)


def _as_bool(value) -> bool:
    return str(value).strip().lower() == "true"


def save_snapshot(codes: pd.DataFrame, *, start: str, test_start: str,
                  out: Path) -> dict:
    """The universe comes from the user's formal DB, NOT today's live list."""
    begin, cut = valid_day(start), valid_day(test_start)
    if begin >= cut or "stock_code" not in codes:
        raise ValueError("Invalid snapshot period or missing stock_code")
    unique = sorted(set(code6(x) for x in codes.stock_code))
    if not unique or out.exists():
        raise ValueError("Empty universe or immutable snapshot already exists")
    out.mkdir(parents=True)
    pd.DataFrame({"stock_code": unique, "scope": "FORMAL_MARKET_DB_ONLY"}).to_csv(
        out / "universe.csv", index=False, encoding="utf-8")
    result = {
        "snapshot_version": "P4_AKSHARE_FORMAL_UNIVERSE_V1",
        "start": str(begin.date()), "test_start_EXCLUDED": str(cut.date()),
        "symbols": len(unique), "universe_sha256": digest(out / "universe.csv"),
        "historic_delisted_from_formal_market_only": True,
        "full_exchange_listed_universe_certified": False,
        "coverage_certified": False, "warning": NOTE,
    }
    (out / "snapshot.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _snapshot(folder: Path, cutoff: str | None = None) -> tuple[dict, pd.DataFrame]:
    meta = json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
    if meta.get("snapshot_version") != "P4_AKSHARE_FORMAL_UNIVERSE_V1":
        raise ValueError("Not a token-free AkShare universe snapshot")
    if cutoff is not None and meta["test_start_EXCLUDED"] != cutoff:
        raise ValueError("Archived test boundary disagrees with snapshot")
    if digest(folder / "universe.csv") != meta["universe_sha256"]:
        raise ValueError("Snapshot contents altered")
    universe = _read(folder / "universe.csv")
    if len(universe) != meta["symbols"] or universe.stock_code.duplicated().any():
        raise ValueError("Snapshot row count or stock codes changed")
    return meta, universe


def _import_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("Run python -m pip install akshare (free; no Token)") from exc
    return ak


def _endpoint(provider, code: str, endpoint: str) -> pd.DataFrame:
    if endpoint == "sina_dividend":
        return provider.stock_history_dividend_detail(symbol=code, indicator="分红")
    if endpoint == "sina_rights":
        return provider.stock_history_dividend_detail(symbol=code, indicator="配股")
    if endpoint == "cninfo_dividend":
        # Modern AKShare exposes stock_dividend_cninfo (singular). Some older
        # wrappers used the historic misspelling; prefer the canonical API.
        method = getattr(provider, "stock_dividend_cninfo", None)
        if method is None:
            method = getattr(provider, "stock_dividents_cninfo", None)
        if method is None:
            raise AttributeError("AkShare lacks stock_dividend_cninfo; upgrade akshare")
        return method(symbol=code)
    raise ValueError(f"Unsupported endpoint: {endpoint}")


def _cols(endpoint: str) -> tuple[str, ...]:
    return {"sina_dividend": SINA_COLS, "cninfo_dividend": CNINFO_COLS,
            "sina_rights": RIGHTS_COLS}[endpoint]


def _prune(table: pd.DataFrame, endpoint: str, cutoff: str) -> tuple[pd.DataFrame, int]:
    """Drop rows whose ex or implementation/publication dates are unknown/future.

    Upstream APIs may return future records; discard before any raw files are
    persisted, so old-test records cannot enter the audit artifacts.
    """
    required = _cols(endpoint)
    if not isinstance(table, pd.DataFrame):
        raise ValueError("Upstream returned non-DataFrame response")
    if table.empty:
        return pd.DataFrame(columns=(["request_code"] if "request_code" in table.columns else []) + list(required)), 0
    if not set(required).issubset(table.columns):
        raise ValueError(f"API schema drift in {endpoint}: {set(required)-set(table)}")
    keepcols = (["request_code"] if "request_code" in table.columns else []) + list(required)
    d = table.loc[:, keepcols].copy()
    exfield, annfield = {
        "sina_dividend": ("除权除息日", "公告日期"),
        "cninfo_dividend": ("除权日", "实施方案公告日期"),
        "sina_rights": ("除权日", "公告日期"),
    }[endpoint]
    cut = valid_day(cutoff)
    keep = []
    for row in d.to_dict("records"):
        try:
            ex, published = valid_day(row[exfield]), valid_day(row[annfield])
            keep.append(ex < cut and published < cut)
        except (ValueError, TypeError):
            keep.append(False)
    return d.loc[keep].copy(), len(d) - sum(keep)


def fetch_batch(snapshot: Path, *, ak, offset: int, limit: int,
                out: Path, wait_seconds: float = 1.0, row_cap: int = 1000) -> dict:
    if offset < 0 or limit < 1 or wait_seconds < 0 or row_cap < 1:
        raise ValueError("Invalid batch settings")
    meta, universe = _snapshot(snapshot)
    chosen = universe.iloc[offset:offset+limit]
    if chosen.empty or out.exists():
        raise ValueError("Empty offset range or immutable batch output already exists")
    raws = {endpoint: [] for endpoint in ENDPOINTS}
    ledger = []
    for code in chosen.stock_code:
        for endpoint in ENDPOINTS:
            status, error, rows, dropped = "SUCCESS", "", 0, 0
            try:
                frame = _endpoint(ak, str(code), endpoint)
                if not isinstance(frame, pd.DataFrame):
                    raise ValueError("API result is not DataFrame")
                rows = len(frame)
                if rows >= row_cap:
                    status = "ROW_LIMIT"
                else:
                    kept, dropped = _prune(frame, endpoint, meta["test_start_EXCLUDED"])
                    if not kept.empty:
                        kept.insert(0, "request_code", str(code))
                        raws[endpoint].append(kept)
            except Exception as exc:
                # AKShare's CNINFO parser directly indexes the announcement
                # column after DataFrame(records). A KeyError may mean either
                # records=[] OR a provider schema change. We cannot distinguish
                # these from the parser exception alone: never call it SUCCESS
                # and NEVER infer that this stock had no corporate actions.
                if (endpoint == "cninfo_dividend" and isinstance(exc, KeyError)
                        and "实施方案公告日期" in str(exc)):
                    status = "UPSTREAM_EMPTY_OR_SCHEMA"
                    error = ("CNINFO parser could not locate 实施方案公告日期; "
                             "upstream may have zero records or schema drift; "
                             "not a verified empty action history")
                else:
                    status = "API_ERROR"
                    error = f"{type(exc).__name__}: {str(exc)[:200]}"
            ledger.append({"stock_code": code, "endpoint": endpoint,
                           "status": status, "response_rows": rows,
                           "dropped_future_or_invalid": dropped, "error": error})
            if wait_seconds:
                time.sleep(wait_seconds)
    if len(ledger) != len(chosen)*len(ENDPOINTS):
        raise AssertionError("Incomplete request ledger")
    out.mkdir(parents=True)
    table = pd.DataFrame(ledger)
    table.to_csv(out / "request_ledger.csv", index=False, encoding="utf-8")
    checksums = {"request_ledger.csv": digest(out / "request_ledger.csv")}
    for endpoint in ENDPOINTS:
        filename = f"raw_{endpoint}.csv"
        frames = raws[endpoint]
        data = (pd.concat(frames, ignore_index=True) if frames else
                pd.DataFrame(columns=("request_code",) + _cols(endpoint)))
        data.to_csv(out / filename, index=False, encoding="utf-8")
        checksums[filename] = digest(out / filename)
    result = {
        "batch_version": "P4_AKSHARE_NO_TOKEN_BATCH_V1",
        "snapshot_sha256": meta["universe_sha256"],
        "test_start_EXCLUDED": meta["test_start_EXCLUDED"],
        "offset": offset, "requested_symbols": len(chosen),
        "api_calls": len(ledger), "successful_calls": int(table.status.eq("SUCCESS").sum()),
        "failed_or_truncated_calls": int(table.status.ne("SUCCESS").sum()),
        "files_sha256": checksums, "provider": "AKSHARE_PUBLIC_SINA_AND_CNINFO",
        "coverage_certified": False, "warning": NOTE,
    }
    (out / "batch.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _number(value) -> float:
    if pd.isna(value) or str(value).strip().lower() in ("", "nan", "none", "nat", "--", "-"):
        return 0.0
    num = float(str(value).replace(",", ""))
    if not np.isfinite(num) or num < 0:
        raise ValueError("Invalid entitlement amount")
    return num


def _normalize(raw: pd.DataFrame, endpoint: str, *, start: str, cutoff: str
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean, rejected = [], []
    lo, cut = valid_day(start), valid_day(cutoff)
    for row in raw.to_dict("records"):
        code = str(row["request_code"])
        try:
            code6(code)
            if endpoint == "sina_dividend":
                if str(row["进度"]).strip() != "实施":
                    continue
                ex = valid_day(row["除权除息日"])
                # Old out-of-window rows remain archived raw evidence. Their
                # inconsistent 1990s dates are not current-period audit issues.
                if not lo <= ex < cut:
                    continue
                ann = valid_day(row["公告日期"])
                reg = valid_day(row["股权登记日"])
                gift, reserve, cash = (_number(row[k])/10 for k in ("送股", "转增", "派息"))
                url = SINA_URL
            elif endpoint == "cninfo_dividend":
                ex = valid_day(row["除权日"])
                if not lo <= ex < cut:
                    continue
                ann = valid_day(row["实施方案公告日期"])
                reg = valid_day(row["股权登记日"])
                gift, reserve, cash = (_number(row[k])/10 for k in ("送股比例", "转增比例", "派息比例"))
                url = CNINFO_URL
            else:
                raise ValueError("Unsupported normalized endpoint")
            if not (ann <= reg < ex):
                raise ValueError("Publication/record/ex-date order invalid")
            if gift + reserve + cash == 0:
                continue
            clean.append({"stock_code": code, "ex_date": str(ex.date()),
                          "published_on": str(ann.date()),
                          "share_multiplier": 1 + gift + reserve,
                          "cash_per_old_share": cash, "record_date": str(reg.date()),
                          "provider": endpoint, "provider_url": url})
        except (KeyError, ValueError, TypeError) as exc:
            rejected.append({"stock_code": code, "endpoint": endpoint,
                             "reason": f"INVALID:{str(exc)[:140]}"})
    return pd.DataFrame(clean, columns=["stock_code", "ex_date", "published_on",
                                            "share_multiplier", "cash_per_old_share",
                                            "record_date", "provider", "provider_url"]), pd.DataFrame(
                                                rejected, columns=["stock_code", "endpoint", "reason"])


def audit_batches(snapshot: Path, batch_root: Path, *, official_events: Path | None = None
                 ) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta, universe = _snapshot(snapshot)
    manifests = sorted(batch_root.glob("batch_*/batch.json"))
    if not manifests:
        raise ValueError("No batch_*/batch.json folders")
    ledgers, raw = [], {endpoint: [] for endpoint in ENDPOINTS}
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("batch_version") != "P4_AKSHARE_NO_TOKEN_BATCH_V1" or
                manifest.get("snapshot_sha256") != meta["universe_sha256"] or
                manifest.get("test_start_EXCLUDED") != meta["test_start_EXCLUDED"]):
            raise ValueError("Batch does not match this snapshot")
        for filename, checksum in manifest["files_sha256"].items():
            if digest(path.parent / filename) != checksum:
                raise ValueError(f"Modified batch file: {filename}")
        ledgers.append(_read(path.parent / "request_ledger.csv"))
        for endpoint in ENDPOINTS:
            raw[endpoint].append(_read(path.parent / f"raw_{endpoint}.csv"))
    ledger = pd.concat(ledgers, ignore_index=True)
    if ledger.duplicated(["stock_code", "endpoint"]).any():
        raise ValueError("Duplicate request/endpoint across batches; resolve retries first")
    if not ledger.stock_code.isin(universe.stock_code).all() or not ledger.endpoint.isin(ENDPOINTS).all():
        raise ValueError("Unexpected code/endpoint in ledger")
    allraw = {key:pd.concat(value, ignore_index=True) for key, value in raw.items()}
    for endpoint, table in allraw.items():
        successful = ledger.loc[(ledger.endpoint.eq(endpoint)) & (ledger.status.eq("SUCCESS")), "stock_code"]
        if not table.request_code.isin(successful).all():
            raise ValueError("Raw data without matching successful provider call")
        # Recheck historical boundary after ingest, independent of fetch filtering.
        kept, removed = _prune(table, endpoint, meta["test_start_EXCLUDED"])
        if removed:
            raise ValueError("Old test/future or invalid-date rows found in persisted batch")
        allraw[endpoint] = kept
    normalized = []
    rejected = []
    for endpoint in ("sina_dividend", "cninfo_dividend"):
        found, bad = _normalize(allraw[endpoint], endpoint, start=meta["start"],
                                cutoff=meta["test_start_EXCLUDED"])
        normalized.append(found)
        rejected.append(bad)
    both = pd.concat([frame for frame in normalized if not frame.empty], ignore_index=True) if any(not frame.empty for frame in normalized) else pd.DataFrame(columns=normalized[0].columns)
    rejects = pd.concat(rejected, ignore_index=True)
    candidates, conflicts = [], []
    for (code, ex), group in both.groupby(["stock_code", "ex_date"]):
        per_source = []
        invalid = False
        for vendor, subset in group.groupby("provider"):
            values = subset[["share_multiplier", "cash_per_old_share", "record_date"]].drop_duplicates()
            if len(values) != 1:
                conflicts.append({"stock_code": code, "ex_date": ex,
                                  "reason": f"CONFLICTING_{vendor}"})
                invalid = True
            else:
                per_source.append(subset.iloc[0])
        if invalid:
            continue
        if len(per_source) == 2:
            first, second = per_source
            if (not np.isclose(float(first.share_multiplier), float(second.share_multiplier), atol=1e-6, rtol=0) or
                    not np.isclose(float(first.cash_per_old_share), float(second.cash_per_old_share), atol=1e-6, rtol=0) or
                    first.record_date != second.record_date):
                conflicts.append({"stock_code": code, "ex_date": ex,
                                  "reason": "SINA_CNINFO_TERMS_MISMATCH"})
                continue
            status = "TWO_PUBLIC_SOURCES_MATCH_UNVERIFIED"
        else:
            status = f"SINGLE_SOURCE_{per_source[0].provider.upper()}_UNVERIFIED"
        chosen = per_source[0]
        candidates.append({"stock_code":code, "ex_date":ex,
                           "published_on":max(x.published_on for x in per_source),
                           "share_multiplier":float(chosen.share_multiplier),
                           "cash_per_old_share":float(chosen.cash_per_old_share),
                           "source_url":CNINFO_URL if len(per_source)==2 else chosen.provider_url,
                           "source_id":f"UNVERIFIED-AKSHARE:{code}:{ex}",
                           "review_status":status,"sources_seen":"|".join(sorted(x.provider for x in per_source))})
    events = pd.DataFrame(candidates, columns=["stock_code","ex_date","published_on",
        "share_multiplier","cash_per_old_share","source_url","source_id","review_status","sources_seen"])
    issues = pd.concat([rejects, pd.DataFrame([{"stock_code":item["stock_code"],
          "endpoint":"cross_source", "reason":item["reason"]} for item in conflicts])], ignore_index=True)
    if issues.empty:
        issues = pd.DataFrame(columns=["stock_code","endpoint","reason"])
    # A historical rights endpoint returns *all* past rights issues. Count
    # relevant hazards only within the archived pre-test audit interval; keep
    # older raw evidence in its immutable batch, not the active hazard count.
    rights_history = allraw["sina_rights"].copy()
    rights_dates = pd.to_datetime(rights_history["除权日"], errors="coerce")
    rights_relevant = rights_dates.ge(valid_day(meta["start"])) & rights_dates.lt(
        valid_day(meta["test_start_EXCLUDED"])
    )
    rights_rows = rights_history.loc[rights_relevant].copy()
    rights_rows["review_status"] = "RIGHTS_SUBSCRIPTION_UNSUPPORTED_FOR_ECONOMIC_LABEL"
    endpoints = ledger.pivot(index="stock_code", columns="endpoint", values="status")
    stock = universe.merge(endpoints.reset_index(), on="stock_code", how="left",validate="one_to_one")
    stock["coverage_status"] = np.select(
        [stock[list(ENDPOINTS)].isna().any(axis=1),
         stock[list(ENDPOINTS)].ne("SUCCESS").any(axis=1)],
        ["NOT_FETCHED","FETCH_FAILED_OR_TRUNCATED"], default="FETCHED_NOT_VERIFIED")
    stock["candidate_events"] = stock.stock_code.map(events.groupby("stock_code").size()).fillna(0).astype(int)
    stock["rights_records"] = stock.stock_code.map(rights_rows.groupby("request_code").size()).fillna(0).astype(int)
    stock["complete_verified"] = False
    controls = []
    if official_events is not None:
        known = _read(official_events)
        for row in known.to_dict("records"):
            day = valid_day(row["ex_date"])
            if not (valid_day(meta["start"]) <= day < valid_day(meta["test_start_EXCLUDED"])):
                continue
            match = events.loc[events.stock_code.eq(str(row["stock_code"])) & events.ex_date.eq(str(day.date()))]
            # Reference issuer controls may be outside the chosen *first*
            # batch. They are NOT mismatches until that stock's two dividend
            # endpoints were actually queried successfully. Distinguish not
            # fetched, partially fetched and genuine conflicting evidence.
            symbol = str(row["stock_code"])
            queried = stock.loc[stock.stock_code.eq(symbol)]
            if queried.empty or queried[list(ENDPOINTS)].isna().all(axis=1).iloc[0]:
                disposition = "NOT_FETCHED_FOR_CONTROL"
            elif not (queried.iloc[0]["sina_dividend"] == "SUCCESS" and
                      queried.iloc[0]["cninfo_dividend"] == "SUCCESS"):
                disposition = "INCOMPLETE_FETCH_FOR_CONTROL"
            else:
                good = (len(match)==1 and
                        np.isclose(float(match.iloc[0].share_multiplier), float(row["share_multiplier"]), rtol=0, atol=1e-6) and
                        np.isclose(float(match.iloc[0].cash_per_old_share), float(row["cash_per_old_share"]), rtol=0, atol=1e-6))
                disposition = "MATCHED_ISSUER_CONTROL" if good else "MISSING_OR_CONFLICTING"
            controls.append({"stock_code":symbol, "ex_date":str(day.date()),
                             "reference_id":row.get("source_id",""),
                             "status":disposition})
    official = pd.DataFrame(controls, columns=["stock_code","ex_date","reference_id","status"])
    # Retain the per-endpoint failure distribution in the report so a 20/30
    # overall success figure cannot conceal one systematically blocked API.
    endpoint_counts = {
        endpoint: {str(status): int((ledger.loc[ledger.endpoint.eq(endpoint), "status"] == status).sum())
                   for status in sorted(ledger.status.unique())}
        for endpoint in ENDPOINTS
    }
    endpoint_error_examples = {
        endpoint: ledger.loc[ledger.endpoint.eq(endpoint) & ledger.status.ne("SUCCESS"),
                             "error"].dropna().astype(str).head(2).tolist()
        for endpoint in ENDPOINTS
    }
    report = {"audit_version":"P4_AKSHARE_NO_TOKEN_COVERAGE_V1",
        "snapshot_sha256":meta["universe_sha256"],
        "test_start_EXCLUDED":meta["test_start_EXCLUDED"],
        "expected_stocks":len(stock),
        "not_fetched":int(stock.coverage_status.eq("NOT_FETCHED").sum()),
        "failed_or_truncated":int(stock.coverage_status.eq("FETCH_FAILED_OR_TRUNCATED").sum()),
        "fetched_not_verified":int(stock.coverage_status.eq("FETCHED_NOT_VERIFIED").sum()),
        "unverified_normalized_candidate_events":len(events),
        "cross_source_conflicts_or_invalid_rows":len(issues),
        "rights_records_requiring_manual_review":len(rights_rows),
        "rights_records_raw_all_history":len(rights_history),
        "rights_records_with_invalid_or_missing_ex_date":int(rights_dates.isna().sum()),
        "endpoint_status_counts":endpoint_counts,
        "endpoint_results":endpoint_counts,
        "endpoint_failure_examples":endpoint_error_examples,
        "endpoint_failure_stocks": {
            ep: [
                {"stock_code": str(rec.stock_code), "status": str(rec.status),
                 "error": str(rec.error)[:200]}
                for rec in ledger.loc[
                    ledger.endpoint.eq(ep) & ledger.status.ne("SUCCESS"),
                    ["stock_code", "status", "error"]
                ].head(5).itertuples(index=False)
            ]
            for ep in ENDPOINTS
        },
        "cninfo_upstream_ambiguous_calls": int(
            (ledger.endpoint.eq("cninfo_dividend") &
             ledger.status.eq("UPSTREAM_EMPTY_OR_SCHEMA")).sum()
        ),
        "issuer_controls_total_in_reference":len(official),
        "issuer_controls_checked":int(official.status.isin(
            ["MATCHED_ISSUER_CONTROL", "MISSING_OR_CONFLICTING"]).sum()),
        "issuer_controls_not_fetched":int(official.status.eq("NOT_FETCHED_FOR_CONTROL").sum()),
        "issuer_controls_sources_incomplete":int(official.status.eq("INCOMPLETE_FETCH_FOR_CONTROL").sum()),
        "issuer_controls_mismatch":int(official.status.eq("MISSING_OR_CONFLICTING").sum()),
        "complete_verified_assertions_created":0,
        "full_historical_action_coverage_certified":False,
        "training_gate_must_remain_closed":True,
        "warning":NOTE}
    return report, events, stock, issues, rights_rows, official


def _model_cut(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("model_version") != "P4_RIDGE_V1":
        raise ValueError("Expected P4_RIDGE_V1 metadata")
    return str(valid_day(data["test_start"]).date())


def main() -> None:
    p = argparse.ArgumentParser(description="No-token free-data P4 action candidates (NOT certified)")
    sub = p.add_subparsers(dest="stage",required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--start",default="2024-12-01")
    snap.add_argument("--reference-metadata",type=Path,required=True)
    snap.add_argument("--market-codes-csv",type=Path)
    snap.add_argument("--out",type=Path,required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--snapshot",type=Path,required=True)
    fetch.add_argument("--reference-metadata",type=Path,required=True)
    fetch.add_argument("--offset",type=int,default=0)
    fetch.add_argument("--limit",type=int,default=20)
    fetch.add_argument("--wait-seconds",type=float,default=1.0)
    fetch.add_argument("--out",type=Path,required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--snapshot",type=Path,required=True)
    audit.add_argument("--reference-metadata",type=Path,required=True)
    audit.add_argument("--batch-root",type=Path,required=True)
    audit.add_argument("--official-controls",type=Path,
                      default=Path("ashare/data/corporate_actions_verified_examples.csv"))
    audit.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    cutoff=_model_cut(args.reference_metadata)
    if args.stage=="snapshot":
        if args.market_codes_csv:
            market=_read(args.market_codes_csv)
        else:
            from .p4_action_feed import load_formal_market_codes
            print("[snapshot] public.market SELECT distinct stock_code ONLY",flush=True)
            market=load_formal_market_codes(args.start,cutoff)
        result=save_snapshot(market,start=args.start,test_start=cutoff,out=args.out)
    elif args.stage=="fetch":
        _snapshot(args.snapshot,cutoff)
        result=fetch_batch(args.snapshot,ak=_import_akshare(),offset=args.offset,
                          limit=args.limit,out=args.out,wait_seconds=args.wait_seconds)
    else:
        _snapshot(args.snapshot,cutoff)
        if args.out.exists():
            raise FileExistsError("Immutable audit path exists; choose another")
        result,events,stocks,issues,rights,controls=audit_batches(
            args.snapshot,args.batch_root,official_events=args.official_controls)
        args.out.mkdir(parents=True)
        for name,frame in (("UNVERIFIED_corporate_actions_candidates.csv",events),
                           ("coverage_by_stock.csv",stocks),
                           ("audit_issues.csv",issues),
                           ("rights_issues_for_review.csv",rights),
                           ("issuer_control_reconciliation.csv",controls)):
            frame.to_csv(args.out/name,index=False,encoding="utf-8-sig")
        (args.out/"feed_coverage_audit.json").write_text(
            json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        (args.out/"COMPLETE_VERIFIED_NOT_ISSUED.txt").write_text(NOTE+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
