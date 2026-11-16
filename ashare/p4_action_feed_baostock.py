"""P4.3-E/2 independent BaoStock dividend corroboration (NO token).

Immutably collect *pre-old-test* annual implemented-dividend records and audit
against existing AKShare candidate data. A successful empty result is NEVER a
completeness certificate. No market DB writes, future labels, model training,
source bypass or rights-issue economic pricing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import pandas as pd
import numpy as np

SOURCE = "https://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5"
VERSION = "P4_BAOSTOCK_UNVERIFIED_DIVIDEND_V1"
RAW_FIELDS = (
    "code", "dividPlanDate", "dividRegistDate", "dividOperateDate",
    "dividCashPsBeforeTax", "dividStocksPs", "dividReserveToStockPs",
)
CAUTION = (
    "Third-party free historical dividends: EMPTY or successful retrieval does NOT "
    "prove complete historical corporate actions. BaoStock is a corroborating "
    "candidate source, not an issuer implementation announcement or a verified "
    "coverage feed. Rights issues, errors, stale revisions and empty results "
    "remain unresolved; never issue COMPLETE_VERIFIED or enable trading models. "
    "Pre-existing inspected test period is excluded."
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code(value: str) -> str:
    x = str(value).strip()
    if not re.fullmatch(r"\d{6}", x):
        raise ValueError(f"Bad stock code {x!r}")
    return x


def _provider_code(code: str) -> str | None:
    code = _code(code)
    if code.startswith(("0", "3")):
        return "sz." + code
    if code.startswith("6"):
        return "sh." + code
    # Beijing / unknown exchanges are NEVER mapped to Shanghai or Shenzhen.
    return None


def _date(value: str, label: str) -> pd.Timestamp:
    x = str(value).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", x):
        raise ValueError(f"Bad {label} YYYY-MM-DD")
    return pd.Timestamp(x).normalize()


def _nullable_day(value) -> pd.Timestamp:
    x = str(value).strip()
    if x.lower() in ("", "none", "nan", "nat", "--"):
        return pd.NaT
    try:
        return _date(x, "event date")
    except (ValueError, TypeError):
        return pd.NaT


def _snapshot(folder: Path, metadata: Path) -> tuple[dict, pd.DataFrame]:
    meta = json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
    archived = json.loads(metadata.read_text(encoding="utf-8"))
    if archived.get("model_version") != "P4_RIDGE_V1":
        raise ValueError("Expected frozen Ridge metadata")
    if meta.get("snapshot_version") != "P4_AKSHARE_FORMAL_UNIVERSE_V1":
        raise ValueError("Expected P4 historical market universe snapshot")
    if meta.get("test_start_EXCLUDED") != archived.get("test_start"):
        raise ValueError("Old test cutoff mismatch")
    universe_path = folder / "universe.csv"
    if _sha(universe_path) != meta["universe_sha256"]:
        raise ValueError("Market stock-universe snapshot SHA256 mismatch")
    stocks = pd.read_csv(universe_path, dtype="string", keep_default_na=False)
    if "stock_code" not in stocks or len(stocks) != meta["symbols"]:
        raise ValueError("Snapshot stock count mismatch")
    codes = [_code(x) for x in stocks.stock_code]
    if len(codes) != len(set(codes)):
        raise ValueError("Duplicate stock codes in snapshot")
    return meta, stocks


def _response_rows(result, *, cap: int) -> tuple[list[dict], int]:
    if str(getattr(result, "error_code", "?")) != "0":
        raise ValueError("BaoStock query response error: " + str(getattr(result, "error_msg", ""))[:130])
    fields = list(getattr(result, "fields", []))
    if not fields:
        raise ValueError("BaoStock response has no column metadata")
    missing = set(RAW_FIELDS) - set(fields)
    if missing:
        raise ValueError("BaoStock schema changed: " + ",".join(sorted(missing)))
    records = []
    while result.next():
        record = result.get_row_data()
        if len(record) != len(fields):
            raise ValueError("BaoStock data/field length disagreement")
        records.append(dict(zip(fields, record)))
        if len(records) >= cap:
            raise OverflowError("BaoStock per-year response reached row cap")
    if str(getattr(result, "error_code", "?")) != "0":
        raise ValueError("BaoStock query iteration failed")
    return records, len(records)


def fetch_query(bs, code: str, year: int, *, begin, cut, cap: int
                ) -> tuple[dict, list[dict]]:
    bs_code = _provider_code(code)
    status, error, row_count, excluded, invalid = "SUCCESS", "", 0, 0, 0
    raw_rows = []
    if bs_code is None:
        status, error = "UNSUPPORTED_EXCHANGE", "No verified SH/SZ BaoStock code mapping"
    else:
        try:
            rs = bs.query_dividend_data(code=bs_code, year=str(year), yearType="operate")
            retrieved, row_count = _response_rows(rs, cap=cap)
            for item in retrieved:
                if item["code"] != bs_code:
                    raise ValueError("Provider returned incorrect security code")
                ex = _nullable_day(item["dividOperateDate"])
                if pd.isna(ex) or ex.year != year:
                    invalid += 1
                    continue
                if ex < begin or ex >= cut:
                    excluded += 1
                    continue
                if any(_nullable_day(item[col]) >= cut for col in ("dividPlanDate", "dividRegistDate")):
                    excluded += 1
                    continue
                raw_rows.append({"request_code": code, "query_year": year,
                                 **{k: str(item[k]) for k in RAW_FIELDS}})
            if invalid:
                status = "UNKNOWN_EX_DATE"
                error = f"{invalid} provider rows lack valid ex-date and were NOT saved"
        except OverflowError as exc:
            status, error = "ROW_LIMIT", str(exc)
            raw_rows = []
        except Exception as exc:
            status, error = "API_ERROR", f"{type(exc).__name__}: {str(exc)[:180]}"
            raw_rows = []
    return {"stock_code": code, "bs_code": bs_code or "", "year": year, "status": status,
            "response_rows": row_count, "saved_rows": len(raw_rows),
            "excluded_out_of_window": excluded, "unknown_ex_date_rows": invalid,
            "error": error}, raw_rows


def collect(snapshot: Path, reference_metadata: Path, *, bs, out: Path,
            codes: list[str], years: list[int], wait_seconds: float = 1.0,
            cap: int = 1000) -> dict:
    """One immutable provider snapshot; does not read or update trading database."""
    meta, stocks = _snapshot(snapshot, reference_metadata)
    if out.exists():
        raise FileExistsError("BaoStock batch already exists: choose new --out")
    begin = _date(meta["start"], "snapshot start")
    cut = _date(meta["test_start_EXCLUDED"], "old test cutoff")
    if not codes or len(codes) != len(set(codes)) or len(codes) > 50 or wait_seconds < 0 or cap < 2:
        raise ValueError("Invalid code request / cap / wait")
    if any(_code(c) not in set(stocks.stock_code) for c in codes):
        raise ValueError("Requested stock outside frozen formal market snapshot")
    if not years or len(years) != len(set(years)) or any(
        not isinstance(y, int) or y < begin.year or y > cut.year for y in years
    ):
        raise ValueError("Year selection outside frozen research window")

    raw_rows = []
    logs = []
    # Login/logout is batched. A failed login is NOT silently an empty dividend.
    login = bs.login()
    if str(getattr(login, "error_code", "?")) != "0":
        raise RuntimeError("BaoStock login failed: " + str(getattr(login, "error_msg", ""))[:180])
    try:
        for code in codes:
            bs_code = _provider_code(code)
            for year in years:
                entry, rows = fetch_query(bs, code, year, begin=begin, cut=cut, cap=cap)
                logs.append(entry)
                raw_rows.extend(rows)
                if wait_seconds and bs_code is not None:
                    time.sleep(wait_seconds)
    finally:
        bs.logout()
    ledger = pd.DataFrame(logs)
    raw = pd.DataFrame(raw_rows, columns=("request_code", "query_year") + RAW_FIELDS)
    out.mkdir(parents=True)
    raw.to_csv(out / "raw_baostock_dividends.csv", index=False, encoding="utf-8")
    ledger.to_csv(out / "request_ledger.csv", index=False, encoding="utf-8")
    manifest = {
        "version": VERSION, "snapshot_sha256": meta["universe_sha256"],
        "test_start_EXCLUDED": str(cut.date()), "start": str(begin.date()),
        "years": years, "stock_codes": codes, "requested_queries": len(logs),
        "successful_queries": int(ledger.status.eq("SUCCESS").sum()),
        "unsuccessful_queries": int(ledger.status.ne("SUCCESS").sum()),
        "saved_candidate_rows": len(raw),
        "files_sha256": {k: _sha(out / k) for k in (
            "raw_baostock_dividends.csv", "request_ledger.csv")},
        "source": "BAOSTOCK_PUBLIC_DIVIDEND", "coverage_certified": False,
        "warning": CAUTION,
    }
    (out / "batch.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def normalize(raw: pd.DataFrame, *, start: str, cutoff: str
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-share BaoStock dividends to same units as P4-E1; no certificates."""
    lo, cut = _date(start, "start"), _date(cutoff, "cutoff")
    candidates, issues = [], []
    for r in raw.to_dict("records"):
        code = str(r["request_code"])
        ex = _nullable_day(r["dividOperateDate"])
        if pd.isna(ex) or not (lo <= ex < cut):
            raise ValueError("Persisted BaoStock row outside frozen pretest interval")
        if r["code"] != _provider_code(code):
            raise ValueError("Wrong BaoStock stock code in raw event")
        try:
            announced = _date(r["dividPlanDate"], "implementation announcement")
            registration = _date(r["dividRegistDate"], "record date")
            if not (announced <= registration < ex):
                raise ValueError("Invalid implementation/registration/ex-date order")
            amounts, blank_fields = [], []
            for name in ("dividStocksPs", "dividReserveToStockPs", "dividCashPsBeforeTax"):
                value = str(r[name]).strip()
                if value.lower() in ("", "none", "nan", "nat", "--"):
                    blank_fields.append(name)
                    amounts.append(0.0)
                else:
                    x = float(value)
                    if not np.isfinite(x) or x < 0:
                        raise ValueError("Invalid per-share action amount " + name)
                    amounts.append(x)
            shares, reserve, cash = amounts
            if blank_fields and (shares + reserve + cash) == 0:
                raise ValueError("Missing entitlement fields without any confirmed nonzero amount")
            if shares + reserve + cash == 0:
                continue
            candidates.append({"stock_code": code, "ex_date": str(ex.date()),
                               "published_on": str(announced.date()),
                               "record_date": str(registration.date()),
                               "share_multiplier": 1.0 + shares + reserve,
                               "cash_per_old_share": cash,
                               "source_id": f"UNVERIFIED-BAOSTOCK:{code}:{ex.date()}",
                               "source_url": SOURCE,
                               "review_status": (
                                   "BAOSTOCK_PARTIAL_AMOUNT_FIELDS_UNVERIFIED" if blank_fields
                                   else "BAOSTOCK_ONLY_UNVERIFIED"),
                               "blank_amount_fields": "|".join(blank_fields)})
        except (ValueError, TypeError, KeyError) as exc:
            issues.append({"stock_code": code, "ex_date": str(ex.date()),
                           "reason": f"INVALID_BAOSTOCK_ROW:{str(exc)[:160]}"})
    candidate_df = pd.DataFrame(candidates, columns=(
        "stock_code", "ex_date", "published_on", "record_date",
        "share_multiplier", "cash_per_old_share", "source_id", "source_url",
        "review_status", "blank_amount_fields"))
    issues_df = pd.DataFrame(issues, columns=("stock_code", "ex_date", "reason"))
    if candidate_df.empty:
        return candidate_df, issues_df
    # Duplicate ex-dates must not arbitrarily choose one event or add twice.
    dup = candidate_df.duplicated(["stock_code", "ex_date"], keep=False)
    for rec in candidate_df.loc[dup].itertuples(index=False):
        issues_df.loc[len(issues_df)] = [rec.stock_code, rec.ex_date, "DUPLICATE_EX_DATE_UNRESOLVED"]
    candidate_df = candidate_df.loc[~dup].reset_index(drop=True)
    return candidate_df, issues_df


def audit(snapshot: Path, reference_metadata: Path, batches: Path, *,
          akshare_events: Path | None = None,
          akshare_stock_coverage: Path | None = None,
          issuer_controls: Path | None = None,
          ) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Verify immutable evidence, compare candidate events, never certify."""
    meta, univ = _snapshot(snapshot, reference_metadata)
    files = sorted(batches.glob("batch_*/batch.json"))
    if not files:
        raise ValueError("No BaoStock batches in batch_*/batch.json")
    seen = set()
    ledgers, raws = [], []
    for fp in files:
        batch = json.loads(fp.read_text(encoding="utf-8"))
        if (batch.get("version") != VERSION or
                batch.get("snapshot_sha256") != meta["universe_sha256"] or
                batch.get("test_start_EXCLUDED") != meta["test_start_EXCLUDED"] or
                batch.get("start") != meta["start"]):
            raise ValueError("BaoStock batch snapshot/boundary mismatch")
        for filename, expected in batch["files_sha256"].items():
            if _sha(fp.parent / filename) != expected:
                raise ValueError("BaoStock batch SHA256 mismatch: " + filename)
        lg = pd.read_csv(fp.parent / "request_ledger.csv", dtype="string", keep_default_na=False)
        raw = pd.read_csv(fp.parent / "raw_baostock_dividends.csv", dtype="string", keep_default_na=False)
        required = {"stock_code", "year", "status", "response_rows"}
        if not required.issubset(lg) or not set(RAW_FIELDS).issubset(raw):
            raise ValueError("Incomplete BaoStock ledger or raw column schema")
        for row in lg.itertuples(index=False):
            key = (str(row.stock_code), str(row.year))
            if key in seen:
                raise ValueError("Overlapping stocks/year requests across BaoStock batches")
            seen.add(key)
        if not lg.stock_code.isin(univ.stock_code).all():
            raise ValueError("BaoStock batch includes stock outside snapshot")
        okkeys = {(str(row.stock_code), str(row.year)) for row in lg.itertuples(index=False) if row.status == "SUCCESS"}
        if any((str(row.request_code), str(row.query_year)) not in okkeys for row in raw.itertuples(index=False)):
            raise ValueError("BaoStock raw rows do not have a successful query")
        ledgers.append(lg)
        raws.append(raw)
    ledger = pd.concat(ledgers, ignore_index=True)
    raw = pd.concat(raws, ignore_index=True)
    return audit_evidence(
        meta, univ, ledger, raw,
        akshare_events=(pd.read_csv(akshare_events, dtype="string", keep_default_na=False,
                                   encoding="utf-8-sig") if akshare_events is not None else None),
        akshare_stock_coverage=(pd.read_csv(akshare_stock_coverage, dtype="string",
                                           encoding="utf-8-sig")
                                if akshare_stock_coverage is not None else None),
        issuer_controls=issuer_controls,
    )


def audit_evidence(meta, univ, ledger, raw, *, akshare_events=None,
                   akshare_stock_coverage=None, issuer_controls=None):
    """Corroborate an authenticated effective view with existing audit rules."""
    candidates, bad = normalize(raw, start=meta["start"], cutoff=meta["test_start_EXCLUDED"])
    query_status = ledger.groupby("stock_code").agg(
        requested_years=("year", "nunique"),
        success_years=("status", lambda x: int(x.eq("SUCCESS").sum())),
        failed_years=("status", lambda x: int(x.ne("SUCCESS").sum())),
        returned_rows=("response_rows", lambda x: sum(int(z) for z in x)),
    ).reset_index()
    stock = univ[["stock_code"]].merge(query_status, on="stock_code", how="left", validate="one_to_one")
    stock["provider_status"] = np.select(
        [stock.requested_years.isna(), stock.failed_years.fillna(0).astype(int).gt(0),
         stock.requested_years.fillna(0).astype(int).lt(
             len(range(_date(meta["start"], "start").year,
                       _date(meta["test_start_EXCLUDED"], "test_start").year + 1)))],
        ["NOT_FETCHED", "FAILED_OR_UNSUPPORTED", "PARTIAL_YEAR_RANGE"],
        default="FETCHED_NOT_VERIFIED",
    )
    stock["complete_verified"] = False
    stock["candidate_events"] = stock.stock_code.map(
        candidates.groupby("stock_code").size()).fillna(0).astype(int)
    stock["no_event_rows_observed"] = stock.candidate_events.eq(0)
    # Source coverage must be known before calling absent events a mismatch.
    ak_status_by_code = {}
    if akshare_stock_coverage is not None:
        other = akshare_stock_coverage
        if not {"stock_code", "coverage_status"} <= set(other):
            raise ValueError("AkShare coverage CSV missing required columns")
        if other.stock_code.duplicated().any():
            raise ValueError("Duplicate AkShare stock coverage rows")
        ak_status_by_code = dict(zip(other.stock_code.astype(str),
                                     other.coverage_status.astype(str)))
        stock = stock.merge(other[["stock_code", "coverage_status"]].rename(
            columns={"coverage_status": "akshare_coverage_status"}),
            on="stock_code", how="left", validate="one_to_one")
    bao_status_by_code = dict(zip(stock.stock_code.astype(str),
                                  stock.provider_status.astype(str)))
    mismatches = []
    if akshare_events is not None:
        ak = akshare_events
        required = {"stock_code", "ex_date", "share_multiplier", "cash_per_old_share"}
        if not required.issubset(ak):
            raise ValueError("AkShare candidate CSV missing required columns")
        for c in candidates.to_dict("records"):
            matched = ak.loc[ak.stock_code.eq(c["stock_code"]) & ak.ex_date.eq(c["ex_date"])]
            if len(matched) == 0:
                # AkShare may not have requested BYD at all in first 10 stocks;
                # that is NOT a disagreement with BaoStock.
                outcome = ("BAOSTOCK_ONLY_UNVERIFIED" if
                           ak_status_by_code.get(c["stock_code"]) == "FETCHED_NOT_VERIFIED"
                           else "AKSHARE_NOT_COMPARABLE_YET")
            elif len(matched) > 1:
                outcome = "AKSHARE_DUPLICATE_UNRESOLVED"
            else:
                # BaoStock legitimately leaves some per-share action fields
                # blank. Normalization keeps a provisional zero for arithmetic,
                # but that zero must NOT become independent evidence of "no
                # bonus / no capitalization". Check only the observed terms.
                observed = set(str(c["blank_amount_fields"]).split("|")) - {""}
                share_equal = np.isclose(
                    float(matched.iloc[0].share_multiplier),
                    float(c["share_multiplier"]), rtol=0, atol=1e-6)
                cash_equal = np.isclose(
                    float(matched.iloc[0].cash_per_old_share),
                    float(c["cash_per_old_share"]), rtol=0, atol=1e-6)
                share_complete = not ({"dividStocksPs", "dividReserveToStockPs"} & observed)
                cash_complete = "dividCashPsBeforeTax" not in observed
                if ("record_date" in matched and matched.iloc[0].record_date
                        and matched.iloc[0].record_date != c["record_date"]):
                    outcome = "BAOSTOCK_AKSHARE_DATE_CONFLICT"
                elif (share_complete and not share_equal) or (cash_complete and not cash_equal):
                    outcome = "BAOSTOCK_AKSHARE_TERMS_CONFLICT"
                elif observed:
                    outcome = "BAOSTOCK_AKSHARE_PARTIAL_AMOUNT_FIELDS_UNRESOLVED"
                else:
                    outcome = "BAOSTOCK_AKSHARE_MATCH_UNVERIFIED"
            mismatches.append({"stock_code": c["stock_code"], "ex_date": c["ex_date"],
                               "result": outcome})
        # AkShare-only records are informative only if full BaoStock year
        # range was successfully queried for the relevant stock.
        for rec in ak.to_dict("records"):
            code = str(rec["stock_code"])
            if code not in set(ledger.stock_code):
                continue
            if not ((candidates.stock_code.eq(code) &
                     candidates.ex_date.eq(rec["ex_date"])).any()):
                outcome = ("AKSHARE_ONLY_UNVERIFIED" if
                           bao_status_by_code.get(code) == "FETCHED_NOT_VERIFIED"
                           else "BAOSTOCK_NOT_COMPARABLE_YET")
                mismatches.append({"stock_code": code,
                                   "ex_date": rec["ex_date"], "result": outcome})
    compare = pd.DataFrame(mismatches, columns=("stock_code", "ex_date", "result"))
    controls = []
    if issuer_controls is not None:
        refs = pd.read_csv(issuer_controls, dtype="string", keep_default_na=False,
                           encoding="utf-8-sig")
        if not {"stock_code", "ex_date", "share_multiplier", "cash_per_old_share",
                "source_id"}.issubset(refs):
            raise ValueError("Issuer reference lacks action terms/provenance")
        known_status = stock.set_index("stock_code")["provider_status"]
        for row in refs.to_dict("records"):
            ex = _date(row["ex_date"], "issuer ex_date")
            if not (_date(meta["start"], "start") <= ex < _date(meta["test_start_EXCLUDED"], "cutoff")):
                continue
            code = _code(row["stock_code"])
            status = known_status.get(code, "NOT_FETCHED")
            if status == "NOT_FETCHED":
                disposition = "NOT_FETCHED_FOR_ISSUER_CONTROL"
            elif status != "FETCHED_NOT_VERIFIED":
                disposition = "INCOMPLETE_FETCH_FOR_ISSUER_CONTROL"
            else:
                found = candidates.loc[candidates.stock_code.eq(code) &
                                       candidates.ex_date.eq(str(ex.date()))]
                matched = (len(found) == 1 and
                    np.isclose(float(found.iloc[0].share_multiplier), float(row["share_multiplier"]),
                               atol=1e-6, rtol=0) and
                    np.isclose(float(found.iloc[0].cash_per_old_share), float(row["cash_per_old_share"]),
                               atol=1e-6, rtol=0))
                if matched and str(found.iloc[0]["blank_amount_fields"]):
                    # Matching a source using placeholder zeros is NOT an
                    # independent issuer control on missing BaoStock fields.
                    disposition = "PARTIAL_AMOUNTS_MATCH_OBSERVED_TERMS_ONLY"
                else:
                    disposition = ("TERMS_MATCH_ISSUER_SAMPLE_UNVERIFIED_COVERAGE" if matched
                                   else "MISSING_OR_CONFLICTING_BAOSTOCK_CANDIDATE")
            controls.append({"stock_code": code, "ex_date": str(ex.date()),
                             "source_id": row["source_id"], "result": disposition})
    control_df = pd.DataFrame(controls, columns=("stock_code", "ex_date", "source_id", "result"))
    counts = {str(k): int(v) for k, v in stock.provider_status.value_counts().items()}
    summary = {
        "audit_version": "P4_BAOSTOCK_FREE_CORROBORATION_V1",
        "snapshot_sha256": meta["universe_sha256"],
        "test_start_EXCLUDED": meta["test_start_EXCLUDED"],
        "stocks_in_snapshot": len(stock),
        "bao_query_rows": len(ledger), "bao_successful_queries": int(ledger.status.eq("SUCCESS").sum()),
        "bao_failed_or_unsupported_queries": int(ledger.status.ne("SUCCESS").sum()),
        "bao_unverified_candidate_events": len(candidates),
        "bao_invalid_or_duplicate_rows": len(bad),
        "bao_stock_status_counts": counts,
        "cross_source_comparisons": int(len(compare)),
        "cross_source_fully_observed_matches": int(compare.result.eq(
            "BAOSTOCK_AKSHARE_MATCH_UNVERIFIED").sum()),
        "cross_source_partial_amount_fields_unresolved": int(compare.result.eq(
            "BAOSTOCK_AKSHARE_PARTIAL_AMOUNT_FIELDS_UNRESOLVED").sum()),
        "cross_source_disagreements_or_missing": int(compare.result.isin([
            "BAOSTOCK_AKSHARE_TERMS_CONFLICT", "BAOSTOCK_AKSHARE_DATE_CONFLICT", "BAOSTOCK_ONLY_UNVERIFIED",
            "AKSHARE_ONLY_UNVERIFIED", "AKSHARE_DUPLICATE_UNRESOLVED"
        ]).sum()),
        "cross_source_not_comparable_yet": int(compare.result.isin([
            "AKSHARE_NOT_COMPARABLE_YET", "BAOSTOCK_NOT_COMPARABLE_YET"
        ]).sum()),
        "issuer_controls_count": len(control_df),
        "issuer_controls_terms_matched": int(control_df.result.eq(
            "TERMS_MATCH_ISSUER_SAMPLE_UNVERIFIED_COVERAGE").sum()),
        "issuer_controls_partial_amount_fields": int(control_df.result.eq(
            "PARTIAL_AMOUNTS_MATCH_OBSERVED_TERMS_ONLY").sum()),
        "issuer_controls_missing_or_conflicting": int(control_df.result.eq(
            "MISSING_OR_CONFLICTING_BAOSTOCK_CANDIDATE").sum()),
        "complete_verified_assertions_created": 0,
        "can_emit_training_labels": False,
        "full_historical_action_coverage_certified": False,
        "warning": CAUTION,
    }
    return summary, candidates, stock, pd.concat([bad.assign(result="INVALID_BAOSTOCK_ROW"),
                compare], ignore_index=True, sort=False), control_df


def main() -> None:
    p = argparse.ArgumentParser(description="BaoStock free P4 historical corporate actions, NOT certified")
    sub = p.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="immutable per-code-per-year dividend evidence")
    fetch.add_argument("--snapshot", type=Path, required=True)
    fetch.add_argument("--reference-metadata", type=Path, required=True)
    fetch.add_argument("--code", action="append", help="specific six-digit stock (repeatable)")
    fetch.add_argument("--offset", type=int, help="optional contiguous snapshot offset")
    fetch.add_argument("--limit", type=int, default=10)
    fetch.add_argument("--wait-seconds", type=float, default=1.0)
    fetch.add_argument("--row-cap", type=int, default=1000)
    fetch.add_argument("--out", type=Path, required=True)
    auditp = sub.add_parser("audit", help="corroborate BaoStock with existing AkShare exports")
    auditp.add_argument("--snapshot", type=Path, required=True)
    auditp.add_argument("--reference-metadata", type=Path, required=True)
    auditp.add_argument("--batch-root", type=Path, required=True)
    auditp.add_argument("--akshare-events", type=Path)
    auditp.add_argument("--akshare-coverage", type=Path)
    auditp.add_argument("--issuer-controls", type=Path,
                        default=Path("ashare/data/corporate_actions_verified_examples.csv"))
    auditp.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    meta, univ = _snapshot(args.snapshot, args.reference_metadata)
    if args.command == "fetch":
        if args.code and args.offset is not None:
            p.error("Use either --code or --offset, not both")
        if args.code:
            codes = [_code(c) for c in args.code]
        else:
            if args.offset is None or args.offset < 0 or args.limit < 1:
                p.error("Provide nonnegative --offset and positive --limit")
            codes = [str(c) for c in univ.iloc[args.offset:args.offset+args.limit].stock_code]
        try:
            import baostock as bs
        except ImportError as exc:
            raise RuntimeError("Install free dependency: python -m pip install baostock") from exc
        years = list(range(_date(meta["start"], "start").year,
                           _date(meta["test_start_EXCLUDED"], "cutoff").year + 1))
        outcome = collect(args.snapshot, args.reference_metadata, bs=bs, out=args.out,
                          codes=codes, years=years,
                          wait_seconds=args.wait_seconds, cap=args.row_cap)
    else:
        if args.out.exists():
            raise FileExistsError("Immutable audit output already exists")
        result, events, stocks, issues, controls = audit(
            args.snapshot, args.reference_metadata, args.batch_root,
            akshare_events=args.akshare_events,
            akshare_stock_coverage=args.akshare_coverage,
            issuer_controls=args.issuer_controls)
        args.out.mkdir(parents=True)
        (args.out / "baostock_corrob_audit.json").write_text(json.dumps(result, indent=2,
                                                              ensure_ascii=False), encoding="utf-8")
        events.to_csv(args.out / "UNVERIFIED_baostock_candidates.csv", index=False, encoding="utf-8-sig")
        stocks.to_csv(args.out / "baostock_coverage_by_stock.csv", index=False, encoding="utf-8-sig")
        issues.to_csv(args.out / "baostock_cross_source_issues.csv", index=False, encoding="utf-8-sig")
        controls.to_csv(args.out / "baostock_issuer_controls.csv", index=False, encoding="utf-8-sig")
        (args.out / "COMPLETE_VERIFIED_NOT_ISSUED.txt").write_text(CAUTION + "\n")
        outcome = result
    print(json.dumps(outcome, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
