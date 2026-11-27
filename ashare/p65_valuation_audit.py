"""Audit archived P6 valuations and triage late settlement receipts without replaying fills."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
from pathlib import Path
import os
from tempfile import TemporaryDirectory

from .p4_action_pipeline import atomic_json, exclusive, read_json
from .p4_subset_inputs import hash_file, parquet_read
from .p55_daily import now_shanghai
from .p55_reports import artifact
from .p6_portfolio_run import action_flags, checked_day, evidence_for
from .p6_settlements import read_receipt

VERSION = "P65_VALUATION_AUDIT_V1"


def marked_value(position):
    if position is None:
        return None
    return round(position["mark"] * (position["shares"] + position["pending_shares"]), 2)


def bench_roster(panel, inputs):
    source_digest, signal = inputs["benchmark_manifest_sha256"], inputs["signal_day"]
    if source_digest is None:
        return set()
    for branch in ("timely", "historical"):
        path = panel / branch / signal / "manifest.json"
        if path.is_file() and hash_file(path) == source_digest:
            manifest = read_json(path)
            if manifest["trade_date"] != signal or not manifest["formal_session"]:
                raise ValueError("Unexpected archived equal-weight manifest")
            spec = manifest["models"]["equal_weight"]
            rows = parquet_read(artifact(path.parent, spec["predictions_file"], spec["predictions_sha256"]))
            if len(rows) != spec["rows"] or rows.stock_code.duplicated().any() or not rows.trade_date.eq(signal).all():
                raise ValueError("Unexpected archived equal-weight roster")
            return set(rows.stock_code)
    raise ValueError(f"Archived equal-weight manifest cannot be verified for {signal}")


def buy_rows(path):
    with (path / "portfolio_trades.csv").open(newline="") as stream:
        return [r for r in csv.DictReader(stream) if r["side"] == "buy"]


def claim_keys(account):
    return Counter((r["day"],r["stock_code"],r["reason"]) for r in account.get("unknown_claims", ()))


def missing_check(stocks, day, targets, originally_checked):
    flags = action_flags(stocks, day, targets)
    return {
        "symbols": len(targets),
        "outside_original_scan": len(targets - originally_checked),
        "witness_gap": sorted(c for c,r in flags.items() if r["status"] == "MISSING_COVERAGE"),
        "unverified_event": sorted(c for c,r in flags.items() if r["status"] == "UNKNOWN"),
    }


def receipt_evidence(root, day, stock, observed_now):
    terms = set()
    details = []
    for folder in sorted((root/day/stock).glob("*")):
        if not (folder/"receipt.json").is_file():
            continue
        review, observed = read_receipt(folder)
        if observed > observed_now:
            raise ValueError("Settlement receipt observation lies in the future")
        keys = ("record_date","cash_per_share","share_multiplier","cash_pay_date","share_list_date","cash_basis")
        terms.add(tuple(Decimal(str(review[k])) if k in {"cash_per_share","share_multiplier"} else review[k]
                        for k in keys))
        details.append({**{k: review[k] for k in keys}, "sha256":hash_file(folder/"receipt.json"),
                        "observed_at":observed.isoformat(), "source_id":review["source_id"]})
    return terms, details


def revision_status(reason, terms, receipts):
    if len(terms) > 1:
        return "CONFLICTING_RECEIPTS_REVIEW_NEEDED"
    if not receipts:
        return "NO_VERIFIED_RECEIPT"
    if reason in {"MISSING_COVERAGE", "MISSING_SAME_DAY_ACTION_COVERAGE"}:
        return "RECEIPT_PRESENT_WHOLE_DAY_COVERAGE_STILL_UNVERIFIED"
    return "RECEIPT_PRESENT_REVIEW_ECONOMIC_ENTITLEMENT"


def revisions(rows, settlement_root, observed_now):
    indexed = defaultdict(list)
    for row in rows:
        indexed[row["claim_day"], row["stock_code"], row["reason"]].append(row)
    result = []
    for (day, stock, reason), claim_rows in sorted(indexed.items()):
        terms, details = receipt_evidence(settlement_root, day, stock, observed_now)
        result.append({"claim_day": day, "stock_code":stock, "reason":reason,
                       "accounts":len({r["account"] for r in claim_rows}),
                       "status":revision_status(reason, terms, details), "receipts":details})
    return result


def audit_track(folder, legacy, panel):
    policy_sha = hash_file(folder.parent / "policy.json")
    previous, previous_sha, day_rows, claims, coverage = {}, None, [], [], []
    for path in sorted(p for p in folder.glob("????-??-??") if p.is_dir()):
        day = path.name
        current, day_sha = checked_day(path, previous_sha, policy_sha)
        summary = read_json(path / "portfolio_summary.json")
        inputs = read_json(path / "inputs.json")
        held = set().union(*(set(r["holdings"]) for r in previous.get("accounts", {}).values()))
        old_bench = set(previous.get("benchmark_weights", {}))
        active = held | old_bench
        # Archive evidence uses the original digest and track. Refuse a changed evidence witness.
        evidence_sha = inputs["closing_evidence_sha256"]
        stocks = {}
        if evidence_sha is not None:
            stocks, actual_sha = evidence_for(legacy, day, summary["track"])
            if actual_sha != evidence_sha:
                raise ValueError(f"Archived evidence differs from original capture on {day}")
        new_bench = bench_roster(panel, inputs) - old_bench
        fills = buy_rows(path)
        new_buys = {r["stock_code"] for r in fills}
        roles = {"opening_holdings":held, "previous_benchmark":old_bench,
                 "same_day_buys":new_buys, "new_benchmark_names":new_bench}
        coverage.append({"day":day, "evidence_sha256":evidence_sha,
                         "roles":{role:missing_check(stocks,day,codes,active) for role,codes in roles.items()},
                         "same_day_buy_exposure_outside_scan":round(sum(float(r["gross"]) for r in fills
                                                                     if r["stock_code"] not in active),2)})
        for account, checkpoint in current["accounts"].items():
            prior = previous.get("accounts", {}).get(account, {})
            before, after = claim_keys(prior), claim_keys(checkpoint)
            for (claim_day, stock, reason), count in (after - before).items():
                claims.append({"claim_day":claim_day,"account":account,"stock_code":stock,"reason":reason,
                               "records":count, "prior_mark_amount":marked_value(prior.get("holdings",{}).get(stock)),
                               "close_mark_amount":marked_value(checkpoint["holdings"].get(stock))})
        rejection_reasons = summary["rejection_reasons"]
        val = summary["valuation_statuses"]
        impacted, impacted_accounts = [], []
        for account,row in current["accounts"].items():
            if row["unknown_claims"]:
                impacted_accounts.append(account)
            blocked = {claim["stock_code"] for claim in row["unknown_claims"]}
            impacted.extend(marked_value(row["holdings"][c]) for c in blocked.intersection(row["holdings"]))
        day_rows.append({"day":day,"summary_sha256":day_sha,"accounts":len(current["accounts"]),
                         "strict_nav_unavailable":len(current["accounts"])-val.get("SIMULATED_CURRENT_MARKS",0),
                         "new_claim_account_stocks":sum(r["claim_day"]==day for r in claims),
                         "accounts_with_claims":len(impacted_accounts),
                         "blocked_buy_attempts":rejection_reasons.get("UNCONFIRMED_ACCOUNT_ECONOMICS",0),
                         "blocked_sell_attempts":rejection_reasons.get("UNRESOLVED_ENTITLEMENT",0),
                         "current_claim_linked_mark_amount":round(sum(impacted),2),
                         "benchmark_daily_unavailable":inputs["benchmark_daily_return"] is None and inputs["signal_day"] is not None,
                         "benchmark_cumulative_unavailable":sum(a["benchmark_return"] is None for a in summary["accounts"])})
        previous,previous_sha = current,day_sha
    return {"track":folder.name,"days":day_rows,"claims":claims,"coverage":coverage,"terminal_sha256":previous_sha}


def write_csv(path, rows, fields):
    with path.open("w",newline="",encoding="utf-8") as stream:
        w = csv.DictWriter(stream,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_report(folder, groups, revision, moment, hashes, settlement_root):
    all_claims = [row|{"track":g["track"]} for g in groups for row in g["claims"]]
    all_days = [row|{"track":g["track"]} for g in groups for row in g["days"]]
    write_csv(folder/"valuation_days.csv",all_days,(
        "track","day","summary_sha256","accounts","strict_nav_unavailable","new_claim_account_stocks",
        "accounts_with_claims","blocked_buy_attempts","blocked_sell_attempts","current_claim_linked_mark_amount",
        "benchmark_daily_unavailable","benchmark_cumulative_unavailable"))
    write_csv(folder/"entitlement_exposures.csv",all_claims,(
        "track","claim_day","account","stock_code","reason","records","prior_mark_amount","close_mark_amount"))
    atomic_json(folder/"coverage_roles.json",{
        "tracks":{g["track"]:g["coverage"] for g in groups},
        "scope":"Membership by role, with outside_original_scan relative to union of prior holdings and prior benchmark. "
                "Frozen buy and benchmark intraday routines do not enforce these evidence flags for new entries."})
    atomic_json(folder/"settlement_revision_review.json",{
        "as_of":moment.isoformat(),"cases":revision,"archives_sha256":hashes,"affects_original_archives":False,
        "review_settlement_root":str(settlement_root),
        "scope":"Triage only. Later verified terms may aid reconciliation and do not authorize rebuilding prior fills, "
                "clearing historical account flags, or establishing the day's complete event coverage."})
    md = ["# P6.5 组合估值核验", "", f"观测时间 {moment.isoformat()}。", "",
          "| 账本 | 交易日 | 严格净值缺失账户日 | 公司行为挡买 | 公司行为挡卖 |",
          "|---|---:|---:|---:|---:|"]
    for g in groups:
        rows = g["days"]
        md.append(f"| {g['track']} | {len(rows)} | {sum(r['strict_nav_unavailable'] for r in rows)} | "
                  f"{sum(r['blocked_buy_attempts'] for r in rows)} | {sum(r['blocked_sell_attempts'] for r in rows)} |")
    md.extend(["","金额按各账户记录分别加总，同股跨策略可能重复。",
               "当日新买与基准新成员的公司行为见证单列，估值复核不改变成交。",
               "收件复核列出实际可见的发行人凭证及其散列。凭证仅能证明其经济条款，整日覆盖仍须独立确认。"])
    (folder/"valuation_audit.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    atomic_json(folder/"manifest.json",{"version":VERSION,"as_of":moment.isoformat(),
        "portfolio_chain_sha256":hashes,"formal_forward_available":any(g["track"]=="forward" and g["days"] for g in groups),
        "read_only_inputs":True, "files":{p.name:hash_file(p) for p in folder.iterdir() if p.is_file()}})


def run(root, legacy, panel, out, review_settlements=None):
    root, out = root.resolve(), out.resolve()
    settlement_root = review_settlements.resolve() if review_settlements else root/"settlements"
    if any(out == source or source in out.parents
           for source in (root, legacy.resolve(), panel.resolve(), settlement_root)):
        raise ValueError("Use an output directory outside all frozen inputs")
    moment = now_shanghai()
    folders = sorted(p for p in root.glob("diagnostic-*") if p.is_dir())
    if (root/"forward").is_dir():
        folders.append(root/"forward")
    groups = [audit_track(f,legacy,panel) for f in folders]
    rows = [row for g in groups for row in g["claims"]]
    revision = revisions(rows,settlement_root,moment)
    hashes = {g["track"]:g["terminal_sha256"] for g in groups}
    with exclusive(out.parent):
        if out.exists():
            raise ValueError("Use a new revision report directory")
        with TemporaryDirectory(dir=out.parent) as tmp:
            stage = Path(tmp)/"valuation"
            stage.mkdir()
            write_report(stage,groups,revision,moment,hashes,settlement_root)
            os.rename(stage,out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("portfolio","legacy","panel","out"):
        p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--review-settlements",type=Path,help="Independent receipt directory for closing review")
    args = p.parse_args()
    print(run(args.portfolio,args.legacy,args.panel,args.out,args.review_settlements))


if __name__ == "__main__":
    main()
