"""Isolated research portfolios using archived P6 bars, point-in-time evidence and symbol-level holds."""
import argparse
from dataclasses import asdict
from datetime import date, timedelta
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import text

from . import p6_portfolio_run as frozen
from . import p6_settlements
from .p4_action_pipeline import atomic_json, exclusive, read_json
from .p4_subset_inputs import hash_file
from .p55_daily import now_shanghai, parse_clock
from .p6_portfolio import STRATEGIES, initial_state, money, positive, step
from .p6_portfolio_report import publish_day
from .p65_valuation_audit import receipt_evidence

VERSION = "P66_EXPLORATORY_SYMBOL_CLAIMS_V1"


def freeze(root, portfolio, diagnostic=False):
    if root.resolve().is_relative_to(portfolio.resolve()) or portfolio.resolve().is_relative_to(root.resolve()):
        raise ValueError("Use a separate exploratory ledger outside the strict portfolio")
    original = read_json(portfolio/"policy.json")
    for module in (frozen, frozen.engine, frozen.backtest, p6_settlements):
        source = Path(module.__file__)
        if hash_file(source) != original["implementation_sha256"][source.name]:
            raise ValueError("Original P6 running implementation differs from frozen file")
    if hash_file(Path(publish_day.__code__.co_filename)) != original["implementation_sha256"]["p6_portfolio_report.py"]:
        raise ValueError("Portfolio report implementation changed")
    if asdict(frozen.CONFIG) != original["config"]:
        raise ValueError("Original P6 configuration changed")
    core = dict(version=VERSION, research_only=True, usable_as_strategy_performance=False,
                usable_as_training_reward=False, diagnostic_only=diagnostic, config=original["config"],
                lanes=list(frozen.LANES), strategies=list(STRATEGIES),
                opening_cutoff="09:15 Asia/Shanghai", strict_policy_sha256=hash_file(portfolio/"policy.json"),
                source_sha256=hash_file(Path(__file__)), known_event_day="quarantine instrument at open",
                unknown_open_coverage="observe; ex-day buyer has no prior-date entitlement",
                closing_unknown="isolate unresolved rights on prior inventory; no estimated cash or shares")
    with exclusive(root):
        path = root/"policy.json"
        if path.exists():
            policy = read_json(path)
            if {k:v for k,v in policy.items() if k != "frozen_at"} != core:
                raise ValueError("Exploratory policy mismatch; use a new version root")
        else:
            policy = core | {"frozen_at":now_shanghai().isoformat()}
            atomic_json(path, policy)
    return policy


def witness(legacy, source_day, cutoff, track):
    """Accept only an archive actually committed, and evidence actually observed, before the cutoff."""
    if source_day:
        for kind in (("timely",) if track == "forward" else ("timely","historical")):
            path = legacy/kind/source_day/"manifest.json"
            if not path.is_file():
                continue
            record = read_json(path)
            ref = record.get("evidence_snapshot") or {}
            if (record["trade_date"] != source_day or not ref.get("root") or
                    parse_clock(record["committed_at"]) > cutoff or parse_clock(ref["observed_at_utc"]) > cutoff):
                continue
            evidence = frozen.artifact(Path(ref["root"])/"dataset","inference_evidence.json",ref["sha256"])
            saved = read_json(evidence)
            if (saved["policy"] != "P5_DUAL_OBSERVED_RESEARCH_WINDOW_V1"
                    or saved["observed_at_utc"] != ref["observed_at_utc"]):
                raise ValueError("Action evidence lineage mismatch")
            return saved["stocks"], {"manifest_sha256":hash_file(path), "evidence_sha256":ref["sha256"],
                                     "observed_at":ref["observed_at_utc"], "committed_at":record["committed_at"]}
    return {}, None


def action_status(stocks, day, scope):
    flags = frozen.action_flags(stocks,day,scope)
    # A positive event remains a warning even if dual-source interval coverage is incomplete.
    for code in scope:
        if any(a[0] == day for a in stocks.get(code, {}).get("actions", [])):
            flags[code] = {"status":"UNKNOWN", "available_at_open":False,
                           "coverage_unverified":flags.get(code,{}).get("status") == "MISSING_COVERAGE"}
    return flags


def close_actions(stocks, day, scope, receipts):
    flags = action_status(stocks, day, scope)
    for code in scope & receipts.keys():
        event = receipts[code]
        incomplete = flags.get(code,{}).get("status") == "MISSING_COVERAGE" or flags.get(code,{}).get("coverage_unverified",False)
        flags[code] = ({"status":"CONFLICTING_RECEIPTS"} if event.get("late_conflict")
                       or event["status"] != "VERIFIED" else event.copy())
        # Cash and share credits occur only after the opening decisions, including for early receipts.
        flags[code].update(available_at_open=False, coverage_unverified=incomplete)
    return flags


def opening_release(claims, prior_witness):
    retained, released = [], []
    for claim in claims:
        row = prior_witness.get(claim["stock_code"], {})
        certified = (claim["reason"] in {"MISSING_COVERAGE","MISSING_SAME_DAY_ACTION_COVERAGE"} and
                     row.get("covered_from", "9999") <= claim["day"] <= row.get("covered_through", "") and
                     not any(a[0] == claim["day"] for a in row.get("actions", [])))
        (released if certified else retained).append(claim)
    return retained, released


def day_marks(bars, day, scope, track, now):
    """A same-evening read may extend closing marks only. Never invent a missing opening fill."""
    missing = sorted(c for c in scope if not positive(bars.get(c,{}).get("close")))
    audit = {"requested":missing,"source":"FROZEN_STRICT_ACCOUNT_BARS","closing_marks_added":[],
             "missing_open":[c for c in sorted(scope) if not positive(bars.get(c,{}).get("open"))]}
    augmented = dict(bars)
    if missing and track == "forward" and now.date().isoformat() == day and now.hour >= 18:
        with frozen.AShareDailyLoader().engine.begin() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            if conn.scalar(text("show transaction_read_only")) != "on":
                raise ValueError("Supplementary closing marks require read-only access")
            rows = conn.execute(text("SELECT stock_code,close FROM public.market WHERE not_billing=1 "
                                     "AND trade_date=:day AND stock_code=ANY(:codes)"),
                                {"day":date.fromisoformat(day),"codes":missing}).all()
        actual = now_shanghai()
        if actual.date().isoformat() != day:
            raise ValueError("Supplementary marks crossed the session boundary")
        if len(rows) != len({row[0] for row in rows}):
            raise ValueError("Ambiguous supplementary closing prices")
        audit.update(source="SAME_EVENING_READ_ONLY_PUBLIC_MARKET_CLOSE",observed_at=actual.isoformat())
        for code, close in sorted(rows):
            if positive(close):
                augmented[code] = dict(augmented.get(code, {}),close=float(close))
                audit["closing_marks_added"].append(code)
    audit["unavailable_closes"] = [c for c in missing if c not in audit["closing_marks_added"]]
    return augmented,audit


def advance(state, day, prior, ranking, bars, strategy, opening, closing, bench_return):
    prior_claims = state["unknown_claims"]
    uncertain_inventory = {c["stock_code"] for c in prior_claims}
    blocked = uncertain_inventory | set(opening)
    # Exact later terms still require a known record-date quantity. The frozen
    # engine's incomplete VERIFIED branch records an unposted entitlement.
    effective_actions = dict(closing)
    for code in uncertain_inventory & closing.keys():
        if closing[code]["status"] == "VERIFIED":
            effective_actions[code] = {"status":"VERIFIED","available_at_open":False}
    quotes = dict(bars)
    for code in blocked & bars.keys():
        quotes[code] = bars[code] | {"can_buy":False,"can_sell":False}
    # P6 blocks every buy when its internal claims list is nonempty. Escrow the historical
    # claims here. Symbol-side restrictions above remain in effect throughout opening fills.
    result, daily, pos, trades, rejects = step(state | {"unknown_claims":[]}, day=day, prior_day=prior,
        ranked=ranking, bars=quotes, strategy=strategy, config=frozen.CONFIG, actions=effective_actions,
        benchmark_return=bench_return)
    new_claims = result["unknown_claims"]
    for claim in new_claims:
        holding = state["holdings"][claim["stock_code"]]
        claim.update(record_shares=holding["shares"], record_day=prior,
                     record_value=money(holding["shares"]*holding["mark"]))
        if claim["stock_code"] in uncertain_inventory:
            claim["record_quantity_unconfirmed"] = True
    result["unknown_claims"] = [*prior_claims,*new_claims]
    for rejected in rejects:
        if rejected["stock_code"] in blocked and rejected["reason"] == "OPEN_SIDE_BLOCKED_OR_UNVERIFIED":
            rejected["reason"] = "SYMBOL_ENTITLEMENT_QUARANTINE"
    acquired = {t["stock_code"] for t in trades if t["side"] == "buy"}
    valuation_watch = {code:closing[code]["status"] for code in acquired & closing.keys()
                       if closing[code]["status"] != "VERIFIED" or closing[code].get("coverage_unverified")}
    # An ex-day buyer does not receive distributions due to the prior record holder.
    affected = {c["stock_code"] for c in result["unknown_claims"]} | set(valuation_watch)
    marked = {p["stock_code"]:p["indicative_value"] for p in pos}
    risk = {"claims":result["unknown_claims"], "new_buyer_valuation_watch":valuation_watch,
            "marked_exposure":{code:marked.get(code,0.) for code in sorted(affected)}}
    risk["stale_close_marks"] = {p["stock_code"]:{"last_mark_date":p["last_mark_date"],
        "carried_value":p["indicative_value"],"reason":"MISSING_FORMAL_CLOSE"} for p in pos if p["stale"]}
    risk["reference_amount"] = money(sum(risk["marked_exposure"].values()))
    risk["asset_share"] = risk["reference_amount"]/result["equity"] if result["equity"] else None
    if affected:
        result.update(uncertain_history=True,last_valuation_uncertain=True)
        daily.update(valuation_status="EXPLORATORY_ACTION_UNCERTAINTY",nav=None,net_return=None,
                     daily_net_return=None,gross_return=None,excess_return=None,max_drawdown=None)
    daily.update(unresolved_claims=len(result["unknown_claims"]),affected_reference_amount=risk["reference_amount"],
                 affected_asset_share=risk["asset_share"],usable_as_training_reward=False)
    return result,daily,pos,trades,rejects,risk


def build_day(day, prior, strict_day, checkpoint, lists, legacy, track, settlements, now):
    cutoff = min(now,parse_clock(day+"T23:59:59.999999+08:00"))
    opening_time = parse_clock(day+"T09:15:00+08:00")
    op_stocks, op_ref = witness(legacy,prior,opening_time,track)
    close_stocks, close_ref = witness(legacy,day,cutoff,track)
    opening_terms, early_digests = p6_settlements.load_settlements(settlements,day,opening_time)
    closing_terms, close_digests = p6_settlements.load_settlements(settlements,day,cutoff)
    quotes = strict_day["bars"]
    old_bench = checkpoint["benchmark_weights"]
    targets, bench_sha = lists.get((prior,"equal_weight"),(None,None))
    planned = {c for lane in frozen.LANES for c in lists.get((prior,lane),([],None))[0][:frozen.CONFIG.entry_rank]}
    held = {c for a in checkpoint["accounts"].values() for c in a["holdings"]}
    roles = {"held":sorted(held),"possible_buys":sorted(planned),"previous_benchmark":sorted(old_bench),
             "target_benchmark":sorted(targets or [])}
    scope = set().union(*roles.values())
    quotes,price_audit = day_marks(quotes,day,scope,track,now)
    op_flags = action_status(op_stocks,day,scope)
    opening = {c for c,f in op_flags.items() if f["status"] != "MISSING_COVERAGE"} | (scope & opening_terms.keys())
    closing = close_actions(close_stocks,day,scope,closing_terms)
    weights, ret, known = frozen.benchmark(old_bench,targets,quotes,closing)
    allocation = set(targets) if targets is not None else set(old_bench)
    if not checkpoint["benchmark_composition_known"] or allocation & closing.keys():
        ret = None
    checkpoint["benchmark_weights"] = weights
    checkpoint["benchmark_composition_known"] = known and (checkpoint["benchmark_composition_known"] or targets is not None)
    risks = {}
    results = {key:[] for key in ("equity","positions","trades","rejections")}
    parents = {}
    for lane in frozen.LANES:
        ranking, parents[lane] = lists.get((prior,lane),(None,None))
        for strategy in STRATEGIES:
            key = lane+"/"+strategy
            state = checkpoint["accounts"].get(key) or initial_state(frozen.CONFIG)
            claims,released = opening_release(state["unknown_claims"],op_stocks)
            state = state | {"unknown_claims":claims}
            state,daily,pos,trades,rejects,risks[key] = advance(state,day,prior,ranking,quotes,strategy,opening,closing,ret)
            risks[key]["opening_released_by_subsequent_coverage"] = released
            checkpoint["accounts"][key] = state
            for name, rows in zip(results,([daily],pos,trades,rejects),strict=True):
                results[name].extend(row | {"model":lane,"strategy":strategy,"trade_date":day} for row in rows)
    inputs = {"signal_day":prior,"rankings_manifest_sha256":parents,"benchmark_manifest_sha256":bench_sha,
              "bars":quotes,"actions":closing,"benchmark_daily_return":ret,
              "opening_cutoff":opening_time.isoformat(),"closing_cutoff":cutoff.isoformat(),
              "opening_witness":op_ref,"closing_witness":close_ref,"opening_receipts":early_digests,
              "closing_receipts":close_digests,"opening_blocked":sorted(opening),"opening_flags":op_flags,
              "price_coverage":price_audit}
    return inputs,results,{"roles":roles,"by_account":risks}


def check_schedule(first, end, source_days):
    from .p66_calendar import planned_open
    day, stop = date.fromisoformat(first),date.fromisoformat(end)
    while day <= stop:
        expected = planned_open(day)
        if expected is None:
            return day.isoformat(),"CALENDAR_UNCONFIRMED"
        if expected and day.isoformat() not in source_days:
            return day.isoformat(),"AWAITING_FORMAL_ACCOUNT_DAY"
        day += timedelta(days=1)
    return None,None


def run(root,portfolio,legacy,panel,group,end,start=None):
    policy = read_json(root/"policy.json")
    diagnostic = start is not None
    if policy["diagnostic_only"] != diagnostic:
        raise ValueError("Diagnostic and prospective books must use independent policies")
    freeze(root,portfolio,diagnostic)
    now = now_shanghai()
    if end > now.date().isoformat() or (end == now.date().isoformat() and now.hour < 18 and diagnostic):
        raise ValueError("End is beyond the formal closing boundary")
    track = "diagnostic" if diagnostic else "forward"
    track_dir = "diagnostic-"+start if diagnostic else "forward"
    folder,source = root/track_dir,portfolio/track_dir
    frozen_at = parse_clock(policy["frozen_at"])
    lists = frozen.archived_lists(panel,legacy,group,track,frozen_at,end)
    first = start if diagnostic else min((d for d,lane in lists if lane in frozen.LANES), default=None)
    if first is None:
        return {"status":"AWAITING_FIRST_POST_FREEZE_SIGNAL","frozen_at":policy["frozen_at"],"new_days":[]}
    if first > end:
        raise ValueError("Invalid date range")
    with exclusive(folder):
        source_days = sorted(p.name for p in source.glob("????-??-??") if p.name <= end and p.is_dir())
        gap,reason = check_schedule(first,end,set(source_days))
        checkpoint = {"accounts":{},"benchmark_weights":{},"benchmark_composition_known":True}
        previous,strict_previous,last = None,None,None
        policy_sha = hash_file(root/"policy.json")
        published = []
        if any(p.name not in source_days for p in folder.glob("????-??-??") if p.name <= end):
            raise ValueError("Previously archived input day disappeared")
        for day in source_days:
            if gap and day >= gap:
                break
            path = source/day
            _, strict_previous = frozen.checked_day(path,strict_previous,policy["strict_policy_sha256"])
            if day < first:
                continue
            dest = folder/day
            if dest.exists():
                checkpoint,previous = frozen.checked_day(dest,previous,policy_sha)
                if read_json(dest/"inputs.json")["strict_day_sha256"] != strict_previous:
                    raise ValueError("Previously consumed strict account day changed")
            else:
                inputs,results,risks = build_day(day,last,read_json(path/"inputs.json"),checkpoint,lists,
                                                 legacy,track,portfolio/"settlements",now)
                inputs["strict_day_sha256"] = strict_previous
                timing = ("HISTORICAL_DIAGNOSTIC" if diagnostic else
                          "SAME_DAY_SIMULATION" if day == now.date().isoformat() else "RECOVERED_SIMULATION")
                with TemporaryDirectory(dir=folder) as tmp:
                    stage = Path(tmp)/"day"
                    stage.mkdir()
                    atomic_json(stage/"inputs.json",inputs)
                    atomic_json(stage/"checkpoint.json",checkpoint)
                    atomic_json(stage/"exploratory_risks.json",risks)
                    publish_day(stage,day,track,timing,results,policy_sha,previous,now)
                    os.rename(stage,dest)
                checkpoint,previous = frozen.checked_day(dest,previous,policy_sha)
                published.append(day)
            last = day
        return {"status":reason or "OK","missing_session":gap,"track":track,"new_days":published,
                "last_day":last,"latest_summary_sha256":previous}


def revision(root, account_track, day, receipts, target):
    """Independent as-of sensitivity to confirmed ordinary cash terms; retain original fills."""
    now = now_shanghai()
    if target.resolve().is_relative_to(root.resolve()) or target.resolve().is_relative_to(receipts.resolve()):
        raise ValueError("Revision output must be outside source archives")
    chain = None
    original_actions = {}
    folder = root/account_track
    for archive in sorted(p for p in folder.glob("????-??-??") if p.name <= day):
        state,chain = frozen.checked_day(archive,chain,hash_file(root/"policy.json"))
        original_actions[archive.name] = read_json(archive/"inputs.json")["actions"]
    if chain is None or archive.name != day:
        raise ValueError("Missing valuation day")
    accounts = {}
    for key, s in state["accounts"].items():
        amounts = []
        recognized = set()
        for item in s["unknown_claims"]:
            terms, sources = receipt_evidence(receipts,item["day"],item["stock_code"],now)
            eligible = [v for v in sources if v["record_date"] == item["record_day"]]
            known_cash = None
            entitlement = (item["day"],item["stock_code"])
            at_close = original_actions[item["day"]].get(item["stock_code"],{})
            if (item["reason"] == "UNKNOWN" and at_close.get("status") == "UNKNOWN"
                    and not item.get("record_quantity_unconfirmed")
                    and len(terms) == 1 and eligible
                    and all(v["cash_basis"] == "VERIFIED_ACCOUNT_CREDIT_PER_OLD_SHARE"
                            and float(v["share_multiplier"]) == 1 for v in eligible)):
                if entitlement in recognized:
                    raise ValueError("Duplicate historical entitlement in revision")
                recognized.add(entitlement)
                known_cash = money(item["record_shares"]*float(eligible[0]["cash_per_share"]))
            amounts.append(item | {"verified_cash_scenario":known_cash,"receipts":sources})
        extra = money(sum(v["verified_cash_scenario"] or 0 for v in amounts))
        accounts[key] = {"original_equity":s["equity"],"verified_cash_scenario_total":extra,
                         "revised_indicative_equity":money(s["equity"]+extra),"claims":amounts,
                         "changes_spendable_cash":False}
    with exclusive(target.parent):
        if target.exists():
            raise FileExistsError(target)
        atomic_json(target,{"observed_at":now.isoformat(),"original_day_sha256":chain,
                            "day":day,"track":account_track,"accounts":accounts,"research_only":True})
    return {"revision":str(target),"sha256":hash_file(target)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command",required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--out",type=Path,required=True)
    f.add_argument("--portfolio",type=Path,required=True)
    f.add_argument("--diagnostic-only",action="store_true")
    r = sub.add_parser("run")
    for name in ("out","portfolio","legacy","panel","group"):
        r.add_argument("--"+name,type=Path,required=True)
    for name in ("end","start"):
        r.add_argument("--"+name,type=date.fromisoformat,required=name=="end")
    v = sub.add_parser("revise")
    for name in ("out","receipts","target"):
        v.add_argument("--"+name,type=Path,required=True)
    v.add_argument("--track-dir",required=True)
    v.add_argument("--day",type=date.fromisoformat,required=True)
    a = cli.parse_args()
    if a.command == "freeze":
        print(freeze(a.out,a.portfolio,a.diagnostic_only))
    elif a.command == "run":
        print(run(a.out,a.portfolio,a.legacy,a.panel,a.group,a.end.isoformat(),a.start.isoformat() if a.start else None))
    else:
        print(revision(a.out,a.track_dir,a.day.isoformat(),a.receipts,a.target))


if __name__ == "__main__":
    main()
