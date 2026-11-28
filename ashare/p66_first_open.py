"""Review the first frozen P6 decision and verify its eventual archived fills."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from math import ceil
from pathlib import Path

from . import backtest, p6_portfolio, p6_portfolio_report, p6_portfolio_run, p6_settlements
from .p4_action_pipeline import atomic_json, exclusive, read_json
from .p4_subset_inputs import hash_file
from .p55_daily import now_shanghai, parse_clock
from .p55_reports import artifact
from .p6_portfolio import PortfolioConfig, money, open_quote, positive, trading_unit
from .p6_portfolio_reconcile import reconcile, require, table
from .p6_portfolio_run import archived_lists, checked_day
from .p66_calendar import next_session

SIGNAL = "2026-11-27"


def account_plan(account, balances, ranking, config):
    lane, strategy = account.split("/")
    require(balances["cash"] == balances["equity"] == config.initial_cash
            and balances["session"] == 1 and balances["last_day"] == SIGNAL and not balances["fees"]
            and not any(balances[k] for k in ("holdings","unknown_claims","cash_due")),
            "Expected clean first-day cash checkpoint")
    cap = ceil(config.max_positions/config.holding_sessions) if strategy == "A" else config.max_positions
    return {"model":lane, "strategy":strategy, "opening_cash":balances["cash"],
            "ordered_candidate_codes":ranking[:config.entry_rank], "max_new_positions":min(len(ranking),cap),
            "per_name_budget_including_fees":money(config.initial_cash*config.max_weight),
            "status":"WAIT_FOR_FORMAL_OPEN" if ranking else "NO_SCORES_STAY_IN_CASH"}


def write_once(path, data):
    with exclusive(path.parent):
        if path.exists():
            stored = read_json(path)
            require({k:v for k,v in stored.items() if k != "generated_at"} == data, "Existing audit differs")
        else:
            atomic_json(path, data | {"generated_at": now_shanghai().isoformat()})
    return path


def audit_fills(plan, trades, bars, config):
    require(len(trades) <= plan["max_new_positions"], "First-session entry limit exceeded")
    previous_rank = -1
    for t in trades:
        code = t["stock_code"]
        ranking = plan["ordered_candidate_codes"]
        require(t["side"] == "buy" and t["signal_date"] == SIGNAL and code in ranking,
                "Fill did not come from the frozen buy roster")
        rank = ranking.index(code)
        require(rank > previous_rank, "First-session purchases do not follow frozen rank")
        previous_rank = rank
        unit = trading_unit(code)
        qty = int(t["shares"])
        require(unit and unit[0] <= qty <= unit[2] and (qty-unit[0]) % unit[1] == 0, "Invalid buy unit")
        quote, reason = open_quote(bars.get(code), "buy", config)
        require(quote and quote == (float(t["opening_price"]),float(t["price"])),
                f"Archived fill differs from the formal open and slippage: {reason}")
        require(money(float(t["gross"])+float(t["fees"])) <= plan["per_name_budget_including_fees"],
                "Trade exceeds the first-open cash budget")


def audit_close(positions, bars, nav):
    for pos in positions:
        close = positive(bars.get(pos["stock_code"], {}).get("close"))
        if close is None:
            require(pos["stale"] == "True" and nav is None, "Missing close marked as confirmed")
        else:
            require(pos["stale"] == "False" and float(pos["mark"]) == close, "Incorrect closing price")


def run(root, legacy, panel):
    group = root/"group_c"
    portfolio = root/"portfolio"
    folder = portfolio/"forward"
    out = root/"p66/first-open"
    policy = read_json(portfolio/"policy.json")
    config = PortfolioConfig(**policy["config"])
    require(asdict(config) == policy["config"], "Unknown account configuration")
    for module in (backtest, p6_portfolio, p6_portfolio_run, p6_portfolio_report, p6_settlements):
        source = Path(module.__file__)
        require(hash_file(source) == policy["implementation_sha256"][source.name], "Frozen implementation changed")
    cutoff = next_session(date.fromisoformat(SIGNAL))
    require(cutoff is not None, "Target session calendar unavailable")
    path = out/"first_open_plan.json"
    if not path.exists() and now_shanghai() >= parse_clock(cutoff.isoformat()+"T09:15:00+08:00"):
        raise ValueError("The initial preparation window has closed")
    lists = archived_lists(panel,legacy,group,"forward",parse_clock(policy["frozen_at"]),SIGNAL)
    capture = {}
    for ledger in (legacy,panel,group):
        manifest = ledger/"timely"/SIGNAL/"manifest.json"
        rec = read_json(manifest)
        for name, spec in rec["models"].items():
            ordered, digest = lists[SIGNAL,name]
            require(digest == hash_file(manifest) and len(ordered) == spec["rows"], "Wrong prospective source")
            artifact(manifest.parent,spec["predictions_file"],spec["predictions_sha256"])
            capture[name] = {"committed_at":rec["committed_at"], "version":spec["model_version"], "count":len(ordered),
                             "manifest_sha256":digest, "predictions_sha256":spec["predictions_sha256"],
                             "weights_sha256":spec.get("weights_sha256")}
    policy_sha = hash_file(portfolio/"policy.json")
    states, initial_sha = checked_day(folder/SIGNAL,None,policy_sha)
    expected = {lane+"/"+s for lane in policy["lanes"] for s in policy["strategies"]}
    require(set(states["accounts"]) == expected, "Incorrect initial account roster")
    plans = {k:account_plan(k, states["accounts"][k],lists[SIGNAL,k.split("/")[0]][0], config)
             for k in sorted(expected)}
    write_once(path, {"track":"PRE_OPEN_PLAN_ONLY", "signal_date":SIGNAL, "target_planned_session":str(cutoff),
                      "policy_sha256":policy_sha,"opening_checkpoint_summary_sha256":initial_sha,
                      "sources":capture,"accounts":plans})
    settled = folder/str(cutoff)
    if not settled.exists():
        late = now_shanghai() >= parse_clock(str(cutoff)+"T23:59:59+08:00")
        return {"plan":str(path),"planned_session":str(cutoff),
                "settlement":"TARGET_ACCOUNT_DAY_MISSING" if late else "AWAITING_REAL_FORMAL_ACCOUNT"}
    _, terminal = checked_day(settled,initial_sha,policy_sha)
    inputs = read_json(settled/"inputs.json")
    require(inputs["signal_day"] == SIGNAL, "Unexpected first trade signal")
    require(all(inputs["rankings_manifest_sha256"][lane] == capture[lane]["manifest_sha256"]
                for lane in policy["lanes"]), "Unexpected ranking source on first trade day")
    actual = table(settled,"trades")
    positions = table(settled,"positions")
    accounts = {}
    summary = read_json(settled/"portfolio_summary.json")
    for row in summary["accounts"]:
        k = row["model"]+"/"+row["strategy"]
        audit_fills(plans[k],actual[k],inputs["bars"],config)
        audit_close(positions[k],inputs["bars"],row["nav"])
        accounts[k] = {**{key:row[key] for key in ("closing_cash","positions","nav","indicative_nav",
                         "valuation_status","trade_count","rejections","cumulative_fees")}, "buys":actual[k]}
    require(set(accounts)==expected, "Missing account settlement")
    result = out/"first_open_settlement_review.json"
    if not result.exists():
        reconcile(folder)
    write_once(result, {"track":"ARCHIVED_SIMULATION_REVIEW","day":str(cutoff),"accounts":accounts,
                        "account_summary_sha256":terminal, "source_plan_sha256":hash_file(path)})
    return {"plan":str(path),"settlement":str(result)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root","legacy","panel"):
        p.add_argument("--"+name, type=Path, required=True)
    args=p.parse_args()
    print(run(args.root,args.legacy,args.panel))
