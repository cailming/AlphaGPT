"""Read-only cross-day reconciliation of archived research accounts, independent of the engine."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dec(value):
    number = Decimal(str(value))
    require(number.is_finite(), "Nonfinite money or quantity")
    return number


def cents(value):
    return dec(value).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)


def integer(value):
    result = dec(value)
    require(result == int(result) and result >= 0, "Negative or fractional shares")
    return int(result)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def table(folder, suffix):
    groups = defaultdict(list)
    with (folder / f"portfolio_{suffix}.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            require(row["trade_date"] == folder.name, "CSV date differs from its archive")
            groups[row["model"] + "/" + row["strategy"]].append(row)
    return groups


def same_metric(actual, expected):
    return abs(dec(actual) - dec(expected)) <= Decimal("1e-10")


def reconcile_account(day, equity, positions, trades, rejections, config, prior, current, actions):
    initial = cents(config["initial_cash"])
    cash = cents(prior["cash"]) if prior else initial
    fees = cents(prior["fees"]) if prior else Decimal(0)
    slippage = cents(prior["slippage"]) if prior else Decimal(0)
    require(cash == cents(equity["opening_cash"]), "Opening cash differs from previous close")
    old = prior["holdings"] if prior else {}
    inventory = Counter({k:integer(v["shares"]) for k,v in old.items()})
    require(len(positions) == len({p["stock_code"] for p in positions}), "Duplicate equity position")
    require(len(positions) == int(equity["positions"]), "Position count mismatch")
    require(len(positions) <= config["max_positions"], "Exceeded maximum inventory count")
    require(len(trades) == int(equity["trade_count"]) and len(rejections) == int(equity["rejections"]),
            "Trade or refusal counts disagree")
    for code, position in old.items():
        if position["pending_shares"] and position["share_list_date"] <= day:
            inventory[code] += integer(position["pending_shares"])
        elif code in actions and actions[code].get("status") == "VERIFIED":
            event = actions[code]
            if (event.get("available_at_open", True) and event.get("share_list_date") == day
                    and not position["pending_shares"] and event.get("record_date") in (None, prior["last_day"])):
                bonus = dec(position["shares"]) * (dec(event["share_multiplier"]) - 1)
                if bonus == int(bonus):
                    inventory[code] += int(bonus)

    sellable = inventory.copy()
    for trade in trades:
        code, side = trade["stock_code"], trade["side"]
        require(trade["signal_date"] < day and bool(trade["signal_date"]), "Same-day or future trading signal")
        require(trade["fill_assumption"] == "FORMAL_OPEN_SIMULATION", "Missing opening execution assumption")
        qty = integer(trade["shares"])
        require(qty > 0 and side in ("buy", "sell") and dec(trade["price"]) > 0, "Invalid fill")
        gross = cents(qty * dec(trade["price"]))
        cost = max(dec(config["min_commission"]), gross * dec(config["commission_rate"]))
        cost += gross * dec(config["transfer_rate"])
        if side == "sell":
            cost += gross * dec(config["stamp_rate_sell"])
            require(qty <= sellable[code], "T+0 or oversold shares")
            sellable[code] -= qty
            inventory[code] -= qty
            cash += gross
        else:
            inventory[code] += qty
            cash -= gross
        require(cents(trade["gross"]) == gross, "Fill gross differs from shares × price")
        fee = cents(trade["fees"])
        require(fee == cents(cost), "Fee differs from frozen schedule")
        fees += fee
        slip = cents(abs(dec(trade["price"]) - dec(trade["opening_price"])) * qty)
        require(slip == cents(trade["slippage"]), "Slippage differs from opening reference")
        slippage += slip
        cash -= fee
        require(cash >= 0, "Negative settled funds after execution")
    cash += cents(equity["action_cash_paid"])
    require(cash == cents(equity["closing_cash"]) == cents(current["cash"]), "Closing cash mismatch")
    require(fees == cents(equity["cumulative_fees"]) == cents(current["fees"]), "Cumulative fee mismatch")
    require(slippage == cents(equity["cumulative_slippage"]) == cents(current["slippage"]), "Cumulative slippage mismatch")
    observed = {p["stock_code"]:integer(p["shares"]) for p in positions}
    require({k:v for k,v in inventory.items() if v} == observed, "Cross-day share balance mismatch")
    require(observed == {k:integer(v["shares"]) for k,v in current["holdings"].items()}, "Checkpoint inventory mismatch")
    marked = Decimal(0)
    for p in positions:
        code = p["stock_code"]
        units = observed[code] + integer(p["pending_shares"])
        value = cents(dec(p["mark"]) * units)
        require(cents(p["indicative_value"]) == value, "Marked stock value mismatch")
        marked += value
    receivable = sum((cents(x["amount"]) for x in current["cash_due"]), Decimal(0))
    old_due = sum((cents(x["amount"]) for x in (prior or {}).get("cash_due", ())), Decimal(0))
    recognized = Decimal(0)
    for code, position in old.items():
        event = actions.get(code, {})
        if (event.get("status") == "VERIFIED" and event.get("cash_per_share")
                and event.get("record_date") in (None, prior["last_day"])):
            # An unresolved prior share grant can make a subsequent entitlement incalculable.
            if not any(claim["stock_code"] == code and claim["day"] == day and claim["reason"] == "VERIFIED"
                       for claim in current["unknown_claims"]):
                recognized += cents(dec(position["shares"]) * dec(event["cash_per_share"]))
    require(old_due + recognized == receivable + cents(equity["action_cash_paid"]),
            "Cash entitlement creation or payment mismatch")
    require(receivable == cents(equity["dividend_receivable"]) and marked == cents(equity["holdings_value"]),
            "Receivables or stock marks disagree")
    total = cash + marked + receivable
    require(total == cents(equity["equity"]) == cents(current["equity"]), "End equity mismatch")
    require(same_metric(equity["indicative_nav"], total / initial), "Indicative NAV mismatch")
    if equity["valuation_status"] == "SIMULATED_CURRENT_MARKS":
        require(same_metric(equity["nav"], total / initial), "NAV mismatch")
        require(same_metric(equity["net_return"], total / initial - 1), "Net return mismatch")
    else:
        require(equity["nav"] == equity["net_return"] == "", "Unconfirmed valuation published a NAV")
    return len(trades)


def reconcile(folder, expected=None):
    policy_path = folder.parent / "policy.json"
    policy, policy_hash = read_json(policy_path), digest(policy_path)
    days = sorted(p for p in folder.glob("????-??-??") if p.is_dir())
    require(days, "No archived account days")
    roster = {l+"/"+s for l in policy["lanes"] for s in policy["strategies"]}
    chain = None
    prior = {}
    fills = refusals = 0
    for path in days:
        day = date.fromisoformat(path.name).isoformat()
        summary_path = path / "portfolio_summary.json"
        summary = read_json(summary_path)
        require(summary["day"] == day and summary["policy_sha256"] == policy_hash
                and summary["previous_sha256"] == chain, "Broken simulation archive chain")
        expected_files = {"checkpoint.json", "inputs.json", "portfolio_daily_equity.csv", "portfolio_positions.csv",
                          "portfolio_trades.csv", "portfolio_rejections.csv", "portfolio_daily.md"}
        require(expected_files.issubset(summary["files"]), "Day lacks required artifact digest")
        for name, sha in summary["files"].items():
            require(Path(name).name == name and sha == digest(path / name), "Archive artifact digest mismatch")
        chain = digest(summary_path)
        equity, positions, trades, rejections = (table(path, suffix) for suffix in
                                                 ("daily_equity","positions","trades","rejections"))
        current = read_json(path / "checkpoint.json")["accounts"]
        actions = read_json(path / "inputs.json")["actions"]
        require(set(current) == set(equity) == roster, "Wrong account roster")
        require(all(set(t).issubset(roster) for t in (positions,trades,rejections)), "Unrecognized account")
        summary_accounts = {r["model"]+"/"+r["strategy"]:r for r in summary["accounts"]}
        require(set(summary_accounts) == roster and len(summary["accounts"]) == len(roster), "Summary roster mismatch")
        require(summary["trade_count"] == sum(map(len, trades.values())), "Summary trade count mismatch")
        require(summary["rejection_reasons"] == dict(Counter(
            row["reason"] for group in rejections.values() for row in group)), "Summary rejection counts mismatch")
        for account in sorted(roster):
            require(len(equity[account]) == 1, "Duplicate account-day")
            for key in ("opening_cash", "closing_cash", "equity", "cumulative_fees", "indicative_nav"):
                require(same_metric(equity[account][0][key], summary_accounts[account][key]),
                        f"{day} {account}: daily summary {key} differs from CSV")
            try:
                fills += reconcile_account(day, equity[account][0], positions[account], trades[account], rejections[account],
                                           policy["config"], prior.get(account), current[account], actions)
            except ValueError as exc:
                raise ValueError(f"{day} {account}: {exc}") from exc
            refusals += len(rejections[account])
        prior = current
    if expected is not None:
        require(chain == expected, "Archives changed since previous run")
    return {"status":"RECONCILED", "directory":str(folder), "first":days[0].name, "last":days[-1].name,
            "days":len(days), "accounts_per_day":len(roster), "account_days":len(days)*len(roster),
            "fills":fills, "rejections":refusals, "archive_chain_sha256":chain,
            "matched_prior_run":expected is not None}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    cli.add_argument("--expect-sha", help="Verify this same terminal chain after a duplicate simulation invocation")
    args = cli.parse_args()
    print(json.dumps(reconcile(args.directory, args.expect_sha), sort_keys=True))


if __name__ == "__main__":
    main()
