"""Deterministic, JSON-checkpointable P6 simulation of archived T-close rankings.

Only the runner supplies the previous formal session's archived list. Its production
access is read-only. This module has no connection to a broker or database.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import math

from .backtest import _money_cost

VERSION = "P6_OPEN_PORTFOLIO_V1"
STRATEGIES = ("A", "B", "C")


@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float = 1_000_000.0
    max_positions: int = 20
    max_weight: float = 0.05
    holding_sessions: int = 5
    entry_rank: int = 20
    buffer_rank: int = 40
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_rate_sell: float = 0.0005
    transfer_rate: float = 0.00001
    slippage_bps: float = 5.0
    # Strict inner band for proxy fills when official limits are unavailable.
    # Applies symmetrically; this is an execution assumption, not a limit price.
    proxy_open_gap: float = 0.04

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in asdict(self).values())
                or self.initial_cash < .01 or not 0 < self.max_weight <= 1
                or any(type(v) is not int or v < 1 for v in
                       (self.max_positions, self.holding_sessions, self.entry_rank, self.buffer_rank))
                or self.entry_rank > self.buffer_rank or not 0 < self.proxy_open_gap < .05
                or not 0 <= self.slippage_bps < 1000 or min(self.commission_rate,
                    self.min_commission, self.stamp_rate_sell, self.transfer_rate) < 0):
            raise ValueError("Invalid research account configuration")


def money(value):
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except (ValueError, TypeError):
        return None


def positive(value):
    value = number(value)
    return value if value is not None and value > 0 else None


def iso(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Expected ISO formal session date")
    return value


def trading_unit(code):
    """Research scope is ordinary A shares. Return minimum, step, order cap."""
    if len(code) != 6 or not code.isascii() or not code.isdigit():
        return None
    if code.startswith("688"):
        return 200, 1, 100000
    if code.startswith(("300", "301")):
        return 100, 100, 300000
    if code.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return 100, 100, 1000000
    if code.startswith(("920", "83", "87", "43")):
        return 100, 1, 1000000
    return None


def initial_state(config):
    return dict(version=VERSION, config=asdict(config), strategy=None, last_day=None,
                session=0, first_decision=None, cash=money(config.initial_cash), holdings={},
                cash_due=[], fees=0., slippage=0., equity=money(config.initial_cash),
                peak=money(config.initial_cash), drawdown=0., benchmark_index=1.,
                uncertain_history=False, unknown_claims=[])


def open_quote(bar, side, cfg):
    if side not in ("buy", "sell"):
        raise ValueError("Unknown simulation side")
    if not bar:
        return None, "MISSING_FORMAL_BAR"
    # Optional explicit fields are fail-closed; absence enables only the proxy.
    if "suspended" in bar and bar["suspended"] is not False:
        return None, "SUSPENDED_OR_UNVERIFIED"
    flag = "can_" + side
    if flag in bar and bar[flag] is not True:
        return None, "OPEN_SIDE_BLOCKED_OR_UNVERIFIED"
    op = positive(bar.get("open"))
    if not op:
        return None, "MISSING_OPEN"
    fill = money(op * (1 + (1 if side == "buy" else -1) * cfg.slippage_bps / 10000))
    if fill <= 0:
        return None, "INVALID_SLIPPED_PRICE"
    if "upper_limit" in bar or "lower_limit" in bar:
        hi, lo = positive(bar.get("upper_limit")), positive(bar.get("lower_limit"))
        if not hi or not lo or lo >= hi or not lo <= op <= hi:
            return None, "INVALID_OPEN_LIMIT_EVIDENCE"
        if side == "buy" and (op >= hi or fill > hi):
            return None, "UP_LIMIT_BUY"
        if side == "sell" and (op <= lo or fill < lo):
            return None, "DOWN_LIMIT_SELL"
    else:
        ref = positive(bar.get("pre_close"))
        previous_close = positive(bar.get("prior_close"))
        if ref and previous_close and abs(ref - previous_close) > 0.010001:
            return None, "CONFLICTING_OPEN_LIMIT_REFERENCE"
        if not ref:
            ref = previous_close
        if not ref:
            return None, "MISSING_OPEN_LIMIT_REFERENCE"
        # A cent of extra protection handles low-price tick rounding.
        if abs(fill - ref) + .01 >= cfg.proxy_open_gap * ref:
            return None, "OPEN_OUTSIDE_CONSERVATIVE_LIMIT_PROXY"
    return (op, fill), None


def add_action(s, code, item, held, day):
    """Book an ex-date entitlement strictly against the prior close's shares."""
    shares = held[code]
    if not shares:
        return
    cash_rate = number(item.get("cash_per_share"))
    multiplier = number(item.get("share_multiplier"))
    cash_day, list_day = item.get("cash_pay_date"), item.get("share_list_date")
    valid = (item.get("status") == "VERIFIED" and cash_rate is not None and cash_rate >= 0
             and multiplier is not None and multiplier >= 1)
    if valid:
        extra = float(shares * (multiplier - 1))
        try:
            valid = (math.isclose(extra, round(extra), rel_tol=0, abs_tol=1e-7)
                     and (not cash_rate or iso(cash_day) >= day)
                     and (not round(extra) or iso(list_day) >= day))
        except (TypeError, ValueError):
            valid = False
    pos = s["holdings"].get(code)
    if not valid or (pos and pos["pending_shares"]) or (pos is None and round(extra)):
        s["unknown_claims"].append({"day": day, "stock_code": code,
                                    "reason": item.get("status", "UNVERIFIED_ACTION")})
        return
    if cash_rate:
        # Pay-date cash becomes deployable at session end; intraday time is unknown.
        s["cash_due"].append(dict(stock_code=code, pay_date=cash_day, amount=money(cash_rate * shares)))
    if round(extra):
        pos["pending_shares"], pos["share_list_date"] = round(extra), list_day
    # Preserve the pre-open marked economic value across a known distribution.
    if pos is not None:
        pos["mark"] = max(0., (pos["mark"] - cash_rate) / multiplier)


def step(state, *, day, prior_day, ranked, bars, strategy, config, actions=None, benchmark_return=None):
    """Advance exactly one exchange session. The input checkpoint is never mutated.

    day/prior_day are consecutive exchange calendar dates provided by the runner;
    None ranking is a missing forecast and [] is an archived zero prediction.
    bars only needs open/close and trusted opening limit levels or reference.
    Unknown or late ex-date terms are recorded against the prior-close inventory.
    """
    iso(day)
    if prior_day is not None and iso(prior_day) >= day:
        raise ValueError("Rankings must precede the simulated opening")
    if (strategy not in STRATEGIES or state["version"] != VERSION or state["config"] != asdict(config)
            or state["strategy"] not in (None, strategy)
            or (state["last_day"] is not None and state["last_day"] != prior_day)
            or (ranked is not None and (prior_day is None or not isinstance(ranked, list)
                 or len(set(ranked)) != len(ranked) or not all(isinstance(c, str) for c in ranked)))):
        raise ValueError("Incompatible checkpoint, duplicate session, or invalid prior list")
    s = deepcopy(state)
    held = {c: p["shares"] for c, p in s["holdings"].items()}
    actions = {code: ((item | {"status": "UNCONFIRMED_RECORD_DATE"})
                     if item.get("record_date") and item["record_date"] != prior_day else item)
               for code, item in (actions or {}).items()}
    holdings = s["holdings"]
    day_no = s["session"] + 1
    cash_start, equity_start = s["cash"], s["equity"]
    trades, rejects = [], []
    action_cash_paid = 0.
    shares_delivered = {}
    sold = set()

    def reject(code, side, reason):
        rejects.append(dict(trade_date=day, stock_code=code, side=side, reason=reason, signal_date=prior_day))

    def record(code, side, shares, quote, fees, decision):
        op, fill = quote
        value = money(shares * fill)
        slip = money(abs(fill - op) * shares)
        s["cash"] = money(s["cash"] + (value if side == "sell" else -value) - fees)
        s["fees"] = money(s["fees"] + fees)
        s["slippage"] = money(s["slippage"] + slip)
        trades.append(dict(trade_date=day, signal_date=decision, stock_code=code, side=side,
                           shares=shares, opening_price=op, price=fill, gross=value, fees=fees,
                           slippage=slip, fill_assumption="FORMAL_OPEN_SIMULATION"))
        if s["cash"] < 0:
            raise ArithmeticError("Negative settled cash")

    for code in sorted(held):
        item = actions.get(code)
        if item and item.get("available_at_open", True):
            add_action(s, code, item, held, day)
        pos = holdings[code]
        if pos["pending_shares"] and pos["share_list_date"] <= day:
            shares_delivered[code] = pos["pending_shares"]
            pos["shares"] += pos["pending_shares"]
            pos["pending_shares"], pos["share_list_date"] = 0, None

    # Anchor the fixed cycle to the first decision session, even for an empty lane.
    if prior_day and s["first_decision"] is None:
        s["first_decision"] = day_no
    cycle = s["first_decision"] is not None and (day_no - s["first_decision"]) % config.holding_sessions == 0
    active_ranking = bool(ranked)
    entry = (ranked or [])[:config.entry_rank]
    retain = set((ranked or [])[:config.buffer_rank] if strategy == "B" else entry)
    for code, pos in sorted(list(holdings.items())):
        due = (strategy == "A" and day_no >= pos["entry_session"] + config.holding_sessions
               or active_ranking and code not in retain and (strategy == "B" or strategy == "C" and cycle))
        if due and not pos["exit_pending"]:
            pos["exit_pending"] = prior_day or day
        if not pos["exit_pending"]:
            continue
        open_issues = any(event["stock_code"] == code for event in s["unknown_claims"])
        # For a late ex-date event, do not let evening evidence cancel a morning sale.
        if open_issues:
            reject(code, "sell", "UNRESOLVED_ENTITLEMENT")
            continue
        if pos["pending_shares"] or pos["entry_session"] >= day_no:
            reject(code, "sell", "AWAITING_LISTING_OR_T_PLUS_ONE")
            continue
        quote, reason = open_quote(bars.get(code), "sell", config)
        if not quote:
            reject(code, "sell", reason)
            continue
        value = money(pos["shares"] * quote[1])
        fees = money(_money_cost(value, config, "sell"))
        if s["cash"] + value < fees:
            reject(code, "sell", "INSUFFICIENT_SETTLEMENT_CASH")
            continue
        record(code, "sell", pos["shares"], quote, fees, pos["exit_pending"])
        sold.add(code)
        del holdings[code]

    cash_due = sum(claim["amount"] for claim in s["cash_due"])
    def open_mark(code, pos):
        return positive((bars.get(code) or {}).get("open")) or pos["mark"]
    marked_at_open = cash_due + s["cash"] + sum(
        open_mark(code, pos) * (pos["shares"] + pos["pending_shares"]) for code, pos in holdings.items())
    buy_cap = (math.ceil(config.max_positions / config.holding_sessions)
               if strategy == "A" else config.max_positions)
    if strategy == "C" and not cycle:
        buy_cap = 0
    bought = 0
    for code in entry:
        if len(holdings) >= config.max_positions or bought >= buy_cap:
            break
        if code in holdings or code in sold:
            continue
        if s["unknown_claims"]:
            reject(code, "buy", "UNCONFIRMED_ACCOUNT_ECONOMICS")
            continue
        if any(not positive((bars.get(c) or {}).get("open")) for c in holdings):
            reject(code, "buy", "MISSING_HELD_OPEN_VALUATION")
            continue
        unit = trading_unit(code)
        if unit is None:
            reject(code, "buy", "UNSUPPORTED_SECURITY_UNIT")
            continue
        quote, reason = open_quote(bars.get(code), "buy", config)
        if not quote:
            reject(code, "buy", reason)
            continue
        minimum, increment, cap = unit
        budget = min(s["cash"], config.max_weight * marked_at_open)
        shares = min(cap, int(budget / quote[1])) // increment * increment
        # Include fees in the target/cash limit before reserving a lot.
        while shares >= minimum:
            value = money(shares * quote[1])
            fees = money(_money_cost(value, config, "buy"))
            if value + fees <= budget:
                break
            shares -= increment
        if shares < minimum:
            reject(code, "buy", "CASH_OR_POSITION_CAP_BELOW_MINIMUM_LOT")
            continue
        record(code, "buy", shares, quote, fees, prior_day)
        holdings[code] = dict(shares=shares, entry_session=day_no, entry_date=day,
                              signal_date=prior_day, mark=quote[0], last_mark_date=None,
                              pending_shares=0, share_list_date=None, exit_pending=None)
        bought += 1

    for code in sorted(held):
        item = actions.get(code)
        if item and (item.get("late_conflict") or item.get("coverage_unverified")):
            s["unknown_claims"].append({"day": day, "stock_code": code, "reason":
                "LATE_CONFLICTING_SETTLEMENT" if item.get("late_conflict") else "MISSING_SAME_DAY_ACTION_COVERAGE"})
        if item and not item.get("available_at_open", True):
            # Late terms cannot retroactively create deployable cash or saleable shares.
            add_action(s, code, item, held, day)
    unpaid = []
    for claim in s["cash_due"]:
        if claim["pay_date"] <= day:
            action_cash_paid = money(action_cash_paid + claim["amount"])
        else:
            unpaid.append(claim)
    s["cash_due"] = unpaid
    s["cash"] = money(s["cash"] + action_cash_paid)
    positions = []
    for code, pos in sorted(holdings.items()):
        mark = positive((bars.get(code) or {}).get("close"))
        if mark:
            pos["mark"], pos["last_mark_date"] = mark, day
        positions.append(dict(trade_date=day, stock_code=code, **pos,
            stale=pos["last_mark_date"] != day, delivered_shares=shares_delivered.get(code, 0),
            indicative_value=money(pos["mark"] * (pos["shares"] + pos["pending_shares"]))))
    opening_flow = sum((t["gross"] if t["side"] == "sell" else -t["gross"]) - t["fees"] for t in trades)
    if money(cash_start + opening_flow + action_cash_paid) != s["cash"]:
        raise ArithmeticError("Daily cash ledger does not reconcile")
    for code in set(held) | set(holdings):
        net_shares = sum(t["shares"] * (1 if t["side"] == "buy" else -1) for t in trades if t["stock_code"] == code)
        if held.get(code, 0) + net_shares + shares_delivered.get(code, 0) != holdings.get(code, {}).get("shares", 0):
            raise ArithmeticError("Daily share ledger does not reconcile")
    cash_due = money(sum(claim["amount"] for claim in unpaid))
    holdings_value = money(sum(p["indicative_value"] for p in positions))
    equity = money(s["cash"] + holdings_value + cash_due)
    uncertain = bool(s["unknown_claims"] or any(p["stale"] for p in positions))
    s["uncertain_history"] |= uncertain
    s["peak"] = max(s["peak"], equity)
    s["drawdown"] = min(s["drawdown"], equity / s["peak"] - 1)
    if benchmark_return is not None:
        benchmark_return = number(benchmark_return)
        if benchmark_return is None or benchmark_return < -1:
            raise ValueError("Invalid benchmark economic return")
    if prior_day is not None:
        if benchmark_return is None:
            s["benchmark_index"] = None
        elif s["benchmark_index"] is not None:
            s["benchmark_index"] *= 1 + benchmark_return
    net = None if uncertain else equity / config.initial_cash - 1
    bench = s["benchmark_index"]
    daily = dict(trade_date=day, signal_date=prior_day, strategy=strategy,
                 signal_status="MISSING" if ranked is None else "ZERO" if not ranked else "ARCHIVED",
                 opening_cash=cash_start, closing_cash=s["cash"], cash=s["cash"], action_cash_paid=action_cash_paid,
                 dividend_receivable=cash_due, holdings_value=holdings_value,
                 equity=equity, indicative_nav=equity/config.initial_cash,
                 valuation_status="UNCONFIRMED_ECONOMIC_ACTION" if s["unknown_claims"] else
                                  "STALE_CLOSE_MARK" if uncertain else "SIMULATED_CURRENT_MARKS",
                 nav=None if uncertain else equity / config.initial_cash, net_return=net,
                 daily_net_return=None if uncertain or state.get("last_valuation_uncertain") else equity/equity_start-1,
                 gross_return=None if uncertain else (equity + s["fees"] + s["slippage"])/config.initial_cash-1,
                 max_drawdown=None if s["uncertain_history"] else s["drawdown"],
                 benchmark_daily_return=benchmark_return, benchmark_return=None if bench is None else bench-1,
                 excess_return=None if net is None or bench is None else net-(bench-1),
                 turnover=sum(t["gross"] for t in trades)/equity_start, cumulative_fees=s["fees"],
                 cumulative_slippage=s["slippage"], positions=len(positions), trade_count=len(trades),
                 rejections=len(rejects), unresolved_claims=len(s["unknown_claims"]),
                 stale_positions=sum(p["stale"] for p in positions))
    s.update(strategy=strategy, last_day=day, last_valuation_uncertain=uncertain, session=day_no, equity=equity)
    return s, daily, positions, trades, rejects
