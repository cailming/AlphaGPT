"""Published exchange sessions and the independent formal-price coverage audit."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import text

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, exclusive, read_json
from .p4_subset_inputs import hash_file
from .p55_daily import now_shanghai
from .p58_fund_snapshot import HOLIDAYS_2026, SSE_2026

DAY = timedelta(days=1)
YEAR_HOLIDAYS = {2026: HOLIDAYS_2026}


def planned_open(day: date) -> bool | None:
    if day.weekday() >= 5:
        return False
    intervals = YEAR_HOLIDAYS.get(day.year)
    if intervals is None:
        return None
    md = day.strftime("%m-%d")
    return not any(start <= md <= end for start, end in intervals)


def next_session(day: date, steps: int = 1) -> date | None:
    """Advance by scheduled sessions, never inferring a closure from missing rows."""
    if steps < 1:
        raise ValueError("Expected a positive number of sessions")
    while steps:
        day += DAY
        status = planned_open(day)
        if status is None:
            return None
        steps -= int(status)
    return day


def coverage(begin, end, observed):
    result = {}
    day = begin
    while day <= end:
        plan = planned_open(day)
        bars = observed.get(day, 0)
        if plan is None:
            status = "SCHEDULE_UNCONFIRMED"
        elif plan:
            status = "FORMAL_OBSERVED" if bars else "SCHEDULED_OPEN_MISSING_FORMAL"
        else:
            status = "UNEXPECTED_FORMAL_ON_CLOSED_DAY" if bars else "SCHEDULED_CLOSED"
        result[day.isoformat()] = {"status": status, "formal_stocks": bars}
        day += DAY
    return result


def recovery_check(root, dates, end):
    files = sorted((root/"monitor").glob("????-??-??/*/status.json"), reverse=True)
    for file in files:
        monitor = read_json(file)
        formal = monitor.get("formal", {})
        if formal.get("pair_support") and monitor["formal_through"] <= end.isoformat():
            break
    else:
        return {"status": "AWAITING_BREADTH_MONITOR"}
    through = monitor["formal_through"]
    by_day = {r["trade_date"]: r["market_session_usable"] for r in formal["pair_support"]}
    # A failed pair affects 20 rolling values. The last of those leaves the
    # 16-step input 35 scheduled sessions after the original failed pair.
    failures = {d for d, ok in by_day.items() if not ok}
    for day, row in dates.items():
        if day >= min(by_day) and row["status"] == "SCHEDULED_OPEN_MISSING_FORMAL":
            failures.add(day)
            # Missing bars also invalidate the next day's return pair.
            after = next_session(date.fromisoformat(day))
            if after:
                failures.add(after.isoformat())
    earliest = next_session(date.fromisoformat(max(failures)), 35) if failures else None
    stale = any(d > through and v["status"] != "SCHEDULED_CLOSED" for d, v in dates.items())
    return {"status": "MONITOR_STALE" if stale else "OBSERVED", "formal_through": through,
            "monitor_sha256": hash_file(file), "failed_or_unobservable_pair_days": sorted(failures),
            "current_missing_20d_inputs": formal["missing_20d_feature_dates"],
            "earliest_clean_signal_if_future_pairs_pass": earliest.isoformat() if earliest else None,
            "frozen_minimum_pair_share": formal["minimum_pair_share"],
            "planned_review": "2026-12-07", "review_due": end >= date(2026,12,7),
            "factors18_archived_scores": monitor["models"].get("factors18", {}).get("scored"),
            "current_c_capture": group_result(root, through)}


def group_result(root, through):
    file = root/"group_c/timely"/through/"manifest.json"
    if not file.is_file():
        return None
    rec = read_json(file)
    return {"scores": rec["models"]["group_c"]["rows"], "manifest_sha256": hash_file(file)}


def audit(root, begin):
    moment = now_shanghai()
    end = moment.date() - (DAY if moment.hour < 18 else timedelta(0))
    if end < begin:
        raise ValueError("Audit begins after the last observable session")
    with AShareDailyLoader().engine.begin() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        rows = conn.execute(text("""SELECT trade_date, count(DISTINCT stock_code)
            FROM public.market WHERE not_billing=1 AND stock_code IS NOT NULL
            AND trade_date BETWEEN :begin AND :end GROUP BY trade_date"""), {"begin":begin,"end":end})
        observed = dict(rows.tuples().all())
    days = coverage(begin, end, observed)
    result = {"observed_at": moment.isoformat(), "through": end.isoformat(), "source": SSE_2026,
              "supported_years": sorted(YEAR_HOLIDAYS), "days": days,
              "missing_formal": [d for d,v in days.items() if v["status"] == "SCHEDULED_OPEN_MISSING_FORMAL"],
              "calendar_unconfirmed": [d for d,v in days.items() if v["status"] == "SCHEDULE_UNCONFIRMED"],
              "enhanced_recovery": recovery_check(root, days, end)}
    out = root/"p66/calendar"/end.isoformat()
    with exclusive(out):
        file = out/(moment.strftime("%Y%m%dT%H%M%S%f")+".json")
        atomic_json(file,result)
    return file


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--begin", type=date.fromisoformat, default=date(2026,9,1))
    args = p.parse_args()
    print(audit(args.root, args.begin))
