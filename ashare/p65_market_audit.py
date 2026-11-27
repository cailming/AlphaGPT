"""Audit two formal-market gaps and their effect on frozen factor windows."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sqlalchemy import text

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, exclusive, read_json
from .p4_subset_inputs import hash_file, parquet_read, parquet_write
from .p54_factors import build_market_factors
from .p55_daily import now_shanghai
from .p58_fund_snapshot import SSE_2026, previous_scheduled_session

BEGIN = date(2026, 9, 1)
INCIDENT = (date(2026, 10, 16), date(2026, 10, 19))
BREADTH_WINDOW = 20
SEQUENCE_WINDOW = 16


def recovery(daily, observed_through, failures):
    """Use observed session positions up to cutoff, then project published open days."""
    by_session = {date.fromisoformat(r["trade_date"]): r["market_session_usable"] for r in daily}
    if observed_through != max(by_session) or not failures.issubset(by_session):
        raise ValueError("Expected a calendar covering both failures and the cutoff")
    cursor = observed_through
    projected = []
    while cursor < date(2026, 12, 31) and len(projected) < BREADTH_WINDOW + SEQUENCE_WINDOW - 2:
        cursor += timedelta(days=1)
        if previous_scheduled_session(cursor) is not None:
            projected.append(cursor)
    days = sorted(by_session) + projected
    index = {day: i for i, day in enumerate(days)}
    factor_days = {days[j] for day in failures for j in range(index[day], index[day] + BREADTH_WINDOW)
                   if j < len(days)}
    sequence_days = {days[j] for day in factor_days
                     for j in range(index[day], min(index[day] + SEQUENCE_WINDOW, len(days)))}
    factor_first = next((day for day in days if day > max(failures) and day not in factor_days), None)
    sequence_first = next((day for day in days if day > max(failures) and day not in sequence_days), None)
    current_window = [day for day in days if day <= observed_through][-SEQUENCE_WINDOW:]
    pending = [d for d in days if observed_through < d <= sequence_first] if sequence_first else []
    other_failures = [d for d, ok in by_session.items() if not ok and d > min(failures) and d not in failures]
    return {
        "market_20d_first_recovered_for_incident": str(factor_first) if factor_first else None,
        "sequence_16d_first_recovered_for_incident": str(sequence_first) if sequence_first else None,
        "impacted_market_20d_sessions": [str(d) for d in sorted(factor_days)],
        "impacted_signal_sessions": [str(d) for d in sorted(sequence_days)],
        "currently_affected_input_sessions": [str(d) for d in current_window if d in factor_days],
        "observed_other_invalid_pair_sessions": [str(d) for d in sorted(other_failures)],
        "scheduled_sessions_still_required": [str(d) for d in pending],
        "projection_conditions": ["All future scheduled sessions have formal bars meeting the existing pair gate.",
                                  "Each stock still passes its own history, evidence, and factor gates."],
        "future_schedule_source": SSE_2026,
    }


def records(conn, sql, params):
    return [dict(row) for row in conn.execute(text(sql), params).mappings()]


def snapshot_reconciliation(conn):
    """Estimate whether closing quote snapshots have the formal daily data contract."""
    controls = []
    for day, table in zip(INCIDENT, ("billing", "billing_bak"), strict=True):
        controls.extend(records(conn, f"""
            WITH last_quote AS (
                SELECT DISTINCT ON (stock_code) stock_code,ticktime,open,high,low,trade,volume,amount
                FROM public.{table} WHERE trade_date=:day AND ticktime >= time '15:00'
                ORDER BY stock_code,ticktime DESC
            )
            SELECT CAST(:day AS date) AS trade_date, count(*) AS formal_stocks,
                count(q.stock_code) AS has_closing_quote,
                count(*) FILTER (WHERE m.close=q.trade) AS exact_close,
                count(*) FILTER (WHERE (m.open,m.high,m.low,m.close)=(q.open,q.high,q.low,q.trade)) AS exact_ohlc,
                count(*) FILTER (WHERE m.volume=q.volume) AS exact_volume,
                count(*) FILTER (WHERE m.amount=q.amount) AS exact_amount,
                count(*) FILTER (WHERE (m.open,m.high,m.low,m.close,m.volume,m.amount)=
                                       (q.open,q.high,q.low,q.trade,q.volume,q.amount)) AS exact_ohlcv_amount,
                count(*) FILTER (WHERE abs(m.close-q.trade)>.01) AS close_disagrees_by_more_than_cent
            FROM public.market m LEFT JOIN last_quote q USING (stock_code)
            WHERE m.trade_date=:day AND m.not_billing=1""", {"day": day}))
        controls[-1]["source"] = table
    provenance = records(conn, """
        WITH last_quote AS (
            SELECT DISTINCT ON (stock_code) stock_code,ticktime,open,high,low,trade,volume,amount,settlement
            FROM public.billing WHERE trade_date=:day ORDER BY stock_code,ticktime DESC
        )
        SELECT count(*) AS temporary_stocks, count(q.stock_code) AS has_same_day_quote,
            min(q.ticktime)::text AS earliest_last_quote, max(q.ticktime)::text AS latest_last_quote,
            count(*) FILTER (WHERE q.stock_code IS NOT NULL
                AND (m.open,m.high,m.low,m.close,m.volume,m.amount,m.pre_close)
                    IS NOT DISTINCT FROM (q.open,q.high,q.low,q.trade,q.volume,q.amount,q.settlement)) AS exact_match
        FROM public.market m LEFT JOIN last_quote q USING (stock_code)
        WHERE m.trade_date=:day AND m.not_billing=2""", {"day": INCIDENT[0]})[0]
    return {"formal_comparisons": controls, "oct16_temp_same_day_quote_comparison": provenance}


def closing_quote_patch(conn, reconciliation):
    """Extract standalone, calibrated research bars for missing formal keys only."""
    controls = reconciliation["formal_comparisons"]
    if any(r["formal_stocks"] == 0 or r["formal_stocks"] != r["exact_ohlcv_amount"] for r in controls):
        raise ValueError("Closing quotes failed exact matched-formal calibration")
    return pd.read_sql_query(text("""
        WITH ranked_quote AS (
            SELECT stock_code,trade_date,id,ticktime,open,high,low,trade,volume,amount,
                rank() OVER (PARTITION BY stock_code ORDER BY ticktime DESC) AS latest_rank,
                count(*) OVER (PARTITION BY stock_code,ticktime) AS key_rows
            FROM public.billing WHERE trade_date=:day AND ticktime>=time '15:00'
        ),
        last_quote AS (
            SELECT * FROM ranked_quote WHERE latest_rank=1 AND key_rows=1
        )
        SELECT q.stock_code,q.trade_date,q.open,q.high,q.low,q.trade AS close,q.volume,q.amount,
               q.id AS source_billing_id,q.ticktime::text AS source_quote_ticktime,p.close AS previous_formal_close
        FROM last_quote q JOIN public.market p
          ON p.stock_code=q.stock_code AND p.trade_date=:prior AND p.not_billing=1
        WHERE q.stock_code ~ '^[0-9]{6}$' AND q.ticktime BETWEEN time '15:00' AND time '15:05'
          AND q.open>0 AND q.trade>0 AND q.low>0 AND q.volume>0 AND q.amount>0 AND p.close>0
          AND q.high>=greatest(q.open,q.trade) AND q.low<=least(q.open,q.trade)
          AND abs(q.open/p.close-1)<=.30 AND abs(q.trade/p.close-1)<=.32
          AND NOT EXISTS (
            SELECT 1 FROM public.market f WHERE f.stock_code=q.stock_code
               AND f.trade_date=q.trade_date AND f.not_billing=1
          )
        ORDER BY q.stock_code
    """), conn, params={"day": INCIDENT[0], "prior": INCIDENT[0] - timedelta(days=1)})


def patch_scenario(raw, keys, patch):
    if patch.empty:
        raise ValueError("Closing quote research patch has no eligible rows")
    original = pd.MultiIndex.from_frame(raw[["stock_code", "trade_date"]])
    supplement = pd.MultiIndex.from_frame(patch[["stock_code", "trade_date"]])
    if supplement.has_duplicates or not original.intersection(supplement).empty:
        raise ValueError("Patch must not contain duplicate or original formal-market keys")
    if set(patch.trade_date) != {INCIDENT[0]}:
        raise ValueError("Patch must be independently scoped to the missing session")
    # The frozen factor function expects an eligibility flag on included input
    # rows. This in-memory scenario adapter is never written to any archive,
    # used for production predictions or read by the frozen live model.
    scenario_rows = patch.loc[:, [c for c in raw if c != "not_billing"]].assign(not_billing=1)
    augmented = pd.concat([raw, scenario_rows], ignore_index=True)
    computed, audit = build_market_factors(augmented, keys)
    return computed, audit["market_session_support"]


def compare_archives(archives, target):
    results = []
    target = target.set_index(["trade_date", "stock_code"])
    fields = ["open", "high", "low", "close", "volume", "amount"]
    for path in archives:
        digest = hash_file(path)
        plans = (path.parent / "inputs.json", path.parent / "input_manifest.json",
                 path.parent.parent / "manifest.json", path.parent / "manifest.json")
        witness = next((p for p in plans if p.is_file()
                        and read_json(p).get("files", {}).get(path.name) == digest), None)
        raw = parquet_read(path)
        raw["trade_date"] = pd.to_datetime(raw.trade_date).dt.date
        raw["stock_code"] = raw.stock_code.astype(str)
        selected = raw.loc[raw.trade_date.isin(INCIDENT)]
        days = {}
        for day in INCIDENT:
            candidate = selected.loc[selected.trade_date.eq(day)].set_index(["trade_date", "stock_code"])
            if candidate.index.duplicated().any():
                raise ValueError(f"Duplicate stock/session in archive {path}")
            formal = candidate.loc[candidate.not_billing.eq(1)] if "not_billing" in candidate else candidate.iloc[0:0]
            common = formal.index.intersection(target.index)
            complete = formal[fields].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
            same = all(pd.to_numeric(formal.loc[common, field]).eq(pd.to_numeric(target.loc[common, field])).all()
                       for field in fields)
            days[str(day)] = {"rows": len(candidate), "formal_rows": len(formal), "complete_formal_ohlcv": int(complete.sum()),
                              "matching_key_rows": len(common), "matching_keys_ohlcv_equal": same if len(common) else None,
                              "extra_complete_formal_rows": int(complete.loc[formal.index.difference(target.index)].sum())}
        results.append({"path": str(path), "sha256": digest, "bound_manifest": str(witness) if witness else None,
                        "first_session": str(raw.trade_date.min()), "last_session": str(raw.trade_date.max()), "days": days})
    return results


def collect(engine, archives, now, research_patch=False):
    cutoff = now.date() + timedelta(days=(now.hour >= 18))
    params = {"first": INCIDENT[0], "second": INCIDENT[1], "begin": BEGIN, "cutoff": cutoff}
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn, conn.begin():
        conn.execute(text("SET TRANSACTION READ ONLY"))
        conn.execute(text("SET LOCAL statement_timeout = '60s'"))
        if conn.scalar(text("SHOW transaction_read_only")) != "on":
            raise RuntimeError("Market gap audit must use a read-only transaction")
        rows = records(conn, """SELECT trade_date, left(stock_code,3) AS prefix, not_billing, count(*) AS rows,
                                count(DISTINCT stock_code) AS stock_count, min(id) AS min_id, max(id) AS max_id
                                FROM public.market WHERE trade_date IN (:first, :second)
                                GROUP BY 1,2,3 ORDER BY 1,2,3""", params)
        status = records(conn, """SELECT trade_date,not_billing,count(*) AS rows,count(DISTINCT stock_code) AS stocks,
                                   min(id) AS min_id,max(id) AS max_id FROM public.market
                                   WHERE trade_date BETWEEN :first - 2 AND :second + 1 GROUP BY 1,2 ORDER BY 1,2""", params)
        # The auxiliary tables below have different price, timestamp and completeness semantics.
        auxiliary = {}
        for table, close in (("market_extend", "NULL"), ("billing_bak", "ticktime"),
                             ("market_minute_snapshot", "trade_minute")):
            fields = f", max({close})::text AS last_observed_tick" if close != "NULL" else ""
            auxiliary[table] = records(conn, f"""SELECT trade_date,count(*) AS rows,
                count(DISTINCT stock_code) AS stocks {fields}
                FROM public.{table} WHERE trade_date IN (:first,:second) GROUP BY 1 ORDER BY 1""", params)
        overlaps = records(conn, """
            SELECT count(*) FILTER (WHERE formal AND provisional) AS in_both,
                   count(*) FILTER (WHERE formal AND NOT provisional) AS formal_only,
                   count(*) FILTER (WHERE provisional AND NOT formal) AS provisional_only
            FROM (
                SELECT stock_code,bool_or(not_billing=1) AS formal,bool_or(not_billing=2) AS provisional
                FROM public.market WHERE trade_date=:first GROUP BY stock_code
            ) AS stock_presence""", params)[0]
        temp_id = conn.scalar(text("""SELECT max(id) FROM public.market
                                     WHERE trade_date=:first AND not_billing=2"""), params)
        nearby_rows = records(conn, """SELECT id, trade_date,not_billing FROM public.market
                                      WHERE id BETWEEN :last_id-1 AND :last_id+2 ORDER BY id""",
                              {"last_id": temp_id}) if temp_id is not None else []
        calendar = list(conn.scalars(text("""SELECT DISTINCT trade_date FROM public.trade_date_list
                                             WHERE trade_date >= :begin AND trade_date < :cutoff ORDER BY 1"""), params))
        raw = pd.read_sql_query(text("""SELECT stock_code, trade_date, open, high, low, close, volume, amount, not_billing
                                       FROM public.market WHERE not_billing=1 AND stock_code IS NOT NULL
                                       AND trade_date >= :begin AND trade_date < :cutoff"""), conn, params=params)
        snapshot_sources = snapshot_reconciliation(conn)
        patch = closing_quote_patch(conn, snapshot_sources) if research_patch else None
    raw["trade_date"] = pd.to_datetime(raw.trade_date).dt.date
    days = sorted(raw.trade_date.unique())
    if not days or any(d not in days for d in INCIDENT):
        raise ValueError("Expected the audited incident within the formal-price window")
    asof = days[-1]
    keys = pd.DataFrame({"stock_code": "688256", "trade_date": days})
    factors, computed = build_market_factors(raw, keys)
    by_day = computed["market_session_support"]
    failed = {date.fromisoformat(r["trade_date"]) for r in by_day if not r["market_session_usable"]}
    if not set(INCIDENT).issubset(failed):
        projection = {"status": "INCIDENT_PAIR_GATE_NOW_PASSES_ON_ONE_OR_BOTH_DAYS"}
    else:
        projection = recovery(by_day, asof, set(INCIDENT))
    sample = factors.loc[factors.trade_date.ge("2026-10-14")].astype(object)
    report = {
        "observed_at": now.isoformat(), "latest_formal_used": str(asof), "read_only_repeatable_read": True,
        "formal_minus_database_trade_calendar": [str(d) for d in sorted(set(days) - set(calendar))],
        "database_trade_calendar_minus_formal": [str(d) for d in sorted(set(calendar) - set(days))],
        "observed_session_calendar": [str(d) for d in days],
        "market_counts_by_flag": status, "counts_by_prefix_and_flag": rows, "oct16_stock_presence": overlaps,
        "oct16_last_temporary_id_neighbours": nearby_rows, "auxiliary_rows_only_not_formal": auxiliary,
        "closing_snapshot_reconciliation": snapshot_sources,
        "formal_ohlcv_archives": compare_archives(archives, raw.loc[raw.trade_date.isin(INCIDENT)]),
        "pair_counts": by_day,
        "pair_policy": {k: computed[k] for k in ("minimum_market_pair_share", "minimum_market_pair_stocks")},
        "window_projection": projection,
        "market_and_reference_stock_factors": sample.where(sample.notna(), None).to_dict("records"),
        "provisional_source_rows_used_for_factors": 0,
    }
    if patch is not None:
        patch["trade_date"] = pd.to_datetime(patch.trade_date).dt.date
        patched, daily = patch_scenario(raw, keys, patch)
        shadow = patched.loc[patched.trade_date.ge("2026-10-14")].astype(object)
        latest_16 = patched.tail(SEQUENCE_WINDOW)
        report["research_quote_patch"] = {
            "rows": len(patch), "source": "public.billing", "trade_date": str(INCIDENT[0]),
            "latest_observed_quote_time_range": [patch.source_quote_ticktime.min(), patch.source_quote_ticktime.max()],
            "only_original_missing_keys": True, "formal_status": "RESEARCH_ONLY_CLOSE_QUOTE",
            "exact_control_reconciliation_required": True,
            "scope": "Preceding formal universe, positive OHLCV and amount, final tick 15:00–15:05, frozen gap and return bounds.",
            "market_pair_counts_scenario": daily,
            "current_market_20d_missing_input_dates_scenario": latest_16.loc[
                latest_16[["market_ret_20", "market_turnover_log_change_20"]].isna().any(axis=1), "trade_date"].tolist(),
            "market_and_reference_stock_factors_scenario": shadow.where(shadow.notna(), None).to_dict("records"),
        }
    return report, patch


def json_value(value):
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return [json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    return value


def publish_output(output, report, patch):
    with exclusive(output.parent):
        if output.exists():
            raise FileExistsError(output)
        with TemporaryDirectory(prefix=".p65-audit-", dir=output.parent) as work:
            staged = Path(work) / "result"
            staged.mkdir()
            if patch is not None:
                path = staged / "close_quote_research_patch.parquet"
                recorded = patch.assign(trade_date=patch.trade_date.astype(str), source_kind="OBSERVED_POST_CLOSE_QUOTE")
                parquet_write(recorded, path)
                report["research_quote_patch"]["files"] = {path.name: hash_file(path)}
            atomic_json(staged / "market_audit.json", json_value(report))
            os.rename(staged, output)
            directory = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    return output / "market_audit.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", default=[])
    parser.add_argument("--build-research-patch", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report, patch = collect(AShareDailyLoader().engine, args.archive, now_shanghai(), args.build_research_patch)
    path = publish_output(args.out, report, patch)
    print(hash_file(path), path)


if __name__ == "__main__":
    main()
