"""Audit historical availability of stored money-flow scores and signal events."""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import re

import pandas as pd
from sqlalchemy import text

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json
from .p4_subset_inputs import hash_file
from .p5_sequences import CHINA

TABLES = ("stock_main_force_daily", "stock_signal_event", "market")
MONEY = ("main_force_score", "activity_score", "holding_score", "distribution_risk")
BACKUP = "stock_main_force_daily_backup_before_backfill_20260818"
SHADOW = "stock_main_force_daily_v2"
AUXILIARY = (BACKUP, SHADOW)
VERSION = "P57_FUND_AVAILABILITY_V1"


def matching_writers(routines):
    """Record writer properties without exposing other database routine bodies."""
    facts = []
    for row in routines:
        body = row["body"]
        normalized = body.replace('"', "").lower()
        for table in TABLES[:2]:
            pattern = rf"\b(insert\s+into|update|delete\s+from)\s+(?:public\.)?{table}\b"
            verbs = sorted(set(re.findall(pattern, normalized)))
            if not verbs:
                continue
            facts.append({
                "routine": row["routine"], "table": table, "write_statements": verbs,
                "can_replace_existing_payload": ("update" in verbs or "delete from" in verbs
                                                  or ("on conflict" in normalized and "do update" in normalized)),
                "assigns_created_at": bool(re.search(r"\bcreated_at\s*=", normalized)),
                "assigns_signal_status_or_score": bool(re.search(r"\b(?:signal_status|signal_score)\s*=", normalized)),
                "source_md5": row["source_md5"],
            })
    return facts


def as_local(field, data_type):
    if data_type == "timestamp with time zone":
        return f"{field} AT TIME ZONE 'Asia/Shanghai'"
    if data_type == "timestamp without time zone":
        return field
    raise ValueError(f"Unexpected timestamp type for {field}: {data_type}")


def load_market(conn, bounds):
    return pd.read_sql_query(text("""
        SELECT trade_date AS session, count(DISTINCT stock_code) AS formal_stocks
          FROM public.market
         WHERE not_billing = 1 AND stock_code IS NOT NULL AND trade_date BETWEEN :start AND :end
         GROUP BY trade_date ORDER BY trade_date
    """), conn, params=bounds)


def load_money(conn, bounds, created):
    return pd.read_sql_query(text(f"""
        WITH official AS (
            SELECT DISTINCT trade_date, stock_code FROM public.market
             WHERE not_billing = 1 AND trade_date BETWEEN :start AND :end
        ), sessions AS (
            SELECT trade_date, lead(trade_date) OVER (ORDER BY trade_date) AS next_date
              FROM (SELECT DISTINCT trade_date FROM official) AS m
        ), base AS (
            SELECT f.*, {created} AS written_local, s.next_date,
                   m.trade_date IS NOT NULL AS formal_match
              FROM public.stock_main_force_daily AS f
              LEFT JOIN sessions s ON s.trade_date = f.trade_date
              LEFT JOIN official m ON m.trade_date = f.trade_date AND m.stock_code = f.stock_code
             WHERE f.trade_date BETWEEN :start AND :end
        )
        SELECT trade_date AS session, count(*) AS rows, count(DISTINCT stock_code) AS stocks,
               count(*) FILTER (WHERE formal_match) AS formal_key_matches,
               count(*) FILTER (WHERE written_local IS NULL) AS unknown_write_time,
               count(*) FILTER (WHERE written_local::date < trade_date) AS before_trade_date,
               count(*) FILTER (WHERE written_local::date = trade_date) AS same_day_write,
               count(*) FILTER (WHERE written_local::date > trade_date) AS after_trade_date,
               count(*) FILTER (WHERE written_local::date >= trade_date
                    AND written_local <= trade_date + time '18:00') AS timestamp_by_signal_cutoff,
               count(*) FILTER (WHERE written_local::date >= trade_date
                    AND written_local < next_date + time '09:30') AS timestamp_before_next_open,
               count(main_force_score) AS main_force_score, count(activity_score) AS activity_score,
               count(holding_score) AS holding_score, count(distribution_risk) AS distribution_risk,
               count(*) FILTER (WHERE formal_match AND main_force_score IS NOT NULL AND activity_score IS NOT NULL
                   AND holding_score IS NOT NULL AND distribution_risk IS NOT NULL) AS complete_formal_feature_rows
          FROM base GROUP BY trade_date ORDER BY trade_date
    """), conn, params=bounds)


def load_signals(conn, bounds, created):
    return pd.read_sql_query(text(f"""
        WITH official AS (
            SELECT DISTINCT trade_date, stock_code FROM public.market
             WHERE not_billing = 1 AND trade_date BETWEEN :start AND :end
        ), base AS (
            SELECT s.*, {created} AS written_local, m.trade_date IS NOT NULL AS formal_match
              FROM public.stock_signal_event AS s
              LEFT JOIN official m ON m.stock_code=s.stock_code AND m.trade_date=s.signal_date
             WHERE s.signal_date BETWEEN :start AND :end
        )
        SELECT signal_date AS session, signal_code, count(*) AS rows, count(DISTINCT stock_code) AS stocks,
               count(*) FILTER (WHERE formal_match) AS formal_key_matches,
               count(signal_score) AS score_rows,
               count(*) FILTER (WHERE written_local IS NULL) AS unknown_write_time,
               count(*) FILTER (WHERE written_local::date < signal_date) AS before_trade_date,
               count(*) FILTER (WHERE written_local::date = signal_date) AS same_day_write,
               count(*) FILTER (WHERE written_local::date > signal_date) AS after_trade_date,
               count(*) FILTER (WHERE written_local::date >= signal_date
                    AND written_local <= signal_date + time '18:00') AS timestamp_by_signal_cutoff,
               count(*) FILTER (WHERE confirm_time IS NULL) AS no_confirmation_time,
               count(*) FILTER (WHERE coalesce(last_trigger_time,signal_time) > time '18:00'
                                      OR confirm_time > time '18:00') AS late_event_or_confirmation,
               count(*) FILTER (WHERE signal_status = 'CONFIRMED') AS confirmed_rows
          FROM base GROUP BY signal_date, signal_code ORDER BY signal_date, signal_code
    """), conn, params=bounds)


def sums(frame, ignore):
    return {k: int(frame[k].sum()) for k in frame.columns if k not in ignore}


def inspect_auxiliary(conn, bounds, catalog, columns):
    """Check existing historic copies before deciding whether this ledger has PIT proof."""
    facts = {}
    established = {row["relation"] for row in catalog}
    for table in AUXILIARY:
        if table not in established:
            continue
        types = columns.loc[columns.table_name.eq(table)].set_index("column_name").data_type
        if not {"stock_code", "trade_date", "created_at"}.issubset(types.index):
            facts[table] = {"status": "MISSING_INDEPENDENT_TIME_EVIDENCE_COLUMNS"}
            continue
        created = as_local("a.created_at", types["created_at"])
        row = conn.execute(text(f"""
            SELECT count(*) AS rows, count(DISTINCT (trade_date, stock_code)) AS unique_keys,
                   count(DISTINCT trade_date) AS days, min(trade_date)::text AS first_signal,
                   max(trade_date)::text AS last_signal, min({created})::text AS first_stored_write,
                   max({created})::text AS last_stored_write,
                   count(*) FILTER (WHERE ({created})::date > trade_date) AS writes_after_signal_day,
                   count(*) FILTER (WHERE ({created})::date >= trade_date
                                AND {created} <= trade_date + time '18:00') AS stamps_by_signal_cutoff
              FROM public.{table} AS a
             WHERE trade_date BETWEEN :start AND :end
        """), bounds).mappings().one()
        facts[table] = dict(row) | {
            "status": "EXISTING_COPY_WITHOUT_VALIDATED_DAILY_ASOF_HISTORY"
        }
        if table != BACKUP or not set(MONEY).issubset(types.index):
            continue
        previous = ", ".join(f"a.{field}" for field in MONEY)
        current = ", ".join(f"f.{field}" for field in MONEY)
        changes = conn.execute(text(f"""
            SELECT count(*) AS matched_rows,
                   count(*) FILTER (WHERE ROW({previous}) IS DISTINCT FROM ROW({current})) AS different_payloads,
                   count(*) FILTER (WHERE a.created_at IS DISTINCT FROM f.created_at) AS different_created_at
              FROM public.{BACKUP} a JOIN public.stock_main_force_daily f USING (trade_date, stock_code)
             WHERE a.trade_date BETWEEN :start AND :end
        """), bounds).mappings().one()
        facts[table]["comparison_to_current_table"] = dict(changes)
    return facts


def audit(engine, start, end, output):
    now = datetime.now(CHINA)
    if start > end or end >= now.date():
        raise ValueError("Audit dates must be ordered and precede today's session")
    bounds = {"start": start, "end": end}
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
        with conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            if conn.scalar(text("SHOW transaction_read_only")) != "on":
                raise RuntimeError("A read-only audit transaction is required")
            db_tz = conn.scalar(text("SHOW TimeZone"))
            columns = pd.read_sql_query(text("""
                SELECT table_name, column_name, data_type
                  FROM information_schema.columns
                 WHERE table_schema='public' AND table_name=ANY(:tables)
                 ORDER BY table_name, ordinal_position
            """), conn, params={"tables": list(TABLES + AUXILIARY)})
            types = columns.set_index(["table_name", "column_name"]).data_type
            catalog = conn.execute(text("""
                SELECT n.nspname AS schema, c.relname AS relation, c.relkind AS type
                FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE n.nspname='public' AND c.relkind IN ('r','p','v','m') AND
                      c.relname ~ '(main_force|signal_event)'
                ORDER BY c.relname
            """)).mappings().all()
            routines = conn.execute(text("""
                SELECT (p.oid::regprocedure)::text AS routine, md5(p.prosrc) AS source_md5, p.prosrc AS body
                FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
                WHERE n.nspname='public' AND p.prokind IN ('f','p') AND
                      p.prosrc ~* '(stock_main_force_daily|stock_signal_event)'
                ORDER BY p.oid
            """)).mappings().all()
            writers = matching_writers(routines)
            market = load_market(conn, bounds)
            money = load_money(conn, bounds, as_local("f.created_at", types["stock_main_force_daily", "created_at"]))
            signals = load_signals(conn, bounds, as_local("s.created_at", types["stock_signal_event", "created_at"]))
            auxiliary = inspect_auxiliary(conn, bounds, catalog, columns)

    market_days = set(market.session)
    relevant_types = sorted({row for row in signals.signal_code if re.search(r"FORCE|BREAKOUT|PULLBACK", row, re.I)})
    relevant = signals.loc[signals.signal_code.isin(relevant_types)]
    totals = sums(money, {"session", "stocks"})
    signal_counts = sums(signals, {"session", "signal_code", "stocks"})
    score_path = "CURRENT_MUTABLE_SCORE_TABLE"
    signals_path = "CURRENT_MUTABLE_EVENT_TABLE"
    by_relation = {table: [w for w in writers if w["table"] == table] for table in TABLES[:2]}
    field_status = {
        "activity_score": score_path, "holding_score": score_path,
        "main_force_score_change": "BOTH_CURRENT_AND_PREVIOUS_SCORE_NEED_POINT_IN_TIME_LINEAGE",
        "distribution_risk": score_path, "breakout_or_dry_pullback_events": signals_path,
    }
    stats = {
        "version": VERSION, "database_transaction": "READ_ONLY_REPEATABLE_READ",
        "audit_clock_asia_shanghai": now.isoformat(), "database_session_timezone": db_tz,
        "timestamp_policy": "Asia/Shanghai for naive row times; timestamp screen alone is not immutable lineage",
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "formal_market": {"sessions": len(market), "stock_days": int(market.formal_stocks.sum())},
        "money": totals | {"days": len(money), "formal_sessions_with_scores": len(set(money.session) & market_days),
                           "first": str(money.session.min()) if len(money) else None,
                           "last": str(money.session.max()) if len(money) else None},
        "signals": signal_counts | {"days": int(signals.session.nunique()), "types": int(signals.signal_code.nunique()),
                                   "relevant_types": relevant_types, "relevant_rows": int(relevant.rows.sum()),
                                   "first": str(signals.session.min()) if len(signals) else None,
                                   "last": str(signals.session.max()) if len(signals) else None},
        "writer_properties": by_relation, "related_catalog_relations": [dict(x) for x in catalog],
        "auxiliary_tables": auxiliary,
        "feature_lineage": field_status, "point_in_time_verified_rows": 0,
        "trusted_dataset_created": False, "paired_model_fit_started": False,
        "decision": "CURRENT_TABLE_VERSIONS_UNADMITTED_WITHOUT_INDEPENDENT_ASOF_SNAPSHOTS",
    }
    output.mkdir(parents=True)
    market.to_csv(output / "formal_market_daily.csv", index=False)
    money.to_csv(output / "money_daily_availability.csv", index=False)
    signals.to_csv(output / "signal_daily_availability.csv", index=False)
    columns.to_csv(output / "column_catalog.csv", index=False)
    atomic_json(output / "fund_availability_audit.json",
                stats | {"files": {file.name: hash_file(file) for file in output.iterdir()}})
    print(f"P5.7 audited {totals['rows']:,} score rows and {signal_counts['rows']:,} event rows; "
          "trusted PIT rows 0", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(AShareDailyLoader().engine, args.start, args.end, args.out)


if __name__ == "__main__":
    main()
