"""Preserve exact, dated versions of mutable production money and signal rows."""
from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

from sqlalchemy import text

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, canonical, exclusive, read_json
from .p4_subset_inputs import hash_file

CHINA = ZoneInfo("Asia/Shanghai")
VERSION = "P58_FUND_SNAPSHOT_V1"
SOURCES = {"stock_main_force_daily": "trade_date", "stock_signal_event": "signal_date",
           "stock_main_force_daily_v2": "trade_date"}
KEY_FIELDS = {"stock_main_force_daily": ["main_force_score", "activity_score", "holding_score", "distribution_risk"],
              "stock_main_force_daily_v2": ["selection_score", "activity_score", "position_state", "overheat_risk"],
              "stock_signal_event": ["signal_code", "signal_status"]}
CUTOFF = time(18)
PHASES = ("close", "next-open", "observe", "preopen")
AUCTION_START = time(9, 15)
SSE_2026 = "https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20251222_10802510.shtml"
HOLIDAYS_2026 = (("01-01", "01-03"), ("02-15", "02-23"), ("04-04", "04-06"),
                 ("05-01", "05-05"), ("06-19", "06-21"), ("09-25", "09-27"), ("10-01", "10-07"))


def previous_scheduled_session(day):
    """Only published exchange-calendar dates qualify for a morning observation."""
    def open_day(d):
        md = d.strftime("%m-%d")
        return d.weekday() < 5 and not any(a <= md <= b for a, b in HOLIDAYS_2026)

    if day.year != 2026 or not open_day(day):
        return None
    earlier = day - timedelta(days=1)
    while earlier.year == 2026 and not open_day(earlier):
        earlier -= timedelta(days=1)
    return earlier if earlier.year == 2026 else None


def wall_clock():
    return datetime.now(CHINA)


def boundary(phase, day, started, finished, formal):
    if started.tzinfo is None or finished.tzinfo is None or finished < started:
        raise ValueError("Unordered or naive observation clocks")
    started, finished = started.astimezone(CHINA), finished.astimezone(CHINA)
    if phase == "preopen":
        if (formal and started.date() == finished.date() and finished.time() < AUCTION_START
                and previous_scheduled_session(started.date()) == day):
            return "NEXT_SESSION_BEFORE_OPENING_AUCTION"
        return "OBSERVATION_ONLY"
    if started.date() != day or finished.date() != day:
        return "CROSSED_SESSION_BOUNDARY"
    if not formal:
        return "NO_FORMAL_SESSION_OBSERVED"
    if phase == "close" and time(15) <= started.time() and finished.time() <= CUTOFF:
        return "BY_CLOSE_1800"
    if phase == "next-open" and started.time() >= CUTOFF:
        return "EVENING_OBSERVATION_FOR_FOLLOWING_SESSION"
    return "OBSERVATION_ONLY"


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_observation(engine, day, *, preopen_on=None):
    """One MVCC view across all sources. Keep PostgreSQL's exact JSON text and its field types."""
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn, conn.begin():
        conn.execute(text("SET TRANSACTION READ ONLY"))
        if conn.scalar(text("SHOW transaction_read_only")) != "on":
            raise RuntimeError("Fund snapshots require a read-only transaction")
        observed = conn.scalar(text("SELECT clock_timestamp()")).astimezone(CHINA)
        if observed.date() != (preopen_on or day):
            raise ValueError("Database observation date and requested capture date disagree")
        if preopen_on:
            last = conn.scalar(text("SELECT max(trade_date) FROM public.market WHERE not_billing=1"))
            if (last != day or day != previous_scheduled_session(preopen_on) or observed.time() >= AUCTION_START):
                raise ValueError("No verified latest formal session before the scheduled opening auction")
        market = conn.execute(text("""
            SELECT count(DISTINCT stock_code) FROM public.market WHERE trade_date=:day AND not_billing=1
        """), {"day": day}).scalar_one()
        prior = conn.scalar(text("""
            SELECT max(trade_date) FROM public.market WHERE trade_date < :day AND not_billing=1
        """), {"day": day})
        catalog = conn.execute(text("""
            SELECT table_name, column_name, data_type FROM information_schema.columns
            WHERE table_schema='public' AND table_name=ANY(:sources) ORDER BY table_name, ordinal_position
        """), {"sources": list(SOURCES)}).mappings()
        types = {source: {} for source in SOURCES}
        for field in catalog:
            types[field["table_name"]][field["column_name"]] = field["data_type"]
        writers = conn.execute(text(r"""
            SELECT p.oid::regprocedure::text AS name, md5(p.prosrc) AS source_md5
            FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
            WHERE n.nspname='public' AND p.prokind IN ('p','f') AND
              p.prosrc ~* '(insert[[:space:]]+into|update|delete[[:space:]]+from)[[:space:]]+(public[.])?(stock_main_force_daily(_v2)?|stock_signal_event)\M'
            ORDER BY 1
        """)).mappings().all()
        rows = {}
        for source, day_field in SOURCES.items():
            if not {"id", "stock_code", day_field, "created_at", *KEY_FIELDS[source]}.issubset(types[source]):
                raise ValueError(f"Missing expected production columns in {source}")
            # Table and column identifiers are fixed constants, never CLI input.
            rows[source] = list(conn.scalars(text(f"""
                SELECT to_jsonb(t)::text FROM public.{source} AS t
                WHERE t.{day_field}=:day ORDER BY t.id
            """), {"day": day}))
    return observed, int(market), prior, types, [dict(w) for w in writers], rows


def build_file(path, source, schema, raw, observed, source_version):
    import polars as pl

    integer = {"smallint", "bigint", "integer"}
    numeric = {"decimal", "numeric", "real", "double precision"}
    columns = {name: (pl.Int64 if kind in integer else pl.Float64 if kind in numeric else pl.String)
               for name, kind in schema.items()}
    columns.update(dict.fromkeys(("signal_date", "snapshot_at", "source_created_at", "source_updated_at",
                                  "data_source", "source_version", "raw_row_json"), pl.String))
    records = []
    original = []
    for payload in raw:
        row = json.loads(payload)
        original.append(row)
        stored = {name: (json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list))
                         else value) for name, value in row.items()}
        records.append(stored | {
            "signal_date": row[SOURCES[source]], "snapshot_at": observed.isoformat(),
            "source_created_at": row["created_at"], "source_updated_at": row.get("updated_at"),
            "data_source": source, "source_version": source_version, "raw_row_json": payload,
        })
    table = pl.DataFrame(records, schema=columns)
    table.write_parquet(path, compression="zstd")
    stocks = {r["stock_code"] for r in original if r["stock_code"]}
    valid_stocks = {r["stock_code"] for r in original if r["stock_code"] and
                    all(r[k] is not None for k in KEY_FIELDS[source])}
    stamps = [datetime.fromisoformat(r["created_at"]) for r in original if r["created_at"]]
    stamps = [t.replace(tzinfo=CHINA) if t.tzinfo is None else t.astimezone(CHINA) for t in stamps]
    source_day = datetime.fromisoformat(original[0][SOURCES[source]]).date() if original else observed.date()
    times = [round((t - datetime.combine(source_day, CUTOFF, CHINA)).total_seconds(), 3) for t in stamps]
    return {"rows": len(original), "stocks": len(stocks), "valid_stocks": len(valid_stocks),
            "required_fields": KEY_FIELDS[source], "file_sha256": hash_file(path),
            "non_null_by_field": {name: len(original) - table[name].null_count() for name in schema},
            "non_null_share_by_field": {name: (1 - table[name].null_count() / len(original) if original else None)
                                        for name in schema},
            "first_source_created_at": min(stamps).isoformat() if stamps else None,
            "last_source_created_at": max(stamps).isoformat() if stamps else None,
            "last_source_write_seconds_after_1800": max(times, default=None),
            "source_stamp_after_observation": sum(stamp > observed for stamp in stamps),
            "source_version": source_version}


def revisions(root, day, phase):
    folder = root / str(day) / phase
    return sorted(folder.glob("r[0-9]*/manifest.json")) if folder.exists() else []


def verified(path):
    manifest = read_json(path)
    if manifest["version"] != VERSION or set(manifest["sources"]) != set(SOURCES):
        raise ValueError("Unexpected archived fund snapshot manifest")
    for source, stats in manifest["sources"].items():
        if hash_file(path.parent / (source + ".parquet")) != stats["file_sha256"]:
            raise ValueError(f"Archived snapshot file changed: {path.parent}/{source}")
    return manifest


def stock_values(file, source):
    import polars as pl

    records = pl.read_parquet(file, columns=["stock_code", *KEY_FIELDS[source]]).iter_rows()
    by_stock = {}
    for stock, *values in records:
        if stock:
            by_stock.setdefault(stock, []).append(json.dumps(values, ensure_ascii=False))
    return {k: sorted(v) for k, v in by_stock.items()}


def prior_change(root, prior, phase, stage):
    if not prior or not (previous := revisions(root, prior, phase)):
        return {"status": "NO_SAME_PHASE_PREVIOUS_FORMAL_SESSION", "previous_formal_day": str(prior) if prior else None}
    path = previous[-1]
    verified(path)
    results = {}
    for source in SOURCES:
        name = source + ".parquet"
        old, now = stock_values(path.parent / name, source), stock_values(stage / name, source)
        overlap = now.keys() & old.keys()
        results[source] = {"added_stocks": len(now.keys() - old.keys()), "removed_stocks": len(old.keys() - now.keys()),
                           "shared_stocks": len(overlap), "changed_core_values": sum(now[k] != old[k] for k in overlap)}
    return {"status": "COMPARED", "previous_formal_day": str(prior), "previous_manifest_sha256": hash_file(path),
            "by_source": results}


def capture(engine, root, phase, *, clock=wall_clock, reader=load_observation):
    if phase not in PHASES:
        raise ValueError("Invalid fund snapshot phase")
    root = Path(root)
    with exclusive(root):
        start = clock().astimezone(CHINA)
        day = start.date()
        if phase == "close" and not time(15) <= start.time() < CUTOFF:
            return {"status": "OUTSIDE_CLOSE_SNAPSHOT_WINDOW", "at": start.isoformat()}
        if phase == "next-open" and start.time() < CUTOFF:
            return {"status": "BEFORE_EVENING_SNAPSHOT_WINDOW", "at": start.isoformat()}
        if phase == "preopen":
            day = previous_scheduled_session(start.date())
            if not day or start.time() >= AUCTION_START:
                return {"status": "OUTSIDE_VERIFIED_PREOPEN_WINDOW", "at": start.isoformat()}
            observation = reader(engine, day, preopen_on=start.date())
        else:
            observation = reader(engine, day)
        observed, formal_count, prior, schema, writers, raw = observation
        if observed.date() != start.date() or observed.tzinfo is None:
            raise ValueError("Observation clock disagrees with capture day")
        if phase != "observe" and not formal_count and not any(raw.values()):
            return {"status": "NO_FORMAL_OR_FUND_DATA_YET", "at": observed.isoformat()}
        fingerprint = {s: hashlib.sha256(b"\n".join(v.encode() for v in raw[s])).hexdigest() for s in SOURCES}
        content = sha([VERSION, schema, writers, formal_count, fingerprint])
        history = revisions(root, day, phase)
        previous_hash = None
        if history:
            last = verified(history[-1])
            if last["content_sha256"] == content:
                return {"status": "ALREADY_ARCHIVED", "manifest": str(history[-1]), "observed_at": observed.isoformat()}
            previous_hash = hash_file(history[-1])
        destination = root / str(day) / phase / f"r{len(history) + 1:05d}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".capture-", dir=root) as temporary:
            stage = Path(temporary) / "snapshot"
            stage.mkdir()
            stats = {s: build_file(stage / (s + ".parquet"), s, schema[s], raw[s], observed,
                                   sha([s, schema[s], writers])) for s in SOURCES}
            change = prior_change(root, prior, phase, stage)
            finish = clock().astimezone(CHINA)
            manifest = {"version": VERSION, "phase": phase, "revision": len(history) + 1,
                        "signal_date": str(day), "snapshot_at": observed.isoformat(), "data_read_started_at": start.isoformat(),
                        "files_ready_at": finish.isoformat(),
                        "availability": boundary(phase, day, start, max(finish, observed), formal_count),
                        "formal_stocks_at_observation": formal_count, "sources": stats,
                        "source_columns": schema, "source_writers": writers, "content_sha256": content,
                        "row_payload_sha256": fingerprint, "previous_revision_manifest_sha256": previous_hash,
                        "previous_day_changes": change, "read_only_repeatable_read": True,
                        "mutable_source_completion_certified": False}
            if phase == "preopen":
                manifest["target_session"] = str(start.date())
                manifest["exchange_calendar"] = {"source": SSE_2026, "year": 2026,
                                                 "opening_auction_cutoff": "09:15 Asia/Shanghai",
                                                 "latest_formal_session_checked": str(day)}
            atomic_json(stage / "manifest.json", manifest)
            stage.rename(destination)
        return {"status": "ARCHIVED", "manifest": str(destination / "manifest.json"), "availability": manifest["availability"],
                "rows": {s: stats[s]["rows"] for s in SOURCES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(AShareDailyLoader().engine, args.out, args.phase), ensure_ascii=False))


if __name__ == "__main__":
    main()
