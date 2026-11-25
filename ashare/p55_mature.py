"""Read immutable prediction journals and mature five-session research labels."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, read_json
from .p4_corporate_actions import load_actions
from .p4_subset_dataset import economic_return
from .p4_subset_inputs import hash_file, parquet_read, parquet_write
from .p5_sequences import CHINA

VERSION = "P55_MATURITY_V1"
OBJECTIVES = ("absolute", "excess", "rank")
GROUPS = ("ALL", "0-4", "5-19", "20+")
COSTS = (5, 10, 25)
HORIZON = 5
MIN_LABEL_COVERAGE = .80
MIN_MATURED_DAYS = 20
OFFICIAL = Path("ashare/data/corporate_actions_verified_examples.csv")
PREDICTION_FIELDS = ("prediction_id", "stock_code", "trade_date", "training_sequences", "score",
                     "liquidity_rank", "objective", "model_version", "model_sha256", "support_group")
BAR_FIELDS = ("open", "high", "low", "close", "volume", "amount")
LABEL_FIELDS = ("prediction_id", "stock_code", "trade_date", "objective", "model_version", "model_sha256",
                "training_sequences", "support_group", "score", "track", "maturity_status",
                "entry_date", "exit_date", "target_return")


def compatible_date(value):
    return date.fromisoformat(value).isoformat() == value


def support_group(number):
    return "0-4" if number < 5 else "5-19" if number < 20 else "20+"


def load_ledger(root):
    entries = []
    predictions = []
    for track_dir in ("timely", "historical"):
        for path in sorted((root / track_dir).glob("????-??-??/manifest.json")):
            manifest = read_json(path)
            signal = manifest["trade_date"]
            kind = manifest["capture_kind"]
            if (signal != path.parent.name or not compatible_date(signal)
                    or kind != ("TIMELY" if track_dir == "timely" else "HISTORICAL")):
                raise ValueError(f"Prediction origin mismatch: {path}")
            eligible = bool(manifest["prospective_eligible"])
            if eligible and (kind != "TIMELY" or not manifest["formal_session"]):
                raise ValueError(f"Ineligible origin labeled prospective: {path}")
            committed = datetime.fromisoformat(manifest["committed_at"])
            frozen = [datetime.fromisoformat(manifest["models"][arm]["model_frozen_at"]) for arm in OBJECTIVES]
            if any(moment.tzinfo is None for moment in frozen):
                raise ValueError(f"Model freeze time must include a timezone: {path}")
            freeze_day = max(moment.astimezone(CHINA).date().isoformat() for moment in frozen)
            if eligible and (committed.tzinfo is None or committed.astimezone(CHINA).date().isoformat() != signal
                             or committed.astimezone(CHINA).time() < time(18) or signal <= freeze_day):
                raise ValueError(f"Prediction was not committed in the timely window: {path}")
            track = "PROSPECTIVE" if eligible else ("TIMELY_NONPROSPECTIVE" if kind == "TIMELY" else "HISTORICAL")
            item = {"track": track, "trade_date": signal, "formal_session": manifest["formal_session"],
                    "prospective_start_after": freeze_day,
                    "status": manifest["status"], "manifest_sha256": hash_file(path), "committed_at": manifest["committed_at"]}
            counts = {}
            for objective in OBJECTIVES:
                spec = manifest["models"][objective]
                configured = spec["predictions_file"]
                if Path(configured).name != configured:
                    raise ValueError(f"Invalid prediction filename: {configured}")
                file = path.parent / configured
                if hash_file(file) != spec["predictions_sha256"]:
                    raise ValueError(f"Frozen prediction changed: {file}")
                rows = parquet_read(file)
                if (not set(PREDICTION_FIELDS).issubset(rows) or len(rows) != spec["rows"]
                        or rows.stock_code.duplicated().any() or rows.prediction_id.duplicated().any()
                        or not rows.stock_code.astype("string").str.fullmatch(r"\d{6}").all()
                        or not rows.trade_date.eq(signal).all()
                        or not rows.objective.eq(objective).all() or not rows.model_sha256.eq(spec["weights_sha256"]).all()
                        or not np.isfinite(rows.score.to_numpy(dtype=float)).all()
                        or not rows.training_sequences.map(support_group).eq(rows.support_group).all()):
                    raise ValueError(f"Bad prediction schema, key or score: {file}")
                if len(rows):
                    predictions.append(rows.loc[:, PREDICTION_FIELDS].assign(track=track))
                counts[objective] = {"rows": len(rows), "abstentions": spec["abstentions"],
                                     "abstention_reasons": spec["abstention_reasons"],
                                     "model_sha256": spec["weights_sha256"]}
            entries.append(item | {"models": counts})
    rows = (pd.concat(predictions, ignore_index=True) if predictions
            else pd.DataFrame(columns=[*PREDICTION_FIELDS, "track"]))
    if rows.prediction_id.duplicated().any():
        raise ValueError("Prediction IDs must be unique across immutable journal types")
    return entries, rows


def select_evidence(ledger, seed, explicit):
    candidates = [explicit] if explicit else sorted((ledger / "evidence").glob("????-??-??/dataset"), reverse=True)
    if not explicit and seed is not None:
        candidates.append(seed)
    if not candidates:
        return {"stocks": {}, "max_covered_date": None, "sha256": None}
    path = candidates[0]
    artifact = path / "inference_evidence.json"
    manifest = read_json(path / "subset_audit.json")
    digest = hash_file(artifact)
    value = read_json(artifact)
    observed = datetime.fromisoformat(value["observed_at_utc"])
    if (digest != manifest["files"][artifact.name] or value["policy"] != "P5_DUAL_OBSERVED_RESEARCH_WINDOW_V1"
            or observed.tzinfo is None or observed > datetime.now(CHINA)):
        raise ValueError(f"Invalid dated dual-source evidence: {path}")
    return value | {"sha256": digest}


def formal_tail(engine, begin, codes, now, cutoff=None):
    """Use the shared formal market calendar from a single read-only DB snapshot."""
    ceiling = now.date() if now.time() < time(18) else now.date() + timedelta(days=1)
    if cutoff:
        if cutoff >= ceiling:
            raise ValueError("Requested cutoff has not reached the formal close boundary")
        ceiling = cutoff + timedelta(days=1)
    session_sql = text("""SELECT DISTINCT trade_date FROM public.market
                           WHERE not_billing=1 AND trade_date >= :begin AND trade_date < :ceiling ORDER BY trade_date""")
    prices_sql = text("""SELECT stock_code,trade_date,open,high,low,close,volume,amount,pre_close
                          FROM public.market WHERE not_billing=1 AND trade_date >= :begin
                          AND trade_date < :ceiling AND stock_code = ANY(:codes)""")
    settings = {"begin": begin, "ceiling": ceiling}
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
        with conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            if conn.scalar(text("SHOW transaction_read_only")) != "on":
                raise RuntimeError("Expected a read-only formal-price snapshot")
            calendar = [pd.Timestamp(day).date().isoformat() for day in conn.scalars(session_sql, settings)]
            raw = (pd.read_sql_query(prices_sql, conn, params=settings | {"codes": codes})
                   if codes else pd.DataFrame(columns=["stock_code", "trade_date", *BAR_FIELDS]))
    if len(raw):
        raw["trade_date"] = pd.to_datetime(raw.trade_date).dt.strftime("%Y-%m-%d")
    if raw.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Formal label bars contain duplicate stock/day keys")
    return calendar, raw


def one_label(stock, signal, calendar, sessions, prices, evidence, official):
    start = sessions.get(signal)
    if start is None or start + HORIZON + 1 >= len(calendar):
        return ("LABEL_PENDING", "", "", None) if start is not None else ("MISSING_FORMAL_SIGNAL", "", "", None)
    days = calendar[start:start + HORIZON + 2]
    entry, exit_day = days[1], days[-1]
    failure = lambda reason: (reason, entry, exit_day, None)
    witness = evidence["stocks"].get(stock)
    if (witness is None or not witness["covered_from"] <= signal <= exit_day <= witness["covered_through"]
            or exit_day > (evidence["max_covered_date"] or "")):
        return failure("MISSING_DUAL_ACTION_EVIDENCE")
    seen = {item[0] for item in witness["actions"] if entry <= item[0] <= exit_day}
    verified = {day: terms for (code, day), terms in official.items()
                if code == stock and entry <= day <= exit_day}
    if seen - set(verified):
        return failure("UNVERIFIED_ECONOMIC_ACTION")
    bars = prices.get(stock)
    if bars is None:
        return failure("MISSING_FORMAL_PRICE_PATH")
    frame = bars.reindex(days)
    values = frame.loc[:, BAR_FIELDS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return failure("MISSING_FORMAL_PRICE_PATH")
    op, hi, low, cl, volume, amount = values.T
    if not ((low > 0) & (volume > 0) & (amount > 0) &
            (hi >= np.maximum(op, cl)) & (low <= np.minimum(op, cl))).all():
        return failure("INVALID_FORMAL_PRICE_PATH")
    jump = (np.abs(op[1:] / cl[:-1] - 1) > .30) | (np.abs(cl[1:] / cl[:-1] - 1) > .32)
    if any(flag and day not in verified for flag, day in zip(jump, days[1:])):
        return failure("UNEXPLAINED_PRICE_JUMP")
    target = economic_return(op[1], op[-1], [verified[d] for d in sorted(verified) if d > entry])
    if not np.isfinite(target) or abs(target) > .8:
        return failure("ECONOMIC_RETURN_OUT_OF_BOUNDS")
    return "MATURE_LABEL", entry, exit_day, float(target)


def daily_report(rows, pools, previous):
    scored, available = len(rows), len(rows.loc[rows.target_return.notna()])
    coverage = available / scored if scored else None
    observed = rows.loc[rows.target_return.notna()]
    pool_observed = pools.loc[pools.target_return.notna()]
    rank_ic = None
    if (coverage is not None and coverage >= MIN_LABEL_COVERAGE and len(observed) >= 5
            and observed.score.nunique() > 1 and observed.target_return.nunique() > 1):
        rank_ic = float(observed.score.rank().corr(observed.target_return.rank()))
    pool_mean = (float(pool_observed.target_return.mean()) if len(pools)
                 and len(pool_observed) / len(pools) >= MIN_LABEL_COVERAGE else None)
    ranked = rows.sort_values(["score", "stock_code"], ascending=[False, True], kind="stable")
    result = {"scores": scored, "labels": available, "label_coverage": coverage, "rank_ic": rank_ic,
              "label_statuses": dict(Counter(rows.maturity_status)), "all_scored_pool_mean": pool_mean}
    for k in (10, 20):
        choice = ranked.head(k)
        selected = set(choice.stock_code)
        prior = previous.get(k)
        present = choice.loc[choice.target_return.notna()]
        top_coverage = len(present) / len(choice) if len(choice) else None
        gross = (float(present.target_return.mean()) if top_coverage is not None and top_coverage >= MIN_LABEL_COVERAGE else None)
        excess = gross - pool_mean if gross is not None and pool_mean is not None else None
        result[f"top{k}"] = {"selected": len(choice), "observed": len(present), "label_coverage": top_coverage,
                              "ranked_before_outcomes": True, "gross_economic_return": gross,
                              "excess_over_all_scored_pool": excess,
                              "net_excess_by_per_side_bps": {str(c): None if excess is None else excess - 2*c/10000 for c in COSTS},
                              "list_replacement": None if not prior else len(prior - selected) / len(prior)}
        previous[k] = selected
    return result


def mean_available(values):
    observed = [value for value in values if value is not None]
    return float(np.mean(observed)) if len(observed) else None


def evaluate(entries, predictions, calendar, raw, evidence, official):
    sessions = {day: i for i, day in enumerate(calendar)}
    by_stock = {code: rows.set_index("trade_date") for code, rows in raw.groupby("stock_code")}
    keyed = {key: one_label(*key, calendar, sessions, by_stock, evidence, official)
             for key in predictions[["stock_code", "trade_date"]].drop_duplicates().itertuples(index=False, name=None)}
    labelled = predictions.copy()
    computed = [keyed[key] for key in predictions[["stock_code", "trade_date"]].itertuples(index=False, name=None)]
    labelled[["maturity_status", "entry_date", "exit_date", "target_return"]] = pd.DataFrame(
        computed, columns=["maturity_status", "entry_date", "exit_date", "target_return"], index=labelled.index)
    days = []
    summaries = {}
    frozen_boundaries = [item["prospective_start_after"] for item in entries if "prospective_start_after" in item]
    for track in ("PROSPECTIVE", "TIMELY_NONPROSPECTIVE", "HISTORICAL"):
        recorded = sorted((item for item in entries if item["track"] == track), key=lambda a: a["trade_date"])
        mature_days = sum(i["formal_session"] and i["trade_date"] in sessions and
                          sessions[i["trade_date"]] + HORIZON + 1 < len(calendar) for i in recorded)
        missing_signal_dates = []
        if track == "PROSPECTIVE" and frozen_boundaries:
            seen = {i["trade_date"] for i in recorded}
            missing_signal_dates = [d for d in calendar if d > min(frozen_boundaries) and d not in seen]
        models = {}
        track_data = labelled.loc[labelled.track.eq(track)]
        for objective in OBJECTIVES:
            groups = {}
            collection = track_data.loc[track_data.objective.eq(objective)]
            for support in GROUPS:
                bucket = collection if support == "ALL" else collection.loc[collection.support_group.eq(support)]
                previous = {}
                prior_signal = None
                diagnostics = []
                for item in recorded:
                    if not item["formal_session"]:
                        continue
                    signal = item["trade_date"]
                    if prior_signal is not None and sessions.get(signal, -10) - sessions.get(prior_signal, -10) != 1:
                        previous.clear()
                    group = bucket.loc[bucket.trade_date.eq(signal)]
                    pool = collection.loc[collection.trade_date.eq(signal)]
                    enough_time = signal in sessions and sessions[signal] + HORIZON + 1 < len(calendar)
                    detail = daily_report(group, pool, previous)
                    detail.update({"signal_matured": enough_time, "capture_status": item["status"], "trade_date": signal,
                                   "track": track, "objective": objective, "support_group": support})
                    prior_signal = signal
                    diagnostics.append(detail)
                days.extend(diagnostics)
                mature = [d for d in diagnostics if d["signal_matured"]]
                scored = sum(d["scores"] for d in mature)
                n = sum(d["labels"] for d in mature)
                values = {
                    "matured_scored_rows": scored, "mature_labels": n, "label_coverage": n / scored if scored else None,
                    "matured_days_with_scores": sum(d["scores"] > 0 for d in mature),
                    "matured_days_with_labels": sum(d["labels"] > 0 for d in mature),
                    "matured_days_with_no_scores": sum(d["scores"] == 0 for d in mature),
                    "mean_daily_rank_ic": mean_available(d["rank_ic"] for d in mature),
                    "rank_ic_days": sum(d["rank_ic"] is not None for d in mature),
                    "label_statuses": dict(Counter(bucket.maturity_status)),
                }
                for k in (10, 20):
                    top = [d[f"top{k}"] for d in mature]
                    values[f"top{k}"] = {"gross": mean_available(d["gross_economic_return"] for d in top),
                                          "gross_excess": mean_available(d["excess_over_all_scored_pool"] for d in top),
                                          "net_excess_by_per_side_bps": {str(c): mean_available(
                                              d["net_excess_by_per_side_bps"][str(c)] for d in top) for c in COSTS},
                                          "list_replacement": mean_available(d[f"top{k}"]["list_replacement"] for d in diagnostics)}
                groups[support] = values
            all_group = groups["ALL"]
            models[objective] = {
                "groups": groups,
                "time_and_label_gate_met": (track == "PROSPECTIVE" and all_group["matured_days_with_labels"] >= MIN_MATURED_DAYS
                                           and (all_group["label_coverage"] or 0.0) >= MIN_LABEL_COVERAGE),
            }
        summaries[track] = {"recorded_dates": len(recorded), "formal_signal_dates": sum(i["formal_session"] for i in recorded),
                            "matured_signal_dates": mature_days, "unrecorded_formal_signal_dates": missing_signal_dates,
                            "zero_prediction_formal_dates": [i["trade_date"] for i in recorded if i["formal_session"]
                                                             and all(not x["rows"] for x in i["models"].values())],
                            "models": models}
    return labelled.loc[:, LABEL_FIELDS], days, summaries


def run(ledger, output, seed, evidence_source, cutoff, official_path):
    entries, predictions = load_ledger(ledger)
    now = datetime.now(CHINA)
    begin = date.fromisoformat(min((r["trade_date"] for r in entries), default=now.date().isoformat()))
    calendar, raw = formal_tail(AShareDailyLoader().engine, begin, sorted(predictions.stock_code.unique()), now, cutoff)
    evidence = select_evidence(ledger, seed, evidence_source)
    official = {(r.stock_code, r.ex_date.date().isoformat()):
                (float(r.share_multiplier), float(r.cash_per_old_share), r.source_id)
                for r in load_actions(official_path).itertuples(index=False)}
    labels, days, report = evaluate(entries, predictions, calendar, raw, evidence, official)
    output.mkdir(parents=True)
    label_path = output / "prediction_labels.parquet"
    parquet_write(labels, label_path)
    day_path = output / "daily_metrics.json"
    atomic_json(day_path, days)
    atomic_json(output / "maturity_audit.json", {
        "version": VERSION, "research_only": True, "evaluated_at": now.isoformat(),
        "formal_through": calendar[-1] if calendar else None, "company_evidence_sha256": evidence["sha256"],
        "issuer_actions_sha256": hash_file(official_path), "input_manifests": entries,
        "label_rows": len(labels), "label_statuses": dict(Counter(labels.maturity_status)),
        "files": {f.name: hash_file(f) for f in [label_path, day_path]}, "tracks": report,
        "forward_rules": {"label": "T+1 open to T+6 open", "min_matured_days": MIN_MATURED_DAYS,
                          "min_label_coverage": MIN_LABEL_COVERAGE, "costs_bps_per_side": list(COSTS),
                          "pool_reference": "paired all-scored stocks with observed labels, when coverage passes",
                          "top_k_picks_use_frozen_scores_before_label_availability": True},
    })
    print(f"Evaluated {len(labels)} original prediction IDs across {len(entries)} immutable manifests", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("ledger", "out"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    p.add_argument("--evidence-seed", type=Path)
    p.add_argument("--evidence-source", type=Path)
    p.add_argument("--cutoff", type=date.fromisoformat)
    p.add_argument("--official-actions", type=Path, default=OFFICIAL)
    a = p.parse_args()
    run(a.ledger, a.out, a.evidence_seed, a.evidence_source, a.cutoff, a.official_actions)


if __name__ == "__main__":
    main()
