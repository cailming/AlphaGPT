"""Independent research-subset qualification and economic-label dataset (F1/F2)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import p4_action_pipeline as source
from .p4_corporate_actions import load_actions
from .p4_mvp import FEATURES
from .p4_subset_inputs import (hash_file, parquet_read, parquet_write, read_verified_factor,
                               ORIGINAL_TEST_EXCLUDED, VERSION as INPUT_VERSION)


VERSION = "P4_SUBSET_GATED_ECONOMIC_V1"
TRUSTED = "TRUSTED_RESEARCH"
LIMITED = "EXPLORATORY_ONLY"
UNUSABLE = "UNAVAILABLE"
FEATURE_LOOKBACK = 21
TRANSFORMER_SEQUENCE = 16
NOTICE_INDEX_SHA256 = "c262fb8e7bc87bf80f2c2f89286b41f98507af32f55177e99ceade84550232e2"
CONFLICT = re.compile(r"CONFLICT|MISMATCH|date order|DUPLICATE_EX_DATE", re.I)
EX_FIELDS = {"sina_dividend": "除权除息日", "sina_rights": "除权日",
             "cninfo_dividend": "除权日", source.BAO: "dividOperateDate"}
GUARD = {"legacy_training_gate": "CLOSED", "strict_complete_verified_assertions_created": 0,
         "strict_can_emit_training_labels": False, "full_historical_action_coverage_certified": False,
         "historical_st_complete": False, "current_st_backfilled": False}


@dataclass
class Evidence:
    actions: dict[str, set[str]]
    conflicts: dict[str, set[str]]
    provider_years: dict[str, set[int]]
    verified: dict[tuple[str, str], tuple[float, float, str]]
    factor_days: dict[str, set[str]]
    provenance: dict


def time_split(signal, exit_day, validation, test):
    if signal < validation:
        return "train" if exit_day < validation else "purged"
    if signal < test:
        return "validation" if exit_day < test else "purged"
    return "test"


def economic_return(entry, exit_price, ordered_events):
    shares = 1.0
    cash = 0.0
    for multiplier, cash_per_share, _ in ordered_events:
        cash += shares * cash_per_share
        shares *= multiplier
    return (shares * exit_price + cash) / entry - 1.0


def load_evidence(inputs, manifest, action_root, notice_index, official_actions):
    with patch("sys.argv", ["p4_action_pipeline", "status", "--output", str(action_root)]):
        args = source.parse_args()
    archive = source.Evidence(args)
    _, checkpoint = source.configure(archive, args)
    if (archive.identity["test_start_EXCLUDED"] != manifest["original_test_excluded"]
            or archive.identity["start"] > manifest["price_begin"]):
        raise ValueError("Source evidence identity does not match historical label period")
    pointer_path = action_root / "latest_report.json"
    pointer = source.read_json(pointer_path)
    strict = pointer["summary"]
    report_dir = Path(pointer["report_dir"])
    if (strict != source.read_json(report_dir / "pipeline_summary.json") or strict["training_gate"] != "CLOSED"
            or any(strict[key] for key in ("complete_verified_assertions_created", "can_emit_training_labels",
                                           "full_historical_action_coverage_certified"))
            or any(strict[key] != value for key, value in archive.identity.items())):
        raise ValueError("Old gate or frozen evidence changed; never use it as a subset certificate")

    actions = defaultdict(set)
    conflicts = defaultdict(set)
    provider_years = defaultdict(set)
    years = range(date.fromisoformat(manifest["price_begin"]).year,
                  date.fromisoformat(manifest["price_end"]).year + 1)
    for stock in archive.codes:
        status = {}
        for task in archive.tasks(stock):
            attempt = archive.effective(task)
            status[task.endpoint, task.year] = attempt.category if attempt else "NOT_FETCHED"
            if attempt is None or not attempt.rows:
                continue
            for row in attempt.rows:
                if task.endpoint == "sina_dividend" and row["进度"].strip() != "实施":
                    continue
                ex_day = row.get(EX_FIELDS[task.endpoint], "")
                if manifest["price_begin"] <= ex_day <= manifest["price_end"]:
                    actions[stock].add(date.fromisoformat(ex_day).isoformat())
        if status.get(("sina_rights", 0)) == source.SUCCESS:
            for year in years:
                suppliers = [("sina_dividend", 0), ("cninfo_dividend", 0), (source.BAO, year)]
                if sum(status.get(key) == source.SUCCESS for key in suppliers) >= 2:
                    provider_years[stock].add(year)

    exclusions = report_dir / "exceptions.csv"
    for row in pd.read_csv(exclusions, dtype="string", keep_default_na=False).itertuples(index=False):
        value = row.date_or_year
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) and CONFLICT.search(row.category):
            conflicts[row.stock_code].add(date.fromisoformat(value).isoformat())
    if hash_file(notice_index) != NOTICE_INDEX_SHA256:
        raise ValueError("The audited BSE notice index changed; check the source digest before adoption")
    notices = pd.read_csv(notice_index, dtype="string", keep_default_na=False)
    if len(notices) != 76 or notices.duplicated(["stock_code", "ex_date"]).any():
        raise ValueError("Unexpected audited BSE notice inventory")
    for row in notices.itertuples(index=False):
        actions[row.stock_code].add(date.fromisoformat(row.ex_date).isoformat())

    verified = {}
    for row in load_actions(official_actions).itertuples(index=False):
        ex = row.ex_date.date().isoformat()
        if ex >= manifest["original_test_excluded"]:
            raise ValueError("Official action touches excluded old holdout")
        actions[row.stock_code].add(ex)
        verified[row.stock_code, ex] = (float(row.share_multiplier), float(row.cash_per_old_share), row.source_id)

    plan_path = inputs / "factor_plan.json"
    plan_hash = hash_file(plan_path)
    records = {}
    stats = Counter()
    for job in source.read_json(plan_path):
        path = inputs / "factors" / (job["stock_code"] + ".json")
        if not path.is_file():
            stats["NOT_CAPTURED"] += 1
            continue
        item = read_verified_factor(path, job, manifest, plan_hash)
        stats[item["status"]] += 1
        if item["status"] == "SUCCESS":
            records[job["stock_code"]] = {row[1] for row in item["rows"]}

    controls_present = {(code, day) for (code, day) in verified if code in records and day in records[code]}
    if records and not controls_present:
        raise ValueError("No independent factor series reproduced any issuer-verified positive control")
    info = {"archive_identity": archive.identity, "strict_checkpoint_state": checkpoint["state"],
            "strict_report_pointer_sha256": hash_file(pointer_path), "conflict_sha256": hash_file(exclusions),
            "notice_index_sha256": hash_file(notice_index), "issuer_events_sha256": hash_file(official_actions),
            "factor_plan_sha256": plan_hash, "factor_captures": dict(stats),
            "factor_issuer_positive_controls": sorted(f"{code}/{day}" for code, day in controls_present)}
    return Evidence(actions, conflicts, provider_years, verified, records, info)


def assess_window(stock, first, signal, entry, exit_day, evidence):
    """Evaluate dated evidence; an entry-open ex-date never earns old-holder rights."""
    actions = evidence.actions.get(stock, set())
    conflicts = evidence.conflicts.get(stock, set())
    factor_days = evidence.factor_days.get(stock)
    evidence_dates = actions | conflicts | (factor_days or set())
    if any(first <= day <= signal for day in evidence_dates):
        return UNUSABLE, "UNADJUSTED_FEATURE_ACTION_WINDOW", []
    future_days = sorted(day for day in evidence_dates if entry <= day <= exit_day)
    if any(day in conflicts for day in future_days):
        return UNUSABLE, "UNRESOLVED_ACTION_CONFLICT", []
    if any((stock, day) not in evidence.verified or day not in (factor_days or set()) for day in future_days):
        return UNUSABLE, "UNVERIFIED_ACTION_DURING_ENTRY_OR_HOLD", []
    if factor_days is None:
        return LIMITED, "NO_INDEPENDENT_FACTOR_WINDOW", []
    years = set(range(int(first[:4]), int(exit_day[:4]) + 1))
    if not years.issubset(evidence.provider_years.get(stock, set())):
        return LIMITED, "INSUFFICIENT_DATED_SOURCE_SUPPORT", []
    payoff = [evidence.verified[stock, day] for day in future_days if day > entry]
    return TRUSTED, "ISSUER_ADJUSTED" if payoff else "CORROBORATED_NO_ACTION_WINDOW", payoff


def build_dataset(raw, candidates, manifest, evidence):
    calendar = np.asarray(manifest["calendar"], dtype="U10")
    ordinal = {day: ix for ix, day in enumerate(calendar)}
    if raw.duplicated(["stock_code", "trade_date"]).any() or candidates.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate stock/day input")
    if len(calendar) != len(set(calendar)) or not (calendar[:-1] < calendar[1:]).all():
        raise ValueError("Calendar is not unique chronological time")
    if set(raw.trade_date) != set(calendar) or calendar[-1] >= manifest["original_test_excluded"]:
        raise ValueError("Snapshot calendar or old holdout boundary violated")
    limit = int(manifest["pool_config"]["top_n"])
    if (len(candidates) == 0 or candidates.liquidity_rank.max() > limit
            or candidates.groupby("trade_date").size().max() > limit
            or not np.isfinite(candidates.loc[:, list(FEATURES)].to_numpy(dtype=float)).all()):
        raise ValueError("Unusable causal candidate feature input")

    evaluation = []
    horizon = int(manifest["pool_config"]["hold_sessions"])
    max_label = float(manifest["pool_config"]["max_abs_label"])
    grouped_prices = raw.loc[raw.stock_code.isin(candidates.stock_code.unique())].groupby("stock_code", sort=False)
    for stock, items in candidates.groupby("stock_code", sort=False):
        prices = grouped_prices.get_group(stock).set_index("trade_date").reindex(calendar)
        opening, close, high, low, volume, amount, pre_close = (
            prices[name].to_numpy(dtype=float) for name in ("open", "close", "high", "low", "volume", "amount", "pre_close"))
        prior = np.r_[np.nan, close[:-1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            jump = ((abs(opening / prior - 1) > .30) | (abs(close / prior - 1) > .32))
        bar_ok = (np.isfinite(opening + close + high + low + volume + amount) & (low > 0) & (volume > 0)
                  & (amount > 0) & (high >= np.maximum(opening, close)) & (low <= np.minimum(opening, close)))
        reference_ok = np.isfinite(pre_close) & (pre_close > 0)
        factor_days = evidence.factor_days.get(stock)
        blocked_feature_days = (evidence.actions.get(stock, set()) | evidence.conflicts.get(stock, set())
                                | (factor_days or set()))
        for row in items.itertuples(index=False):
            signal = row.trade_date
            index = ordinal[signal]
            if index != row.session_id:
                raise ValueError("Causal feature trading session mismatch")
            first_i, entry_i, exit_i = index - FEATURE_LOOKBACK + 1, index + 1, index + horizon + 1
            causal_ok = (first_i > 0 and bar_ok[first_i:index+1].all() and reference_ok[first_i:index+1].all()
                         and not jump[first_i:index+1].any())
            sequence_first = first_i - TRANSFORMER_SEQUENCE + 1
            sequence_ok = (sequence_first > 0 and factor_days is not None and bar_ok[sequence_first:index+1].all()
                           and reference_ok[sequence_first:index+1].all() and not jump[sequence_first:index+1].any()
                           and not any(calendar[sequence_first] <= day <= signal for day in blocked_feature_days)
                           and set(range(int(calendar[sequence_first][:4]), int(signal[:4]) + 1)).issubset(
                               evidence.provider_years.get(stock, set())))
            split = "outside"
            entry, exit_day, dataset_label, diagnostic_raw = "", "", None, None
            grade, reason, event_ids = UNUSABLE, "INSUFFICIENT_PRICE_PATH", ""
            if exit_i >= len(calendar):
                reason = "EXIT_OUTSIDE_FROZEN_PRICES"
            elif first_i < 1 or not bar_ok[first_i:exit_i+1].all() or not reference_ok[first_i:exit_i+1].all():
                reason = "INVALID_OR_MISSING_DAILY_BAR"
            else:
                first, entry, exit_day = str(calendar[first_i]), str(calendar[entry_i]), str(calendar[exit_i])
                split = time_split(signal, exit_day, manifest["val_start"], manifest["test_start"])
                diagnostic_raw = float(opening[exit_i] / opening[entry_i] - 1.0)
                if split == "purged":
                    reason = "FORWARD_LABEL_CROSSES_SPLIT"
                elif not causal_ok:
                    reason = "ABNORMAL_UNADJUSTED_FEATURE_PRICE_JUMP"
                else:
                    grade, reason, payoff = assess_window(stock, first, signal, entry, exit_day, evidence)
                    if grade != UNUSABLE and any(jump[position] and (stock, str(calendar[position])) not in evidence.verified
                                                for position in range(entry_i, exit_i + 1)):
                        grade, reason = UNUSABLE, "UNEXPLAINED_FORWARD_PRICE_JUMP"
                    if grade == TRUSTED:
                        gross = economic_return(opening[entry_i], opening[exit_i], payoff)
                        if not np.isfinite(gross) or abs(gross) > max_label:
                            grade, reason = UNUSABLE, "ECONOMIC_LABEL_OUTSIDE_BOUND"
                        else:
                            dataset_label = float(gross)
                            event_ids = "|".join(event[2] for event in payoff)
            evaluation.append({"stock_code": stock, "trade_date": signal, "entry_date": entry, "exit_date": exit_day,
                               "split": split, "sample_grade": grade, "reason": reason, "target_return": dataset_label,
                               "raw_return_diagnostic_only": diagnostic_raw, "event_sources": event_ids,
                               "causal_feature_ready": bool(causal_ok), "sequence_feature_evidence_ready": bool(sequence_ok)})
    evaluated = pd.DataFrame(evaluation)
    return candidates.merge(evaluated, on=["stock_code", "trade_date"], how="inner", validate="one_to_one")


def distribution(values):
    column = values.to_numpy(dtype=float)
    if not len(column):
        return {"n": 0}
    result = {f"p{int(q * 100):02d}": float(np.quantile(column, q)) for q in (.01, .05, .5, .95, .99)}
    return result | {"n": len(column), "mean": float(column.mean()), "std": float(column.std()),
                     "min": float(column.min()), "max": float(column.max()), "positive_fraction": float((column > 0).mean())}


def finalize(inputs, output, action_root, notices, official):
    manifest_path = inputs / "inputs.json"
    manifest = source.read_json(manifest_path)
    if manifest["version"] != INPUT_VERSION or manifest["original_test_excluded"] != ORIGINAL_TEST_EXCLUDED:
        raise ValueError("Unsupported input version or cutoff")
    for filename, sha in manifest["files"].items():
        if hash_file(inputs / filename) != sha:
            raise ValueError("Input snapshot digest mismatch")
    evidence = load_evidence(inputs, manifest, action_root, notices, official)
    print("Evidence loaded:", evidence.provenance["factor_captures"], flush=True)
    features = parquet_read(inputs / "causal_features.parquet")
    result = build_dataset(parquet_read(inputs / "formal_ohlcv.parquet"), features, manifest, evidence)
    trusted = result.loc[result.sample_grade.eq(TRUSTED)].copy()
    if len(result) != len(features) or not trusted.target_return.notna().all() or result.loc[
            ~result.sample_grade.eq(TRUSTED), "target_return"].notna().any():
        raise ValueError("Labels leaked outside the independently qualified sample set")
    output.mkdir(parents=True)
    training_columns = ["stock_code", "trade_date", "session_id", "liquidity_rank", "entry_date", "exit_date", "split",
                        "sample_grade", "reason", "event_sources", "target_return", "sequence_feature_evidence_ready", *FEATURES]
    indicators = features.merge(result[["stock_code", "trade_date", "causal_feature_ready"]],
                                on=["stock_code", "trade_date"], validate="one_to_one")
    inference = indicators.loc[indicators.causal_feature_ready].drop(columns="causal_feature_ready")
    parquet_write(trusted[training_columns], output / "training_dataset.parquet")
    parquet_write(result[["stock_code", "trade_date", "entry_date", "exit_date", "split", "sample_grade", "reason",
                          "raw_return_diagnostic_only", "event_sources", "causal_feature_ready"]], output / "sample_quality.parquet")
    parquet_write(inference, output / "causal_inference_features.parquet")
    stocks = (result.groupby(["stock_code", "sample_grade"]).size().unstack(fill_value=0)
              .reindex(columns=[TRUSTED, LIMITED, UNUSABLE], fill_value=0))
    stocks["stock_grade"] = np.select([stocks[TRUSTED] > 0, stocks[LIMITED] > 0], [TRUSTED, LIMITED], default=UNUSABLE)
    stocks.to_csv(output / "stock_quality.csv")
    daily = result.groupby(["trade_date", "sample_grade"]).size().unstack(fill_value=0).reindex(
        columns=[TRUSTED, LIMITED, UNUSABLE], fill_value=0)
    daily.to_csv(output / "daily_quality.csv")
    fold_counts = {}
    for fold in ("train", "validation", "test"):
        rows = trusted.loc[trusted.split.eq(fold)]
        fold_counts[fold] = {"samples": len(rows), "stocks": int(rows.stock_code.nunique()),
                             "days": int(rows.trade_date.nunique()), "first_signal": rows.trade_date.min() if len(rows) else None,
                             "last_signal": rows.trade_date.max() if len(rows) else None,
                             "last_label_exit": rows.exit_date.max() if len(rows) else None,
                             "label_distribution": distribution(rows.target_return)}
    summary = {"version": VERSION, **GUARD, "subset_training_allowed": all(item["samples"] > 0 for item in fold_counts.values()),
               "subset_training_research_only": True, "reference_price_not_action_clearance": True,
               "old_test_excluded": ORIGINAL_TEST_EXCLUDED, "test_kind": "HISTORICAL_TIME_HOLDOUT_NOT_VIRGIN_MARKET_DATA",
               "feature_names": list(FEATURES), "feature_lookback": FEATURE_LOOKBACK,
               "sequence_eligibility_length": TRANSFORMER_SEQUENCE,
               "val_start": manifest["val_start"], "test_start": manifest["test_start"], "price_end": manifest["price_end"],
               "inputs_sha256": hash_file(manifest_path), "past_only_candidate_selection": True,
               "future_quality_does_not_rerank_or_backfill_candidates": True,
               "pool": manifest["pool"], "candidate_rows": len(result),
               "candidate_stocks": int(result.stock_code.nunique()), "trusted_stocks": int(trusted.stock_code.nunique()),
               "daily_trusted_stocks_median": float(daily[TRUSTED].median()),
               "daily_trusted_stocks_max": int(daily[TRUSTED].max()), "labelled_rows": len(trusted),
               "stock_grades": {name: int(value) for name, value in stocks.stock_grade.value_counts().items()},
               "sample_grades": {name: int(value) for name, value in result.sample_grade.value_counts().items()},
               "by_reason": {name: int(value) for name, value in result.reason.value_counts().items()},
               "splits": fold_counts, "label_distribution": distribution(trusted.target_return),
               "evidence": evidence.provenance, "files": {path.name: hash_file(path) for path in output.iterdir()}}
    source.atomic_json(output / "subset_audit.json", summary)
    print("Qualified", len(trusted), "rows across", summary["trusted_stocks"], "stocks", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--actions", type=Path, default=source.OUTPUT)
    parser.add_argument("--notice-index", required=True, type=Path)
    parser.add_argument("--official-actions", type=Path, default=Path("ashare/data/corporate_actions_verified_examples.csv"))
    args = parser.parse_args()
    finalize(args.inputs, args.out, args.actions, args.notice_index, args.official_actions)


if __name__ == "__main__":
    main()
