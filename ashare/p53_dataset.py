"""Chronological monthly research cohorts and audited P5 development datasets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import p4_action_pipeline as ledger
from .p4_mvp import FEATURES, PoolConfig
from .p4_subset_dataset import FEATURE_LOOKBACK, GUARD, TRANSFORMER_SEQUENCE, load_evidence
from .p4_subset_inputs import ORIGINAL_TEST_EXCLUDED, RISK_FILE, hash_file, parquet_read, parquet_write
from .p4_subset_transformer import fit_sequence_scaler
from .p5_dataset import Boundaries, FOLDS, NOTICE_INDEX, VERSION, label_windows, read_recent_evidence
from .p5_evidence import RECENT_SOURCES, capture_observations
from .p5_sequences import VERSION as SEQUENCE_POLICY
from .p5_sequences import build_causal_population, price_window_flags, sequence_indexes

POOL_POLICY = "P53_MONTHLY_PRIOR_SESSION_LIQUIDITY_V1"
EVIDENCE_START = date.fromisoformat(ORIGINAL_TEST_EXCLUDED)
ONE_DAY = timedelta(days=1)
PASS_REASONS = {"OBSERVED_CLEAN_WINDOW", "ISSUER_ADJUSTED"}


def monthly_memberships(features, calendar, begin, end, *, size=400, retain=300, lookback=63, minimum=20):
    """Rank from prior sessions only, with bounded incumbent retention."""
    if not 1 <= size <= 500 or not 0 <= retain <= size or not 1 <= minimum <= lookback:
        raise ValueError("Invalid monthly cohort parameters")
    iso = pd.DatetimeIndex(calendar).strftime("%Y-%m-%d")
    iso = pd.Index(iso)
    if not iso.is_unique or not iso.is_monotonic_increasing:
        raise ValueError("The calendar must be unique and chronological")
    if features.duplicated(["stock_code", "session_id"]).any():
        raise ValueError("Duplicate feature stock/session")
    members = []
    previous = set()
    for month in pd.period_range(begin, end, freq="M"):
        start = month.start_time.date().isoformat()
        cutoff = int(iso.searchsorted(start, side="left")) - 1
        if cutoff < 0:
            continue
        history = features.loc[features.session_id.between(max(0, cutoff - lookback + 1), cutoff)]
        rank = (history.groupby("stock_code", as_index=False)
                .agg(eligible_days=("session_id", "size"), median_rank=("liquidity_rank", "median"),
                     last_seen=("session_id", "max")))
        rank = (rank.loc[rank.eligible_days.ge(minimum)]
                .sort_values(["median_rank", "eligible_days", "last_seen", "stock_code"],
                             ascending=[True, False, False, True], kind="stable"))
        incumbent = rank.loc[rank.stock_code.isin(previous)].head(retain)
        added = rank.loc[~rank.stock_code.isin(incumbent.stock_code)].head(size - len(incumbent))
        chosen = pd.concat([incumbent, added], ignore_index=True)
        current = set(chosen.stock_code)
        chosen["retained"] = chosen.stock_code.isin(previous)
        chosen["month"] = str(month)
        chosen["selection_cutoff"] = iso[cutoff]
        chosen["pool_rank"] = np.arange(1, len(chosen) + 1)
        members.append(chosen)
        previous = current
    if not members:
        raise ValueError("No month has a usable preceding formal session")
    return pd.concat(members, ignore_index=True)


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if start > end:
            raise ValueError("Inverted evidence interval")
        if merged and start <= merged[-1][1] + ONE_DAY:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def uncovered(intervals, start, end):
    for lo, hi in intervals:
        if hi < start:
            continue
        if lo > end:
            break
        if lo > start:
            yield start, min(end, lo - ONE_DAY)
        start = max(start, hi + ONE_DAY)
    if start <= end:
        yield start, end


@dataclass
class RecentEvidence:
    coverage: dict
    events: dict
    provenance: list

    def covered(self, stock, start, end):
        return any(lo <= start <= end <= hi for lo, hi in self.coverage.get(stock, ()))


def collect_recent(roots):
    coverage, events, facts = defaultdict(list), defaultdict(set), []
    for root in roots:
        plan, dual, observed, audit = read_recent_evidence(root / "inputs")
        dates = tuple(date.fromisoformat(plan[key]) for key in ("start", "end"))
        for stock in dual:
            coverage[stock].append(dates)
        for stock, action_dates in observed.items():
            events[stock].update(action_dates)
        facts.append({"root": str(root), "stocks": len(dual), **audit})
    return RecentEvidence({k: merge_intervals(v) for k, v in coverage.items()}, events, facts)


@dataclass
class ResearchEvidence:
    legacy: object
    legacy_begin: date
    recent: RecentEvidence

    def covered(self, stock, first, last):
        first, last = date.fromisoformat(first), date.fromisoformat(last)
        if first < EVIDENCE_START:
            end = min(last, EVIDENCE_START - ONE_DAY)
            years = set(range(first.year, end.year + 1))
            if (first < self.legacy_begin or stock not in self.legacy.factor_days
                    or not years.issubset(self.legacy.provider_years.get(stock, set()))):
                return False
        return last < EVIDENCE_START or self.recent.covered(stock, max(first, EVIDENCE_START), last)

    def observed_action_dates(self, stock):
        return (self.legacy.actions.get(stock, set()) | self.legacy.conflicts.get(stock, set())
                | self.legacy.factor_days.get(stock, set()) | self.recent.events.get(stock, set()))

    def inference_snapshot(self, pool, end):
        rows = {}
        observed = []
        for stock in sorted(pool):
            intervals = self.recent.coverage.get(stock, [])
            if not intervals:
                continue
            first, last = intervals[-1]
            if last > end:
                raise ValueError("Evidence exceeds frozen price scope")
            actions = [[day, day] for day in sorted(self.observed_action_dates(stock))
                       if first.isoformat() <= day <= last.isoformat()]
            rows[stock] = {"covered_from": first.isoformat(), "covered_through": last.isoformat(), "actions": actions}
            observed.append(last)
        return {"policy": "P5_DUAL_OBSERVED_RESEARCH_WINDOW_V1",
                "event_date_role": "price-effective quarantine, not inferred publication time",
                "max_covered_date": max(observed).isoformat() if observed else None,
                "observed_at_utc": max((i["latest_observation_utc"] for i in self.recent.provenance
                                        if i["latest_observation_utc"]), default=None),
                "stocks": rows}


def frozen_source(source):
    inputs = source / "inputs"
    manifest = ledger.read_json(inputs / "inputs.json")
    file = inputs / "formal_ohlcv.parquet"
    digest = manifest["files"][file.name]
    if hash_file(file) != digest:
        raise ValueError("Frozen formal bars failed digest verification")
    return manifest, file, digest


def packs(out, jobs, limit):
    chunks = defaultdict(list)
    for stock, lo, hi, weight in jobs[:limit]:
        provider = ledger.bao._provider_code(stock)
        if provider:
            chunks[(lo, hi)].append({"stock_code": stock, "provider_code": provider, "coverage_opportunities": weight})
    result = []
    for (lo, hi), subset in sorted(chunks.items()):
        for index in range(0, len(subset), 220):
            root = out / "evidence" / f"batch-{len(result) + 1:02d}"
            folder = root / "inputs"
            folder.mkdir(parents=True)
            plan = folder / "evidence_plan.json"
            payload = {"kind": "STABLE_RESEARCH_POOL_INTERVAL_COMPLEMENT", "start": lo.isoformat(),
                       "end": hi.isoformat(), "providers": list(RECENT_SOURCES), "jobs": subset[index:index + 220]}
            ledger.atomic_json(plan, payload)
            digest = hash_file(plan)
            ledger.atomic_json(folder / "inputs.json", {"files": {plan.name: digest}})
            result.append({"root": str(root), "sha256": digest, "jobs": len(payload["jobs"]),
                           "start": payload["start"], "end": payload["end"]})
    return result


def interval_jobs(anchors, idx, calendar, bounds, config, recent):
    iso = pd.DatetimeIndex(calendar).date
    scope = {}
    before = Counter()
    for row in anchors.iloc[idx].itertuples(index=False):
        signal = row.trade_date
        exit_day = row.session_id + config.hold_sessions + 1
        first = row.session_id - FEATURE_LOOKBACK - TRANSFORMER_SEQUENCE + 2
        fold = bounds.split(signal, iso[exit_day].isoformat()) if exit_day < len(iso) else None
        if fold is None or first < 0 or iso[exit_day] < EVIDENCE_START:
            continue
        lo, hi = max(EVIDENCE_START, iso[first]), iso[exit_day]
        before[f"{fold}_recent_windows"] += 1
        if recent.covered(row.stock_code, lo, hi):
            before[f"{fold}_already_dual_covered"] += 1
            continue
        # Prioritize coverage of evaluation windows, never use observed return values.
        state = scope.setdefault(row.stock_code, [lo, hi, {}])
        state[0] = min(state[0], lo)
        state[1] = max(state[1], hi)
        state[2][fold] = state[2].get(fold, 0) + 1
    jobs = []
    for stock, (lo, hi, counts) in scope.items():
        opportunities = int(sum(counts.values()))
        weight = counts.get("validation", 0) + counts.get("test", 0)
        jobs.extend((stock, a, b, opportunities, weight)
                    for a, b in uncovered(recent.coverage.get(stock, ()), lo, hi))
    jobs.sort(key=lambda row: (-row[4], -row[3], row[0], row[1].isoformat()))
    return [row[:4] for row in jobs], dict(before)


def plan(source, supplements, out, validation, diagnostic, size, retain, limit):
    out = out.resolve()
    source = source.resolve(strict=True)
    roots = list(dict.fromkeys([source, *(path.resolve(strict=True) for path in supplements)]))
    if not 300 <= size <= 500 or not 0 <= limit <= 220:
        raise ValueError("Research pool size must be 300–500 and targeted interval cap 0–220")
    manifest, prices, digest = frozen_source(source)
    bounds = Boundaries(**(manifest["bounds"] | {"val_start": validation, "test_start": diagnostic}))
    config = PoolConfig(**manifest["pool_config"])
    if hash_file(RISK_FILE) != manifest["risk_events_sha256"]:
        raise ValueError("Frozen risk events changed")
    raw = parquet_read(prices)
    start = date.fromisoformat(bounds.train_start).replace(day=1) - timedelta(days=100)
    events = pd.read_csv(RISK_FILE, dtype={"stock_code": "string"})
    population = build_causal_population(raw, start, date.fromisoformat(bounds.price_end), config, events=events)
    if population.calendar.strftime("%Y-%m-%d").tolist() != manifest["calendar"]:
        raise ValueError("Frozen calendar changed")
    memberships = monthly_memberships(population.features, population.calendar, bounds.train_start,
                                      bounds.price_end, size=size, retain=retain)
    signals = population.features.assign(month=population.features.trade_date.str.slice(0, 7))
    anchors = (signals.loc[signals.trade_date.ge(bounds.train_start)]
               .merge(memberships[["month", "stock_code"]], on=["month", "stock_code"], validate="many_to_one")
               .drop(columns="month"))
    members = memberships.stock_code.unique()
    if not len(anchors):
        raise ValueError("No causal anchors in the research cohort")
    features = population.features.loc[population.features.stock_code.isin(members)].reset_index(drop=True)
    idx, _ = sequence_indexes(features, anchors, TRANSFORMER_SEQUENCE)
    recent = collect_recent(roots)
    missing, before = interval_jobs(anchors, idx, population.calendar, bounds, config, recent)

    folder = out / "inputs"
    folder.mkdir(parents=True)
    (folder / prices.name).symlink_to(prices)
    files = {prices.name: digest}
    for name, frame in (("monthly_memberships.parquet", memberships), ("anchors.parquet", anchors),
                        ("features.parquet", features)):
        file = folder / name
        parquet_write(frame, file)
        files[name] = hash_file(file)
    batches = packs(out, missing, limit)
    by_month = []
    for month, sub in memberships.groupby("month", sort=True):
        by_month.append({"month": month, "size": len(sub), "cutoff": sub.selection_cutoff.iloc[0],
                         "retained": int(sub.retained.sum())})
    report = {"policy": POOL_POLICY, "bounds": asdict(bounds), "pool_config": asdict(config),
              "target_monthly_size": size, "reserved_incumbent_slots": retain, "months": by_month,
              "ever_members": int(len(members)), "initially_dual_supported": len(recent.coverage),
              "already_available": before, "missing_intervals": len(missing),
              "planned_new_jobs": sum(a["jobs"] for a in batches), "batches": batches,
              "prior_evidence_roots": [str(path) for path in roots],
              "prior_evidence_digests": [row["recent_plan_sha256"] for row in recent.provenance],
              "market_population": population.stats, "calendar": manifest["calendar"],
              "risk_events_sha256": manifest["risk_events_sha256"], "files": files}
    ledger.atomic_json(folder / "research_pool_plan.json", report)
    print({k: report[k] for k in ("months", "ever_members", "already_available", "planned_new_jobs", "batches")}, flush=True)
    return report


def fold_for(day, bounds):
    return "train" if day < bounds.val_start else "validation" if day < bounds.test_start else "test"


def audit_quality(quality, memberships, raw, calendar, bounds):
    days = pd.DataFrame({"trade_date": calendar})
    days = days.loc[days.trade_date.between(bounds.train_start, bounds.price_end)].assign(
        month=lambda frame: frame.trade_date.str.slice(0, 7))
    expected = days.merge(memberships[["month", "stock_code"]], on="month", validate="many_to_many")
    keys = ["stock_code", "trade_date"]
    expected = expected.merge(quality[keys + ["reason"]], on=keys, how="left", validate="one_to_one")
    missing = expected.reason.isna()
    if missing.any():
        src = raw.loc[raw.stock_code.isin(memberships.stock_code.unique())].copy()
        flags = price_window_flags(src, pd.to_datetime(calendar), allow_auxiliary_fallback=True)
        flags["trade_date"] = flags.trade_date.dt.strftime("%Y-%m-%d")
        observed = expected.loc[missing, keys].merge(flags[keys + ["valid_price_window"]],
                                                     on=keys, how="left", validate="one_to_one")
        expected.loc[missing, "reason"] = np.select(
            [observed.valid_price_window.isna(), observed.valid_price_window.eq(False)],
            ["MISSING_FORMAL_PRICE", "INVALID_OR_NONCONSECUTIVE_CAUSAL_PRICE_WINDOW"],
            default="PRICE_LIQUIDITY_FEATURE_OR_RISK_GATE")
    expected["split"] = expected.trade_date.map(lambda day: fold_for(day, bounds))
    expected["valid_label"] = expected.reason.isin(PASS_REASONS)
    observed = expected.groupby(["split", "trade_date"], as_index=False).agg(
        candidate_stocks=("stock_code", "size"), valid_labels=("valid_label", "sum"))
    daily = days[["trade_date"]].assign(split=lambda f: f.trade_date.map(lambda d: fold_for(d, bounds)))
    daily = daily.merge(observed, how="left", on=["split", "trade_date"]).fillna(0)
    daily[["candidate_stocks", "valid_labels"]] = daily[["candidate_stocks", "valid_labels"]].astype(int)
    return expected.drop(columns="month"), daily


def audit_inference(anchors, ready, calendar, evidence):
    """Measure retrospective signal eligibility without asking whether a label matured."""
    calendar = pd.DatetimeIndex(calendar).strftime("%Y-%m-%d")
    accepted = np.zeros(len(anchors), dtype=bool)
    context = FEATURE_LOOKBACK + TRANSFORMER_SEQUENCE - 2
    known = {stock: evidence.observed_action_dates(stock) for stock in anchors.stock_code.unique()}
    for index, stock, day, session in anchors.iloc[ready][["stock_code", "trade_date", "session_id"]].itertuples():
        if session < context:
            continue
        first = calendar[session - context]
        accepted[index] = (evidence.covered(stock, first, day)
                           and not any(first <= event <= day for event in known[stock]))
    price_ready = np.zeros(len(anchors), dtype=bool)
    price_ready[ready] = True
    return (anchors[["trade_date"]].assign(price_qualified_sequences=price_ready,
                                          research_inference_stocks=accepted)
            .groupby("trade_date", as_index=False).sum())


def summarize(dataset, quality, daily):
    splits = {}
    for fold in FOLDS:
        select = dataset.loc[dataset.split.eq(fold)]
        sub = daily.loc[daily.split.eq(fold)]
        rejected = quality.loc[quality.split.eq(fold) & ~quality.valid_label, "reason"].value_counts()
        breadth = sub.valid_labels
        splits[fold] = {"samples": len(select), "stocks": int(select.stock_code.nunique()),
                        "candidate_days": len(sub), "days": int(breadth.gt(0).sum()),
                        "days_with_at_least_twenty_stocks": int(breadth.ge(20).sum()),
                        "median_valid_stocks_per_day": float(breadth.median()) if len(sub) else None,
                        "first_signal": select.trade_date.min() if len(select) else None,
                        "last_signal": select.trade_date.max() if len(select) else None,
                        "last_label_exit": select.exit_date.max() if len(select) else None,
                        "inference_days_with_100_to_300": int(sub.research_inference_stocks.between(100, 300).sum()),
                        "median_research_inference_stocks": float(sub.research_inference_stocks.median()) if len(sub) else None,
                        "rejection_reasons": {str(k): int(v) for k, v in rejected.items()}}
    return splits


def build(plan_root, out, v1, notice_index, action_root, official):
    plan_root = plan_root.resolve(strict=True)
    plan_path = plan_root / "inputs" / "research_pool_plan.json"
    info = ledger.read_json(plan_path)
    base = plan_path.parent
    if info["policy"] != POOL_POLICY or any(hash_file(base / f) != s for f, s in info["files"].items()):
        raise ValueError("Stable pool input digest or policy differs")
    cfg = PoolConfig(**info["pool_config"])
    bounds = Boundaries(**info["bounds"])
    raw, memberships, features, anchors = (parquet_read(base / name) for name in
        ("formal_ohlcv.parquet", "monthly_memberships.parquet", "features.parquet", "anchors.parquet"))
    roots = [Path(path) for path in info["prior_evidence_roots"]]
    for item in info["batches"]:
        path = Path(item["root"])
        if hash_file(path / "inputs" / "evidence_plan.json") != item["sha256"]:
            raise ValueError("Research cohort capture plan changed")
        roots.append(path)
    recent = collect_recent(roots)
    if [i["recent_plan_sha256"] for i in recent.provenance[:len(info["prior_evidence_roots"])]] != info["prior_evidence_digests"]:
        raise ValueError("Previously committed evidence root changed")
    if recent.coverage and not sum(row["recent_positive_event_matches"] for row in recent.provenance):
        raise ValueError("No joint positive event check across available evidence roots")
    old_inputs = v1 / "inputs"
    legacy_manifest = ledger.read_json(old_inputs / "inputs.json")
    evidence = ResearchEvidence(load_evidence(old_inputs, legacy_manifest, action_root, notice_index, official),
                                date.fromisoformat(legacy_manifest["price_begin"]), recent)
    calendar = pd.to_datetime(info["calendar"])
    ids, sequence = sequence_indexes(features, anchors, TRANSFORMER_SEQUENCE)
    dataset, accepted, quality = label_windows(raw, anchors, ids, calendar, bounds, cfg, evidence)
    quality, daily = audit_quality(quality, memberships, raw, info["calendar"], bounds)
    daily = daily.merge(audit_inference(anchors, ids, calendar, evidence), on="trade_date", how="left")
    for name in ("price_qualified_sequences", "research_inference_stocks"):
        daily[name] = daily[name].fillna(0).astype(int)
    if dataset.empty:
        dataset = dataset.assign(split=pd.Series(dtype=str), exit_date=pd.Series(dtype=str))
    splits = summarize(dataset, quality, daily)

    out.mkdir(parents=True)
    inputs = out / "inputs"
    inputs.symlink_to(base, target_is_directory=True)
    dest = out / "dataset"
    dest.mkdir()
    parquet_write(anchors, dest / "causal_anchors.parquet")
    parquet_write(quality, dest / "anchor_quality.parquet")
    daily.to_csv(dest / "daily_coverage.csv", index=False)
    parquet_write(memberships, dest / "monthly_memberships.parquet")
    ledger.atomic_json(dest / "inference_evidence.json", evidence.inference_snapshot(memberships.stock_code.unique(),
                                                                                    date.fromisoformat(bounds.price_end)))
    qualified = all(s["samples"] for s in splits.values())
    seq_facts = {}
    scaler_facts = {}
    if qualified:
        lookup = np.full(len(anchors), -1, dtype=np.int64)
        lookup[ids] = np.arange(len(ids))
        selected_sequences = sequence[lookup[accepted]]
        history = np.unique(sequence)
        compact = np.full(len(features), -1, dtype=np.int64)
        compact[history] = np.arange(len(history))
        features = features.iloc[history].reset_index(drop=True)
        selected_sequences = compact[selected_sequences]
        parquet_write(features, dest / "causal_inference_features.parquet")
        parquet_write(dataset, dest / "training_dataset.parquet")
        transforms = out / "transformer"
        transforms.mkdir()
        for fold in FOLDS:
            positions = np.flatnonzero(dataset.split.eq(fold).to_numpy())
            path = transforms / f"{fold}_sequences.npz"
            np.savez_compressed(path, training_dataset_rows=positions, causal_feature_rows=selected_sequences[positions])
            seq_facts[fold] = {"rows": len(positions), "sha256": hash_file(path)}
        mean, scale, used = fit_sequence_scaler(features[list(FEATURES)].to_numpy(dtype=float),
                                               selected_sequences[dataset.split.eq("train").to_numpy()])
        scaler = transforms / "train_only_scaler.npz"
        np.savez(scaler, mean=mean, scale=scale, feature_names=np.asarray(FEATURES))
        scaler_facts = {"file": scaler.name, "fit_fold": "train", "sha256": hash_file(scaler),
                        "unique_feature_rows": used}
    files = {file.name: hash_file(file) for file in dest.iterdir()}
    audit = {"version": VERSION, **GUARD, **asdict(bounds), "research_end_exclusive": bounds.research_end_exclusive,
             "v1_old_test_excluded": ORIGINAL_TEST_EXCLUDED, "subset_training_allowed": qualified,
             "prospective_test_started": False, "subset_training_research_only": True, "feature_names": list(FEATURES),
             "feature_lookback": FEATURE_LOOKBACK, "sequence_eligibility_length": TRANSFORMER_SEQUENCE,
             "sequence_policy": SEQUENCE_POLICY, "v2_sequence_policy": SEQUENCE_POLICY,
             "research_pool_policy": POOL_POLICY, "months": info["months"],
             "test_kind": "HISTORICAL_RETROSPECTIVE_DEVELOPMENT_ONLY", "splits": splits,
             "daily_inference_coverage_is_retrospective_not_timely": True,
             "latest_signal_coverage": daily.iloc[-1].to_dict(),
             "candidate_anchor_rows": len(anchors), "trusted_rows": len(dataset),
             "frozen_input_sha256": hash_file(plan_path), "frozen_price_sha256": info["files"]["formal_ohlcv.parquet"],
             "feature_history_derived_from_future_label_eligibility": False,
             "pool": info["market_population"] | {"pool_config": asdict(cfg)},
             "action_evidence": {"legacy": evidence.legacy.provenance, "recent": recent.provenance},
             "files": files}
    audit_path = dest / "subset_audit.json"
    ledger.atomic_json(audit_path, audit)
    if qualified:
        preflight = ledger.read_json(v1 / "transformer" / "transformer_preflight.json")
        ledger.atomic_json(out / "transformer" / "transformer_preflight.json", {
            "status": "SEQUENCE_INPUTS_READY", "market_model_fitted": False, "legacy_training_gate": "CLOSED",
            "sequence_length": TRANSFORMER_SEQUENCE, "folds": seq_facts, "scaler": scaler_facts,
            "dataset_audit_sha256": hash_file(audit_path),
            "causal_features_sha256": files["causal_inference_features.parquet"],
            "labels_sha256": files["training_dataset.parquet"],
            "initial_training_config": preflight["initial_training_config"],
        })
    print({"eligible": qualified, "splits": splits}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    design = sub.add_parser("plan")
    design.add_argument("--source", type=Path, required=True)
    design.add_argument("--supplement", type=Path, action="append", default=[])
    design.add_argument("--out", type=Path, required=True)
    design.add_argument("--validation-start", default="2026-06-12")
    design.add_argument("--diagnostic-start", default="2026-08-17")
    design.add_argument("--size", type=int, default=400)
    design.add_argument("--retain", type=int, default=300)
    design.add_argument("--capture-limit", type=int, default=120)
    capture = sub.add_parser("capture")
    capture.add_argument("--plan", type=Path, required=True)
    produce = sub.add_parser("build")
    produce.add_argument("--plan", type=Path, required=True)
    produce.add_argument("--out", type=Path, required=True)
    produce.add_argument("--v1", type=Path, required=True)
    produce.add_argument("--notice-index", type=Path, default=NOTICE_INDEX)
    produce.add_argument("--legacy-actions", type=Path, default=ledger.OUTPUT)
    produce.add_argument("--official-actions", type=Path, default=ledger.CONTROLS)
    args = parser.parse_args()
    if args.cmd == "plan":
        plan(args.source, args.supplement, args.out, args.validation_start, args.diagnostic_start,
             args.size, args.retain, args.capture_limit)
    elif args.cmd == "capture":
        info = ledger.read_json(args.plan / "inputs" / "research_pool_plan.json")
        for job in info["batches"]:
            folder = Path(job["root"]) / "inputs"
            if hash_file(folder / "evidence_plan.json") != job["sha256"]:
                raise ValueError("New observation plan digest changed")
            capture_observations(folder)
    else:
        build(args.plan, args.out, args.v1, args.notice_index, args.legacy_actions, args.official_actions)


if __name__ == "__main__":
    main()
