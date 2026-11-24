"""Prepare a rolling P5 research dataset with mature economic-return labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text

from . import p4_action_pipeline as source
from .daily import AShareDailyLoader
from .p4_mvp import FEATURES, PoolConfig, build_pool
from .p4_subset_dataset import FEATURE_LOOKBACK, GUARD, TRANSFORMER_SEQUENCE, TRUSTED, economic_return
from .p4_subset_inputs import ORIGINAL_TEST_EXCLUDED, RISK_FILE, hash_file, parquet_read, parquet_write
from .p5_evidence import RECENT_SOURCES, capture_observations

VERSION = "P5_ROLLING_GATED_ECONOMIC_V2"
INPUTS_VERSION = "P5_ROLLING_INPUTS_V1"
NOTICE_INDEX = Path("reports/ashare_p4_action_pipeline/followup/bse-notices.VlYHW4/verification-final/official_notice_index.csv")
FOLDS = ("train", "validation", "test")


@dataclass(frozen=True)
class Boundaries:
    train_start: str
    val_start: str
    test_start: str
    price_end: str

    def __post_init__(self):
        values = list(asdict(self).values())
        if values != [date.fromisoformat(value).isoformat() for value in values] or sorted(set(values)) != values:
            raise ValueError("P5 training, validation, diagnostic, and observed cutoff must increase")

    @property
    def warmup_begin(self):
        return (date.fromisoformat(self.train_start) - timedelta(days=140)).isoformat()

    @property
    def research_end_exclusive(self):
        return (date.fromisoformat(self.price_end) + timedelta(days=1)).isoformat()

    @staticmethod
    def rolling(price_end, train_start=None, val_start=None, test_start=None):
        end = pd.Timestamp(price_end)
        diagnostic = pd.Timestamp(test_start) if test_start else end - pd.DateOffset(months=1)
        validation = pd.Timestamp(val_start) if val_start else diagnostic - pd.DateOffset(months=2)
        train = pd.Timestamp(train_start) if train_start else validation - pd.DateOffset(months=12)
        return Boundaries(*(t.date().isoformat() for t in (train, validation, diagnostic, end)))

    def split(self, signal, exit_day):
        if signal < self.train_start or exit_day > self.price_end:
            return None
        if signal < self.val_start:
            return "train" if exit_day < self.val_start else None
        if signal < self.test_start:
            return "validation" if exit_day < self.test_start else None
        return "test"


def verify_formal_cutoff(day):
    capture_time = datetime.now(ZoneInfo("Asia/Shanghai"))
    first_unavailable = capture_time.date() + timedelta(days=capture_time.time() >= time(18))
    if date.fromisoformat(day) >= first_unavailable:
        raise ValueError("Requested price cutoff is outside the formal daily availability boundary")


def load_formal(engine, bounds):
    verify_formal_cutoff(bounds.price_end)
    query = text("""
        SELECT stock_code, trade_date, open, high, low, close, volume, amount, pre_close, not_billing
          FROM public.market
         WHERE not_billing=1 AND stock_code IS NOT NULL AND trade_date BETWEEN :begin AND :end
    """)
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
        with conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            if conn.scalar(text("show transaction_read_only")) != "on":
                raise ValueError("P5 input requires a read-only snapshot")
            batches = pd.read_sql_query(query, conn.execution_options(stream_results=True),
                                        params={"begin": bounds.warmup_begin, "end": bounds.price_end},
                                        chunksize=100000)
            raw = pd.concat(batches, ignore_index=True)
    raw["trade_date"] = pd.to_datetime(raw.trade_date).dt.strftime("%Y-%m-%d")
    raw = raw.sort_values(["stock_code", "trade_date"], kind="stable").reset_index(drop=True)
    if raw.empty or raw.trade_date.min() < bounds.warmup_begin or raw.trade_date.max() != bounds.price_end:
        raise ValueError("Frozen price end must be an actual formal session inside the requested snapshot")
    return raw


def prepare(out, v1, bounds, limit, config):
    if not 1 <= limit <= 220:
        raise ValueError("P5 recent evidence cap must be between 1 and 220 stocks")
    archive = v1 / "inputs"
    previous_jobs = source.read_json(archive / "factor_plan.json")
    previous_manifest = source.read_json(archive / "inputs.json")
    previous_plan = hash_file(archive / "factor_plan.json")
    if previous_plan != previous_manifest["files"]["factor_plan.json"]:
        raise ValueError("Archived P4 factor plan digest mismatch")
    old_universe = {job["stock_code"]: job["provider_code"] for job in previous_jobs}
    raw = load_formal(AShareDailyLoader().engine, bounds)
    events = pd.read_csv(RISK_FILE, dtype={"stock_code": "string"})
    train_end = (date.fromisoformat(bounds.val_start) - timedelta(days=1)).isoformat()
    historical = build_pool(raw.loc[raw.trade_date.le(train_end)], date.fromisoformat(bounds.train_start),
                            date.fromisoformat(train_end), config, events=events).data
    shortlist = (historical.loc[historical.stock_code.isin(old_universe)]
                 .groupby("stock_code", as_index=False)
                 .agg(anchor_count=("trade_date", "size"), first_anchor=("trade_date", "min"),
                      mean_rank=("liquidity_rank", "mean"))
                 .sort_values(["anchor_count", "mean_rank", "first_anchor", "stock_code"],
                              ascending=[False, True, True, True]).head(limit))
    if shortlist.empty:
        raise ValueError("No prior research stocks for targeted P5 capture")
    days = sorted(raw.trade_date.unique().tolist())
    plan = {"start": ORIGINAL_TEST_EXCLUDED, "end": bounds.price_end, "providers": list(RECENT_SOURCES),
            "job_selection_last_date_inclusive": train_end,
            "jobs": [{"stock_code": r.stock_code, "provider_code": old_universe[r.stock_code],
                      "training_liquidity_anchor_count": int(r.anchor_count),
                      "first_training_anchor": r.first_anchor.date().isoformat()}
                     for r in shortlist.itertuples(index=False)]}
    if plan["start"] > plan["end"]:
        # An entirely historical window needs only the old archive.
        plan["jobs"] = []
    out.mkdir(parents=True)
    inputs = out / "inputs"
    inputs.mkdir()
    parquet_write(raw, inputs / "formal_ohlcv.parquet")
    source.atomic_json(inputs / "evidence_plan.json", plan)
    files = {name: hash_file(inputs / name) for name in ("formal_ohlcv.parquet", "evidence_plan.json")}
    source.atomic_json(inputs / "inputs.json", {
        "version": INPUTS_VERSION, "bounds": asdict(bounds), "pool_config": asdict(config),
        "calendar": days, "price_rows": len(raw), "price_stocks": raw.stock_code.nunique(),
        "prior_factor_plan_sha256": previous_plan, "risk_events_sha256": hash_file(RISK_FILE), "files": files,
        "recent_evidence_planned": len(plan["jobs"]), "data_source": "READ_ONLY_FORMAL_PUBLIC_MARKET",
    })
    print(f"P5 prepared {len(raw):,} formal prices, {len(days)} sessions, {len(plan['jobs'])} evidence stocks", flush=True)


def write_evidence_plan(out, prior, plan):
    out.mkdir(parents=True)
    inputs = out / "inputs"
    inputs.mkdir()
    plan_path = inputs / "evidence_plan.json"
    source.atomic_json(plan_path, plan)
    source.atomic_json(inputs / "inputs.json", {
        "version": "P5_IMMUTABLE_EVIDENCE_EXTENSION_V1",
        "prior_plan_sha256": hash_file(prior / "inputs/evidence_plan.json"),
        "files": {"evidence_plan.json": hash_file(plan_path)},
    })
    print(f"P5 evidence plan {plan['start']} to {plan['end']} for {len(plan['jobs'])} stocks", flush=True)


def reframe(out, prior, diagnostic_start):
    """Reuse immutable observations while changing the diagnostic start boundary only."""
    original = (prior / "inputs").resolve(strict=True)
    manifest = source.read_json(original / "inputs.json")
    if manifest["version"] != INPUTS_VERSION or any(hash_file(original / name) != digest
                                                   for name, digest in manifest["files"].items()):
        raise ValueError("Prior P5 price and evidence inputs failed digest verification")
    boundaries = Boundaries(**(manifest["bounds"] | {"test_start": diagnostic_start}))
    destination = out / "inputs"
    out.mkdir(parents=True)
    destination.mkdir()
    for name in (*manifest["files"], "observations"):
        item = original / name
        if item.exists():
            (destination / name).symlink_to(item.resolve(strict=True), target_is_directory=item.is_dir())
    source.atomic_json(destination / "inputs.json", manifest | {"bounds": asdict(boundaries)})
    print(f"Reframed diagnostic start {manifest['bounds']['test_start']} to {boundaries.test_start}", flush=True)


def prepare_live_evidence(out, prior, diagnostics, price_end, limit):
    if not 1 <= limit <= 220:
        raise ValueError("P5 contemporaneous evidence must be bounded to 1–220 stocks")
    verify_formal_cutoff(price_end)
    plan_path = prior / "inputs/evidence_plan.json"
    plan = source.read_json(plan_path)
    if source.read_json(prior / "inputs/inputs.json")["files"][plan_path.name] != hash_file(plan_path):
        raise ValueError("Previous evidence plan digest mismatch")
    old = {job["stock_code"] for job in plan["jobs"]}
    rows = pd.read_csv(diagnostics, dtype={"stock_code": "string"})
    if set(rows.trade_date.astype(str)) != {price_end} or rows.stock_code.duplicated().any():
        raise ValueError("Candidate diagnostics must represent one unique matching signal date")
    eligible = rows["v2_price_sequence_available"].map({True: True, False: False, "True": True, "False": False})
    if eligible.isna().any():
        raise ValueError("Candidate diagnostics have unknown eligibility values")
    rows = rows.loc[eligible & ~rows.stock_code.isin(old)].sort_values(["liquidity_rank", "stock_code"])
    jobs = [{"stock_code": code, "provider_code": provider} for code in rows.stock_code
            if (provider := source.bao._provider_code(code))][:limit]
    scope_start = (date.fromisoformat(price_end) - timedelta(days=100)).isoformat()
    write_evidence_plan(out, prior, {"kind": "LIVE_SUPPLEMENT", "start": scope_start, "end": price_end,
                                    "jobs": jobs, "providers": list(RECENT_SOURCES),
                                    "candidate_diagnostics_sha256": hash_file(diagnostics)})


@dataclass
class WindowEvidence:
    legacy: object
    legacy_start: str
    recent_start: str
    recent_end: str
    recent_dual: set
    recent_events: dict
    provenance: dict

    def covered(self, stock, first, last):
        if first < self.recent_start:
            legacy_last = min(last, ORIGINAL_TEST_EXCLUDED)
            years = set(range(int(first[:4]), int(legacy_last[:4]) + 1))
            if (first < self.legacy_start or stock not in self.legacy.factor_days
                    or not years.issubset(self.legacy.provider_years.get(stock, set()))):
                return False
        return last < self.recent_start or (stock in self.recent_dual and last <= self.recent_end)

    def observed_action_dates(self, stock):
        return (self.legacy.actions.get(stock, set()) | self.legacy.conflicts.get(stock, set())
                | self.legacy.factor_days.get(stock, set()) | self.recent_events.get(stock, set()))

    def inference_snapshot(self):
        items = {}
        for stock in sorted(self.recent_dual):
            legacy_start = self.legacy_start if self.covered(
                stock, self.legacy_start, (date.fromisoformat(self.recent_start) - timedelta(days=1)).isoformat()
            ) else self.recent_start
            items[stock] = {
                "covered_from": legacy_start,
                "covered_through": self.recent_end,
                "actions": [[event, event] for event in sorted(self.observed_action_dates(stock))
                            if legacy_start <= event <= self.recent_end],
            }
        return {"policy": "P5_DUAL_OBSERVED_RESEARCH_WINDOW_V1",
                "event_date_role": "price-effective quarantine, not inferred publication time",
                "observed_at_utc": self.provenance["latest_observation_utc"],
                "max_covered_date": self.recent_end, "stocks": items}


def read_recent_evidence(inputs):
    from .p5_evidence import observation_path, verified_observation

    plan_path = inputs / "evidence_plan.json"
    plan = source.read_json(plan_path)
    plan_digest = hash_file(plan_path)
    if source.read_json(inputs / "inputs.json")["files"][plan_path.name] != plan_digest:
        raise ValueError("Recent evidence input plan digest differs")
    stats = Counter()
    recent_events = defaultdict(set)
    both = set()
    positive_matches = 0
    latest = None
    evidence_files = {}
    for job in plan["jobs"]:
        stock = job["stock_code"]
        active = {}
        for provider in RECENT_SOURCES:
            path = observation_path(inputs, stock, provider)
            if not path.is_file():
                stats[provider + "/NOT_CAPTURED"] += 1
                continue
            observation = verified_observation(path, job, provider, plan, plan_digest)
            status = observation["status"]
            stats[provider + "/" + status] += 1
            evidence_files[path.name] = hash_file(path)
            latest = max(latest or observation["observed_at_utc"], observation["observed_at_utc"])
            if provider == "bao_adjust":
                event_dates = {row[1] for row in observation["rows"]}
            else:
                event_dates = {row["除权除息日"] for row in observation["rows"]}
            recent_events[stock].update(event_dates)
            if status == "SUCCESS":
                active[provider] = event_dates
        if len(active) == len(RECENT_SOURCES):
            both.add(stock)
            positive_matches += len(active["bao_adjust"] & active["sina_dividend"])
    result = {"recent_plan_sha256": plan_digest, "recent_counts": dict(stats),
              "recent_evidence_files": evidence_files, "recent_dual_supported_stocks": len(both),
              "recent_positive_event_matches": positive_matches, "latest_observation_utc": latest}
    return plan, both, recent_events, result


def read_window_evidence(inputs, v1, action_root, notice_index, official):
    from .p4_subset_dataset import load_evidence

    old_inputs = v1 / "inputs"
    legacy_manifest = source.read_json(old_inputs / "inputs.json")
    legacy = load_evidence(old_inputs, legacy_manifest, action_root, notice_index, official)
    plan, both, recent_events, facts = read_recent_evidence(inputs)
    if both and not facts["recent_positive_event_matches"]:
        raise ValueError("Recent action sources did not independently reproduce any common positive event")
    return WindowEvidence(
        legacy, legacy_manifest["price_begin"], plan["start"], plan["end"], both, recent_events,
        facts | {"old": legacy.provenance},
    )


def read_prior_inference_evidence(prior):
    folder = prior / "dataset"
    manifest = source.read_json(folder / "subset_audit.json")
    evidence_path = folder / "inference_evidence.json"
    digest = hash_file(evidence_path)
    if manifest["version"] != VERSION or manifest["files"][evidence_path.name] != digest:
        raise ValueError("Previous P5 inference evidence manifest mismatch")
    evidence = source.read_json(evidence_path)
    if evidence["policy"] != "P5_DUAL_OBSERVED_RESEARCH_WINDOW_V1":
        raise ValueError("Invalid previous observation policy")
    return evidence, digest


def extend_evidence(out, prior, price_end, limit):
    """Freeze only the next observation interval for supported stocks, without rehashing labels."""
    if not 1 <= limit <= 220:
        raise ValueError("Evidence extension cap must be between 1 and 220 stocks")
    verify_formal_cutoff(price_end)
    earlier, prior_digest = read_prior_inference_evidence(prior)
    start = (date.fromisoformat(earlier["max_covered_date"]) + timedelta(days=1)).isoformat()
    if start > price_end:
        raise ValueError("New evidence end must be later than the existing ceiling")
    # Preserve the priority order of the immediately previous capture, then fill from its evidence envelope.
    previous_plan = source.read_json(prior / "inputs/evidence_plan.json")
    priority = {job["stock_code"]: index for index, job in enumerate(previous_plan["jobs"])}
    eligible = sorted((code for code, record in earlier["stocks"].items()
                       if record["covered_through"] == earlier["max_covered_date"]),
                      key=lambda code: (priority.get(code, len(priority)), code))
    jobs = [{"stock_code": code, "provider_code": provider} for code in eligible
            if (provider := source.bao._provider_code(code))][:limit]
    if not jobs:
        raise ValueError("Evidence extension has no supported prior stocks")
    write_evidence_plan(out, prior, {"kind": "INCREMENTAL", "start": start, "end": price_end,
                                    "jobs": jobs, "providers": list(RECENT_SOURCES),
                                    "prior_inference_evidence_sha256": prior_digest})


def export_live_evidence(out, prior):
    from .p5_sequences import VERSION as SEQUENCE_POLICY

    previous, prior_digest = read_prior_inference_evidence(prior)
    inputs = out / "inputs"
    manifest = source.read_json(inputs / "inputs.json")
    if manifest["prior_plan_sha256"] != hash_file(prior / "inputs/evidence_plan.json"):
        raise ValueError("Inference supplement requested different previous observation scope")
    plan, dual, recent_events, facts = read_recent_evidence(inputs)
    result = previous | {"stocks": {code: row.copy() for code, row in previous["stocks"].items()}}
    if plan["kind"] == "INCREMENTAL":
        if prior_digest != plan["prior_inference_evidence_sha256"] or plan["end"] <= previous["max_covered_date"]:
            raise ValueError("Incremental action window does not join the previous observation envelope")
        for stock in dual:
            row = result["stocks"][stock]
            previous_end = date.fromisoformat(row["covered_through"])
            if (previous_end + timedelta(days=1)).isoformat() != plan["start"]:
                raise ValueError("New observations cannot bridge a missing source interval")
            row["covered_through"] = plan["end"]
            row["actions"] += [[day, day] for day in sorted(recent_events[stock])]
    elif plan["kind"] == "LIVE_SUPPLEMENT":
        for stock in dual:
            if stock in result["stocks"]:
                raise ValueError("Live supplementary evidence repeated an existing supported stock")
            result["stocks"][stock] = {"covered_from": plan["start"], "covered_through": plan["end"],
                                        "actions": [[day, day] for day in sorted(recent_events[stock])]}
    else:
        raise ValueError("Unexpected P5 live evidence plan type")
    result["max_covered_date"] = max(previous["max_covered_date"], plan["end"])
    timestamps = [t for t in (previous["observed_at_utc"], facts["latest_observation_utc"]) if t]
    result["observed_at_utc"] = max(timestamps) if timestamps else None
    folder = out / "dataset"
    folder.mkdir()
    evidence_path = folder / "inference_evidence.json"
    source.atomic_json(evidence_path, result)
    source.atomic_json(folder / "subset_audit.json",
                       {"version": VERSION, **GUARD, "subset_training_allowed": False,
                        "sequence_policy": SEQUENCE_POLICY, "v2_sequence_policy": SEQUENCE_POLICY,
                        "audit_kind": "INFERENCE_EVIDENCE_ONLY", "parent_inference_evidence_sha256": prior_digest,
                        "action_evidence": facts, "files": {evidence_path.name: hash_file(evidence_path)}})
    print({"live_export_dual_stocks_added_or_refreshed": len(dual), "coverage_end": result["max_covered_date"]}, flush=True)


def label_windows(raw, anchors, ready_positions, calendar, boundaries, config, evidence):
    """Apply eligibility and maturity to the same V2 anchor/sequence definition."""
    from .p5_sequences import price_window_flags

    ready = set(ready_positions.tolist())
    sequence_span = FEATURE_LOOKBACK + TRANSFORMER_SEQUENCE - 1
    anchors = anchors.reset_index(drop=True)
    useful = set(anchors.stock_code.unique())
    scoped_raw = raw.loc[raw.stock_code.isin(useful)]
    raw_by_stock = scoped_raw.groupby("stock_code", sort=False)
    flags = price_window_flags(scoped_raw, calendar, allow_auxiliary_fallback=True)
    flags["usable_reference"] = flags.vendor_reference_available | flags.auxiliary_reference_fallback
    flags_by_stock = flags.groupby("stock_code", sort=False)
    iso = pd.DatetimeIndex(calendar).strftime("%Y-%m-%d").to_numpy(dtype="U10")
    label_rows, label_anchor_ids, quality = [], [], []
    for stock, group in anchors.groupby("stock_code", sort=False):
        bars = raw_by_stock.get_group(stock).set_index("trade_date").reindex(iso)
        opening, close, high, low, volume, amount = (
            bars[name].to_numpy(dtype=float) for name in ("open", "close", "high", "low", "volume", "amount"))
        reference = flags_by_stock.get_group(stock).set_index("session_id").usable_reference.reindex(
            range(len(iso)), fill_value=False).to_numpy(dtype=bool)
        valid = (np.isfinite(opening + close + high + low + volume + amount) & (low > 0)
                 & (volume > 0) & (amount > 0) & reference & (high >= np.maximum(opening, close))
                 & (low <= np.minimum(opening, close)))
        prior = np.r_[np.nan, close[:-1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            unexplained_jump = (abs(opening / prior - 1) > .30) | (abs(close / prior - 1) > .32)
        observed = evidence.observed_action_dates(stock)
        for r in group.itertuples():
            session = int(r.session_id)
            first, entry, exit_index = session + 1 - sequence_span, session + 1, session + config.hold_sessions + 1
            reason = None
            fold, entry_day, exit_day = "", "", ""
            if session >= len(iso) or r.trade_date != iso[session]:
                raise ValueError("P5 sequence xref does not agree with frozen calendar")
            if exit_index >= len(iso):
                reason = "UNMATURED_FINAL_WINDOW"
            else:
                entry_day, exit_day = str(iso[entry]), str(iso[exit_index])
                fold = boundaries.split(r.trade_date, exit_day)
                if fold is None:
                    reason = "CROSSES_FOLD_OR_UNMATURED"
                elif r.Index not in ready or first < 1:
                    reason = "MISSING_CAUSAL_FEATURE_SEQUENCE"
                elif not valid[entry:exit_index + 1].all():
                    reason = "INVALID_FORWARD_FORMAL_WINDOW"
                elif not evidence.covered(stock, str(iso[first]), exit_day):
                    reason = "INSUFFICIENT_DUAL_ACTION_EVIDENCE"
                elif any(iso[first] <= event <= r.trade_date for event in observed):
                    reason = "UNADJUSTED_FEATURE_ACTION_WINDOW"
                else:
                    events = sorted(event for event in observed if entry_day <= event <= exit_day)
                    if any((stock, event) not in evidence.legacy.verified
                           or event in evidence.legacy.conflicts.get(stock, set())
                           or event not in evidence.legacy.factor_days.get(stock, set()) for event in events):
                        reason = "UNVERIFIED_ACTION_IN_LABEL_WINDOW"
                    elif any(unexplained_jump[index] and str(iso[index]) not in events
                             for index in range(entry, exit_index + 1)):
                        reason = "UNEXPLAINED_FORWARD_PRICE_JUMP"
                    else:
                        payoffs = [evidence.legacy.verified[stock, event] for event in events if event > entry_day]
                        target = economic_return(opening[entry], opening[exit_index], payoffs)
                        if not np.isfinite(target) or abs(target) > config.max_abs_label:
                            reason = "ECONOMIC_RETURN_OUTSIDE_FROZEN_LIMIT"
                        else:
                            reason = "ISSUER_ADJUSTED" if payoffs else "OBSERVED_CLEAN_WINDOW"
                            label_rows.append({
                                "entry_date": entry_day, "exit_date": exit_day, "split": fold, "sample_grade": TRUSTED,
                                "sequence_feature_evidence_ready": True, "reason": reason, "target_return": float(target),
                                "event_sources": "|".join(payoff[2] for payoff in payoffs),
                            })
                            label_anchor_ids.append(r.Index)
            quality.append({"stock_code": stock, "trade_date": r.trade_date, "split": fold or "",
                            "reason": reason})
    selected = pd.concat([anchors.iloc[label_anchor_ids].reset_index(drop=True), pd.DataFrame(label_rows)], axis=1)
    return selected, np.asarray(label_anchor_ids, dtype=np.int64), pd.DataFrame(quality)


def build(out, v1, notice_index, action_root, official):
    from .p4_subset_transformer import fit_sequence_scaler
    from .p5_sequences import VERSION as SEQUENCE_POLICY, build_causal_population, sequence_indexes

    inputs = out / "inputs"
    manifest_path = inputs / "inputs.json"
    manifest = source.read_json(manifest_path)
    legacy_transformer = source.read_json(v1 / "transformer/transformer_preflight.json")
    if manifest["version"] != INPUTS_VERSION:
        raise ValueError("Unsupported P5 input snapshot")
    if any(hash_file(inputs / name) != digest for name, digest in manifest["files"].items()):
        raise ValueError("One or more P5 snapshot files changed")
    if (hash_file(RISK_FILE) != manifest["risk_events_sha256"]
            or hash_file(v1 / "inputs/factor_plan.json") != manifest["prior_factor_plan_sha256"]):
        raise ValueError("Referenced frozen evidence changed")
    bounds = Boundaries(**manifest["bounds"])
    config = PoolConfig(**manifest["pool_config"])
    raw = parquet_read(inputs / "formal_ohlcv.parquet")
    events = pd.read_csv(RISK_FILE, dtype={"stock_code": "string"})
    feature_begin = (date.fromisoformat(bounds.train_start) - timedelta(days=46))
    population = build_causal_population(raw, feature_begin, date.fromisoformat(bounds.price_end), config, events=events)
    if population.calendar.strftime("%Y-%m-%d").tolist() != manifest["calendar"]:
        raise ValueError("P5 population sessions differ from frozen price inputs")
    anchors = population.anchors.loc[population.anchors.trade_date.ge(bounds.train_start)].reset_index(drop=True)
    features = population.features.loc[population.features.stock_code.isin(anchors.stock_code.unique())].reset_index(drop=True)
    anchor_ids, source_indices = sequence_indexes(features, anchors, TRANSFORMER_SEQUENCE)
    evidence = read_window_evidence(inputs, v1, action_root, notice_index, official)
    dataset, trusted, quality = label_windows(raw, anchors, anchor_ids, population.calendar, bounds, config, evidence)
    reasons = {str(name): int(value) for name, value in quality.reason.value_counts().items()}
    if dataset.empty or any(not dataset.split.eq(fold).any() for fold in FOLDS):
        check = out / "dataset_rejection.json"
        source.atomic_json(check, {"reasons": reasons, "evidence": evidence.provenance,
                                   "bounds": asdict(bounds), "trusted_rows": len(dataset)})
        raise ValueError(f"No mature P5 rows in one or more required folds; see {check}")
    # Keep history for every price-qualified sequence, including anchors without a mature label.
    if not np.isin(trusted, anchor_ids).all():
        raise ValueError("Trusted anchor has no sequence")
    row_lookup = np.full(len(anchors), -1, dtype=np.int64)
    row_lookup[anchor_ids] = np.arange(len(anchor_ids))
    index = source_indices[row_lookup[trusted]]
    history_rows = np.unique(source_indices)
    compact = np.full(len(features), -1, dtype=np.int64)
    compact[history_rows] = np.arange(len(history_rows))
    features = features.iloc[history_rows].reset_index(drop=True)
    index = compact[index]
    output = out / "dataset"
    sequences = out / "transformer"
    output.mkdir()
    sequences.mkdir()
    parquet_write(dataset, output / "training_dataset.parquet")
    parquet_write(features, output / "causal_inference_features.parquet")
    parquet_write(anchors, output / "causal_anchors.parquet")
    parquet_write(quality, output / "anchor_quality.parquet")
    source.atomic_json(output / "inference_evidence.json", evidence.inference_snapshot())

    seq_info, folds = {}, {}
    train_index = None
    for fold in FOLDS:
        selected = np.flatnonzero(dataset.split.eq(fold).to_numpy())
        seq = index[selected]
        path = sequences / (fold + "_sequences.npz")
        np.savez_compressed(path, training_dataset_rows=selected, causal_feature_rows=seq)
        seq_info[fold] = {"rows": len(seq), "sha256": hash_file(path)}
        part = dataset.iloc[selected]
        folds[fold] = {"samples": len(part), "stocks": part.stock_code.nunique(), "days": part.trade_date.nunique(),
                       "first_signal": str(part.trade_date.min()), "last_signal": str(part.trade_date.max()),
                       "last_label_exit": str(part.exit_date.max()),
                       "days_with_at_least_five_stocks": int(part.groupby("trade_date").stock_code.nunique().ge(5).sum())}
        if fold == "train":
            train_index = seq
    mean, scale, used = fit_sequence_scaler(features[list(FEATURES)].to_numpy(dtype=float), train_index)
    scaler_path = sequences / "train_only_scaler.npz"
    np.savez(scaler_path, mean=mean, scale=scale, feature_names=np.asarray(FEATURES))
    dataset_files = {item.name: hash_file(item) for item in output.iterdir()}
    audit = {
        "version": VERSION, **GUARD, "feature_names": list(FEATURES), "feature_lookback": FEATURE_LOOKBACK,
        "sequence_eligibility_length": TRANSFORMER_SEQUENCE, "subset_training_allowed": True,
        "subset_training_research_only": True,
        "sequence_policy": SEQUENCE_POLICY, "v2_sequence_policy": SEQUENCE_POLICY,
        "test_kind": "ALREADY_VIEWED_HISTORICAL_DEVELOPMENT_DIAGNOSTIC",
        **asdict(bounds), "research_end_exclusive": bounds.research_end_exclusive,
        "v1_old_test_excluded": ORIGINAL_TEST_EXCLUDED, "prospective_test_started": False,
        "train_label_end_exclusive": bounds.val_start, "validation_label_end_exclusive": bounds.test_start,
        "pool": population.stats | {"pool_config": asdict(config)},
        "frozen_input_sha256": hash_file(manifest_path), "eligibility_counts": reasons,
        "candidate_anchor_rows": len(anchors), "trusted_rows": len(dataset), "splits": folds,
        "feature_history_derived_from_future_label_eligibility": False,
        "rolling_policy": {"training_months_default": 12, "validation_months_default": 2,
                           "historical_diagnostic_months_default": 1, "frequency": "monthly",
                           "no_automatic_jobs_created": True},
        "action_evidence": evidence.provenance,
        "current_risk_snapshot_sha256": manifest["risk_events_sha256"],
        "frozen_price_sha256": manifest["files"]["formal_ohlcv.parquet"],
        "files": dataset_files,
    }
    audit_path = output / "subset_audit.json"
    source.atomic_json(audit_path, audit)
    source.atomic_json(sequences / "transformer_preflight.json", {
        "status": "SEQUENCE_INPUTS_READY", "market_model_fitted": False, "legacy_training_gate": "CLOSED",
        "dataset_audit_sha256": hash_file(audit_path), "sequence_length": TRANSFORMER_SEQUENCE, "folds": seq_info,
        "causal_features_sha256": dataset_files["causal_inference_features.parquet"],
        "labels_sha256": dataset_files["training_dataset.parquet"],
        "initial_training_config": legacy_transformer["initial_training_config"],
        "scaler": {"file": scaler_path.name, "fit_fold": "train", "sha256": hash_file(scaler_path),
                   "unique_feature_rows": used},
    })
    print({"dataset_version": VERSION, "folds": folds, "reasons": reasons,
           "recent_dual_stocks": len(evidence.recent_dual)}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "capture", "build"):
        sub = modes.add_parser(command)
        sub.add_argument("--out", type=Path, required=True)
        if command != "capture":
            sub.add_argument("--v1", type=Path, required=True)
        if command == "prepare":
            sub.add_argument("--price-end", required=True)
            sub.add_argument("--train-start")
            sub.add_argument("--validation-start")
            sub.add_argument("--diagnostic-start")
            sub.add_argument("--evidence-limit", type=int, default=160)
            sub.add_argument("--top-n", type=int, default=800)
        elif command == "build":
            sub.add_argument("--notice-index", type=Path, default=NOTICE_INDEX)
            sub.add_argument("--legacy-actions", type=Path, default=source.OUTPUT)
            sub.add_argument("--official-actions", type=Path, default=source.CONTROLS)
    live = modes.add_parser("prepare-live-evidence")
    live.add_argument("--out", type=Path, required=True)
    live.add_argument("--prior", type=Path, required=True)
    live.add_argument("--candidate-diagnostics", type=Path, required=True)
    live.add_argument("--price-end", required=True)
    live.add_argument("--evidence-limit", type=int, default=120)
    for command in ("extend-evidence", "export-live-evidence"):
        sub = modes.add_parser(command)
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--prior", type=Path, required=True)
        if command == "extend-evidence":
            sub.add_argument("--price-end", required=True)
            sub.add_argument("--evidence-limit", type=int, default=220)
    frame = modes.add_parser("reframe")
    frame.add_argument("--out", type=Path, required=True)
    frame.add_argument("--prior", type=Path, required=True)
    frame.add_argument("--diagnostic-start", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out, args.v1, Boundaries.rolling(args.price_end, args.train_start, args.validation_start,
                                                    args.diagnostic_start), args.evidence_limit,
                PoolConfig(top_n=args.top_n))
    elif args.command == "capture":
        plan = source.read_json(args.out / "inputs/evidence_plan.json")
        if (args.out / "inputs/evidence_plan.json").is_symlink():
            raise ValueError("A reframed input bundle is read-only; capture into a new observation plan")
        if plan["jobs"]:
            capture_observations(args.out / "inputs")
        else:
            print("Window precedes the legacy archive ceiling; no upstream requests")
    elif args.command == "prepare-live-evidence":
        prepare_live_evidence(args.out, args.prior, args.candidate_diagnostics, args.price_end, args.evidence_limit)
    elif args.command == "extend-evidence":
        extend_evidence(args.out, args.prior, args.price_end, args.evidence_limit)
    elif args.command == "export-live-evidence":
        export_live_evidence(args.out, args.prior)
    elif args.command == "reframe":
        reframe(args.out, args.prior, args.diagnostic_start)
    else:
        build(args.out, args.v1, args.notice_index, args.legacy_actions, args.official_actions)


if __name__ == "__main__":
    main()
