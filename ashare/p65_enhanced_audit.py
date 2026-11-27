"""Read-only comparison of frozen enhanced-model eligibility and an active-only C diagnostic."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, time
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sqlalchemy import text

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, exclusive, read_json
from .p4_subset_inputs import RISK_FILE, hash_file, parquet_read
from .p4_transformer_train import load_model, predict, scale_sequences, verify_digest
from .p53_experiments import FACTOR_MODEL_VERSION
from .p54_factors import P54_FEATURES
from .p55_daily import FORMAL_AVAILABILITY, checked_manifest, is_timely, now_shanghai, parse_clock
from .p55_panel import dependencies, factor_sequences, fetch_panel_features, legacy_details, validate_existing
from .p58_group_c import ARM, contract, read_capture
from .p56_ablation import VERSION as TRAIN_VERSION, masked_sequences

VERSION = "P65_ENHANCED_READ_ONLY_DIAGNOSTIC_V1"
PATCH_DAY = "2026-10-16"
PATCH_PRIOR_DAY = "2026-10-15"
PATCH_CHECK_DAY = "2026-10-19"
PATCH_FILE = "close_quote_research_patch.parquet"
PATCH_PRICE_FIELDS = ("open", "high", "low", "close", "volume", "amount")
DIAGNOSTIC_TRACK = "READ_ONLY_DIAGNOSTIC"
QUOTE_TRACK = "RESEARCH_QUOTE_PATCH_DIAGNOSTIC"
REPORT_FILE = "enhanced_audit.json"
BASE_REPORT_FILES = frozenset(("latest_missing_factor_dates.csv", "latest_factor_candidates.csv"))
SCORES_FILE = "research_scenario_scores.csv"


def patch_market_rows(raw, patch):
    required = {"stock_code", "trade_date", *PATCH_PRICE_FIELDS, "source_billing_id",
                "source_quote_ticktime", "previous_formal_close", "source_kind"}
    if not required.issubset(patch) or not len(patch):
        raise ValueError("Expected a nonempty vetted quote patch")
    patch = patch.copy()
    days = pd.to_datetime(patch.trade_date, errors="raise").dt.strftime("%Y-%m-%d")
    observed = pd.to_datetime(raw.trade_date, errors="raise").dt.strftime("%Y-%m-%d")
    symbols = patch.stock_code.astype("string")
    prices = patch[list(PATCH_PRICE_FIELDS)].apply(pd.to_numeric, errors="raise")
    previous = raw.loc[observed.eq(PATCH_PRIOR_DAY)].set_index("stock_code").close
    matched = symbols.map(pd.to_numeric(previous, errors="raise"))
    tick = pd.to_timedelta(patch.source_quote_ticktime, errors="raise")
    if (not days.eq(PATCH_DAY).all() or symbols.duplicated().any()
            or not symbols.str.fullmatch(r"\d{6}").fillna(False).all()
            or symbols.isin(raw.loc[observed.eq(PATCH_DAY), "stock_code"]).any()
            or patch.source_billing_id.isna().any() or patch.source_billing_id.duplicated().any()
            or not patch.source_kind.eq("OBSERVED_POST_CLOSE_QUOTE").all()
            or not tick.between(pd.Timedelta(hours=15), pd.Timedelta(hours=15, minutes=5)).all()
            or not np.isfinite(prices).all(axis=None) or not prices.gt(0).all(axis=None)
            or not (prices.low.le(prices[["open", "close"]].min(axis=1))
                    & prices.high.ge(prices[["open", "close"]].max(axis=1))).all()
            or not prices.open.div(matched).sub(1).abs().le(.30).all()
            or not prices.close.div(matched).sub(1).abs().le(.32).all()
            or matched.isna().any() or not np.array_equal(matched.to_numpy(dtype=float),
                   pd.to_numeric(patch.previous_formal_close, errors="raise").to_numpy(dtype=float))):
        raise ValueError("Quote patch conflicts with source bars, calibration window, or prior close")
    # P54's pure factor builder consumes a formal-row selector. Tag only this temporary
    # in-memory adapter to pass its filter; the archived patch retains its own provenance.
    adapter = prices.assign(stock_code=symbols.astype(raw.stock_code.dtype), trade_date=pd.to_datetime(days).dt.date,
                            not_billing=1)
    return pd.concat([raw, adapter], ignore_index=True)


def load_research_patch(path, expected):
    verify_digest(path, expected)
    manifest = read_json(path)
    entry = manifest["research_quote_patch"]
    controls = manifest["closing_snapshot_reconciliation"]["formal_comparisons"]
    controls_by_day = {r["trade_date"]: r for r in controls}
    if (entry["trade_date"] != PATCH_DAY or entry["source"] != "public.billing"
            or entry["formal_status"] != "RESEARCH_ONLY_CLOSE_QUOTE"
            or entry["only_original_missing_keys"] is not True
            or set(controls_by_day) != {PATCH_DAY, PATCH_CHECK_DAY}
            or any(r["formal_stocks"] <= 0 or r["formal_stocks"] != r["exact_ohlcv_amount"] for r in controls)):
        raise ValueError("Research quote lineage is incompatible with the calibrated incident")
    digest = entry["files"][PATCH_FILE]
    override = path.parent / PATCH_FILE
    verify_digest(override, digest)
    patch = parquet_read(override)
    if len(patch) != entry["rows"]:
        raise ValueError("Research quote patch row count differs from recorded lineage")
    return patch, {"manifest_sha256": expected, "artifact_sha256": digest, "date": PATCH_DAY}


def latest_formal(engine, moment):
    sql = text("SELECT max(trade_date) FROM public.market WHERE not_billing=1 AND "
               "(trade_date < :day OR (trade_date = :day AND :closed))")
    with engine.begin() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        if conn.scalar(text("SHOW transaction_read_only")) != "on":
            raise RuntimeError("Enhanced diagnostic requires read-only prices")
        day = conn.scalar(sql, {"day": moment.date(), "closed": moment.time() >= time(18)})
    if day is None:
        raise ValueError("No available formal session")
    return pd.Timestamp(day).date()


def eligibility(matrix, sequences, active):
    if matrix.ndim != 2 or matrix.shape[1] != len(P54_FEATURES) or active.shape != (len(P54_FEATURES),) or not active.any():
        raise ValueError("Unexpected frozen 18-column feature mask")
    missing = ~np.isfinite(matrix[sequences])
    return missing, ~missing.any(axis=(1, 2)), ~missing[:, :, active].any(axis=(1, 2))


def active_standardized(matrix, sequences, active, mean, scale):
    """Keep the frozen projection dimension and zero only inactive standardized columns."""
    raw = matrix[sequences][:, :, active]
    if not np.isfinite(raw).all():
        raise ValueError("Cannot score a missing active feature")
    result = np.zeros((*sequences.shape, len(P54_FEATURES)), dtype=np.float32)
    result[:, :, active] = ((raw - np.asarray(mean)[active]) / np.asarray(scale)[active]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite active feature after scaling")
    return result


def report_windows(missing, dates, active):
    rows = []
    for j, name in enumerate(P54_FEATURES):
        mask = missing[:, :, j]
        for day in sorted(np.unique(dates[mask])):
            rows.append({"feature": name, "active_c": bool(active[j]), "input_date": str(day)[:10],
                         "missing_stock_days": int((mask & (dates == day)).sum())})
    return rows


def score_difference(candidate, old):
    candidate = np.asarray(candidate)
    old = np.asarray(old)
    if candidate.shape != old.shape or not len(old) or not np.isfinite(candidate).all() or not np.isfinite(old).all():
        raise ValueError("Expected a nonempty finite paired score comparison")
    return {"rows": len(old), "max_abs_error": float(np.max(np.abs(candidate - old))),
            "bitwise_equal": bool(np.array_equal(candidate, old)),
            "within_atol_1e_6": bool(np.allclose(candidate, old, atol=1e-6, rtol=0))}


def archive_compare(model, scores, codes, folder, spec):
    path = folder / spec["predictions_file"]
    verify_digest(path, spec["predictions_sha256"])
    original = parquet_read(path)
    if (original.stock_code.duplicated().any() or len(original) != spec["rows"]
            or not original.model_sha256.eq(model["weights_sha256"]).all()
            or set(original.stock_code) != set(codes)):
        raise ValueError("Recomputed complete cohort does not match its immutable prediction file")
    old = original.set_index("stock_code").loc[codes, "score"].to_numpy(dtype=float)
    return score_difference(scores, old) | {"archive_sha256": spec["predictions_sha256"]}


def parents(day, legacy_root, panel, *, timely_by=None):
    # A historical run is checked against its historical lineage, if one exists.
    key = "timely"
    if timely_by is None and (panel / "historical" / day / "manifest.json").is_file():
        key = "historical"
    legacy_day = legacy_root / key / day
    peer_day = panel / key / day
    if not (legacy_day / "manifest.json").is_file() or not (peer_day / "manifest.json").is_file():
        raise ValueError(f"Matching frozen candidate cohort is missing for {day}")
    original = checked_manifest(legacy_day)
    parent_digest = hash_file(legacy_day / "manifest.json")
    peer = validate_existing(peer_day, parent_digest, hash_file(panel / "policy.json"))
    if any(rec["trade_date"] != day or rec["capture_kind"].lower() != key or not rec["formal_session"]
           for rec in (original, peer)):
        raise ValueError("Unexpected session or parent lineage")
    if timely_by is not None:
        for rec in (original, peer):
            started = parse_clock(rec["capture_started_at"])
            committed = parse_clock(rec["committed_at"])
            if (not is_timely(None, started, committed)
                    or committed.date().isoformat() != day or committed > timely_by
                    or rec["status"] == "MISSED_CAPTURE_WINDOW"):
                raise ValueError("Scheduled diagnostic requires two timely same-session parent captures")
    return original, peer, legacy_day, peer_day


def parent_hashes(pair):
    return {name: hash_file(folder / "manifest.json")
            for name, folder in zip(("legacy", "panel"), pair[2:], strict=True)}


def run_day(engine, moment, day, paths, metadata, compare=False, research_patch=None, matched_parents=None):
    pair = matched_parents if matched_parents is not None else parents(day, paths.legacy, paths.panel)
    original, peer, legacy_day, peer_day = pair
    meta18, meta_c, policy_c = metadata
    minimum = policy_c["minimum_market_stocks"]
    active = np.asarray(policy_c["input_mask"], dtype=bool)
    if any(day <= meta["selection_last_exit"] for meta in (meta18, meta_c)):
        raise ValueError("Audit day is inside a frozen model selection window")
    verify_digest(RISK_FILE, original["risk_events_sha256"])
    cohort = legacy_details(legacy_day, original)
    rows, population, indices, raw, codes = fetch_panel_features(engine, moment, day, meta18, original, cohort, RISK_FILE)
    factor_raw = patch_market_rows(raw, research_patch) if research_patch is not None else raw
    outcomes = Counter(rows.loc[rows.outcome.ne("SCORED"), "outcome"])
    baseline = {"date": day, "candidates": len(rows), "base_eligible": len(codes),
                "source_price_rows": len(raw), "archived_full18_scores": peer["models"]["factors18"]["rows"],
                "parent_manifest_sha256": parent_hashes(pair)}
    if not codes:
        result = baseline | {"full18_eligible": 0, "group_c_frozen_eligible": 0, "group_c_active_eligible": 0,
                             "status": "ABSTAINED", "full18_real_scores": 0, "group_c_real_scores": 0,
                             "input_stock_days": 0, "inactive_only_rejections": 0, "nonfinite_factors": {},
                             "frozen_abstentions": dict(outcomes), "active_only_diagnostic_abstentions": dict(outcomes)}
        if research_patch is not None:
            result["research_quote_rows"] = len(research_patch)
        return result, [], [], []
    matrix, original_good, remapped, factor_audit, missing_names = factor_sequences(
        factor_raw, population, indices, minimum)
    missing, full, only_active = eligibility(matrix, remapped, active)
    if not np.array_equal(original_good, full):
        raise ValueError("Frozen eligibility did not match the equivalent full mask")
    dates = population.features.trade_date.to_numpy()[indices]
    factor_days = report_windows(missing, dates, active)
    counts = {name: {"missing_stock_days": int(missing[:, :, j].sum()),
                     "affected_stocks": int(missing[:, :, j].any(axis=1).sum()),
                     "active_c": bool(active[j])}
              for j, name in enumerate(P54_FEATURES)}
    strict = dict(outcomes | Counter({"FACTOR_SEQUENCE_INCOMPLETE": int((~full).sum())}))
    hypothetical = dict(outcomes | Counter({"ACTIVE_FACTOR_SEQUENCE_INCOMPLETE": int((~only_active).sum())}))
    result = baseline | {"full18_eligible": int(full.sum()), "group_c_frozen_eligible": int(full.sum()),
                         "group_c_active_eligible": int(only_active.sum()),
                         "inactive_only_rejections": int((only_active & ~full).sum()),
                         "input_stock_days": int(missing.shape[0] * missing.shape[1]),
                         "frozen_abstentions": strict, "active_only_diagnostic_abstentions": hypothetical,
                         "nonfinite_factors": counts}
    if research_patch is not None:
        result["research_quote_rows"] = len(research_patch)
        result["market_pair_support"] = [r for r in factor_audit["market_session_support"]
                                         if r["trade_date"] in {PATCH_DAY, PATCH_CHECK_DAY}]
    scenario_scores = []
    details = [dict(stock_code=code, full18_eligible=bool(f), group_c_active_eligible=bool(a),
                    missing_factors=names, missing_active_c="|".join(
                        name for name, bad, on in zip(P54_FEATURES, flags, active, strict=True) if bad and on))
               for code, f, a, names, flags in zip(codes, full, only_active, missing_names, missing.any(axis=1), strict=True)]
    complete = remapped[full]
    if complete.size:
        import torch

        selected = np.asarray(codes)[full].tolist()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network, _ = load_model(paths.full18, device, version=FACTOR_MODEL_VERSION, feature_names=P54_FEATURES)
        baseline18 = predict(network, scale_sequences(matrix, complete, meta18["mean"], meta18["scale"]),
                             device, meta18["architecture"]["batch_size"])
        result["full18_real_scores"] = len(baseline18)
        network, _ = load_model(paths.group_model, device, version=TRAIN_VERSION, feature_names=P54_FEATURES)
        standardized = masked_sequences(scale_sequences(matrix, complete, meta_c["mean"], meta_c["scale"]), ARM)
        proposal = active_standardized(matrix, complete, active, meta_c["mean"], meta_c["scale"])
        common = (device, meta_c["architecture"]["batch_size"])
        strict_c, proposed_c = predict(network, standardized, *common), predict(network, proposal, *common)
        result["group_c_real_scores"] = len(strict_c)
        if research_patch is not None:
            scenario_scores = [dict(stock_code=code, signal_date=day, frozen_full18_score=float(f),
                                    frozen_group_c_score=float(c), track=QUOTE_TRACK)
                               for code, f, c in zip(selected, baseline18, strict_c, strict=True)]
        result["c_complete_common_comparison"] = score_difference(proposed_c, strict_c) | {
            "standardized_tensors_bitwise_equal": bool(np.array_equal(standardized, proposal))}
        if not result["c_complete_common_comparison"]["within_atol_1e_6"]:
            raise ArithmeticError("Active-only inference differs on identical complete samples")
        if compare:
            c_folder = paths.group_ledger / "historical" / day
            c_audit = read_capture(c_folder, hash_file(paths.group_ledger / "policy.json"))
            if (c_audit["panel_manifest_sha256"] != result["parent_manifest_sha256"]["panel"]
                    or c_audit["legacy_manifest_sha256"] != result["parent_manifest_sha256"]["legacy"]):
                raise ValueError("Historical group C predictions have different parents")
            result["frozen_c_archive_comparison"] = archive_compare(meta_c, proposed_c, selected, c_folder,
                                                                    c_audit["models"]["group_c"])
            result["frozen_full18_archive_comparison"] = archive_compare(meta18, baseline18, selected, peer_day,
                                                                         peer["models"]["factors18"])
    else:
        result.update(full18_real_scores=0, group_c_real_scores=0)
    result["status"] = "SCORED" if result["full18_real_scores"] else "ABSTAINED"
    return result, factor_days, details, scenario_scores


def require_scheduled_day(day, moment, formal_day=None):
    if day != moment.date() or moment.time() < FORMAL_AVAILABILITY:
        raise ValueError("Scheduled diagnostic requires today's Beijing after-close window")
    if formal_day is not None and formal_day != day:
        raise ValueError("Scheduled diagnostic requires today's formal market session")


def checked_report(out, day, moment, lineage, parents_sha, compare_day):
    path = out / REPORT_FILE
    if path.is_symlink():
        raise ValueError("Existing diagnostic manifest must be a local file")
    report = read_json(path)
    observed = parse_clock(report["observed_at"])
    expected_files = BASE_REPORT_FILES | ({SCORES_FILE} if lineage["research_patch"] is not None else set())
    if (any(report.get(key) != val for key, val in lineage.items()) or report.get("forward_eligible") is not False
            or report["latest_formal"]["date"] != day or report["latest_formal"]["parent_manifest_sha256"] != parents_sha
            or report["comparison"]["date"] != compare_day or observed.date().isoformat() != day
            or observed.time() < FORMAL_AVAILABILITY or observed > moment
            or set(report["files"]) != expected_files
            or {p.name for p in out.iterdir()} != expected_files | {REPORT_FILE}):
        raise ValueError("Existing diagnostic output does not match its scheduled-day lineage")
    for name in expected_files:
        member = out / name
        if member.is_symlink() or not member.is_file():
            raise ValueError("Existing diagnostic artifact must be a local file")
        verify_digest(member, report["files"][name])
    return path


def run(paths):
    out = paths.out.resolve()
    if any(out.is_relative_to(folder.resolve()) for folder in (paths.legacy, paths.panel, paths.group_ledger)):
        raise ValueError("Write diagnostic artifacts outside frozen prediction ledgers")
    moment = now_shanghai()
    scheduled = paths.scheduled_day
    if scheduled is not None:
        require_scheduled_day(scheduled, moment)
    engine = AShareDailyLoader().engine
    signal = latest_formal(engine, moment)
    if scheduled is not None:
        require_scheduled_day(scheduled, moment, signal)
    signal_text = signal.isoformat()
    meta18, dep = dependencies(paths.full18, paths.source)
    p55 = read_json(paths.panel / "policy.json")
    if p55["dependencies"] != dep:
        raise ValueError("Frozen full18 dependencies differ from reconstructed audit inputs")
    meta_c, dep_c = contract(paths.group_model, paths.source, paths.panel)
    policy_c = read_json(paths.group_ledger / "policy.json")
    if {k: v for k, v in policy_c.items() if k != "frozen_at"} != dep_c or parse_clock(policy_c["frozen_at"]) > moment:
        raise ValueError("Frozen group C dependencies differ from reconstructed audit inputs")
    metadata = meta18, meta_c, policy_c
    patch, patch_lineage = (load_research_patch(paths.research_patch_manifest, paths.research_patch_manifest_sha256)
                            if paths.research_patch_manifest else (None, None))
    if patch is not None and signal_text <= PATCH_CHECK_DAY:
        raise ValueError("The documented research patch requires a later signal date")
    lineage = {"version": VERSION, "track": QUOTE_TRACK if patch_lineage is not None else DIAGNOSTIC_TRACK,
               "research_patch": patch_lineage, "full18_weights_sha256": meta18["weights_sha256"],
               "group_c_weights_sha256": meta_c["weights_sha256"],
               "group_c_scaler_sha256": policy_c["train_only_scaler_sha256"],
               "active_c_features": [f for f, enabled in zip(P54_FEATURES, policy_c["input_mask"], strict=True) if enabled],
               "group_c_policy_sha256": hash_file(paths.group_ledger / "policy.json"),
               "full18_policy_sha256": hash_file(paths.panel / "policy.json")}
    compare_day = paths.compare.isoformat()
    pair = parents(signal_text, paths.legacy, paths.panel, timely_by=moment) if scheduled is not None else None
    parents_sha = parent_hashes(pair) if pair is not None else None
    if out.exists() and scheduled is not None:
        return checked_report(out, signal_text, moment, lineage, parents_sha, compare_day)
    latest, dates, candidates, scenario_scores = run_day(engine, moment, signal_text, paths, metadata,
                                                         research_patch=patch, matched_parents=pair)
    compare, _, _, _ = run_day(engine, moment, compare_day, paths, metadata, compare=True)
    if "frozen_c_archive_comparison" not in compare or not all(compare[key]["within_atol_1e_6"] for key in
            ("c_complete_common_comparison", "frozen_c_archive_comparison", "frozen_full18_archive_comparison")):
        raise ValueError("Frozen complete-input score comparison did not pass")
    with exclusive(out.parent):
        if out.exists():
            if scheduled is not None:
                return checked_report(out, signal_text, now_shanghai(), lineage, parents_sha, compare_day)
            raise ValueError("Choose a unique diagnostic output directory")
        if scheduled is not None and now_shanghai().date() != scheduled:
            raise ValueError("Scheduled diagnostic finished after its Beijing day")
        with TemporaryDirectory(dir=out.parent) as temp:
            stage = Path(temp) / "diagnostic"
            stage.mkdir()
            pd.DataFrame(dates, columns=["feature", "active_c", "input_date", "missing_stock_days"]).to_csv(
                stage / "latest_missing_factor_dates.csv", index=False)
            pd.DataFrame(candidates, columns=["stock_code", "full18_eligible", "group_c_active_eligible",
                                               "missing_factors", "missing_active_c"]).to_csv(stage / "latest_factor_candidates.csv", index=False)
            if patch_lineage is not None:
                pd.DataFrame(scenario_scores, columns=["stock_code", "signal_date", "frozen_full18_score",
                                                       "frozen_group_c_score", "track"]).to_csv(
                    stage / SCORES_FILE, index=False)
            manifest = lineage | {"observed_at": moment.isoformat(), "forward_eligible": False,
                                  "latest_formal": latest, "comparison": compare,
                                  "files": {p.name: hash_file(p) for p in stage.iterdir()}}
            atomic_json(stage / REPORT_FILE, manifest)
            os.rename(stage, out)
    return out / REPORT_FILE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("legacy", "panel", "source", "full18", "group-model", "group-ledger", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--compare", type=date.fromisoformat, default=date(2026, 9, 23))
    parser.add_argument("--scheduled-day", type=date.fromisoformat)
    parser.add_argument("--research-patch-manifest", type=Path)
    parser.add_argument("--research-patch-manifest-sha256")
    args = parser.parse_args()
    if bool(args.research_patch_manifest) != bool(args.research_patch_manifest_sha256):
        parser.error("A quote patch requires both the manifest and its verified SHA256")
    print(run(args))


if __name__ == "__main__":
    main()
