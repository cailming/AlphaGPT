"""Evaluate C against all six immutable peers on the same archived signal dates."""
from __future__ import annotations

from datetime import date, time, timedelta

import numpy as np
import pandas as pd

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, exclusive, read_json
from .p4_subset_inputs import hash_file, parquet_read
from .p55_daily import checked_manifest, now_shanghai
from .p55_mature import MIN_LABEL_COVERAGE, MIN_MATURED_DAYS, formal_tail
from .p55_nightly import publish_revision
from .p55_panel import validate_existing
from .p55_reports import (LABEL_FIELDS, LABEL_REPORT, REPORT_AUDIT, artifact, ranked_daily, summarize)
from .p58_group_c import ARM, VERSION, read_capture, track
from .p55_daily import parse_clock


def load_cohorts(folder, policy, digest, panel, legacy, six_labels, audited):
    rec = read_capture(folder, digest)
    day = rec["trade_date"]
    if folder.name != day or folder.parent.name != rec["capture_kind"].lower():
        raise ValueError("C capture path and origin disagree")
    origin = rec["parent_kind"]
    parent = legacy / origin / day
    peer = panel / origin / day
    for directory, key in ((parent, "legacy_manifest_sha256"), (peer, "panel_manifest_sha256")):
        actual = hash_file(directory / "manifest.json")
        if rec[key] != actual or actual not in audited:
            raise ValueError("C manifest source is absent from the six-method evaluation revision")
    original = checked_manifest(parent)
    six = validate_existing(peer, rec["legacy_manifest_sha256"], policy["panel_policy_sha256"])
    start, finish = (parse_clock(rec[k]) for k in ("started_at", "committed_at"))
    origin_track = track(rec["capture_kind"], day, start, finish, policy, six["formal_session"], six["prospective_eligible"])
    if orig_mismatch(rec, origin_track, six, original):
        raise ValueError("C evaluation track or prospective provenance disagrees")
    groups, specs = {}, {}
    for directory, entry in ((parent, original), (peer, six), (folder, rec)):
        for lane, spec in entry["models"].items():
            values = parquet_read(artifact(directory, spec["predictions_file"], spec["predictions_sha256"]))
            if (len(values) != spec["rows"] or values.prediction_id.duplicated().any()
                    or values.stock_code.duplicated().any() or not values.trade_date.eq(day).all()
                    or not values.objective.eq(lane).all() or not values.model_sha256.eq(spec["weights_sha256"]).all()
                    or not np.isfinite(values.score.to_numpy(float)).all()):
                raise ValueError("Inconsistent archived forecast")
            if lane != ARM:
                values = values.merge(six_labels[["prediction_id", *LABEL_FIELDS]], how="left",
                                      on="prediction_id", validate="one_to_one")
                if values.maturity_status.isna().any():
                    raise ValueError("Forecast missing from frozen six-method label report")
            groups[lane], specs[lane] = values, spec
    if len(groups) != 7:
        raise ValueError("Expected C and six unchanged methods")
    # Every C forecast is a member of the original absolute-return cohort. Borrow its
    # exact same stock/session economic outcome, including its unresolved label status.
    labels = groups["absolute"][["stock_code", *LABEL_FIELDS]]
    groups[ARM] = groups[ARM].merge(labels, how="left", on="stock_code", validate="one_to_one")
    if groups[ARM].maturity_status.isna().any():
        raise ValueError("No auditable original economic-label status for a C forecast")
    return rec, groups, specs


def orig_mismatch(rec, computed_track, panel, original):
    return (computed_track != rec["track"] or rec["prospective_eligible"] != (computed_track == "PROSPECTIVE")
            or (computed_track == "PROSPECTIVE" and not (original["prospective_eligible"] and panel["prospective_eligible"])))


def capture_gaps(calendar, policy_stamp, observed_timely, observed_prospective):
    """Disclose the deployment session even when its forecasts cannot count towards review."""
    moment = parse_clock(policy_stamp)
    first_due = moment.date()
    if moment.time() >= time(18):
        first_due += timedelta(days=1)
    missing_timely = [d for d in calendar if d >= first_due.isoformat() and d not in observed_timely]
    missing_forward = [d for d in calendar if d > moment.date().isoformat() and d not in observed_prospective]
    return missing_timely, missing_forward


def review_state(summary):
    states = {}
    empty = {"status": "INSUFFICIENT_PROSPECTIVE_DATA", "jointly_valid_days": 0, "label_coverage": None,
             "rank_ic_days": 0, "full_top20_days": 0, "disjoint_joint_hold_windows": 0}
    for scope in ("OWN", "COMMON7"):
        match = next((r for r in summary if r["track"] == "PROSPECTIVE"
                      and r["lane"] == ARM and r["scope"] == scope), None)
        states[scope] = {k: match[k] for k in empty} if match else empty.copy()
    return states


def compare_day(rec, cohorts, specs, sessions, calendar, history):
    common = set.intersection(*(set(frame.stock_code) for frame in cohorts.values()))
    day, track_name = rec["trade_date"], rec["track"]
    ordinal = sessions.get(day)
    mature = ordinal is not None and ordinal + 6 < len(calendar)
    rows = []
    for lane, frame in cohorts.items():
        paired = frame.loc[frame.stock_code.isin(common)]
        for scope, cohort in (("OWN", frame), ("COMMON7", paired)):
            key = track_name, lane, scope
            last, prior = history.get(key, (None, {}))
            if last is None or ordinal is None or ordinal != last + 1:
                prior = {}
            metrics = ranked_daily(cohort, cohort, lane == "equal_weight", prior)
            history[key] = ordinal, prior
            denominator = specs[lane].get("candidate_count", specs[lane]["rows"] + specs[lane]["abstentions"])
            rows.append(metrics | {"day":day,"track":track_name,"scope":scope,"support_group":"ALL",
                "lane":lane,"model_sha256":specs[lane]["weights_sha256"],"signal_session":ordinal,
                "signal_matured":mature,"formal":rec["formal_session"],"captured":True,"candidates":denominator,
                "prediction_coverage":len(cohort)/denominator if denominator else None,
                "entry_date":calendar[ordinal + 1] if mature else None,"exit_date":calendar[ordinal + 6] if mature else None})
    return rows


def report(out, panel, legacy, asof):
    source = panel / "reports" / asof.isoformat() / REPORT_AUDIT
    if not source.exists():
        return {"status":"DEFERRED","reason":"WAITING_FOR_SIX_METHOD_MATURITY"}
    audit = read_json(source)
    labels = parquet_read(artifact(source.parent, LABEL_REPORT, audit["files"][LABEL_REPORT]))
    if labels.prediction_id.duplicated().any():
        raise ValueError("Duplicate original prediction ID")
    digest = hash_file(out / "policy.json")
    policy = read_json(out / "policy.json")
    if policy["panel_policy_sha256"] != audit["panel_policy_sha256"]:
        raise ValueError("Original six-method evaluation policy changed")
    cutoff = date.fromisoformat(audit["formal_through"])
    if cutoff > asof:
        raise ValueError("Six-method report used future data")
    audited = {i["sha256"] for i in audit["input_manifests"]}
    with exclusive(out):
        def build(stage):
            files = sorted(p for kind in ("historical","timely") for p in (out / kind).glob("*/manifest.json")
                           if p.parent.name <= asof.isoformat())
            freeze_date = parse_clock(policy["frozen_at"]).date()
            start = min([freeze_date, *(date.fromisoformat(p.parent.name) for p in files)])
            calendar, _ = formal_tail(AShareDailyLoader().engine, start, [], now_shanghai(), cutoff)
            sessions = {d:i for i,d in enumerate(calendar)}
            records, daily, history = [], [], {}
            timely, prospective = set(), set()
            all_ids = set()
            for path in files:
                rec, groups, specs = load_cohorts(path.parent, policy, digest, panel, legacy, labels, audited)
                ids = set(groups[ARM].prediction_id)
                if all_ids.intersection(ids):
                    raise ValueError("Duplicate C prediction ID across days or tracks")
                all_ids.update(ids)
                if rec["capture_kind"] == "TIMELY" and rec["formal_session"]:
                    timely.add(rec["trade_date"])
                if rec["track"] == "PROSPECTIVE":
                    prospective.add(rec["trade_date"])
                records.append({"day":rec["trade_date"],"track":rec["track"],"sha256":hash_file(path)})
                daily.extend(compare_day(rec, groups, specs, sessions, calendar, history))
            missing_timely, missing_forward = capture_gaps(calendar, policy["frozen_at"], timely, prospective)
            summary = summarize(daily)
            stage.mkdir()
            csv = stage / "daily_comparison.csv"
            pd.DataFrame(daily).to_csv(csv, index=False)
            atomic_json(stage / "comparison.json", {"version":VERSION,"evaluated_at":now_shanghai().isoformat(),
                "asof":asof.isoformat(),"formal_through":audit["formal_through"],"six_method_audit_sha256":hash_file(source),
                "c_policy_sha256":digest,"captures":records,"missing_c_prospective_sessions":missing_forward,
                "missing_c_timely_sessions_since_policy_freeze":missing_timely,
                "columns_scope":"OWN or all seven archived forecasts intersected before label filtering",
                "comparison_track":"Aligned to C policy eligibility. Historical capture of timely parents remains historical.",
                "review_thresholds": {"minimum_joint_mature_sessions": MIN_MATURED_DAYS,
                                      "minimum_mature_label_coverage": MIN_LABEL_COVERAGE},
                "group_c_forward_review":review_state(summary),"summary":summary,
                "daily_sha256":hash_file(csv),"auto_prediction_ability_pass":False})
        dest = publish_revision(out/"reports", asof.isoformat(),"comparison.json",build)
        data = read_json(dest / "comparison.json")
        artifact(dest, "daily_comparison.csv", data["daily_sha256"])
        return {"status":"REPORTED","asof":data["asof"],"captures":len(data["captures"]),"output":str(dest)}
