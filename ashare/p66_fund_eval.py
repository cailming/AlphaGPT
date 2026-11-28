"""Evaluate archived fund rankings only after scheduled economic-return windows mature."""
from __future__ import annotations

from collections import Counter
from datetime import date, time, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .p4_action_pipeline import atomic_json, read_json
from .p4_corporate_actions import load_actions
from .p4_subset_inputs import hash_file, parquet_read, parquet_write
from .p55_mature import COSTS, MIN_LABEL_COVERAGE, OFFICIAL, formal_tail, one_label, select_evidence
from .p58_fund_snapshot import sha
from .p66_calendar import next_session
from .p66_fund_research import read_capture


def label_path(day, entry, observed, available_day):
    scheduled = [day] + [next_session(day, i) for i in range(1, 7)]
    if any(d is None for d in scheduled):
        return "SCHEDULE_UNCONFIRMED", []
    if str(scheduled[1]) != entry:
        raise ValueError("Calendar and frozen entry date disagree")
    window = list(map(str, scheduled))
    if scheduled[-1] > available_day:
        return "LABEL_PENDING", window
    if not set(window).issubset(observed):
        return "MISSING_PLANNED_FORMAL_SESSION", window
    return "READY", window


def metric(frame, name):
    ranked = frame.loc[frame[name].notna()].sort_values([name, "stock_code"], ascending=[False, True])
    labelled = ranked.dropna(subset=["economic_return"])
    n = len(ranked)
    ratio = len(labelled)/n if n else None
    ok = ratio is not None and ratio >= MIN_LABEL_COVERAGE
    equal = float(labelled.economic_return.mean()) if ok else None
    rank_ic = None
    if ok and len(labelled) >= 5 and labelled[name].nunique() > 1 and labelled.economic_return.nunique() > 1:
        rank_ic = float(labelled[name].rank().corr(labelled.economic_return.rank()))
    quintiles = []
    bins = np.ceil(5 * ranked[name].rank(pct=True))  # equal scores share a quantile
    for group in range(1, 6):
        values = ranked.loc[bins.eq(group), "economic_return"]
        quotient = values.count()/len(values) if len(values) else 0
        quintiles.append({"quintile":group,"selected":len(values),"labelled":int(values.count()),
                          "mean":float(values.mean()) if quotient >= MIN_LABEL_COVERAGE else None})
    tops = {}
    for k in (10, 20):
        top = ranked.head(k)
        values = top.economic_return.dropna()
        gross = float(values.mean()) if len(values) == k else None
        excess = gross-equal if gross is not None and equal is not None else None
        tops[str(k)] = {"selected":len(top),"labelled":len(values),"mean":gross,"excess":excess,
                        "net_excess_by_per_side_bps":{str(cost):excess-2*cost/10_000 if excess is not None else None for cost in COSTS}}
    return {"eligible":n, "labelled":len(labelled), "label_coverage":ratio, "rank_ic":rank_ic,
            "equal_weight_return":equal, "top":tops,"quintiles":quintiles}


def summarize(frame, name):
    own = metric(frame, name)
    common = frame.dropna(subset=[name, "ma20"])
    return {"factor":own, "paired_factor":metric(common, name), "paired_ma20":metric(common, "ma20")}


def evaluate(out, legacy, policy, now, engine):
    captures = []
    codes = set()
    for path in sorted((out/"predictions").glob("????-??-??/*/manifest.json")):
        capture = read_capture(path, policy)
        frame = parquet_read(path.parent/"scores.parquet")
        codes.update(frame.stock_code)
        captures.append((path, capture, frame))
    if not captures:
        return None
    first = min(date.fromisoformat(rec["signal"]) for _,rec,_ in captures)
    calendar, raw = formal_tail(engine, first, sorted(codes), now)
    available = now.date() - timedelta(days=now.time() < time(18))
    evidence = select_evidence(legacy, None, None)
    official = {(row.stock_code, row.ex_date.date().isoformat()):
                (float(row.share_multiplier), float(row.cash_per_old_share), row.source_id)
                for row in load_actions(OFFICIAL).itertuples(index=False)}
    prices = {code:bars.set_index("trade_date") for code,bars in raw.groupby("stock_code")}
    report, label_rows = [], []
    for path, rec, frame in captures:
        signal, entry = rec["signal"], rec["entry"]
        status, dates = label_path(date.fromisoformat(signal), entry, calendar, available)
        sessions = {d:i for i,d in enumerate(dates)}
        labels = []
        for code in frame.stock_code:
            label, _, _, value = (one_label(code, signal, dates, sessions, prices, evidence, official) if status == "READY"
                                   else (status, "", "", None))
            labels.append(value)
            label_rows.append({"signal":signal, "entry":entry, "exit":dates[-1] if dates else None,
                               "source":rec["source"], "stock_code":code, "label_status":label, "economic_return":value})
        frame = frame.assign(economic_return=pd.Series(labels, dtype=float))
        report.append({"signal":signal,"entry":entry,"exit":dates[-1] if dates else None,
                       "source":rec["source"], "capture_sha256":hash_file(path),
                       "window_status":status,"evaluation":{f:summarize(frame,f) for f in rec["factors"]}})
    lineage = {"reports":report, "formal_through":calendar[-1] if calendar else None,
               "company_evidence_sha256":evidence["sha256"],"issuer_actions_sha256":hash_file(OFFICIAL),
               "label_statuses":dict(Counter(x["label_status"] for x in label_rows)), "labels_sha256":sha(label_rows)}
    root = out/"evaluations"
    root.mkdir(exist_ok=True)
    dest = root/sha(lineage)
    if not dest.exists():
        with TemporaryDirectory(dir=root) as tmp:
            stage = Path(tmp)/"report"
            stage.mkdir()
            parquet_write(pd.DataFrame(label_rows,columns=("signal","entry","exit","source","stock_code","label_status","economic_return")),
                          stage/"labels.parquet")
            atomic_json(stage/"report.json", lineage | {"evaluated_at":now.isoformat(),"policy_sha256":policy,
                        "label_file_sha256":hash_file(stage/"labels.parquet")})
            stage.rename(dest)
    stored = read_json(dest/"report.json")
    if (stored["policy_sha256"] != policy or any(stored[key] != value for key,value in lineage.items())
            or hash_file(dest/"labels.parquet") != stored["label_file_sha256"]):
        raise ValueError("Fund evaluation archive failed integrity verification")
    return dest
