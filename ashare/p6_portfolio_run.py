"""Archive daily paper simulations from immutable predictions and read-only formal bars."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, timedelta
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from . import backtest, p6_settlements

from . import p6_portfolio as engine
from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, exclusive, read_json
from .p4_subset_inputs import hash_file, parquet_read
from .p55_daily import now_shanghai, parse_clock
from .p55_mature import formal_tail
from .p55_reports import artifact, read_panel
from .p6_portfolio import PortfolioConfig, STRATEGIES, initial_state, positive, step
from .p6_portfolio_report import publish_day

LANES = ("ma20_reversal", "factors18", "group_c", "absolute", "excess", "rank")
CONFIG = PortfolioConfig()


def freeze(root):
    core = {"version":engine.VERSION, "config":asdict(CONFIG), "strategies":list(STRATEGIES),
            "lanes":list(LANES), "implementation_sha256":{Path(f).name:hash_file(Path(f)) for f in
                (engine.__file__, backtest.__file__, p6_settlements.__file__, __file__, publish_day.__code__.co_filename)},
            "forward": "Only timely PROSPECTIVE signals of a session whose 18:00 boundary is after policy freeze; execute next formal open",
            "benchmark": "previous archived equal-weight roster, gross daily open rebalance, overnight drift; incomplete economics unavailable",
            "research_only": True}
    with exclusive(root):
        path = root/"policy.json"
        if path.exists():
            old = read_json(path)
            if {k:v for k,v in old.items() if k != "frozen_at"} != core:
                raise ValueError("Changed portfolio policy or implementation requires a new version directory")
            return old
        result = core | {"frozen_at":now_shanghai().isoformat()}
        atomic_json(path,result)
        return result


def usable_signal(day, committed, freeze):
    boundary = parse_clock(day+"T18:00:00+08:00")
    published = parse_clock(committed)
    return freeze < boundary <= published and published.date().isoformat() == day


def archived_lists(panel, legacy, group, track, freeze, end):
    records, all_rows, _, _ = read_panel(panel,legacy)
    scores = {(kind,day,lane):frame.sort_values(["score","stock_code"],ascending=[False,True]).stock_code.tolist()
              for (kind,day,lane),frame in all_rows.groupby(["track","trade_date","objective"])}
    out = {}
    # A timely record takes precedence if a diagnostic replay exists for the same day.
    for rec in sorted(records,key=lambda r: r["track"] != "HISTORICAL"):
        day,lane,kind = rec["day"],rec["lane"],rec["track"]
        if day > end or not rec["formal"] or (track == "forward" and (kind != "PROSPECTIVE" or not usable_signal(day,read_json(Path(rec["source"]))["committed_at"],freeze))):
            continue
        out[day,lane] = (scores.get((kind,day,lane),[]),hash_file(Path(rec["source"])))
    if group and (group/"policy.json").is_file():
        from .p58_group_c import read_capture
        digest = hash_file(group/"policy.json")
        for folder in ("historical","timely"):
            for path in sorted((group/folder).glob("????-??-??/manifest.json")):
                day = path.parent.name
                if day > end or (track == "forward" and folder != "timely"):
                    continue
                record = read_capture(path.parent,digest)
                if (record["trade_date"] != day or record["capture_kind"] != folder.upper()
                        or not record["formal_session"]):
                    continue
                if track == "forward" and (not record["prospective_eligible"]
                        or not usable_signal(day,record["committed_at"],freeze)
                        or day <= parse_clock(read_json(group/"policy.json")["frozen_at"]).date().isoformat()):
                    continue
                spec = record["models"]["group_c"]
                data = parquet_read(artifact(path.parent,spec["predictions_file"],spec["predictions_sha256"]))
                parent_kind = record["parent_kind"]
                if parent_kind not in ("historical","timely") or any(
                        hash_file(ledger/parent_kind/day/"manifest.json") != record[key]
                        for ledger,key in ((panel,"panel_manifest_sha256"),(legacy,"legacy_manifest_sha256"))):
                    raise ValueError("C capture ancestry differs from original scored lists")
                ordered = data.sort_values(["score","stock_code"],ascending=[False,True]).stock_code.tolist()
                if (data.stock_code.duplicated().any() or data.prediction_id.duplicated().any()
                        or not data.score.map(math.isfinite).all() or len(data) != spec["rows"]
                        or not data.stock_code.astype("string").str.fullmatch(r"\d{6}").all()
                        or not data.trade_date.eq(day).all() or not data.model_sha256.eq(spec["weights_sha256"]).all()
                        or not data.model_version.eq(spec["model_version"]).all() or not data.objective.eq("group_c").all()
                        or ordered[:20] != spec["top20"]):
                    raise ValueError("Invalid archived C roster")
                out[day,"group_c"] = ordered, hash_file(path)
    return out


def evidence_for(legacy, day, track):
    # Historical replays can use their explicitly archived witness only for diagnostic closing marks.
    folders = ["timely"] if track == "forward" else ["timely","historical"]
    for folder in folders:
        path = legacy/folder/day/"manifest.json"
        if not path.is_file():
            continue
        ref = read_json(path).get("evidence_snapshot") or {}
        if ref.get("root"):
            witness = artifact(Path(ref["root"])/"dataset","inference_evidence.json",ref["sha256"])
            obj = read_json(witness)
            if obj["policy"] != "P5_DUAL_OBSERVED_RESEARCH_WINDOW_V1" or obj["observed_at_utc"] != ref["observed_at_utc"]:
                raise ValueError("Unexpected source evidence for portfolio")
            if track == "forward" and parse_clock(obj["observed_at_utc"]).date().isoformat() != day:
                return {},None
            return obj["stocks"],ref["sha256"]
    return {},None


def action_flags(stocks,day,codes):
    result = {}
    for code in codes:
        witness = stocks.get(code)
        if not witness or not witness["covered_from"] <= day <= witness["covered_through"]:
            result[code] = {"status":"MISSING_COVERAGE","available_at_open":False}
        elif any(row[0] == day for row in witness["actions"]):
            # Observation-only feeds don't establish cash payment or share listing dates.
            result[code] = {"status":"UNKNOWN","available_at_open":False}
    return result


def benchmark(previous, targets, bars, actions):
    """Gross theoretical fractional-share comparator; do not drop missing constituents."""
    overnight = 1.
    for code,weight in previous.items():
        bar = bars.get(code,{})
        op,old = positive(bar.get("open")),positive(bar.get("prior_close"))
        if not op or not old or code in actions:
            overnight = None
            break
        overnight += weight * (op/old-1)
    if targets is None:
        # No fresh portfolio instruction: retain the former basket with drift.
        if previous and overnight is None:
            return {},None,False
        allocation = {c:w*bars[c]["open"]/bars[c]["prior_close"]/overnight for c,w in previous.items()}
    else:
        allocation = {c:1/len(targets) for c in targets}
    end = {}
    for code,weight in allocation.items():
        bar = bars.get(code,{})
        op,cl = positive(bar.get("open")),positive(bar.get("close"))
        if not op or not cl:
            return {},None,False
        end[code] = weight * cl/op
    intraday = sum(end.values()) if end else 1.
    return {c:w/intraday for c,w in end.items()}, (overnight*intraday-1 if overnight is not None else None), True


def checked_day(path, prev_hash, policy_sha):
    summary = read_json(path/"portfolio_summary.json")
    if summary["day"] != path.name or summary["previous_sha256"] != prev_hash or summary["policy_sha256"] != policy_sha:
        raise ValueError("Broken account archive chain")
    for name,sha in summary["files"].items():
        artifact(path,name,sha)
    return read_json(path/"checkpoint.json"),hash_file(path/"portfolio_summary.json")


def run(root,legacy,panel,group,track,end,start=None):
    if not (root/"policy.json").is_file():
        raise ValueError("Freeze this paper policy before the first run")
    policy = freeze(root)
    frozen_day = parse_clock(policy["frozen_at"]).date().isoformat()
    if track == "forward" and start:
        raise ValueError("Prospective series always starts at its frozen boundary")
    if track == "diagnostic" and not start:
        raise ValueError("Use an explicit diagnostic start")
    folder = root/("forward" if track == "forward" else ("diagnostic-"+start))
    policy_sha = hash_file(root/"policy.json")
    now = now_shanghai()
    if track == "forward" and (end < frozen_day or (end == now.date().isoformat() == frozen_day and now.hour < 18)):
        return {"status":"AWAITING_FIRST_POST_FREEZE_SIGNAL","frozen_at":policy["frozen_at"]}
    first = date.fromisoformat(frozen_day if track == "forward" else start)
    if date.fromisoformat(end) < first:
        raise ValueError("Invalid simulation bounds")
    with exclusive(folder):
        lists = archived_lists(panel,legacy,group,track,parse_clock(policy["frozen_at"]),end)
        universe = {code for (day,_),(codes,_) in lists.items() if day >= first.isoformat() for code in codes}
        calendar,raw = formal_tail(AShareDailyLoader().engine, first-timedelta(days=10), sorted(universe),now,date.fromisoformat(end))
        bars_by_day = {day:g.set_index("stock_code").to_dict("index") for day,g in raw.groupby("trade_date")}
        checkpoints = {"accounts":{}, "benchmark_weights":{}, "benchmark_composition_known":True}
        previous_sha = None
        locked_bars = {}
        if track == "forward":
            first_signal = min((d for (d,lane) in lists if lane in LANES), default=None)
            if first_signal is None:
                return {"status":"AWAITING_FIRST_POST_FREEZE_SIGNAL","frozen_at":policy["frozen_at"]}
        else:
            first_signal = first.isoformat()
        published = []
        for i,day in enumerate(calendar):
            if day < first_signal:
                continue
            path = folder/day
            if path.exists():
                checkpoints,previous_sha = checked_day(path,previous_sha,policy_sha)
                locked_bars = read_json(path/"inputs.json")["bars"]
                continue
            prior = calendar[i-1] if i and calendar[i-1] >= first_signal else None
            bars = bars_by_day.get(day,{})
            old_bars = locked_bars
            previous_accounts = checkpoints["accounts"]
            needed = set(checkpoints["benchmark_weights"])
            for state in previous_accounts.values():
                needed.update(state["holdings"])
            for key in ((d, lane) for d in (day,prior) for lane in (*LANES,"equal_weight")):
                needed.update(lists.get(key,([],None))[0])
            quotes = {code:{k:positive(row.get(k)) for k in ("open","close","pre_close")} |
                      {"prior_close":positive(old_bars.get(code,{}).get("close"))} for code,row in bars.items() if code in needed}
            stocks,evidence_sha = evidence_for(legacy,day,track)
            orig = checkpoints["accounts"]
            active = set(checkpoints["benchmark_weights"])
            for state in orig.values():
                active.update(state["holdings"])
            actions = action_flags(stocks,day,active)
            # Only terms actually archived by the simulated close can affect that day.
            cutoff = min(now,parse_clock(day+"T23:59:59.999999+08:00"))
            verified,settlement_hashes = p6_settlements.load_settlements(root/"settlements",day,cutoff)
            for code, terms in verified.items():
                if code in active:
                    missing = actions.get(code, {}).get("status") == "MISSING_COVERAGE"
                    actions[code] = terms | {"coverage_unverified": missing}
            targets,base_sha = lists.get((prior,"equal_weight"),(None,None))
            known = checkpoints["benchmark_composition_known"]
            weights,ret,known_now = benchmark(checkpoints["benchmark_weights"],targets,quotes,actions)
            if not known:
                ret = None
                known_now = targets is not None and known_now
            checkpoints["benchmark_weights"],checkpoints["benchmark_composition_known"] = weights,known_now
            results = {key:[] for key in ("equity","positions","trades","rejections")}
            source = {}
            for lane in LANES:
                ranking,sha = lists.get((prior,lane),(None,None))
                source[lane] = sha
                for strategy in STRATEGIES:
                    key = lane + "/" + strategy
                    state,daily,pos,trades,refused = step(orig.get(key) or initial_state(CONFIG),day=day,
                        prior_day=prior,ranked=ranking,bars=quotes,strategy=strategy,config=CONFIG,
                        actions=actions,benchmark_return=ret)
                    orig[key] = state
                    for name,rows in zip(results,([daily],pos,trades,refused),strict=True):
                        results[name].extend(dict(r,model=lane,strategy=strategy,trade_date=day) for r in rows)
            timing = ("HISTORICAL_DIAGNOSTIC" if track=="diagnostic" else
                      "SAME_DAY_SIMULATION" if day==now.date().isoformat() else "RECOVERED_SIMULATION")
            with TemporaryDirectory(dir=folder) as tmp:
                stage = Path(tmp)/"day"
                stage.mkdir()
                atomic_json(stage/"checkpoint.json",checkpoints)
                atomic_json(stage/"inputs.json",{"signal_day":prior,"rankings_manifest_sha256":source,
                    "benchmark_manifest_sha256":base_sha,"closing_evidence_sha256":evidence_sha,
                    "settlement_evidence_sha256":settlement_hashes,"bars":quotes,"actions":actions,"benchmark_daily_return":ret})
                publish_day(stage,day,track,timing,results,policy_sha,previous_sha,now)
                os.rename(stage,path)
            checkpoints,previous_sha = checked_day(path,previous_sha,policy_sha)
            locked_bars = quotes
            published.append(day)
        return {"status":"OK","track":track,"new_days":published,"last_day":calendar[-1] if calendar else None,
                "latest_summary_sha256":previous_sha,"output":str(folder)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["freeze","run"])
    p.add_argument("--out",type=Path,required=True)
    for name in ("legacy","panel","group"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--track",choices=["forward","diagnostic"],default="forward")
    p.add_argument("--start",type=date.fromisoformat)
    p.add_argument("--end",type=date.fromisoformat)
    args = p.parse_args()
    if args.action=="freeze":
        print(freeze(args.out))
        return
    if not (args.panel and args.legacy and args.end):
        p.error("run requires --panel --legacy --end")
    print(run(args.out,args.legacy,args.panel,args.group,args.track,args.end.isoformat(),
              args.start.isoformat() if args.start else None))


if __name__=="__main__":
    main()
