"""Evaluate immutable P55 forecast panels and publish read-only maturity reports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, read_json
from .p4_corporate_actions import load_actions
from .p4_subset_inputs import hash_file, parquet_read, parquet_write
from .p55_mature import (GROUPS, HORIZON, MIN_LABEL_COVERAGE, MIN_MATURED_DAYS, OBJECTIVES, OFFICIAL,
                         PREDICTION_FIELDS, formal_tail, load_ledger, one_label, select_evidence, support_group)
from .p5_sequences import CHINA

VERSION = "P55_PANEL_REPORT_V1"
PANEL_VERSION = "P55_DAILY_PANEL_V1"
PANEL_LANES = ("factors18", "ma20_reversal", "equal_weight")
LANES = (*OBJECTIVES, *PANEL_LANES)
RESEARCH_COST_BPS_PER_SIDE = 10
TRACKS = ("PROSPECTIVE", "TIMELY_NONPROSPECTIVE", "HISTORICAL")
NAMES = {"absolute": "十因子绝对收益", "excess": "十因子超额收益", "rank": "十因子百分位",
         "factors18": "十八因子 Transformer", "ma20_reversal": "MA20 反转", "equal_weight": "等权股票池"}
LABEL_FIELDS = ("maturity_status", "entry_date", "exit_date", "target_return")


def timestamp(value):
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise ValueError("Capture and freeze timestamps require a timezone")
    return moment.astimezone(CHINA)


def artifact(folder, filename, sha):
    if Path(filename).name != filename:
        raise ValueError(f"Expected a local immutable artifact name: {filename}")
    path = folder / filename
    if hash_file(path) != sha:
        raise ValueError(f"Captured file digest mismatch: {path}")
    return path


def model_capture(record, track, lane, spec, source):
    candidates = spec.get("candidate_count", spec["rows"] + spec["abstentions"])
    counts = spec["rows"], spec["abstentions"], candidates
    if (any(type(n) is not int or n < 0 for n in counts) or counts[0] + counts[1] != counts[2]
            or sum(spec["abstention_reasons"].values()) != counts[1]):
        raise ValueError(f"Inconsistent capture denominators for {lane}/{record['trade_date']}")
    kind = spec.get("portfolio_kind", "RANKED")
    if kind != ("EQUAL_WEIGHT" if lane == "equal_weight" else "RANKED"):
        raise ValueError(f"Unexpected portfolio kind for {lane}: {kind}")
    status = "ABSTAINED" if record["status"] == "SCORED" and counts[0] == 0 else record["status"]
    freeze = timestamp(spec["model_frozen_at"])
    if "policy_frozen_at" in record:
        freeze = max(freeze, timestamp(record["policy_frozen_at"]))
    return {"day": record["trade_date"], "track": track, "lane": lane, "version": spec["model_version"],
            "model_sha256": spec["weights_sha256"], "freeze_day": freeze.date().isoformat(),
            "status": status, "formal": bool(record["formal_session"]), "portfolio_kind": kind,
            "captured": True, "candidates": counts[2], "abstentions": counts[1],
            "abstention_reasons": spec["abstention_reasons"], "source": str(source)}


def read_panel(panel, legacy):
    records, frames, provenance = [], [], []
    old_manifests = {}
    if legacy is not None:
        old_records, old_rows = load_ledger(legacy)
        for entry in old_records:
            folder = "historical" if entry["track"] == "HISTORICAL" else "timely"
            path = legacy / folder / entry["trade_date"] / "manifest.json"
            old = read_json(path)
            if hash_file(path) != entry["manifest_sha256"]:
                raise ValueError("Legacy ledger changed during evaluation")
            old_manifests[entry["trade_date"], old["capture_kind"]] = entry["manifest_sha256"]
            provenance.append({"source": str(path), "sha256": entry["manifest_sha256"]})
            for lane in OBJECTIVES:
                records.append(model_capture(old, entry["track"], lane, old["models"][lane], path))
        if len(old_rows):
            frames.append(old_rows.assign(portfolio_weight=np.nan))
    policy = panel / "policy.json"
    policy_digest = hash_file(policy) if policy.is_file() else None
    for folder, capture in (("timely", "TIMELY"), ("historical", "HISTORICAL")):
        for path in sorted((panel / folder).glob("????-??-??/manifest.json")):
            raw = read_json(path)
            day = raw["trade_date"]
            if (policy_digest is None or raw["schema_version"] != PANEL_VERSION or raw["capture_kind"] != capture
                    or day != path.parent.name or date.fromisoformat(day).isoformat() != day
                    or set(raw["models"]) != set(PANEL_LANES) or raw["policy_sha256"] != policy_digest
                    or raw["uses_future_labels"] is not False):
                raise ValueError(f"Unexpected panel manifest or policy: {path}")
            old_key = day, raw.get("legacy_capture_kind", capture)
            if legacy is not None and old_manifests.get(old_key) != raw["legacy_manifest_sha256"]:
                raise ValueError(f"Panel lost its original V2 capture dependency: {path}")
            published = timestamp(raw["committed_at"])
            freeze_dates = [timestamp(raw["policy_frozen_at"]).date(),
                            *(timestamp(s["model_frozen_at"]).date() for s in raw["models"].values())]
            eligible = bool(raw["prospective_eligible"])
            if eligible and (capture != "TIMELY" or not raw["formal_session"]
                             or published.date().isoformat() != day or published.time() < time(18)
                             or date.fromisoformat(day) <= max(freeze_dates)):
                raise ValueError(f"Panel capture is outside its prospective window: {path}")
            track = "PROSPECTIVE" if eligible else ("HISTORICAL" if capture == "HISTORICAL" else "TIMELY_NONPROSPECTIVE")
            provenance.append({"source": str(path), "sha256": hash_file(path)})
            for lane, spec in raw["models"].items():
                meta = model_capture(raw, track, lane, spec, path)
                meta["freeze_day"] = max(freeze_dates).isoformat()
                pred = artifact(path.parent, spec["predictions_file"], spec["predictions_sha256"])
                artifact(path.parent, spec["candidates_file"], spec["candidates_sha256"])
                table = parquet_read(pred)
                if (not set((*PREDICTION_FIELDS, "portfolio_weight")).issubset(table)
                        or len(table) != spec["rows"] or table.stock_code.duplicated().any()
                        or table.prediction_id.duplicated().any()
                        or not table.stock_code.astype("string").str.fullmatch(r"\d{6}").all()
                        or not table.trade_date.eq(day).all() or not table.objective.eq(lane).all()
                        or not table.model_version.eq(spec["model_version"]).all()
                        or not table.model_sha256.eq(spec["weights_sha256"]).all()
                        or not table.training_sequences.map(support_group).eq(table.support_group).all()
                        or not np.isfinite(table.score.to_numpy(dtype=float)).all()):
                    raise ValueError(f"Malformed frozen panel predictions: {pred}")
                if lane == "equal_weight" and len(table):
                    weights = table.portfolio_weight.to_numpy(dtype=float)
                    if not (table.score.eq(0).all() and np.allclose(weights, 1.0 / len(table))):
                        raise ValueError(f"Unexpected equal-weight predictions: {pred}")
                for k in (10, 20):
                    ordered = table.sort_values(["score", "stock_code"], ascending=(False, True)).stock_code.tolist()
                    expected = [] if lane == "equal_weight" else ordered[:k]
                    if expected != spec[f"top{k}"]:
                        raise ValueError(f"Published Top-{k} does not match frozen scores: {pred}")
                records.append(meta)
                if len(table):
                    frames.append(table.loc[:, [*PREDICTION_FIELDS, "portfolio_weight"]].assign(track=track))
    fields = [*PREDICTION_FIELDS, "portfolio_weight", "track"]
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=fields)
    if rows.prediction_id.duplicated().any():
        raise ValueError("Forecast IDs must remain unique across all publication tracks")
    return records, rows, provenance, policy_digest


def with_labels(forecasts, calendar, raw, evidence, official):
    sessions = {d: i for i, d in enumerate(calendar)}
    price_paths = {code: data.set_index("trade_date") for code, data in raw.groupby("stock_code")}
    keys = list(forecasts[["stock_code", "trade_date"]].itertuples(index=False, name=None))
    labels = {key: one_label(*key, calendar, sessions, price_paths, evidence, official) for key in set(keys)}
    column_values = pd.DataFrame([labels[key] for key in keys], columns=LABEL_FIELDS, index=forecasts.index)
    return pd.concat([forecasts, column_values], axis=1)


def panel_coverage(rows):
    observed = int(rows.target_return.notna().sum())
    return observed, observed / len(rows) if len(rows) else None


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def amount(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def set_replacement(previous, choices):
    return len(previous - choices) / len(previous) if previous else None


def ranked_daily(rows, pool, equal, prior):
    labels, coverage = panel_coverage(rows)
    observed = rows.loc[rows.target_return.notna()]
    _, pool_coverage = panel_coverage(pool)
    pool_return = (float(pool.target_return.mean()) if pool_coverage is not None
                   and pool_coverage >= MIN_LABEL_COVERAGE else None)
    rank_ic = None
    if (not equal and coverage is not None and coverage >= MIN_LABEL_COVERAGE and labels >= 5
            and observed.score.nunique() > 1 and observed.target_return.nunique() > 1):
        rank_ic = float(observed.score.rank().corr(observed.target_return.rank()))
    cost = 2 * RESEARCH_COST_BPS_PER_SIDE / 10000
    pnl = float(observed.target_return.mean()) if equal and (coverage or 0) >= MIN_LABEL_COVERAGE else None
    portfolio_names = set(rows.stock_code)
    output = {"predictions": len(rows), "labels": labels, "label_coverage": coverage,
              "pending": int(rows.maturity_status.eq("LABEL_PENDING").sum()), "rank_ic": rank_ic,
              "label_reasons": dict(Counter(rows.maturity_status)), "equal_weight_pool_return": pool_return,
              "portfolio_return": pnl,
              "portfolio_net_excess": pnl - pool_return - cost if pnl is not None and pool_return is not None else None,
              "portfolio_replacement": set_replacement(prior.get("portfolio"), portfolio_names) if equal else None}
    tops = rows.sort_values(["score", "stock_code"], ascending=(False, True), kind="stable")
    prior["portfolio"] = portfolio_names
    for k in (10, 20):
        chosen = rows.iloc[:0] if equal else tops.head(k)
        names = chosen.stock_code.tolist()
        count, selected_coverage = panel_coverage(chosen)
        gross = float(chosen.target_return.mean()) if count == k else None
        excess = gross - pool_return - cost if gross is not None and pool_return is not None else None
        new = set(names)
        output.update({f"top{k}_stocks": names, f"top{k}_labels": count, f"top{k}_label_coverage": selected_coverage,
                       f"top{k}_full": count == k and pool_return is not None,
                       f"top{k}_return": gross, f"top{k}_net_excess": excess,
                       f"top{k}_replacement": set_replacement(prior.get(k), new) if not equal else None})
        prior[k] = new
    return output


def captures_with_gaps(records, calendar):
    observed = {(r["track"], r["day"], r["lane"], r["model_sha256"]) for r in records}
    identities = {(r["lane"], r["model_sha256"]): r for r in records}
    if len({lane for lane, _ in identities}) != len(identities):
        raise ValueError("A changed roster requires a new immutable forecast journal")
    missing = []
    for (lane, model), r in identities.items():
        for day in calendar:
            key = "PROSPECTIVE", day, lane, model
            if day > r["freeze_day"] and key not in observed:
                missing.append(r | {"track": "PROSPECTIVE", "day": day, "formal": True,
                                    "status": "MISSING_PROSPECTIVE_CAPTURE", "captured": False,
                                    "candidates": None, "abstentions": None, "abstention_reasons": {}, "source": None})
    return [*records, *missing]


def evaluate_days(records, labelled, calendar):
    sessions = {day: i for i, day in enumerate(calendar)}
    empty = labelled.iloc[0:0]
    groups = {(track, day, lane): rows for (track, day, lane), rows in labelled.groupby(
        ["track", "trade_date", "objective"], sort=False)}
    by_day = defaultdict(list)
    for record in captures_with_gaps(records, calendar):
        by_day[record["track"], record["day"]].append(record)
    history = {}
    output = []
    for (track, day), captures in sorted(by_day.items()):
        scored = {r["lane"]: groups.get((track, day, r["lane"]), empty) for r in captures}
        common = None
        if set(scored) == set(LANES) and all(r["captured"] and r["formal"] for r in captures):
            # Intersect forecasts before any label outcomes are considered.
            common = set.intersection(*(set(frame.stock_code) for frame in scored.values()))
        for record in captures:
            lane = record["lane"]
            original = scored[lane]
            scopes = {"OWN": original}
            if common is not None:
                scopes["COMMON"] = original.loc[original.stock_code.isin(common)]
            mature = day in sessions and sessions[day] + HORIZON + 1 < len(calendar)
            for scope, cohort in scopes.items():
                for support in GROUPS:
                    chosen = cohort if support == "ALL" else cohort.loc[cohort.support_group.eq(support)]
                    hkey = track, lane, scope, support
                    previous_day, previous = history.get(hkey, (None, {}))
                    if not record["formal"] or previous_day not in sessions or day not in sessions \
                            or sessions[day] - sessions[previous_day] != 1:
                        previous = {}
                    metrics = ranked_daily(chosen, cohort, record["portfolio_kind"] == "EQUAL_WEIGHT", previous)
                    if record["captured"] and record["formal"]:
                        history[hkey] = day, previous
                    else:
                        history.pop(hkey, None)
                    baseline = {"day": day, "track": track, "scope": scope, "support_group": support, "lane": lane,
                                "version": record["version"], "model_sha256": record["model_sha256"],
                                "status": record["status"], "captured": record["captured"], "formal": record["formal"],
                                "signal_session": sessions.get(day), "signal_matured": mature,
                                "entry_date": calendar[sessions[day] + 1] if mature else None,
                                "exit_date": calendar[sessions[day] + HORIZON + 1] if mature else None,
                                "candidates": record["candidates"] if support == "ALL" else None,
                                "abstentions": record["abstentions"] if support == "ALL" else None,
                                "abstention_reasons": record["abstention_reasons"] if support == "ALL" else {},
                                "forecast_excluded_by_common_scope": len(original) - len(cohort) if scope == "COMMON" else 0,
                                "prediction_coverage": len(chosen) / record["candidates"] if support == "ALL"
                                and record["candidates"] else None}
                    output.append(baseline | metrics)
    return output


def hac_mean(observations):
    points = {int(i): float(y) for i, y in observations}
    n = len(points)
    center = float(np.mean(list(points.values()))) if n else None
    result = {"n": n, "mean": center, "lag": HORIZON - 1, "standard_error": None, "approximate_95_percent_interval": None}
    if n < 2 * HORIZON:
        return result
    centered = {i: y - center for i, y in points.items()}
    cov = sum(y * y for y in centered.values())
    for shift in range(1, HORIZON):
        cov += 2 * (1 - shift / HORIZON) * sum(v * centered.get(i - shift, 0.0) for i, v in centered.items())
    se = float(np.sqrt(max(0.0, cov) / (n * (n - 1))))
    return result | {"standard_error": se, "approximate_95_percent_interval": [center - 1.96*se, center + 1.96*se]}


def disjoint_holds(rows):
    latest_exit = ""
    count = 0
    for row in sorted(rows, key=lambda r: (r["exit_date"], r["entry_date"])):
        if row["entry_date"] > latest_exit:
            latest_exit = row["exit_date"]
            count += 1
    return count


def summarize(days):
    groups = defaultdict(list)
    for row in days:
        groups[row["track"], row["lane"], row["scope"], row["support_group"], row["model_sha256"]].append(row)
    result = []
    for (track, lane, scope, support, sha), rows in groups.items():
        mature = [r for r in rows if r["signal_matured"] and r["formal"] and r["captured"]]
        aggregate = sum(r["predictions"] for r in mature)
        labels = sum(r["labels"] for r in mature)
        n_days = sum(r["labels"] > 0 for r in mature)
        coverage = labels / aggregate if aggregate else None
        ics = [(r["signal_session"], r["rank_ic"]) for r in mature if r["rank_ic"] is not None]
        top = [(r["signal_session"], r["top20_net_excess"]) for r in mature if r["top20_full"]]
        joint = [r for r in mature if r["rank_ic"] is not None and r["top20_full"]]
        replacements = [r["top20_replacement"] for r in rows if r["top20_replacement"] is not None]
        portfolio = [(r["signal_session"], r["portfolio_net_excess"]) for r in mature if r["portfolio_net_excess"] is not None]
        portfolio_replacements = [r["portfolio_replacement"] for r in rows if r["portfolio_replacement"] is not None]
        accepted = (track == "PROSPECTIVE" and len(joint) >= MIN_MATURED_DAYS
                    and (coverage or 0.0) >= MIN_LABEL_COVERAGE)
        state = ("BASELINE_NO_RANK_GATE" if lane == "equal_weight" else
                 "READY_FOR_REVIEW" if accepted else "INSUFFICIENT_PROSPECTIVE_DATA" if track == "PROSPECTIVE"
                 else "DEVELOPMENT_DIAGNOSTIC_ONLY")
        result.append({"track": track, "lane": lane, "scope": scope, "support_group": support, "model_sha256": sha,
                       "mature_label_sessions": n_days, "mature_predictions": aggregate, "mature_labels": labels,
                       "pending_predictions": sum(r["pending"] for r in rows), "missing_mature_labels": aggregate - labels,
                       "label_coverage": coverage, "rank_ic_days": len(ics), "full_top20_days": len(top),
                       "jointly_valid_days": len(joint), "disjoint_joint_hold_windows": disjoint_holds(joint),
                       "rank_ic": hac_mean(ics), "top20_net_excess": hac_mean(top),
                       "top20_replacement": float(np.mean(replacements)) if replacements else None,
                       "portfolio_net_excess": hac_mean(portfolio),
                       "portfolio_replacement": float(np.mean(portfolio_replacements)) if portfolio_replacements else None,
                       "status": state,
                       "missing_formal_days": [r["day"] for r in rows if not r["captured"]],
                       "zero_prediction_days": [r["day"] for r in rows if r["captured"] and r["formal"]
                                                and not r["predictions"]]})
    return result


REPORT_AUDIT = "panel_report_audit.json"
DAILY_REPORT = "daily.md"
CUMULATIVE_REPORT = "cumulative.csv"
LABEL_REPORT = "prediction_labels.parquet"
TABLE_KEYS = ("day", "track", "scope", "support_group", "lane", "version", "model_sha256",
              "status", "captured", "formal", "signal_matured", "entry_date", "exit_date",
              "candidates", "predictions", "prediction_coverage", "abstentions", "abstention_reasons",
              "forecast_excluded_by_common_scope", "labels", "pending", "label_coverage", "label_reasons",
              "rank_ic", "equal_weight_pool_return", "portfolio_return", "portfolio_net_excess",
              "portfolio_replacement", "top10_stocks", "top10_labels", "top10_full", "top10_label_coverage",
              "top10_return", "top10_net_excess", "top10_replacement", "top20_stocks", "top20_labels",
              "top20_full", "top20_label_coverage", "top20_return", "top20_net_excess", "top20_replacement")


def tally_cell(mapping):
    return "、".join(f"{key}（{value}）" for key, value in sorted(mapping.items())) or "—"


def counts(value):
    return "—" if value is None else str(value)


def daily_markdown(days, summary, calendar, focus=None):
    focus = focus or max((r["day"] for r in days if r["captured"]), default=None)
    final_day = calendar[-1] if calendar else "无正式行情"
    lines = ["# P5.5 每日预测与成熟评价", "",
             f"本页记录 {focus or '暂无预测记录'}，正式行情截至 {final_day}。", "",
             "标签使用 T+1 至 T+6 开盘的研究经济收益。原始名单先由冻结评分确定，后续缺标签保留原位置。"
             "Top-10 与 Top-20 净超额按同一评估池的毛收益计算，每侧扣 10 bp。", ""]
    groups = sorted({r["track"] for r in days if r["day"] == focus})
    for track in groups:
        rows = [r for r in days if r["day"] == focus and r["scope"] == "OWN"
                and r["track"] == track and r["support_group"] == "ALL"]
        lines.extend([f"## {track}", "",
                      "| 模型 | 候选 | 预测 | 覆盖 | 弃权 | 标签 | 待成熟 | 标签覆盖 | IC | Top-20 净超额 | 名单更替 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for lane in LANES:
            for r in [entry for entry in rows if entry["lane"] == lane]:
                lines.append(f"| {NAMES[lane]} | {counts(r['candidates'])} | {r['predictions']} | "
                             f"{percent(r['prediction_coverage'])} | {counts(r['abstentions'])} | {r['labels']} | "
                             f"{r['pending']} | {percent(r['label_coverage'])} | {amount(r['rank_ic'])} | "
                             f"{percent(r['top20_net_excess'])} | {percent(r['top20_replacement'])} |")
        lines.append("")
        for r in rows:
            lines.append(f"- **{NAMES[r['lane']]}**。版本 {r['version']}。{r['status']}。"
                         f"弃权 {tally_cell(r['abstention_reasons'])}。"
                         f"标签状态 {tally_cell(r['label_reasons'])}。")
            if r["top20_stocks"]:
                for k in (10, 20):
                    lines.append(f"  - Top-{k} " + "、".join(r[f"top{k}_stocks"]))
            elif r["lane"] == "equal_weight" and r["predictions"]:
                lines.append("  - 对全部入池股票等权，完整名单保存在原冻结账本。")
        lines.append("")
        common = [r for r in days if r["day"] == focus and r["scope"] == "COMMON"
                  and r["track"] == track and r["support_group"] == "ALL"]
        if common:
            lines.extend(["共有 " + str(common[0]["predictions"]) + " 只股票同时具备六条轨道的冻结预测，"
                          f"其中 {common[0]['labels']} 只已有有效标签。共有池按冻结预测名单求交集。", ""])
    lines.extend(["## 前瞻累计", "",
                  "| 模型 | 范围 | 训练支持组 | 预测 | 已计算标签 | 标签覆盖 | 平均 IC (有效日) | Top-20 净超额 (完整日) | 不重叠区间 | 状态 |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---|"])
    for s in summary:
        if s["track"] != "PROSPECTIVE":
            continue
        lines.append(f"| {NAMES[s['lane']]} | {s['scope']} | {s['support_group']} | "
                     f"{s['mature_predictions']} | {s['mature_labels']} | {percent(s['label_coverage'])} | "
                     f"{amount(s['rank_ic']['mean'])} ({s['rank_ic_days']}) | "
                     f"{percent(s['top20_net_excess']['mean'])} ({s['full_top20_days']}) | "
                     f"{s['disjoint_joint_hold_windows']} | {s['status']} |")
    gap = sorted({day for s in summary if s["track"] == "PROSPECTIVE" for day in s["missing_formal_days"]})
    zero = sorted({day for s in summary if s["track"] == "PROSPECTIVE" and s["scope"] == "OWN"
                   and s["support_group"] == "ALL" for day in s["zero_prediction_days"]})
    lines.extend(["", "前瞻缺失交易日 " + ("、".join(gap) or "无") + "。",
                  "至少一个模型记录零预测的前瞻日期 " + ("、".join(zero) or "无") + "。", "",
                  "进入初步人工评审需要 20 个同时具备 IC 与完整 Top-20 的前瞻成熟信号日，"
                  "并且成熟预测标签覆盖不低于 80%。报告按实际正式交易日间距计算 4 阶 HAC 近似区间，"
                  "另列互不重叠的持有区间数量。五日持有窗可以重叠。等权基准无排名验收。", ""])
    for s in summary:
        if s["track"] != "PROSPECTIVE" or s["support_group"] != "ALL":
            continue
        if s["lane"] == "equal_weight":
            lines.append(f"- 等权股票池 {s['scope']} 的平均组合净超额为 {percent(s['portfolio_net_excess']['mean'])}。"
                         f"名单更替 {percent(s['portfolio_replacement'])}。")
        intervals = []
        for name, field in (("IC", "rank_ic"), ("Top-20 净超额", "top20_net_excess")):
            interval = s[field]["approximate_95_percent_interval"]
            if interval is not None:
                intervals.append(f"{name} [{amount(interval[0])}, {amount(interval[1])}]")
        if intervals:
            lines.append(f"- {NAMES[s['lane']]} {s['scope']} 的 95% HAC 近似区间为 " + "，".join(intervals) + "。")
    return "\n".join(lines).rstrip() + "\n"


def csv_row(day):
    row = {key: day[key] for key in TABLE_KEYS}
    for key in ("abstention_reasons", "top10_stocks", "top20_stocks", "label_reasons"):
        row[key] = json.dumps(row[key], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return row


def run(ledger, output, seed, evidence_source=None, cutoff=None, official_path=OFFICIAL, legacy_ledger=None,
        focus=None):
    records, forecasts, input_manifests, policy_digest = read_panel(ledger, legacy_ledger)
    now = datetime.now(CHINA)
    if cutoff is not None and not isinstance(cutoff, date):
        cutoff = date.fromisoformat(cutoff)
    if cutoff is not None:
        end = cutoff.isoformat()
        records = [r for r in records if r["day"] <= end]
        forecasts = forecasts.loc[forecasts.trade_date.le(end)]
    start = min([now.date(), *(date.fromisoformat(r["day"]) for r in records),
                 *(date.fromisoformat(r["freeze_day"]) + timedelta(days=1) for r in records)])
    calendar, raw = formal_tail(AShareDailyLoader().engine, start, sorted(forecasts.stock_code.unique()), now, cutoff)
    evidence = select_evidence(legacy_ledger or ledger, seed, evidence_source)
    official = {(r.stock_code, r.ex_date.date().isoformat()):
                (float(r.share_multiplier), float(r.cash_per_old_share), r.source_id)
                for r in load_actions(official_path).itertuples(index=False)}
    labelled = with_labels(forecasts, calendar, raw, evidence, official)
    days = evaluate_days(records, labelled, calendar)
    summary = summarize(days)
    output.mkdir(parents=True)
    parquet_write(labelled, output / LABEL_REPORT)
    pd.DataFrame([csv_row(r) for r in days], columns=TABLE_KEYS).to_csv(output / CUMULATIVE_REPORT, index=False)
    (output / DAILY_REPORT).write_text(daily_markdown(days, summary, calendar, focus), encoding="utf-8")
    audit = {"version": VERSION, "evaluated_at": now.isoformat(), "research_only": True,
             "formal_through": calendar[-1] if calendar else None, "panel_policy_sha256": policy_digest,
             "input_manifests": input_manifests, "company_evidence_sha256": evidence["sha256"],
             "issuer_actions_sha256": hash_file(official_path), "label_rows": len(labelled),
             "label_statuses": dict(Counter(labelled.maturity_status)), "summary": summary,
             "forward_rules": {"minimum_joint_prospective_sessions": MIN_MATURED_DAYS,
                               "minimum_mature_label_coverage": MIN_LABEL_COVERAGE,
                               "research_cost_bps_per_side": RESEARCH_COST_BPS_PER_SIDE,
                               "complete_top20_requires_all_selected_labels": True,
                               "common_scope": "all six matching-track forecast lists intersected before labels",
                               "common_support_buckets": "lane-specific training support; ALL is the paired comparison",
                               "sufficient_data_status": "READY_FOR_REVIEW",
                               "auto_prediction_ability_pass": False, "hac_lag_sessions": HORIZON - 1},
             "files": {name: hash_file(output / name) for name in (DAILY_REPORT, CUMULATIVE_REPORT, LABEL_REPORT)}}
    atomic_json(output / REPORT_AUDIT, audit)
    return audit


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--ledger", type=Path, required=True)
    cli.add_argument("--legacy-ledger", type=Path)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--evidence-seed", type=Path)
    cli.add_argument("--evidence-source", type=Path)
    cli.add_argument("--cutoff", type=date.fromisoformat)
    cli.add_argument("--official-actions", type=Path, default=OFFICIAL)
    cli.add_argument("--focus")
    a = cli.parse_args()
    result = run(a.ledger, a.out, a.evidence_seed, a.evidence_source, a.cutoff, a.official_actions, a.legacy_ledger, a.focus)
    print({"formal_through": result["formal_through"], "label_rows": result["label_rows"],
           "label_statuses": result["label_statuses"]}, flush=True)


if __name__ == "__main__":
    main()
