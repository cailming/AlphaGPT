"""Publish one self-contained, reconcilable, research-only portfolio day."""
from collections import Counter

import pandas as pd

from .p4_action_pipeline import atomic_json
from .p4_subset_inputs import hash_file


def publish_day(stage, day, track, timing, results, policy_sha, previous_sha, now):
    names = {"equity":"daily_equity","positions":"positions","trades":"trades","rejections":"rejections"}
    extra = {"equity":["nav","equity","valuation_status"],"positions":["stock_code","shares","indicative_value"],
             "trades":["stock_code","side","shares","price","fees"],"rejections":["stock_code","side","reason"]}
    for kind,suffix in names.items():
        base = ["model","strategy","trade_date",*extra[kind]]
        df = pd.DataFrame(results[kind])
        df.reindex(columns=[*base,*[k for k in df if k not in base]]).to_csv(
            stage/f"portfolio_{suffix}.csv",index=False)
    accounts = results["equity"]
    flags = dict(Counter(r["valuation_status"] for r in accounts))
    rejected = dict(Counter(r["reason"] for r in results["rejections"]))
    if len({(r["model"],r["strategy"]) for r in accounts}) != len(accounts):
        raise ValueError("Duplicate account day")
    lines = [f"# 模拟组合日报 · {day}", "", f"归档类别 {track} / {timing}。正式开盘价为模拟成交假设。",
             "", "| 模型 | 策略 | 期初现金 | 期末现金 | 持仓数 | 净值 | 模拟买卖 | 拒单 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in accounts:
        nav = f"{r['nav']:.6f}" if r["nav"] is not None else "估值待核实"
        lines.append(f"| {r['model']} | {r['strategy']} | {r['opening_cash']:.2f} | {r['closing_cash']:.2f}"
                     f" | {r['positions']} | {nav} | {r['trade_count']} | {r['rejections']} |")
    lines += ["",f"估值状态 {flags}。拒单和延迟原因 {rejected}。",
              "同日完整账户指标见 portfolio_daily_equity.csv，含累计费用、累计滑点、毛收益、费后收益、基准、换手和回撤。",
              "当公司行为条款或持仓价格缺少证据时，indicative_nav 仅供核账，正式收益留空。"]
    (stage/"portfolio_daily.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    atomic_json(stage/"portfolio_summary.json",{
        "day":day,"track":track,"timing":timing,"observed_at":now.isoformat(),
        "policy_sha256":policy_sha,"previous_sha256":previous_sha,"accounts":accounts,"research_only":True,
        "trade_count":len(results["trades"]),"rejection_reasons":rejected,"valuation_statuses":flags,
        "files":{p.name:hash_file(p) for p in sorted(stage.iterdir()) if p.is_file()}})
