"""P4.3-D: read-only source-price tracing for previously flagged large labels.

Trace T+1 to T+1+H opens from public.market and verify the exported
unadjusted label. A price jump is NOT proof of a corporate action.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

from .daily import AShareDailyLoader

WARNING = (
    "SOURCE PRICE RECONCILIATION ONLY. No adjusted prices, corporate-action "
    "table, historical ST certification, exchange fill verification or trading "
    "PnL. A large open-price jump MAY be genuine market movement, adjustment "
    "or bad upstream data; review an independently dated exchange/company "
    "announcement before classifying it. NO old-test labels are processed."
)


def select_cases(outliers: pd.DataFrame, *, test_start: str, hold_sessions: int = 5,
                 limit_per_split: int = 30) -> pd.DataFrame:
    """Fail closed on test-period rows; select largest magnitudes per old split."""
    if hold_sessions < 1 or limit_per_split < 1:
        raise ValueError("Invalid trace horizon or case limit")
    needed = {"split", "stock_code", "trade_date", "session_id",
              "label_end_session", "target_return"}
    if needed - set(outliers):
        raise ValueError(f"Missing outlier fields: {sorted(needed-set(outliers))}")
    if outliers.empty:
        raise ValueError("No outlier candidates")
    d = outliers.copy()
    if not d["split"].isin(("train", "validation")).all():
        raise ValueError("Only train and validation outliers are allowed")
    d["trade_date"] = pd.to_datetime(d.trade_date, errors="raise").dt.normalize()
    if d.trade_date.isna().any() or d.trade_date.ge(pd.Timestamp(test_start)).any():
        raise ValueError("Old inspected test period MUST be excluded")
    if d.stock_code.isna().any() or not d.stock_code.astype("string").str.fullmatch(r"\d{6}").all():
        raise ValueError("Invalid outlier stock codes")
    if d.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Repeated outlier stock-date")
    for col in ("session_id", "label_end_session", "target_return"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
        if not np.isfinite(d[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {col} in outliers")
    if not (d.label_end_session - d.session_id).eq(hold_sessions+1).all():
        raise ValueError("Unexpected label-end session gap")
    d["absolute_label"] = d.target_return.abs()
    d = d.sort_values(["split", "absolute_label", "trade_date", "stock_code"],
                      ascending=[True, False, True, True], kind="stable")
    return d.groupby("split", sort=True).head(limit_per_split).reset_index(drop=True)


def trace_price_paths(cases: pd.DataFrame, raw: pd.DataFrame,
                      calendar: pd.DatetimeIndex, *, hold_sessions: int = 5,
                      jump_threshold: float = 0.15, compare_tolerance: float = 1e-7,
                      test_start: str | None = None
                      ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Independently recompute the exported open-to-open research label."""
    if hold_sessions < 1 or jump_threshold <= 0 or compare_tolerance <= 0:
        raise ValueError("Invalid price audit settings")
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Invalid formal-market calendar")
    required = {"stock_code", "trade_date", "open", "high", "low", "close"}
    if required - set(raw):
        raise ValueError("Missing OHLC source fields")
    bars = raw.copy()
    bars["stock_code"] = bars.stock_code.astype("string").str.strip()
    bars["trade_date"] = pd.to_datetime(bars.trade_date, errors="raise").dt.normalize()
    if bars.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate formal OHLC bar, cannot reconcile safely")
    for field in ("open", "high", "low", "close"):
        bars[field] = pd.to_numeric(bars[field], errors="coerce")
    # Indexed only by the requested stock codes; no broad market row scan here.
    lookups = bars.set_index(["stock_code", "trade_date"])
    cal = pd.DatetimeIndex(calendar).normalize()
    cal_position = {day: i for i, day in enumerate(cal)}
    items, daily = [], []
    for row in cases.itertuples(index=False):
        signal_date = pd.Timestamp(row.trade_date).normalize()
        if signal_date not in cal_position:
            raise ValueError(f"Signal date is missing in market calendar: {signal_date}")
        pos = cal_position[signal_date]
        # Use the global market calendar, not just the requested stock's dates:
        # an unquoted session is a gap, NEVER silently jump to its next bar.
        if pos + hold_sessions + 1 >= len(cal):
            items.append({
                "split": row.split, "stock_code": row.stock_code,
                "trade_date": signal_date, "target_return": float(row.target_return),
                "status": "INSUFFICIENT_MARKET_CALENDAR",
                "complete_price_path": False,
            })
            continue
        session_dates = cal[pos+1:pos+hold_sessions+2]
        if test_start is not None and session_dates[-1] >= pd.Timestamp(test_start):
            raise ValueError("Price trace label path would cross excluded test boundary")
        open_prices, invalid_bars = [], False
        for offset, day in enumerate(session_dates, start=1):
            key = (row.stock_code, day)
            if key not in lookups.index:
                item = {"split": row.split, "stock_code": row.stock_code,
                        "trade_date": signal_date, "offset_after_signal": offset,
                        "market_session_date": day, "open": np.nan,
                        "close": np.nan, "valid_bar": False}
                open_prices.append(np.nan)
            else:
                entry = lookups.loc[key]
                op = float(entry.open)
                high, low, close = map(float, (entry.high, entry.low, entry.close))
                ok = (np.isfinite([op, high, low, close]).all()
                      and op > 0 and low > 0 and low <= op <= high
                      and low <= close <= high)
                invalid_bars |= not ok
                item = {"split": row.split, "stock_code": row.stock_code,
                        "trade_date": signal_date, "offset_after_signal": offset,
                        "market_session_date": day, "open": op,
                        "close": close, "valid_bar": bool(ok)}
                open_prices.append(op)
            daily.append(item)
        opens = np.asarray(open_prices, dtype=float)
        complete = bool(np.isfinite(opens).all() and (opens > 0).all()
                        and not invalid_bars)
        estimate = float(opens[-1] / opens[0] - 1) if complete else np.nan
        error = abs(estimate - float(row.target_return)) if complete else np.nan
        jumps = opens[1:] / opens[:-1] - 1 if complete else np.array([])
        maxjump = float(np.max(np.abs(jumps))) if len(jumps) else np.nan
        jump_day = (session_dates[int(np.argmax(np.abs(jumps)))+1]
                    if len(jumps) else pd.NaT)
        status = ("MISSING_OR_INVALID_BAR" if not complete
                  else ("LABEL_MISMATCH" if error > compare_tolerance else "RECOMPUTED_OK"))
        items.append({
            "split": row.split, "stock_code": row.stock_code,
            "trade_date": signal_date, "target_return": float(row.target_return),
            "entry_trade_date": session_dates[0], "exit_trade_date": session_dates[-1],
            "entry_open": float(opens[0]) if np.isfinite(opens[0]) else np.nan,
            "exit_open": float(opens[-1]) if np.isfinite(opens[-1]) else np.nan,
            "recomputed_label": estimate, "absolute_label_difference": error,
            "maximum_absolute_consecutive_open_jump": maxjump,
            "largest_jump_on": jump_day,
            "jump_exceeds_review_threshold": bool(maxjump > jump_threshold) if complete else False,
            "complete_price_path": complete, "status": status,
        })
    overview = pd.DataFrame(items)
    summary = {
        "trace_version": "P4_UNADJUSTED_OPEN_PATH_TRACE_V1",
        "requested_cases": int(len(cases)),
        "complete_price_paths": int(overview.complete_price_path.sum()),
        "recomputed_equal_within_tolerance": int(overview.status.eq("RECOMPUTED_OK").sum()),
        "missing_or_invalid_paths": int(overview.status.eq("MISSING_OR_INVALID_BAR").sum()),
        "calendar_shortfall": int(overview.status.eq("INSUFFICIENT_MARKET_CALENDAR").sum()),
        "label_mismatch_paths": int(overview.status.eq("LABEL_MISMATCH").sum()),
        "large_consecutive_open_jumps_for_review": int(
            overview.get("jump_exceeds_review_threshold", pd.Series(dtype=bool)).sum()
        ),
        "jump_threshold": float(jump_threshold),
        "compare_tolerance": float(compare_tolerance),
        "warning": WARNING,
    }
    return summary, overview, pd.DataFrame(daily)


def load_formal_source_cases(cases: pd.DataFrame, *,
                             hold_sessions: int = 5, engine=None,
                             test_start: str | None = None
                             ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Two SELECTs only: market dates and requested stock OHLC, no writes."""
    db = engine if engine is not None else AShareDailyLoader().engine
    begin = pd.Timestamp(cases.trade_date.min()).date()
    finish = (pd.Timestamp(cases.trade_date.max()) +
              timedelta(days=max(40, (hold_sessions+1) * 6))).date()
    if test_start is not None:
        finish = min(finish, (pd.Timestamp(test_start) - timedelta(days=1)).date())
    if finish < begin:
        raise ValueError("No pre-test market dates remain to trace")
    params = {"begin": begin, "finish": finish}
    calendar_sql = text("""
        SELECT DISTINCT trade_date FROM public.market
         WHERE not_billing=1 AND trade_date BETWEEN :begin AND :finish
         ORDER BY trade_date
    """)
    prices_sql = text("""
        SELECT stock_code, trade_date, open, high, low, close
          FROM public.market
         WHERE not_billing=1
           AND trade_date BETWEEN :begin AND :finish
           AND stock_code IN :codes
    """).bindparams(bindparam("codes", expanding=True))
    codes = sorted(set(cases.stock_code.astype(str)))
    if not codes:
        raise ValueError("No stock codes to audit")
    with db.connect() as conn:
        cal_rows = pd.read_sql_query(calendar_sql, conn, params=params)
        bars = pd.read_sql_query(prices_sql, conn, params={**params, "codes": codes})
    cal = pd.DatetimeIndex(pd.to_datetime(cal_rows.trade_date)).normalize()
    if cal.empty:
        raise ValueError("No formal market calendar in audited window")
    return bars, cal


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only price tracing for P4 train/validation outliers")
    p.add_argument("--outliers", type=Path, required=True)
    p.add_argument("--reference-metadata", type=Path, required=True)
    p.add_argument("--limit-per-split", type=int, default=30)
    p.add_argument("--jump-threshold", type=float, default=0.15)
    p.add_argument("--out", type=Path, default=Path("reports/ashare_p4_price_trace"))
    args = p.parse_args()
    with args.reference_metadata.open(encoding="utf-8") as fp:
        meta = json.load(fp)
    if meta.get("model_version") != "P4_RIDGE_V1":
        raise ValueError("Unexpected reference metadata")
    hold = int(meta["pool_config"]["hold_sessions"])
    cases = select_cases(pd.read_csv(args.outliers, encoding="utf-8-sig",
                                     dtype={"stock_code": "string"}),
                         test_start=meta["test_start"], hold_sessions=hold,
                         limit_per_split=args.limit_per_split)
    def log(msg: str):
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
    log(f"[1/3] Read-only fetch market calendar and OHLC for {len(cases)} cases")
    raw, calendar = load_formal_source_cases(
        cases, hold_sessions=hold, test_start=meta["test_start"]
    )
    log("[2/3] Recompute T+1 to T+6 opens and flag missing bars and large jumps")
    summary, overview, daily = trace_price_paths(
        cases, raw, calendar, hold_sessions=hold,
        jump_threshold=args.jump_threshold, test_start=meta["test_start"]
    )
    summary["reference_test_start_EXCLUDED"] = meta["test_start"]
    summary["outliers_input"] = str(args.outliers)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "price_trace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8"
    )
    overview.to_csv(args.out / "price_trace_cases.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(args.out / "price_trace_bars.csv", index=False, encoding="utf-8-sig")
    log(f"[3/3] Saved source trace: {args.out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
