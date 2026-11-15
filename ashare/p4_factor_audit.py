"""P4.3-C: train/validation-only factor and label audit (read-only research).

The already-inspected 2026 test labels are NEVER used in audit statistics.
Reads the same formal OHLCV as P4 MVP and reconstructs the dynamic pool;
P4.1 pool_selected.csv alone lacks features and label_end_session.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "ret_1", "ret_5", "ret_10", "close_ma20_dev",
    "volume_ratio_20", "amount_ratio_20", "range_intraday",
    "return_intraday", "volatility_20", "log_liquidity_20",
)
WARNING = (
    "TRAIN/VALIDATION FACTOR AND LABEL AUDIT ONLY. Previously inspected test "
    "labels are excluded from every audit table. Five-session forward labels "
    "overlap, and unadjusted OHLC prices may contain corporate-action artifacts. "
    "Extreme labels are candidates for manual investigation, NOT confirmed "
    "corporate actions, executable return, or portfolio PnL. Future independent "
    "holdout or predeclared walk-forward is required for new models."
)


def _validate_pool(data: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    required = {"stock_code", "trade_date", "session_id", "label_end_session",
                "target_return", *FEATURE_NAMES}
    if required - set(data):
        raise ValueError(f"Missing required pool columns: {sorted(required - set(data))}")
    if data.empty:
        raise ValueError("Empty dynamic pool")
    d = data.copy()
    d["trade_date"] = pd.to_datetime(d.trade_date, errors="raise").dt.normalize()
    if d.trade_date.isna().any() or d.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Missing dates or duplicate stock/date")
    if d.stock_code.isna().any() or not d.stock_code.astype("string").str.fullmatch(r"\d{6}").all():
        raise ValueError("Invalid stock codes")
    if not isinstance(calendar, pd.DatetimeIndex) or not calendar.is_monotonic_increasing or calendar.has_duplicates:
        raise ValueError("Invalid market calendar")
    sid = pd.Series(np.arange(len(calendar)), index=calendar)
    mapped = d.trade_date.map(sid)
    if mapped.isna().any() or not np.array_equal(
        mapped.to_numpy(dtype=int), pd.to_numeric(d.session_id).to_numpy(dtype=int)
    ):
        raise ValueError("Pool signal session_id does not match market calendar")
    label = pd.to_numeric(d.target_return, errors="coerce")
    end = pd.to_numeric(d.label_end_session, errors="coerce")
    if (label.notna() != end.notna()).any():
        raise ValueError("Missing label and end-session must agree")
    if not np.isfinite(label.dropna().to_numpy(dtype=float)).all():
        raise ValueError("Invalid label")
    valid_end = end.dropna().to_numpy(dtype=float)
    if (not np.isfinite(valid_end).all() or
            np.any(valid_end != np.floor(valid_end)) or
            np.any(valid_end >= len(calendar)) or
            np.any(valid_end <= 0)):
        raise ValueError("Invalid label end-session index")
    if not np.isfinite(d[list(FEATURE_NAMES)].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite research features")
    return d


def split_audit_pool(data: pd.DataFrame, calendar: pd.DatetimeIndex, *,
                     train_fraction: float = 0.6, val_fraction: float = 0.2
                     ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Exactly mirror fit_splits' date cuts and label-end embargo.

    Do not read test TARGET values, including when calculating diagnostics.
    """
    if (train_fraction <= 0 or val_fraction <= 0 or
            train_fraction + val_fraction >= 0.9):
        raise ValueError("Invalid chronological fractions")
    # Establish the fixed date split using signal dates alone, BEFORE
    # touching label columns. Exclude all test rows from schema/label checks
    # and downstream analysis, even if they contain corrupted future labels.
    signal_dates = pd.to_datetime(data.trade_date, errors="raise").dt.normalize()
    if signal_dates.isna().any():
        raise ValueError("Missing signal dates")
    dates = pd.DatetimeIndex(sorted(signal_dates.unique()))
    if len(dates) < 80:
        raise ValueError("Need at least 80 pool signal dates")
    train_cut = int(len(dates) * train_fraction)
    test_cut = int(len(dates) * (train_fraction + val_fraction))
    val_start, test_start = dates[train_cut], dates[test_cut]
    test_rows = int(signal_dates.ge(test_start).sum())
    d = _validate_pool(data.loc[signal_dates.lt(test_start)].copy(), calendar)
    sid = pd.Series(np.arange(len(calendar)), index=calendar)
    vsid, tsid = int(sid[val_start]), int(sid[test_start])
    labelled = d.target_return.notna()
    train_signal = d.trade_date.lt(val_start)
    val_signal = d.trade_date.ge(val_start) & d.trade_date.lt(test_start)
    train = d.loc[train_signal & labelled & d.label_end_session.lt(vsid)].copy()
    val = d.loc[val_signal & labelled & d.label_end_session.lt(tsid)].copy()
    refit_eligible = d.trade_date.lt(test_start) & labelled & d.label_end_session.lt(tsid)
    if len(train) < 100 or len(val) < 40:
        raise ValueError("Insufficient purged train/validation samples")
    assert train.label_end_session.max() < vsid
    assert val.label_end_session.max() < tsid
    summary = {
        "train_start": str(train.trade_date.min().date()),
        "validation_start": str(val_start.date()),
        "test_start_EXCLUDED": str(test_start.date()),
        "test_end_EXCLUDED": str(dates[-1].date()),
        "train_last_label_date": str(calendar[int(train.label_end_session.max())].date()),
        "validation_last_label_date": str(calendar[int(val.label_end_session.max())].date()),
        "selected_signal_days": len(dates),
        "train_signal_rows_pre_label_checks": int(train_signal.sum()),
        "validation_signal_rows_pre_label_checks": int(val_signal.sum()),
        "train_rows": len(train), "validation_rows": len(val),
        "train_purged_or_missing_label_rows": int(train_signal.sum() - len(train)),
        "validation_purged_or_missing_label_rows": int(val_signal.sum() - len(val)),
        "eligible_refit_pretest_samples": int(refit_eligible.sum()),
        "test_rows_EXCLUDED_without_reading_labels": test_rows,
        "purge_rule": "label_end_session < next_split_first_session",
    }
    return summary, train, val


def factor_diagnostics(train: pd.DataFrame, val: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily cross-sectional Spearman IC; validation direction comes from train."""
    rows = []
    for split, frame in (("train", train), ("validation", val)):
        for day, group in frame.groupby("trade_date", sort=True):
            y = group.target_return.rank()
            for factor in FEATURE_NAMES:
                x = group[factor].rank()
                ic = float(x.corr(y)) if len(group) >= 3 else float("nan")
                rows.append({
                    "split": split, "trade_date": day, "feature": factor,
                    "observations": len(group),
                    "rank_ic": ic if np.isfinite(ic) else np.nan,
                })
    daily = pd.DataFrame(rows)
    aggregates = daily.groupby(["split", "feature"], sort=True).agg(
        signal_days=("trade_date", "size"),
        valid_ic_days=("rank_ic", "count"),
        mean_ic=("rank_ic", "mean"),
        median_ic=("rank_ic", "median"),
        ic_std=("rank_ic", "std"),
        share_positive_days=("rank_ic", lambda s: s.dropna().gt(0).mean()),
    ).reset_index()
    train_means = aggregates.loc[aggregates.split.eq("train"), ["feature", "mean_ic"]]
    signs = train_means.set_index("feature").mean_ic.apply(
        lambda x: 1 if pd.notna(x) and x > 0 else (-1 if pd.notna(x) and x < 0 else 0)
    )
    aggregates["train_defined_direction"] = aggregates.feature.map(signs).astype(int)
    aggregates["signed_mean_ic_train_direction"] = (
        aggregates.mean_ic * aggregates.train_defined_direction
    )
    return daily, aggregates


def label_diagnostics(train: pd.DataFrame, val: pd.DataFrame, *,
                      extreme_threshold: float = 0.20,
                      outlier_limit: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect large unadjusted label changes, NOT prove corporate actions."""
    if not 0 < extreme_threshold < 1 or outlier_limit < 0:
        raise ValueError("Invalid label audit thresholds")
    stats = []
    outliers = []
    for split, frame in (("train", train), ("validation", val)):
        labels = frame.target_return.to_numpy(dtype=float)
        positive = labels > 0
        item = {
            "split": split, "sample_rows": len(frame),
            "signal_days": int(frame.trade_date.nunique()),
            "unique_stock_codes": int(frame.stock_code.nunique()),
            "mean_label": float(np.mean(labels)),
            "median_label": float(np.median(labels)),
            "std_label": float(np.std(labels)),
            "positive_share": float(positive.mean()),
            "abs_gt_10pct_rows": int((np.abs(labels) > .10).sum()),
            "abs_gt_20pct_rows": int((np.abs(labels) > .20).sum()),
            "abs_gt_threshold_rows": int((np.abs(labels) > extreme_threshold).sum()),
        }
        for name, pct in (("p01", .01), ("p05", .05), ("p25", .25),
                          ("p50", .50), ("p75", .75), ("p95", .95), ("p99", .99)):
            item[name] = float(np.quantile(labels, pct))
        stats.append(item)
        if outlier_limit:
            candidates = frame.loc[frame.target_return.abs().gt(extreme_threshold)]
            columns = ["trade_date", "stock_code", "session_id", "label_end_session",
                       "target_return", "ret_1", "return_intraday", "close_ma20_dev"]
            top = candidates.assign(
                absolute_target=candidates.target_return.abs()
            ).sort_values(
                ["absolute_target", "trade_date", "stock_code"],
                ascending=[False, True, True], kind="stable"
            ).head(outlier_limit)
            outliers.append(top.loc[:, columns + ["absolute_target"]].assign(split=split))
    return pd.DataFrame(stats), (
        pd.concat(outliers, ignore_index=True) if outliers else
        pd.DataFrame(columns=["split", "trade_date", "stock_code", "target_return"])
    )


def audit_pool(data: pd.DataFrame, calendar: pd.DatetimeIndex, *,
               train_fraction: float = 0.6, val_fraction: float = 0.2,
               extreme_threshold: float = 0.20, outlier_limit: int = 200
               ) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split, train, val = split_audit_pool(
        data, calendar, train_fraction=train_fraction, val_fraction=val_fraction
    )
    ic_daily, ic_summary = factor_diagnostics(train, val)
    labels, outliers = label_diagnostics(train, val,
                                        extreme_threshold=extreme_threshold,
                                        outlier_limit=outlier_limit)
    report = {
        "audit_version": "P4_TRAIN_VAL_FACTOR_AUDIT_V1",
        "split": split, "factor_names": list(FEATURE_NAMES),
        "extreme_label_threshold": float(extreme_threshold),
        "outlier_rows_saved": len(outliers),
        "test_labels_used_in_audit": False,
        "warning": WARNING,
    }
    return report, ic_daily, ic_summary, labels, outliers


def main() -> None:
    from .p4_mvp import FEATURES, PoolConfig, build_pool, load_formal_ohlcv, RESEARCH_WARNING
    from .risk_events import load_risk_events
    if tuple(FEATURES) != FEATURE_NAMES:
        raise RuntimeError("P4 feature contract changed; update factor audit first")
    p = argparse.ArgumentParser(description="P4.3-C read-only train/validation factor audit")
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 3, 1))
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--top-n", type=int, default=800)
    p.add_argument("--min-price", type=float, default=3.0)
    p.add_argument("--min-amount", type=float, default=0)
    p.add_argument("--consecutive", type=int, default=40)
    p.add_argument("--hold-sessions", type=int, default=5)
    p.add_argument("--train-fraction", type=float, default=0.6)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--extreme-label", type=float, default=0.2)
    p.add_argument("--outlier-limit", type=int, default=200)
    p.add_argument("--db-chunk-size", type=int, default=50000)
    p.add_argument("--risk-events", type=Path,
                   default=Path("ashare/data/risk_events_verified_examples.csv"))
    p.add_argument("--no-risk-events", action="store_true")
    p.add_argument("--reference-metadata", type=Path, default=None,
                   help="optional old Ridge model_metadata.json; fail on split/config mismatch")
    p.add_argument("--out", type=Path, default=Path("reports/ashare_p4_factor_audit"))
    args = p.parse_args()
    if args.end < args.start or args.db_chunk_size < 1:
        p.error("Invalid date range or DB chunk size")
    cfg = PoolConfig(top_n=args.top_n, min_price=args.min_price,
                     min_amount=args.min_amount, consecutive_sessions=args.consecutive,
                     hold_sessions=args.hold_sessions)
    def log(msg):
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
    log("[1/4] SELECT formal OHLCV only; no production DB writes")
    raw = load_formal_ohlcv(args.start, args.end,
                            hold_sessions=cfg.hold_sessions,
                            chunksize=args.db_chunk_size, progress=log)
    if raw.empty:
        raise SystemExit("No formal OHLCV rows")
    events = None
    if not args.no_risk_events:
        if not args.risk_events.is_file():
            p.error(f"Risk event file missing: {args.risk_events}")
        events = load_risk_events(args.risk_events)
    log("[2/4] Rebuild past-only pool + forward labels; isolate train and validation")
    pool = build_pool(raw, args.start, args.end, cfg, events=events)
    report, daily, factors, labels, outliers = audit_pool(
        pool.data, pool.calendar, train_fraction=args.train_fraction,
        val_fraction=args.val_fraction, extreme_threshold=args.extreme_label,
        outlier_limit=args.outlier_limit
    )
    report["dataset_start"] = args.start.isoformat()
    report["dataset_end"] = args.end.isoformat()
    report["pool_config"] = pool.stats["pool_config"]
    report["pool_stats"] = {
        "selected_rows": pool.stats["selected_rows"],
        "selected_days": pool.stats["selected_days"],
        "selected_stock_codes": pool.stats["selected_stock_codes"],
    }
    report["data_source"] = "public.market WHERE not_billing=1"
    report["risk_event_file"] = None if args.no_risk_events else str(args.risk_events)
    report["research_warning"] = RESEARCH_WARNING
    if args.reference_metadata is not None:
        with args.reference_metadata.open(encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("model_version") != "P4_RIDGE_V1":
            raise ValueError("Not P4_RIDGE_V1 reference metadata")
        if (metadata.get("validation_start") != report["split"]["validation_start"] or
                metadata.get("test_start") != report["split"]["test_start_EXCLUDED"] or
                metadata.get("refit_samples") != report["split"]["eligible_refit_pretest_samples"] or
                metadata.get("pool_config") != report["pool_config"]):
            raise ValueError("Reconstructed P4 pool or split differs from reference Ridge run")
        report["reference_metadata_match"] = True
    # Fail before writing anything if dates or reference metadata disagree.
    folder = args.out / f"{args.start}_{args.end}"
    folder.mkdir(parents=True, exist_ok=True)
    log("[3/4] Write train / validation factor IC, label tails, and samples for review")
    (folder / "factor_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    daily.to_csv(folder / "factor_ic_daily.csv", index=False, encoding="utf-8-sig")
    factors.to_csv(folder / "factor_ic_summary.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(folder / "label_quality.csv", index=False, encoding="utf-8-sig")
    outliers.to_csv(folder / "label_outliers.csv", index=False, encoding="utf-8-sig")
    log(f"[4/4] Audit saved: {folder}")
    print(json.dumps({"split": report["split"], "output": str(folder),
                      "test_labels_used_in_audit": False,
                      "warning": WARNING}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
