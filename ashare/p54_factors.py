"""Build an immutable, common-sample P5.4 market-context feature experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .p4_action_pipeline import atomic_json, read_json
from .p4_mvp import FEATURES
from .p4_subset_inputs import hash_file, parquet_read, parquet_write
from .p4_subset_transformer import fit_sequence_scaler

FACTOR_VERSION = "P54_MARKET_ENVIRONMENT_V1"
EXTRA_FEATURES = (
    "market_ret_5", "market_ret_20", "market_advancer_share_5", "market_turnover_log_change_20",
    "stock_relative_ret_20", "stock_breakout_20", "stock_volume_dryup_20", "stock_dry_volume_pullback_5",
)
P54_FEATURES = (*FEATURES, *EXTRA_FEATURES)
FOLDS = ("train", "validation", "test")
KEYS = ["stock_code", "trade_date"]
PRICE_FIELDS = ["open", "high", "low", "close", "volume", "amount"]
LABEL_FIELDS = {"target_return", "label_end_session", "entry_date", "exit_date", "split", "sample_grade"}
MIN_MARKET_PAIR_SHARE = .70


def _keys(frame):
    normalized = frame.loc[:, KEYS].copy()
    normalized["stock_code"] = normalized.stock_code.astype("string").str.strip()
    normalized["trade_date"] = pd.to_datetime(normalized.trade_date, errors="raise").dt.normalize()
    if (normalized.isna().any().any() or normalized.stock_code.eq("").any() or normalized.duplicated().any()):
        raise ValueError("Expected unique nonempty stock/session keys")
    return normalized


def _market_daily(bars, calendar, min_market_stocks):
    paired = bars.loc[bars.price_pair_valid]
    day = paired.groupby("trade_date").agg(
        pair_stocks=("one_day_return", "size"),
        median_stock_return=("one_day_return", "median"),
        advancer_share=("advancing", "mean"),
        median_turnover_log_change=("turnover_log_change", "median"),
    ).reindex(calendar)
    valid = bars.loc[bars.valid_bar].groupby("trade_date").size().reindex(calendar, fill_value=0)
    day["valid_formal_stocks"] = valid
    day["pair_stocks"] = day.pair_stocks.fillna(0).astype(int)
    day["pair_share"] = day.pair_stocks.div(pd.concat([valid, valid.shift(1)], axis=1).max(axis=1))
    day["market_session_usable"] = day.pair_stocks.ge(min_market_stocks) & day.pair_share.ge(MIN_MARKET_PAIR_SHARE)
    market = day[["median_stock_return", "advancer_share", "median_turnover_log_change"]].where(
        day.market_session_usable)
    log_return = np.log1p(market.median_stock_return)
    day["market_ret_5"] = np.expm1(log_return.rolling(5).sum())
    day["market_ret_20"] = np.expm1(log_return.rolling(20).sum())
    day["market_advancer_share_5"] = market.advancer_share.rolling(5).mean()
    day["market_turnover_log_change_20"] = market.median_turnover_log_change.rolling(20).sum()
    return day


def _stock_factors(bars, keys):
    stocks = bars.loc[bars.stock_code.isin(keys.stock_code.unique())].copy()
    grouped = stocks.groupby("stock_code", sort=False)

    def prior_rolling(column, count, statistic):
        prior = grouped[column].shift()
        return prior.groupby(stocks.stock_code, sort=False).transform(
            lambda series: getattr(series.rolling(count), statistic)()
        )

    complete = grouped.price_pair_valid.transform(lambda series: series.rolling(20).sum().eq(20))
    close_20_ago = grouped.close.shift(20)
    stock_return = stocks.close.div(close_20_ago).sub(1)
    prior_high_20 = prior_rolling("high", 20, "max")
    breakout = stocks.close.div(prior_high_20).sub(1)
    volume_dryup = 1.0 - stocks.volume.div(prior_rolling("volume", 20, "median"))
    prior_peak = prior_rolling("close", 5, "max")
    prior_ma = prior_rolling("close", 20, "mean")
    pullback_depth = 1.0 - stocks.close.div(prior_peak)
    uptrend_pullback = grouped.close.shift().gt(close_20_ago) & stocks.close.ge(prior_ma)
    pullback = (pullback_depth.clip(lower=0) * volume_dryup.clip(lower=0)).where(uptrend_pullback, 0.0)
    stocks["stock_ret_20"] = stock_return.where(complete)
    stocks["stock_breakout_20"] = breakout.where(complete)
    stocks["stock_volume_dryup_20"] = volume_dryup.where(complete)
    stocks["stock_dry_volume_pullback_5"] = pullback.where(complete)
    return stocks.loc[:, [*KEYS, "stock_ret_20", *EXTRA_FEATURES[5:]]]


def build_market_factors(raw, keys, *, min_market_stocks=100):
    """Compute trailing proxies at signal close without reading a future label or session."""
    if keys.empty or min_market_stocks < 2 or LABEL_FIELDS.intersection(keys):
        raise ValueError("Need feature-only keys and a nontrivial market breadth minimum")
    keys = _keys(keys)
    prices = raw.loc[pd.to_numeric(raw.not_billing, errors="coerce").eq(1), [*KEYS, *PRICE_FIELDS]].copy()
    prices["trade_date"] = pd.to_datetime(prices.trade_date, errors="raise").dt.normalize()
    cutoff = keys.trade_date.max()
    prices = prices.loc[prices.trade_date.le(cutoff)].reset_index(drop=True)
    if prices.empty:
        raise ValueError("No formal OHLCV through the last signal")
    prices[KEYS] = _keys(prices)
    prices[PRICE_FIELDS] = prices[PRICE_FIELDS].apply(pd.to_numeric, errors="coerce")
    bars = prices.sort_values(KEYS, kind="stable").reset_index(drop=True)
    calendar = pd.DatetimeIndex(bars.trade_date.unique()).sort_values()
    bars["session_id"] = calendar.get_indexer(bars.trade_date)
    bars["valid_bar"] = (np.isfinite(bars[PRICE_FIELDS]).all(axis=1) & bars.low.gt(0)
                         & bars.volume.gt(0) & bars.amount.gt(0)
                         & bars.high.ge(bars[["open", "close"]].max(axis=1))
                         & bars.low.le(bars[["open", "close"]].min(axis=1)))
    groups = bars.groupby("stock_code", sort=False)
    previous = groups[["session_id", "close", "amount"]].shift()
    previous_valid = groups.valid_bar.shift(fill_value=False)
    consecutive = bars.session_id.sub(previous.session_id).eq(1)
    opening_gap = bars.open.div(previous.close).sub(1)
    bars["one_day_return"] = bars.close.div(previous.close).sub(1)
    bars["price_pair_valid"] = (bars.valid_bar & previous_valid & consecutive
                                & opening_gap.abs().le(.30) & bars.one_day_return.abs().le(.32))
    bars["advancing"] = bars.one_day_return.gt(0)
    safe_amount = bars.amount.where(bars.price_pair_valid)
    bars["turnover_log_change"] = np.log(safe_amount) - np.log(previous.amount.where(bars.price_pair_valid))
    daily = _market_daily(bars, calendar, min_market_stocks).rename_axis("trade_date")
    stocks = _stock_factors(bars, keys)
    output = (keys.merge(stocks, how="left", on=KEYS, validate="one_to_one", sort=False)
              .merge(daily.loc[:, list(EXTRA_FEATURES[:4])], how="left", on="trade_date",
                     validate="many_to_one", sort=False))
    output["stock_relative_ret_20"] = output.stock_ret_20.sub(output.market_ret_20)
    output["trade_date"] = output.trade_date.dt.strftime("%Y-%m-%d")
    supported = np.isfinite(output[list(EXTRA_FEATURES)]).all(axis=1)
    day_counts = daily[["pair_stocks", "valid_formal_stocks", "pair_share", "market_session_usable"]].reset_index()
    day_counts["trade_date"] = day_counts.trade_date.dt.strftime("%Y-%m-%d")
    day_report = [{k: (None if pd.isna(v) else v) for k, v in row.items()}
                  for row in day_counts.to_dict("records")]
    audit = {
        "version": FACTOR_VERSION, "feature_names": list(P54_FEATURES), "extra_factors": list(EXTRA_FEATURES),
        "feature_rows": len(output), "all_extra_factors_finite_rows": int(supported.sum()),
        "nonfinite_by_factor": {name: int((~np.isfinite(output[name])).sum()) for name in EXTRA_FEATURES},
        "last_source_session": calendar[-1].strftime("%Y-%m-%d"), "last_signal_session": cutoff.strftime("%Y-%m-%d"),
        "formal_bar_count_at_or_before_last_signal": len(bars), "market_sessions_total": len(daily),
        "market_sessions_with_usable_daily_pairs": int(daily.market_session_usable.sum()),
        "minimum_market_pair_stocks": min_market_stocks, "minimum_market_pair_share": MIN_MARKET_PAIR_SHARE,
        "market_session_support": day_report,
        "market_return": "compound median paired formal-bar stock returns, 5 and 20 sessions",
        "market_sentiment": "5-session average of the share of paired names with positive daily returns",
        "market_activity": "sum of daily median paired log turnover changes over 20 sessions",
        "relative_strength": "stock 20-session return minus 20-session compounded market proxy",
        "breakout": "close divided by the highest high from the previous 20 sessions, minus one",
        "dry_volume": "one minus volume relative to its previous 20-session median",
        "dry_pullback": "positive depth from previous 5-session closing high multiplied by positive volume dryness when the prior close exceeds the 20-session lag and current close holds the prior 20-session mean",
        "market_adjustment_policy": "formal raw-bar robust proxy with paired valid bars and gap veto; no full-market action certification",
        "training_labels_opened_for_factor_values": False, "past_and_signal_close_inputs_only": True,
    }
    return output.loc[:, [*KEYS, *EXTRA_FEATURES]], audit


def _verify(path, expected):
    if hash_file(path) != expected:
        raise ValueError(f"P5.4 input hash mismatch: {path}")


def _unique_file_link(source, target):
    target.symlink_to(source.resolve(strict=True))


def _new_layout(root, dataset, audit):
    reports, transformer = root / "dataset", root / "transformer"
    reports.mkdir(parents=True)
    transformer.mkdir()
    for name in audit["files"]:
        if name not in {"causal_inference_features.parquet", "training_dataset.parquet"}:
            _unique_file_link(dataset / name, reports / name)
    return reports, transformer


def _publish_preflight(reports, transformer, audit, preflight, selected, matrix, names):
    files = {name: hash_file(reports / name) for name in audit["files"]}
    new_audit = reports / "subset_audit.json"
    atomic_json(new_audit, audit | {"files": files})
    means, scales, fit_rows = fit_sequence_scaler(matrix, selected["train"][0])
    scaler = transformer / "train_only_scaler.npz"
    np.savez(scaler, mean=means, scale=scales, feature_names=np.asarray(names))
    folds = {name: {"rows": len(values[0]), "sha256": hash_file(transformer / f"{name}_sequences.npz")}
             for name, values in selected.items()}
    atomic_json(transformer / "transformer_preflight.json", preflight | {
        "dataset_audit_sha256": hash_file(new_audit),
        "causal_features_sha256": files["causal_inference_features.parquet"],
        "labels_sha256": files["training_dataset.parquet"], "folds": folds,
        "scaler": {"file": scaler.name, "fit_fold": "train", "sha256": hash_file(scaler), "unique_feature_rows": fit_rows},
    })


def _filtered_labels(label_source, selected, total_rows):
    labels = parquet_read(label_source)
    retained = np.concatenate([pair[1] for pair in selected.values()])
    if len(labels) != total_rows or len(np.unique(retained)) != len(retained):
        raise ValueError("Paired source labels disagree with the frozen fold indexes")
    for name, (_, positions) in selected.items():
        if not labels.iloc[positions].split.eq(name).all():
            raise ValueError(f"Paired source includes an invalid {name} label key")
    labels = labels.iloc[np.sort(retained)].reset_index(drop=True)
    remap = np.full(total_rows, -1, dtype=np.int64)
    remap[np.sort(retained)] = np.arange(len(retained))
    splits = {}
    for name in FOLDS:
        rows = labels.loc[labels.split.eq(name)]
        daily = rows.groupby("trade_date").size()
        splits[name] = {"samples": len(rows), "stocks": rows.stock_code.nunique(), "days": len(daily),
                        "days_with_at_least_five_stocks": int(daily.ge(5).sum()),
                        "days_with_at_least_twenty_stocks": int(daily.ge(20).sum()),
                        "first_signal": str(rows.trade_date.min()), "last_signal": str(rows.trade_date.max()),
                        "last_label_exit": str(rows.exit_date.max())}
    selected = {name: (index, remap[positions]) for name, (index, positions) in selected.items()}
    return labels, selected, splits


def freeze_market_variant(source, out, *, min_market_stocks=100, paired_control=None):
    """Bind the expanded features to the original labels or a shared finite-window subset."""
    if out.exists() or (paired_control is not None and paired_control.exists()):
        raise ValueError("Use new output directories for immutable P5.4 bundles")
    if paired_control is not None:
        expanded_path, control_path = out.resolve(), paired_control.resolve()
        if expanded_path == control_path or expanded_path in control_path.parents or control_path in expanded_path.parents:
            raise ValueError("Paired experiment and control need independent output directories")
    dataset = source / "dataset"
    transformer = source / "transformer"
    audit_path = dataset / "subset_audit.json"
    preflight_path = transformer / "transformer_preflight.json"
    audit, preflight = read_json(audit_path), read_json(preflight_path)
    if (tuple(audit["feature_names"]) != FEATURES or not audit["subset_training_allowed"]
            or preflight["dataset_audit_sha256"] != hash_file(audit_path)
            or preflight["status"] != "SEQUENCE_INPUTS_READY"
            or preflight["sequence_length"] != audit["sequence_eligibility_length"]
            or preflight["scaler"]["fit_fold"] != "train"
            or preflight["causal_features_sha256"] != audit["files"]["causal_inference_features.parquet"]
            or preflight["labels_sha256"] != audit["files"]["training_dataset.parquet"]):
        raise ValueError("Expected an audited ten-factor P5 source bundle")
    for name, digest in audit["files"].items():
        if Path(name).name != name:
            raise ValueError("Unexpected nested P5 dataset filename")
        _verify(dataset / name, digest)
    raw_path = source / "inputs/formal_ohlcv.parquet"
    _verify(raw_path, audit["frozen_price_sha256"])
    features = parquet_read(dataset / "causal_inference_features.parquet")
    extras, factor_report = build_market_factors(parquet_read(raw_path), features, min_market_stocks=min_market_stocks)
    originals = _keys(features)
    if (not originals.stock_code.equals(extras.stock_code)
            or not originals.trade_date.dt.strftime("%Y-%m-%d").equals(extras.trade_date)):
        raise ValueError("Market factor expansion changed feature key order")
    expanded = features.join(extras.loc[:, list(EXTRA_FEATURES)])
    matrix = expanded.loc[:, list(P54_FEATURES)].to_numpy(dtype=float)
    row_we_have = np.isfinite(matrix).all(axis=1)
    selected, fold_coverage = {}, {}
    total_labels = sum(preflight["folds"][name]["rows"] for name in FOLDS)
    for fold in FOLDS:
        path = transformer / f"{fold}_sequences.npz"
        record = preflight["folds"][fold]
        _verify(path, record["sha256"])
        with np.load(path, allow_pickle=False) as saved:
            index = saved["causal_feature_rows"]
            label_index = saved["training_dataset_rows"]
        if (index.shape != (record["rows"], preflight["sequence_length"]) or index.dtype.kind not in "iu"
                or label_index.shape != (record["rows"],) or label_index.dtype.kind not in "iu"
                or record["rows"] <= 0 or (index < 0).any() or (index >= len(expanded)).any()
                or (label_index < 0).any() or (label_index >= total_labels).any()
                or len(np.unique(label_index)) != len(label_index)):
            raise ValueError(f"P5.4 cannot preserve malformed {fold} keys")
        finite = row_we_have[index].all(axis=1)
        fold_coverage[fold] = {"source_sequences": len(index),
                               "all_inputs_finite_sequences": int(finite.sum()),
                               "lost_sequences": int((~finite).sum())}
        selected[fold] = index[finite], label_index[finite]
    exact = all(not value["lost_sequences"] for value in fold_coverage.values())
    shared = paired_control is not None and all(value["all_inputs_finite_sequences"] for value in fold_coverage.values())
    ready = exact or shared
    factor_report |= {"source_audit_sha256": hash_file(audit_path), "source_preflight_sha256": hash_file(preflight_path),
                      "source_price_snapshot_sha256": audit["frozen_price_sha256"], "fold_coverage": fold_coverage,
                      "identical_train_validation_and_test_anchors": ready,
                      "exact_match_to_source_anchors": exact, "model_inner_architecture_changed": False,
                      "scaler_fit_fold": "train" if ready else None,
                      "status": ("EXACT_PAIRED_FEATURES_READY" if exact else "FILTERED_PAIRED_FEATURES_READY" if shared
                                 else "REJECTED_INCOMPLETE_PAIRED_WINDOWS")}
    out.mkdir(parents=True)
    atomic_json(out / "factor_audit.json", factor_report)
    if not ready:
        raise ValueError(f"Incomplete expanded factors on source sequences; inspect {out / 'factor_audit.json'}")

    destination, experiment = _new_layout(out, dataset, audit)
    parquet_write(expanded, destination / "causal_inference_features.parquet")
    label_file = "training_dataset.parquet"
    if exact:
        splits = audit.get("splits", {})
        _unique_file_link(dataset / label_file, destination / label_file)
        for fold in FOLDS:
            path = f"{fold}_sequences.npz"
            _unique_file_link(transformer / path, experiment / path)
    else:
        label_rows, selected, splits = _filtered_labels(dataset / label_file, selected, total_labels)
        parquet_write(label_rows, destination / label_file)
        for fold, (index, targets) in selected.items():
            np.savez_compressed(experiment / f"{fold}_sequences.npz", causal_feature_rows=index,
                                training_dataset_rows=targets)
    common = audit | {"source_subset_audit_sha256": hash_file(audit_path), "splits": splits,
                      "paired_sample_fold_coverage": fold_coverage,
                      "factor_audit_sha256": hash_file(out / "factor_audit.json")}
    expanded_audit = common | {
        "version": FACTOR_VERSION, "feature_names": list(P54_FEATURES), "base_feature_names": list(FEATURES),
        "source_dataset_version": audit["version"], "exact_paired_feature_rows": exact,
    }
    _publish_preflight(destination, experiment, expanded_audit, preflight, selected, matrix, P54_FEATURES)
    if paired_control is not None:
        control_data, control_transformer = _new_layout(paired_control, dataset, audit)
        base_feature_file = "causal_inference_features.parquet"
        _unique_file_link(dataset / base_feature_file, control_data / base_feature_file)
        _unique_file_link(destination / label_file, control_data / label_file)
        for fold in FOLDS:
            path = f"{fold}_sequences.npz"
            _unique_file_link(experiment / path, control_transformer / path)
        base_matrix = features.loc[:, list(FEATURES)].to_numpy(dtype=float)
        _publish_preflight(control_data, control_transformer, common, preflight, selected, base_matrix, FEATURES)
    print(f"{FACTOR_VERSION}: {len(P54_FEATURES)} inputs on {len(expanded):,} feature rows; paired folds {fold_coverage}",
          flush=True)
    return factor_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paired-control", type=Path)
    parser.add_argument("--minimum-market-stocks", type=int, default=100)
    args = parser.parse_args()
    freeze_market_variant(args.source, args.out, min_market_stocks=args.minimum_market_stocks,
                          paired_control=args.paired_control)


if __name__ == "__main__":
    main()
