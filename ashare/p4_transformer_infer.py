"""Score historical feature-only inputs or the latest formal PostgreSQL bars."""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text
import torch

from .daily import AShareDailyLoader
from .p4_action_pipeline import atomic_json, read_json
from .p4_mvp import FEATURES, PoolConfig, RidgeModel, build_pool
from .p4_subset_dataset import FEATURE_LOOKBACK
from .p4_subset_inputs import RISK_FILE, hash_file, parquet_read, parquet_write
from .p4_transformer_train import load_model, predict, scale_sequences, verify_digest

CHINA = ZoneInfo("Asia/Shanghai")
FORMAL_BAR_READY = time(18)


def inference_indexes(features, asof, length, counts, minimum):
    forbidden = {"target_return", "label_end_session", "entry_date", "exit_date", "split", "sample_grade"}
    if forbidden & set(features.columns) or features.duplicated(["stock_code", "session_id"]).any():
        raise ValueError("Inference requires unique feature-only stock sessions")
    past = features.loc[features.trade_date.le(asof)].reset_index(drop=True)
    current = past.index[past.trade_date.eq(asof)]
    positions = {(r.stock_code, r.session_id): i for i, r in enumerate(past.itertuples(index=False))}
    anchors, sequence = [], []
    status = {"causal_candidates_on_date": len(current), "insufficient_training_sequences": 0, "sequence_gaps": 0}
    for i in current:
        stock, session = past.loc[i, ["stock_code", "session_id"]]
        if counts.get(stock, 0) < minimum:
            status["insufficient_training_sequences"] += 1
            continue
        rows = [positions.get((stock, index)) for index in range(session + 1 - length, session + 1)]
        if any(index is None for index in rows):
            status["sequence_gaps"] += 1
            continue
        anchors.append(i)
        sequence.append(rows)
    selected = past.iloc[anchors][["stock_code", "trade_date", "liquidity_rank", *FEATURES]].copy()
    return selected, past[list(FEATURES)].to_numpy(dtype=float), np.asarray(sequence, dtype=np.int64).reshape(-1, length), status


def latest_ohlcv(engine, now):
    first_unavailable = now.date() + timedelta(days=now.time() >= FORMAL_BAR_READY)
    latest = text("SELECT max(trade_date) FROM public.market WHERE not_billing=1 AND trade_date < :unavailable")
    prices = text("""
        SELECT stock_code, trade_date, open, high, low, close, volume, amount, pre_close, not_billing
          FROM public.market
         WHERE not_billing=1 AND stock_code IS NOT NULL AND trade_date BETWEEN :begin AND :end
    """)
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
        with conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            if conn.scalar(text("show transaction_read_only")) != "on":
                raise RuntimeError("Formal OHLCV requires a read-only snapshot")
            asof = conn.scalar(latest, {"unavailable": first_unavailable})
            if asof is None:
                raise ValueError("No formal market session available")
            asof = pd.Timestamp(asof).date()
            if (now.date() - asof).days > 10:
                raise ValueError(f"Latest formal OHLCV is stale: {asof}")
            bounds = {"begin": asof - timedelta(days=140), "end": asof}
            batches = pd.read_sql_query(prices, conn.execution_options(stream_results=True),
                                        params=bounds, chunksize=75000)
            raw = pd.concat(batches, ignore_index=True)
    return asof, raw


def price_quality_keys(raw, calendar):
    """Match the F2 causal price-window checks, using only past observed bars."""
    raw = raw.copy()
    raw["trade_date"] = pd.to_datetime(raw.trade_date).dt.normalize()
    raw["stock_code"] = raw.stock_code.astype("string").str.strip()
    cols = ["open", "close", "high", "low", "volume", "amount", "pre_close"]
    raw[cols] = raw[cols].apply(pd.to_numeric, errors="coerce")
    raw = raw.sort_values(["stock_code", "trade_date"])
    raw["session"] = raw.trade_date.map({day: i for i, day in enumerate(calendar)}).astype(int)
    grouped = raw.groupby("stock_code", sort=False)
    prior = grouped.close.shift(1)
    ref_ok = np.isfinite(raw.pre_close) & raw.pre_close.gt(0)
    bar_ok = (np.isfinite(raw[cols]).all(axis=1) & raw.low.gt(0) & raw.volume.gt(0) & raw.amount.gt(0)
              & raw.high.ge(raw[["open", "close"]].max(axis=1)) & raw.low.le(raw[["open", "close"]].min(axis=1)))
    continuous = raw.session.sub(grouped.session.shift(1)).eq(1)
    jump = raw.open.div(prior).sub(1).abs().gt(.30) | raw.close.div(prior).sub(1).abs().gt(.32)
    valid = (ref_ok & bar_ok & continuous & ~jump).groupby(raw.stock_code, sort=False).transform(
        lambda values: values.rolling(FEATURE_LOOKBACK, min_periods=FEATURE_LOOKBACK).sum().eq(FEATURE_LOOKBACK))
    return raw.loc[valid, ["stock_code", "trade_date"]]


def latest_features(info, raw, asof, events):
    pool = build_pool(raw, asof - timedelta(days=46), asof, PoolConfig(**info["pool_config"]), events=events)
    features = pool.data.drop(columns=["target_return", "label_end_session"])
    features = features.merge(price_quality_keys(raw, pool.calendar), on=["stock_code", "trade_date"],
                              validate="one_to_one")
    features["trade_date"] = features.trade_date.dt.strftime("%Y-%m-%d")
    return features


def score(source, model_folder, output, asof, live, device):
    model, info = load_model(model_folder, device)
    policy_path = model_folder / "evaluation_plan.json"
    verify_digest(policy_path, info["evaluation_plan_sha256"])
    policy = read_json(policy_path)["prospective"]
    bar_snapshot_at = datetime.now(CHINA)
    freeze_at = datetime.fromisoformat(info["frozen_at_utc"]).astimezone(CHINA)
    audit_path = source / "dataset/subset_audit.json"
    verify_digest(audit_path, info["dataset_audit_sha256"])
    if live:
        latest, raw = latest_ohlcv(AShareDailyLoader().engine, bar_snapshot_at)
        asof = latest.isoformat()
        risk_events = pd.read_csv(RISK_FILE, dtype={"stock_code": "string"})
        features = latest_features(info, raw, latest, risk_events)
        quality = "FORMAL_OHLCV_PASSED_ACTION_AND_HISTORICAL_ST_COVERAGE_PARTIAL"
    else:
        date.fromisoformat(asof)
        if asof >= info["old_test_excluded"]:
            raise ValueError("Historical replay must stay before the original holdout")
        feature_path = source / "dataset/causal_inference_features.parquet"
        verify_digest(feature_path, read_json(audit_path)["files"][feature_path.name])
        features = parquet_read(feature_path)
        quality = "FROZEN_CAUSAL_OHLCV_PASSED"
    if asof <= info["selection_last_exit"]:
        raise ValueError("Prediction date is before the last model-selection label exit")
    if output.exists():
        raise FileExistsError(f"Prediction artifact already exists: {output}")

    cfg = info["architecture"]
    rows, matrix, indices, counts = inference_indexes(features, asof, cfg["sequence_length"],
                                                       info["train_stock_counts"], policy["candidate_min_training_sequences"])
    if rows.empty and not live:
        raise ValueError(f"No eligible causal sequence at {asof}: {counts}")
    values = (np.empty(0, dtype=np.float64) if rows.empty else
              predict(model, scale_sequences(matrix, indices, info["mean"], info["scale"]),
                      device, cfg["batch_size"]).astype(np.float64))
    ridge_path = source / "model/ridge_model.npz"
    ridge_meta = read_json(source / "model/model_metadata.json")
    verify_digest(ridge_path, ridge_meta["model_sha256"])
    if ridge_meta["dataset_audit_sha256"] != info["dataset_audit_sha256"]:
        raise ValueError("Ridge comparison is from a different snapshot")
    result = rows[["stock_code", "trade_date", "liquidity_rank"]].copy().assign(
        predicted_economic_return=values, ridge=RidgeModel.load(ridge_path).predict(rows),
        negative_ma20=-rows.close_ma20_dev.to_numpy(), train_mean=info["train_mean_return"],
        training_sequences=rows.stock_code.map(info["train_stock_counts"]), model_version=info["version"],
        model_sha256=info["weights_sha256"], data_quality=quality)
    result = result.sort_values(["predicted_economic_return", "stock_code"], ascending=[False, True]).reset_index(drop=True)
    output.mkdir(parents=True)
    parquet = output / "predictions.parquet"
    if len(result):
        parquet_write(result, parquet)
    else:
        import polars as pl

        schema = {name: pl.String for name in ("stock_code", "trade_date", "model_version", "model_sha256", "data_quality")}
        schema.update({name: pl.Int64 for name in ("liquidity_rank", "training_sequences")})
        schema.update({name: pl.Float64 for name in ("predicted_economic_return", "ridge", "negative_ma20", "train_mean")})
        pl.DataFrame(schema={column: schema[column] for column in result}).write_parquet(parquet, compression="zstd")
    timely = live and asof == bar_snapshot_at.date().isoformat() and date.fromisoformat(asof) > freeze_at.date()
    atomic_json(output / "inference_audit.json", {
        **counts, "asof": asof, "rows": len(result), "feature_columns": list(FEATURES),
        "model_version": info["version"], "weights_sha256": info["weights_sha256"],
        "ridge_weights_sha256": ridge_meta["model_sha256"], "predictions_sha256": hash_file(parquet),
        "evaluation_plan_sha256": info["evaluation_plan_sha256"],
        "prediction_context": "LATEST_FORMAL_POSTGRESQL" if live else "HISTORICAL_FEATURE_ONLY_REPLAY",
        "snapshot_clock_asia_shanghai": bar_snapshot_at.isoformat(),
        "inference_status": "SCORED" if len(result) else "ABSTAINED_NO_QUALIFIED_SEQUENCE",
        "prospective_session_capture": timely,
        "prospective_eligible": timely and bool(len(result)),
        "model_frozen_asia_shanghai": freeze_at.isoformat(), "data_quality": quality,
        "risk_source_sha256": hash_file(RISK_FILE) if live else None,
        "uses_future_labels": False, "uses_current_st_backfill": False, "research_only": True,
    })
    print(f"{asof} {len(result)} causal predictions, {quality}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--asof")
    mode.add_argument("--latest", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score(args.source, args.model, args.out, args.asof, args.latest, device)


if __name__ == "__main__":
    main()
