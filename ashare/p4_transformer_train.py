"""Train and freeze the P4 research Transformer without opening the test fold."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .p4_action_pipeline import atomic_json, read_json
from .p4_subset_dataset import GUARD, TRUSTED, time_split
from .p4_subset_inputs import hash_file, parquet_read, parquet_write
from .p4_subset_model import FEATURES, dataset_summary, metrics

VERSION = "P4.4_TRANSFORMER_V1"
ARCHITECTURE = dict(model_width=48, attention_heads=4, encoder_layers=2, feedforward_width=96)
EVALUATION_PLAN = {
    "historical_test": "single comparison on the frozen test sequence population",
    "baselines": ["frozen_ridge", "matched_train_ridge", "negative_ma20", "train_mean"],
    "ridge_alphas": [0.01, 0.1, 1.0, 10.0],
    "rank_ic": "unweighted mean daily Spearman, minimum 5 observations, omit undefined days",
    "direction": "three-way sign including zero",
    "confidence": "absolute prediction; threshold fixed at validation quantile",
    "confidence_quantile": 0.8,
    "top_k": [10, 20],
    "turnover": "fraction of prior names replaced across successive observed lists",
    "cost_bps_per_side": [5, 10, 25],
    "cohort": "equal-weight five-session event cohorts; no compounding or fill assumption",
    "prospective": {
        "start": "first formal session strictly after model and protocol freeze in Asia/Shanghai",
        "minimum_matured_sessions": 20, "minimum_scored_label_coverage": 0.8,
        "compare": ["frozen_transformer", "frozen_ridge", "negative_ma20", "train_mean"],
        "candidate_min_training_sequences": 20,
        "primary_metrics": ["daily_rank_ic", "top20_net_excess_at_10bps_per_side", "label_coverage"],
        "prediction_gate": "positive mean daily Rank IC and positive Top20 net excess; report uncertainty",
        "labels": "T+1 to T+6 economic returns from independently checked action windows",
        "retune_on_evaluation": False,
    },
}


class ReturnTransformer(nn.Module):
    def __init__(self, sequence_length, feature_count, model_width, attention_heads,
                 encoder_layers, feedforward_width):
        super().__init__()
        self.input_projection = nn.Linear(feature_count, model_width)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_width, nhead=attention_heads,
                                       dim_feedforward=feedforward_width, dropout=0.0, batch_first=True),
            num_layers=encoder_layers, enable_nested_tensor=False)
        self.head = nn.Linear(model_width, 1)
        position = torch.sin(torch.arange(sequence_length, dtype=torch.float32)[:, None]
                             / torch.arange(1, model_width + 1, dtype=torch.float32)[None, :])
        self.register_buffer("position", position, persistent=False)
        self.register_buffer("causal", torch.ones(sequence_length, sequence_length, dtype=torch.bool).triu(1),
                             persistent=False)

    def encode(self, x):
        projected = self.input_projection(x)
        return self.encoder(projected + self.position.to(projected.dtype), mask=self.causal, is_causal=True)

    def forward(self, x):
        return self.head(self.encode(x)[:, -1]).flatten()


def verify_digest(path, expected):
    if hash_file(path) != expected:
        raise ValueError(f"Frozen file digest mismatch: {path}")


def scale_sequences(matrix, indices, mean, scale):
    return ((matrix[indices] - mean) / scale).astype(np.float32)


class FrozenData:
    def __init__(self, root, *, audit_reader=dataset_summary):
        self.root = root
        self.dataset = root / "dataset"
        preflight = root / "transformer/transformer_preflight.json"
        self.audit = audit_reader(self.dataset)
        self.feature_names = tuple(self.audit["feature_names"])
        self.end_exclusive = (self.audit["research_end_exclusive"] if "research_end_exclusive" in self.audit
                              else self.audit["old_test_excluded"])
        self.record = read_json(preflight)
        self.audit_sha256 = hash_file(self.dataset / "subset_audit.json")
        self.preflight_sha256 = hash_file(preflight)
        config = self.record["initial_training_config"]
        if (self.record["status"] != "SEQUENCE_INPUTS_READY" or self.record["legacy_training_gate"] != "CLOSED"
                or self.record["dataset_audit_sha256"] != self.audit_sha256
                or self.record["causal_features_sha256"] != self.audit["files"]["causal_inference_features.parquet"]
                or self.record["labels_sha256"] != self.audit["files"]["training_dataset.parquet"]
                or self.record["sequence_length"] != self.audit["sequence_eligibility_length"]
                or any(config[name] != value for name, value in ARCHITECTURE.items())
                or self.record["scaler"]["fit_fold"] != "train"):
            raise ValueError("Dataset and Transformer preflight have incompatible provenance")
        scaler_file = root / "transformer/train_only_scaler.npz"
        verify_digest(scaler_file, self.record["scaler"]["sha256"])
        with np.load(scaler_file, allow_pickle=False) as artifact:
            self.mean, self.scale = artifact["mean"], artifact["scale"]
            if (tuple(artifact["feature_names"]) != self.feature_names or self.mean.shape != (len(self.feature_names),)
                    or self.scale.shape != self.mean.shape or not np.isfinite(self.mean).all()
                    or not np.isfinite(self.scale).all() or (self.scale <= 0).any()):
                raise ValueError("Invalid frozen training scaler")
        self.config = config | {"sequence_length": self.record["sequence_length"], "feature_count": len(self.feature_names)}
        self.features = parquet_read(self.dataset / "causal_inference_features.parquet")
        if (self.features.duplicated(["stock_code", "session_id"]).any()
                or {"target_return", "split", "exit_date", "sample_grade"} & set(self.features.columns)):
            raise ValueError("Expected unique feature-only candidate rows")
        self.matrix = self.features[list(self.feature_names)].to_numpy(dtype=float)

    def fold(self, name):
        import polars as pl

        if name not in ("train", "validation", "test"):
            raise ValueError("Invalid frozen fold")
        specification = self.record["folds"][name]
        path = self.root / "transformer" / (name + "_sequences.npz")
        verify_digest(path, specification["sha256"])
        with np.load(path, allow_pickle=False) as artifact:
            targets, indices = artifact["training_dataset_rows"], artifact["causal_feature_rows"]
        count = specification["rows"]
        if (count <= 0 or targets.shape != (count,) or indices.shape != (count, self.config["sequence_length"])
                or targets.dtype.kind not in "iu" or indices.dtype.kind not in "iu"
                or (targets < 0).any() or len(np.unique(targets)) != count
                or (indices < 0).any() or (indices >= len(self.features)).any()):
            raise ValueError(f"Malformed frozen {name} indexes")
        selected = (pl.scan_parquet(self.dataset / "training_dataset.parquet")
                    .with_row_index("row_id").filter(pl.col("split") == name).collect())
        lookup = pd.DataFrame(selected.to_dict(as_series=False)).set_index("row_id")
        if len(lookup) != self.audit["splits"][name]["samples"] or not np.isin(targets, lookup.index).all():
            raise ValueError(f"Frozen {name} indexes refer to the wrong fold")
        rows = lookup.loc[targets].reset_index(drop=True)
        codes = self.features.stock_code.to_numpy()[indices]
        sessions = self.features.session_id.to_numpy()[indices]
        anchor = self.features.trade_date.to_numpy()[indices[:, -1]]
        if (not (codes == rows.stock_code.to_numpy()[:, None]).all() or not (np.diff(sessions, axis=1) == 1).all()
                or not np.array_equal(sessions[:, -1], rows.session_id.to_numpy())
                or not np.array_equal(anchor, rows.trade_date.to_numpy())
                or not rows.sample_grade.eq(TRUSTED).all() or not rows.sequence_feature_evidence_ready.all()
                or not (rows.trade_date.lt(rows.entry_date) & rows.entry_date.lt(rows.exit_date)
                        & rows.exit_date.le(self.audit["price_end"]) & rows.exit_date.lt(self.end_exclusive)).all()
                or any(time_split(r.trade_date, r.exit_date, self.audit["val_start"], self.audit["test_start"]) != name
                       for r in rows.itertuples(index=False))):
            raise ValueError(f"Frozen {name} sequence keys, qualification, or timing invalid")
        x = scale_sequences(self.matrix, indices, self.mean, self.scale)
        y = rows.target_return.to_numpy(dtype=np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Nonfinite inputs in {name}")
        return rows, x, y


def make_model(specification):
    return ReturnTransformer(specification["sequence_length"], specification["feature_count"],
                             **{name: specification[name] for name in ARCHITECTURE})


def atomic_torch(path, obj):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(obj, stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            folder_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(folder_fd)
            finally:
                os.close(folder_fd)
        finally:
            temporary.unlink(missing_ok=True)


def load_model(folder, device, *, version=VERSION, feature_names=FEATURES):
    info = read_json(folder / "model_metadata.json")
    weights = folder / "best_model.pt"
    verify_digest(weights, info["weights_sha256"])
    if (info["version"] != version or tuple(info["feature_names"]) != tuple(feature_names)
            or info["architecture"]["feature_count"] != len(feature_names)
            or any(info[key] != value for key, value in GUARD.items())
            or any(info["architecture"][name] != value for name, value in ARCHITECTURE.items())):
        raise ValueError("Model contract is incompatible")
    model = make_model(info["architecture"]).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.eval(), info


def predict(model, x, device, batch=64):
    model.eval()
    with torch.inference_mode():
        inputs = torch.as_tensor(x, dtype=torch.float32, device=device)
        values = torch.cat([model(part).float() for part in inputs.split(batch)]).cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Model produced nonfinite scores")
    return values


def precision_for(requested):
    if requested == "fp32":
        return "fp32"
    if torch.cuda.is_bf16_supported():
        return "bf16"
    if requested == "bf16":
        print("BF16 unavailable; falling back to FP32", flush=True)
    return "fp32"


def train(source, output, requested, resume):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal training")
    if resume and not (output / "checkpoint.pt").is_file():
        raise ValueError("Resume requires an existing epoch checkpoint")
    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _fit(FrozenData(source), output, precision_for(requested), resume)


def _fit(data, output, precision, resume, *, version=VERSION, plan=EVALUATION_PLAN, label_transform=None,
         experiment_metadata=None):
    device = torch.device("cuda:0")
    cfg = data.config
    identity = {"version": version, "audit_sha256": data.audit_sha256, "preflight_sha256": data.preflight_sha256,
                "configuration": cfg, "precision": precision, "torch": str(torch.__version__),
                "gpu": torch.cuda.get_device_name(device), "evaluation_plan": plan}
    if experiment_metadata is not None:
        identity["experiment"] = experiment_metadata
    if (output / "model_metadata.json").exists():
        raise ValueError("Training already completed; its selected model is frozen")
    policy_file = output / "evaluation_plan.json"
    if resume:
        if read_json(policy_file) != plan:
            raise ValueError("Evaluation plan changed since training started")
    else:
        atomic_json(policy_file, plan)
    rows_train, x, y = data.fold("train")
    rows_val, vx, vy = data.fold("validation")
    validation_labels = rows_val
    if label_transform is not None:
        y = label_transform(rows_train)
        vy = label_transform(rows_val)
        validation_labels = rows_val.assign(target_return=vy)
    validation_x = torch.as_tensor(vx, device=device)
    x = torch.as_tensor(x, device=device)
    y = torch.as_tensor(y, device=device)
    torch.manual_seed(cfg["seed"])
    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], fused=True)
    checkpoint = output / "checkpoint.pt"
    batch, total = cfg["batch_size"], cfg["max_epochs"]
    model_state = None
    history, best_ic, best_epoch = [], -float("inf"), 0
    if resume:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Checkpoint dataset, architecture, precision, policy or device changed")
        model.load_state_dict(state["state"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_ic, best_epoch, model_state = state["history"], state["ic"], state["epoch"], state["best"]
    else:
        state = {"identity": identity, "state": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "history": history, "ic": best_ic, "epoch": best_epoch, "best": model_state}
        atomic_torch(checkpoint, state)

    print(f"{len(x)} training / {len(vx)} validation sequences on {identity['gpu']} ({precision})", flush=True)
    for epoch in range(len(history) + 1, total + 1):
        if len(history) - best_epoch >= cfg["early_stop_patience"]:
            break
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        order = torch.randperm(len(x), device=device, generator=torch.Generator(device=device).manual_seed(cfg["seed"] + epoch))
        loss_sum = torch.zeros((), device=device)
        for step, index in enumerate(order.split(batch), 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                score = model(x[index])
                loss = nn.functional.smooth_l1_loss(score.float(), y[index])
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(index)
            if step % 250 == 0:
                print(f"epoch {epoch}/{total}, batch {step}/{len(order.split(batch))}, "
                      f"train_loss={loss_sum.item()/min(step*batch,len(x)):.7f}", flush=True)
        scores = predict(model, validation_x, device, batch)
        val = metrics(validation_labels, scores)
        if val["mean_daily_rank_ic"] is None or not torch.isfinite(loss_sum).item():
            raise ValueError("Training loss or validation Rank IC is invalid")
        absolute = np.abs(scores - vy)
        record = {"epoch": epoch, "train_loss": loss_sum.item()/len(x),
                  "validation_loss": float(np.where(absolute < 1, 0.5 * absolute**2, absolute - 0.5).mean()),
                  "validation_rank_ic": val["mean_daily_rank_ic"], "validation_direction_accuracy": val["direction_accuracy"],
                  "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.monotonic()-started,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}
        history.append(record)
        if val["mean_daily_rank_ic"] > best_ic:
            best_ic, best_epoch = val["mean_daily_rank_ic"], epoch
            model_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        atomic_torch(checkpoint, {"identity": identity, "state": model.state_dict(), "optimizer": optimizer.state_dict(),
                                  "history": history, "ic": best_ic, "epoch": best_epoch, "best": model_state})
        print(record, flush=True)

    model.load_state_dict(model_state)
    before = predict(model, validation_x, device, batch)
    weights = output / "best_model.pt"
    atomic_torch(weights, model_state)
    restored = make_model(cfg).to(device)
    restored.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    after = predict(restored, validation_x, device, batch)
    if not np.array_equal(before, after):
        raise RuntimeError("Restored Transformer does not reproduce validation scores")
    support = {str(code): int(count) for code, count in rows_train.stock_code.value_counts().items()}
    report = {
        "version": version, **GUARD, "research_only": True, "dataset_version": data.audit["version"],
        "training_identity": identity, "feature_names": list(data.audit.get("feature_names", FEATURES)), "architecture": cfg,
        "mean": data.mean.tolist(), "scale": data.scale.tolist(), "dataset_audit_sha256": data.audit_sha256,
        "weights_sha256": hash_file(weights), "evaluation_plan_sha256": hash_file(policy_file),
        "train_first_signal": str(rows_train.trade_date.min()), "train_last_signal": str(rows_train.trade_date.max()),
        "train_last_exit": str(rows_train.exit_date.max()), "selection_last_exit": str(rows_val.exit_date.max()),
        "old_test_excluded": (data.audit["v1_old_test_excluded"] if experiment_metadata is not None
                              else data.audit["old_test_excluded"]), "test_start": data.audit["test_start"],
        "holdout_kind": data.audit["test_kind"], "pool_config": data.audit["pool"]["pool_config"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "train_stock_counts": support,
        "train_mean_return": float(rows_train.target_return.mean()), "training_rows": len(rows_train),
        "validation_rows": len(rows_val), "epochs": history, "best_epoch": best_epoch,
        "best_validation_rank_ic": best_ic, "early_stopped": len(history) < total,
        "reload_predictions_match_exactly": True, "reload_predictions_checked": len(before),
        "model_training_accessed_test_labels": False,
    }
    if experiment_metadata is not None:
        validation_path = output / "validation_predictions.parquet"
        parquet_write(rows_val[["stock_code", "trade_date", "target_return"]].assign(score=after), validation_path)
        report.update(experiment_metadata, validation_predictions_sha256=hash_file(validation_path),
                      validation_first_signal=str(rows_val.trade_date.min()),
                      validation_last_signal=str(rows_val.trade_date.max()))
    atomic_json(output / "model_metadata.json", report)
    print(f"FROZEN best epoch {best_epoch}, validation IC {best_ic:.6f}, reload={len(before)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.source, args.out, args.precision, args.resume)


if __name__ == "__main__":
    main()
