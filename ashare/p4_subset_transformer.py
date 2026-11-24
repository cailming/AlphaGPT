"""RTX Transformer hardware and causal sequence-shape preflight; no market fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def preflight(batch_size, sequence):
    import torch
    from torch import nn

    if not 1 <= batch_size <= 256 or not 4 <= sequence <= 64:
        raise ValueError("Bounded GPU smoke-test dimensions required")
    if not torch.cuda.is_available():
        raise ValueError("A CUDA-enabled Torch environment is required")
    device = torch.device("cuda:0")
    before = torch.cuda.mem_get_info(device)
    if before[0] < 768 * 1024**2:
        raise ValueError("Not enough free GPU memory for preflight")
    torch.manual_seed(0)
    model_width = 48
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=model_width, nhead=4, dim_feedforward=96,
                                   dropout=0.0, batch_first=True), num_layers=2,
        enable_nested_tensor=False,
    ).to(device).eval()
    projection = nn.Linear(10, model_width, device=device).eval()
    head = nn.Linear(model_width, 1, device=device).eval()
    position = torch.sin(torch.arange(sequence, dtype=torch.float32, device=device)[:, None]
                         / torch.arange(1, model_width+1, dtype=torch.float32, device=device)[None, :])
    causal = torch.ones((sequence, sequence), device=device, dtype=torch.bool).triu(diagonal=1)
    sample = torch.zeros(batch_size, sequence, 10, device=device)
    changed = sample.clone()
    changed[:, -1, :] = 1
    with torch.inference_mode():
        before_change = encoder(projection(sample) + position, mask=causal, is_causal=True)
        after_change = encoder(projection(changed) + position, mask=causal, is_causal=True)
        output = head(before_change[:, -1]).flatten()
        if (output.shape != (batch_size,) or not torch.isfinite(output).all()
                or not torch.allclose(before_change[:, :-1], after_change[:, :-1], rtol=1e-6, atol=1e-6)):
            raise ValueError("GPU output shape or causal mask smoke check failed")
    encoder.train()
    projection.train()
    head.train()
    bf16 = torch.cuda.is_bf16_supported()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
        scores = head(encoder(projection(changed) + position, mask=causal, is_causal=True)[:, -1]).flatten()
        loss = nn.functional.smooth_l1_loss(scores, torch.zeros_like(scores))
    loss.backward()
    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
           for module in (encoder, projection, head) for parameter in module.parameters()):
        raise ValueError("GPU gradient smoke check failed")
    torch.cuda.synchronize()
    return {"status": "CUDA_SMOKE_PASSED", "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
            "cuda": torch.version.cuda, "bf16_supported": bool(bf16),
            "device_memory_bytes": before[1], "batch": batch_size, "sequence": sequence,
            "uses_market_data": False, "causal_mask_checked": True, "synthetic_backward_checked": True,
            "transformer_training_executed": False}


def sequence_indexes(features, labels, length):
    import numpy as np

    if features.duplicated(["stock_code", "session_id"]).any():
        raise ValueError("Duplicate causal sequence keys")
    keys = {(row.stock_code, int(row.session_id)): position for position, row in enumerate(features.itertuples(index=False))}
    indexes = {split: ([], []) for split in ("train", "validation", "test")}
    for target_pos, row in enumerate(labels.itertuples(index=False)):
        if not row.sequence_feature_evidence_ready:
            continue
        start, end = int(row.session_id) + 1 - length, int(row.session_id) + 1
        positions = [keys.get((row.stock_code, session)) for session in range(start, end)]
        if any(position is None for position in positions):
            continue
        if features.trade_date.iloc[positions[-1]] != row.trade_date:
            raise ValueError("Sequence anchor does not agree with forecast date")
        indexes[row.split][0].append(target_pos)
        indexes[row.split][1].append(positions)
    return {fold: (np.asarray(targets, dtype=np.int64), np.asarray(rows, dtype=np.int64).reshape(-1, length))
            for fold, (targets, rows) in indexes.items()}


def fit_sequence_scaler(features, training_indexes):
    import numpy as np

    used = np.unique(training_indexes)
    if not len(used):
        raise ValueError("No eligible training sequence for normalization")
    training_features = features[used]
    if not np.isfinite(training_features).all():
        raise ValueError("Transformer training features must be finite")
    mean, scale = training_features.mean(axis=0), training_features.std(axis=0)
    return mean, np.where(scale > 0, scale, 1.0), len(used)


def prepare_sequences(dataset, output, length, batch_size):
    import numpy as np

    from ashare.p4_action_pipeline import atomic_json
    from ashare.p4_subset_inputs import hash_file, parquet_read
    from ashare.p4_subset_model import dataset_summary, checked_folds

    summary = dataset_summary(dataset)
    if length != summary["sequence_eligibility_length"]:
        raise ValueError("Sequence length must match the certified feature-window policy")
    labels = parquet_read(dataset / "training_dataset.parquet")
    checked_folds(labels, summary)
    features = parquet_read(dataset / "causal_inference_features.parquet")
    bundle = sequence_indexes(features, labels, length)
    if any(not len(targets) for targets, _ in bundle.values()):
        raise ValueError("No qualified Transformer sequences in at least one fold")
    columns = summary["feature_names"]
    mean, scale, used = fit_sequence_scaler(features[columns].to_numpy(dtype=float), bundle["train"][1])
    smoke = preflight(batch_size, length)
    output.mkdir(parents=True)
    scaler = output / "train_only_scaler.npz"
    np.savez(scaler, feature_names=np.asarray(columns), mean=mean, scale=scale)
    counts = {}
    for fold, (targets, indexes) in bundle.items():
        path = output / (fold + "_sequences.npz")
        np.savez_compressed(path, training_dataset_rows=targets, causal_feature_rows=indexes)
        counts[fold] = {"rows": len(targets), "sha256": hash_file(path)}
    record = {"status": "SEQUENCE_INPUTS_READY",
              "market_model_fitted": False, "sequence_length": length, "folds": counts, "gpu_preflight": smoke,
              "scaler": {"file": scaler.name, "fit_fold": "train", "unique_feature_rows": used, "sha256": hash_file(scaler)},
              "initial_training_config": {"model_width": 48, "attention_heads": 4, "encoder_layers": 2,
                                          "feedforward_width": 96, "batch_size": batch_size, "seed": 0,
                                          "objective": "smooth_l1_return", "optimizer": "AdamW", "learning_rate": 0.0003,
                                          "selection_metric": "validation_mean_daily_rank_ic", "early_stop_patience": 5,
                                          "max_epochs": 30, "bf16": smoke["bf16_supported"]},
              "dataset_audit_sha256": hash_file(dataset / "subset_audit.json"),
              "labels_sha256": hash_file(dataset / "training_dataset.parquet"),
              "causal_features_sha256": hash_file(dataset / "causal_inference_features.parquet"),
              "legacy_training_gate": "CLOSED"}
    atomic_json(output / "transformer_preflight.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--sequence", type=int, default=16)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if (args.dataset is None) != (args.out is None):
        parser.error("Sequence preparation requires both --dataset and --out")
    report = (prepare_sequences(args.dataset, args.out, args.sequence, args.batch) if args.dataset
              else preflight(args.batch, args.sequence))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
