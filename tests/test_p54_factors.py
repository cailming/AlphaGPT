"""Regression coverage for causal P5.4 market-context construction and pairing."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np
import pandas as pd

from ashare.p4_action_pipeline import atomic_json, read_json
from ashare.p4_mvp import FEATURES
from ashare.p4_subset_inputs import hash_file
from ashare.p54_factors import EXTRA_FEATURES, FACTOR_VERSION, P54_FEATURES, build_market_factors, freeze_market_variant


def market_fixture():
    days = pd.bdate_range("2026-01-05", periods=48)
    rows = []
    for i, code_options in enumerate((("000001", 1.005), ("000002", 0.998), ("000003", 1.012))):
        code, multiplier = code_options
        for j, day in enumerate(days):
            close = (20 + i) * multiplier**j
            rows.append((code, day, close, close*1.01, close*.99, close, 100.0, (1000+i)*1.01**j, 1))
    bars = pd.DataFrame(rows, columns=["stock_code", "trade_date", "open", "high", "low", "close",
                                       "volume", "amount", "not_billing"])
    return bars, days


class MarketFactorTests(TestCase):
    def test_formal_trend_sentiment_activity_relative_strength_and_breakout(self):
        raw, days = market_fixture()
        keys = pd.DataFrame({"stock_code": ["000002", "000001"], "trade_date": [days[35], days[34]]})
        factors, audit = build_market_factors(raw, keys, min_market_stocks=3)
        self.assertEqual(factors.stock_code.tolist(), ["000002", "000001"])
        self.assertAlmostEqual(factors.market_ret_5.iloc[0], 1.005**5 - 1)
        self.assertAlmostEqual(factors.market_ret_20.iloc[0], 1.005**20 - 1)
        self.assertAlmostEqual(factors.market_advancer_share_5.iloc[0], 2/3)
        self.assertAlmostEqual(factors.market_turnover_log_change_20.iloc[0], 20 * np.log(1.01))
        self.assertAlmostEqual(factors.stock_relative_ret_20.iloc[0], 0.998**20 - 1.005**20)
        self.assertAlmostEqual(factors.stock_breakout_20.iloc[1], 1.005 / 1.01 - 1)
        self.assertEqual(audit["all_extra_factors_finite_rows"], 2)
        self.assertEqual(audit["last_source_session"], days[35].strftime("%Y-%m-%d"))

    def test_dry_pullback_requires_lower_volume_and_uptrend(self):
        raw, days = market_fixture()
        code = "000003"
        mask = raw.stock_code.eq(code)
        prior = raw.loc[mask & raw.trade_date.eq(days[40]), "close"].item()
        current = raw.index[mask & raw.trade_date.eq(days[41])][0]
        raw.loc[current, ["open", "high", "low", "close", "volume"]] = [
            prior*.98, prior*.985, prior*.975, prior*.98, 40.0
        ]
        keys = pd.DataFrame({"stock_code": [code, code], "trade_date": [days[40], days[41]]})
        factors, _ = build_market_factors(raw, keys, min_market_stocks=3)
        self.assertEqual(factors.stock_dry_volume_pullback_5.iloc[0], 0)
        self.assertAlmostEqual(factors.stock_volume_dryup_20.iloc[1], .6)
        self.assertAlmostEqual(factors.stock_dry_volume_pullback_5.iloc[1], .012)
        raw.loc[current, "volume"] = 110.0
        higher, _ = build_market_factors(raw, keys, min_market_stocks=3)
        self.assertEqual(higher.stock_dry_volume_pullback_5.iloc[1], 0)

    def test_future_dates_and_unsettled_billing_do_not_change_earlier_factors(self):
        raw, days = market_fixture()
        keys = pd.DataFrame({"stock_code": ["000001", "000001"], "trade_date": [days[32], days[42]]})
        first, _ = build_market_factors(raw, keys.iloc[:1], min_market_stocks=3)
        changed = raw.copy()
        changed.loc[changed.trade_date.gt(days[32]), "amount"] *= 1000
        unfinalized = raw.iloc[:4].assign(not_billing=0, close=10000)
        with_future, _ = build_market_factors(pd.concat([changed, unfinalized]), keys, min_market_stocks=3)
        pd.testing.assert_frame_equal(first.reset_index(drop=True), with_future.iloc[:1].reset_index(drop=True))

    def test_gaps_anomalies_and_low_market_overlap_do_not_impute_factors(self):
        raw, days = market_fixture()
        missing = raw.loc[~(raw.stock_code.eq("000001") & raw.trade_date.eq(days[30]))].copy()
        keys = pd.DataFrame({"stock_code": ["000001", "000002"], "trade_date": [days[34], days[34]]})
        output, audit = build_market_factors(missing, keys, min_market_stocks=2)
        self.assertTrue(np.isnan(output.stock_breakout_20.iloc[0]))
        self.assertTrue(np.isfinite(output.stock_breakout_20.iloc[1]))
        self.assertTrue(np.isnan(output.market_ret_20).all())
        self.assertEqual(audit["all_extra_factors_finite_rows"], 0)
        indexes = raw.index[raw.stock_code.eq("000001") & raw.trade_date.eq(days[30])]
        raw.loc[indexes, ["open", "high", "low", "close"]] *= 1.6
        after_jump, _ = build_market_factors(raw, keys.iloc[:1], min_market_stocks=2)
        self.assertTrue(np.isnan(after_jump.stock_relative_ret_20.iloc[0]))

    def test_label_columns_and_duplicate_formal_keys_are_rejected(self):
        raw, days = market_fixture()
        keys = pd.DataFrame({"stock_code": ["000001"], "trade_date": [days[34]]})
        with self.assertRaisesRegex(ValueError, "feature-only"):
            build_market_factors(raw, keys.assign(target_return=99), min_market_stocks=2)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_market_factors(pd.concat([raw, raw.iloc[[1]]]), keys, min_market_stocks=2)


class PairedVariantTests(TestCase):
    def prepare_fixture(self, root):
        raw, days = market_fixture()
        input_dir, data, sequence = root / "inputs", root / "dataset", root / "transformer"
        for path in [input_dir, data, sequence]:
            path.mkdir(parents=True)
        latest = raw.loc[raw.stock_code.eq("000001") & raw.trade_date.ge(days[25])].reset_index(drop=True)
        features = latest[["stock_code", "trade_date"]].assign(session_id=np.arange(25, len(days)))
        for index, name in enumerate(FEATURES):
            features[name] = np.arange(len(features), dtype=float) + index
        raw_file = input_dir / "formal_ohlcv.parquet"
        feature_file = data / "causal_inference_features.parquet"
        label_file = data / "training_dataset.parquet"
        for path in (raw_file, feature_file, label_file):
            path.write_text(path.name)
        folds = {}
        for i, fold in enumerate(("train", "validation", "test")):
            path = sequence / f"{fold}_sequences.npz"
            np.savez(path, causal_feature_rows=np.array([[i, i+1]], dtype=np.int64),
                     training_dataset_rows=np.array([i], dtype=np.int64))
            folds[fold] = {"rows": 1, "sha256": hash_file(path)}
        atomic_json(data / "subset_audit.json", {
            "version": "P5_ROLLING_GATED_ECONOMIC_V2", "feature_names": list(FEATURES),
            "subset_training_allowed": True, "sequence_eligibility_length": 2,
            "frozen_price_sha256": hash_file(raw_file),
            "files": {p.name: hash_file(p) for p in (feature_file, label_file)},
        })
        atomic_json(sequence / "transformer_preflight.json", {
            "dataset_audit_sha256": hash_file(data / "subset_audit.json"), "status": "SEQUENCE_INPUTS_READY",
            "sequence_length": 2, "scaler": {"fit_fold": "train"}, "folds": folds,
            "initial_training_config": {"model_width": 48, "nhead": 4},
            "causal_features_sha256": hash_file(feature_file), "labels_sha256": hash_file(label_file),
        })
        return {raw_file: raw, feature_file: features}, label_file

    def test_exact_paired_immutability_preserves_training_labels_and_sequence_digest(self):
        with TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "source", Path(tmp) / "expanded"
            fixtures, labels = self.prepare_fixture(root)
            with mock.patch("ashare.p54_factors.parquet_read", side_effect=fixtures.__getitem__), mock.patch(
                    "ashare.p54_factors.parquet_write", side_effect=lambda frame, path: frame.to_csv(path, index=False)):
                audit = freeze_market_variant(root, out, min_market_stocks=3)
            self.assertTrue(audit["identical_train_validation_and_test_anchors"])
            self.assertEqual(audit["feature_names"], list(P54_FEATURES))
            self.assertEqual(read_json(out / "dataset/subset_audit.json")["version"], FACTOR_VERSION)
            self.assertEqual(hash_file(labels), hash_file(out / "dataset/training_dataset.parquet"))
            for fold in audit["fold_coverage"]:
                path = f"transformer/{fold}_sequences.npz"
                self.assertEqual(hash_file(root / path), hash_file(out / path))
            with np.load(out / "transformer/train_only_scaler.npz", allow_pickle=False) as scaler:
                self.assertEqual(len(scaler["mean"]), len(FEATURES) + len(EXTRA_FEATURES))
                np.testing.assert_allclose(scaler["mean"][:len(FEATURES)], np.arange(len(FEATURES)) + .5)

    def test_incomplete_context_keeps_rejection_audit_without_cherry_picking(self):
        with TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "source", Path(tmp) / "expanded"
            fixtures, _ = self.prepare_fixture(root)
            with mock.patch("ashare.p54_factors.parquet_read", side_effect=fixtures.__getitem__):
                with self.assertRaisesRegex(ValueError, "Incomplete expanded factors"):
                    freeze_market_variant(root, out, min_market_stocks=4)
            audit = read_json(out / "factor_audit.json")
            self.assertEqual(audit["status"], "REJECTED_INCOMPLETE_PAIRED_WINDOWS")
            self.assertEqual(audit["fold_coverage"]["test"]["all_inputs_finite_sequences"], 0)
            self.assertFalse((out / "transformer").exists())

    def test_filtered_control_and_expanded_inputs_share_all_label_and_index_bytes(self):
        with TemporaryDirectory() as tmp:
            root, out, control = (Path(tmp) / name for name in ("source", "expanded", "control"))
            fixtures, labels = self.prepare_fixture(root)
            days = pd.bdate_range("2026-01-05", periods=48)
            raw_file = root / "inputs/formal_ohlcv.parquet"
            fixtures[raw_file] = fixtures[raw_file].loc[~(fixtures[raw_file].stock_code.eq("000001")
                                                        & fixtures[raw_file].trade_date.eq(days[43]))]
            fixtures[labels] = pd.DataFrame({"stock_code": ["000001"]*4,
                                             "split": ["train", "validation", "test", "test"],
                                             "trade_date": ["2026-02-10", "2026-02-12", "2026-02-17", "2026-03-05"],
                                             "exit_date": ["2026-02-16", "2026-02-18", "2026-02-24", "2026-03-11"],
                                             "target_return": [.123, -.098, .109, -.77]})
            seq_file = root / "transformer/test_sequences.npz"
            np.savez(seq_file, causal_feature_rows=np.array([[3, 4], [17, 18]], dtype=np.int64),
                     training_dataset_rows=np.array([2, 3], dtype=np.int64))
            preflight_path = root / "transformer/transformer_preflight.json"
            preflight = read_json(preflight_path)
            preflight["folds"]["test"] = {"rows": 2, "sha256": hash_file(seq_file)}
            atomic_json(preflight_path, preflight)
            with mock.patch("ashare.p54_factors.parquet_read", side_effect=fixtures.__getitem__), mock.patch(
                    "ashare.p54_factors.parquet_write", side_effect=lambda frame, path: frame.to_csv(path, index=False)):
                audit = freeze_market_variant(root, out, min_market_stocks=3, paired_control=control)
            self.assertEqual(audit["status"], "FILTERED_PAIRED_FEATURES_READY")
            self.assertEqual(audit["fold_coverage"]["test"]["lost_sequences"], 1)
            label_path = "dataset/training_dataset.parquet"
            self.assertEqual(hash_file(out / label_path), hash_file(control / label_path))
            self.assertEqual(pd.read_csv(out / label_path).target_return.tolist(), [.123, -.098, .109])
            self.assertEqual(read_json(control / "dataset/subset_audit.json")["splits"]["test"]["samples"], 1)
            for fold in ("train", "validation", "test"):
                path = f"transformer/{fold}_sequences.npz"
                self.assertEqual(hash_file(out / path), hash_file(control / path))
