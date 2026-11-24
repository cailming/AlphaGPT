"""Regression checks for causal sequences, comparison, and resumable model selection."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from ashare.p4_mvp import FEATURES, PoolConfig
    from ashare.p4_transformer_evaluate import diagnostics
    from ashare.p4_transformer_infer import inference_indexes, price_quality_keys
    from ashare.p4_transformer_train import (ARCHITECTURE, EVALUATION_PLAN, _fit, atomic_torch,
                                             load_model, make_model, predict)


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class TransformerTests(unittest.TestCase):
    def test_network_prefix_is_causal(self):
        torch.manual_seed(0)
        model = make_model(dict(ARCHITECTURE, sequence_length=16, feature_count=10)).eval()
        first = torch.zeros(2, 16, 10)
        changed = first.clone()
        changed[:, -1] = 2
        with torch.inference_mode():
            torch.testing.assert_close(model.encode(first)[:, :-1], model.encode(changed)[:, :-1])
            self.assertEqual(tuple(model(first).shape), (2,))

    def test_inference_cannot_fill_a_gap_or_check_future_outcomes(self):
        rows = pd.DataFrame([dict(stock_code=stock, session_id=i, liquidity_rank=rank,
                                  trade_date=f"2026-04-{i:02d}", **dict.fromkeys(FEATURES, 1.0))
                             for stock, rank, sessions in (("000001", 1, [1, 2, 3, 4]),
                                                           ("000002", 2, [1, 3, 4]),
                                                           ("000003", 3, [1, 2, 3, 4])) for i in sessions])
        picks, _, sequence, counts = inference_indexes(rows, "2026-04-03", 3,
                                                        {"000001": 22, "000002": 20}, 20)
        self.assertEqual(picks.stock_code.tolist(), ["000001"])
        self.assertEqual(sequence.tolist(), [[0, 1, 2]])
        self.assertEqual(counts, {"causal_candidates_on_date": 3, "sequence_gaps": 1,
                                  "insufficient_training_sequences": 1})
        with self.assertRaises(ValueError):
            inference_indexes(rows.assign(target_return=.1), "2026-04-03", 3, {}, 20)

    def test_price_guard_excludes_a_jump_without_using_later_bars(self):
        calendar = pd.bdate_range("2026-01-01", periods=44)
        raw = pd.DataFrame(dict(stock_code="000001", trade_date=calendar, open=10., close=10., low=10.,
                                high=10., volume=100., amount=1000., pre_close=10.))
        raw.loc[22:, ["open", "high", "low", "close"]] = 5.
        ready = price_quality_keys(raw, calendar)
        self.assertIn(calendar[21], ready.trade_date.tolist())
        self.assertNotIn(calendar[22], ready.trade_date.tolist())
        self.assertNotIn(calendar[42], ready.trade_date.tolist())
        self.assertIn(calendar[43], ready.trade_date.tolist())

    def test_common_sample_diagnostics_account_for_ties_cost_and_replacement(self):
        frame = pd.DataFrame([
            {"stock_code": code, "trade_date": day, "target_return": .02 if code == chosen else .0}
            for day, chosen in (("2026-03-23", "001"), ("2026-03-24", "002")) for code in ("001", "002")
        ])
        plan = EVALUATION_PLAN | {"top_k": [1]}
        result, daily = diagnostics(frame, np.array([.01, .0, .0, .01]), .005, plan)
        self.assertEqual(result["confidence"]["test_coverage"], .5)
        self.assertAlmostEqual(result["top1"]["mean_net_excess_over_gross_pool_by_per_side_cost_bps"]["10"], .008)
        self.assertEqual(daily[1]["top1_lists_replaced"], 1.0)
        constant, _ = diagnostics(frame, np.ones(4), 1., plan)
        self.assertIsNone(constant["mean_daily_rank_ic"])
        self.assertEqual(constant["top1"]["mean_list_replacement"], 0.)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required")
    def test_epoch_checkpoint_resume_equals_uninterrupted_fit(self):
        rng = np.random.default_rng(17)
        config = dict(ARCHITECTURE, sequence_length=16, feature_count=10, seed=0, batch_size=16,
                      max_epochs=3, early_stop_patience=5, learning_rate=.0003)
        train_x = rng.normal(size=(64, 16, 10)).astype(np.float32)
        evaluation_x = rng.normal(size=(32, 16, 10)).astype(np.float32)

        def fold(name):
            if name not in ("train", "validation"):
                self.fail("Training requested a historical test fold")
            x = train_x if name == "train" else evaluation_x
            y = x[:, -1, 0] / 20
            day, exit_day = ("2025-06-01", "2025-06-09") if name == "train" else ("2026-01-05", "2026-01-13")
            return pd.DataFrame(dict(stock_code=[f"{i:06d}" for i in range(len(x))], trade_date=day,
                                     exit_date=exit_day, target_return=y)), x, y

        audit = dict(version="fixture", old_test_excluded="2026-06-12", test_start="2026-03-23",
                     test_kind="synthetic", pool={"pool_config": PoolConfig().__dict__})
        data = SimpleNamespace(fold=fold, config=config, audit_sha256="x", preflight_sha256="y", audit=audit,
                               mean=np.zeros(10), scale=np.ones(10))
        interrupted = False

        def abort_after_first_epoch(path, contents):
            nonlocal interrupted
            atomic_torch(path, contents)
            if path.name == "checkpoint.pt" and len(contents["history"]) == 1 and not interrupted:
                interrupted = True
                raise InterruptedError("Test interruption after first atomic checkpoint")

        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "resumed", Path(folder) / "reference"
            first.mkdir()
            second.mkdir()
            with patch("ashare.p4_transformer_train.atomic_torch", side_effect=abort_after_first_epoch):
                with self.assertRaises(InterruptedError):
                    _fit(data, first, "fp32", False)
            _fit(data, first, "fp32", True)
            _fit(data, second, "fp32", False)
            gpu = torch.device("cuda:0")
            resumed, meta = load_model(first, gpu)
            reference, expected = load_model(second, gpu)
            self.assertEqual(len(meta["epochs"]), 3)
            self.assertEqual(meta["best_epoch"], expected["best_epoch"])
            np.testing.assert_array_equal(predict(resumed, evaluation_x, gpu), predict(reference, evaluation_x, gpu))
