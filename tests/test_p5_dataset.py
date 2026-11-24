"""Rolling date, eligibility, and immutable action extension regressions."""
from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ashare import p4_action_pipeline as source
from ashare.p4_mvp import PoolConfig
from ashare.p4_subset_inputs import hash_file
from ashare.p5_dataset import (INPUTS_VERSION, VERSION, Boundaries, WindowEvidence, export_live_evidence,
                               extend_evidence, label_windows, prepare_live_evidence, read_recent_evidence, reframe)


class RollingTests(unittest.TestCase):
    def test_roll_and_purge_until_label_matures(self):
        bounds = Boundaries.rolling("2026-11-23")
        self.assertEqual((bounds.train_start, bounds.val_start, bounds.test_start, bounds.price_end),
                         ("2025-08-23", "2026-08-23", "2026-10-23", "2026-11-23"))
        self.assertEqual(bounds.split("2026-08-13", "2026-08-21"), "train")
        self.assertIsNone(bounds.split("2026-08-13", "2026-08-24"))
        self.assertEqual(bounds.split("2026-10-13", "2026-10-22"), "validation")
        self.assertIsNone(bounds.split("2026-10-13", "2026-10-23"))
        self.assertIsNone(bounds.split("2026-11-19", "2026-11-24"))
        with self.assertRaises(ValueError):
            Boundaries("2026-01-01", "2026-02-01", "2026-02-01", "2026-03-01")

    def test_v1_success_never_certifies_later_missing_observation(self):
        legacy = SimpleNamespace(factor_days={"000001": set()}, provider_years={"000001": {2025, 2026}})
        available = WindowEvidence(legacy, "2025-01-02", "2026-06-12", "2026-11-23", {"000001"}, {}, {})
        unavailable = WindowEvidence(legacy, "2025-01-02", "2026-06-12", "2026-11-23", set(), {}, {})
        self.assertTrue(available.covered("000001", "2026-05-01", "2026-06-30"))
        self.assertFalse(unavailable.covered("000001", "2026-05-01", "2026-06-30"))
        self.assertFalse(available.covered("000001", "2026-11-20", "2026-11-24"))
        self.assertFalse(available.covered("000001", "2025-01-01", "2025-01-12"))

    def test_missing_auxiliary_reference_is_distinct_from_business_action(self):
        calendar = pd.bdate_range("2026-05-04", periods=100)
        iso = calendar.strftime("%Y-%m-%d")
        raw = pd.DataFrame({"stock_code": "000001", "trade_date": iso, "open": 10., "close": 10.,
                            "low": 10., "high": 10., "volume": 100., "amount": 1000., "pre_close": 10.,
                            "not_billing": 1})
        raw.loc[67, "pre_close"] = np.nan
        anchors = pd.DataFrame([{"stock_code": "000001", "trade_date": iso[64], "session_id": 64}])
        bounds = Boundaries(iso[32], iso[74], iso[88], iso[-1])
        legacy = SimpleNamespace(actions={}, conflicts={}, factor_days={"000001": set()},
                                 provider_years={"000001": {2026}}, verified={})
        accepted = WindowEvidence(legacy, "2025-01-02", "2026-06-12", iso[-1], {"000001"}, {}, {})
        labels, indexes, quality = label_windows(raw, anchors, np.array([0]), calendar, bounds, PoolConfig(), accepted)
        self.assertEqual((len(labels), indexes.tolist(), quality.reason.tolist()), (1, [0], ["OBSERVED_CLEAN_WINDOW"]))
        self.assertEqual(labels.target_return.iloc[0], 0)
        accepted.recent_events = {"000001": {iso[67]}}
        rejected, _, quality = label_windows(raw, anchors, np.array([0]), calendar, bounds, PoolConfig(), accepted)
        self.assertTrue(rejected.empty)
        self.assertEqual(quality.reason.tolist(), ["UNVERIFIED_ACTION_IN_LABEL_WINDOW"])

    def test_live_observation_jobs_are_completion_bounded_and_disjoint(self):
        with tempfile.TemporaryDirectory() as root:
            earlier = Path(root) / "existing"
            inputs = earlier / "inputs"
            inputs.mkdir(parents=True)
            source.atomic_json(inputs / "evidence_plan.json", {"jobs": [{"stock_code": "000001"}]})
            old_digest = hash_file(inputs / "evidence_plan.json")
            source.atomic_json(inputs / "inputs.json", {"files": {"evidence_plan.json": old_digest}})
            candidates = Path(root) / "candidates.csv"
            pd.DataFrame({"stock_code": ["000001", "000002", "000003"],
                          "trade_date": ["2026-11-23"]*3, "liquidity_rank": [1, 2, 3],
                          "v2_price_sequence_available": [True, True, False]}).to_csv(candidates, index=False)
            fresh = Path(root) / "supplement"
            with patch("ashare.p5_dataset.verify_formal_cutoff"):
                prepare_live_evidence(fresh, earlier, candidates, "2026-11-23", 120)
            plan, supported, _, audit = read_recent_evidence(fresh / "inputs")
            self.assertEqual([j["stock_code"] for j in plan["jobs"]], ["000002"])
            self.assertEqual(supported, set())
            self.assertEqual(audit["recent_counts"], {"bao_adjust/NOT_CAPTURED": 1, "sina_dividend/NOT_CAPTURED": 1})
            self.assertLess(date.fromisoformat(plan["start"]), date.fromisoformat(plan["end"]))
            self.assertEqual(hash_file(inputs / "evidence_plan.json"), old_digest)

    def test_increment_never_bridges_vendor_failure_and_keeps_the_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            earlier = Path(root) / "existing"
            inputs, dataset = earlier / "inputs", earlier / "dataset"
            inputs.mkdir(parents=True)
            dataset.mkdir()
            source.atomic_json(inputs / "evidence_plan.json", {"jobs": [{"stock_code": "000001"}, {"stock_code": "000002"}]})
            old_path = dataset / "inference_evidence.json"
            old_stock = {"covered_from": "2026-09-01", "covered_through": "2026-11-23", "actions": []}
            base = {"policy": "P5_DUAL_OBSERVED_RESEARCH_WINDOW_V1", "observed_at_utc": "2026-11-23T13:00:00Z",
                    "max_covered_date": "2026-11-23", "stocks": {stock: old_stock for stock in ("000001", "000002")}}
            source.atomic_json(old_path, base)
            old_digest = hash_file(old_path)
            source.atomic_json(dataset / "subset_audit.json", {"version": VERSION, "files": {old_path.name: old_digest}})
            refresh = Path(root) / "refresh"
            with patch("ashare.p5_dataset.verify_formal_cutoff"):
                extend_evidence(refresh, earlier, "2026-11-24", 220)
            newer = refresh / "inputs"
            plan = source.read_json(newer / "evidence_plan.json")
            plan_sha = hash_file(newer / "evidence_plan.json")
            self.assertEqual(plan["start"], "2026-11-24")
            observations = newer / "observations"
            observations.mkdir()
            for provider in plan["providers"]:
                item = {"job": plan["jobs"][0], "provider": provider, "status": "SUCCESS",
                        "start": plan["start"], "end": plan["end"], "rows": [],
                        "observed_at_utc": "2026-11-24T13:00:00Z", "plan_sha256": plan_sha}
                source.atomic_json(observations / f"000001.{provider}.json",
                                   {"observation": item, "sha256": hashlib.sha256(source.canonical(item)).hexdigest()})
            export_live_evidence(refresh, earlier)
            outcome = source.read_json(refresh / "dataset/inference_evidence.json")
            self.assertEqual(outcome["stocks"]["000001"]["covered_through"], "2026-11-24")
            self.assertEqual(outcome["stocks"]["000002"]["covered_through"], "2026-11-23")
            self.assertEqual(hash_file(old_path), old_digest)

    def test_reframe_preserves_the_shared_inputs_and_other_cutoffs(self):
        from dataclasses import asdict

        with tempfile.TemporaryDirectory() as root:
            prior = Path(root) / "base"
            inputs = prior / "inputs"
            inputs.mkdir(parents=True)
            source.atomic_json(inputs / "evidence_plan.json", {"jobs": []})
            (inputs / "formal_ohlcv.parquet").write_bytes(b"test fixture")
            (inputs / "observations").mkdir()
            original = {"version": INPUTS_VERSION, "bounds": asdict(Boundaries.rolling("2026-11-23")),
                        "files": {name: hash_file(inputs / name) for name in ("evidence_plan.json", "formal_ohlcv.parquet")}}
            source.atomic_json(inputs / "inputs.json", original)
            current = Path(root) / "permitted"
            reframe(current, prior, "2026-09-23")
            revised = current / "inputs"
            self.assertEqual(source.read_json(revised / "inputs.json"), original | {
                "bounds": original["bounds"] | {"test_start": "2026-09-23"}})
            self.assertEqual(source.read_json(inputs / "inputs.json"), original)
            self.assertTrue((revised / "observations").is_symlink())
            self.assertEqual((revised / "formal_ohlcv.parquet").resolve(), (inputs / "formal_ohlcv.parquet").resolve())
