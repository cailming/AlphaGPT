"""Checks fixed-frame support comparisons and fail-closed latest scoring."""
from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from ashare.p4_mvp import FEATURES
from ashare.p5_infer import floor_summary, qualification, select_latest, validation
from ashare.p5_sequences import CausalPopulation


class SupportFloorTests(unittest.TestCase):
    def test_fixed_scores_and_pool_across_support_floors(self):
        rows = pd.DataFrame(
            [(str(i), day, float(i * .01), float(i * .01), "validation")
             for day in ("2026-09-02", "2026-09-03") for i in range(8)
             if day == "2026-09-02" or i < 2],
            columns=("stock_code", "trade_date", "target_return", "score", "split"))
        reports, by_day = floor_summary(rows, {str(i): i * 5 for i in range(1, 8)})
        self.assertEqual([reports[key]["samples"] for key in ("0", "5", "10", "20")], [10, 8, 6, 4])
        self.assertEqual(reports["20"]["days_with_no_predictions"], 1)
        self.assertEqual(reports["20"]["daily_median_rows"], 2.0)
        self.assertAlmostEqual(reports["20"]["top10_excess_over_fixed_daily_pool"], 0.02)
        self.assertEqual(by_day["floor_20_rows"].tolist(), [4, 0])
        self.assertAlmostEqual(reports["0"]["mean_daily_rank_ic"], 1.0)

    def test_rejects_other_folds_duplicate_rows_and_nonfinite_scores(self):
        frame = pd.DataFrame(
            [("000001", "2026-09-02", 0.03, 0.02, "validation")],
            columns=("stock_code", "trade_date", "target_return", "score", "split"))
        for bad in (frame.assign(split="test"), pd.concat([frame, frame]), frame.assign(score=float("nan"))):
            with self.assertRaises(ValueError):
                floor_summary(bad, {})

    def test_empty_positive_floors_have_no_metrics(self):
        frame = pd.DataFrame([("000001", "2026-09-02", 0.03, 0.02)],
                             columns=("stock_code", "trade_date", "target_return", "score"))
        reports, _ = floor_summary(frame, {})
        self.assertIsNone(reports["5"]["direction_accuracy"])
        self.assertIsNone(reports["5"]["top20_excess_over_fixed_daily_pool"])
        self.assertEqual(reports["5"]["coverage"], 0)

    def test_validation_can_start_on_the_exclusive_training_cutoff(self):
        frame = pd.DataFrame([("000001", "2026-09-02", 0.03, 0.02)],
                             columns=("stock_code", "trade_date", "target_return", "score"))
        info = dict(objective="absolute", validation_predictions_sha256="validated-digest", weights_sha256="weights",
                    validation_first_signal="2026-09-02", validation_last_signal="2026-09-02",
                    fit_cutoff="2026-09-02", fit_cutoff_is_exclusive=True, version="P5", v2_sequence_policy="V2",
                    train_stock_counts={})
        with (TemporaryDirectory() as folder, patch("ashare.p5_infer.read_json", return_value=info),
              patch("ashare.p5_infer.parquet_read", return_value=frame),
              patch("ashare.p4_transformer_train.verify_digest")):
            self.assertEqual(validation(Path(folder), Path(folder) / "valid")["rows"], 1)
            info["fit_cutoff_is_exclusive"] = False
            with self.assertRaises(ValueError):
                validation(Path(folder), Path(folder) / "rejected")


class LiveCandidateTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2026-08-03", periods=60)
        self.signal = self.calendar[-1].date().isoformat()
        frame = pd.DataFrame(
            [(f"{i:06d}", date.date().isoformat(), j, 1200 if j < 59 else i + 1, *([.1] * len(FEATURES)))
             for i in range(5) for j, date in enumerate(self.calendar)
             if not (i == 4 and j == 57)],
            columns=("stock_code", "trade_date", "session_id", "liquidity_rank", *FEATURES))
        self.population = CausalPopulation(frame, frame.loc[frame.trade_date.eq(self.signal)].copy(),
                                           frame, self.calendar, {})
        full = {"covered_from": "2026-08-03", "covered_through": self.signal, "actions": []}
        action_date = self.calendar[-5].date().isoformat()
        self.evidence = {"max_covered_date": self.signal, "stocks": {
            "000000": full,
            "000001": {**full, "actions": [[action_date, action_date]]},
            "000002": {**full, "covered_through": action_date},
            "000003": {**full, "actions": [[action_date, "2026-11-01"]]},
            "000004": full,
        }}
        self.info = {"architecture": {"sequence_length": 16}, "support_floor": 0,
                     "train_stock_counts": {"000000": 6, "000003": 20}}

    def test_uncapped_history_and_observation_windows(self):
        details, positions, idx, counts = select_latest(self.population, self.signal, self.info, self.evidence)
        results = details.set_index("stock_code").outcome.to_dict()
        self.assertEqual(results, {
            "000000": "SCORED",
            "000001": "COMPANY_ACTION_INSIDE_CONTEXT",
            "000002": "CAUSAL_ACTION_EVIDENCE_WINDOW_UNAVAILABLE",
            "000003": "SCORED",
            "000004": "NO_CONTIGUOUS_FEATURE_SEQUENCE",
        })
        self.assertEqual(details.iloc[positions].stock_code.tolist(), ["000000", "000003"])
        self.assertEqual(idx.shape, (2, 16))
        self.assertEqual(counts, {"0": 2, "5": 2, "10": 1, "20": 1})
        self.assertGreater(self.population.features.iloc[idx[0][0]].liquidity_rank, 800)

    def test_support_floor_abstains_for_otherwise_valid_candidate(self):
        self.info["support_floor"] = 20
        details, positions, _, _ = select_latest(self.population, self.signal, self.info, self.evidence)
        self.assertEqual(details.iloc[positions].stock_code.tolist(), ["000003"])
        self.assertEqual(details.set_index("stock_code").loc["000000", "outcome"], "INSUFFICIENT_TRAINING_SEQUENCES")

    def test_requires_coverage_of_entire_feature_lookback(self):
        self.evidence["stocks"]["000000"]["covered_from"] = self.calendar[-35].date().isoformat()
        details, _, _, _ = select_latest(self.population, self.signal, self.info, self.evidence)
        self.assertEqual(details.set_index("stock_code").loc["000000", "outcome"],
                         "CAUSAL_ACTION_EVIDENCE_WINDOW_UNAVAILABLE")

    def test_records_fallback_feature_dates_by_stock(self):
        feature_date = self.calendar[-3].date().isoformat()
        self.population.features["auxiliary_reference_fallback"] = (
            self.population.features.stock_code.eq("000000") & self.population.features.trade_date.eq(feature_date))
        details, _, _, _ = select_latest(self.population, self.signal, self.info, self.evidence)
        audits = details.set_index("stock_code")
        self.assertEqual(audits.loc["000000", "auxiliary_reference_fallback_feature_dates"], feature_date)
        self.assertEqual(audits.loc["000000", "auxiliary_reference_fallback_feature_sessions"], 1)
        self.assertEqual(audits.loc["000003", "auxiliary_reference_fallback_feature_sessions"], 0)


class NoModelQualificationTests(unittest.TestCase):
    def test_uses_only_frozen_training_fold_counts(self):
        train = pd.DataFrame({"stock_code": ["000001"] * 4 + ["000002"] * 2})
        details = pd.DataFrame({"stock_code": ["000001"], "outcome": ["SCORED"]})
        data = SimpleNamespace(config={"sequence_length": 16}, audit={
            "feature_names": list(FEATURES), "pool": {"pool_config": {}}, "v2_sequence_policy": "V2",
            "splits": {"validation": {"last_label_exit": "2026-09-02"}}},
            audit_sha256="audit", fold=Mock(return_value=(train, None, None)))
        with (TemporaryDirectory() as folder, patch("ashare.p5_train.open_p5_dataset", return_value=data),
              patch("ashare.p5_infer.live_population", return_value=(
                  None, date(2026, 9, 3), None, details, [0], None, {"qualified_by_support_floor": {"0": 1}},
              )) as live):
            result = qualification(Path(folder), Path(folder) / "qualification")
        self.assertEqual(result["qualified"], 1)
        self.assertFalse(result["prediction_performed"])
        self.assertEqual(result["candidate_outcomes"], {"QUALIFIED_FOR_SCORING": 1})
        self.assertEqual(live.call_args.args[2]["train_stock_counts"], {"000001": 4, "000002": 2})
        data.fold.assert_called_once_with("train")


if __name__ == "__main__":
    unittest.main()
