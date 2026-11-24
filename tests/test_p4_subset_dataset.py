"""Synthetic guards for the independent P4 subset contract."""
import unittest

import numpy as np
import pandas as pd

from ashare import p4_subset_dataset as subset
from ashare.p4_mvp import FEATURES, RidgeModel
from ashare.p4_subset_inputs import validate_factor_rows
from ashare.p4_subset_model import infer_rows
from ashare.p4_subset_transformer import fit_sequence_scaler, sequence_indexes


class SubsetQualityTests(unittest.TestCase):
    def evidence(self, known=None, conflicts=None, factors=None, issuer=None, support=True):
        return subset.Evidence({"000001": set(known or [])}, {"000001": set(conflicts or [])},
                               {"000001": {2025}} if support else {}, issuer or {},
                               {"000001": set(factors)} if factors is not None else {}, {})

    def assess(self, evidence):
        return subset.assess_window("000001", "2025-03-03", "2025-04-02", "2025-04-03", "2025-04-11", evidence)

    def test_empty_factor_result_needs_dated_provider_support(self):
        self.assertEqual(self.assess(self.evidence(factors=[], support=False))[0], subset.LIMITED)
        self.assertEqual(self.assess(self.evidence(factors=None))[0], subset.LIMITED)
        self.assertEqual(self.assess(self.evidence(factors=[]))[0], subset.TRUSTED)

    def test_official_actions_use_ex_date_boundaries_and_order_of_cash(self):
        entry, inside = "2025-04-03", "2025-04-07"
        issuer = {("000001", day): (2.0, 0.5, day) for day in (entry, inside)}
        grade, reason, payoff = self.assess(self.evidence(known=issuer_days(issuer), factors=[entry, inside], issuer=issuer))
        self.assertEqual((grade, reason), (subset.TRUSTED, "ISSUER_ADJUSTED"))
        self.assertEqual(payoff, [issuer[("000001", inside)]])
        self.assertAlmostEqual(subset.economic_return(10, 3, [(2, .5, "one"), (2, .25, "two")]), .3)
        self.assertEqual(self.assess(self.evidence(factors=[inside]))[1], "UNVERIFIED_ACTION_DURING_ENTRY_OR_HOLD")
        self.assertEqual(self.assess(self.evidence(factors=["2025-04-02"]))[1], "UNADJUSTED_FEATURE_ACTION_WINDOW")
        self.assertEqual(self.assess(self.evidence(conflicts=[inside], factors=[inside], issuer=issuer))[1],
                         "UNRESOLVED_ACTION_CONFLICT")

    def test_label_exit_on_boundary_is_purged(self):
        validation, test = "2025-07-01", "2025-10-01"
        self.assertEqual(subset.time_split("2025-06-30", validation, validation, test), "purged")
        self.assertEqual(subset.time_split("2025-09-30", test, validation, test), "purged")
        self.assertEqual(subset.time_split(validation, "2025-07-08", validation, test), "validation")

    def test_factor_request_rejects_wrong_symbol_and_outside_boundary(self):
        for code, date in (("sz.000002", "2025-03-01"), ("sz.000001", "2026-06-12")):
            with self.assertRaises(ValueError):
                validate_factor_rows([[code, date, "1", "1", "1"]], "sz.000001", "2025-01-02", "2026-06-11")

    def test_forward_quality_does_not_backfill_causal_candidates_or_emit_unsafe_labels(self):
        calendar = pd.bdate_range("2025-03-03", periods=60).strftime("%Y-%m-%d").tolist()
        raw = pd.DataFrame([{"stock_code": code, "trade_date": day, "open": 10., "close": 10.,
                             "high": 10., "low": 10., "volume": 100., "amount": 1000., "pre_close": 10.}
                            for code in ("000001", "000002") for day in calendar])
        candidates = pd.DataFrame([{"stock_code": code, "trade_date": calendar[40], "liquidity_rank": rank, "session_id": 40,
                                    **dict.fromkeys(FEATURES, .1)} for rank, code in enumerate(("000001", "000002"), 1)])
        manifest = {"calendar": calendar, "original_test_excluded": "2026-06-12", "val_start": calendar[50],
                    "test_start": calendar[55], "pool_config": {"top_n": 2, "hold_sessions": 5, "max_abs_label": .8}}
        result = subset.build_dataset(raw, candidates, manifest, self.evidence(factors=[], conflicts=[calendar[43]]))
        self.assertEqual(result.stock_code.tolist(), candidates.stock_code.tolist())
        self.assertEqual(result.sample_grade.tolist(), [subset.UNUSABLE, subset.LIMITED])
        self.assertEqual(result.causal_feature_ready.tolist(), [True, True])
        self.assertTrue(result.target_return.isna().all())

    def test_f3_inference_rejects_future_label_columns_and_before_end_date(self):
        model = RidgeModel(np.ones(10), np.zeros(10), np.ones(10), 0, 1)
        features = pd.DataFrame([{"stock_code": "000001", "trade_date": "2025-06-02", "liquidity_rank": 1,
                                  **dict.fromkeys(FEATURES, .1)}])
        with self.assertRaises(ValueError):
            infer_rows(model, features.assign(target_return=.5), "2025-06-02", "2025-05-28")
        with self.assertRaises(ValueError):
            infer_rows(model, features, "2025-06-02", "2025-06-02")
        self.assertAlmostEqual(infer_rows(model, features, "2025-06-02", "2025-05-28").predicted_economic_return.iloc[0], 1.)

    def test_transformer_sequences_never_jump_a_gap_or_read_after_anchor(self):
        features = pd.DataFrame([{"stock_code": "000001", "session_id": number, "trade_date": f"2025-06-{number:02d}"}
                                 for number in (1, 2, 3, 5, 6)])
        labels = pd.DataFrame([{"stock_code": "000001", "session_id": number, "trade_date": f"2025-06-{number:02d}",
                                "sequence_feature_evidence_ready": True, "split": "test"} for number in (3, 6)])
        indices = sequence_indexes(features, labels, 3)
        self.assertEqual(indices["test"][0].tolist(), [0])
        self.assertEqual(indices["test"][1].tolist(), [[0, 1, 2]])

    def test_transformer_normalization_only_uses_distinct_training_sequence_rows(self):
        data = np.array([[1., 1.], [3., 1.], [5., 1.], [999., 999.]])
        mean, scale, used = fit_sequence_scaler(data, np.array([[0, 1], [1, 2]]))
        np.testing.assert_allclose(mean, [3., 1.])
        np.testing.assert_allclose(scale, [np.sqrt(8/3), 1.])
        self.assertEqual(used, 3)


def issuer_days(issuer):
    return {day for _, day in issuer}
