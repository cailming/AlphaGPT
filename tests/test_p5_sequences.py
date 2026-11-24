"""Cover the distinct signal pool and contiguous context qualification contracts."""
import unittest

import numpy as np
import pandas as pd

from ashare.p4_mvp import FEATURES, PoolConfig
from ashare.p5_sequences import (VENDOR_REFERENCE, build_causal_population, price_window_flags, refusal_diagnostics,
                                 sequence_indexes)


def example_market():
    days = pd.bdate_range("2026-01-05", periods=80)
    raw = pd.DataFrame([
        {"stock_code": code, "trade_date": day, "open": 10., "high": 11., "low": 9., "close": 10.,
         "pre_close": 10., "volume": 100., "amount": amount, "not_billing": 1}
        for code, amounts in (("000001", np.full(80, 1500.)),
                              ("000002", np.repeat([1000., 4000.], [66, 14])))
        for day, amount in zip(days, amounts, strict=True)
    ])
    return days, raw


class CausalSequenceTests(unittest.TestCase):
    def test_signal_pool_does_not_cap_contiguous_context(self):
        days, raw = example_market()
        pop = build_causal_population(raw, days[60].date(), days[-1].date(), PoolConfig(top_n=1))
        anchor = pop.anchors.tail(1)
        self.assertEqual(anchor.stock_code.tolist(), ["000002"])
        v1, _ = sequence_indexes(pop.anchors, anchor)
        v2, rows = sequence_indexes(pop.features, anchor)
        self.assertEqual(v1.tolist(), [])
        self.assertEqual(v2.tolist(), [0])
        context = pop.features.iloc[rows[0]]
        self.assertEqual(context.stock_code.unique().tolist(), ["000002"])
        self.assertTrue(context.liquidity_rank.eq(2).any())
        self.assertTrue(context.session_id.diff().iloc[1:].eq(1).all())
        self.assertEqual(context.trade_date.iloc[-1], days[-1].strftime("%Y-%m-%d"))
        self.assertNotIn("target_return", context)
        extended = raw.copy()
        extended.loc[extended.trade_date.eq(days[-1]), "close"] = 100.
        same = build_causal_population(extended, days[60].date(), days[-2].date(), PoolConfig(top_n=1))
        before = build_causal_population(raw, days[60].date(), days[-2].date(), PoolConfig(top_n=1))
        pd.testing.assert_frame_equal(same.features, before.features)
        pd.testing.assert_frame_equal(same.anchors, before.anchors)

    def test_missing_context_is_never_filled_and_outcomes_are_rejected(self):
        days, raw = example_market()
        pop = build_causal_population(raw, days[60].date(), days[-1].date(), PoolConfig(top_n=1))
        anchor = pop.anchors.tail(1)
        gaps = pop.features.loc[~(pop.features.stock_code.eq("000002") & pop.features.session_id.eq(73))]
        idx, rows = sequence_indexes(gaps, anchor)
        self.assertEqual(idx.tolist(), [])
        self.assertEqual(rows.shape, (0, 16))
        with self.assertRaises(ValueError):
            sequence_indexes(pop.features.assign(target_return=0.), anchor)

    def test_known_action_blocks_only_when_visible_inside_the_feature_lookback(self):
        days, raw = example_market()
        end = days[-1].date()
        pop = build_causal_population(raw, days[50].date(), end, PoolConfig(top_n=1))
        anchor = pop.anchors.tail(1)
        actions = pd.DataFrame({"stock_code": ["000002"], "ex_date": [days[56].strftime("%Y-%m-%d")],
                                "published_on": [days[55].strftime("%Y-%m-%d")]})
        blocked, _ = sequence_indexes(pop.features, anchor, known_actions=actions, calendar=pop.calendar)
        self.assertEqual(blocked.tolist(), [])
        later, _ = sequence_indexes(pop.features, anchor, known_actions=actions.assign(published_on="2027-01-01"),
                                    calendar=pop.calendar)
        self.assertEqual(later.tolist(), [0])

    def test_per_stock_diagnosis_preserves_v1_reason_precedence(self):
        days, raw = example_market()
        pop = build_causal_population(raw, days[60].date(), days[-1].date(), PoolConfig(top_n=1))
        flags = price_window_flags(raw, pop.calendar)
        day = days[-1].strftime("%Y-%m-%d")
        _, [failure] = refusal_diagnostics(pop, flags, day, {"000002": 10}, 16)
        self.assertEqual(failure["v1_outcome"], "INSUFFICIENT_TRAINING_SEQUENCES")
        self.assertTrue(failure["outside_top_n_context_dates"])
        self.assertFalse(failure["v1_sequence_available"])
        self.assertTrue(failure["v2_floor_10_eligible"])
        self.assertFalse(failure["v2_floor_20_eligible"])
        _, [blocked] = refusal_diagnostics(pop, flags, day, {"000002": 20}, 16)
        self.assertEqual(blocked["v1_outcome"], "HISTORICAL_SEQUENCE_GAPS")
        self.assertTrue(blocked["v2_floor_20_eligible"])

    def test_quality_marks_real_jump_and_keeps_finite_features(self):
        days, raw = example_market()
        cut = raw.stock_code.eq("000002") & raw.trade_date.ge(days[70])
        raw.loc[cut, ["open", "high", "low", "close"]] /= 2
        pop = build_causal_population(raw, days[60].date(), days[-1].date(), PoolConfig(top_n=2))
        flag = price_window_flags(raw, pop.calendar).loc[lambda data: data.stock_code.eq("000002")].set_index("trade_date")
        self.assertEqual(flag.loc[days[70], "bar_status"], "UNEXPLAINED_PRICE_JUMP")
        self.assertFalse(flag.loc[days[-1], "valid_price_window"])
        self.assertFalse(pop.anchors.loc[lambda data: data.trade_date.eq(days[-1].strftime("%Y-%m-%d"))].stock_code.eq("000002").any())
        self.assertTrue(np.isfinite(pop.features[list(FEATURES)]).all().all())

    def test_auxiliary_reference_only_can_use_valid_contiguous_prior_close(self):
        days, raw = example_market()
        affected = raw.stock_code.eq("000002") & raw.trade_date.eq(days[57])
        raw.loc[affected, "pre_close"] = np.nan
        cfg = PoolConfig(top_n=2)
        start, end = days[48].date(), days[-1].date()
        strict = build_causal_population(raw, start, end, cfg, reference_policy=VENDOR_REFERENCE)
        repaired = build_causal_population(raw, start, end, cfg)
        anchor = strict.anchors.loc[strict.anchors.trade_date.eq(days[-1].strftime("%Y-%m-%d")) &
                                    strict.anchors.stock_code.eq("000002")]
        self.assertEqual(sequence_indexes(strict.features, anchor)[0].tolist(), [])
        accepted, indices = sequence_indexes(repaired.features, anchor)
        self.assertEqual(accepted.tolist(), [0])
        self.assertTrue(repaired.features.iloc[indices[0]].auxiliary_reference_fallback.any())
        self.assertTrue(raw.loc[affected, "pre_close"].isna().all())

        raw.loc[raw.stock_code.eq("000002") & raw.trade_date.eq(days[56]), "volume"] = 0
        flags = price_window_flags(raw, repaired.calendar, allow_auxiliary_fallback=True)
        candidate = flags.loc[flags.stock_code.eq("000002") & flags.trade_date.eq(days[57])].iloc[0]
        self.assertFalse(candidate.auxiliary_reference_fallback)
        self.assertFalse(candidate.valid_price_window)
