from datetime import date

import pandas as pd
import unittest

from ashare.p53_dataset import audit_inference, audit_quality, merge_intervals, monthly_memberships, uncovered
from ashare.p5_dataset import Boundaries


class StablePoolTests(unittest.TestCase):
    def test_monthly_selection_only_uses_previous_sessions(self):
        days = pd.bdate_range("2026-01-01", "2026-04-03")
        lines = []
        for i, day in enumerate(days):
            for stock, rank in (("000001", 2), ("000002", 3), ("000003", 9)):
                if day.month >= 3 and stock == "000003":
                    rank = 1
                lines.append((stock, i, rank, day.date().isoformat()))
        frame = pd.DataFrame(lines, columns=["stock_code", "session_id", "liquidity_rank", "trade_date"])
        kwargs = dict(size=2, retain=1, lookback=8, minimum=5)
        complete = monthly_memberships(frame, days, "2026-02-01", "2026-04-03", **kwargs)
        first = monthly_memberships(frame.loc[frame.trade_date.lt("2026-03-01")], days[days.month < 3],
                                    "2026-02-01", "2026-02-28", **kwargs)
        pd.testing.assert_frame_equal(first, complete.loc[complete.month.eq("2026-02")].reset_index(drop=True))
        assert complete.loc[complete.month.eq("2026-03"), "stock_code"].tolist() == ["000001", "000002"]
        assert complete.loc[complete.month.eq("2026-04"), "stock_code"].tolist() == ["000001", "000003"]
        assert (complete.selection_cutoff.str.slice(0, 7) < complete.month).all()


    def test_intervals_merge_only_contiguous_provider_confirmed_periods(self):
        d = date.fromisoformat
        merged = merge_intervals([(d("2026-09-02"), d("2026-09-03")), (d("2026-09-04"), d("2026-09-06")),
                                  (d("2026-09-08"), d("2026-09-13"))])
        assert merged == [(d("2026-09-02"), d("2026-09-06")), (d("2026-09-08"), d("2026-09-13"))]
        assert list(uncovered(merged, d("2026-09-01"), d("2026-09-14"))) == [
            (d("2026-09-01"), d("2026-09-01")), (d("2026-09-07"), d("2026-09-07")),
            (d("2026-09-14"), d("2026-09-14"))]
        with self.assertRaisesRegex(ValueError, "Inverted"):
            merge_intervals([(d("2026-09-05"), d("2026-09-04"))])


    def test_daily_denominator_retains_rejected_and_empty_evaluation_dates(self):
        calendar = ["2026-09-01", "2026-09-02", "2026-09-03"]
        bounds = Boundaries(*calendar, "2026-10-01")
        members = pd.DataFrame([{"month": "2026-09", "stock_code": "000001"}])
        observed = pd.DataFrame([{"stock_code": "000001", "trade_date": "2026-09-01",
                                  "reason": "INSUFFICIENT_DUAL_ACTION_EVIDENCE"}])
        raw = pd.DataFrame([{"stock_code": "000001", "trade_date": "2026-09-02", "not_billing": 1,
                             "open": 10.0, "close": 10.0, "high": 10.0, "low": 10.0,
                             "volume": 1, "amount": 1, "pre_close": 10}])
        detail, daily = audit_quality(observed, members, raw, calendar, bounds)
        assert detail.reason.tolist() == ["INSUFFICIENT_DUAL_ACTION_EVIDENCE",
                                          "INVALID_OR_NONCONSECUTIVE_CAUSAL_PRICE_WINDOW", "MISSING_FORMAL_PRICE"]
        assert daily.candidate_stocks.tolist() == [1, 1, 1]
        assert daily.valid_labels.tolist() == [0, 0, 0]

    def test_inference_counts_use_no_post_signal_events_or_mature_labels(self):
        calendar = pd.bdate_range("2026-08-03", periods=40)
        signal = calendar[37].date().isoformat()
        following = calendar[38].date().isoformat()
        anchors = pd.DataFrame([{"stock_code": "000001", "trade_date": signal, "session_id": 37},
                                {"stock_code": "000002", "trade_date": signal, "session_id": 37}])

        class Evidence:
            def covered(self, stock, first, last):
                return first <= signal and last == signal

            def observed_action_dates(self, stock):
                return {following} if stock == "000001" else {signal}

        result = audit_inference(anchors, [0, 1], calendar, Evidence())
        assert result.price_qualified_sequences.tolist() == [2]
        assert result.research_inference_stocks.tolist() == [1]
