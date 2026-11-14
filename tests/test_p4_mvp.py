"""P4 MVP causal-universe, gap-safe labels, chronological training tests."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from ashare.p4_mvp import (
    FEATURES, PoolConfig, RidgeModel, build_pool, fit_splits,
    load_formal_ohlcv, rank_diagnostics,
)


def synthetic(periods: int = 160) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    rows = []
    codes = ("000001", "000002", "000003", "000004", "000005")
    for s, code in enumerate(codes):
        for i, t in enumerate(dates):
            px = 10 + 3 * s + .025 * i + .4 * np.sin(i / 3.3 + s / 2.1)
            close = px * (1 + .002 * np.cos(i / 4 + s))
            rows.append({
                "stock_code": code, "trade_date": t,
                "open": px, "close": close,
                "high": max(px, close) * 1.02,
                "low": min(px, close) * .98,
                "volume": 90000 * (s + 1) * (1 + .07 * np.cos(i / 5 + s)),
                "amount": 2e6 * (s + 1) * (1 + .03 * np.sin(i / 7 + s)),
                "not_billing": 1,
            })
    return pd.DataFrame(rows), dates


def cfg(*, top_n=4, hold=3, streak=25):
    return PoolConfig(
        top_n=top_n, min_price=3,
        consecutive_sessions=streak, hold_sessions=hold,
    )


class P4MVPTests(unittest.TestCase):
    def setUp(self):
        self.bars, self.dates = synthetic()
        self.start = self.dates[35].date()
        self.end = self.dates[150].date()

    def pool(self, *, raw=None, config=None, events=None):
        return build_pool(
            self.bars if raw is None else raw,
            self.start, self.end,
            cfg() if config is None else config,
            events=events,
        )

    def test_dynamic_pool_uses_only_past_liquidity(self):
        out = self.pool()
        self.assertEqual(out.stats["pool_cap"], 4)
        day = out.data.loc[out.data.trade_date == self.dates[50]]
        self.assertEqual(set(day.stock_code), {
            "000002", "000003", "000004", "000005"
        })
        self.assertTrue((day.liquidity_rank <= 4).all())
        self.assertEqual(out.stats["historical_st_complete"], False)

    def test_future_bar_changes_label_not_past_factors_or_pool(self):
        baseline = self.pool()
        changed = self.bars.copy()
        code = "000005"
        t = self.dates[51]
        idx = changed.index[
            (changed.stock_code == code) & (changed.trade_date == t)
        ][0]
        changed.loc[idx, ["open", "high", "low", "close"]] = [
            1000, 1050, 900, 1020
        ]
        changed.loc[idx, "amount"] = 9e11
        rerun = self.pool(raw=changed)
        prior = baseline.data.loc[
            baseline.data.trade_date.eq(self.dates[50])
        ].sort_values("stock_code").reset_index(drop=True)
        now = rerun.data.loc[
            rerun.data.trade_date.eq(self.dates[50])
        ].sort_values("stock_code").reset_index(drop=True)
        self.assertEqual(prior.stock_code.tolist(), now.stock_code.tolist())
        np.testing.assert_allclose(
            prior[list(FEATURES)].to_numpy(dtype=float),
            now[list(FEATURES)].to_numpy(dtype=float),
        )
        # Changed future label might be excluded by the large-return guard;
        # either outcome is allowed, but no future feature may change.
        self.assertTrue(rerun.stats["selection_is_future_data_free"])

    def test_missing_prior_session_blocks_eligibility(self):
        lost_date = self.dates[48]
        missing = self.bars.loc[
            ~((self.bars.stock_code == "000005")
              & (self.bars.trade_date == lost_date))
        ].copy()
        out = self.pool(raw=missing, config=cfg(top_n=5))
        self.assertFalse(
            ((out.data.stock_code == "000005")
             & out.data.trade_date.eq(self.dates[50])).any()
        )

    def test_missing_future_session_does_not_jump_label(self):
        first = self.pool(config=cfg(top_n=5))
        before = first.data.loc[
            first.data.stock_code.eq("000005")
            & first.data.trade_date.eq(self.dates[50])
        ].iloc[0]
        self.assertTrue(np.isfinite(before.target_return))
        missing = self.bars.loc[
            ~((self.bars.stock_code == "000005")
              & self.bars.trade_date.eq(self.dates[51]))
        ].copy()
        after = self.pool(raw=missing, config=cfg(top_n=5))
        row = after.data.loc[
            after.data.stock_code.eq("000005")
            & after.data.trade_date.eq(self.dates[50])
        ].iloc[0]
        self.assertTrue(np.isnan(row.target_return))

    def test_duplicate_formal_rows_and_bad_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.pool(raw=pd.concat([self.bars, self.bars.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, "configuration"):
            PoolConfig(consecutive_sessions=5)

    def test_known_risk_event_removes_only_after_publication(self):
        warning = pd.DataFrame([{
            "stock_code": "000005", "event_type": "RISK_STAR_ST",
            "effective_date": self.dates[80].date().isoformat(),
            "published_on": self.dates[79].date().isoformat(),
            "source_url": "https://example.org/notice",
            "source_id": "sample-01",
        }])
        out = self.pool(config=cfg(top_n=5), events=warning)
        self.assertTrue(
            ((out.data.stock_code == "000005")
             & out.data.trade_date.eq(self.dates[75])).any()
        )
        self.assertFalse(
            ((out.data.stock_code == "000005")
             & out.data.trade_date.eq(self.dates[80])).any()
        )
        self.assertGreater(out.stats["blocked_known_risk_event_rows"], 0)

    def test_chronological_purge_and_unseen_test_labels(self):
        pool = self.pool()
        model, report, heldout, latest = fit_splits(pool)
        self.assertEqual(report["model_version"], "P4_RIDGE_V1")
        self.assertLess(
            report["train_last_label_date"], report["test_start"]
        )
        self.assertTrue(
            (heldout.trade_date >= pd.Timestamp(report["test_start"])).all()
        )
        self.assertEqual(len(latest), 4)
        self.assertEqual(len(model.coefficients), len(FEATURES))
        self.assertGreater(report["test_metrics"]["evaluated_days"], 5)
        self.assertIn("topk_net_return_proxy", report["test_metrics"])

        # Do not peek at heldout labels during model selection or fitting.
        changed = self.pool()
        cut = pd.Timestamp(report["test_start"])
        mask = changed.data.trade_date >= cut
        changed.data.loc[mask, "target_return"] *= -1
        changed_model, changed_report, _, _ = fit_splits(changed)
        self.assertEqual(
            report["selection_alpha"], changed_report["selection_alpha"]
        )
        np.testing.assert_allclose(
            model.coefficients, changed_model.coefficients
        )
        self.assertAlmostEqual(model.intercept, changed_model.intercept)

    def test_saved_model_predictions_reproduce_exactly(self):
        pool = self.pool()
        model, report, test, latest = fit_splits(pool)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ridge_model.npz"
            model.save(path)
            again = RidgeModel.load(path)
            np.testing.assert_allclose(
                again.predict(latest), model.predict(latest)
            )
            self.assertEqual(again.features, FEATURES)

    def test_rank_metrics_not_portfolio_pnl(self):
        df = pd.DataFrame([
            {"trade_date": "2026-01-01", "stock_code": "000001",
             "predicted_return": .5, "target_return": .04},
            {"trade_date": "2026-01-01", "stock_code": "000002",
             "predicted_return": .4, "target_return": .02},
            {"trade_date": "2026-01-01", "stock_code": "000003",
             "predicted_return": .1, "target_return": -.05},
        ])
        result = rank_diagnostics(df, top_k=1, roundtrip_cost_bps=25)
        self.assertEqual(result["topk_positive_share"], 1.0)
        self.assertAlmostEqual(result["topk_mean_forward_return"], .04)
        self.assertAlmostEqual(result["topk_net_return_proxy"], .0375)
        self.assertIn("NOT portfolio PnL", result["warning"])

    def test_loader_only_reads_formal_market(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        with engine.begin() as c:
            c.exec_driver_sql("ATTACH DATABASE ':memory:' AS public")
            c.exec_driver_sql(
                "CREATE TABLE public.market "
                "(stock_code TEXT, trade_date DATE, open REAL, high REAL,"
                " low REAL, close REAL, volume REAL, amount REAL,"
                " not_billing INTEGER)"
            )
            c.exec_driver_sql(
                "INSERT INTO public.market VALUES "
                "('000001','2026-08-31',10,11,9,10,100,1000,1),"
                "('000002','2026-08-31',10,11,9,10,100,1000,2)"
            )
        out = load_formal_ohlcv(
            date(2026, 8, 31), date(2026, 8, 31),
            engine=engine, include_label_tail=False,
        )
        self.assertEqual(out.stock_code.tolist(), ["000001"])


if __name__ == "__main__":
    unittest.main()
