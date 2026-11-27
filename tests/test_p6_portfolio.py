import json
import unittest

from ashare.p6_portfolio import PortfolioConfig, initial_state, open_quote, step, trading_unit


S1 = "688001"
S2 = "600001"
BARS = {S1: {"open": 91., "pre_close": 91., "close": 91.},
        S2: {"open": 12., "pre_close": 12., "close": 12.}}
DAYS = ["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19",
        "2026-11-20", "2026-11-23", "2026-11-24", "2026-11-25"]


class PortfolioTest(unittest.TestCase):
    def setUp(self):
        self.config = PortfolioConfig()

    def advance(self, state, i, ranks, *, bars=BARS, strategy="A", **kw):
        return step(state, day=DAYS[i], prior_day=DAYS[i-1] if i else None, ranked=ranks,
                    bars=bars, strategy=strategy, config=self.config, **kw)

    def start(self, ranks, strategy="A"):
        initial = initial_state(self.config)
        state, *_ = self.advance(initial, 0, None, strategy=strategy)
        return self.advance(state, 1, ranks, strategy=strategy)

    def test_freeze_input_units_budget_and_cash_ledger(self):
        s, d, _, fills, _ = self.start([S1, S2])
        self.assertEqual(d["positions"], 2)
        self.assertEqual(fills[1]["shares"] % 100, 0)
        self.assertGreaterEqual(fills[0]["shares"], 200)
        self.assertNotEqual(fills[0]["shares"] % 100, 0)
        for t in fills:
            self.assertLessEqual(t["gross"] + t["fees"], 50000)
            self.assertEqual(t["signal_date"], DAYS[0])
        self.assertAlmostEqual(d["opening_cash"] - d["closing_cash"],
                               sum(f["gross"] + f["fees"] for f in fills))
        self.assertGreater(d["gross_return"], d["net_return"])
        with self.assertRaises(ValueError):
            self.advance(s, 1, [S1])
        with self.assertRaises(ValueError):
            self.advance(s, 2, [S1, S1])
        with self.assertRaises(ValueError):
            self.advance(s, 2, [], strategy="C")
        self.assertEqual(trading_unit("920001")[:2], (100, 1))
        self.assertIsNone(trading_unit("510300"))
        self.assertEqual(json.loads(json.dumps(s)), s)

    def test_hold_five_and_deferred_exit_even_if_reranked(self):
        s, *_ = self.start([S1])
        for i in range(2, 6):
            s, _, _, fills, _ = self.advance(s, i, [])
            self.assertFalse(fills)
        s, d, positions, fills, rejects = self.advance(s, 6, None, bars={})
        self.assertFalse(fills)
        self.assertIsNone(d["nav"])
        self.assertEqual(positions[0]["stock_code"], S1)
        self.assertEqual(rejects[0]["reason"], "MISSING_FORMAL_BAR")
        restored = json.loads(json.dumps(s))
        next_state = self.advance(s, 7, [S1])
        self.assertEqual(next_state, self.advance(restored, 7, [S1]))
        _, d, positions, fills, _ = next_state
        self.assertEqual(positions, [])
        self.assertEqual([a["side"] for a in fills], ["sell"])
        self.assertIsNotNone(d["nav"])
        self.assertIsNone(d["max_drawdown"])

    def test_three_distinct_policies_and_missing_lists(self):
        for strategy in ("A", "B", "C"):
            s, *_ = self.start([S1], strategy)
            s, _, _, fills, _ = self.advance(s, 2, None, strategy=strategy)
            self.assertFalse(fills)
            s, _, _, fills, _ = self.advance(s, 3, [S2], strategy=strategy)
            sides = [f["side"] for f in fills]
            self.assertEqual(sides, {"A": ["buy"], "B": ["sell", "buy"], "C": []}[strategy])
            for i in (4, 5, 6):
                s, _, _, fills, _ = self.advance(s, i, [S2], strategy=strategy)
            if strategy == "C":
                self.assertEqual([t["side"] for t in fills], ["sell", "buy"])

    def test_no_hindsight_or_late_action_leak_at_open(self):
        state, *_ = self.start([S1], "B")
        altered = {code: (bar | {"close": 800., "volume": 0., "high": 2000., "low": 0.}) for code, bar in BARS.items()}
        baseline = self.advance(state, 2, [S2], strategy="B")
        revised = self.advance(state, 2, [S2], bars=altered, strategy="B",
                               actions={S1: {"status": "UNKNOWN", "available_at_open": False}})
        self.assertEqual(baseline[3], revised[3])
        self.assertIsNone(revised[1]["net_return"])
        self.assertEqual(state["session"], 2)
        self.assertEqual(state["unknown_claims"], [])

    def test_settlement_of_cash_and_bonus_and_ex_date_entry(self):
        s, _, pos, _, _ = self.start([S2])
        n = pos[0]["shares"]
        action = {"status": "VERIFIED", "cash_per_share": .5, "share_multiplier": 1.1,
                  "cash_pay_date": DAYS[4], "share_list_date": DAYS[5]}
        ex_bars = {S2: {"open": 10.45, "close": 10.45, "pre_close": 10.45}}
        s, d, pos, _, _ = self.advance(s, 2, [], actions={S2: action}, bars=ex_bars)
        self.assertEqual(pos[0]["shares"], n)
        self.assertEqual(pos[0]["pending_shares"], n // 10)
        self.assertEqual(d["dividend_receivable"], n / 2)
        cash = s["cash"]
        for i in (3, 4, 5):
            s, d, pos, _, _ = self.advance(s, i, [], bars=ex_bars)
        self.assertEqual(s["cash"] - cash, n / 2)
        self.assertEqual(d["dividend_receivable"], 0)
        self.assertEqual(pos[0]["shares"], n + n // 10)
        self.assertEqual(pos[0]["pending_shares"], 0)
        blank, *_ = self.advance(initial_state(self.config), 0, None)
        fresh, d, _, _, _ = self.advance(blank, 1, [S2], actions={S2: action})
        self.assertEqual(d["dividend_receivable"], 0)
        self.assertFalse(fresh["cash_due"])

    def test_stale_unknown_or_missing_terms_preserve_shares(self):
        s, _, positions, *_ = self.start([S1], "B")
        qty = positions[0]["shares"]
        s, d, pos, fills, rejects = self.advance(s, 2, [S2], strategy="B",
            actions={S1: {"status": "VERIFIED", "cash_per_share": 1., "share_multiplier": 1.}})
        self.assertEqual(d["valuation_status"], "UNCONFIRMED_ECONOMIC_ACTION")
        self.assertIsNone(d["net_return"])
        self.assertEqual(pos[0]["shares"], qty)
        self.assertFalse(fills)
        self.assertIn("UNRESOLVED_ENTITLEMENT", [r["reason"] for r in rejects])

    def test_confirmed_morning_and_late_night_conflict_on_sold_stock(self):
        s, _, pos, *_ = self.start([S2], "B")
        claim = {"status": "VERIFIED", "available_at_open": True, "cash_per_share": .25,
                 "share_multiplier": 1, "cash_pay_date": DAYS[4], "record_date": DAYS[1]}
        normal = self.advance(s, 2, [S1], strategy="B", actions={S2: claim})
        for flag in ("late_conflict", "coverage_unverified"):
            sts, row, _, trades, _ = self.advance(s, 2, [S1], strategy="B", actions={S2: claim | {flag: True}})
            self.assertEqual(trades, normal[3])
            self.assertEqual(row["dividend_receivable"], pos[0]["shares"] * .25)
            self.assertIsNone(row["nav"])
            self.assertEqual(sts["unknown_claims"][0]["stock_code"], S2)

    def test_limit_suspension_and_bad_inputs(self):
        base = {"pre_close": 10., "lower_limit": 9., "upper_limit": 11.}
        self.assertEqual(open_quote(base | {"open": 11.}, "buy", self.config)[1], "UP_LIMIT_BUY")
        self.assertEqual(open_quote(base | {"open": 9.}, "sell", self.config)[1], "DOWN_LIMIT_SELL")
        self.assertIsNotNone(open_quote({"open": 10.}, "sell", self.config)[1])
        self.assertIsNotNone(open_quote({"open": 10., "pre_close": 9.}, "sell", self.config)[1])
        self.assertIsNotNone(open_quote(base | {"open": 10., "suspended": True}, "buy", self.config)[1])
        self.assertIsNotNone(open_quote(base | {"open": 10., "can_buy": "False"}, "buy", self.config)[1])
        with self.assertRaises(ValueError):
            PortfolioConfig(initial_cash=float("nan"))

    def test_open_budget_and_staggering(self):
        stocks = ["600%03d" % i for i in range(20)]
        bars = {code: {"open": 10., "close": 20., "pre_close": 10.} for code in stocks}
        state, *_ = self.advance(initial_state(self.config), 0, None)
        _, d, _, fills, _ = self.advance(state, 1, stocks, bars=bars, benchmark_return=.01)
        self.assertEqual(len(fills), 4)
        self.assertEqual(d["excess_return"], d["net_return"] - d["benchmark_return"])
        self.assertLessEqual(max(t["gross"] for t in fills), 50000)
        _, d2, _, _, _ = self.advance(state, 1, [], benchmark_return=None)
        self.assertIsNone(d2["excess_return"])


if __name__ == "__main__":
    unittest.main()
