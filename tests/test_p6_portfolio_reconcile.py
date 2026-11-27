"""Independent ledger reconciliation rejects economic and archival corruption."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from ashare.p4_action_pipeline import atomic_json
from ashare.p6_portfolio import PortfolioConfig, initial_state, step
from ashare.p6_portfolio_report import publish_day
from ashare.p6_portfolio_reconcile import digest, reconcile, reconcile_account


class ReconciliationTests(unittest.TestCase):
    def setup_fill(self):
        cfg = PortfolioConfig()
        state, equity, positions, trades, rejects = step(initial_state(cfg), day="2026-11-26",
            prior_day="2026-11-25", ranked=["688001"], bars={"688001":{"open":31., "pre_close":31., "close":31.25}},
            strategy="A", config=cfg)
        return cfg, state, equity, positions, trades, rejects

    def test_archive_checks_and_rerun_fingerprint(self):
        cfg, state, equity, positions, trades, rejects = self.setup_fill()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / "policy.json"
            atomic_json(policy, {"lanes":["test"],"strategies":["A"],"config":asdict(cfg)})
            path = root / "diagnostic-2026-11-26" / "2026-11-26"
            path.mkdir(parents=True)
            atomic_json(path / "checkpoint.json", {"accounts":{"test/A":state}})
            atomic_json(path / "inputs.json", {"actions":{}})
            tables = {k:[row | {"model":"test","strategy":"A"} for row in v]
                      for k,v in (("equity",[equity]),("trades",trades),("positions",positions),("rejections",rejects))}
            publish_day(path, path.name, "diagnostic", "HISTORICAL_DIAGNOSTIC", tables, digest(policy), None,
                        datetime.now(timezone.utc))
            result = reconcile(path.parent)
            self.assertEqual(result["fills"], 1)
            self.assertTrue(reconcile(path.parent, result["archive_chain_sha256"])["matched_prior_run"])
            with self.assertRaisesRegex(ValueError, "changed since"):
                reconcile(path.parent, "wrong")
            with (path / "portfolio_positions.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                reconcile(path.parent)

    def test_same_session_sale_and_malformed_nav_rejected(self):
        cfg,state,eq,positions,trades,rejects = self.setup_fill()
        eq = eq | {"trade_count":2}
        bad = [*trades, trades[0] | {"side":"sell"}]
        with self.assertRaisesRegex(ValueError, "T\\+0 or oversold"):
            reconcile_account("2026-11-26", eq, positions, bad, rejects, asdict(cfg), None, state, {})
        with self.assertRaisesRegex(ValueError, "NAV mismatch"):
            reconcile_account("2026-11-26", eq | {"trade_count":1,"nav":2}, positions, trades, rejects,
                              asdict(cfg), None, state, {})

    def test_dividend_accrual_delivery_and_cash_payment_reconcile(self):
        cfg = PortfolioConfig()
        state, *_ = step(initial_state(cfg), day="2026-11-26", prior_day="2026-11-25", ranked=["600001"],
                         bars={"600001":{"open":10., "pre_close":10., "close":10.}}, strategy="A", config=cfg)
        action = {"600001": {"status":"VERIFIED", "cash_per_share":.5, "share_multiplier":2.,
                             "cash_pay_date":"2026-11-30", "share_list_date":"2026-11-30"}}
        for day, prior, events in (("2026-11-27","2026-11-26",action),("2026-11-30","2026-11-27",{})):
            following, equity, holdings, trades, rejections = step(
                state, day=day, prior_day=prior, ranked=[], bars={"600001":{"open":5., "close":5.}},
                strategy="A", config=cfg, actions=events)
            reconcile_account(day, equity, holdings, trades, rejections, asdict(cfg), state, following, events)
            state = following
        self.assertGreater(equity["action_cash_paid"], 0)
        self.assertEqual(equity["dividend_receivable"], 0)


if __name__ == "__main__":
    unittest.main()
