"""P3.3-C: historical event evidence obeys publication and effective dates."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd

from ashare.risk_events import (
    MODE, filter_signals_by_risk_events, load_risk_events,
)


def events():
    return pd.DataFrame([
        ("002731", "RISK_STAR_ST", "2026-07-07", "2026-07-06",
         "https://example.org/002731", "2026-067"),
        ("688496", "RISK_STAR_ST", "2026-05-12", "2026-05-09",
         "https://example.org/688496", "2026-033"),
    ], columns=["stock_code", "event_type", "effective_date",
                "published_on", "source_url", "source_id"])


def sigs(*records):
    return pd.DataFrame(records, columns=["stock_code", "signal_date", "score"])


class RiskEventTests(unittest.TestCase):
    def test_two_known_2026_risk_stocks_are_blocked(self):
        days = [
            "2026-08-31", "2026-09-01", "2026-09-10",
            "2026-09-11", "2026-09-14",
        ]
        signals = sigs(("002731", "2026-08-31", 10),
                       ("688496", "2026-09-10", 9),
                       ("000001", "2026-09-10", 8))
        out = filter_signals_by_risk_events(signals, events(), days)
        self.assertEqual(out.stats["blocked_unique_signal_rows"], 2)
        self.assertEqual(out.signals.stock_code.tolist(), ["000001"])
        self.assertEqual(out.stats["status"], MODE)
        self.assertFalse(out.stats["all_unflagged_symbols_proven_normal"])
        self.assertEqual(out.stats["signals_with_no_events_for_stock"], 1)
        self.assertEqual(out.blocked_by_day.block_reasons.tolist(),
                         ["RISK_WARNING", "RISK_WARNING"])

    def test_cannot_leak_future_publication_or_future_effective_date(self):
        days = ["2026-05-08", "2026-05-11", "2026-05-12", "2026-05-13"]
        inputs = sigs(("688496", "2026-05-08", 1),
                      ("688496", "2026-05-11", 2),
                      ("688496", "2026-05-12", 3))
        out = filter_signals_by_risk_events(inputs, events(), days)
        self.assertEqual(out.stats["blocked_unique_signal_rows"], 2)
        self.assertEqual(out.signals.signal_date.dt.strftime("%Y-%m-%d").tolist(),
                         ["2026-05-08"])

    def test_same_day_publication_does_not_reach_that_morning(self):
        frame = pd.DataFrame([
            ("688496", "HALT", "2026-09-11", "2026-09-11",
             "https://example.org/halt", "HALT-9-11"),
        ], columns=events().columns)
        days = ["2026-09-10", "2026-09-11", "2026-09-14"]
        inputs = sigs(("688496", "2026-09-10", 1),
                      ("688496", "2026-09-11", 2))
        out = filter_signals_by_risk_events(inputs, frame, days)
        self.assertEqual(out.stats["blocked_unique_signal_rows"], 1)
        self.assertEqual(out.signals.signal_date.dt.strftime("%Y-%m-%d").tolist(),
                         ["2026-09-10"])

    def test_risk_clear_and_resume_are_respected(self):
        e = pd.DataFrame([
            ("000001", "RISK_ST", "2026-03-03", "2026-03-02",
             "https://example.org/1", "r1"),
            ("000001", "RISK_NORMAL", "2026-03-10", "2026-03-09",
             "https://example.org/2", "r2"),
            ("000001", "HALT", "2026-03-04", "2026-03-03",
             "https://example.org/3", "h1"),
            ("000001", "RESUME", "2026-03-10", "2026-03-09",
             "https://example.org/4", "h2"),
        ], columns=events().columns)
        days = ["2026-03-03", "2026-03-04", "2026-03-09",
                "2026-03-10", "2026-03-11"]
        out = filter_signals_by_risk_events(
            sigs(("000001", "2026-03-03", 1),
                 ("000001", "2026-03-10", 2)), e, days
        )
        self.assertEqual(out.stats["blocked_unique_signal_rows"], 1)
        self.assertEqual(out.stats["halt_block_hits"], 1)
        self.assertEqual(out.signals.signal_date.dt.date.astype(str).tolist(),
                         ["2026-03-10"])

    def test_late_disclosure_does_not_rewrite_earlier_decisions(self):
        # The Sep 7 announcement confirms a Sep 1 halt retrospectively.
        e = pd.DataFrame([
            ("002731", "HALT", "2026-09-01", "2026-09-07",
             "https://example.org/retrospective", "2026-109"),
        ], columns=events().columns)
        days = ["2026-09-02", "2026-09-03", "2026-09-07", "2026-09-08"]
        out = filter_signals_by_risk_events(
            sigs(("002731", "2026-09-02", 1),
                 ("002731", "2026-09-07", 2)), e, days
        )
        self.assertEqual(out.stats["blocked_unique_signal_rows"], 1)
        self.assertEqual(out.signals.signal_date.dt.strftime("%Y-%m-%d").tolist(),
                         ["2026-09-02"])

    def test_checked_in_announcement_examples_block_the_two_cases(self):
        path = Path(__file__).resolve().parents[1] / "ashare" / "data" / "risk_events_verified_examples.csv"
        recorded = load_risk_events(path)
        out = filter_signals_by_risk_events(
            sigs(("002731", "2026-08-31", 10),
                 ("688496", "2026-09-10", 9)),
            recorded,
            ["2026-08-31", "2026-09-01", "2026-09-10", "2026-09-11"],
        )
        self.assertEqual(out.stats["blocked_unique_signal_rows"], 2)
        self.assertEqual(set(out.blocked_by_day.stock_code), {"002731", "688496"})

    def test_invalid_event_or_duplicate_transition_is_rejected(self):
        bad = events()
        bad.loc[0, "event_type"] = "MYSTERY"
        with self.assertRaisesRegex(ValueError, "event_type"):
            filter_signals_by_risk_events(sigs(("002731", "2026-08-31", 1)),
                                          bad, ["2026-08-31", "2026-09-01"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            filter_signals_by_risk_events(sigs(("002731", "2026-08-31", 1)),
                                          pd.concat([events(), events()]),
                                          ["2026-08-31", "2026-09-01"])

    def test_end_of_range_signal_cannot_execute(self):
        out = filter_signals_by_risk_events(
            sigs(("002731", "2026-09-30", 1)), events(),
            ["2026-09-29", "2026-09-30"]
        )
        self.assertEqual(out.stats["signals_without_next_session_in_range"], 1)
        self.assertEqual(out.stats["blocked_unique_signal_rows"], 0)

    def test_csv_load_and_url_validation(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "risk.csv"
            events().to_csv(path, index=False)
            parsed = load_risk_events(path)
            self.assertEqual(parsed.stock_code.tolist(), ["002731", "688496"])
            bad = events()
            bad.loc[0, "source_url"] = "non-url"
            bad.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "source_url"):
                load_risk_events(path)


if __name__ == "__main__":
    unittest.main()
