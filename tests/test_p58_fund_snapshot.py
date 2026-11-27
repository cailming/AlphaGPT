import json
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ashare.p58_fund_snapshot import CHINA, KEY_FIELDS, SOURCES, boundary, capture, previous_scheduled_session, verified


def moment(day=26, hour=17, minute=55):
    return datetime(2026, 11, day, hour, minute, tzinfo=CHINA)


class Example:
    def __init__(self):
        self.now = moment()
        self.score = 4.32125
        self.market = 100
        self.empty = False

    def read(self, _, day, *, preopen_on=None):
        self.calls = getattr(self, "calls", 0) + 1
        schema = {}
        rows = {}
        for source, column in SOURCES.items():
            signal = source == "stock_signal_event"
            fields = {field: "character varying" if signal or field == "position_state" else "numeric"
                      for field in KEY_FIELDS[source]}
            schema[source] = {"id": "bigint", "stock_code": "character varying", column: "date",
                              "created_at": "timestamp without time zone", "extra_data": "jsonb", **fields}
            row = {k: ("CONFIRMED" if kind == "character varying" else self.score) for k, kind in fields.items()}
            row.update(id=100, stock_code="000001", created_at=f"{day}T15:00:01", extra_data={"k": 2}, **{column: str(day)})
            rows[source] = [] if self.empty else [json.dumps(row)]
        return self.now, self.market, day - timedelta(days=1), schema, [], rows

    def run(self, directory, phase="close"):
        return capture(None, Path(directory), phase, clock=lambda: self.now, reader=self.read)


class SnapshotTests(unittest.TestCase):
    def test_clock_and_incomplete_formal(self):
        early, late = moment(), moment(hour=18, minute=1)
        self.assertEqual(boundary("close", early.date(), early, late, True), "OBSERVATION_ONLY")
        self.assertEqual(boundary("observe", early.date(), early, early, True), "OBSERVATION_ONLY")
        self.assertEqual(boundary("next-open", early.date(), late, late, True),
                         "EVENING_OBSERVATION_FOR_FOLLOWING_SESSION")
        self.assertEqual(boundary("close", early.date(), early, early, False), "NO_FORMAL_SESSION_OBSERVED")

    def test_outside_window_never_queries_sources(self):
        example = Example()
        with TemporaryDirectory() as root:
            self.assertEqual(example.run(root, "next-open")["status"], "BEFORE_EVENING_SNAPSHOT_WINDOW")
            example.now = moment(hour=18, minute=0)
            self.assertEqual(example.run(root)["status"], "OUTSIDE_CLOSE_SNAPSHOT_WINDOW")
            self.assertFalse(hasattr(example, "calls"))

    def test_no_source_before_formal(self):
        example = Example()
        example.empty = True
        example.market = 0
        with TemporaryDirectory() as root:
            self.assertEqual(example.run(root)["status"], "NO_FORMAL_OR_FUND_DATA_YET")
            self.assertEqual(list(Path(root).rglob("manifest.json")), [])

    def test_preopen_next_exchange_session_and_real_clock(self):
        from datetime import date
        example = Example()
        example.now = moment(day=27, hour=8)
        with TemporaryDirectory() as root:
            result = example.run(root, "preopen")
            manifest = verified(Path(result["manifest"]))
            self.assertEqual(manifest["signal_date"], "2026-11-26")
            self.assertEqual(manifest["target_session"], "2026-11-27")
            self.assertTrue(manifest["snapshot_at"].startswith("2026-11-27T"))
            self.assertEqual(manifest["availability"], "NEXT_SESSION_BEFORE_OPENING_AUCTION")
            example.now = moment(day=27, hour=9, minute=15)
            self.assertEqual(example.run(root, "preopen")["status"], "OUTSIDE_VERIFIED_PREOPEN_WINDOW")
        self.assertIsNone(previous_scheduled_session(date(2026, 10, 6)))
        self.assertEqual(previous_scheduled_session(date(2026, 10, 8)), date(2026, 9, 30))
        self.assertIsNone(previous_scheduled_session(date(2027, 1, 4)))

    def test_immutable_raw_typed_columns_empty_and_edits(self):
        import polars as pl
        example = Example()
        with TemporaryDirectory() as root:
            first = Path(example.run(root)["manifest"])
            saved = first.read_bytes()
            self.assertEqual(example.run(root)["status"], "ALREADY_ARCHIVED")
            source = "stock_main_force_daily"
            table = pl.read_parquet(first.parent / (source + ".parquet"))
            self.assertEqual(json.loads(table["raw_row_json"][0])["main_force_score"], example.score)
            self.assertEqual(table["main_force_score"][0], example.score)
            self.assertEqual(table["stock_code"][0], "000001")
            self.assertIsNone(table["source_updated_at"][0])
            example.score = 9.5
            revision = Path(example.run(root)["manifest"])
            self.assertNotEqual(first, revision)
            self.assertTrue(verified(revision)["previous_revision_manifest_sha256"])
            self.assertEqual(first.read_bytes(), saved)
            example.empty = True
            gap = verified(Path(example.run(root)["manifest"]))
            self.assertEqual(gap["sources"][source]["rows"], 0)
            self.assertIsNone(gap["sources"][source]["non_null_share_by_field"]["main_force_score"])

    def test_prior_day_and_damaged_file(self):
        example = Example()
        with TemporaryDirectory() as root:
            example.run(root)
            example.now = moment(day=27)
            example.score = 7.1
            next_day = Path(example.run(root)["manifest"])
            stats = verified(next_day)["previous_day_changes"]["by_source"]["stock_main_force_daily"]
            self.assertEqual((stats["shared_stocks"], stats["changed_core_values"]), (1, 1))
            (next_day.parent / "stock_signal_event.parquet").write_bytes(b"incomplete")
            with self.assertRaisesRegex(ValueError, "Archived snapshot file changed"):
                example.run(root)


if __name__ == "__main__":
    unittest.main()
