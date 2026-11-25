"""Guard against accepting scores or events based on an overwritten timestamp."""
import unittest

from ashare.p57_fund_audit import as_local, matching_writers


class MoneyAvailabilityTests(unittest.TestCase):
    def test_conflict_update_overwrites_current_payload_and_creation_time(self):
        meta = {"routine": "daily_writer(date)", "source_md5": "a"}
        body = """
        INSERT INTO "public"."stock_main_force_daily" (...) VALUES (...)
        ON CONFLICT (stock_code, trade_date) DO UPDATE SET created_at = now(), holding_score = excluded.holding_score;
        """
        [facts] = matching_writers([dict(meta, body=body)])
        self.assertEqual(facts["table"], "stock_main_force_daily")
        self.assertTrue(facts["can_replace_existing_payload"])
        self.assertTrue(facts["assigns_created_at"])

    def test_shadow_table_does_not_pass_for_requested_table(self):
        rows = [{"routine": "shadow(date)", "source_md5": "a",
                 "body": "INSERT INTO public.stock_main_force_daily_v2 VALUES (1);"}]
        self.assertEqual(matching_writers(rows), [])

    def test_event_revisions_possible_even_without_changing_created_at(self):
        facts = matching_writers([{"routine": "event()", "source_md5": "a",
                                   "body": "UPDATE public.stock_signal_event SET signal_status='INVALID';"}])[0]
        self.assertTrue(facts["assigns_signal_status_or_score"])
        self.assertTrue(facts["can_replace_existing_payload"])
        self.assertFalse(facts["assigns_created_at"])

    def test_database_timestamp_types_are_explicit(self):
        self.assertEqual(as_local("f.created_at", "timestamp without time zone"), "f.created_at")
        self.assertIn("Asia/Shanghai", as_local("f.created_at", "timestamp with time zone"))
        with self.assertRaises(ValueError):
            as_local("f.created_at", "text")
