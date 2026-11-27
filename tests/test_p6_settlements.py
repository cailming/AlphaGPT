from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import unittest
from unittest.mock import patch

from ashare import p6_settlements as mod


class SettlementTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base/"receipts"
        self.pdf = self.base/"fixture.pdf"
        self.pdf.write_bytes(b"%PDF-1.4 explicit synthetic test fixture, not an issuer document\n%%EOF\n")
        self.review_path = self.base/"review.json"
        self.index = self.base/"source.json"
        self.day = "2026-11-27"
        self.opening = datetime.fromisoformat(self.day+"T09:15:00+08:00")
        self.review = dict(stock_code="688001",ex_date=self.day,record_date="2026-11-26",
            source_id="fixture",source_url="https://static.cninfo.com.cn/finalpage/fixture.PDF",
            source_sha256=hashlib.sha256(self.pdf.read_bytes()).hexdigest(),
            published_at="2026-11-25T19:00:00+08:00", reviewed_at="2026-11-26T15:00:00+08:00",
            reviewer="synthetic reviewer",pdf_page=2,evidence_excerpt="synthetic fixture terms",
            cash_per_share=.25,share_multiplier=1.,cash_pay_date="2026-11-30",share_list_date=None,
            cash_basis="VERIFIED_ACCOUNT_CREDIT_PER_OLD_SHARE",scope="COMPLETE_ORDINARY_A_SHARE_CASH_AND_BONUS_ONLY",
            review_status="ISSUER_IMPLEMENTATION_NOTICE_VERIFIED")
        self.index.write_text(json.dumps([dict(stock_code="688001",ex_date=self.day,
            official_notice_id="fixture",source_url=self.review["source_url"],published_on="2026-11-25",
            file_sha256=self.review["source_sha256"])]))

    def commit(self, at, **changes):
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return at.astimezone(tz)
        self.review_path.write_text(json.dumps(self.review | changes))
        with patch.object(mod, "datetime", Clock):
            return mod.capture(self.root,self.review_path,self.pdf,self.index)

    def test_digest_cashdate_receipt_and_idempotence(self):
        when = self.opening-timedelta(hours=1)
        path = self.commit(when)
        digest = mod.hash_file(path/"receipt.json")
        self.assertEqual(self.commit(self.opening+timedelta(hours=3)),path)
        events, shas = mod.load_settlements(self.root,self.day,self.opening)
        self.assertEqual(shas,[digest])
        self.assertTrue(events["688001"]["available_at_open"])
        self.assertEqual(events["688001"]["cash_pay_date"],"2026-11-30")
        (path/"source.pdf").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError,"digest mismatch"):
            mod.load_settlements(self.root,self.day,self.opening)

    def test_real_clock_prevents_retrospective_open_eligibility(self):
        receipt_time = self.opening+timedelta(hours=1)
        self.commit(receipt_time)
        self.assertEqual(mod.load_settlements(self.root,self.day,self.opening),({},[]))
        events,_ = mod.load_settlements(self.root,self.day,receipt_time)
        self.assertFalse(events["688001"]["available_at_open"])
        self.commit(self.opening+timedelta(days=1),cash_per_share=.9)
        events,shas = mod.load_settlements(self.root,self.day,self.opening+timedelta(days=2))
        self.assertEqual(len(shas),1)
        self.assertEqual(events["688001"]["cash_per_share"],.25)

    def test_intraday_conflict_preserves_morning_facts_and_end_unknown(self):
        self.commit(self.opening-timedelta(minutes=30))
        self.commit(self.opening+timedelta(hours=1),cash_per_share=.30)
        events,shas = mod.load_settlements(self.root,self.day,self.opening+timedelta(hours=2))
        self.assertEqual(len(shas),2)
        self.assertTrue(events["688001"]["late_conflict"])
        self.assertEqual(events["688001"]["cash_per_share"],.25)

    def test_reject_incomplete_unsupported_or_unanchored_terms(self):
        for args in (dict(cash_pay_date=None),dict(share_multiplier=1.1),dict(cash_basis="PRETAX_ONLY"),
                     dict(ex_date="2026-11-25"),dict(source_url="https://evil.example/foo.pdf"),
                     dict(source_id="not-in-reviewed-index"),dict(reviewed_at="2026-11-29T00:00:00+08:00")):
            with self.subTest(args=args), self.assertRaises((ValueError,TypeError)):
                self.commit(self.opening,**args)
        self.assertEqual(mod.load_settlements(self.root,self.day,self.opening),({},[]))


if __name__ == "__main__":
    unittest.main()
