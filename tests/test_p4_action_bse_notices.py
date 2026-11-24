"""Offline checks of issuer identity, gross terms, and dated notice matching."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from ashare import p4_action_bse_notice_verify as verifier
from ashare import p4_action_bse_notices as collect


NOTICE = {"notice_stock_code": "832522", "notice_short_name": "纳科诺尔"}
TARGET = {"record_date": "2025-05-27", "ex_date": "2025-05-28", "cash_pre_tax": "0.3", "capitalization": "0.4"}
TEXT = """
证券代码：832522 证券简称：纳科诺尔 公告编号：2025-001
2024年年度权益分派实施公告
现将权益分派事宜公告如下：
一、权益分派方案
向全体股东每10股转增4股，（其中以股票发行溢价形成的资本公积金每10股转增4股，
不需要纳税；以其他资本公积每10股转增0股，需要纳税），每10股派3.00元人民币现金。
2、扣税说明
公司按10%的税率代扣代缴所得税后，实际每10股派发2.70元。
二、权益登记日与除权除息日
本次权益分派权益登记日为：2025 年5月27日
除权除息日为：2025 年5月28日
"""


class IssuerNoticeTests(unittest.TestCase):
    def test_primary_terms_exclude_tax_and_capitalization_commentary(self):
        amounts, _, _ = verifier.extract(TEXT, NOTICE, TARGET)
        self.assertEqual(amounts, {"cash_pre_tax": "3.00", "capitalization": "4"})

    def test_conflicting_date_and_value_do_not_qualify(self):
        for source in (TEXT.replace("5月28日", "5月29日"), TEXT.replace("3.00元", "3.01元"),
                       TEXT.replace("832522", "920522"), TEXT.replace("转增4股，", "送红股4股，", 1)):
            with self.subTest(source=source), self.assertRaises(ValueError):
                verifier.extract(source, NOTICE, TARGET)

    def test_only_expected_official_attachment_paths_are_allowed(self):
        self.assertTrue(collect.attachment_url("finalpage/2025-05-19/1223589847.PDF").startswith("https://static.cninfo.com.cn/"))
        for path in ("//another.example/test.pdf", "finalpage/2025-05-19/../7.pdf", "finalpage/2025-05-19/1.pdf?x=y"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                collect.attachment_url(path)

    def test_pdf_text_order_recovers_unit_moved_across_parentheses(self):
        layout = TEXT.replace("转增4股，（", "转增4（", 1).replace("不需要纳税；", "不需要纳税；股，", 1)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "layout.txt").write_text(layout, encoding="utf-8")
            notice = NOTICE | {"id": "1", "text": "layout.txt", "pdf": "notice.pdf"}
            with patch.object(verifier.subprocess, "run") as pdf_to_text:
                pdf_to_text.return_value.stdout = TEXT.encode()
                (terms, _, _), evidence = verifier.read_notice(folder, notice, TARGET, folder, "920522", "2025-05-28")
            self.assertEqual(terms["capitalization"], "4")
            self.assertEqual(evidence["mode"], "pdf_text_order")
            self.assertEqual((folder / evidence["text_file"]).read_text(), TEXT)
