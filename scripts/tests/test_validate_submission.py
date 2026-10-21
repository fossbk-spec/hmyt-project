# -*- coding: utf-8 -*-
"""Kiểm thử phân loại PR của validate_submission.py. Chạy: python -m unittest discover -s scripts/tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_submission as vs  # noqa: E402


class ClassifyPrTest(unittest.TestCase):
    def test_pr_ha_tang_khong_bi_chan(self):
        files = ["scripts/check_upstream.py", ".github/workflows/review-tool-selftest.yml", "README.md", "rubric/README.md"]
        self.assertEqual(vs.classify_pr(files)[0], "infrastructure")
        self.assertEqual(vs.validate_pr_files(files), [])

    def test_sua_template_va_tai_lieu_la_ha_tang(self):
        files = ["submissions/_TEMPLATE/README.md", "CONTRIBUTING.md"]
        self.assertEqual(vs.classify_pr(files)[0], "infrastructure")
        self.assertEqual(vs.validate_pr_files(files), [])

    def test_pr_nop_bai_tron_file_ha_tang_bi_chan(self):
        files = ["submissions/x_nhom99/README.md", "scripts/validate_submission.py"]
        self.assertEqual(vs.classify_pr(files)[0], "submission")
        errs = vs.validate_pr_files(files)
        self.assertTrue(any("ngoài thư mục nộp bài" in e and "scripts/validate_submission.py" in e for e in errs))

    def test_pr_nop_bai_sua_template_bi_chan(self):
        errs = vs.validate_pr_files(["submissions/x_nhom99/a.md", "submissions/_TEMPLATE/README.md"])
        self.assertTrue(any("_TEMPLATE" in e for e in errs))

    def test_pr_hai_thu_muc_nop_bai_bi_chan(self):
        errs = vs.validate_pr_files(["submissions/a_n1/x.md", "submissions/b_n2/y.md"])
        self.assertTrue(any("NHIỀU thư mục" in e for e in errs))

    def test_file_nam_ngay_trong_submissions_khong_tinh_la_thu_muc(self):
        self.assertEqual(vs.classify_pr(["submissions/README.md"])[0], "infrastructure")
        errs = vs.validate_pr_files(["submissions/a_n1/x.md", "submissions/README.md"])
        self.assertTrue(any("submissions/README.md" in e for e in errs))

    def test_bai_vi_du_hien_co_van_hop_le(self):
        folder = vs.SUBMISSIONS_DIR / "suy_tim_risk_dxai_vidumau"
        self.assertEqual(vs.classify_pr([f"submissions/{folder.name}/README.md"])[0], "submission")
        self.assertEqual(vs.validate_pr_files([f"submissions/{folder.name}/README.md"]), vs.validate_folder(folder))


if __name__ == "__main__":
    unittest.main()
