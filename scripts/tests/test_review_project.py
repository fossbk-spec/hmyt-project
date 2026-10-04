# -*- coding: utf-8 -*-
"""Kiểm thử cho scripts/review_project.py — chạy: python -m unittest discover scripts/tests

Mỗi test dựng 1 bài nộp giả trong thư mục tạm (không đụng vào submissions/).
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import review_project as rp  # noqa: E402

SAMPLE = ROOT / "submissions" / "suy_tim_risk_dxai_vidumau"
CONFIG = rp.load_config()


def opts(**kw):
    base = dict(milestone=None, rubric=None, online=False, reference_dir=None, deadline=None,
                pr_body_text=None, max_beyond=6, max_viva=6)
    base.update(kw)
    return types.SimpleNamespace(**base)


def notebook(*cells, executed=True):
    out = []
    for i, src in enumerate(cells, 1):
        out.append({"cell_type": "code", "execution_count": i if executed else None, "metadata": {},
                    "outputs": [], "source": src})
    return json.dumps({"cells": out, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


class Sub:
    """Bài nộp giả: Sub(tmp, "3.2", "sa_sut_tri_tue").file("src/a.py", "...")"""

    def __init__(self, tmp, topic_id, slug, group="nhom99", members=("20210001", "20210002", "20210003")):
        self.path = Path(tmp) / f"{slug}_{group}"
        self.path.mkdir(parents=True)
        self.file("submission.json", json.dumps({"topic_slug": slug, "topic_id": topic_id, "group_code": group,
                                                 "members": list(members), "submitted_at": "2026-12-15"}))
        self.file("README.md", "# Tóm tắt\n\nĐề tài thử nghiệm.\n")

    def file(self, rel, content):
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return self

    def review(self, **kw):
        return rp.review_folder(self.path, opts(**kw), CONFIG)


def finding(result, check_id):
    for sec in result["sections"]:
        for cr in sec["criteria"]:
            for f in cr["findings"]:
                if f.check_id == check_id:
                    return f
    for f in result["topic_checks"] + result["integrity"]:
        if f.check_id == check_id:
            return f
    raise AssertionError(f"Không có finding {check_id}")


class ConfigTest(unittest.TestCase):
    def test_self_test_passes(self):
        self.assertEqual(rp.self_test(), [])

    def test_every_topic_has_profile(self):
        _, _, catalog = CONFIG
        self.assertEqual(len(catalog), 15)
        self.assertEqual(catalog["suy_tim_risk_dxai"]["rubric"], "lab_3_4")


class LeakageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_scaler_smote_fillna_before_split(self):
        s = Sub(self.tmp, "3.2", "sa_sut_tri_tue").file("src/model.ipynb", notebook(
            "import pandas as pd\nfrom sklearn.preprocessing import StandardScaler\n"
            "from imblearn.over_sampling import SMOTE\nfrom sklearn.model_selection import train_test_split\n"
            "df = pd.read_csv('data.csv')\ndf = df.fillna(df.mean())",
            "scaler = StandardScaler()\nX = scaler.fit_transform(df.drop(columns='y'))\ny = df['y']",
            "X_res, y_res = SMOTE().fit_resample(X, y)",
            "X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)"))
        r = s.review()
        leak_fit = finding(r, "M2.LEAK_FIT")
        self.assertEqual(leak_fit.status, rp.FAIL)
        self.assertEqual(leak_fit.severity, "critical")
        self.assertIn("cell 2", " ".join(leak_fit.evidence))
        self.assertEqual(finding(r, "M2.LEAK_RESAMPLE").status, rp.FAIL)
        self.assertEqual(finding(r, "M2.LEAK_FILLNA").status, rp.WARN)
        self.assertTrue(any(f.check_id == "M2.LEAK_FIT" for f in r["critical"]))

    def test_shuffle_is_not_a_split(self):
        s = Sub(self.tmp, "3.2", "sa_sut_tri_tue").file("src/a.py", (
            "from sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import MinMaxScaler\n"
            "df = df.sample(frac=1, random_state=0)\nX = MinMaxScaler().fit_transform(df[cols])\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"))
        self.assertEqual(finding(s.review(), "M2.LEAK_FIT").status, rp.FAIL)

    def test_fit_on_test_set(self):
        s = Sub(self.tmp, "3.2", "sa_sut_tri_tue").file("src/a.py", (
            "from sklearn.model_selection import train_test_split\nfrom sklearn.impute import KNNImputer\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)\n"
            "imp = KNNImputer()\nX_test = imp.fit_transform(X_test)\n"))
        f = finding(s.review(), "M2.LEAK_FIT")
        self.assertEqual(f.status, rp.FAIL)
        self.assertIn("Test", f.detail)

    def test_clean_pipeline_passes(self):
        s = Sub(self.tmp, "3.2", "sa_sut_tri_tue").file("src/a.py", (
            "from sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\n"
            "from imblearn.over_sampling import SMOTE\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)\n"
            "sc = StandardScaler()\nX_train_s = sc.fit_transform(X_train)\nX_test_s = sc.transform(X_test)\n"
            "X_tr_sm, y_tr_sm = SMOTE(random_state=42).fit_resample(X_train_s, y_train)\n"))
        r = s.review()
        self.assertEqual(finding(r, "M2.LEAK_FIT").status, rp.PASS)
        self.assertEqual(finding(r, "M2.LEAK_RESAMPLE").status, rp.PASS)
        self.assertEqual(finding(r, "M2.LEAK_FILLNA").status, rp.PASS)
        self.assertEqual(r["critical"], [])

    def test_cv_on_resampled_data_warns(self):
        r = rp.review_folder(SAMPLE, opts(), CONFIG)
        self.assertEqual(finding(r, "M2.LEAK_CV_RESAMPLED").status, rp.WARN)

    def test_patient_level_split_detected(self):
        s = Sub(self.tmp, "2.2", "day_mat").file("src/split.py", (
            "from sklearn.model_selection import GroupShuffleSplit\n"
            "gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)\n"
            "tr, te = next(gss.split(df, groups=df.patient_id))\n"
            "assert set(df.patient_id.iloc[tr]).isdisjoint(set(df.patient_id.iloc[te]))\n"))
        r = s.review()
        self.assertEqual(finding(r, "M2.SPLIT_PATIENT").status, rp.PASS)
        self.assertEqual(finding(r, "M2.SPLIT_ASSERT").status, rp.PASS)

    def test_record_level_split_fails_for_images(self):
        s = Sub(self.tmp, "2.2", "day_mat").file("src/split.py", (
            "from sklearn.model_selection import train_test_split\n"
            "train_df, test_df = train_test_split(df, test_size=0.2)\n"))
        self.assertEqual(finding(s.review(), "M2.SPLIT_PATIENT").status, rp.FAIL)


class TopicRequirementTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_ptbxl_random_split_is_critical(self):
        s = Sub(self.tmp, "1.2", "roi_loan_nhip_tim").file("src/a.py", (
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test = train_test_split(X, test_size=0.1)\n"))
        r = s.review()
        self.assertEqual(finding(r, "T.NO_RANDOM_SPLIT").status, rp.FAIL)
        self.assertEqual(finding(r, "T.STRAT_FOLD").status, rp.FAIL)
        self.assertTrue(any(f.check_id == "T.NO_RANDOM_SPLIT" for f in r["critical"]))

    def test_ptbxl_strat_fold_passes(self):
        s = Sub(self.tmp, "1.2", "roi_loan_nhip_tim").file("src/a.py", (
            "train = df[df.strat_fold <= 8]\nval = df[df.strat_fold == 9]\ntest = df[df.strat_fold == 10]\n"))
        r = s.review()
        self.assertEqual(finding(r, "T.STRAT_FOLD").status, rp.PASS)
        self.assertEqual(finding(r, "T.NO_RANDOM_SPLIT").status, rp.PASS)
        self.assertEqual(finding(r, "M2.SPLIT_PATIENT").status, rp.PASS)

    def test_accuracy_only_fails_metrics(self):
        s = Sub(self.tmp, "3.2", "sa_sut_tri_tue").file("src/a.py", (
            "from sklearn.metrics import accuracy_score\nprint(accuracy_score(y_test, pred))\n"))
        f = finding(s.review(), "M4.METRICS")
        self.assertEqual(f.status, rp.FAIL)
        self.assertIn("Accuracy", f.detail)

    def test_missing_deliverables_are_human_readable(self):
        f = finding(rp.review_folder(SAMPLE, opts(), CONFIG), "S.TOPIC_FILES")
        self.assertEqual(f.status, rp.WARN)
        self.assertIn("1_eda.ipynb", f.detail)
        self.assertNotIn("(^|/)", f.detail)
        self.assertNotIn("\\.", f.detail)

    def test_milestone_gating(self):
        s = Sub(self.tmp, "3.2", "sa_sut_tri_tue")
        r = s.review(milestone="m1")
        self.assertEqual([sec["id"] for sec in r["sections"]], ["M0", "M1"])
        self.assertFalse(any(f.check_id.startswith("T.") and f.check_id != "T.MISSINGNESS"
                             for f in r["topic_checks"]))


class IntegrityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_phi_columns(self):
        s = Sub(self.tmp, "3.2", "sa_sut_tri_tue").file("data/raw.csv", "Họ tên,age,CCCD\nA,70,0123\n")
        f = finding(s.review(), "I.PHI")
        self.assertEqual(f.status, rp.FAIL)
        self.assertEqual(f.severity, "critical")

    def test_member_names_flagged(self):
        s = Sub(self.tmp, "3.2", "sa_sut_tri_tue", members=("Nguyen Van A", "20210002"))
        self.assertEqual(finding(s.review(), "S.MEMBERS").status, rp.FAIL)

    def test_result_traceability(self):
        s = Sub(self.tmp, "3.2", "sa_sut_tri_tue")
        s.file("report/bao_cao.md", "## Kết quả\n\n| Mô hình | AUC | F1 |\n|---|---|---|\n| RF | 0,873 | 71,2% |\n")
        f = finding(s.review(), "I.RESULT_TRACE")
        self.assertEqual(f.status, rp.WARN)
        s.file("results/metrics.json", json.dumps({"rf": {"auc": 0.87321, "f1": 0.7118}}))
        self.assertEqual(finding(s.review(), "I.RESULT_TRACE").status, rp.PASS)

    def test_copy_of_other_submission_detected(self):
        s = Sub(self.tmp, "3.4", "suy_tim_risk_dxai")
        shutil.copytree(SAMPLE / "src", s.path / "src")
        f = finding(s.review(), "I.SIMILARITY")
        self.assertEqual(f.status, rp.WARN)
        self.assertEqual(f.severity, "critical")
        self.assertIn("vidumau", f.detail)

    def test_orphan_citekey(self):
        s = Sub(self.tmp, "3.2", "sa_sut_tri_tue")
        s.file("report/paper.md", "## Y văn\n\nTheo [@aracri2025bridging] và [@ghost2099fake].\n")
        s.file("report/references.bib", "@article{aracri2025bridging,\n title={X},\n doi={10.1/x}\n}\n")
        f = finding(s.review(), "I.CITATION_INTEGRITY")
        self.assertEqual(f.status, rp.FAIL)
        self.assertIn("ghost2099fake", f.detail)


class SafetyTest(unittest.TestCase):
    """Công cụ chạy trên PR của sinh viên: tuyệt đối không thực thi code, không theo symlink."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_never_executes_student_code(self):
        marker = Path(self.tmp) / "PWNED"
        payload = f"open({str(marker)!r}, 'w').write('x')\n"
        s = Sub(self.tmp, "3.2", "sa_sut_tri_tue").file("src/evil.py", payload)
        s.file("src/evil.ipynb", notebook(payload))
        s.file("src/__init__.py", payload)
        s.review()
        self.assertFalse(marker.exists())

    @unittest.skipIf(os.name == "nt", "symlink cần quyền đặc biệt trên Windows")
    def test_symlinks_not_followed(self):
        secret = Path(self.tmp) / "secret.txt"
        secret.write_text("TOKEN=ghp_" + "a" * 36, encoding="utf-8")
        s = Sub(self.tmp, "3.2", "sa_sut_tri_tue")
        os.symlink(secret, s.path / "leak.md")
        r = s.review()
        dumped = json.dumps(rp.to_json(r), ensure_ascii=False)
        self.assertNotIn("TOKEN=", dumped)
        self.assertNotIn("leak.md", dumped)


class CliTest(unittest.TestCase):
    def test_pr_files_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / "changed.txt"
            changed.write_text("submissions/suy_tim_risk_dxai_vidumau/README.md\n", encoding="utf-8")
            out = Path(tmp) / "r.md"
            js = Path(tmp) / "r.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = rp.main(["--pr-files", str(changed), "--output", str(out), "--json", str(js)])
            self.assertEqual(code, 0)
            self.assertIn("suy_tim_risk_dxai_vidumau", out.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(js.read_text(encoding="utf-8"))["topic"]["id"], "3.4")

    def test_pr_without_submission_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / "changed.txt"
            changed.write_text("README.md\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rp.main(["--pr-files", str(changed)]), 0)

    def test_fail_on_critical(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Sub(tmp, "3.2", "sa_sut_tri_tue").file("data/x.csv", "ho_ten,age\nA,1\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rp.main([str(s.path), "--fail-on", "critical"]), 1)
                self.assertEqual(rp.main([str(s.path)]), 0)


if __name__ == "__main__":
    unittest.main()
