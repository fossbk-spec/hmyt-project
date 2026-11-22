"""Synthetic tests only. No public patient identifiers or images are needed."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from chexchonet_pipeline_ml_support_dl import (
    SAFE_TABULAR_FEATURES, align_branch_predictions, build_tabular_records,
    make_patient_split, prepare_metadata, validate_tabular_features,
)


class PairingAndSplitTests(unittest.TestCase):
    def test_patient_separation(self):
        frame = pd.DataFrame({"patient_id": [f"synthetic_{i//2}" for i in range(160)],
                              "cxr_filename": [f"image_{i}.jpg" for i in range(160)],
                              "composite_slvh_dlv": [(i//2)%3 == 0 for i in range(160)]})
        result = make_patient_split(frame, seed=42)
        assert result.groupby("patient_id")["split"].nunique().max() == 1
        sets = {s: set(result.loc[result["split"].eq(s), "patient_id"]) for s in ("train", "val", "test")}
        assert sets["train"].isdisjoint(sets["test"])
        assert sets["train"].isdisjoint(sets["val"])
        assert sets["val"].isdisjoint(sets["test"])
        pd.testing.assert_frame_equal(result, make_patient_split(frame, seed=42))

    def test_preparation(self):
        frame = pd.DataFrame({"patient_id": ["synthetic_a", "synthetic_b"],
                              "cxr_filename": ["a.jpg", "b.jpg"], "age": [18, 90],
                              "sex": ["f", "m"], "slvh": [0, 1], "dlv": [0, 0],
                              "composite_slvh_dlv": [0, 1]})
        result = prepare_metadata(frame)
        self.assertEqual(result.age_scaled.tolist(), [0., 1.])
        self.assertEqual(result.sex_binary.tolist(), [0., 1.])

    def test_label_measurements_are_rejected(self):
        for feature in ("ivsd", "lvpwd", "lvidd", "slvh", "dlv", "composite_slvh_dlv"):
            with self.assertRaises(ValueError):
                validate_tabular_features([*SAFE_TABULAR_FEATURES, feature])

    def test_unknown_feature_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_tabular_features(["unreviewed_feature"])

    def test_pairing_respects_keys_not_row_order(self):
        dl = pd.DataFrame({"patient_id": ["synthetic_a", "synthetic_a"],
                           "cxr_filename": ["a.jpg", "b.jpg"], "dl_probability": [.1, .9]})
        ml = pd.DataFrame({"patient_id": ["synthetic_a", "synthetic_a"],
                           "cxr_filename": ["b.jpg", "a.jpg"], "ml_probability": [.8, .2]})
        self.assertEqual(align_branch_predictions(dl, ml).ml_probability.tolist(), [.2, .8])

    def test_missing_pair_is_rejected(self):
        dl = pd.DataFrame({"patient_id": ["synthetic_a"], "cxr_filename": ["a.jpg"], "dl_probability": [.1]})
        ml = pd.DataFrame({"patient_id": ["synthetic_b"], "cxr_filename": ["b.jpg"], "ml_probability": [.2]})
        with self.assertRaises(ValueError):
            align_branch_predictions(dl, ml)

    def test_duplicate_pair_is_rejected(self):
        dl = pd.DataFrame({"patient_id": ["synthetic_a"]*2, "cxr_filename": ["a.jpg"]*2,
                           "dl_probability": [.1, .2]})
        ml = pd.DataFrame({"patient_id": ["synthetic_a"], "cxr_filename": ["a.jpg"], "ml_probability": [.2]})
        with self.assertRaises(ValueError):
            align_branch_predictions(dl, ml)


if __name__ == "__main__":
    unittest.main()
