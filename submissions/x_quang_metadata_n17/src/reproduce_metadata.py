"""Reproduce the notebook's metadata comparison on validation data only."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chexchonet_pipeline_ml_support_dl import (
    SAFE_TABULAR_FEATURES, prepare_metadata, make_patient_split,
    build_tabular_records, validate_tabular_features,
)


def main(metadata_path, output_dir):
    output_dir = output_dir.resolve()
    submission = Path(__file__).resolve().parents[1]
    if output_dir.is_relative_to(submission):
        raise ValueError("Choose a private output directory outside the submission")
    frame = pd.read_csv(metadata_path, dtype={"patient_id": str})
    if not set(SAFE_TABULAR_FEATURES).issubset(frame.columns):
        frame = prepare_metadata(frame)
    if "split" not in frame:
        frame = make_patient_split(frame, seed=42)
    assert frame.groupby("patient_id")["split"].nunique().max() == 1
    groups = {s: set(frame.loc[frame["split"].eq(s), "patient_id"]) for s in ("train", "val", "test")}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    SEED, TARGET = 42, "composite_slvh_dlv"
    OUTPUT_DIR = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = frame
    display = lambda value: print(value.to_string(index=False))
    notebook_path = Path(__file__).with_name("N17_CheXchoNet_ML_Support_DL_Colab.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][10]["source"])
    # Retain the original baseline algorithm but do not even predict the test rows.
    source = source.replace("ml_test_predictions = predict_ml(test_rows)", "")
    scope = dict(globals(), **locals())
    exec(compile(source, str(notebook_path)+"#cell10", "exec"), scope)
    scope["ml_validation"].to_csv(output_dir/"validation_metrics.csv", index=False)
    receipt = {"scope": "Training fits models; validation selects model/threshold; no test scores",
               "python": platform.python_version(), "sklearn": sklearn.__version__, "seed": SEED,
               "manifest_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
               "public_notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
               "selected_model": scope["BEST_ML_NAME"],
               "validation_results": scope["ml_validation"].to_dict(orient="records")}
    (output_dir/"metadata_reproduction_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("Private outputs:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.metadata, args.output_dir)
