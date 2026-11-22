"""Core data utilities for the CheXchoNet Week 5-6 workflow.

The model notebook imports this module.  The helpers deliberately keep patient
splitting separate from image loading so they can be audited without a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_COLUMNS = ("slvh", "dlv", "composite_slvh_dlv")
SAFE_TABULAR_FEATURES = (
    "age_scaled",
    "sex_binary",
    "heart_transplant",
    "lung_transplant",
    "pacemaker_or_icd",
)
LEAKAGE_COLUMNS = (
    "ivsd",
    "lvpwd",
    "lvidd",
    *LABEL_COLUMNS,
)
SAFE_BINARY_COLUMNS = (
    "heart_transplant",
    "lung_transplant",
    "pacemaker_or_icd",
)
REQUIRED_COLUMNS = (
    "patient_id",
    "cxr_filename",
    "age",
    "sex",
    *LABEL_COLUMNS,
)


def find_dataset_root(search_root: Path | str) -> Path:
    """Return the directory containing both metadata.csv and images/."""
    search_root = Path(search_root)
    candidates = [search_root, *search_root.rglob("metadata.csv")]
    for candidate in candidates:
        root = candidate if candidate.is_dir() else candidate.parent
        if (root / "metadata.csv").is_file() and (root / "images").is_dir():
            return root
    raise FileNotFoundError(
        f"Could not find metadata.csv and images/ below: {search_root}"
    )


def prepare_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, remove unusable rows and create demographic features."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    cleaned = frame.copy()
    cleaned["patient_id"] = cleaned["patient_id"].astype(str).str.strip()
    cleaned["cxr_filename"] = cleaned["cxr_filename"].astype(str).str.strip()
    cleaned["sex"] = cleaned["sex"].astype(str).str.upper().str.strip()
    cleaned["age"] = pd.to_numeric(cleaned["age"], errors="coerce")
    for label in LABEL_COLUMNS:
        cleaned[label] = pd.to_numeric(cleaned[label], errors="coerce")

    valid = (
        cleaned["patient_id"].ne("")
        & cleaned["cxr_filename"].str.lower().str.endswith((".jpg", ".jpeg", ".png"))
        & cleaned["age"].between(18, 90)
        & cleaned["sex"].isin(["F", "M"])
    )
    for label in LABEL_COLUMNS:
        valid &= cleaned[label].isin([0, 1])

    cleaned = cleaned.loc[valid].reset_index(drop=True)
    cleaned["sex_binary"] = cleaned["sex"].map({"F": 0.0, "M": 1.0})
    cleaned["age_scaled"] = (cleaned["age"] - 18.0) / 72.0
    for column in SAFE_BINARY_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = np.nan
        else:
            values = pd.to_numeric(cleaned[column], errors="coerce")
            cleaned[column] = values.where(values.isin([0, 1]))
    return cleaned


def validate_tabular_features(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Validate that the classical ML branch uses only approved predictors."""
    features = tuple(feature_names)
    leaked = sorted(set(features).intersection(LEAKAGE_COLUMNS))
    if leaked:
        raise ValueError(
            "Features contain target leakage: " + ", ".join(leaked)
        )
    unsupported = sorted(set(features) - set(SAFE_TABULAR_FEATURES))
    if unsupported:
        raise ValueError(
            "Unsupported tabular features: " + ", ".join(unsupported)
        )
    return features


def build_tabular_records(
    frame: pd.DataFrame,
    label: str = "composite_slvh_dlv",
    feature_names: Iterable[str] = SAFE_TABULAR_FEATURES,
) -> pd.DataFrame:
    """Return one ML row per X-ray encounter, aligned by patient and filename."""
    features = validate_tabular_features(feature_names)
    required = {"patient_id", "cxr_filename", label, *features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing tabular columns: {', '.join(missing)}")

    keys = ["patient_id", "cxr_filename"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate patient_id and cxr_filename keys")

    columns = [*keys]
    if "split" in frame.columns:
        columns.append("split")
    columns.extend(features)
    columns.append(label)
    return frame.loc[:, columns].copy().reset_index(drop=True)


def align_branch_predictions(
    dl_predictions: pd.DataFrame,
    ml_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Align image and metadata probabilities for the same X-ray encounter."""
    keys = ["patient_id", "cxr_filename"]
    requirements = (
        ("DL", dl_predictions, "dl_probability"),
        ("ML", ml_predictions, "ml_probability"),
    )
    for branch_name, frame, probability_column in requirements:
        missing = sorted(set([*keys, probability_column]) - set(frame.columns))
        if missing:
            raise ValueError(
                f"{branch_name} predictions are missing columns: {', '.join(missing)}"
            )
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate {branch_name} prediction keys")

    dl_keys = set(map(tuple, dl_predictions[keys].astype(str).to_numpy()))
    ml_keys = set(map(tuple, ml_predictions[keys].astype(str).to_numpy()))
    if dl_keys != ml_keys:
        raise ValueError("ML and DL prediction keys do not match")

    return dl_predictions.merge(
        ml_predictions[[*keys, "ml_probability"]],
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )


def _patient_table(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if label not in frame.columns:
        raise ValueError(f"Unknown label column: {label}")
    return (
        frame.groupby("patient_id", as_index=False)[label]
        .max()
        .rename(columns={label: "patient_label"})
    )


def _safe_stratify(values: Iterable[int]) -> np.ndarray | None:
    values = np.asarray(list(values))
    _, counts = np.unique(values, return_counts=True)
    return values if len(counts) > 1 and counts.min() >= 2 else None


def make_patient_split(
    frame: pd.DataFrame,
    *,
    label: str = "composite_slvh_dlv",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val/test at patient level, approximately stratified by label."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1")

    patients = _patient_table(frame, label)
    train_patients, holdout = train_test_split(
        patients,
        train_size=train_size,
        random_state=seed,
        stratify=_safe_stratify(patients["patient_label"]),
    )
    relative_val = val_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(
        holdout,
        train_size=relative_val,
        random_state=seed + 1,
        stratify=_safe_stratify(holdout["patient_label"]),
    )

    split_map = {
        **{patient: "train" for patient in train_patients["patient_id"]},
        **{patient: "val" for patient in val_patients["patient_id"]},
        **{patient: "test" for patient in test_patients["patient_id"]},
    }
    result = frame.copy()
    result["split"] = result["patient_id"].map(split_map)
    if result["split"].isna().any():
        raise RuntimeError("At least one patient was not assigned to a split")
    return result


def compute_pos_weight(frame: pd.DataFrame, label: str) -> float:
    """Return N_negative / N_positive for BCEWithLogitsLoss(pos_weight=...)."""
    values = pd.to_numeric(frame[label], errors="coerce").dropna()
    positives = int((values == 1).sum())
    negatives = int((values == 0).sum())
    if positives == 0:
        raise ValueError(f"Label {label!r} has no positive examples")
    return negatives / positives


def resolve_image_path(row: pd.Series, images_dir: Path | str) -> Path:
    """Resolve an image from cxr_filename; cxr_path in the CSV is legacy metadata."""
    return Path(images_dir) / str(row["cxr_filename"])


def audit_split(frame: pd.DataFrame, label: str = "composite_slvh_dlv") -> pd.DataFrame:
    """Create a compact count/prevalence table for slides and reports."""
    rows = []
    for split_name, part in frame.groupby("split", sort=False):
        rows.append(
            {
                "split": split_name,
                "images": len(part),
                "patients": part["patient_id"].nunique(),
                "positive": int(part[label].sum()),
                "prevalence": float(part[label].mean()),
            }
        )
    return pd.DataFrame(rows)
