"""Check aggregate export consistency; never invent unavailable patient-level results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def verify(directory):
    directory = Path(directory)
    with (directory / "test_metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with (directory / "dl_training_history.csv").open(encoding="utf-8-sig", newline="") as f:
        history = list(csv.DictReader(f))
    thresholds = json.loads((directory / "selected_thresholds.json").read_text())
    expected_models = {"ML: HistGradientBoosting": "ml_threshold",
                       "DL: EfficientNet-B0": "dl_threshold",
                       "Fusion: ML supports DL": "fusion_threshold"}
    assert len(rows) == 3 and {r["model"] for r in rows} == set(expected_models)
    for row in rows:
        for field in ("threshold", "precision", "recall", "f2", "roc_auc", "pr_auc"):
            value = float(row[field])
            assert math.isfinite(value) and 0 <= value <= 1, field
        precision, recall = float(row["precision"]), float(row["recall"])
        score = 5 * precision * recall / (4 * precision + recall)
        assert math.isclose(score, float(row["f2"]), abs_tol=1e-12), row["model"]
        assert math.isclose(float(row["threshold"]), thresholds[expected_models[row["model"]]], abs_tol=1e-12)
    assert history and [int(r["epoch"]) for r in history] == list(range(1, len(history)+1))
    for row in history:
        assert math.isfinite(float(row["train_loss"])) and float(row["train_loss"]) >= 0
        assert 0 <= float(row["val_pr_auc"]) <= 1
    best = max(history, key=lambda r: float(r["val_pr_auc"]))
    receipt = directory / "colab_results_receipt.json"
    if receipt.exists():
        hashes = json.loads(receipt.read_text())["source_sha256"]
        for name in ("test_metrics.csv", "dl_training_history.csv", "selected_thresholds.json"):
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == hashes[name], name
    return {"models": len(rows), "epochs": len(history), "best_validation_epoch": int(best["epoch"]),
            "checks": "F2, thresholds, numeric ranges, epoch order and available source hashes agree",
            "not_verified": "AUROC/AP from raw predictions, exact test cohort, checkpoint and confidence intervals"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().results), indent=2))
