# Cardiac structure screening from chest X rays

Based on Machine learning and deep learning

Group **N17**. Members: **202414865, 202414821, 202414845**.

The project combines a five-feature metadata model with EfficientNet-B0 for
paired chest X-ray examinations. Logistic regression combines their two
probabilities. The target is the CheXchoNet composite label for severe left
ventricular hypertrophy or a dilated left ventricle, not all cardiovascular disease.

## Current results

The current aggregate test export gives the following results. These are
image/examination-level scores from the supplied Colab output, not the earlier
local metadata validation measurements.

| Model | Precision | Recall | F2 | ROC AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Metadata ML | 0.1427 | 0.9986 | 0.4540 | 0.6059 | 0.1919 |
| EfficientNet-B0 | 0.2425 | 0.7025 | 0.5093 | 0.7381 | 0.3141 |
| ML and DL fusion | 0.2309 | 0.7727 | 0.5258 | 0.7413 | 0.3172 |

Fusion increases recall by 7.01 percentage points versus the image branch, with
a 1.17-point precision decrease. The small AP increase does not establish
statistical significance. Validation AP peaks in the first of four recorded
epochs, then decreases. The notebook fits fusion and selects its threshold on
the same validation observations; this limitation remains in the supplied run.

The three original Colab exports are preserved byte-for-byte in `results/`.
`colab_results_receipt.json` records their hashes and the consistency checks.
Per-image predictions, checkpoint hashes and the exact Colab environment were
not supplied, so independent AUROC/AP recalculation, patient-bootstrap confidence
intervals and checkpoint verification are still outstanding.

## Submission status

This is a prepared **M3** snapshot, not a claim that every final rubric item is
complete. As confirmed by the submitting team, `topic_id` is the project title:
`CheXchoNet Cardiac Structure Screening based on Machine learning and deep learning`.
The descriptive slug is `x_quang_metadata`. This custom project does not have a
topic-specific profile in the class repository, so its advisory review uses the
general rubric. It is not topic 3.4, the separate 299-patient tabular heart-failure lab.

Individual CRediT roles also require confirmation by the members. No role or
contribution percentage has been invented.

## Contents

- `report/N17-CheXchoNet-Report.md`: readable report, including measured results and limitations.
- `report/N17-CheXchoNet-Report-Public.docx`: report in the retained HUST format.
- `report/figures/`: aggregate figures and the model architecture.
- `src/`: Colab notebook, data utilities, and local reproducibility helpers.
- `slides/`: HUST presentation updated with the Colab exports.
- `results/`: aggregate evidence only, without patient records.
- `docs/`: data access, AI disclosure, and evidence limitations.
- `tests/`: synthetic unit tests for pairing and partition separation.

## Dataset and access

Use [CheXchoNet version 1.0.0](https://physionet.org/content/chexchonet/1.0.0/),
DOI [10.13026/kp08-ws25](https://doi.org/10.13026/kp08-ws25).
Obtain access through the official source and comply with its current terms.
Read the dataset's own `LICENSE.txt`. This submission does not redistribute
X-rays, metadata rows, authentication cookies or access credentials.

The audited local manifest has 71,589 images from 24,689 patients. Patient-level
partitions contain 17,282 training, 3,703 validation and 3,704 test patients.
Multiple images from one patient stay in a single partition. The five metadata
predictors are age, sex, heart transplant, lung transplant, and pacemaker/ICD.
Echo measurements that define the labels are excluded as predictors.

## Reproduce the Colab workflow

1. Place the officially obtained dataset ZIP or extracted directory in
   `MyDrive/Hoc_may_2_0/DL_Xray/CheXchoNet/` on your own Google Drive.
2. Open `src/N17_CheXchoNet_ML_Support_DL_Colab.ipynb` in Colab and select a GPU.
3. Run the cells in order. Authorise Drive mounting in your own account. The
   notebook contains the data helper functions and does not require an external
   helper upload. Never add a password or token to its cells.
4. The notebook writes to
   `MyDrive/Hoc_may_2_0/DL_Xray/CheXchoNet/outputs_ml_support_dl/`.
5. Archive `test_metrics.csv`, `dl_training_history.csv`,
   `selected_thresholds.json`, the model checkpoint and test predictions privately.
   Do not upload row-level predictions or large weights to this class repository.
6. Record the runtime versions and checkpoint/split hashes. The exported notebook
   is an unexecuted source copy; the three CSV/JSON exports are the available
   evidence of the reported Colab run.

The notebook checks missing packages but does not lock Colab's installed versions.
`requirements-colab-target.txt` gives a proposed compatible target, not a recovered
lockfile for the historical Colab run. The historical Colab environment cannot
be reconstructed from three aggregate files alone.

## Local checks

Use Python 3.12 and install the metadata-only environment separately from Colab:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/verify_colab_exports.py --results results
```

To repeat only the metadata baseline, pass your private manifest path and an
output directory outside the submission:

```powershell
python src/reproduce_metadata.py --metadata "D:\Hoc_may_2_0\DL_Xray\CheXchoNet\processed\metadata_with_split.csv" --output-dir "D:\Hoc_may_2_0\private_reproduction"
```

This helper computes validation scores only and never evaluates test outcomes.
The local package versions are pinned to the inspected environment; they are
not evidence of the Colab package versions.

From the class repository root:

```powershell
python scripts/validate_submission.py submissions/x_quang_metadata_n17
python scripts/review_project.py submissions/x_quang_metadata_n17 --milestone m3
```

If the instructor later requests a different slug, rename the folder and update
`submission.json` and those commands consistently. A technical validator pass
does not establish that every rubric criterion is satisfied.

## References and integrity

Report references are in `report/references.bib`. The report and
`docs/ai_disclosure_log.md` disclose AI assistance. Do not publish dataset files,
patient identifiers, full student names, secrets, or files larger than 10 MB.
The research prototype has not demonstrated clinical deployment readiness.
