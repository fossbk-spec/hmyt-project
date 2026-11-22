# Data and evidence scope

Dataset: CheXchoNet 1.0.0, official source
https://physionet.org/content/chexchonet/1.0.0/ .

This repository contains no dataset rows or X-ray images. Public aggregate plots
summarise prevalence, validation decisions and supplied Colab scores. Dataset
access and reuse are governed by the official resource and its own license.

The local split audit reports no overlap of patient identifiers across train,
validation and test. The notebook splits patients, then maps all their X-rays to
the same partition. Preprocessors fit training rows only. Fusion aligns both
branches on patient_id and cxr_filename and rejects missing or duplicate pairs.
The task does not use blood laboratory results or label-defining echo values as
metadata predictors.

`results/validation_receipt.json` is the original local metadata receipt. Its
notebook hash refers to the historical source, not the formatting-sanitised
public notebook. `public_source_hashes.json` records the public source hashes.

`results/test_metrics.csv`, `dl_training_history.csv` and
`selected_thresholds.json` are the user's Colab exports, imported on 1 November
2026. Import time is not evidence of the training date. Their bytes are preserved.
The separate import receipt records the SHA256 values and consistency checks.

The provided aggregate exports do not contain test sample size, record keys,
class support, checkpoint identity or runtime versions. We do not infer a test
confusion matrix or a confidence interval from an assumed cohort. The checkpoint,
per-examination predictions and split/runtime receipt should be retained privately.

Known limitations include sparse metadata, image-level weighting of patients
with repeated examinations, fusion fit/threshold reuse of the validation set,
falling image validation AP after epoch 1, and no demonstrated external or
prospective clinical validation. A small numerical fusion gain is not proof of
statistical significance, calibrated risk or clinical benefit.
