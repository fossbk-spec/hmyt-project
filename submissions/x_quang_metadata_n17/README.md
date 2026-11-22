# CheXchoNet cardiac structure screening

Based on Machine learning and deep learning

Group N17. Members: 202414821, 202414865, 202414845.

## HW12–15 submission

Snapshot: 22 November 2026. The project pairs chest X-rays with metadata from
the same examination to screen for severe left ventricular hypertrophy or
left ventricular dilation. It does not predict all cardiovascular diseases.

- Current slides: [HW12–15 project review](slides/N17-CheXchoNet-HW12-15-Project-Review-Public.pptx).
- Current [IEEE Word report](report/HW12-15/N17-CheXchoNet-IEEE-Report-Public.docx) and [readable report](report/N17-CheXchoNet-Report.md).
- [Data and reporting audit](report/HW12-15/N17-CheXchoNet-Data-and-Reporting-Audit-Public.docx) and [TRIPOD+AI evidence map](report/HW12-15/N17-CheXchoNet-TRIPOD-AI-Evidence-Map.csv).
- [V2 source, notebooks, tests and aggregate results](src/HW12-15/).
- [Dataset access and summary](docs/data_access/).
- [Reference PDFs and source inventory](report/HW12-15/references/).

Earlier Week 7/M3 presentations, source files and results remain as historical
artifacts. Use the HW12–15 paths above for the current submission.

## Measured results and limits

| Experiment | AP | ROC-AUC | F2 | Scope |
| --- | ---: | ---: | ---: | --- |
| Historical metadata ML | 0.1919 | 0.6059 | 0.4540 | Earlier Colab test export |
| Historical EfficientNet-B0 | 0.3141 | 0.7381 | 0.5093 | Earlier Colab test export |
| Historical late fusion | 0.3172 | 0.7413 | 0.5258 | Earlier Colab test export |
| Week 12 selected metadata LR | 0.1462 | 0.5220 | 0.2000 | Development audit only |

The Week 12 search compared 12 candidates with three patient-group folds within
the training partition. It selected age-only logistic regression, then refitted
the classifier. Calibration and threshold fitting used separate patient groups.
On the previously viewed development group, calibrated Brier score was 0.1159,
recall 0.2170 and specificity 0.8133. The audit includes 500 whole-patient
bootstrap resamples, conditional on the fitted model. These development values
are not independent final-test scores or a direct comparison with the old run.

The team confirmed that V2 GPU training has not run. There are no new V2 DL or
fusion results. Historical fusion reused its validation observations for fitting
and threshold selection; only three aggregate Colab exports are available
locally, without the original weights or examination-level predictions.
The small historical fusion gain has no demonstrated statistical significance.
Clinical utility, external validation and subgroup evidence remain unestablished.

## Reproduction

1. Obtain [CheXchoNet 1.0.0](https://physionet.org/content/chexchonet/1.0.0/)
   through PhysioNet and comply with its access terms and licence.
2. Open `src/HW12-15/N17-CheXchoNet-V2-ML-Support-DL-Colab.ipynb` in Colab.
   Use a GPU and the already uploaded official dataset under your private Drive.
3. For metadata-only reproduction, install `src/HW12-15/requirements-metadata.txt`.
   From `src/HW12-15`, run `python -m unittest discover -s tests -v`, then follow
   `N17-CheXchoNet-Run-Instructions.txt` for the two metadata scripts.
4. Keep patient rows, images, role manifests, predictions, model weights and
   credentials private. Public results in this submission contain aggregates only.

The audited cohort contains 71,589 images from 24,689 patients, including 9,861
positive images. Splits keep each patient in one role. ML and DL predictions
pair by patient ID and CXR filename. V2 defaults to age and sex; the separate
Week 12 selection retained age only. Label-defining echo measurements are excluded.

## Reporting status

`milestone: final` identifies the requested review stage, not completion of all
scientific validation. The project title is the registered topic identifier;
the custom slug `x_quang_metadata` uses the general rubric.

References 01 (Bhave 2024) and 02 (Huang 2020) have official source links but
their PDFs could not be downloaded in this environment. The inventory states
this explicitly. Reference 04 is a dataset record. Member contributions,
funding/conflicts and Mendeley account actions have not been independently
confirmed. See the audit for remaining reporting gaps.

Code and documentation received AI assistance, disclosed in
`docs/ai_disclosure_log.md`. The team remains responsible for checking all claims.
## Local smoke test

```powershell
cd src/HW12-15
python -m pip install -r requirements-metadata.txt
python -m unittest discover -s tests -v
```

The test uses synthetic records. Actual training additionally requires the official private dataset.
