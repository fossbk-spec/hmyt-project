# Progress Test 2 — N12
## Glaucoma Detection Using the Cup-to-Disc Ratio in Fundus Images

| Field | Value |
|---|---|
| Course | ET4248 — Machine Learning in Healthcare |
| Team code | N12 |
| Project title | Glaucoma Detection Using the Cup-to-Disc Ratio in Fundus Images |
| Student 1 | Pham Hong Anh — 202414822 |
| Student 2 | Cao Thi Thanh Thao — 202414869 |
| GitHub fork URL | |
| Working branch | `gd5-proposed-model` |

---

# A. Literature Review

## A1. Search Strategy

The literature review focuses on automated optic disc (OD) and optic cup (OC) segmentation, vertical cup-to-disc ratio (vCDR) measurement, and glaucoma screening using retinal fundus images.

| Source | Search concepts | Period | Search results | Included studies |
|---|---|---:|---:|---:|
| PubMed | optic disc, optic cup, segmentation, glaucoma, fundus | 2019–2024 | Not recorded | Selected relevant studies |
| IEEE Xplore | optic disc segmentation, cup-to-disc ratio, deep learning | 2019–2024 | Not recorded | Selected relevant studies |
| Scopus | fundus image, glaucoma screening, U-Net, vCDR | 2019–2024 | Not recorded | Selected relevant studies |
| Google Scholar | automated optic disc and cup segmentation, CDR measurement | 2019–2024 | Not recorded | Selected relevant studies |

**Inclusion criteria**
- Studies published between 2019 and 2024 in journals or conferences.
- Studies addressing optic disc or optic cup segmentation, vCDR measurement, or glaucoma screening from color fundus photographs.
- Studies using machine learning or deep learning methods.
- Studies reporting quantitative evaluation metrics, such as Dice, IoU, vCDR error, AUC, sensitivity, or specificity.

**Exclusion criteria**
- Studies unrelated to fundus-image-based glaucoma assessment.
- Studies without a relevant automated analysis method.
- Studies that do not provide sufficient methodological or evaluation information.
- Non-primary sources, except reviews used to identify relevant research directions.

## A2. Literature Matrix

| No. | Study | Dataset / data source | Method | Main evaluation focus | External validation |
|---:|---|---|---|---|---|
| 1 | Orlando et al. (2020), REFUGE Challenge | REFUGE | Benchmarking of automated methods for optic disc/cup segmentation and glaucoma assessment | OD/OC segmentation and glaucoma-related measurements | Multiple methods evaluated under a common challenge protocol |
| 2 | Yu et al. (2019), robust optic disc and cup segmentation | RIGA, DRISHTI-GS, RIM-ONE | Deep-learning-based joint OD/OC segmentation | Segmentation performance and CDR-related assessment | Multiple datasets considered |
| 3 | Bian et al. (2020), anatomy-guided cascade network | Fundus-image datasets | Anatomy-guided cascaded segmentation | OD and OC segmentation | Dataset-dependent |
| 4 | Chaurasia et al. (2024), automated cup-to-disc ratio quantification | Fundus images | Automated OD/OC analysis and CDR quantification | CDR measurement for glaucoma screening | Dataset-dependent |
| 5 | Gao et al. (2024), automated vCDR measurement | Fundus images | Automated vCDR estimation | vCDR measurement | Dataset-dependent |
| 6 | Virbukaite et al. (2024), CNN ensemble | REFUGE, Drishti-GS, RIM-ONE | Ensemble of attention-based U-Net variants with majority voting | OD/OC segmentation and glaucoma-stage assessment | Yes, across separate dataset test sets |
| 7 | Liu et al. (2019), joint optic disc and cup segmentation using semi-supervised conditional GANs | ORIGA, REFUGE, RIM-ONE, DRISHTI-GS | Semi-supervised conditional GAN framework | OD/OC segmentation, CDR error, glaucoma discrimination | Multiple datasets used |
| 8 | Tian et al. (2020), graph convolutional network-based segmentation | REFUGE | Graph convolutional network | OD/OC segmentation | Dataset-dependent |
| 9 | Chen et al. (2024), MobileNetV3-based MBG-Net | Fundus images | Lightweight MobileNetV3-based segmentation network | Efficient OD/OC segmentation | Dataset-dependent |
| 10 | Kim et al. (2022), Mask R-CNN and MAPNet | Fundus images | Mask R-CNN and MAPNet-based approach | Optic disc/cup analysis | Dataset-dependent |
| 11 | Ma et al. (2024), review of automated glaucoma assessment | Published studies | Review of computational methods | Summary of methods and evaluation challenges | Not applicable |

## A3. Research Gaps

| Gap ID | Research gap | Evidence from literature | Project response |
|---|---|---|---|
| G1 | Optic cup segmentation is more challenging than optic disc segmentation because the cup boundary can be less distinct and has lower contrast. | Studies 1, 2, 6, and 7 | Report OD and OC Dice scores separately instead of relying only on an overall score. |
| G2 | The optic cup occupies a smaller image region than the background and optic disc, creating class-imbalance problems during training. | Studies 2, 3, 6, and 9 | Compare standard cross-entropy, class-weighted cross-entropy, and a combined weighted cross-entropy plus Dice loss. |
| G3 | Performance comparisons across studies are difficult because datasets, annotation protocols, preprocessing, and evaluation splits differ. | Studies 1–10 | Evaluate the experiments on the same RIM-ONE DL split and report the dataset and metrics explicitly. |

## A4. PROBAST+AI Appraisal of a Key Study

**Selected study:** Orlando et al. (2020), REFUGE Challenge.

| Domain | Appraisal | Rationale |
|---|---|---|
| Participants / data sources | Unclear | The study uses a defined challenge dataset, but the representativeness of the dataset for broader clinical populations must be considered. |
| Predictors | Low concern | The input consists of retinal fundus images, which are directly relevant to the intended screening task. |
| Outcome | Low concern | The challenge defines relevant outcomes for optic disc/cup segmentation and glaucoma assessment. |
| Analysis | Unclear | Multiple submitted methods are evaluated, but the analysis of each method must be considered separately when judging bias and generalizability. |
| Applicability | Some concerns | Challenge performance may not directly translate to performance in other clinical settings or on different fundus-camera populations. |

---

# B. Results and Project Timeline

## B1. Baseline and Proposed Model Results

### Dataset and evaluation protocol

| Item | Recorded value |
|---|---|
| Dataset | RIM-ONE DL |
| Total image-mask pairs | 485 |
| Image size | 256 × 256 pixels |
| Image format | RGB |
| Segmentation classes | Background (0), disc-only (1), cup (2) |
| Training set | 271 images |
| Validation set | 68 images |
| Test set | 146 images |
| Model | 3-class U-Net |

The dataset was divided into training, validation, and test subsets. Patient-level independence could not be verified because subject identifiers were not available in the recorded split information.

### Model architecture

The implemented model is a compact 3-class U-Net. Its encoder uses feature channels of 16, 32, and 64, followed by a 128-channel bottleneck. The decoder uses upsampling layers and skip connections to recover spatial information and produce a three-class segmentation mask.

### Validation results

| Experiment | Loss configuration | Optimizer | Learning rate | Epochs | Validation OD Dice | Validation OC Dice | Mean Dice |
|---|---|---|---:|---:|---:|---:|---:|
| Baseline | Cross-Entropy | Adam | 0.001 | 30 | 0.8276 | 0.3047 | 0.5661 |
| Experiment 2 | Weighted Cross-Entropy, weights [1, 1, 2] | Adam | 0.0003 | 15 | 0.8394 | 0.4149 | 0.6271 |
| Proposed configuration | Weighted Cross-Entropy + Dice Loss | Adam | 0.0001 | 15 | 0.8787 | 0.4667 | 0.6727 |

The combined weighted cross-entropy and Dice-loss configuration achieved the highest validation mean Dice score among the three recorded experiments. The improvement was particularly notable for optic cup segmentation.

### Independent test-set results

The selected configuration was evaluated on the 146-image test set.

| Structure | Test Dice |
|---|---:|
| Optic Disc (OD) | 0.890537 |
| Optic Cup (OC) | 0.462512 |
| Mean Dice | 0.676525 |

**Result files**
- `UNet_Results/final_test_summary.csv`
- `UNet_Results/test_per_image_dice.csv`
- `UNet_Results/combined_loss_experiment/training_history_combined.csv`
- `UNet_Results/combined_loss_experiment/best_unet_combined.pt`

## B2. Loss-Function Comparison

The experiments compared three training configurations:

1. Baseline U-Net trained with standard cross-entropy loss.
2. U-Net trained with class-weighted cross-entropy, using class weights `[1, 1, 2]`.
3. U-Net trained with a combination of class-weighted cross-entropy and Dice loss.

| Comparison | Mean validation Dice | Difference from baseline |
|---|---:|---:|
| Baseline Cross-Entropy | 0.5661 | — |
| Weighted Cross-Entropy | 0.6271 | +0.0610 |
| Weighted Cross-Entropy + Dice | 0.6727 | +0.1066 |

The combined-loss configuration produced the best recorded validation result. However, the learning rates and training durations differed between configurations. Therefore, this comparison should be interpreted as a comparison of training configurations rather than a strictly controlled experiment isolating the effect of the loss function alone.

## B3. Comparison with Published Studies

Published studies commonly report strong optic disc and optic cup segmentation results on benchmark datasets such as REFUGE, DRISHTI-GS, and RIM-ONE. The current project reports a test mean Dice of 0.6765, with a substantially higher OD Dice (0.8905) than OC Dice (0.4625).

Direct numerical comparison should be interpreted cautiously because studies may differ in:
- Dataset version and image composition.
- Training, validation, and test partitioning.
- Annotation protocols and mask definitions.
- Image resolution and preprocessing.
- Model architecture and training configuration.
- Whether results are reported on validation or independent test sets.

The relatively low OC Dice indicates that optic cup segmentation remains the main technical challenge in the current implementation.

## B4. Project Timeline

| Milestone | Planned timing | Main tasks | Status |
|---|---|---|---|
| M3 | Week 10 | Complete literature review, dataset preparation, baseline implementation, and initial experiments. | Completed |
| M4 | Week 12 | Compare loss configurations, select the best model, evaluate on the test set, and prepare the progress report. | Completed |
| Final | Week 15 | Improve cup segmentation, complete CDR calculation, conduct error analysis, and finalize the report and presentation. | In progress |

## B5. Risks, Blockers, and Open Questions

| Risk / limitation | Potential impact | Planned action |
|---|---|---|
| Lower optic cup segmentation performance | May reduce the reliability of subsequent CDR estimation. | Inspect difficult cases and investigate loss weighting, preprocessing, and model improvements. |
| Patient-level split is not verified | Possible data leakage if images from the same subject appear in different subsets. | Check dataset metadata and split-generation records. |
| CDR calculation from predicted masks is pending | The current segmentation results do not yet demonstrate a complete automated CDR pipeline. | Implement vertical cup and disc diameter extraction and validate the measurements. |
| CDR error metrics are not yet evaluated | Measurement accuracy cannot yet be quantified. | Evaluate MAE, RMSE, and correlation against reference CDR values. |
| Different learning rates and epoch counts across experiments | The loss-function comparison is not a strictly controlled ablation. | Conduct controlled experiments with matched training settings. |
| No external validation | Generalization to other datasets and clinical settings remains unknown. | Consider evaluation on an independent fundus-image dataset if permitted. |

---

# C. FAIR, TRIPOD+AI, and PROBAST+AI

## C1. Dataset Documentation and FAIR Principles

| Item | Description |
|---|---|
| Dataset name | RIM-ONE DL |
| Data type | Retinal color fundus photographs and optic disc/cup segmentation masks |
| Number of image-mask pairs | 485 |
| Image dimensions used | 256 × 256 pixels |
| Image format | RGB |
| Mask labels | 0 = background; 1 = disc-only; 2 = cup |
| Dataset version | RIM-ONE DL; exact release/version identifier not recorded |
| Download date | Not recorded |
| Checksum | Not recorded |
| Licence / data-use agreement | Must follow the dataset provider's terms |
| Patient identifiers | No patient-level identifiers were available in the recorded split information |

### Dataset limitations
- The available split information does not establish patient-level independence.
- The dataset represents a specific collection of fundus images and may not capture all variations in imaging devices or clinical populations.
- Some masks were previously flagged for cup pixels outside the optic disc. These cases should be reviewed and documented as part of quality control.
- External validation on an independent dataset has not been completed.

## C2. Research Question and Reporting Guideline

**Primary research question**

Can a U-Net-based model accurately segment the optic disc and optic cup in retinal fundus images from the RIM-ONE DL dataset?

**Secondary research question**

Do class-weighted cross-entropy and combined weighted cross-entropy plus Dice loss improve optic cup segmentation compared with standard cross-entropy loss?

**Selected reporting framework:** TRIPOD+AI, used as a supplementary reporting reference.

TRIPOD+AI is primarily intended for prediction-model studies. Since the current project focuses on image segmentation, not all checklist items are directly applicable. Segmentation-specific reporting should also describe the dataset, annotations, preprocessing, model architecture, training procedure, evaluation metrics, and test-set protocol.

| Reporting item | Current status |
|---|---|
| Research objective and intended use | Reported |
| Dataset source and sample size | Reported |
| Input data and target labels | Reported |
| Model architecture | Reported |
| Training configurations | Reported |
| Evaluation metrics | Reported |
| Data-splitting procedure | Counts reported; split-generation details require verification |
| Patient-level separation | Not verified |
| External validation | Not completed |
| CDR measurement performance | Pending |
| Clinical utility assessment | Not completed |

## C3. PROBAST+AI Self-Appraisal

| Domain | Current concern | Evidence / explanation | Risk-reduction action |
|---|---|---|---|
| Participants / data sources | Some concerns | RIM-ONE DL contains 485 image-mask pairs, but patient-level independence and population representativeness are not established. | Document dataset characteristics and verify the split protocol where possible. |
| Predictors | Low concern | The model uses RGB fundus images as inputs. | Document image preprocessing and ensure that all transformations are applied consistently. |
| Outcome | Some concerns | OD and OC masks provide segmentation targets, but annotation quality and anomalous masks require review. | Audit masks and document label definitions and quality-control procedures. |
| Analysis | Some concerns | Three training configurations were compared, but learning rates and epoch counts differed. | Run controlled comparisons and report uncertainty and per-structure metrics. |
| Applicability | Some concerns | Results are based on one dataset and do not yet establish clinical performance. | Conduct external validation and evaluate CDR measurement against reference values. |

## C4. Reproducibility and Reporting Checklist

- [x] Dataset name and total number of image-mask pairs reported.
- [x] Image dimensions and segmentation label definitions reported.
- [x] Training, validation, and test sample counts reported.
- [x] Baseline model and proposed training configurations described.
- [x] OD and OC Dice scores reported separately.
- [ ] Exact dataset release identifier and download date recorded.
- [ ] Dataset checksum recorded.
- [ ] Patient-level independence verified.
- [ ] Preprocessing and augmentation fully documented from the implementation.
- [ ] All anomalous masks reviewed and documented.
- [ ] Controlled loss-function comparison completed.
- [ ] Automated CDR calculation implemented and evaluated.
- [ ] External validation completed.

AUC, calibration, and Brier score are not primary metrics for the current segmentation-only experiments. They may become relevant if a separate glaucoma classification model is developed.

---

# D. AI Use Declaration

| Tool | Purpose |
|---|---|
| ChatGPT | Assistance with report organization, literature-review structuring, and explanation of machine-learning concepts. |
| Google Colab | Model development and experimentation. |
| Python | Data processing, model training, evaluation, and visualization. |
| PyTorch | U-Net implementation and model training. |
| GitHub | Version control and project-file management. |

All reported experimental results should be checked against the corresponding source code, logs, and result files before submission. The team is responsible for verifying the correctness of the final report and all cited literature.
