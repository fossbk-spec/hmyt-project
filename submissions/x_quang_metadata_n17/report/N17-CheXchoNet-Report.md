# CheXchoNet cardiac structure screening

HW12–15 snapshot, 22 November 2026. Group N17: 202414821, 202414865, 202414845.

This readable copy follows the current IEEE manuscript. The accompanying Word file preserves the equations, figures and two-column layout. Historical DL/fusion exports and the new metadata development audit are separate experiments.

Development of CheXchoNet Cardiac Structure Screening

based on Machine learning and deep learning

202414821    202414865    202414845

School of Electrical and Electronic Engineering, Hanoi University of Science and Technology

Group N17   |   ET4248   |   Instructor Dr. Hoang Quang Huy

Abstract—This study develops a paired chest X-ray and metadata pipeline for screening severe left ventricular hypertrophy or left ventricular dilatation. CheXchoNet provides 71,589 examinations from 24,689 patients. An earlier experiment combined metadata ML and EfficientNet-B0 through logistic late fusion, with exported fusion F2 0.5258 and average precision 0.3172. V2 separates patients used for branch selection, fusion, calibration and threshold choice. A new training-group cross-validation experiment compared 12 metadata configurations; age-only Logistic Regression was selected and retrained. Its development average precision was 0.1462 and ROC-AUC 0.5220, with a 95% patient-bootstrap interval of 0.4547–0.5882 for ROC-AUC. Sigmoid calibration reduced Brier score from 0.2474 to 0.1159 on the development group, but weak discrimination remained. New V2 image training and real-image fusion have not been completed. The work supports an auditable research workflow rather than a clinically validated diagnostic system.

Keywords—cardiac screening; chest radiography; metadata fusion; probability calibration; patient separation

I. INTRODUCTION

Chest radiographs provide an opportunity to investigate structural cardiac abnormalities using examinations already acquired in routine care. Bhave et al. linked radiographs to echocardiographic measurements and reported discrimination for severe left ventricular hypertrophy (SLVH), dilated left ventricle (DLV) and their composite [1]. Their results motivate this task; they are not performance estimates for our implementation.

A review of imaging and electronic-record fusion describes early, joint and late strategies, with heterogeneous evidence across tasks [2]. Pal et al. examine demographic and anatomical encoding in CheXchoNet SLVH classifiers [3]. Their balanced SLVH-only experiments cannot be compared directly with average precision at the natural prevalence of our composite endpoint.

Our question is whether metadata provides useful support to an image model when both observations belong to the same examination. We compare metadata-only, image-only and late-fusion outputs, while separating evidence from the earlier Colab run and the revised V2 protocol. The Week 12 extension adds measured metadata tuning, feature selection and retraining. It does not establish that new fusion improves performance.

II. METHODOLOGY

A. Dataset and clinical target

CheXchoNet v1.0.0 supplies PA radiographs, metadata and echo-derived labels [4]. The source cohort is from Columbia University Irving Medical Center, with radiographs collected during 2013–2018 and echo linkage within 12 months [1]. The audited local release contains 71,589 examinations and 24,689 adults aged 18–90. Each row represents an examination; patients have variable numbers of images. The target y is the supplied binary composite SLVH OR DLV, with 9,861 positive images (13.77%). No ECG sampling rate applies to this image dataset.

Prediction time T0 is defined as availability of the acquired radiograph and its recorded age and sex, before using echocardiographic outcome information. The dataset is not a complete laboratory panel. IVSD, LVPWD and LVIDD define outcome information and are excluded as predictors. Transplant and pacemaker/ICD flags are reserved for a retrospective extension because their availability at T0 is unverified.

TABLE I. PATIENT ROLES IN V2

| Role | Images | Patients |
| --- | --- | --- |
| Train | 50,233 | 17,282 |
| Branch selection | 2,548 | 925 |
| Fusion fitting | 3,956 | 1,296 |
| Calibration | 2,254 | 741 |
| Threshold | 2,212 | 741 |
| Historical test | 10,386 | 3,704 |

Original train and historical-test assignments are preserved. Original validation patients are subdivided 25/35/20/20% into branch selection, fusion fitting, calibration and threshold choice. Every patient has one role. The local audit found zero missing referenced images, duplicate joint keys, cross-role patients or inconsistencies between the composite and component labels. These checks do not validate the clinical accuracy of labels.

B. Preprocessing and feature extraction

Metadata features are age_scaled = (age − 18)/72 and sex_binary, encoded F = 0 and M = 1. Imputation and standardization fit only training observations. The image pipeline converts each image to three channels, resizes to 256 pixels, center-crops to 224 × 224 and uses ImageNet normalization. Training adds rotation up to five degrees and no horizontal flip. Crop adequacy requires further visual review.

The image branch uses pretrained EfficientNet-B0 [5]. Let fθ(I) be its final convolutional activation and z = GAP(fθ(I)) its pooled 1,280-dimensional image representation. A one-logit classifier produces pDL = sigmoid(wᵀz + b). These learned image features replace ECG-specific spectral descriptors. Source code implements extraction through torchvision; no LightGBM feature bundling is used for the small metadata vector.

Fig. 1. Metadata and matched image branches with late fusion and calibration.

C. Training and metadata optimization

The metadata candidates are Logistic Regression (LR) with class weighting and HistGradientBoosting (HGB) with balanced training weights. LR minimizes weighted binary log loss plus L2 regularization. The image branch minimizes weighted binary cross entropy, with training positive weight 43,393/6,840 ≈ 6.344. Its planned optimizer is AdamW with learning rate 10⁻⁴, weight decay 10⁻⁴, batch 32, at most 10 epochs and early-stopping patience 3. The checkpoint is selected by branch-selection AP.

Week 12 uses three-fold StratifiedGroupKFold entirely within training patients. Nine LR candidates combine C ∈ {0.1, 1, 10} with age-and-sex, age-only or sex-only features. Three HGB candidates use (learning rate, leaves) ∈ {(0.05, 7), (0.05, 15), (0.1, 15)}, with 120 iterations and age-and-sex inputs. Mean fold AP selects the candidate and feature set. The selected model is refitted on all training examinations; branch-selection data are not used in cross-validation fitting.

Feature-group ablation supplies a selection analysis suited to two source variables. Permutation importance reports the decrease in development AP when the retained feature is shuffled ten times. It describes predictive reliance, not causation or a clinically independent importance estimate. The Week 12 selected model is saved separately and has not yet been integrated into a new image-fusion run.

D. Fusion calibration and operating points

Branch predictions are joined one-to-one using both patient_id and cxr_filename; labels and key sets must agree. Unweighted LR combines standardized branch scores on fusion-fitting patients only. Separate unweighted sigmoid calibrators then fit on calibration patients, using logit-transformed raw scores. Calibration assessment follows the distinction between ranking and probability reliability [6].

The threshold group chooses the largest F2 subject to observed specificity ≥0.80. This is an experimental constraint, not a guarantee in another cohort or an endorsed referral policy. F2 = 5TP/(5TP + 4FN + FP). For the measured Week 12 metadata audit, only the selected metadata pipeline is calibrated and assessed; no image output is substituted or simulated.

E. Evaluation and reproducibility

AP uses average_precision_score, not trapezoidal PR-curve integration. Other measures include ROC-AUC, accuracy, precision, recall, specificity, F2, Brier score and review fraction. The metadata development audit uses 500 whole-patient bootstrap samples with fixed fitted models. It is image-weighted and conditional on the trained model; it does not include training variability or repair prior reuse of development outcomes. Slope and intercept are reported without uncertainty intervals.

The scientific report is mapped to TRIPOD+AI and a preliminary PROBAST+AI assessment in the accompanying audit [7,8]. FAIR documentation provides version, data dictionary, access conditions and hashes. V2 records exact runtime packages, split/source hashes, predictions and model state; best and last DL checkpoints serve different purposes. GPU checkpoint resumption is implemented but remains unverified locally.

III. RESULTS AND DISCUSSION

A. Metadata selection and feature reduction

TABLE II. TRAINING-GROUP CV FEATURE COMPARISON

| Candidate | Best mean AP |
| --- | --- |
| LR age and sex | 0.1506 |
| LR age only | 0.1550 |
| LR sex only | 0.1392 |
| HGB age and sex | 0.1516 |

Age-only LR with C = 0.1 was selected; equal-ranking age-only C values are resolved in prespecified candidate order. This reduces the metadata input from two variables to one. All C values gave the same AP ranking within each LR feature group, so these results do not show a benefit from stronger LR tuning. Mean age-only CV AP was 0.1550 with fold SD 0.0049, which is variability across folds, not a confidence interval.

Fig. 2. LR feature-group ablation using training patients only.

TABLE III. DEVELOPMENT METADATA AUDIT

| Metric | Estimate | 95% CI |
| --- | --- | --- |
| AP | 0.1462 | 0.1166–0.1870 |
| ROC-AUC | 0.5220 | 0.4547–0.5882 |
| Precision | 0.1523 | 0.1097–0.2010 |
| Recall | 0.2170 | 0.1420–0.3005 |
| Specificity | 0.8133 | 0.7783–0.8447 |
| F2 | 0.2000 | 0.1371–0.2674 |
| Brier | 0.1159 | 0.0927–0.1423 |

On 2,548 branch-selection images, calibrated metadata accuracy was 0.7335. The threshold was 0.16752, with TN 1,795, FP 412, FN 267 and TP 74; 19.07% of images were flagged. The always-negative accuracy is 86.62% at this prevalence, so accuracy 73.35% is not evidence of superior screening. ROC-AUC confidence limits include 0.5, consistent with weak metadata discrimination.

Permuting retained age reduced development AP by 0.0120 on average (repeat SD 0.0043). This SD is not a patient-bootstrap confidence interval. Raw Brier score 0.2474 decreased to 0.1159 after calibration. A constant prevalence score has Brier approximately 0.1159 here, so the calibrated score provides little overall improvement over that reference. Calibration-in-the-large was −0.0619 and joint slope 0.6178; these do not justify a claim of clinical calibration.

Fig. 3. Quantile-binned development calibration; a coarse diagnostic, not clinical validation.

B. Earlier image and fusion evidence

TABLE IV. HISTORICAL COLAB EXPORTS

| Branch | AP | ROC-AUC | F2 |
| --- | --- | --- | --- |
| Metadata ML | 0.1919 | 0.6059 | 0.4540 |
| Image DL | 0.3141 | 0.7381 | 0.5093 |
| Late fusion | 0.3172 | 0.7413 | 0.5258 |

The earlier fusion F2 exceeds image DL by 0.0165, with recall increasing from 0.7025 to 0.7727 and precision decreasing from 0.2425 to 0.2309. There is no paired uncertainty estimate for that difference. Only three historical aggregate exports are available locally; examination predictions, checkpoint verification and exact Colab runtime are missing. Historical fusion fitting and threshold choice shared validation observations. These numbers are not new V2 test results.

TABLE V. RELATED EVIDENCE AND COMPARABILITY

| Study | Reported result | Comparison limit |
| --- | --- | --- |
| Bhave [1] | Composite AUROC 0.80 internal; 0.76 external | Different model and split |
| Pal [3] | SLVH AUROC 0.816 | Balanced subset and different target |
| N17 historical | Fusion AUROC 0.7413 | Aggregate export only |
| N17 Week 12 | Metadata AUROC 0.5220 | Development patients only |

Published results support feasibility rather than a leaderboard claim. Natural-prevalence composite AP cannot be equated with balanced SLVH-only AUPRC. The current optimization shows a small metadata development gain over the previous age-and-sex selection AP 0.1355, but no statistical or clinical improvement is established. Improving experimental separation and evidence saving should not be described as improved DL accuracy.

C. Limitations and remaining evidence

The source cohort is selected for PA radiography and echo availability in one US health system. Repeated examinations, delayed reference labels, narrow metadata and possible device shortcuts limit applicability. There is no formal sample-size calculation for a clinical deployment or external evaluation. Subgroup fairness, device ablation, Grad-CAM review and prospective workload impact remain unmeasured. The historical test has already been viewed.

V2 GPU training and real-image fusion remain pending. A separate external cohort with the same target has not been supplied. Calibration slope/intercept confidence intervals and decision-curve benefit are absent. The filled checklist marks these gaps explicitly instead of declaring compliance. No ethics waiver, independent clinical audit or all-author conflict-of-interest statement is invented.

IV. CONCLUSION

N17 has built a paired metadata and radiograph research pipeline and strengthened it with patient-separated development roles and evidence persistence. The Week 12 experiment completed feature selection, tuning and retraining; age-only LR yielded development AP 0.1462 and ROC-AUC 0.5220. Its weak discrimination shows that metadata support must be tested rather than assumed. Historical fusion results motivate a new experiment but do not establish reliable or clinical benefit. Next steps are a complete V2 GPU run, verified checkpoint recovery, frozen fusion comparison and independent validation.

REPRODUCIBILITY AND AI DISCLOSURE

Local metadata measurements use Python 3.12.10 and scikit-learn 1.6.1; receipts record other exact versions and hashes. The code-only package contains no patient rows, images, private predictions, credentials or weights. AI assistance was used for code scaffolding, documentation, literature organization and language editing. The members must review citations, outputs and the final text before submission. Individual CRediT roles, funding, conflicts and Mendeley collection status require author confirmation; no Mendeley-linked fields are claimed.

REFERENCES

[1] S. Bhave et al., “Deep learning to detect left ventricular structural abnormalities in chest X-rays,” Eur. Heart J., vol. 45, no. 22, pp. 2002–2012, 2024, doi: 10.1093/eurheartj/ehad782.

[2] S.-C. Huang et al., “Fusion of medical imaging and electronic health records using deep learning: a systematic review and implementation guidelines,” npj Digit. Med., vol. 3, art. 136, 2020, doi: 10.1038/s41746-020-00341-z.

[3] B. Pal, R. Chellappa, and M. Umair, “Encoding of demographic and anatomical information in chest X-ray-based severe left ventricular hypertrophy classifiers,” Biomedicines, vol. 13, art. 2140, 2025, doi: 10.3390/biomedicines13092140.

[4] P. Elias and S. Bhave, “CheXchoNet: A chest radiograph dataset with gold standard echocardiography labels,” PhysioNet, v1.0.0, 2024, doi: 10.13026/kp08-ws25.

[5] M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. ICML, PMLR 97, pp. 6105–6114, 2019.

[6] B. Van Calster et al., “Calibration: the Achilles heel of predictive analytics,” BMC Med., vol. 17, art. 230, 2019, doi: 10.1186/s12916-019-1466-7.

[7] G. S. Collins et al., “TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods,” BMJ, vol. 385, e078378, 2024, doi: 10.1136/bmj-2023-078378.

[8] K. G. M. Moons et al., “PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods,” BMJ, vol. 388, e082505, 2025, doi: 10.1136/bmj-2024-082505.

