#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
review_project.py — Kiểm tra bài nộp đồ án môn học ET4248 theo rubric và
đề xuất phương án nâng cao chất lượng.

Bổ sung (KHÔNG thay thế) cho validate_submission.py:

  * validate_submission.py — CỔNG BẮT BUỘC: cách ly thư mục, schema
    submission.json, secret, file nặng. FAIL → PR không được merge.
  * review_project.py      — CÔNG CỤ TƯ VẤN theo Khung Quản Lý & Rubric 100
    điểm (https://fossbk-spec.github.io/hmyt-book/du_an_mon_hoc): tiêu chí
    nào đã có / chưa có bằng chứng, nghi vấn rò rỉ dữ liệu, các vấn đề liên
    quan chính sách trừ điểm (Phần IV), và 3 tầng phương án nâng cao chất
    lượng. Kết quả KHÔNG phải điểm chính thức — giảng viên chấm cuối cùng.

An toàn khi chạy trên PR của sinh viên: script CHỈ đọc file (json, regex,
ast.parse) — không import, không exec, không chạy notebook/mã nguồn của bài
nộp, không theo symlink.

Cách dùng:
  python scripts/review_project.py submissions/<topic_slug>_<ma_nhom>
  python scripts/review_project.py submissions/<...> --milestone m2
  python scripts/review_project.py submissions/<...> --online
      (+ xác minh DOI qua Crossref, nhánh gd1–gd7 của repo nhóm, so khớp với
       code minh họa công khai của đề tài)
  python scripts/review_project.py --all --output-dir review_reports
      (giảng viên: review cả lớp + bảng tổng hợp)
  python scripts/review_project.py --pr-files changed_files.txt   (dùng trong CI)
  python scripts/review_project.py --self-test                    (kiểm tra cấu hình rubric/)

Cấu hình: rubric/rubric.json, rubric/topics/<topic_slug>.json,
rubric/improvements.json — xem rubric/README.md. Yêu cầu Python ≥ 3.9, chỉ
dùng thư viện chuẩn (không cần pip install).
"""

from __future__ import annotations

import argparse
import ast
import bisect
import datetime as dt
import difflib
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tokenize
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = ROOT / "submissions"
RUBRIC_DIR = ROOT / "rubric"
TEMPLATE_NAME = "_TEMPLATE"
RUBRIC_URL = "https://fossbk-spec.github.io/hmyt-book/du_an_mon_hoc"

MAX_TEXT_BYTES = 3 * 1024 * 1024
MAX_NOTEBOOK_BYTES = 15 * 1024 * 1024
MAX_DOCX_XML_BYTES = 30 * 1024 * 1024
HTTP_TIMEOUT = 15
USER_AGENT = f"hmyt-project-review/{VERSION} (+https://github.com/fossbk-spec/hmyt-project)"

PASS, WARN, FAIL, MANUAL, NA = "PASS", "WARN", "FAIL", "MANUAL", "NA"
ICON = {PASS: "✅", WARN: "⚠️", FAIL: "❌", MANUAL: "👤", NA: "➖"}
SEV_ORDER = {"critical": 0, "major": 1, "minor": 2, "info": 3}
SEV_LABEL = {"critical": "nghiêm trọng", "major": "quan trọng", "minor": "nhỏ", "info": "thông tin"}
MILESTONE_ORDER = ["M0", "M1", "M2", "M3", "M4", "FINAL"]
MILESTONE_ALIASES = {"m0": "M0", "m1": "M1", "m2": "M2", "m3": "M3", "m4": "M4", "final": "FINAL"}

TEXT_EXT = {".md", ".markdown", ".txt", ".rst", ".tex"}
REPORT_DIRS = {"report", "reports", "paper", "papers", "bao_cao", "baocao", "manuscript"}
JUNK_DIRS = {"__pycache__", ".ipynb_checkpoints", ".venv", "venv", ".idea", ".vscode",
             "node_modules", ".pytest_cache", ".mypy_cache"}
JUNK_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}
WEIGHT_EXT = {".pt", ".pth", ".ckpt", ".h5", ".hdf5", ".pkl", ".pickle", ".joblib", ".onnx",
              ".bin", ".safetensors", ".keras", ".tflite"}
OUTPUT_EXT = {".json", ".csv", ".tsv", ".txt", ".log"}
SLIDE_EXT = {".pdf", ".pptx", ".ppt", ".key", ".odp", ".html"}

# Chính sách trừ điểm — Phần IV của Khung Quản Lý & Rubric
PENALTY_LEAK = "Phần IV.1 — rò rỉ dữ liệu: trừ 50% điểm cột mốc phát hiện lỗi, sửa và nộp lại PR trong 03 ngày."
PENALTY_PHI = "Phần IV.2 — lộ dữ liệu định danh bệnh nhân: đình chỉ nghiệm thu + trừ 30 điểm tổng kết."
PENALTY_FREERIDER = "Phần IV.3 — thành viên không có commit hoặc K_peer < 0.5: 0 điểm phần kỹ thuật của cột mốc."
PENALTY_LATE = "Phần IV.4 — nộp muộn: trừ 10%/24 giờ; muộn quá 72 giờ = 0 điểm cột mốc."
PENALTY_FABRICATION = ("Phần IV.5 — sao chép/bịa số liệu/trích dẫn ma: hủy kết quả dự án (0 điểm) "
                       "và chuyển Hội đồng Kỷ luật.")
PENALTY_ACCURACY_ONLY = "Rubric Milestone 4 — chỉ báo cáo Accuracy trên dữ liệu mất cân bằng: trừ 3 điểm."


# ---------------------------------------------------------------------------
# Mẫu hình nhận diện (regex) — dùng chung cho mọi đề tài
# ---------------------------------------------------------------------------

# Bộ biến đổi "học" thống kê từ dữ liệu: fit trước khi chia tập = rò rỉ.
# Giá trị = mức độ nghiêm trọng nếu fit trên toàn bộ dữ liệu.
TRANSFORMERS = {
    "StandardScaler": "critical", "MinMaxScaler": "critical", "RobustScaler": "critical",
    "MaxAbsScaler": "critical", "QuantileTransformer": "critical", "PowerTransformer": "critical",
    "SimpleImputer": "critical", "KNNImputer": "critical", "IterativeImputer": "critical",
    "SelectKBest": "critical", "SelectPercentile": "critical", "RFE": "critical", "RFECV": "critical",
    "SelectFromModel": "critical", "SequentialFeatureSelector": "critical", "TargetEncoder": "critical",
    "PCA": "major", "TruncatedSVD": "major", "FastICA": "major", "KernelPCA": "major",
    "TfidfVectorizer": "major", "KBinsDiscretizer": "major", "CountVectorizer": "minor",
}
PIPELINE_FUNCS = {"Pipeline", "make_pipeline"}
RESAMPLER_CALLS = {"fit_resample", "fit_sample"}
SPLIT_FUNCS = {"train_test_split", "random_split", "iterative_train_test_split"}
SPLIT_CLASSES = {"GroupShuffleSplit", "GroupKFold", "StratifiedGroupKFold", "LeaveOneGroupOut",
                 "LeavePGroupsOut", "KFold", "StratifiedKFold", "RepeatedStratifiedKFold",
                 "RepeatedKFold", "ShuffleSplit", "StratifiedShuffleSplit", "TimeSeriesSplit",
                 "PredefinedSplit", "MultilabelStratifiedKFold", "IterativeStratification"}
CV_FUNCS = {"cross_val_score", "cross_validate", "cross_val_predict", "learning_curve",
            "validation_curve"}
SEARCH_CLASSES = {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV",
                  "HalvingRandomSearchCV", "BayesSearchCV"}

PATIENT_SPLIT_RX = (r"GroupKFold|GroupShuffleSplit|StratifiedGroupKFold|LeaveOneGroupOut|LeavePGroupsOut"
                    r"|groups\s*=|strat_fold"
                    r"|train_test_split\(\s*[^,\n]*(unique|_ids\b|patients|subjects)"
                    r"|\.isin\(\s*[^)\n]*(_ids\b|patients|subjects)")
SPLIT_ASSERT_RX = r"^\s*assert\b.*(isdisjoint|intersection|&|len\(\s*set)"
FILLNA_GLOBAL_RX = r"\.fillna\([^)\n]*\.(mean|median|mode)\("
STRAT_FOLD_SPLIT_RX = r"strat_fold\W*\s*(==|<=|>=|<|>|!=|\.isin)"

CLASSIC_MODELS = [
    "LogisticRegression", "LinearRegression", "Ridge", "Lasso", "ElasticNet", "RidgeClassifier",
    "SGDClassifier", "RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier",
    "ExtraTreesRegressor", "GradientBoostingClassifier", "GradientBoostingRegressor",
    "HistGradientBoostingClassifier", "AdaBoostClassifier", "BaggingClassifier",
    "DecisionTreeClassifier", "DecisionTreeRegressor", "KNeighborsClassifier", "SVC", "LinearSVC",
    "SVR", "GaussianNB", "MultinomialNB", "BernoulliNB", "MLPClassifier", "MLPRegressor",
    "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor", "CatBoostClassifier",
    "CatBoostRegressor", "VotingClassifier", "StackingClassifier", "CoxPHFitter",
    "RandomSurvivalForest",
]
DEEP_MODELS = [
    ("ResNet", r"\b(?i:resnet\d*|xresnet1d\d*)\b"),
    ("DenseNet", r"\b(?i:densenet\d*)\b"),
    ("EfficientNet", r"\b(?i:efficientnet\w*)\b"),
    ("MobileNet", r"\b(?i:mobilenet\w*)\b"),
    ("VGG", r"\b(?i:vgg\d+\w*)\b"),
    ("ConvNeXt", r"\b(?i:convnext\w*)\b"),
    ("ViT/Swin", r"\b(vit_\w+|ViT\w*|VisionTransformer|(?i:swin\w*))\b"),
    ("U-Net", r"\b(U-?Net\w*|Unet\w*|ResUNet\w*)\b"),
    ("LSTM/GRU", r"\b(LSTM|GRU)\s*\("),
    ("CNN-1D", r"\bConv1[dD]\s*\("),
    ("CNN-2D tự thiết kế", r"\bConv2[dD]\s*\("),
    ("Transformer", r"\bTransformerEncoder\w*\s*\("),
    ("BERT/LLM", r"\b(AutoModel\w*|Bert\w*Model|BertFor\w+|\w+BERT\w*|Longformer\w*|RoBERTa\w*)\b"),
]
CUSTOM_MODEL_RX = (r"class\s+(\w+)\s*\(\s*(?:nn\.Module|torch\.nn\.Module|keras\.Model|tf\.keras\.Model"
                   r"|pl\.LightningModule|LightningModule)\s*\)")

XAI_METHODS = {
    "shap": (r"\bshap\b|TreeExplainer|KernelExplainer|DeepExplainer|GradientExplainer|LinearExplainer",
             "SHAP"),
    "gradcam": (r"(?i)grad[-_ ]?cam|pytorch_grad_cam|score[-_]?cam|eigen[-_]?cam|layer[-_]?cam", "Grad-CAM"),
    "saliency": (r"(?i)saliency|integrated[-_ ]?gradients|captum|occlusion|deeplift|guided[-_ ]?backprop",
                 "Saliency / Integrated Gradients"),
    "lime": (r"\blime\b|LimeTabularExplainer|LimeTextExplainer|LimeImageExplainer", "LIME"),
    "attention": (r"(?i)attention[-_ ]?(weight|map|score|visuali)|output_attentions|bertviz|attn_weights",
                  "Attention weights"),
    "segmentation_map": (r"(?i)overlay[^\n]*mask|mask[^\n]*overlay|segmentation[-_ ]map", "Bản đồ phân vùng"),
    "permutation": (r"permutation_importance|feature_importances_", "Feature importance / Permutation"),
}
XAI_BY_MODALITY = {
    "image": ["gradcam", "saliency", "segmentation_map"],
    "signal": ["gradcam", "saliency", "shap", "attention"],
    "tabular": ["shap", "lime"],
    "nlp": ["attention", "lime", "shap", "saliency"],
}
XAI_TEXT_RX = r"(?i)\bSHAP\b|grad-?cam|saliency|attention|\bLIME\b|bản đồ nhiệt|heatmap|\bXAI\b|giải thích"

METRIC_PATTERNS = {
    "accuracy": (r"accuracy_score|\b[Aa]ccuracy\b|độ chính xác", "Accuracy"),
    "sensitivity": (r"recall_score|\b(?i:sensitivity|recall)\b|\bTPR\b|độ nhạy|classification_report",
                    "Sensitivity/Recall"),
    "specificity": (r"\b(?i:specificity)\b|\bTNR\b|độ đặc hiệu|tn\s*/\s*\(\s*tn\s*\+\s*fp\s*\)",
                    "Specificity"),
    "f1": (r"f1_score|\bF1\b|(?i:f1-score)|classification_report|fbeta", "F1"),
    "roc_auc": (r"roc_auc_score|roc_curve|RocCurveDisplay|\bROC\b|\bAUROC\b|\bAUC\b", "ROC-AUC"),
    "pr_auc": (r"average_precision_score|precision_recall_curve|PrecisionRecallDisplay|\bPR-?AUC\b|\bAUPRC\b",
               "PR-AUC"),
    "mcc": (r"matthews_corrcoef|\bMCC\b", "MCC"),
    "precision": (r"precision_score|\b(?i:precision)\b|\bPPV\b|classification_report", "Precision"),
    "dice": (r"\b(?i:dice)\b|dice_coef|DiceLoss|DiceMetric", "Dice"),
    "iou": (r"\b(?i:iou)\b|(?i:jaccard)|MeanIoU", "IoU"),
    "kappa": (r"cohen_kappa_score|\b(?i:kappa)\b", "Cohen's Kappa"),
    "mae": (r"mean_absolute_error|mean_squared_error|\bMAE\b|\bRMSE\b", "MAE/RMSE"),
}
METRIC_SETS = {
    "binary": ["sensitivity", "specificity", "f1", "roc_auc", "pr_auc"],
    "multiclass": ["sensitivity", "specificity", "f1", "roc_auc", "pr_auc"],
    "multilabel": ["sensitivity", "specificity", "f1", "roc_auc", "pr_auc"],
    "segmentation": ["dice", "iou"],
    "ner": ["precision", "sensitivity", "f1"],
    "regression": ["mae"],
}
INFORMATIVE_METRICS = {"sensitivity", "specificity", "f1", "roc_auc", "pr_auc", "mcc", "dice", "iou",
                       "kappa", "precision", "mae"}

CI_RX = (r"(?i)bootstrap|95\s*%\s*CI|confidence[-_ ]interval|khoảng tin cậy|\bCI\s*95|"
         r"np\.percentile\([^)\n]*2\.5|stats\.bootstrap")
CI_NUMERIC_RX = (r"(?i)(95\s*%\s*CI|CI\s*95|khoảng tin cậy)[^\n]{0,40}\d+[.,]\d+\s*[-–]\s*\d+[.,]\d+|"
                 r"\d+[.,]\d+\s*[\(\[]\s*\d+[.,]\d+\s*[-–,;]\s*\d+[.,]\d+\s*[\)\]]")
STD_RX = r"±|\+/-|\.std\("
CALIBRATION_RX = r"(?i)calibration_curve|CalibrationDisplay|CalibratedClassifierCV|brier|reliability diagram|hiệu chuẩn|calibration"

EDA_SIGNALS = [
    ("Phân bố nhãn / mất cân bằng", r"value_counts\(|countplot|bincount|Counter\(|class[_ ]distribution|\.hist\(|histplot"),
    ("Dữ liệu khuyết", r"isna\(|isnull\(|missingno|msno\.|missing"),
    ("Tương quan", r"\.corr\(|heatmap|\bcorrcoef\(|pairplot"),
    ("Trực quan hóa", r"plt\.|sns\.|px\.|plotly|matplotlib|seaborn"),
    ("Nhân khẩu học (tuổi/giới)", r"(?i)['\"](age|sex|gender|tuoi|gioi_tinh)['\"]|\b(age|sex|gender)\b"),
    ("Thống kê mô tả", r"\.describe\(|\.info\(|\.shape\b"),
    ("Mẫu tín hiệu/ảnh", r"imshow|wfdb\.plot|plot_wfdb|show_batch|plot_images|\.plot\([^)\n]*(signal|ecg|lead)"),
]
PREPROCESS_BY_MODALITY = {
    "signal": r"butter|filtfilt|sosfilt|iirnotch|\bnotch|bandpass|band_pass|highpass|high_pass|lowpass|pywt|wavelet|neurokit|ecg_clean|savgol|detrend|baseline[_ ]wander|resample_poly|biosppy",
    "image": r"CLAHE|createCLAHE|equalizeHist|transforms\.\w+|albumentations|\bA\.\w+\(|RandomHorizontalFlip|RandomRotation|RandomResizedCrop|Normalize\(|Resize\(|cv2\.resize|\bcrop\w*|\bROI\b|windowing|hounsfield",
    "tabular": r"SimpleImputer|KNNImputer|IterativeImputer|\bMICE\b|fillna|OneHotEncoder|get_dummies|OrdinalEncoder|StandardScaler|MinMaxScaler|RobustScaler|outlier|\bIQR\b|winsor\w*|\.clip\(|log1p",
    "nlp": r"\w*Tokenizer|tokeniz\w*|TfidfVectorizer|CountVectorizer|\.lower\(\)|re\.sub|stopword\w*|lemmat\w*|stemm\w*|underthesea|pyvi|spacy|scispacy|negex|NegEx|sent_tokenize|max_length|truncation",
}
REG_SIGNALS_RX = (r"Dropout|dropout|weight_decay|EarlyStopping|early_stopping|patience|ReduceLROnPlateau|"
                  r"CosineAnnealing\w*|OneCycleLR|warmup|BatchNorm\w*|LayerNorm|kernel_regularizer|"
                  r"GridSearchCV|RandomizedSearchCV|optuna|cross_val_score|cross_validate|StratifiedKFold|"
                  r"max_depth|min_samples_leaf|penalty\s*=|\bC\s*=|reg_lambda|reg_alpha|class_weight|label_smoothing")
CURVES_RX = r"val_loss|history\.history|learning_curve|validation_curve|train_loss|loss_curve|evals_result"
TRACK_DASH_RX = r"\bwandb\b|tensorboard|SummaryWriter|\bmlflow\b|neptune|comet_ml|TensorBoardLogger|WandbLogger"
TRACK_FILE_RX = r"CSVLogger|to_csv\([^)\n]*(log|history|metric)|logging\.basicConfig|json\.dump\([^)\n]*(history|log|metric|result)"
SEED_RX = (r"random_state\s*=\s*\d+|np\.random\.seed\(|\brandom\.seed\(|torch\.manual_seed\(|"
           r"tf\.random\.set_seed\(|seed_everything\(|set_seed\(|\bSEED\s*=\s*\d+|RANDOM_STATE\s*=\s*\d+|"
           r"default_rng\(\s*\d+")
TORCH_SEED_RX = r"torch\.manual_seed\(|seed_everything\(|set_seed\(|tf\.random\.set_seed\("
DL_FRAMEWORK_RX = r"^\s*(import|from)\s+(torch|tensorflow|keras|lightning|pytorch_lightning)\b"
ABS_PATH_LOCAL_RX = r"""['"](?:[A-Za-z]:[\\/]|/home/|/Users/|/mnt/|/root/)"""
ABS_PATH_CLOUD_RX = r"""['"](?:/content/|/kaggle/)"""
ENTRYPOINT_RX = r"(?im)^\s*(\$\s*)?(make\b|bash\s+\S*run\w*\.sh|sh\s+\S*run\w*\.sh|python3?\s+\S*(main|run_all|run_pipeline|pipeline|reproduce|train)\w*\.py|python3?\s+-m\s+\w+|dvc\s+repro|snakemake|docker\s+(compose\s+)?run)"
README_RUN_HEADING_RX = r"(?i)cách chạy|chạy lại|tái lập|reproduc|how to run|hướng dẫn chạy|usage|cài đặt|installation|quick ?start"
README_PLACEHOLDERS = ["(điền tên đề tài)", "(mã nhóm — không dùng tên thật)", "(vd Random Forest + SHAP)",
                       "(1-2 chỉ số chính", "(Điền các bước"]

AI_DISCLOSURE_RX = (r"(?i)AI and AI-Assisted Technologies Disclosure|AI[- ](assisted[- ])?disclosure|"
                    r"Tuyên bố (về )?(việc )?(sử dụng )?(AI|trí tuệ nhân tạo)|Khai báo (sử dụng )?(AI|trí tuệ nhân tạo)")
CREDIT_RX = r"(?i)\bCRediT\b|đóng góp (của )?(tác giả|thành viên)|vai trò (của )?(tác giả|thành viên)"
CREDIT_ROLES = ["Conceptualization", "Methodology", "Software", "Validation", "Formal analysis", "Investigation",
                "Data curation", "Writing", "Visualization", "Supervision", "Project administration"]
PROBLEM_IO_RX = r"(?i)\binput\b|đầu vào|\(X\)|\boutput\b|đầu ra|\(Y\)|biến mục tiêu|nhãn (dự đoán|đầu ra|mục tiêu)"
PROBLEM_CONTEXT_RX = r"(?i)đặt vấn đề|giới thiệu|introduction|bối cảnh|tính cấp thiết|motivation|phát biểu bài toán|ý nghĩa lâm sàng"
PROBLEM_COST_RX = (r"(?i)false negative|false positive|âm tính giả|dương tính giả|\bFN\b|\bFP\b|bỏ sót|"
                   r"báo động giả|chi phí sai|bất đối xứng|asymmetric")
ABLATION_RX = r"(?i)ablation|nghiên cứu loại bỏ|loại bỏ thành phần|bỏ module|w/o\b|without\s+\w+"
ERROR_CODE_RX = r"confusion_matrix|ConfusionMatrixDisplay|multilabel_confusion_matrix|crosstab"
ERROR_TEXT_RX = (r"(?i)phân tích lỗi|error analysis|ca (dự đoán )?sai|dự đoán sai|misclassif|false negative|"
                 r"false positive|âm tính giả|dương tính giả|ma trận nhầm lẫn|confusion matrix")
LIMITATION_RX = r"(?i)hạn chế|limitation"
PROPOSED_TEXT_RX = r"(?i)mô hình đề xuất|kiến trúc đề xuất|phương pháp đề xuất|proposed (model|method|architecture|approach)"

DEFAULT_PAPER_SECTIONS = [
    ["Abstract / Tóm tắt", "abstract|tóm tắt"],
    ["Introduction / Đặt vấn đề", "introduction|giới thiệu|đặt vấn đề|mở đầu"],
    ["Related Work / Y văn", "related work|y văn|tổng quan|literature|công trình liên quan"],
    ["Methods / Phương pháp", "method|phương pháp"],
    ["Results / Kết quả", "results?|kết quả"],
    ["Discussion / Thảo luận", "discussion|thảo luận|bàn luận"],
    ["Conclusion / Kết luận", "conclusions?|kết luận"],
    ["References / Tài liệu tham khảo", "references|tài liệu tham khảo|bibliography"],
]

# Cột định danh trực tiếp (PHI/PII) — tên đã bỏ dấu, viết thường, nối bằng "_"
PHI_DIRECT = {"name", "full_name", "fullname", "patient_name", "ho_ten", "hoten", "ho_va_ten", "ten_benh_nhan",
              "ten_bn", "address", "dia_chi", "diachi", "phone", "phone_number", "so_dien_thoai", "sdt",
              "email", "cccd", "cmnd", "so_cccd", "so_cmnd", "ssn", "bhyt", "so_bhyt", "ma_bhyt",
              "insurance_number", "passport"}
PHI_QUASI = {"dob", "date_of_birth", "birth_date", "ngay_sinh", "mrn", "medical_record_number", "so_benh_an",
             "ma_benh_an"}
MIMIC_FILE_RX = (r"(?i)(^|/)(admissions|patients|noteevents|discharge|radiology|diagnoses_icd|procedures_icd|"
                 r"chartevents|labevents|icustays|prescriptions|d_icd_diagnoses|omr)\.(csv|csv\.gz|parquet)$")

DOI_RX = r"\b10\.\d{4,9}/[^\s\"'<>{}\[\]()|,;]+"
CITEKEY_PANDOC_RX = r"\[[^\]\n]*@([\w:\-]+)[^\]\n]*\]"
CITEKEY_LATEX_RX = r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]+)\}"
CITEKEY_HMYT_RX = r"`([a-z]+\d{4}[a-z][a-z0-9]*)`"


# ---------------------------------------------------------------------------
# Cấu trúc dữ liệu
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    check_id: str
    title: str
    status: str
    severity: str = "major"
    detail: str = ""
    evidence: list = field(default_factory=list)
    suggestion: str = ""
    penalty: str = ""


@dataclass
class CodeUnit:
    """1 file .py hoặc 1 notebook — các cell nối theo thứ tự chạy."""
    path: str
    kind: str
    lines: list = field(default_factory=list)   # dòng đã bỏ comment (giữ số dòng)
    locs: list = field(default_factory=list)    # nhãn vị trí của từng dòng
    cells: list = field(default_factory=list)   # [(offset, source đã làm sạch)]
    import_lines: set = field(default_factory=set)  # chỉ số dòng thuộc câu lệnh import


@dataclass
class Notebook:
    path: str
    n_code: int = 0
    n_executed: int = 0
    exec_counts: list = field(default_factory=list)
    n_errors: int = 0
    n_images: int = 0
    output_text: str = ""
    markdown: str = ""


@dataclass
class TextDoc:
    path: str
    kind: str
    text: str
    is_report: bool = False


class Context:
    """Toàn bộ dữ liệu đã đọc từ 1 thư mục nộp bài."""

    def __init__(self, folder: Path, opts):
        self.folder = folder
        self.opts = opts
        self.notes: list[str] = []
        self.files: list[Path] = []
        self.rel_files: list[str] = []
        self.code_units: list[CodeUnit] = []
        self.notebooks: list[Notebook] = []
        self.text_docs: list[TextDoc] = []
        self.bib_entries: dict = {}
        self.manifest: dict = {}
        self.profile: dict = {}
        self.modalities: list[str] = []
        self.milestone = "FINAL"
        self.rubric_name = "standard"
        self.rubric_def: dict = {}
        self._leak = None
        self._git = None
        self._models = None

    # -- truy vấn tiện ích -------------------------------------------------
    @property
    def report_docs(self):
        docs = [d for d in self.text_docs if d.is_report]
        return docs

    @property
    def readme(self):
        for d in self.text_docs:
            if d.kind == "readme":
                return d
        return None

    def docs_for(self, scope: str):
        if scope == "report":
            return self.report_docs or [d for d in self.text_docs if d.kind != "nb-output"]
        if scope == "readme":
            return [self.readme] if self.readme else []
        return [d for d in self.text_docs if d.kind != "nb-output"]

    def grep_code(self, pattern, flags=0, limit=None, imports=False):
        """Tìm trong code (đã bỏ comment). Mặc định bỏ qua dòng import: import
        mà không dùng không phải bằng chứng đã áp dụng kỹ thuật."""
        rx = re.compile(pattern, flags | re.M)
        hits = []
        for u in self.code_units:
            for i, line in enumerate(u.lines):
                if not imports and i in u.import_lines:
                    continue
                if line.strip() and rx.search(line):
                    hits.append((u.locs[i], line.strip()))
                    if limit and len(hits) >= limit:
                        return hits
        return hits

    def grep_text(self, pattern, scope="text", flags=0, limit=None):
        rx = re.compile(pattern, flags)
        hits = []
        for d in self.docs_for(scope):
            for i, line in enumerate(d.text.split("\n")):
                if rx.search(line):
                    hits.append((f"{d.path}:{i + 1}", line.strip()))
                    if limit and len(hits) >= limit:
                        return hits
        return hits

    def headings(self, scope="report"):
        out = []
        for d in self.docs_for(scope):
            plain = d.kind in ("pdf", "docx", "txt", "rst")
            for i, line in enumerate(d.text.split("\n")):
                h = heading_of(line, plain)
                if h:
                    out.append((f"{d.path}:{i + 1}", h))
        return out

    def find_section(self, keyword_rx, scope="report"):
        keyword_rx = re.sub(r"^\(\?i\)", "", keyword_rx)
        rx = re.compile(r"^\W*(?:[IVXLC]+\.|\d+(?:\.\d+)*\.?|[A-Z]\.)?\s*\W*(?:" + keyword_rx + r")", re.I)
        return [(loc, h) for loc, h in self.headings(scope) if rx.search(h)]

    def files_matching(self, pattern, flags=re.I):
        rx = re.compile(pattern, flags)
        return [f for f in self.rel_files if rx.search(f)]


def heading_of(line: str, plain: bool = False):
    """Tiêu đề mục: Markdown `#`, dòng in đậm đứng riêng; với PDF/DOCX/TXT thêm
    dòng ngắn đánh số (`II. RELATED WORK`, `3.1 Dữ liệu`) và `Abstract—...`."""
    m = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
    if m:
        return m.group(1).strip()
    s = line.strip()
    if not s:
        return None
    if plain:
        m = re.match(r"^(Abstract|Index Terms|Keywords|Tóm tắt|Từ khóa)\s*[—–:-]", s, re.I)
        if m:
            return m.group(1)
    if len(s) > 90:
        return None
    if s.startswith("**") and s.endswith("**") and len(s) > 4:
        return s.strip("*").strip()
    if plain and re.match(r"^(?:[IVXLC]+\.|\d+(?:\.\d+)*\.?|[A-Z]\.)\s+\S", s):
        return s
    return None


# ---------------------------------------------------------------------------
# Đọc file (chỉ đọc — không thực thi gì)
# ---------------------------------------------------------------------------

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def read_text(path: Path, limit=MAX_TEXT_BYTES):
    try:
        if path.stat().st_size > limit:
            return None
        return nfc(path.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return None


def walk_files(folder: Path) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
        d = Path(dirpath)
        dirnames[:] = sorted(n for n in dirnames if n != ".git" and not (d / n).is_symlink())
        for name in sorted(filenames):
            p = d / name
            if not p.is_symlink() and p.is_file():
                out.append(p)
    return out


def strip_comments(src: str) -> str:
    """Bỏ comment Python nhưng giữ nguyên số dòng (để bằng chứng trỏ đúng dòng)."""
    lines = src.split("\n")
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                r, c = tok.start
                if 0 < r <= len(lines):
                    lines[r - 1] = lines[r - 1][:c].rstrip()
        return "\n".join(lines)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return "\n".join("" if ln.lstrip().startswith("#") else ln for ln in lines)


def clean_python_source(src: str) -> str:
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    lines = src.split("\n")
    if lines and lines[0].lstrip().startswith("%%"):
        magic = lines[0].strip().split()[0]
        if magic in ("%%time", "%%timeit", "%%capture"):
            lines[0] = ""
        else:  # %%bash, %%html, %%writefile... — không phải Python
            return "\n".join("" for _ in lines)
    lines = ["" if re.match(r"\s*[%!]", ln) else ln for ln in lines]
    return strip_comments("\n".join(lines))


def build_code_unit(rel: str, kind: str, cells) -> CodeUnit:
    unit = CodeUnit(rel, kind)
    for cell_no, src in cells:
        clean = clean_python_source(src)
        offset = len(unit.lines)
        cell_lines = clean.split("\n")
        for j, line in enumerate(cell_lines):
            unit.lines.append(line)
            unit.locs.append(f"{rel}:{j + 1}" if cell_no is None else f"{rel} [cell {cell_no}, dòng {j + 1}]")
        unit.cells.append((offset, clean))
        try:
            for node in ast.walk(ast.parse(clean)):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    unit.import_lines.update(range(offset + node.lineno - 1, offset + (node.end_lineno or node.lineno)))
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            for j, line in enumerate(cell_lines):
                if re.match(r"\s*(import|from)\s+\w", line):
                    unit.import_lines.add(offset + j)
    return unit


def parse_notebook(rel: str, raw: str):
    try:
        nb = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None, None
    if not isinstance(nb, dict):
        return None, None
    meta = Notebook(rel)
    code_cells, md_parts, out_parts = [], [], []
    for idx, cell in enumerate(nb.get("cells") or [], start=1):
        if not isinstance(cell, dict):
            continue
        src = cell.get("source", "")
        src = "".join(map(str, src)) if isinstance(src, list) else str(src)
        if cell.get("cell_type") == "markdown":
            md_parts.append(nfc(src))
        elif cell.get("cell_type") == "code":
            code_cells.append((idx, src))
            meta.n_code += 1
            ec = cell.get("execution_count")
            if isinstance(ec, int):
                meta.n_executed += 1
                meta.exec_counts.append(ec)
            for out in cell.get("outputs") or []:
                if not isinstance(out, dict):
                    continue
                otype = out.get("output_type")
                if otype == "stream":
                    t = out.get("text", "")
                    out_parts.append("".join(map(str, t)) if isinstance(t, list) else str(t))
                elif otype in ("execute_result", "display_data"):
                    data = out.get("data") if isinstance(out.get("data"), dict) else {}
                    t = data.get("text/plain", "")
                    out_parts.append("".join(map(str, t)) if isinstance(t, list) else str(t))
                    if any(k.startswith("image/") for k in data):
                        meta.n_images += 1
                elif otype == "error":
                    meta.n_errors += 1
    meta.markdown = "\n\n".join(md_parts)
    meta.output_text = "\n".join(out_parts)
    return meta, build_code_unit(rel, "ipynb", code_cells)


def latex_to_text(src: str) -> str:
    src = re.sub(r"(?<!\\)%.*", "", src)
    src = re.sub(r"\\(?:sub)*section\*?\{([^}]*)\}", r"\n# \1\n", src)
    src = re.sub(r"\\begin\{abstract\}", "\n# Abstract\n", src)
    src = re.sub(r"\\begin\{IEEEkeywords\}", "\n# Index Terms\n", src)
    return src


def docx_to_text(path: Path):
    try:
        with zipfile.ZipFile(path) as z:
            info = z.getinfo("word/document.xml")
            if info.file_size > MAX_DOCX_XML_BYTES:
                return None
            xml = z.read(info).decode("utf-8", errors="ignore")
    except (zipfile.BadZipFile, KeyError, OSError):
        return None
    paras = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        texts = re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", p)
        if not texts:
            continue
        t = html.unescape("".join(texts))
        style = re.search(r'<w:pStyle w:val="([^"]+)"', p)
        if style and re.match(r"(?i)(heading|title|u\d|berschrift|titre)", style.group(1)):
            t = "# " + t
        paras.append(t)
    return nfc("\n".join(paras))


def pdf_to_text(path: Path, notes: list):
    exe = shutil.which("pdftotext")
    if not exe:
        notes.append(f"Không đọc được nội dung PDF `{path.name}` (máy thiếu `pdftotext`/poppler-utils).")
        return None
    try:
        r = subprocess.run([exe, "-layout", "-l", "60", str(path), "-"], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return nfc(r.stdout.decode("utf-8", errors="ignore"))


def parse_bib(text: str) -> dict:
    entries = {}
    starts = list(re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text))
    for i, m in enumerate(starts):
        if m.group(1).lower() in ("comment", "string", "preamble"):
            continue
        body = text[m.end(): starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        doi = re.search(r"\bdoi\s*=\s*[{\"]\s*([^}\"]+)", body, re.I)
        title = re.search(r"\btitle\s*=\s*[{\"](.+?)[}\"]\s*,?\s*\n", body, re.I | re.S)
        entries[m.group(2)] = {
            "doi": doi.group(1).strip() if doi else "",
            "title": re.sub(r"[{}]", "", title.group(1)).strip() if title else "",
        }
    return entries


def is_report_path(rel: str) -> bool:
    parts = rel.split("/")
    if len(parts) > 1 and parts[0].lower() in REPORT_DIRS:
        return True
    return bool(re.search(r"(?i)(report|bao_?cao|paper|manuscript|ieee)", Path(rel).stem))


def build_context(folder: Path, opts, catalog: dict) -> Context:
    ctx = Context(folder, opts)
    ctx.files = walk_files(folder)
    ctx.rel_files = [f.relative_to(folder).as_posix() for f in ctx.files]

    for f, rel in zip(ctx.files, ctx.rel_files):
        parts = set(rel.split("/")[:-1])
        if parts & JUNK_DIRS or "wandb" in parts:
            continue  # rác hoặc log — chỉ dùng để phát hiện, không đọc nội dung
        ext = f.suffix.lower()
        if ext == ".py":
            src = read_text(f)
            if src is not None:
                ctx.code_units.append(build_code_unit(rel, "py", [(None, src)]))
        elif ext == ".ipynb":
            raw = read_text(f, MAX_NOTEBOOK_BYTES)
            if raw is None:
                ctx.notes.append(f"Bỏ qua notebook quá lớn: `{rel}`.")
                continue
            meta, unit = parse_notebook(rel, raw)
            if meta is None:
                ctx.notes.append(f"Notebook lỗi định dạng JSON: `{rel}`.")
                continue
            ctx.notebooks.append(meta)
            ctx.code_units.append(unit)
            if meta.markdown.strip():
                ctx.text_docs.append(TextDoc(rel, "nb-markdown", meta.markdown))
            if meta.output_text.strip():
                ctx.text_docs.append(TextDoc(rel, "nb-output", meta.output_text))
        elif ext in TEXT_EXT:
            text = read_text(f)
            if text is None:
                continue
            if f.name.lower() == "readme.md" and len(rel.split("/")) == 1:
                ctx.text_docs.append(TextDoc(rel, "readme", text))
                continue
            kind = "tex" if ext == ".tex" else ext.lstrip(".")
            if ext == ".tex":
                text = latex_to_text(text)
            ctx.text_docs.append(TextDoc(rel, kind, text, is_report_path(rel)))
        elif ext == ".docx":
            text = docx_to_text(f)
            if text:
                ctx.text_docs.append(TextDoc(rel, "docx", text, is_report_path(rel)))
        elif ext == ".pdf" and is_report_path(rel):
            text = pdf_to_text(f, ctx.notes)
            if text:
                ctx.text_docs.append(TextDoc(rel, "pdf", text, True))
        elif ext == ".bib":
            text = read_text(f)
            if text:
                ctx.bib_entries.update(parse_bib(text))

    manifest_path = folder / "submission.json"
    if manifest_path.is_file():
        try:
            ctx.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(ctx.manifest, dict):
                ctx.manifest = {}
        except (json.JSONDecodeError, OSError):
            ctx.manifest = {}

    slug = str(ctx.manifest.get("topic_slug", ""))
    topic_id = str(ctx.manifest.get("topic_id", ""))
    ctx.profile = catalog.get(slug) or {}
    group = ctx.profile.get("group")
    if not group and re.match(r"^\d+\.", topic_id):
        group = int(topic_id.split(".")[0])
    ctx.modalities = ctx.profile.get("modalities") or {1: ["signal"], 2: ["image"], 3: ["tabular"],
                                                       4: ["nlp"]}.get(group, ["tabular"])

    ms = (opts.milestone or str(ctx.manifest.get("milestone", "")) or "final").lower()
    ctx.milestone = MILESTONE_ALIASES.get(ms, "FINAL")
    ctx.rubric_name = opts.rubric or ctx.profile.get("rubric") or "standard"
    return ctx


# ---------------------------------------------------------------------------
# Phân tích rò rỉ dữ liệu bằng AST (thứ tự fit / split / resample)
# ---------------------------------------------------------------------------

def call_name(node):
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
    return None


def unparse(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 — ast.unparse có thể lỗi với node lạ
        return ""


def is_train_like(src: str) -> bool:
    return bool(re.search(r"(?i)train|xtr|(^|[^a-z])(x|y|df)_?tr([^a-z]|$)|_tr\b", src))


def is_test_like(src: str) -> bool:
    return bool(re.search(r"(?i)test|xte\b|xval|(^|[^a-z])(x|y|df)_?(te|val|valid)([^a-z]|$)", src))


def is_full_like(src: str) -> bool:
    base = re.split(r"[\[\.(]", src.strip(), maxsplit=1)[0]
    return base.lower() in {"x", "df", "data", "dataset", "features", "x_all", "x_full", "df_all", "all_data",
                            "df_full", "x_data", "x_scaled"}


def analyze_leakage(units: list[CodeUnit]) -> dict:
    """Trả về các sự kiện nghi vấn rò rỉ theo từng file/notebook.

    Quy tắc (heuristic, cần người xác nhận): trong cùng 1 file/notebook, bước
    fit của scaler/imputer/feature selection/SMOTE xuất hiện TRƯỚC lần chia
    tập (train_test_split, *.split(), CV, strat_fold...) mà đối số không phải
    tập Train → nghi vấn fit trên toàn bộ dữ liệu.
    """
    res = {"leaks": [], "unresolved": [], "fits_ok": [], "resample_ok": [], "cv_resampled": [],
           "fillna": [], "splits": [], "pipelines": [], "imblearn_pipeline": False}
    for unit in units:
        events = []
        transformer_vars, splitter_vars, search_vars, resampled = {}, set(), set(), set()
        for offset, src in unit.cells:
            if not src.strip():
                continue
            try:
                tree = ast.parse(src)
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                continue
            nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.Call))]
            nodes.sort(key=lambda n: (n.lineno, n.col_offset, 0 if isinstance(n, ast.Assign) else 1))
            for n in nodes:
                g = offset + n.lineno - 1
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                    cname = call_name(n.value)
                    names = [t.id for t in n.targets if isinstance(t, ast.Name)]
                    tuple_names = [e.id for t in n.targets if isinstance(t, (ast.Tuple, ast.List))
                                   for e in t.elts if isinstance(e, ast.Name)]
                    if cname in TRANSFORMERS:
                        for nm in names:
                            transformer_vars[nm] = cname
                    elif cname in PIPELINE_FUNCS:
                        inner = {call_name(c) for c in ast.walk(n.value) if isinstance(c, ast.Call)}
                        inner |= {transformer_vars.get(x.id) for x in ast.walk(n.value) if isinstance(x, ast.Name)}
                        if inner & set(TRANSFORMERS):
                            for nm in names:
                                transformer_vars[nm] = "Pipeline"
                            res["pipelines"].append((unit.locs[g], unit.lines[g].strip()))
                    elif cname in SPLIT_CLASSES:
                        splitter_vars.update(names)
                    elif cname in SEARCH_CLASSES:
                        search_vars.update(names)
                    elif cname in RESAMPLER_CALLS:
                        resampled.update(names + tuple_names)
                    continue
                if not isinstance(n, ast.Call):
                    continue
                cname = call_name(n)
                func = n.func
                recv = func.value if isinstance(func, ast.Attribute) else None
                first_arg = unparse(n.args[0]) if n.args else ""
                if cname in SPLIT_FUNCS or cname in CV_FUNCS:
                    events.append((g, "split", cname))
                    if cname in CV_FUNCS and len(n.args) > 1 and isinstance(n.args[1], ast.Name) \
                            and n.args[1].id in resampled:
                        events.append((g, "cv_resampled", cname))
                elif cname == "split" and recv is not None and (
                        (isinstance(recv, ast.Name) and recv.id in splitter_vars)
                        or call_name(recv) in SPLIT_CLASSES):
                    events.append((g, "split", "split()"))
                elif cname == "sample" and any(
                        k.arg == "frac" and isinstance(k.value, ast.Constant)
                        and isinstance(k.value.value, (int, float)) and k.value.value < 1 for k in n.keywords):
                    events.append((g, "split", "sample(frac<1)"))  # frac=1 chỉ là xáo trộn, không phải chia tập
                elif cname == "fit" and isinstance(recv, ast.Name) and recv.id in search_vars:
                    events.append((g, "split", "search.fit"))
                    if n.args and isinstance(n.args[0], ast.Name) and n.args[0].id in resampled:
                        events.append((g, "cv_resampled", "GridSearch/RandomizedSearch"))
                if cname in ("fit", "fit_transform"):
                    cls = None
                    if isinstance(recv, ast.Name) and recv.id in transformer_vars:
                        cls = transformer_vars[recv.id]
                    elif call_name(recv) in TRANSFORMERS:
                        cls = call_name(recv)
                    if cls:
                        events.append((g, "fit", (cls, first_arg)))
                elif cname in RESAMPLER_CALLS:
                    events.append((g, "resample", first_arg))
        # strat_fold dạng so sánh (PTB-XL) cũng là 1 bước chia tập
        for i, line in enumerate(unit.lines):
            if re.search(STRAT_FOLD_SPLIT_RX, line):
                events.append((i, "split", "strat_fold"))
            if "imblearn.pipeline" in line:
                res["imblearn_pipeline"] = True
        events.sort(key=lambda e: e[0])
        split_lines = [e[0] for e in events if e[1] == "split"]
        first_split = split_lines[0] if split_lines else None
        for g, kind, info in events:
            loc, code = unit.locs[g], unit.lines[g].strip()
            if kind == "split":
                res["splits"].append((loc, code))
            elif kind == "fit":
                cls, arg = info
                sev = TRANSFORMERS.get(cls, "critical")
                if is_test_like(arg):
                    res["leaks"].append(("critical", loc, code, f"{cls} fit trên tập Test/Validation"))
                elif first_split is None:
                    (res["fits_ok"] if is_train_like(arg) else res["unresolved"]).append((loc, code))
                elif g < first_split and not is_train_like(arg):
                    res["leaks"].append((sev, loc, code, f"{cls} fit trên `{arg or '?'}` TRƯỚC bước chia tập"))
                elif g > first_split and is_full_like(arg):
                    res["leaks"].append(("minor", loc, code, f"{cls} fit trên `{arg}` (có vẻ là toàn bộ dữ liệu)"))
                else:
                    res["fits_ok"].append((loc, code))
            elif kind == "resample":
                if is_test_like(info):
                    res["leaks"].append(("critical", loc, code, "SMOTE/resampling áp dụng lên tập Test"))
                elif first_split is not None and g < first_split and not is_train_like(info):
                    res["leaks"].append(("critical", loc, code, "SMOTE/resampling TRƯỚC bước chia tập"))
                else:
                    res["resample_ok"].append((loc, code))
            elif kind == "cv_resampled":
                res["cv_resampled"].append((loc, code))
        for i, line in enumerate(unit.lines):
            if (first_split is None or i < first_split) and re.search(FILLNA_GLOBAL_RX, line):
                res["fillna"].append((unit.locs[i], line.strip()))
    return res


# ---------------------------------------------------------------------------
# Tiện ích khác: git, mạng, số liệu, độ tương đồng
# ---------------------------------------------------------------------------

def git(args, cwd, timeout=20):
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def git_history(ctx: Context):
    if ctx._git is not None:
        return ctx._git
    info = {"available": False, "commits": []}
    top = git(["rev-parse", "--show-toplevel"], ctx.folder)
    if top:
        top = top.strip()
        rel = os.path.relpath(ctx.folder.resolve(), top)
        out = git(["log", "--no-merges", "--format=%H%x1f%an%x1f%aI%x1f%cI%x1f%s", "--", rel], top)
        if out is not None:
            info["available"] = True
            for line in out.splitlines():
                parts = line.split("\x1f")
                if len(parts) == 5:
                    info["commits"].append(dict(zip(["sha", "author", "adate", "cdate", "subject"], parts)))
    ctx._git = info
    return info


def http_get(url: str, accept: str | None = None, max_bytes=20 * 1024 * 1024):
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:  # noqa: S310 — chỉ https cố định/đã lọc
        return r.status, r.read(max_bytes)


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def verify_doi(doi: str, bib_title: str = ""):
    """→ (trạng thái, ghi chú): ok | mismatch | notfound | error."""
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")
    try:
        _, body = http_get(url, accept="application/json")
        data = json.loads(body.decode("utf-8", errors="ignore"))
        titles = data.get("message", {}).get("title") or [""]
        real = titles[0] if titles else ""
        if bib_title and real:
            ratio = difflib.SequenceMatcher(None, norm_title(bib_title), norm_title(real)).ratio()
            if ratio < 0.75:
                return "mismatch", f"tiêu đề Crossref: “{real[:90]}”"
        return "ok", real[:90]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "notfound", "Crossref không tìm thấy DOI"
        return "error", f"HTTP {e.code}"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return "error", str(e)[:80]


NUM_RX = re.compile(r"(?<![\w.,])(\d{1,4}[.,]\d{1,4})(?!\d|[.,]\d)\s*(%?)")


def report_numbers(ctx: Context) -> list[tuple[str, str]]:
    """Số thập phân trong các dòng bảng kết quả của báo cáo (bỏ dòng y văn)."""
    out = []
    for d in ctx.report_docs:
        for line in d.text.split("\n"):
            if not ("|" in line or "&" in line):
                continue
            if re.search(r"(?i)@|công bố|published|y văn|et al|\b(19|20)\d{2}\b|^\s*\|?\s*:?-{3,}", line):
                continue
            for m in NUM_RX.finditer(line):
                out.append((m.group(1), m.group(2)))
    return out


def output_numbers(ctx: Context) -> tuple[list, list]:
    texts, sources = [], []
    for nb in ctx.notebooks:
        if nb.output_text.strip():
            texts.append(nb.output_text)
            sources.append(nb.path)
    for f, rel in zip(ctx.files, ctx.rel_files):
        if f.suffix.lower() in OUTPUT_EXT and not is_report_path(rel) and f.name != "submission.json" \
                and not re.search(r"(?i)requirements|readme|/data/|^data/", rel):
            t = read_text(f, 5 * 1024 * 1024)
            if t:
                texts.append(t)
                sources.append(rel)
    values = set()
    for t in texts:
        for m in re.finditer(r"-?\d+\.\d+(?:[eE]-?\d+)?|-?\d+", t[:3_000_000]):
            try:
                v = float(m.group(0))
            except ValueError:
                continue
            if math.isfinite(v):
                values.update((v, v * 100, v / 100))  # tỷ lệ ↔ phần trăm
    return sorted(values), sources


def traced(value: float, decimals: int, outputs: list) -> bool:
    """Số trong báo cáo (làm tròn `decimals` chữ số) khớp 1 giá trị output trong nửa đơn vị làm tròn."""
    half = 0.5 * 10 ** -decimals + 1e-9
    i = bisect.bisect_left(outputs, value - half)
    return i < len(outputs) and outputs[i] <= value + half


SIMILARITY_CACHE: dict = {}


def code_shingles(units: list[CodeUnit]) -> set:
    shingles = set()
    for u in units:
        norm = []
        for line in u.lines:
            s = re.sub(r"\s+", " ", line.strip()).lower()
            if len(s) < 12 or s.startswith(("import ", "from ", "print(", "plt.show", "display(")):
                continue
            norm.append(s)
        for i in range(len(norm) - 2):
            shingles.add(hashlib.md5("\n".join(norm[i:i + 3]).encode()).hexdigest()[:16])
    return shingles


def text_shingles(texts: list[str]) -> set:
    out = set()
    for t in texts:
        words = re.findall(r"\w+", t.lower())
        for i in range(len(words) - 7):
            out.add(hashlib.md5(" ".join(words[i:i + 8]).encode()).hexdigest()[:16])
    return out


def folder_fingerprint(folder: Path):
    key = str(folder.resolve())
    if key in SIMILARITY_CACHE:
        return SIMILARITY_CACHE[key]
    units, texts = [], []
    for f in walk_files(folder):
        rel = f.relative_to(folder).as_posix()
        if set(rel.split("/")[:-1]) & JUNK_DIRS:
            continue
        if f.suffix == ".py":
            src = read_text(f)
            if src is not None:
                units.append(build_code_unit(rel, "py", [(None, src)]))
        elif f.suffix == ".ipynb":
            raw = read_text(f, MAX_NOTEBOOK_BYTES)
            if raw:
                _, unit = parse_notebook(rel, raw)
                if unit:
                    units.append(unit)
        elif f.suffix in (".md", ".tex") and is_report_path(rel):
            t = read_text(f)
            if t:
                texts.append(t)
    fp = (code_shingles(units), text_shingles(texts))
    SIMILARITY_CACHE[key] = fp
    return fp


def containment(a: set, b: set) -> float:
    return len(a & b) / len(a) if a else 0.0


# ---------------------------------------------------------------------------
# Các phép kiểm tra (mỗi hàm trả về 1 Finding)
# ---------------------------------------------------------------------------

CHECKS: dict = {}


def check(check_id, title):
    def deco(fn):
        CHECKS[check_id] = (title, fn)
        return fn
    return deco


def F(ctx, cid, status, severity="major", detail="", evidence=None, suggestion="", penalty=""):
    return Finding(cid, CHECKS[cid][0] if cid in CHECKS else cid, status, severity, detail,
                   [f"`{loc}` — {snippet(code)}" if code else f"`{loc}`" for loc, code in (evidence or [])],
                   suggestion, penalty)


def snippet(s: str, limit=110) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > limit:
        s = s[: limit - 1] + "…"
    return "`" + s.replace("`", "'") + "`"


def leak(ctx):
    if ctx._leak is None:
        ctx._leak = analyze_leakage(ctx.code_units)
    return ctx._leak


def models_found(ctx):
    if ctx._models is not None:
        return ctx._models
    found = {}
    for name in CLASSIC_MODELS:
        hits = ctx.grep_code(r"\b" + name + r"\s*\(", limit=1)
        if hits:
            found[name] = hits[0]
    for name, rx in DEEP_MODELS:
        hits = ctx.grep_code(rx, limit=1)
        if hits:
            found[name] = hits[0]
    for u in ctx.code_units:
        for i, line in enumerate(u.lines):
            m = re.search(CUSTOM_MODEL_RX, line)
            if m:
                found[f"{m.group(1)} (nn.Module tự định nghĩa)"] = (u.locs[i], line.strip())
    ctx._models = found
    return found


def override(ctx, key, default=None):
    return (ctx.profile.get("overrides") or {}).get(key, default)


# -- M0: Hồ sơ nộp bài ------------------------------------------------------

@check("S.MANIFEST", "submission.json khớp tên thư mục và ngân hàng đề tài")
def c_manifest(ctx):
    m = ctx.manifest
    if not m:
        return F(ctx, "S.MANIFEST", FAIL, "critical", "Thiếu hoặc lỗi cú pháp `submission.json`.",
                 suggestion="Copy `submissions/_TEMPLATE/submission.json` và điền đủ trường bắt buộc.")
    problems, warns = [], []
    slug, gid, code = str(m.get("topic_slug", "")), str(m.get("topic_id", "")), str(m.get("group_code", ""))
    if slug and code and ctx.folder.name != f"{slug}_{code}":
        warns.append(f"tên thư mục `{ctx.folder.name}` ≠ `<topic_slug>_<group_code>` = `{slug}_{code}`")
    if not ctx.profile:
        warns.append(f"`topic_slug` = `{slug}` không có trong ngân hàng đề tài (rubric/topics/) — nếu là đề tài "
                     f"tự đề xuất cần giảng viên duyệt; công cụ dùng yêu cầu chung theo nhóm đề tài")
    elif gid and gid != str(ctx.profile.get("topic_id")):
        problems.append(f"`topic_id` = `{gid}` nhưng đề tài `{slug}` có mã `{ctx.profile.get('topic_id')}`")
    ms = m.get("milestone")
    if ms and str(ms).lower() not in MILESTONE_ALIASES:
        warns.append(f"`milestone` = `{ms}` không hợp lệ (m1, m2, m3, m4, final)")
    link = m.get("repo_link_optional")
    if link and not re.match(r"^https://github\.com/[\w.-]+/[\w.-]+/?$", str(link)):
        warns.append("`repo_link_optional` nên là URL dạng https://github.com/<owner>/<repo>")
    if problems:
        return F(ctx, "S.MANIFEST", FAIL, "major", "; ".join(problems + warns),
                 suggestion="Sửa `submission.json` cho khớp mã đề tài trong ngân hàng đề tài.")
    if warns:
        return F(ctx, "S.MANIFEST", WARN, "minor", "; ".join(warns),
                 suggestion="Đồng bộ tên thư mục và các trường trong `submission.json`.")
    return F(ctx, "S.MANIFEST", PASS, "info", f"Đề tài {gid} `{slug}`, nhóm `{code}`.")


@check("S.README", "README.md tóm tắt bài làm đã điền (không còn nội dung mẫu)")
def c_readme(ctx):
    rd = ctx.readme
    if not rd:
        return F(ctx, "S.README", FAIL, "major", "Thiếu `README.md` ở gốc thư mục nộp bài.",
                 suggestion="Copy README từ `_TEMPLATE` và điền tóm tắt + cách chạy lại.")
    left = [p for p in README_PLACEHOLDERS if p.lower() in rd.text.lower()]
    if left:
        return F(ctx, "S.README", FAIL, "major", "README còn nội dung mẫu: " + ", ".join(f"“{p}”" for p in left),
                 suggestion="Thay toàn bộ phần trong ngoặc bằng thông tin thật của nhóm.")
    if len(rd.text.strip()) < 300:
        return F(ctx, "S.README", WARN, "minor", "README quá ngắn (< 300 ký tự).",
                 suggestion="Bổ sung: đề tài, mô hình chính, kết quả nổi bật, cấu trúc thư mục, cách chạy lại.")
    return F(ctx, "S.README", PASS, "info")


@check("S.MEMBERS", "Thành viên ghi bằng MSSV (không dùng họ tên thật)")
def c_members(ctx):
    members = ctx.manifest.get("members") or []
    if not isinstance(members, list) or not members:
        return F(ctx, "S.MEMBERS", FAIL, "major", "`members` trống hoặc không phải danh sách.")
    names = [x for x in members if isinstance(x, str) and re.search(r"[^\W\d_]{2,}\s+[^\W\d_]{2,}", x)]
    odd = [x for x in members if not (isinstance(x, str) and re.fullmatch(r"\d{6,10}", x.strip()))]
    if names:
        return F(ctx, "S.MEMBERS", FAIL, "major", f"{len(names)} giá trị có vẻ là họ tên thật.",
                 suggestion="Chỉ dùng MSSV — repo công khai vĩnh viễn (CONTRIBUTING Mục 3).",
                 penalty="Quy tắc quyền riêng tư của repo nộp bài.")
    if odd:
        return F(ctx, "S.MEMBERS", WARN, "minor",
                 f"{len(odd)}/{len(members)} giá trị không giống MSSV (6–10 chữ số): "
                 + ", ".join(f"`{str(x)[:30]}`" for x in odd[:4]),
                 suggestion="Kiểm tra lại `members` trong `submission.json`.")
    return F(ctx, "S.MEMBERS", PASS, "info", f"{len(members)} thành viên.")


@check("S.REPORT_PRESENT", "Có báo cáo đọc được (report/: .md/.tex/.docx/.pdf)")
def c_report_present(ctx):
    reports = ctx.report_docs
    pdfs = ctx.files_matching(r"^(report|reports|paper|bao_cao)/.*\.pdf$")
    if reports:
        kinds = sorted({d.kind for d in reports})
        return F(ctx, "S.REPORT_PRESENT", PASS, "info",
                 f"{len(reports)} tài liệu báo cáo ({', '.join(kinds)}): "
                 + ", ".join(f"`{d.path}`" for d in reports[:4]))
    if pdfs:
        return F(ctx, "S.REPORT_PRESENT", WARN, "minor",
                 "Báo cáo chỉ có PDF, không trích xuất được văn bản → các tiêu chí về báo cáo cần review thủ công.",
                 suggestion="Nộp kèm file nguồn (.tex Overleaf hoặc .md) để kiểm tra tự động và tái lập.")
    return F(ctx, "S.REPORT_PRESENT", FAIL, "major", "Không tìm thấy báo cáo trong `report/`.",
             suggestion="Đặt báo cáo (IEEE paper / báo cáo rút gọn) vào `report/`.")


@check("S.TOPIC_FILES", "Sản phẩm bàn giao theo hướng dẫn của đề tài")
def c_topic_files(ctx):
    expected = [e for e in ctx.profile.get("expected_files", [])
                if milestone_reached(ctx, e.get("milestone", "M0"))]
    if not expected:
        return F(ctx, "S.TOPIC_FILES", NA, "info", "Đề tài không quy định tên file bàn giao cụ thể.")
    missing = [e for e in expected if not ctx.files_matching(e["pattern"])]
    if not missing:
        return F(ctx, "S.TOPIC_FILES", PASS, "info", f"Đủ {len(expected)} sản phẩm bàn giao.")
    sev = "major" if any(e.get("severity") == "major" for e in missing) else "minor"
    return F(ctx, "S.TOPIC_FILES", WARN, sev,
             "Thiếu: " + "; ".join(f"{e['title']} (`{e['pattern']}`)" for e in missing),
             suggestion="Đặt tên/tách file đúng sản phẩm bàn giao trong hướng dẫn đề tài để GV đối chiếu nhanh.")


# -- M1 ---------------------------------------------------------------------

@check("M1.PROBLEM", "Phát biểu bài toán: Input/Output, bối cảnh lâm sàng, chi phí FP vs FN")
def c_problem(ctx):
    parts = {
        "Input/Output (X, Y)": ctx.grep_text(PROBLEM_IO_RX, "report", limit=1),
        "Bối cảnh lâm sàng / tính cấp thiết": ctx.find_section(PROBLEM_CONTEXT_RX)
        or ctx.grep_text(PROBLEM_CONTEXT_RX, "report", limit=1),
        "Chi phí sai lầm bất đối xứng (FP vs FN)": ctx.grep_text(PROBLEM_COST_RX, "report", limit=1),
    }
    have = [k for k, v in parts.items() if v]
    miss = [k for k, v in parts.items() if not v]
    ev = [v[0] for v in parts.values() if v]
    if not miss:
        return F(ctx, "M1.PROBLEM", PASS, "info", "Đủ 3 thành phần.", ev)
    status = WARN if have else FAIL
    return F(ctx, "M1.PROBLEM", status, "major", "Chưa thấy: " + "; ".join(miss), ev,
             "Mục Đặt vấn đề cần nêu rõ đối tượng bệnh nhân, Input (X), Output (Y) và phân tích hậu quả "
             "bỏ sót (FN) so với báo động giả (FP) — từ đó chọn độ đo/ngưỡng phù hợp.")


@check("M1.LIT_COUNT", "Khảo sát y văn đủ số lượng bài báo")
def c_lit_count(ctx):
    need = int(override(ctx, "min_references", 10))
    report_text = "\n".join(d.text for d in ctx.docs_for("report"))
    dois = {m.group(0).rstrip(".") for m in re.finditer(DOI_RX, report_text)}
    keys = set(re.findall(CITEKEY_HMYT_RX, report_text))
    for m in re.finditer(CITEKEY_PANDOC_RX, report_text):
        keys.update(re.findall(r"@([\w:\-]+)", m.group(0)))
    for m in re.finditer(CITEKEY_LATEX_RX, report_text):
        keys.update(k.strip() for k in m.group(1).split(","))
    ref_items = 0
    sec = re.split(r"(?im)^\W*(?:\d+\.?\s*)?(?:references|tài liệu tham khảo|bibliography)\b.*$", report_text)
    if len(sec) > 1:
        ref_items = len(re.findall(r"(?m)^\s*(\[\d+\]|\d+\.\s|[-*]\s)", sec[-1]))
    n = max(len(ctx.bib_entries), len(dois), len(keys), ref_items)
    detail = (f"Ước lượng {n} tài liệu (bib: {len(ctx.bib_entries)}, DOI: {len(dois)}, citekey: {len(keys)}, "
              f"mục tham khảo: {ref_items}); yêu cầu ≥ {need}.")
    if n >= need:
        return F(ctx, "M1.LIT_COUNT", PASS, "info", detail)
    status = WARN if n >= max(1, math.ceil(need / 2)) else FAIL
    return F(ctx, "M1.LIT_COUNT", status, "major", detail,
             suggestion=f"Bổ sung y văn tới ≥ {need} bài (Q1/Q2, 3–5 năm gần nhất), lưu vào `references.bib` "
                        f"kèm DOI đã xác minh qua Crossref; không dùng blog/diễn đàn.")


@check("M1.LIT_MATRIX", "Bảng ma trận so sánh y văn (Dataset, phương pháp, kết quả, hạn chế)")
def c_lit_matrix(ctx):
    best = None
    for d in ctx.docs_for("report"):
        lines = d.text.split("\n")
        for i, line in enumerate(lines[:-1]):
            if "|" in line and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
                head = line.lower()
                has_data = re.search(r"dataset|dữ liệu|data", head)
                has_res = re.search(r"auc|f1|accuracy|kết quả|metric|độ đo|hiệu năng|result", head)
                has_lim = re.search(r"hạn chế|nhược điểm|limitation|weakness|khoảng trống|gap", head)
                if has_data and has_res:
                    cand = (2 if has_lim else 1, f"{d.path}:{i + 1}", line.strip())
                    best = max(best, cand) if best else cand
        if d.kind == "tex":
            for m in re.finditer(r"(?s)\\begin\{tabular\}.*?\\end\{tabular\}", d.text):
                block = m.group(0).lower()
                if re.search(r"dataset|dữ liệu", block) and re.search(r"auc|f1|accuracy|kết quả", block):
                    lim = 2 if re.search(r"hạn chế|nhược điểm|limitation|weakness|gap", block) else 1
                    cand = (lim, d.path, "LaTeX tabular")
                    best = max(best, cand) if best else cand
    if not best:
        return F(ctx, "M1.LIT_MATRIX", FAIL, "major", "Không thấy bảng so sánh y văn.",
                 suggestion="Lập bảng ma trận ≥10 bài: Tác giả/Năm | Dataset | Kiến trúc | F1/AUC | XAI | "
                            "Hạn chế — rồi chỉ ra khoảng trống mà nhóm giải quyết.")
    if best[0] == 1:
        return F(ctx, "M1.LIT_MATRIX", WARN, "minor", "Có bảng so sánh nhưng thiếu cột hạn chế/nhược điểm.",
                 [best[1:]], "Thêm cột “Hạn chế” cho từng bài — đây là tiêu chí phân biệt mức Đạt và Xuất sắc.")
    return F(ctx, "M1.LIT_MATRIX", PASS, "info", evidence=[best[1:]])


@check("M1.EDA", "EDA: phân bố nhãn, dữ liệu khuyết, tương quan, trực quan hóa đặc thù y sinh")
def c_eda(ctx):
    got, ev = [], []
    for label, rx in EDA_SIGNALS:
        hits = ctx.grep_code(rx, limit=1)
        if hits:
            got.append(label)
            ev.append(hits[0])
    if "Dữ liệu khuyết" not in got and ctx.grep_text(r"(?i)khuyết|missing", "report", limit=1):
        got.append("Dữ liệu khuyết")
    eda_files = ctx.files_matching(r"(?i)eda|explor|kham_pha|1_")
    detail = f"Dấu hiệu EDA: {', '.join(got) or 'không có'}."
    if eda_files:
        detail += f" File EDA: {', '.join(f'`{x}`' for x in eda_files[:3])}."
    if len(got) >= 4 and "Trực quan hóa" in got:
        return F(ctx, "M1.EDA", PASS, "info", detail, ev[:3])
    miss = [label for label, _ in EDA_SIGNALS if label not in got][:4]
    status = WARN if len(got) >= 2 else FAIL
    return F(ctx, "M1.EDA", status, "major", detail + " Còn thiếu: " + ", ".join(miss) + ".", ev[:3],
             "Notebook EDA nên có: phân bố nhãn (mức mất cân bằng), ma trận/biểu đồ dữ liệu khuyết, phân bố "
             "tuổi/giới, ma trận tương quan và trực quan mẫu ảnh/tín hiệu.")


@check("M1.GIT", "Kỷ luật Git: nhánh gd1–gd7, commit chuẩn, Issue báo cáo tuần")
def c_git(ctx):
    hist = git_history(ctx)
    lines = []
    commits = hist["commits"]
    if commits:
        conv = [c for c in commits if re.match(
            r"^(feat|fix|docs|style|refactor|perf|test|chore|build|ci|data|exp|revert)(\([^)]+\))?!?: .+", c["subject"])]
        lines.append(f"{len(commits)} commit chạm thư mục, {len(conv)}/{len(commits)} đúng chuẩn "
                     f"`<type>(<scope>): <subject>`")
    link = str(ctx.manifest.get("repo_link_optional") or "")
    if not link:
        return F(ctx, "M1.GIT", MANUAL, "info", "; ".join(lines + ["chưa khai báo `repo_link_optional`"]),
                 suggestion="Khai báo repo nhóm trong `submission.json` (`repo_link_optional`) để GV kiểm tra "
                            "nhánh gd1–gd7, Issue báo cáo tuần và PR nghiệm thu từng mốc.")
    if not ctx.opts.online:
        return F(ctx, "M1.GIT", MANUAL, "info", "; ".join(lines + [f"repo nhóm: {link} (chạy `--online` để "
                                                                  f"kiểm tra nhánh gd1–gd7)"]))
    if not re.match(r"^https://github\.com/[\w.-]+/[\w.-]+/?$", link):
        return F(ctx, "M1.GIT", MANUAL, "info", f"repo nhóm `{link}` không phải URL GitHub hợp lệ.")
    out = git(["ls-remote", "--heads", link.rstrip("/")], ctx.folder, timeout=30)
    if out is None:
        return F(ctx, "M1.GIT", MANUAL, "info", f"Không truy cập được {link} (private hoặc lỗi mạng).")
    branches = [ln.split("refs/heads/")[-1] for ln in out.splitlines() if "refs/heads/" in ln]
    upto = {"M0": 0, "M1": 2, "M2": 4, "M3": 5, "M4": 6, "FINAL": 7}[ctx.milestone]
    need = [f"gd{i}" for i in range(1, upto + 1)]
    have = [b for b in need if any(x.startswith(b) for x in branches)]
    detail = "; ".join(lines + [f"nhánh gd có trên repo nhóm: {', '.join(have) or 'không có'} / cần {len(need)}"])
    if len(have) == len(need):
        return F(ctx, "M1.GIT", PASS, "info", detail)
    return F(ctx, "M1.GIT", WARN, "minor", detail,
             suggestion="Tạo đủ nhánh `gd1-proposal`…`gd7-ieee-paper` theo Khung quản lý; mỗi mốc 1 PR nghiệm thu.")


# -- M2 ---------------------------------------------------------------------

@check("M2.SPLIT_PATIENT", "Chia tập ở mức bệnh nhân (patient-level split)")
def c_split_patient(ctx):
    lk = leak(ctx)
    split_cfg = ctx.profile.get("split") or {}
    if not ctx.code_units:
        return F(ctx, "M2.SPLIT_PATIENT", FAIL, "major", "Chưa có mã nguồn.")
    if split_cfg.get("one_row_per_patient"):
        if lk["splits"]:
            return F(ctx, "M2.SPLIT_PATIENT", PASS, "info",
                     "Hồ sơ đề tài: mỗi dòng = 1 bệnh nhân độc lập → chia theo dòng là chia mức bệnh nhân. "
                     + split_cfg.get("note", ""), lk["splits"][:2])
        return F(ctx, "M2.SPLIT_PATIENT", FAIL, "major", "Không tìm thấy bước chia Train/Test trong code.",
                 suggestion="Chia Train/Test (stratified) trước mọi bước fit.")
    hits = ctx.grep_code(PATIENT_SPLIT_RX, limit=3)
    if hits:
        return F(ctx, "M2.SPLIT_PATIENT", PASS, "info", split_cfg.get("note", ""), hits)
    if lk["splits"]:
        return F(ctx, "M2.SPLIT_PATIENT", FAIL, "major",
                 "Có chia tập nhưng không thấy dấu hiệu chia theo bệnh nhân (GroupKFold / groups= / patient_id"
                 + (" / strat_fold" if "signal" in ctx.modalities else "") + "). " + split_cfg.get("note", ""),
                 lk["splits"][:2],
                 "Dùng `GroupKFold`/`GroupShuffleSplit(groups=patient_id)` hoặc chia trên danh sách ID bệnh nhân "
                 "duy nhất; nếu dữ liệu thật sự 1 bản ghi = 1 bệnh nhân, chứng minh bằng assert + nêu trong báo cáo.",
                 "Nếu cùng 1 bệnh nhân nằm ở cả Train và Test: " + PENALTY_LEAK)
    return F(ctx, "M2.SPLIT_PATIENT", WARN, "major", "Không tìm thấy bước chia tập trong code đã nộp.",
             suggestion="Đưa đủ code chia tập vào bài nộp để kiểm toán Zero-Leakage.")


@check("M2.SPLIT_ASSERT", "Có assert tự động chứng minh Train ∩ Test = ∅ (theo patient_id)")
def c_split_assert(ctx):
    if (ctx.profile.get("split") or {}).get("one_row_per_patient"):
        hits = ctx.grep_code(r"^\s*assert\b", limit=2)
        return F(ctx, "M2.SPLIT_ASSERT", NA, "info", "Đề tài 1 dòng = 1 bệnh nhân — không bắt buộc.", hits,
                 "Khuyến nghị: `assert not df.duplicated().any()` và assert chỉ số Train/Test không giao nhau.")
    hits = ctx.grep_code(SPLIT_ASSERT_RX, limit=3)
    if hits:
        return F(ctx, "M2.SPLIT_ASSERT", PASS, "info", evidence=hits)
    return F(ctx, "M2.SPLIT_ASSERT", WARN, "major", "Chưa thấy assertion kiểm tra độc lập bệnh nhân.",
             suggestion="Thêm: `assert set(train.patient_id).isdisjoint(set(test.patient_id))` ngay sau bước chia "
                        "— đây là yêu cầu mức Xuất sắc của tiêu chí Zero-Leakage (8 đ).")


@check("M2.LEAK_FIT", "Scaler/Imputer/Feature selection chỉ fit trên Train")
def c_leak_fit(ctx):
    lk = leak(ctx)
    leaks = [x for x in lk["leaks"] if "SMOTE" not in x[3]]
    if leaks:
        worst = min(leaks, key=lambda x: SEV_ORDER[x[0]])[0]
        status = FAIL if worst == "critical" else WARN
        ev = [(loc, code) for _, loc, code, _ in leaks]
        detail = "Nghi vấn: " + "; ".join(sorted({x[3] for x in leaks}))[:400]
        return F(ctx, "M2.LEAK_FIT", status, worst, detail, ev,
                 "Chia tập TRƯỚC, rồi `fit`/`fit_transform` chỉ trên Train và `transform` cho Val/Test; tốt nhất "
                 "gói vào `sklearn.pipeline.Pipeline` để CV tự cô lập. Nếu bước fit chỉ phục vụ trực quan hóa "
                 "EDA (vd PCA), tách hẳn khỏi pipeline mô hình và ghi chú rõ.",
                 PENALTY_LEAK if status == FAIL else "")
    if lk["unresolved"]:
        return F(ctx, "M2.LEAK_FIT", WARN, "minor",
                 "Có bước fit trong file không có bước chia tập — không xác định được thứ tự, cần xem thủ công.",
                 lk["unresolved"][:3], "Đặt bước chia tập và bước fit trong cùng 1 pipeline/notebook.")
    if lk["fits_ok"] or lk["pipelines"]:
        return F(ctx, "M2.LEAK_FIT", PASS, "info",
                 f"{len(lk['fits_ok'])} bước fit trên Train/sau khi chia; {len(lk['pipelines'])} Pipeline.",
                 (lk["fits_ok"] + lk["pipelines"])[:2])
    return F(ctx, "M2.LEAK_FIT", NA, "info", "Không dùng scaler/imputer/feature selection.")


@check("M2.LEAK_RESAMPLE", "SMOTE/oversampling chỉ áp dụng trên Train")
def c_leak_resample(ctx):
    lk = leak(ctx)
    leaks = [x for x in lk["leaks"] if "SMOTE" in x[3]]
    if leaks:
        return F(ctx, "M2.LEAK_RESAMPLE", FAIL, "critical", "; ".join(sorted({x[3] for x in leaks})),
                 [(loc, code) for _, loc, code, _ in leaks],
                 "SMOTE tuyệt đối chỉ `fit_resample` trên Train (sau khi chia); đánh giá trên Test gốc chưa bị "
                 "resample. Với CV: dùng `imblearn.pipeline.Pipeline([('smote', SMOTE()), ('clf', ...)])`.",
                 PENALTY_LEAK)
    if lk["resample_ok"]:
        return F(ctx, "M2.LEAK_RESAMPLE", PASS, "info", evidence=lk["resample_ok"][:2])
    return F(ctx, "M2.LEAK_RESAMPLE", NA, "info", "Không dùng SMOTE/oversampling.")


@check("M2.LEAK_CV_RESAMPLED", "Không cross-validate trên dữ liệu đã SMOTE (lạc quan giả tạo)")
def c_leak_cv_resampled(ctx):
    lk = leak(ctx)
    if lk["cv_resampled"] and not lk["imblearn_pipeline"]:
        return F(ctx, "M2.LEAK_CV_RESAMPLED", WARN, "major",
                 "CV chạy trên dữ liệu đã SMOTE: mẫu tổng hợp ở fold validation được nội suy từ fold train → "
                 "điểm CV cao giả tạo (xem ví dụ minh họa đề tài 3.4, Mục 4.3).", lk["cv_resampled"][:3],
                 "Đưa SMOTE vào `imblearn.pipeline.Pipeline` rồi mới `cross_val_score`; chỉ báo cáo con số trên "
                 "Test gốc hoặc CV đúng quy trình.")
    if lk["cv_resampled"]:
        return F(ctx, "M2.LEAK_CV_RESAMPLED", WARN, "minor", "Có CV trên biến đã resample nhưng có dùng "
                 "imblearn Pipeline — kiểm tra lại thủ công.", lk["cv_resampled"][:2])
    return F(ctx, "M2.LEAK_CV_RESAMPLED", PASS if lk["resample_ok"] else NA, "info")


@check("M2.LEAK_FILLNA", "Điền khuyết bằng thống kê toàn bộ dữ liệu trước khi chia")
def c_leak_fillna(ctx):
    lk = leak(ctx)
    if lk["fillna"]:
        return F(ctx, "M2.LEAK_FILLNA", WARN, "major",
                 "`fillna(mean/median)` trước bước chia tập = dùng thống kê của cả tập Test.", lk["fillna"][:3],
                 "Dùng `SimpleImputer`/`KNNImputer`/`IterativeImputer` fit trên Train (hoặc trong Pipeline).",
                 PENALTY_LEAK)
    return F(ctx, "M2.LEAK_FILLNA", PASS, "info")


@check("M2.PREPROCESS", "Tiền xử lý đặc thù y sinh theo loại dữ liệu")
def c_preprocess(ctx):
    found, ev = {}, []
    for mod in ctx.modalities:
        rx = PREPROCESS_BY_MODALITY.get(mod)
        if not rx:
            continue
        for u in ctx.code_units:
            for i, line in enumerate(u.lines):
                for m in re.finditer(rx, line):
                    tok = m.group(0).lower()
                    if tok not in found:
                        found[tok] = (u.locs[i], line.strip())
    ev = list(found.values())[:3]
    label = "/".join(ctx.modalities)
    if len(found) >= 2:
        return F(ctx, "M2.PREPROCESS", PASS, "info", f"Kỹ thuật ({label}): " + ", ".join(list(found)[:8]), ev,
                 "Mức Xuất sắc còn yêu cầu GIẢI THÍCH lý do y sinh của tham số (vd tần số cắt, cửa sổ HU) trong "
                 "báo cáo — GV đánh giá thủ công.")
    hints = {"signal": "lọc nhiễu baseline wander (high-pass ~0,5 Hz), notch 50/60 Hz, wavelet",
             "image": "chuẩn hóa ánh sáng (CLAHE), cắt ROI, augmentation hợp lý lâm sàng",
             "tabular": "điền khuyết (fit trên Train), mã hóa biến phân loại, xử lý ngoại lai",
             "nlp": "chuẩn hóa viết tắt y khoa, tokenization, xử lý phủ định, văn bản dài"}
    sug = "; ".join(hints[m] for m in ctx.modalities if m in hints)
    return F(ctx, "M2.PREPROCESS", WARN if found else FAIL, "major",
             f"Ít dấu hiệu tiền xử lý đặc thù ({label}): " + (", ".join(found) or "không có"), ev,
             f"Bổ sung tiền xử lý đặc thù: {sug} — và giải thích lý do y sinh.")


@check("M2.BASELINE", "Có ≥ 2 mô hình Baseline đối chứng")
def c_baseline(ctx):
    found = models_found(ctx)
    names = list(found)
    detail = f"Mô hình phát hiện: {', '.join(names) or 'không có'}."
    if len(names) >= 2:
        return F(ctx, "M2.BASELINE", PASS, "info", detail, list(found.values())[:3])
    return F(ctx, "M2.BASELINE", WARN if names else FAIL, "major", detail, list(found.values())[:2],
             "Xây ≥ 2 baseline (vd Logistic Regression + Random Forest, hoặc CNN chuẩn) và ghi bảng độ đo cơ "
             "sở làm mốc tham chiếu trước khi làm mô hình đề xuất.")


@check("M2.PR", "PR nghiệm thu đúng mẫu, có Biomedical Reproducibility Checklist")
def c_pr(ctx):
    body = ctx.opts.pr_body_text
    if body is None:
        return F(ctx, "M2.PR", MANUAL, "info", "Kiểm tra mô tả PR (mẫu `.github/PULL_REQUEST_TEMPLATE.md`).")
    checked = len(re.findall(r"(?im)^\s*[-*]\s*\[[xX]\]", body))
    unchecked = len(re.findall(r"(?im)^\s*[-*]\s*\[ \]", body))
    if checked + unchecked == 0:
        return F(ctx, "M2.PR", FAIL, "minor", "Mô tả PR không có checklist.",
                 suggestion="Dùng mẫu PR của repo và đánh dấu Biomedical Reproducibility Checklist.")
    if unchecked:
        return F(ctx, "M2.PR", WARN, "minor", f"Checklist: {checked}/{checked + unchecked} mục đã đánh dấu.",
                 suggestion="Hoàn thành hoặc giải thích các mục chưa đánh dấu trong mô tả PR.")
    return F(ctx, "M2.PR", PASS, "info", f"Checklist đầy đủ ({checked} mục).")


# -- M3 ---------------------------------------------------------------------

@check("M3.PROPOSED", "Mô hình đề xuất có căn cứ (khác baseline)")
def c_proposed(ctx):
    found = models_found(ctx)
    custom = [k for k in found if "tự định nghĩa" in k]
    text = ctx.grep_text(PROPOSED_TEXT_RX, "report", limit=2)
    ev = text + [found[k] for k in custom][:2]
    if len(found) >= 3 or custom or text:
        return F(ctx, "M3.PROPOSED", MANUAL, "info",
                 f"{len(found)} mô hình, {len(custom)} kiến trúc tự định nghĩa; tính sáng tạo/căn cứ khoa học "
                 f"do GV đánh giá (10 đ).", ev)
    return F(ctx, "M3.PROPOSED", WARN, "major", "Chưa thấy mô hình đề xuất khác các baseline.", ev,
             "Nêu rõ mô hình đề xuất, cơ chế cải tiến so với baseline và căn cứ y văn (vd hàm loss chống mất "
             "cân bằng, kết hợp đặc trưng cục bộ + ngữ cảnh).")


@check("M3.ABLATION", "Ablation study định lượng đóng góp từng thành phần")
def c_ablation(ctx):
    code = ctx.files_matching(r"(?i)ablation") or [loc for loc, _ in ctx.grep_code(r"(?i)ablation", limit=1)]
    sec = ctx.find_section(r"ablation|nghiên cứu loại bỏ|loại bỏ thành phần")
    text = ctx.grep_text(ABLATION_RX, "report", limit=2)
    ev = [(c, "") for c in code[:1]] + (sec[:1] or text[:1])
    if code and (sec or text):
        return F(ctx, "M3.ABLATION", PASS, "info", evidence=ev)
    if code or sec or text:
        return F(ctx, "M3.ABLATION", WARN, "major", "Có dấu hiệu ablation nhưng chưa đủ cả code và bảng/mục "
                 "trong báo cáo.", ev,
                 "Trình bày bảng ablation: mỗi dòng bỏ/đổi 1 thành phần (module, hàm loss, augmentation, SMOTE, "
                 "nhóm đặc trưng) và Δ độ đo so với mô hình đầy đủ.")
    return F(ctx, "M3.ABLATION", FAIL, "major", "Không có ablation study.",
             suggestion="Thiết kế ablation để chứng minh vì sao mô hình đề xuất tốt hơn baseline.")


@check("M3.REGULARIZATION", "Kỹ thuật huấn luyện & kiểm soát quá khớp")
def c_regularization(ctx):
    sig = {}
    for loc, code in ctx.grep_code(REG_SIGNALS_RX):
        for m in re.finditer(REG_SIGNALS_RX, code):
            sig.setdefault(m.group(0).strip(), (loc, code))
    curves = ctx.grep_code(CURVES_RX, limit=1) or ctx.grep_text(r"(?i)learning curve|đường cong (học|loss)|"
                                                                r"val(idation)? loss", "report", limit=1)
    detail = f"Kỹ thuật: {', '.join(list(sig)[:8]) or 'không có'}; đồ thị loss/learning curve: " \
             f"{'có' if curves else 'chưa thấy'}."
    ev = list(sig.values())[:2] + curves[:1]
    if len(sig) >= 2 and curves:
        return F(ctx, "M3.REGULARIZATION", PASS, "info", detail, ev)
    return F(ctx, "M3.REGULARIZATION", WARN if sig else FAIL, "major", detail, ev,
             "Dùng Early Stopping trên Val loss, Dropout/Weight decay (DL) hoặc CV + giới hạn độ sâu (ML cổ "
             "điển); vẽ đường cong Train/Val để chứng minh không quá khớp.")


@check("M3.TRACKING", "Theo dõi huấn luyện (W&B / TensorBoard)")
def c_tracking(ctx):
    dash = ctx.grep_code(TRACK_DASH_RX, limit=2) + ctx.grep_text(r"wandb\.ai/|tensorboard\.dev", "text", limit=1)
    if dash:
        return F(ctx, "M3.TRACKING", PASS, "info", evidence=dash[:2])
    if override(ctx, "tracking_optional"):
        return F(ctx, "M3.TRACKING", NA, "info", "Không bắt buộc với đề tài này.")
    files = ctx.grep_code(TRACK_FILE_RX, limit=2)
    if files:
        return F(ctx, "M3.TRACKING", WARN, "minor", "Có log ra file nhưng chưa có dashboard.", files,
                 "Tích hợp Weights & Biases/TensorBoard và dán link dashboard công khai vào README.")
    return F(ctx, "M3.TRACKING", FAIL, "minor", "Không lưu log huấn luyện.",
             suggestion="Tích hợp W&B (`wandb.init`, `wandb.log`) hoặc TensorBoard `SummaryWriter`.")


# -- M4 ---------------------------------------------------------------------

def metric_presence(ctx):
    pres = {}
    for key, (rx, _) in METRIC_PATTERNS.items():
        hit = ctx.grep_code(rx, limit=1) or ctx.grep_text(rx, "report", limit=1)
        if hit:
            pres[key] = hit[0]
    return pres


def required_metrics(ctx):
    sets = ctx.profile.get("metric_set") or "binary"
    sets = sets if isinstance(sets, list) else [sets]
    req = []
    for s in sets:
        for k in METRIC_SETS.get(s, []):
            if k not in req:
                req.append(k)
    return req, sets


@check("M4.METRICS", "Hệ thống đa độ đo y tế phù hợp loại bài toán")
def c_metrics(ctx):
    pres = metric_presence(ctx)
    req, sets = required_metrics(ctx)
    miss = [METRIC_PATTERNS[k][1] for k in req if k not in pres]
    got = [METRIC_PATTERNS[k][1] for k in pres]
    detail = f"Loại bài toán: {', '.join(sets)}. Độ đo có: {', '.join(got) or 'không có'}."
    if "accuracy" in pres and not (set(pres) & INFORMATIVE_METRICS):
        return F(ctx, "M4.METRICS", FAIL, "major", detail + " CHỈ có Accuracy.", [pres["accuracy"]],
                 "Báo cáo tối thiểu " + ", ".join(METRIC_PATTERNS[k][1] for k in req) + ".",
                 PENALTY_ACCURACY_ONLY)
    if not pres:
        return F(ctx, "M4.METRICS", FAIL, "major", "Không thấy độ đo đánh giá.",
                 suggestion="Báo cáo " + ", ".join(METRIC_PATTERNS[k][1] for k in req) + ".")
    if miss:
        return F(ctx, "M4.METRICS", WARN, "major", detail + " Thiếu: " + ", ".join(miss) + ".",
                 list(pres.values())[:2],
                 "Bổ sung " + ", ".join(miss) + (" (tính theo từng nhãn + macro/micro)" if "multilabel" in sets
                                                 else "") + "; ưu tiên Sensitivity khi FN nguy hiểm hơn FP.")
    return F(ctx, "M4.METRICS", PASS, "info", detail, list(pres.values())[:2])


@check("M4.CI", "Khoảng tin cậy 95% (Bootstrap)")
def c_ci(ctx):
    sug = ("Bootstrap 1000–2000 lần trên tập Test (lấy mẫu lại theo bệnh nhân) → báo cáo độ đo dạng "
           "`0,85 (95% CI 0,79–0,90)`.")
    code = ctx.grep_code(CI_RX, limit=2)
    numeric = ctx.grep_text(CI_NUMERIC_RX, "report", limit=1)
    if code or numeric:
        return F(ctx, "M4.CI", PASS, "info", evidence=(code + numeric)[:2])
    mention = ctx.grep_text(CI_RX, "report", limit=1)
    if mention:
        return F(ctx, "M4.CI", WARN, "major", "Báo cáo có nhắc tới khoảng tin cậy nhưng chưa thấy code tính "
                 "hoặc số liệu CI cụ thể.", mention, sug)
    std = ctx.grep_code(STD_RX, limit=1) + ctx.grep_text(STD_RX, "report", limit=1)
    return F(ctx, "M4.CI", WARN, "major",
             "Có mean ± std nhưng chưa có 95% CI." if std else "Chưa có khoảng tin cậy.", std[:1], sug)


@check("M4.CALIBRATION", "Đường cong hiệu chuẩn (Calibration curve / Brier)")
def c_calibration(ctx):
    sets = required_metrics(ctx)[1]
    if set(sets) <= {"segmentation", "ner", "regression"}:
        return F(ctx, "M4.CALIBRATION", NA, "info", "Không áp dụng cho loại bài toán này.")
    sug = "Vẽ `CalibrationDisplay` + Brier score: bác sĩ dùng XÁC SUẤT dự báo, không chỉ nhãn."
    code = ctx.grep_code(CALIBRATION_RX, limit=1)
    if code:
        return F(ctx, "M4.CALIBRATION", PASS, "info", evidence=code)
    text = ctx.grep_text(CALIBRATION_RX, "report", limit=1)
    if text:
        return F(ctx, "M4.CALIBRATION", WARN, "minor", "Chỉ thấy nhắc trong báo cáo, chưa thấy code.", text, sug)
    return F(ctx, "M4.CALIBRATION", WARN, "minor", "Chưa có calibration curve.", suggestion=sug)


@check("M4.XAI", "Khả năng giải thích (Grad-CAM / SHAP) phù hợp loại dữ liệu")
def c_xai(ctx):
    expected = ctx.profile.get("xai_expected") or sorted({k for m in ctx.modalities
                                                          for k in XAI_BY_MODALITY.get(m, [])})
    used = {}
    for key, (rx, label) in XAI_METHODS.items():
        hit = ctx.grep_code(rx, limit=1)
        if hit:
            used[key] = hit[0]
    figs = [f for f in ctx.rel_files if Path(f).suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
            and re.search(r"(?i)shap|cam|saliency|xai|explain|attention|giai_thich", f)]
    nb_figs = sum(nb.n_images for nb in ctx.notebooks)
    text = ctx.find_section(r"xai|khả năng giải thích|giải thích|explainab|interpretab|shap|grad-?cam") or \
        ctx.grep_text(XAI_TEXT_RX, "report", limit=1)
    good = [k for k in used if k in expected]
    labels = ", ".join(XAI_METHODS[k][1] for k in used) or "không có"
    exp_labels = " hoặc ".join(XAI_METHODS[k][1] for k in expected)
    ev = list(used.values())[:2] + [(f, "") for f in figs[:2]] + text[:1]
    severity = "critical" if ctx.profile.get("xai_mandatory") else "major"
    if not used:
        return F(ctx, "M4.XAI", FAIL, severity, f"Không có module XAI (cần {exp_labels}).", ev,
                 f"Tích hợp {exp_labels}; trực quan hóa ≥3 ca True Positive và ≥2 ca FP/FN, đối chiếu giải "
                 f"phẫu/sinh lý.")
    if not good:
        return F(ctx, "M4.XAI", WARN, "major", f"XAI dùng: {labels}; với dữ liệu này nên dùng {exp_labels}.", ev,
                 f"Bổ sung {exp_labels} để giải thích từng ca, không chỉ độ quan trọng toàn cục.")
    if not (figs or nb_figs) or not text:
        return F(ctx, "M4.XAI", WARN, "minor", f"Có {labels} trong code nhưng chưa thấy hình/diễn giải trong "
                 f"báo cáo.", ev, "Đưa hình XAI vào báo cáo kèm đoạn đối chiếu lâm sàng.")
    return F(ctx, "M4.XAI", PASS, "info", f"XAI: {labels}.", ev)


@check("M4.XAI_CASES", "XAI theo ca: ≥3 TP + ≥2 FP/FN, đối chiếu giải phẫu/sinh lý")
def c_xai_cases(ctx):
    hits = ctx.grep_text(r"(?i)true positive|false (positive|negative)|\bTP\b|\bFP\b|\bFN\b|dương tính (thật|giả)|"
                         r"âm tính giả|ca (bệnh|số)\s*\d", "report", limit=3)
    return F(ctx, "M4.XAI_CASES", MANUAL, "info",
             "GV kiểm tra số ca và mức khớp kiến thức lâm sàng." + (" Có nhắc tới TP/FP/FN trong báo cáo."
                                                                     if hits else " Chưa thấy nhắc tới ca TP/FP/FN."),
             hits[:2], "" if hits else "Chọn ≥3 ca TP và ≥2 ca FP/FN, trình bày XAI từng ca và lý giải y học.")


@check("M4.ERROR_ANALYSIS", "Phân tích lỗi sai (confusion matrix + nguyên nhân y học)")
def c_error(ctx):
    code = ctx.grep_code(ERROR_CODE_RX, limit=1)
    text = ctx.find_section(r"phân tích lỗi|error analysis|ca dự đoán sai") or \
        ctx.grep_text(ERROR_TEXT_RX, "report", limit=1)
    if code and text:
        return F(ctx, "M4.ERROR_ANALYSIS", PASS, "info", evidence=code + text[:1])
    if code or text:
        return F(ctx, "M4.ERROR_ANALYSIS", WARN, "major", "Có confusion matrix hoặc nhắc lỗi, nhưng chưa phân "
                 "tích nguyên nhân.", code + text[:1],
                 "Phân tích từng nhóm lỗi: nhiễu tín hiệu/ảnh, ca giáp ranh, nhãn hiếm — kèm ví dụ cụ thể.")
    return F(ctx, "M4.ERROR_ANALYSIS", FAIL, "major", "Không có phân tích lỗi.",
             suggestion="Vẽ confusion matrix trên Test và phân tích nguyên nhân y học của các ca sai.")


# -- FINAL ------------------------------------------------------------------

@check("F.PAPER_STRUCTURE", "Cấu trúc báo cáo đủ các mục bắt buộc")
def c_paper(ctx):
    sections = override(ctx, "report_sections") or DEFAULT_PAPER_SECTIONS
    have, miss, ev = [], [], []
    for label, rx in sections:
        hit = ctx.find_section(rx)
        (have if hit else miss).append(label)
        if hit:
            ev.append(hit[0])
    detail = f"Có {len(have)}/{len(sections)} mục" + (f"; thiếu: {', '.join(miss)}" if miss else "")
    if not miss:
        return F(ctx, "F.PAPER_STRUCTURE", PASS, "info", detail)
    status = WARN if len(have) >= len(sections) / 2 else FAIL
    return F(ctx, "F.PAPER_STRUCTURE", status, "major", detail + ".", ev[:2],
             "Bổ sung các mục còn thiếu theo cấu trúc IMRAD; mỗi mục có tiêu đề rõ ràng.")


@check("F.IEEE_FORMAT", "Định dạng bài báo chuẩn IEEE (Overleaf IEEEtran / Markdown đủ Abstract + Index Terms)")
def c_ieee(ctx):
    tex = [d for d in ctx.report_docs if d.kind == "tex"]
    if any(re.search(r"IEEEtran", (ctx.folder / d.path).read_text(encoding="utf-8", errors="ignore"))
           for d in tex if (ctx.folder / d.path).is_file()):
        return F(ctx, "F.IEEE_FORMAT", PASS, "info", "Dùng template IEEEtran.")
    abstract = ctx.find_section(r"abstract|tóm tắt")
    keywords = ctx.grep_text(r"(?i)index terms|keywords|từ khóa", "report", limit=1)
    if abstract and keywords:
        return F(ctx, "F.IEEE_FORMAT", PASS, "info", "Có Abstract + Index Terms.", abstract[:1] + keywords)
    return F(ctx, "F.IEEE_FORMAT", WARN, "minor", "Chưa thấy template IEEE hoặc Abstract + Index Terms.",
             suggestion="Viết trên Overleaf template IEEEtran (conference) — bảng/hình theo chuẩn xuất bản.")


@check("F.LIMITATIONS", "Nêu trung thực hạn chế của nghiên cứu")
def c_limitations(ctx):
    sec = ctx.find_section(r"hạn chế|limitation")
    if sec:
        return F(ctx, "F.LIMITATIONS", PASS, "info", evidence=sec[:1])
    text = ctx.grep_text(LIMITATION_RX, "report", limit=1)
    if text:
        return F(ctx, "F.LIMITATIONS", PASS, "info", "Nêu hạn chế trong phần Thảo luận.", text)
    return F(ctx, "F.LIMITATIONS", FAIL, "major", "Không có mục Hạn chế.",
             suggestion="Nêu hạn chế: cỡ mẫu, đơn trung tâm, thiếu external validation, nguy cơ rò rỉ/thiên vị.")


@check("F.AI_DISCLOSURE", "Tuyên bố sử dụng AI (mục Disclosure + docs/ai_disclosure_log.md)")
def c_ai_disclosure(ctx):
    sec = ctx.grep_text(AI_DISCLOSURE_RX, "report", limit=1)
    log = ctx.files_matching(r"(^|/)ai_disclosure\w*\.md$")
    ev = sec + [(x, "") for x in log[:1]]
    if sec and log:
        return F(ctx, "F.AI_DISCLOSURE", PASS, "info", evidence=ev)
    miss = ([] if sec else ["mục “AI and AI-Assisted Technologies Disclosure” trước Tài liệu tham khảo"]) + \
           ([] if log else ["file `docs/ai_disclosure_log.md` lưu prompt quan trọng"])
    return F(ctx, "F.AI_DISCLOSURE", WARN if (sec or log) else FAIL, "major", "Thiếu: " + "; ".join(miss) + ".",
             ev, "Khai báo: tên công cụ AI, mục đích sử dụng cụ thể, và tuyên bố nhóm chịu trách nhiệm 100% "
                 "nội dung (Phần VII của Khung quản lý).")


@check("F.CREDIT", "Đóng góp tác giả theo CRediT")
def c_credit(ctx):
    hits = ctx.grep_text(CREDIT_RX, "text", limit=1)
    roles = [r for r in CREDIT_ROLES if ctx.grep_text(re.escape(r), "text", limit=1)]
    if hits or len(roles) >= 3:
        return F(ctx, "F.CREDIT", PASS, "info", evidence=hits[:1])
    return F(ctx, "F.CREDIT", WARN, "minor", "Chưa thấy bảng đóng góp tác giả CRediT.",
             suggestion="Thêm mục CRediT (Conceptualization, Methodology, Software, ...) theo MSSV từng thành viên.")


@check("F.REQUIREMENTS", "requirements.txt / environment.yml cố định phiên bản")
def c_requirements(ctx):
    req = ctx.files_matching(r"(^|/)(requirements[\w.-]*\.txt|environment\.ya?ml|pyproject\.toml|Pipfile(\.lock)?|"
                             r"poetry\.lock|uv\.lock|conda-lock\.ya?ml)$")
    if not req:
        readme_pip = ctx.grep_text(r"pip install", "readme", limit=1)
        return F(ctx, "F.REQUIREMENTS", FAIL, "major",
                 "Không có file môi trường." + (" README chỉ liệt kê `pip install` không cố định phiên bản."
                                                if readme_pip else ""), readme_pip,
                 "Tạo `requirements.txt` bằng `pip freeze` (lọc gói cần thiết), dạng `scikit-learn==1.5.2`.")
    pinned = total = 0
    for rel in req:
        if rel.endswith(".txt") or rel.endswith((".yml", ".yaml")):
            t = read_text(ctx.folder / rel) or ""
            for line in t.splitlines():
                s = line.strip()
                if not s or s.startswith(("#", "-r", "-e", "--", "name:", "channels:", "dependencies:", "prefix:")):
                    continue
                s = s.lstrip("- ").strip()
                if not s or s.endswith(":"):
                    continue
                total += 1
                if re.search(r"==|===|@\s*\w+://|(?<![<>=!~])=\d", s):
                    pinned += 1
        elif rel.endswith(".lock") or "lock" in rel:
            pinned += 1
            total += 1
    ratio = pinned / total if total else 0
    detail = f"{', '.join(f'`{r}`' for r in req)} — {pinned}/{total} gói cố định phiên bản."
    if total and ratio >= 0.8:
        return F(ctx, "F.REQUIREMENTS", PASS, "info", detail)
    return F(ctx, "F.REQUIREMENTS", WARN, "minor", detail,
             suggestion="Cố định phiên bản mọi gói (`==`) để kết quả tái lập được.")


@check("F.README_RUN", "README hướng dẫn tái lập (lý tưởng: 1 lệnh)")
def c_readme_run(ctx):
    rd = ctx.readme
    if not rd:
        return F(ctx, "F.README_RUN", FAIL, "major", "Không có README.")
    if "(Điền các bước" in rd.text:
        return F(ctx, "F.README_RUN", FAIL, "major", "Mục cách chạy lại còn để nội dung mẫu.")
    heading = [h for h in ctx.headings("readme") if re.search(README_RUN_HEADING_RX, h[1])]
    blocks = re.findall(r"```[\w-]*\n(.*?)```", rd.text, re.S)
    cmd_blocks = [b for b in blocks if re.search(r"(?m)^\s*(\$\s*)?(python|pip|conda|bash|sh|make|jupyter|docker|"
                                                 r"dvc|cd)\b", b)]
    entry = re.search(ENTRYPOINT_RX, "\n".join(cmd_blocks))
    data_src = re.search(r"https?://|data/", rd.text)
    if heading and cmd_blocks and entry:
        return F(ctx, "F.README_RUN", PASS, "info", f"Lệnh tái lập: `{entry.group(0).strip()[:60]}`.", heading[:1])
    if cmd_blocks:
        return F(ctx, "F.README_RUN", WARN, "minor",
                 "Có lệnh chạy nhưng chưa gộp thành 1 lệnh tái lập toàn bộ." + ("" if data_src
                                                                                  else " Chưa ghi nguồn dữ liệu."),
                 heading[:1], "Thêm `run_all.sh`/`Makefile`/`python main.py` chạy từ dữ liệu thô → bảng kết quả + "
                              "hình trong báo cáo; ghi rõ link tải dữ liệu.")
    return F(ctx, "F.README_RUN", FAIL, "major", "README chưa có hướng dẫn chạy lại.",
             suggestion="Ghi: cài môi trường, tải dữ liệu (link), lệnh chạy, thời gian/phần cứng dự kiến.")


@check("F.SEED", "Cố định random seed")
def c_seed(ctx):
    if not ctx.code_units:
        return F(ctx, "F.SEED", NA, "info", "Không có mã nguồn.")
    hits = ctx.grep_code(SEED_RX, limit=2)
    dl = ctx.grep_code(DL_FRAMEWORK_RX, limit=1, imports=True)
    if dl and not ctx.grep_code(TORCH_SEED_RX, limit=1):
        return F(ctx, "F.SEED", WARN, "minor", "Dùng deep learning nhưng chưa cố định seed của framework.",
                 hits[:1], "Gọi `torch.manual_seed`/`tf.random.set_seed` (hoặc `lightning.seed_everything`) và "
                           "bật chế độ deterministic khi có thể.")
    if hits:
        return F(ctx, "F.SEED", PASS, "info", evidence=hits[:1])
    return F(ctx, "F.SEED", WARN, "minor", "Không thấy cố định seed.",
             suggestion="Đặt `RANDOM_STATE = 42` dùng chung cho split, mô hình, SMOTE, bootstrap.")


@check("F.ABS_PATH", "Không hard-code đường dẫn tuyệt đối")
def c_abs_path(ctx):
    local = ctx.grep_code(ABS_PATH_LOCAL_RX, limit=3)
    cloud = ctx.grep_code(ABS_PATH_CLOUD_RX, limit=3)
    if local:
        return F(ctx, "F.ABS_PATH", FAIL, "major", "Đường dẫn tuyệt đối của máy cá nhân.", local,
                 "Dùng đường dẫn tương đối từ gốc dự án (`Path(__file__).parent / 'data'`) hoặc tham số dòng lệnh.")
    if cloud:
        return F(ctx, "F.ABS_PATH", WARN, "minor", "Đường dẫn Colab/Kaggle — không chạy được ở máy khác.", cloud,
                 "Đưa đường dẫn dữ liệu vào biến cấu hình/tham số, mặc định là thư mục tương đối `data/`.")
    return F(ctx, "F.ABS_PATH", PASS, "info")


@check("F.NOTEBOOK_EXEC", "Notebook đã chạy tuần tự từ đầu (Restart & Run All), không lỗi")
def c_nb_exec(ctx):
    nbs = [nb for nb in ctx.notebooks if nb.n_code]
    if not nbs:
        return F(ctx, "F.NOTEBOOK_EXEC", NA, "info", "Không có notebook.")
    issues = []
    for nb in nbs:
        if nb.n_executed < nb.n_code:
            issues.append(f"`{nb.path}`: {nb.n_code - nb.n_executed}/{nb.n_code} cell chưa chạy")
        elif nb.exec_counts != list(range(1, len(nb.exec_counts) + 1)):
            issues.append(f"`{nb.path}`: thứ tự chạy không tuần tự ({nb.exec_counts[:6]}…)")
        if nb.n_errors:
            issues.append(f"`{nb.path}`: {nb.n_errors} cell lỗi")
    if not issues:
        return F(ctx, "F.NOTEBOOK_EXEC", PASS, "info", f"{len(nbs)} notebook chạy tuần tự, có output.")
    return F(ctx, "F.NOTEBOOK_EXEC", WARN, "minor", "; ".join(issues[:4]),
             suggestion="Kernel → Restart & Run All trước khi commit để output khớp đúng code nộp.")


@check("F.JUNK", "Repo sạch: không file rác, trọng số/dữ liệu nặng")
def c_junk(ctx):
    junk, heavy = [], []
    for f, rel in zip(ctx.files, ctx.rel_files):
        parts = rel.split("/")
        if set(parts[:-1]) & JUNK_DIRS or parts[-1] in JUNK_FILES:
            junk.append(rel)
        try:
            size = f.stat().st_size
        except OSError:
            continue
        if f.suffix.lower() in WEIGHT_EXT and size > 5 * 1024 * 1024:
            heavy.append(f"{rel} ({size / 1048576:.1f}MB)")
    if heavy:
        return F(ctx, "F.JUNK", WARN, "major", "Trọng số/file nhị phân lớn: " + ", ".join(heavy[:3]),
                 suggestion="Đưa trọng số lên HuggingFace/Drive, README ghi link + checksum.")
    if junk:
        return F(ctx, "F.JUNK", WARN, "minor", "File rác: " + ", ".join(f"`{x}`" for x in junk[:4]),
                 suggestion="Xóa và thêm vào `.gitignore` (`__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`).")
    return F(ctx, "F.JUNK", PASS, "info")


@check("F.SLIDES", "Slide thuyết trình")
def c_slides(ctx):
    files = [f for f in ctx.rel_files if f.startswith("slides/") and Path(f).suffix.lower() in SLIDE_EXT]
    link = ctx.grep_text(r"docs\.google\.com/presentation|canva\.com|slides\.com|gamma\.app", "text", limit=1)
    if files or link:
        return F(ctx, "F.SLIDES", PASS, "info", evidence=[(x, "") for x in files[:1]] + link[:1])
    return F(ctx, "F.SLIDES", FAIL, "minor", "Chưa có slide trong `slides/`.",
             suggestion="Slide 10–12 phút: bài toán lâm sàng → kiến trúc → bảng đối chứng → XAI → hạn chế.")


@check("F.QA", "Vấn đáp phản biện (mọi thành viên trả lời)")
def c_qa(ctx):
    return F(ctx, "F.QA", MANUAL, "info", "Đánh giá tại buổi bảo vệ — xem câu hỏi gợi ý ở cuối báo cáo.")


# -- Liêm chính & chính sách trừ điểm (Phần IV) ------------------------------

def norm_col(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.strip().strip('"\'').lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


@check("I.PHI", "Không có cột định danh bệnh nhân (PHI/PII) trong dữ liệu nộp")
def c_phi(ctx):
    direct, quasi, dicom = [], [], []
    for f, rel in zip(ctx.files, ctx.rel_files):
        ext = f.suffix.lower()
        if ext in (".csv", ".tsv"):
            try:
                with open(f, "rb") as fh:
                    head = fh.read(65536)
            except OSError:
                continue
            first = head.decode("utf-8", errors="ignore").split("\n", 1)[0]
            cols = [norm_col(c) for c in re.split(r"[,;\t]", first)]
            d = sorted({c for c in cols if c in PHI_DIRECT})
            q = sorted({c for c in cols if c in PHI_QUASI})
            if d:
                direct.append((rel, "cột: " + ", ".join(d)))
            if q:
                quasi.append((rel, "cột: " + ", ".join(q)))
        elif ext == ".dcm":
            dicom.append(rel)
    if direct:
        return F(ctx, "I.PHI", FAIL, "critical", "Dữ liệu chứa cột định danh trực tiếp.", direct[:3],
                 "Xóa ngay khỏi PR VÀ lịch sử Git (file đã push vẫn còn trong lịch sử), chỉ dùng dữ liệu công khai "
                 "đã khử định danh; báo giảng viên.", PENALTY_PHI)
    if quasi or dicom:
        ev = quasi[:2] + [(x, "") for x in dicom[:2]]
        return F(ctx, "I.PHI", WARN, "major",
                 "Có định danh gián tiếp (ngày sinh/mã bệnh án) hoặc file DICOM (header có thể chứa PatientName).",
                 ev, "Xác nhận dữ liệu đã khử định danh (DICOM: kiểm tra/xóa tag PatientName, PatientID, "
                     "PatientBirthDate bằng pydicom).", PENALTY_PHI)
    return F(ctx, "I.PHI", PASS, "info")


@check("I.DATA_LICENSE", "Dữ liệu commit đúng giấy phép (không phân phối lại dữ liệu cần credentialed access)")
def c_data_license(ctx):
    mimic = ctx.files_matching(MIMIC_FILE_RX)
    raw = [f for f in ctx.rel_files if Path(f).suffix.lower() in {".dat", ".hea", ".edf", ".dcm", ".nii"}
           or (f.startswith("data/") and Path(f).suffix.lower() in {".png", ".jpg", ".jpeg"})]
    if mimic:
        return F(ctx, "I.DATA_LICENSE", WARN, "critical",
                 "Có file mang tên bảng MIMIC: " + ", ".join(f"`{x}`" for x in mimic[:4]),
                 suggestion="MIMIC-III/IV bản đầy đủ chỉ cấp cho người có credentialed access và DUA của PhysioNet "
                            "KHÔNG cho phép phân phối lại — xóa khỏi PR (và lịch sử Git), chỉ commit script trích "
                            "xuất. Chỉ bản Demo open-access mới được chia sẻ; xác nhận với giảng viên.",
                 penalty=PENALTY_PHI)
    if len(raw) > 50:
        return F(ctx, "I.DATA_LICENSE", WARN, "minor", f"Commit {len(raw)} file dữ liệu thô (ảnh/tín hiệu).",
                 suggestion="Dùng script tải dữ liệu từ nguồn gốc (PhysioNet/Kaggle) kèm checksum thay vì commit.")
    return F(ctx, "I.DATA_LICENSE", PASS, "info")


@check("I.CITATION_INTEGRITY", "Trích dẫn truy vết được (citekey ∈ .bib, DOI có thật)")
def c_citations(ctx):
    text = "\n".join(d.text for d in ctx.docs_for("report"))
    used = set()
    for m in re.finditer(CITEKEY_PANDOC_RX, text):
        used.update(re.findall(r"@([\w:\-]+)", m.group(0)))
    for m in re.finditer(CITEKEY_LATEX_RX, text):
        used.update(k.strip() for k in m.group(1).split(","))
    hmyt_keys = set(re.findall(CITEKEY_HMYT_RX, text))
    bib = ctx.bib_entries
    problems, warns, ev = [], [], []
    if bib:
        orphan = sorted((used | hmyt_keys) - set(bib))
        if orphan:
            problems.append(f"{len(orphan)} citekey không có trong .bib: " + ", ".join(f"`{k}`" for k in orphan[:6]))
        no_doi = [k for k, v in bib.items() if not v["doi"]]
        if no_doi:
            warns.append(f"{len(no_doi)}/{len(bib)} mục .bib thiếu DOI")
    elif used or hmyt_keys:
        warns.append(f"báo cáo dùng {len(used | hmyt_keys)} citekey nhưng không nộp file .bib để đối chiếu")
    if ctx.opts.online:
        dois = {v["doi"]: v["title"] for v in bib.values() if v["doi"]}
        for m in re.finditer(DOI_RX, text):
            dois.setdefault(m.group(0).rstrip("."), "")
        notfound, mismatch, errors = [], [], 0
        for doi, title in list(dois.items())[:40]:
            st, note = verify_doi(doi, title)
            if st == "notfound":
                notfound.append(doi)
            elif st == "mismatch":
                mismatch.append(f"{doi} ({note})")
            elif st == "error":
                errors += 1
        if notfound:
            problems.append(f"{len(notfound)} DOI không tồn tại trên Crossref: " + ", ".join(notfound[:4]))
        if mismatch:
            warns.append(f"{len(mismatch)} DOI có tiêu đề lệch với .bib: " + "; ".join(mismatch[:3]))
        if errors:
            warns.append(f"{errors} DOI không xác minh được (lỗi mạng)")
        if dois and not (notfound or mismatch or errors):
            ev.append((f"{len(dois)} DOI", "đã xác minh qua Crossref"))
    if problems:
        return F(ctx, "I.CITATION_INTEGRITY", FAIL, "major", "; ".join(problems + warns), ev,
                 "Mỗi trích dẫn phải khớp 1 mục .bib có DOI đã xác minh; xóa/sửa trích dẫn không truy vết được.",
                 "Nếu là trích dẫn ma: " + PENALTY_FABRICATION)
    if warns:
        return F(ctx, "I.CITATION_INTEGRITY", WARN, "minor", "; ".join(warns), ev,
                 "Nộp `references.bib` có trường `doi` cho mọi mục; kiểm tra bằng `--online` (Crossref).")
    if not (used or hmyt_keys or bib):
        return F(ctx, "I.CITATION_INTEGRITY", NA, "info", "Không có trích dẫn dạng citekey để đối chiếu.")
    return F(ctx, "I.CITATION_INTEGRITY", PASS, "info",
             "" if ctx.opts.online else "Chưa xác minh DOI (chạy `--online`).", ev)


@check("I.RESULT_TRACE", "Số liệu báo cáo truy vết được tới output thực của code")
def c_result_trace(ctx):
    nums = report_numbers(ctx)
    if not nums:
        return F(ctx, "I.RESULT_TRACE", NA, "info", "Không thấy bảng số liệu trong báo cáo.")
    outputs, sources = output_numbers(ctx)
    if not sources:
        return F(ctx, "I.RESULT_TRACE", WARN, "major",
                 f"Báo cáo có {len(nums)} số liệu trong bảng nhưng bài nộp không có output để đối chiếu "
                 f"(notebook đã chạy, results.json/csv, log).",
                 suggestion="Commit notebook đã chạy (có output) hoặc file kết quả (`results.json`, "
                            "`ablation_results.json`) do code sinh ra — mọi con số trong báo cáo phải truy vết được.",
                 penalty="Nếu số liệu bịa đặt: " + PENALTY_FABRICATION)
    found, missing = 0, []
    for raw, pct in nums:
        d = len(re.split(r"[.,]", raw)[1])
        v = float(raw.replace(",", "."))
        if traced(v, d, outputs):
            found += 1
        else:
            missing.append(raw + pct)
    ratio = found / len(nums)
    detail = f"{found}/{len(nums)} số liệu trong bảng báo cáo tìm thấy trong output ({', '.join(sources[:3])})."
    if ratio >= 0.7:
        return F(ctx, "I.RESULT_TRACE", PASS, "info", detail)
    return F(ctx, "I.RESULT_TRACE", WARN, "major" if ratio < 0.3 else "minor",
             detail + " Chưa khớp (kiểm tra thủ công): " + ", ".join(dict.fromkeys(missing))[:200] + ".",
             suggestion="Chạy lại toàn bộ pipeline, cập nhật báo cáo từ output mới nhất (tránh chép tay).")


@check("I.SIMILARITY", "Không sao chép bài nhóm khác / code minh họa công khai")
def c_similarity(ctx):
    mine_code, mine_text = folder_fingerprint(ctx.folder)
    if len(mine_code) < 15 and len(mine_text) < 50:
        return F(ctx, "I.SIMILARITY", NA, "info", "Quá ít nội dung để so khớp.")
    refs = []
    if SUBMISSIONS_DIR.is_dir():
        for other in sorted(SUBMISSIONS_DIR.iterdir()):
            if other.is_dir() and other.name != TEMPLATE_NAME and other.resolve() != ctx.folder.resolve():
                refs.append((f"submissions/{other.name}", folder_fingerprint(other)))
    for rd in ctx.opts.reference_dir or []:
        p = Path(rd)
        if p.is_dir():
            refs.append((f"{p.name} (tham chiếu)", folder_fingerprint(p)))
    if ctx.opts.online:
        units = []
        for url in ctx.profile.get("reference_code_urls", []):
            try:
                _, body = http_get(url)
                _, unit = parse_notebook(url.rsplit("/", 1)[-1], body.decode("utf-8", errors="ignore"))
                if unit:
                    units.append(unit)
            except (urllib.error.URLError, OSError, ValueError):
                ctx.notes.append(f"Không tải được code tham chiếu {url}.")
        if units:
            refs.append(("code minh họa công khai của đề tài", (code_shingles(units), set())))
    scores = []
    for name, (code_fp, text_fp) in refs:
        c = containment(mine_code, code_fp) if len(mine_code) >= 15 else 0
        t = containment(mine_text, text_fp) if len(mine_text) >= 50 else 0
        if c or t:
            scores.append((max(c, t), name, c, t))
    scores.sort(reverse=True)
    top = [(name, f"code trùng {c:.0%}, báo cáo trùng {t:.0%}") for s, name, c, t in scores[:3] if s >= 0.15]
    if not refs:
        return F(ctx, "I.SIMILARITY", NA, "info", "Không có bài nộp/tài liệu tham chiếu để so khớp.")
    best = scores[0][0] if scores else 0
    if best >= 0.6:
        return F(ctx, "I.SIMILARITY", WARN, "critical", f"Mức trùng lặp cao (≥ 60%) với {scores[0][1]}.", top,
                 "Nếu tái sử dụng code/ví dụ có sẵn, ghi rõ nguồn và phần nhóm tự phát triển; bài nộp phải mở rộng "
                 "khác biệt so với ví dụ minh họa.", PENALTY_FABRICATION)
    if best >= 0.3:
        return F(ctx, "I.SIMILARITY", WARN, "minor", "Trùng lặp đáng chú ý — GV xem xét.", top,
                 "Ghi nguồn cho phần tái sử dụng; nhấn mạnh phần đóng góp riêng của nhóm.")
    return F(ctx, "I.SIMILARITY", PASS, "info", f"So khớp với {len(refs)} nguồn; trùng cao nhất {best:.0%}.", top)


@check("I.CONTRIBUTION", "Bằng chứng đóng góp của từng thành viên (Zero Freerider)")
def c_contribution(ctx):
    hist = git_history(ctx)
    members = ctx.manifest.get("members") or []
    if not hist["available"]:
        return F(ctx, "I.CONTRIBUTION", MANUAL, "info", "Không đọc được lịch sử Git.")
    authors = sorted({c["author"] for c in hist["commits"]})
    detail = f"{len(hist['commits'])} commit, {len(authors)} tác giả commit / {len(members)} thành viên."
    if len(members) >= 2 and len(authors) <= 1:
        return F(ctx, "I.CONTRIBUTION", MANUAL, "info",
                 detail + " Repo nộp bài thường chỉ do 1 người đẩy — đối chiếu lịch sử commit trên repo nhóm "
                          "và phiếu đánh giá đồng đẳng.", penalty=PENALTY_FREERIDER)
    return F(ctx, "I.CONTRIBUTION", MANUAL, "info", detail)


@check("I.DEADLINE", "Nộp đúng hạn")
def c_deadline(ctx):
    if not ctx.opts.deadline:
        return F(ctx, "I.DEADLINE", NA, "info", "Không cung cấp `--deadline`.")
    try:
        tz = dt.timezone(dt.timedelta(hours=7))
        deadline = dt.datetime.fromisoformat(ctx.opts.deadline)
        if deadline.tzinfo is None:
            if len(ctx.opts.deadline) <= 10:
                deadline = deadline.replace(hour=23, minute=59, second=59)
            deadline = deadline.replace(tzinfo=tz)
    except ValueError:
        return F(ctx, "I.DEADLINE", MANUAL, "info", f"`--deadline` không hợp lệ: {ctx.opts.deadline}.")
    hist = git_history(ctx)
    when, src = None, ""
    if hist["commits"]:
        when, src = dt.datetime.fromisoformat(hist["commits"][0]["cdate"]), "commit cuối"
    elif ctx.manifest.get("submitted_at"):
        try:
            when = dt.datetime.fromisoformat(str(ctx.manifest["submitted_at"])).replace(tzinfo=tz)
            src = "`submitted_at`"
        except ValueError:
            pass
    if not when:
        return F(ctx, "I.DEADLINE", MANUAL, "info", "Không xác định được thời điểm nộp.")
    late_h = (when - deadline).total_seconds() / 3600
    if late_h <= 0:
        return F(ctx, "I.DEADLINE", PASS, "info", f"Nộp lúc {when.isoformat()} ({src}).")
    if late_h > 72:
        return F(ctx, "I.DEADLINE", FAIL, "critical", f"Muộn {late_h:.0f} giờ ({src}).", penalty=PENALTY_LATE)
    return F(ctx, "I.DEADLINE", WARN, "major", f"Muộn {late_h:.0f} giờ ({src}) → trừ "
             f"{math.ceil(late_h / 24) * 10}% điểm cột mốc.", penalty=PENALTY_LATE)


INTEGRITY_CHECKS = ["I.PHI", "I.DATA_LICENSE", "I.CITATION_INTEGRITY", "I.RESULT_TRACE", "I.SIMILARITY",
                    "I.CONTRIBUTION", "I.DEADLINE"]


# ---------------------------------------------------------------------------
# Yêu cầu đặc thù đề tài (khai báo trong rubric/topics/<slug>.json)
# ---------------------------------------------------------------------------

def eval_topic_check(ctx: Context, spec: dict) -> Finding:
    flags = 0 if spec.get("case_sensitive") else re.I
    where = spec.get("where", "any")
    patterns = spec.get("any_of", [])

    def search(rx):
        if where == "files":
            return [(f, "") for f in ctx.files_matching(rx, flags)][:2]
        hits = []
        if where in ("code", "any"):
            hits += ctx.grep_code(rx, flags, limit=2)
        if where in ("report", "text", "any"):
            hits += ctx.grep_text(rx, "report" if where == "report" else "text", flags, limit=2)
        return hits

    matched = {rx: search(rx) for rx in patterns}
    hit_patterns = [rx for rx, h in matched.items() if h]
    ev = list(dict.fromkeys(h for rx in hit_patterns for h in matched[rx][:1]))[:3]
    sev = spec.get("severity", "major")
    fail_status = spec.get("fail_status") or (FAIL if sev in ("critical", "major") else WARN)
    base = Finding(spec["id"], spec["title"], PASS, "info", spec.get("pass_detail", ""), [], "",
                   spec.get("penalty", ""))
    if spec.get("type", "require") == "forbid":
        if hit_patterns:
            if any(search(rx) for rx in spec.get("unless_any", [])):
                base.status = spec.get("unless_status", PASS)
                base.severity = "minor" if base.status == WARN else "info"
                base.detail = spec.get("unless_detail", "")
                base.evidence = [f"`{loc}` — {snippet(c)}" if c else f"`{loc}`" for loc, c in ev]
                base.suggestion = spec.get("suggestion", "") if base.status != PASS else ""
                return base
            return Finding(spec["id"], spec["title"], fail_status, sev, spec.get("fail_detail", ""),
                           [f"`{loc}` — {snippet(c)}" if c else f"`{loc}`" for loc, c in ev],
                           spec.get("suggestion", ""), spec.get("penalty", ""))
        return base
    need = int(spec.get("min_distinct", 1))
    if len(hit_patterns) >= need:
        base.evidence = [f"`{loc}` — {snippet(c)}" if c else f"`{loc}`" for loc, c in ev]
        return base
    where_label = {"code": "mã nguồn", "report": "báo cáo", "text": "báo cáo/README/notebook",
                   "any": "mã nguồn và báo cáo", "files": "danh sách file"}.get(where, where)
    return Finding(spec["id"], spec["title"], fail_status, sev,
                   spec.get("fail_detail", f"Chưa thấy đủ bằng chứng trong {where_label} "
                                           f"({len(hit_patterns)}/{need} dấu hiệu)."),
                   [f"`{loc}` — {snippet(c)}" if c else f"`{loc}`" for loc, c in ev],
                   spec.get("suggestion", ""), spec.get("penalty", ""))


# ---------------------------------------------------------------------------
# Điều phối review + dựng báo cáo
# ---------------------------------------------------------------------------

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(rubric_dir: Path = RUBRIC_DIR):
    rubric = load_json(rubric_dir / "rubric.json")
    improvements = load_json(rubric_dir / "improvements.json")
    catalog = {}
    for p in sorted((rubric_dir / "topics").glob("*.json")):
        prof = load_json(p)
        catalog[prof["topic_slug"]] = prof
    return rubric, improvements, catalog


def milestone_reached(ctx: Context, ms: str) -> bool:
    if not ctx.rubric_def.get("gated", True):
        return True
    return MILESTONE_ORDER.index(ms) <= MILESTONE_ORDER.index(ctx.milestone)


def criterion_status(findings):
    sts = [f.status for f in findings]
    if FAIL in sts:
        return "🔴", "Nguy cơ < 70%"
    if WARN in sts:
        return "🟡", "Dấu hiệu mức Đạt (70–89%)"
    if PASS in sts:
        return "🟢", "Dấu hiệu mức Xuất sắc (≥ 90%)"
    return "👤", "GV đánh giá"


def run_check(ctx: Context, cid: str) -> Finding:
    """Lỗi bất ngờ của 1 check không được làm hỏng cả báo cáo (CI tư vấn)."""
    try:
        return CHECKS[cid][1](ctx)
    except Exception as e:  # noqa: BLE001
        return Finding(cid, CHECKS[cid][0], MANUAL, "info", f"Lỗi công cụ khi kiểm tra ({type(e).__name__}: "
                       f"{str(e)[:120]}) — cần review thủ công.")


def review_folder(folder: Path, opts, config) -> dict:
    rubric_cfg, improvements, catalog = config
    ctx = build_context(folder, opts, catalog)
    if ctx.rubric_name not in rubric_cfg["rubrics"]:
        ctx.notes.append(f"Không có rubric `{ctx.rubric_name}` — dùng `standard`.")
        ctx.rubric_name = "standard"
    ctx.rubric_def = rubric_cfg["rubrics"][ctx.rubric_name]

    sections, findings = [], {}
    crit_ids = {}
    for ms in ctx.rubric_def["milestones"]:
        if not milestone_reached(ctx, ms["id"]):
            continue
        sec = {"id": ms["id"], "title": ms["title"], "weight": ms.get("weight", 0), "criteria": []}
        for cr in ms["criteria"]:
            items = []
            for cid in cr["checks"]:
                if cid not in findings:
                    findings[cid] = run_check(ctx, cid)
                items.append(findings[cid])
            crit = {"id": cr["id"], "title": cr["title"], "points": cr.get("points", 0),
                    "excellent": cr.get("excellent", ""), "findings": items}
            crit_ids[cr["id"]] = crit
            sec["criteria"].append(crit)
        sections.append(sec)

    topic_extra = []
    for spec in ctx.profile.get("checks", []):
        if not milestone_reached(ctx, spec.get("milestone", "M0")):
            continue
        try:
            f = eval_topic_check(ctx, spec)
        except (re.error, KeyError, TypeError) as e:
            f = Finding(spec.get("id", "?"), spec.get("title", "?"), MANUAL, "info",
                        f"Lỗi cấu hình check đề tài ({e}) — chạy `--self-test`.")
        findings[spec["id"]] = f
        crit = spec.get("criterion")
        if isinstance(crit, dict):
            crit = crit.get(ctx.rubric_name)
        if crit in crit_ids:
            crit_ids[crit]["findings"].append(f)
        else:
            topic_extra.append(f)

    integrity = []
    for cid in INTEGRITY_CHECKS:
        if cid not in findings:
            findings[cid] = run_check(ctx, cid)
        integrity.append(findings[cid])

    all_f = list(findings.values())
    counts = {s: sum(1 for f in all_f if f.status == s) for s in (PASS, WARN, FAIL, MANUAL, NA)}
    critical = [f for f in all_f if f.status in (FAIL, WARN) and f.severity == "critical"]

    code_blob = "\n".join("\n".join(u.lines) for u in ctx.code_units)
    text_blob = "\n".join(d.text for d in ctx.text_docs)
    beyond = []
    for item in improvements["items"]:
        mods = item.get("modalities", ["*"])
        if "*" not in mods and not set(mods) & set(ctx.modalities):
            continue
        if item.get("topics") and ctx.profile.get("topic_slug") not in item["topics"]:
            continue
        if item.get("detect") and re.search(item["detect"], code_blob + "\n" + text_blob, re.I):
            continue
        beyond.append(item)
    viva = []
    for q in improvements.get("viva_questions", []):
        blob = code_blob + "\n" + text_blob if q.get("where") == "any" else code_blob
        if re.search(q["detect"], blob, re.I):
            viva.append(q["question"])
    viva = (ctx.profile.get("viva_questions", []) + viva)[: opts.max_viva]

    return {
        "folder": folder.as_posix() if not folder.is_absolute() else os.path.relpath(folder, ROOT),
        "topic": {"id": ctx.profile.get("topic_id") or ctx.manifest.get("topic_id"),
                  "slug": ctx.manifest.get("topic_slug"), "title": ctx.profile.get("title", ""),
                  "page": ctx.profile.get("page", ""), "modalities": ctx.modalities,
                  "track": ctx.profile.get("track", "standard"), "benchmarks": ctx.profile.get("benchmarks", [])},
        "group_code": ctx.manifest.get("group_code"),
        "n_members": len(ctx.manifest.get("members") or []),
        "rubric": {"name": ctx.rubric_name, "title": ctx.rubric_def["title"],
                   "source": ctx.rubric_def.get("source", RUBRIC_URL)},
        "milestone": ctx.milestone if ctx.rubric_def.get("gated", True) else None,
        "scanned": {"code_files": sum(1 for u in ctx.code_units if u.kind == "py"),
                    "notebooks": len(ctx.notebooks), "report_docs": len(ctx.report_docs),
                    "bib_entries": len(ctx.bib_entries), "files": len(ctx.files)},
        "counts": counts,
        "critical": critical,
        "sections": sections,
        "topic_checks": topic_extra,
        "integrity": integrity,
        "beyond": beyond[: opts.max_beyond],
        "topic_improvements": ctx.profile.get("improvements", []),
        "viva": viva,
        "notes": ctx.notes,
        "online": bool(opts.online),
    }


def fmt_finding(f: Finding, show_suggestion=True) -> list[str]:
    head = f"- {ICON[f.status]} **{f.title}** `{f.check_id}`"
    if f.detail:
        head += f" — {f.detail}"
    out = [head]
    for e in f.evidence[:4]:
        out.append(f"  - {e}")
    if len(f.evidence) > 4:
        out.append(f"  - … và {len(f.evidence) - 4} vị trí khác")
    if show_suggestion and f.suggestion and f.status in (FAIL, WARN, MANUAL, NA):
        out.append(f"  - 💡 {f.suggestion}")
    if f.penalty and f.status in (FAIL, WARN):
        out.append(f"  - ⚖️ {f.penalty}")
    return out


def render_markdown(r: dict) -> str:
    L = []
    t = r["topic"]
    L.append(f"# 📋 Báo cáo kiểm tra theo rubric — `{r['folder']}`")
    L.append("")
    L.append("> **Kết quả kiểm tra tự động (phân tích tĩnh), không phải điểm.** Công cụ chỉ phát hiện *có/không "
             "có dấu hiệu* đáp ứng từng tiêu chí của "
             f"[Khung Quản Lý & Rubric]({RUBRIC_URL}); chất lượng và điểm chính thức do giảng viên đánh giá. "
             "Mục ❌/⚠️ có thể là dương tính giả — sinh viên giải thích trong PR nếu không đồng ý.")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    title = f"{t['id'] or '?'} — {t['title']}" if t["title"] else str(t["id"])
    if t["page"]:
        title = f"[{title}]({t['page']})"
    L.append(f"| Đề tài | {title} (`{t['slug']}`) · dữ liệu: {', '.join(t['modalities'])} · "
             f"{'cấp tốc' if t['track'] == 'accelerated' else 'tiêu chuẩn 15 tuần'} |")
    L.append(f"| Nhóm | `{r['group_code']}` · {r['n_members']} thành viên |")
    ms = f" · mốc đánh giá: **{r['milestone']}** (tích lũy từ M1)" if r["milestone"] else ""
    L.append(f"| Rubric | [{r['rubric']['title']}]({r['rubric']['source']}){ms} |")
    s = r["scanned"]
    L.append(f"| Đã quét | {s['files']} file: {s['code_files']} .py, {s['notebooks']} notebook, "
             f"{s['report_docs']} tài liệu báo cáo, {s['bib_entries']} mục .bib |")
    L.append(f"| Công cụ | `review_project.py` v{VERSION} · {dt.date.today().isoformat()}"
             f"{' · có kiểm tra trực tuyến' if r['online'] else ' · offline (thêm `--online` để xác minh DOI)'} |")
    if t["benchmarks"]:
        L.append("| Mốc y văn cần đối chiếu | " + "; ".join(
            f"`{b['citekey']}` — {b['note']}" for b in t["benchmarks"]) + " |")
    L.append("")

    c = r["counts"]
    L.append("## 1. Tóm tắt")
    L.append("")
    L.append(f"✅ {c[PASS]} đạt · ⚠️ {c[WARN]} cần cải thiện · ❌ {c[FAIL]} chưa đạt · 👤 {c[MANUAL]} cần GV "
             f"đánh giá · ➖ {c[NA]} không áp dụng")
    L.append("")
    L.append("| Mốc | Tiêu chí rubric | Điểm | Dự báo (tự động) | ✅ | ⚠️ | ❌ | 👤 |")
    L.append("|---|---|:---:|---|:---:|:---:|:---:|:---:|")
    for sec in r["sections"]:
        for cr in sec["criteria"]:
            icon, label = criterion_status(cr["findings"])
            n = {st: sum(1 for f in cr["findings"] if f.status == st) for st in (PASS, WARN, FAIL, MANUAL)}
            L.append(f"| {sec['id']} | {cr['title']} | {cr['points'] or '—'} | {icon} {label} | {n[PASS]} | "
                     f"{n[WARN]} | {n[FAIL]} | {n[MANUAL]} |")
    L.append("")

    if r["critical"]:
        L.append("## 2. 🚨 Vấn đề nghiêm trọng (liên quan chính sách trừ điểm — Phần IV)")
        L.append("")
        for f in r["critical"]:
            L.extend(fmt_finding(f))
        L.append("")
    else:
        L.append("## 2. 🚨 Vấn đề nghiêm trọng")
        L.append("")
        L.append("Không phát hiện vấn đề nghiêm trọng (rò rỉ dữ liệu rõ ràng, dữ liệu định danh, sao chép cao).")
        L.append("")

    L.append("## 3. Chi tiết theo rubric")
    L.append("")
    for sec in r["sections"]:
        weight = f" — {sec['weight']}%" if sec["weight"] else ""
        L.append(f"### {sec['title']}{weight}")
        L.append("")
        for cr in sec["criteria"]:
            icon, label = criterion_status(cr["findings"])
            pts = f" ({cr['points']} đ)" if cr["points"] else ""
            L.append(f"#### {icon} {cr['title']}{pts}")
            if cr["excellent"]:
                L.append(f"> Mức Xuất sắc: {cr['excellent']}")
            L.append("")
            for f in cr["findings"]:
                L.extend(fmt_finding(f))
            L.append("")
    if r["topic_checks"]:
        L.append("### Yêu cầu đặc thù của đề tài")
        L.append("")
        for f in r["topic_checks"]:
            L.extend(fmt_finding(f))
        L.append("")

    L.append("## 4. Liêm chính học thuật & dữ liệu (Phần IV)")
    L.append("")
    for f in r["integrity"]:
        L.extend(fmt_finding(f))
    L.append("")

    L.append("## 5. 💡 Phương án nâng cao chất lượng")
    L.append("")
    all_findings = [f for sec in r["sections"] for cr in sec["criteria"] for f in cr["findings"]] + \
        r["topic_checks"] + r["integrity"]
    seen, uniq = set(), []
    for f in all_findings:
        if f.check_id not in seen:
            seen.add(f.check_id)
            uniq.append(f)
    fails = sorted([f for f in uniq if f.status == FAIL and f.suggestion], key=lambda f: SEV_ORDER[f.severity])
    warns = sorted([f for f in uniq if f.status == WARN and f.suggestion], key=lambda f: SEV_ORDER[f.severity])
    L.append("### Tầng 1 — Sửa ngay (đang ở mức “Cần cải thiện” hoặc dính chính sách trừ điểm)")
    L.append("")
    L.extend([f"{i}. **{f.title}** ({SEV_LABEL[f.severity]}) — {f.suggestion}" for i, f in enumerate(fails, 1)]
             or ["Không có."])
    L.append("")
    L.append("### Tầng 2 — Nâng lên mức “Xuất sắc” của rubric")
    L.append("")
    L.extend([f"{i}. **{f.title}** — {f.suggestion}" for i, f in enumerate(warns, 1)] or ["Không có."])
    L.append("")
    L.append("### Tầng 3 — Vượt rubric, hướng tới bài báo khoa học (chọn 2–3 hướng phù hợp)")
    L.append("")
    if r["beyond"]:
        for i, it in enumerate(r["beyond"], 1):
            ref = f" *(Tham khảo: {it['ref']})*" if it.get("ref") else ""
            L.append(f"{i}. **{it['title']}** — {it['how']}{ref}")
    else:
        L.append("Nhóm đã có dấu hiệu áp dụng các hướng nâng cao phổ biến.")
    if r["topic_improvements"]:
        L.append("")
        L.append("**Gợi ý riêng của đề tài** (từ trang đề tài trong ngân hàng):")
        L.append("")
        L.extend(f"- {x}" for x in r["topic_improvements"])
    L.append("")

    manual = [f for f in uniq if f.status == MANUAL]
    L.append("## 6. Dành cho giảng viên/TA")
    L.append("")
    if manual:
        L.append("**Cần đánh giá thủ công:**")
        L.append("")
        L.extend(f"- [ ] {f.title} — {f.detail}" for f in manual)
        L.append("")
    if r["viva"]:
        L.append("**Câu hỏi vấn đáp gợi ý** (dựa trên kỹ thuật nhóm đã dùng):")
        L.append("")
        L.extend(f"{i}. {q}" for i, q in enumerate(r["viva"], 1))
        L.append("")
    if r["notes"]:
        L.append("**Ghi chú của công cụ:**")
        L.append("")
        L.extend(f"- {n}" for n in dict.fromkeys(r["notes"]))
        L.append("")
    return "\n".join(L)


def to_json(r: dict) -> dict:
    def conv(x):
        if isinstance(x, Finding):
            return asdict(x)
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, list):
            return [conv(v) for v in x]
        return x
    return conv(r)


def render_class_summary(results: list[dict]) -> str:
    L = ["# 📊 Tổng hợp kiểm tra bài nộp cả lớp", "",
         f"`review_project.py` v{VERSION} · {dt.date.today().isoformat()} · {len(results)} bài nộp · "
         "kết quả tự động, không phải điểm.", "",
         "| Bài nộp | Đề tài | Rubric · mốc | 🚨 | ❌ | ⚠️ | ✅ | 👤 | Ưu tiên xử lý |",
         "|---|---|---|:---:|:---:|:---:|:---:|:---:|---|"]
    for r in sorted(results, key=lambda r: (-len(r["critical"]), -r["counts"][FAIL], r["folder"])):
        c = r["counts"]
        first = (r["critical"] or [f for sec in r["sections"] for cr in sec["criteria"] for f in cr["findings"]
                                   if f.status == FAIL] or [None])[0]
        prio = f"{first.check_id}: {first.title}" if first else "—"
        L.append(f"| `{Path(r['folder']).name}` | {r['topic']['id']} | {r['rubric']['name']} · "
                 f"{r['milestone'] or '—'} | {len(r['critical'])} | {c[FAIL]} | {c[WARN]} | {c[PASS]} | "
                 f"{c[MANUAL]} | {prio} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Tự kiểm tra cấu hình rubric/ (chạy khi giảng viên sửa JSON)
# ---------------------------------------------------------------------------

def self_test(rubric_dir: Path = RUBRIC_DIR) -> list[str]:
    errors = []
    try:
        rubric, improvements, catalog = load_config(rubric_dir)
    except (OSError, json.JSONDecodeError, KeyError) as e:
        return [f"Không đọc được cấu hình: {e}"]
    for name, rb in rubric["rubrics"].items():
        crit_ids = set()
        total = 0
        for ms in rb["milestones"]:
            if rb.get("gated", True) and ms["id"] not in MILESTONE_ORDER:
                errors.append(f"rubric `{name}`: mốc `{ms['id']}` không thuộc {MILESTONE_ORDER}")
            for cr in ms["criteria"]:
                crit_ids.add(cr["id"])
                total += cr.get("points", 0)
                for cid in cr["checks"]:
                    if cid not in CHECKS:
                        errors.append(f"rubric `{name}` / `{cr['id']}`: check `{cid}` không tồn tại")
        if rb.get("total_points") and total != rb["total_points"]:
            errors.append(f"rubric `{name}`: tổng điểm {total} ≠ {rb['total_points']}")
        rb["_crit_ids"] = crit_ids
    ids = {}
    for slug, prof in catalog.items():
        for key in ("topic_id", "topic_slug", "title", "modalities", "metric_set"):
            if key not in prof:
                errors.append(f"topics/{slug}.json: thiếu `{key}`")
        if prof.get("topic_id") in ids:
            errors.append(f"topics/{slug}.json: trùng topic_id với {ids[prof['topic_id']]}")
        ids[prof.get("topic_id")] = slug
        rb_name = prof.get("rubric", "standard")
        if rb_name not in rubric["rubrics"]:
            errors.append(f"topics/{slug}.json: rubric `{rb_name}` không tồn tại")
        for m in prof.get("modalities", []):
            if m not in PREPROCESS_BY_MODALITY:
                errors.append(f"topics/{slug}.json: modality `{m}` không hợp lệ")
        sets = prof.get("metric_set")
        for s in sets if isinstance(sets, list) else [sets]:
            if s not in METRIC_SETS:
                errors.append(f"topics/{slug}.json: metric_set `{s}` không hợp lệ")
        for k in prof.get("xai_expected", []):
            if k not in XAI_METHODS:
                errors.append(f"topics/{slug}.json: xai_expected `{k}` không hợp lệ")
        seen = set()
        for spec in prof.get("checks", []):
            sid = spec.get("id", "?")
            if sid in seen or sid in CHECKS:
                errors.append(f"topics/{slug}.json: id check `{sid}` bị trùng")
            seen.add(sid)
            for key in ("id", "title", "milestone", "any_of"):
                if key not in spec:
                    errors.append(f"topics/{slug}.json: check `{sid}` thiếu `{key}`")
            if spec.get("milestone") not in MILESTONE_ORDER:
                errors.append(f"topics/{slug}.json: check `{sid}` có milestone không hợp lệ")
            if spec.get("severity", "major") not in SEV_ORDER:
                errors.append(f"topics/{slug}.json: check `{sid}` có severity không hợp lệ")
            crit = spec.get("criterion")
            crits = crit.values() if isinstance(crit, dict) else [crit] if crit else []
            all_crit = set().union(*(rb["_crit_ids"] for rb in rubric["rubrics"].values()))
            for c in crits:
                if c not in all_crit:
                    errors.append(f"topics/{slug}.json: check `{sid}` trỏ tới tiêu chí `{c}` không tồn tại")
            for rx in spec.get("any_of", []) + spec.get("unless_any", []):
                try:
                    re.compile(rx)
                except re.error as e:
                    errors.append(f"topics/{slug}.json: regex lỗi trong `{sid}`: {e}")
        for ef in prof.get("expected_files", []):
            try:
                re.compile(ef["pattern"])
            except (re.error, KeyError) as e:
                errors.append(f"topics/{slug}.json: expected_files lỗi: {e}")
    for it in improvements.get("items", []) + improvements.get("viva_questions", []):
        try:
            re.compile(it.get("detect", ""))
        except re.error as e:
            errors.append(f"improvements.json: regex lỗi ({it.get('id', it.get('question', '?'))[:40]}): {e}")
    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def submission_folders_from_pr(changed: list[str]) -> list[Path]:
    names = []
    for raw in changed:
        parts = Path(raw.strip().replace("\\", "/")).parts
        if len(parts) >= 2 and parts[0] == "submissions" and parts[1] != TEMPLATE_NAME and parts[1] not in names:
            names.append(parts[1])
    return [SUBMISSIONS_DIR / n for n in names if (SUBMISSIONS_DIR / n).is_dir()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Kiểm tra bài nộp đồ án ET4248 theo rubric + gợi ý nâng cao.")
    ap.add_argument("folders", nargs="*", help="Thư mục nộp bài (submissions/<topic_slug>_<ma_nhom>)")
    ap.add_argument("--all", action="store_true", help="Review mọi thư mục trong submissions/ (trừ _TEMPLATE)")
    ap.add_argument("--pr-files", help="File liệt kê đường dẫn thay đổi trong PR (1 dòng/file)")
    ap.add_argument("--pr-body", help="File chứa mô tả PR (để kiểm tra checklist)")
    ap.add_argument("--milestone", choices=["m1", "m2", "m3", "m4", "final"],
                    help="Mốc đánh giá (mặc định: trường `milestone` trong submission.json, hoặc final)")
    ap.add_argument("--rubric", help="Ép dùng rubric khác (vd standard, lab_3_4)")
    ap.add_argument("--online", action="store_true",
                    help="Bật kiểm tra mạng: DOI (Crossref), nhánh gd1–gd7 của repo nhóm, code minh họa công khai")
    ap.add_argument("--reference-dir", action="append",
                    help="Thư mục code tham chiếu để so khớp trùng lặp (có thể lặp lại)")
    ap.add_argument("--deadline", help="Hạn nộp YYYY-MM-DD hoặc ISO 8601 (mặc định múi giờ +07:00)")
    ap.add_argument("--output", help="Ghi báo cáo Markdown ra file (mặc định in ra màn hình)")
    ap.add_argument("--json", help="Ghi kết quả JSON ra file")
    ap.add_argument("--output-dir", help="Với --all: ghi báo cáo từng nhóm vào thư mục này")
    ap.add_argument("--fail-on", choices=["never", "critical", "fail"], default="never",
                    help="Mã thoát ≠ 0 khi có vấn đề nghiêm trọng (critical) hoặc bất kỳ ❌ (fail)")
    ap.add_argument("--max-beyond", type=int, default=6, help="Số gợi ý Tầng 3 tối đa")
    ap.add_argument("--max-viva", type=int, default=6, help="Số câu hỏi vấn đáp gợi ý tối đa")
    ap.add_argument("--self-test", action="store_true", help="Kiểm tra tính hợp lệ của cấu hình rubric/")
    opts = ap.parse_args(argv)

    if opts.self_test:
        errors = self_test()
        if errors:
            print("[FAILED] Cấu hình rubric/ có lỗi:")
            for e in errors:
                print(f"  ❌ {e}")
            return 1
        _, _, catalog = load_config()
        print(f"[PASSED] Cấu hình hợp lệ: {len(CHECKS)} check chung, {len(catalog)} hồ sơ đề tài.")
        return 0

    opts.pr_body_text = None
    if opts.pr_body:
        try:
            opts.pr_body_text = Path(opts.pr_body).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            opts.pr_body_text = ""

    if opts.pr_files:
        folders = submission_folders_from_pr(Path(opts.pr_files).read_text(encoding="utf-8").splitlines())
        if not folders:
            msg = "Không có thư mục nộp bài nào thay đổi trong PR — bỏ qua review theo rubric.\n"
            print(msg)
            if opts.output:
                Path(opts.output).write_text(msg, encoding="utf-8")
            return 0
    elif opts.all:
        folders = [p for p in sorted(SUBMISSIONS_DIR.iterdir()) if p.is_dir() and p.name != TEMPLATE_NAME]
    else:
        folders = [Path(f) for f in opts.folders]
    if not folders:
        ap.print_help()
        return 2
    for f in folders:
        if not f.is_dir():
            print(f"Không tìm thấy thư mục: {f}", file=sys.stderr)
            return 2
        if f.name == TEMPLATE_NAME:
            print(f"'{TEMPLATE_NAME}' là thư mục mẫu — hãy copy sang thư mục riêng của nhóm.", file=sys.stderr)
            return 2

    config = load_config()
    results = [review_folder(f, opts, config) for f in folders]

    if opts.all or len(results) > 1:
        md = render_class_summary(results)
        if opts.output_dir:
            out_dir = Path(opts.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for r in results:
                name = Path(r["folder"]).name
                (out_dir / f"{name}.md").write_text(render_markdown(r), encoding="utf-8")
                (out_dir / f"{name}.json").write_text(json.dumps(to_json(r), ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
            md += f"\nBáo cáo chi tiết từng nhóm: `{out_dir.as_posix()}/<thư-mục>.md`\n"
        else:
            md += "\n" + "\n\n---\n\n".join(render_markdown(r) for r in results)
    else:
        md = render_markdown(results[0])

    if opts.output:
        Path(opts.output).write_text(md, encoding="utf-8")
        print(f"Đã ghi báo cáo: {opts.output}")
    else:
        print(md)
    if opts.json:
        payload = to_json(results[0]) if len(results) == 1 else [to_json(r) for r in results]
        Path(opts.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if opts.fail_on == "critical" and any(r["critical"] for r in results):
        return 1
    if opts.fail_on == "fail" and any(r["counts"][FAIL] for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
