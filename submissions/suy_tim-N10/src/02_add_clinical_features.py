# 02_add_clinical_features.py
# MIMIC-IV v3.1 - Add OMR, comorbidities, and HF medication groups
# Input : D:\MIMIC_IV_3_1\processed\hf_features_v1.csv.gz
# Output: D:\MIMIC_IV_3_1\processed\hf_model_dataset_final.csv.gz
#
# IMPORTANT:
# - This script DOES NOT globally impute/scale/one-hot encode the dataset.
# - Data-dependent preprocessing must be fitted on TRAIN only later.
# - Comorbidity/medication rules below are pragmatic project phenotypes,
#   not validated clinical phenotyping algorithms.

from pathlib import Path
import gc
import re
import numpy as np
import pandas as pd

# ============================================================
# 0. CONFIG
# ============================================================
ROOT = Path(r"D:\MIMIC_IV_3_1")
DATA_DIR = ROOT / "hosp"
OUT_DIR = ROOT / "processed"

INPUT_FEATURES = OUT_DIR / "hf_features_v1.csv.gz"

DX_FILE = DATA_DIR / "diagnoses_icd.csv.gz"
PRESC_FILE = DATA_DIR / "prescriptions.csv.gz"
OMR_FILE = DATA_DIR / "omr.csv.gz"

OUT_FINAL = OUT_DIR / "hf_model_dataset_final.csv.gz"
OUT_MISSING = OUT_DIR / "hf_model_missingness_final.csv"
OUT_SUMMARY = OUT_DIR / "hf_model_summary_final.txt"
OUT_OMR_COVERAGE = OUT_DIR / "hf_omr_feature_coverage.csv"

CHUNK_PRESC = 500_000
CHUNK_OMR = 500_000

for p in [INPUT_FEATURES, DX_FILE, PRESC_FILE, OMR_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"Khong tim thay: {p}")

print("=== STEP 2: Add clinical features ===")
print("Input :", INPUT_FEATURES)
print("Output:", OUT_FINAL)

# ============================================================
# 1. LOAD V1 FEATURE TABLE
# ============================================================
print("\n[1/6] Doc hf_features_v1...")

df = pd.read_csv(
    INPUT_FEATURES,
    parse_dates=["admittime", "dischtime"]
)

assert df["hadm_id"].is_unique, "hadm_id bi trung trong hf_features_v1!"
assert not df[["subject_id", "hadm_id"]].duplicated().any(), "subject_id/hadm_id bi trung!"

df["subject_id"] = df["subject_id"].astype("int64")
df["hadm_id"] = df["hadm_id"].astype("int64")

cohort_hadm_ids = set(df["hadm_id"].tolist())
cohort_subject_ids = set(df["subject_id"].tolist())

# Structural missing indicator
df["has_prior_admission"] = (df["prior_admissions"] > 0).astype("int8")

print("Patients  :", df["subject_id"].nunique())
print("Admissions:", len(df))
print("Positive  :", int(df["readmit_30d"].sum()))
print("Rate      :", f'{100*df["readmit_30d"].mean():.2f}%')

# ============================================================
# 2. COMORBIDITY FLAGS FROM ICD
# ============================================================
print("\n[2/6] Tao comorbidity flags tu diagnoses_icd...")

dx = pd.read_csv(
    DX_FILE,
    usecols=["hadm_id", "icd_code", "icd_version"]
)
dx = dx.dropna(subset=["hadm_id", "icd_code", "icd_version"]).copy()
dx["hadm_id"] = dx["hadm_id"].astype("int64")
dx = dx[dx["hadm_id"].isin(cohort_hadm_ids)].copy()

dx["icd_code_norm"] = (
    dx["icd_code"]
    .astype(str)
    .str.upper()
    .str.replace(".", "", regex=False)
    .str.strip()
)

# Helpers
def starts_any(s, prefixes):
    out = pd.Series(False, index=s.index)
    for p in prefixes:
        out |= s.str.startswith(p)
    return out

is9 = dx["icd_version"].eq(9)
is10 = dx["icd_version"].eq(10)
code = dx["icd_code_norm"]

# Pragmatic ICD phenotype definitions
dx["diabetes"] = (
    (is9 & code.str.startswith("250"))
    | (is10 & starts_any(code, ["E08", "E09", "E10", "E11", "E12", "E13"]))
)

dx["ckd"] = (
    (is9 & code.str.startswith("585"))
    | (is10 & code.str.startswith("N18"))
)

dx["copd"] = (
    (is9 & starts_any(code, ["491", "492", "496"]))
    | (is10 & code.str.startswith("J44"))
)

dx["atrial_fibrillation"] = (
    (is9 & starts_any(code, ["42731", "42732"]))
    | (is10 & code.str.startswith("I48"))
)

# CAD / ischemic heart disease
dx["cad"] = (
    (is9 & code.str.match(r"^(410|411|412|413|414)"))
    | (is10 & code.str.match(r"^I2[0-5]"))
)

dx["anemia"] = (
    (is9 & code.str.match(r"^(280|281|282|283|284|285)"))
    | (is10 & code.str.match(r"^D(5[0-9]|6[0-4])"))
)

comorb_cols = [
    "diabetes", "ckd", "copd",
    "atrial_fibrillation", "cad", "anemia"
]

comorb = (
    dx.groupby("hadm_id")[comorb_cols]
    .max()
    .astype("int8")
    .reset_index()
)

df = df.merge(comorb, on="hadm_id", how="left")

for c in comorb_cols:
    df[c] = df[c].fillna(0).astype("int8")

print("Comorbidity prevalence:")
for c in comorb_cols:
    print(f"  {c:20s}: {100*df[c].mean():6.2f}%")

del dx, comorb
gc.collect()

# ============================================================
# 3. MEDICATION GROUPS FROM PRESCRIPTIONS
# ============================================================
print("\n[3/6] Tao HF medication group flags tu prescriptions...")

# Regex intentionally conservative enough for common generic/brand strings
MED_PATTERNS = {
    "beta_blocker": re.compile(
        r"\b(metoprolol|carvedilol|bisoprolol|atenolol|propranolol)\b",
        flags=re.I
    ),
    "acei_arb_arni": re.compile(
        r"\b(lisinopril|enalapril|captopril|ramipril|benazepril|"
        r"losartan|valsartan|candesartan|irbesartan|telmisartan|olmesartan|"
        r"sacubitril|entresto)\b",
        flags=re.I
    ),
    "mra": re.compile(
        r"\b(spironolactone|eplerenone)\b",
        flags=re.I
    ),
    "loop_diuretic": re.compile(
        r"\b(furosemide|lasix|bumetanide|bumex|torsemide|demadex)\b",
        flags=re.I
    ),
    "sglt2_inhibitor": re.compile(
        r"\b(dapagliflozin|farxiga|empagliflozin|jardiance|"
        r"canagliflozin|invokana|ertugliflozin|steglatro)\b",
        flags=re.I
    ),
}

# Maintain one set of hadm_ids per med group, memory-light
med_hadm = {k: set() for k in MED_PATTERNS}

for i, chunk in enumerate(pd.read_csv(
    PRESC_FILE,
    usecols=["hadm_id", "drug"],
    chunksize=CHUNK_PRESC,
    low_memory=False
)):
    chunk = chunk.dropna(subset=["hadm_id", "drug"]).copy()
    if chunk.empty:
        continue

    chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
    chunk = chunk[chunk["hadm_id"].isin(cohort_hadm_ids)].copy()
    if chunk.empty:
        continue

    drugs = chunk["drug"].astype(str)

    for name, pattern in MED_PATTERNS.items():
        mask = drugs.str.contains(pattern, na=False)
        if mask.any():
            med_hadm[name].update(chunk.loc[mask, "hadm_id"].tolist())

    if (i + 1) % 10 == 0:
        print(f"  prescriptions chunks: {i+1}")

for name in MED_PATTERNS:
    df[name] = df["hadm_id"].isin(med_hadm[name]).astype("int8")
    print(f"  {name:20s}: {100*df[name].mean():6.2f}%")

del med_hadm
gc.collect()

# ============================================================
# 4. OMR: BMI, WEIGHT, BLOOD PRESSURE, eGFR
# ============================================================
print("\n[4/6] Doc OMR theo chunk va lay gia tri gan discharge nhat...")

# Check schema first
omr_header = pd.read_csv(OMR_FILE, nrows=0)
required = {"subject_id", "chartdate", "result_name", "result_value"}
missing_cols = required - set(omr_header.columns)
if missing_cols:
    raise ValueError(
        "OMR thieu cac cot mong doi: "
        + ", ".join(sorted(missing_cols))
        + f"\nCac cot hien co: {list(omr_header.columns)}"
    )

# Admission windows for subject-level OMR matching
adm_window = df[
    ["subject_id", "hadm_id", "admittime", "dischtime"]
].copy()
adm_window["admit_date"] = adm_window["admittime"].dt.normalize()
adm_window["discharge_date"] = adm_window["dischtime"].dt.normalize()

# Keep latest candidate per hadm_id and feature
best_omr = {}

def parse_first_number(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+", str(x).replace(",", ""))
    if not m:
        return np.nan
    try:
        return float(m.group())
    except Exception:
        return np.nan

def parse_bp(x):
    if pd.isna(x):
        return (np.nan, np.nan)
    s = str(x).strip()
    m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", s)
    if not m:
        return (np.nan, np.nan)
    return (float(m.group(1)), float(m.group(2)))

def classify_omr_name(name):
    s = str(name).strip().lower()
    if "blood pressure" in s:
        return "bp"
    if "bmi" in s or "body mass index" in s:
        return "bmi"
    if "egfr" in s or "estimated glomerular filtration" in s:
        return "egfr"
    if "weight" in s:
        return "weight"
    return None

# We also collect observed matching names for audit
observed_names = set()

for i, chunk in enumerate(pd.read_csv(
    OMR_FILE,
    usecols=["subject_id", "chartdate", "result_name", "result_value"],
    chunksize=CHUNK_OMR,
    low_memory=False
)):
    chunk = chunk.dropna(subset=["subject_id", "chartdate", "result_name"]).copy()
    if chunk.empty:
        continue

    chunk["subject_id"] = chunk["subject_id"].astype("int64")
    chunk = chunk[chunk["subject_id"].isin(cohort_subject_ids)].copy()
    if chunk.empty:
        continue

    chunk["feature_type"] = chunk["result_name"].map(classify_omr_name)
    chunk = chunk[chunk["feature_type"].notna()].copy()
    if chunk.empty:
        continue

    observed_names.update(chunk["result_name"].dropna().astype(str).unique().tolist())

    chunk["chartdate"] = pd.to_datetime(chunk["chartdate"], errors="coerce")
    chunk = chunk.dropna(subset=["chartdate"]).copy()
    chunk["chartdate"] = chunk["chartdate"].dt.normalize()

    # subject_id join then keep OMR measurements during the current admission dates
    merged = chunk.merge(
        adm_window[
            ["subject_id", "hadm_id", "admit_date", "discharge_date"]
        ],
        on="subject_id",
        how="inner"
    )

    merged = merged[
        (merged["chartdate"] >= merged["admit_date"])
        & (merged["chartdate"] <= merged["discharge_date"])
    ].copy()

    if merged.empty:
        continue

    # Parse values
    # BP -> two features
    bp_rows = merged[merged["feature_type"].eq("bp")].copy()
    if not bp_rows.empty:
        bp_vals = bp_rows["result_value"].apply(parse_bp)
        bp_rows["sbp_last"] = [x[0] for x in bp_vals]
        bp_rows["dbp_last"] = [x[1] for x in bp_vals]

        for feat in ["sbp_last", "dbp_last"]:
            temp = (
                bp_rows.dropna(subset=[feat])
                .sort_values("chartdate")
                .drop_duplicates("hadm_id", keep="last")
                [["hadm_id", "chartdate", feat]]
            )
            for row in temp.itertuples(index=False):
                key = (int(row.hadm_id), feat)
                prev = best_omr.get(key)
                if prev is None or row.chartdate >= prev[0]:
                    best_omr[key] = (row.chartdate, float(getattr(row, feat)))

    # Numeric single-value OMR
    for ftype, out_name in [
        ("bmi", "bmi_last"),
        ("egfr", "egfr_last"),
        ("weight", "weight_raw_last"),
    ]:
        temp = merged[merged["feature_type"].eq(ftype)].copy()
        if temp.empty:
            continue

        temp["value_num"] = temp["result_value"].apply(parse_first_number)
        temp = (
            temp.dropna(subset=["value_num"])
            .sort_values("chartdate")
            .drop_duplicates("hadm_id", keep="last")
            [["hadm_id", "chartdate", "result_name", "value_num"]]
        )

        for row in temp.itertuples(index=False):
            value = float(row.value_num)

            # Convert weight to kg when OMR label clearly says lb/lbs/pound
            if out_name == "weight_raw_last":
                label = str(row.result_name).lower()
                if "lb" in label or "pound" in label:
                    value = value * 0.45359237
                out_feature = "weight_kg_last"
            else:
                out_feature = out_name

            key = (int(row.hadm_id), out_feature)
            prev = best_omr.get(key)
            if prev is None or row.chartdate >= prev[0]:
                best_omr[key] = (row.chartdate, value)

    if (i + 1) % 10 == 0:
        print(f"  OMR chunks: {i+1}; latest values kept: {len(best_omr):,}")

# Convert dict to wide table
omr_records = [
    {"hadm_id": hadm_id, "feature": feature, "value": value}
    for (hadm_id, feature), (_, value) in best_omr.items()
]

if omr_records:
    omr_long = pd.DataFrame(omr_records)
    omr_wide = (
        omr_long.pivot_table(
            index="hadm_id",
            columns="feature",
            values="value",
            aggfunc="last"
        )
        .reset_index()
    )
    omr_wide.columns.name = None
    df = df.merge(omr_wide, on="hadm_id", how="left")
else:
    print("WARNING: Khong tim thay OMR features trong admission windows.")

# Ensure columns exist even if coverage = 0
for c in [
    "bmi_last", "weight_kg_last",
    "sbp_last", "dbp_last", "egfr_last"
]:
    if c not in df.columns:
        df[c] = np.nan

# Plausibility filters - set implausible values to NaN, do not impute
plausible_ranges = {
    "bmi_last": (10, 80),
    "weight_kg_last": (25, 300),
    "sbp_last": (50, 260),
    "dbp_last": (20, 160),
    "egfr_last": (1, 200),
}

for c, (lo, hi) in plausible_ranges.items():
    bad = df[c].notna() & ~df[c].between(lo, hi)
    if bad.any():
        print(f"  {c}: set {int(bad.sum())} implausible values -> NaN")
        df.loc[bad, c] = np.nan

coverage = pd.DataFrame({
    "feature": list(plausible_ranges.keys()),
    "non_missing_n": [int(df[c].notna().sum()) for c in plausible_ranges],
    "coverage_pct": [100*df[c].notna().mean() for c in plausible_ranges],
})
coverage.to_csv(OUT_OMR_COVERAGE, index=False)

print("OMR coverage:")
print(coverage.to_string(index=False))

print("\nObserved relevant OMR result_name values:")
for x in sorted(observed_names):
    print(" ", x)

del best_omr, omr_records
gc.collect()

# ============================================================
# 5. BASIC CLINICAL PLAUSIBILITY FOR LABS
#    Only impossible/extreme data-quality values -> NaN.
#    These are broad QC ranges, not diagnostic cutoffs.
# ============================================================
print("\n[5/6] Basic data-quality checks...")

lab_ranges = {
    "creatinine_last": (0.1, 30),
    "bun_last": (1, 300),
    "sodium_last": (90, 190),
    "potassium_last": (1.5, 10),
    "hemoglobin_last": (2, 25),
    "wbc_last": (0.1, 200),
    "platelet_last": (1, 2000),
    "glucose_last": (10, 1500),
    "bicarbonate_last": (2, 60),
    "chloride_last": (50, 150),
    "calcium_last": (2, 20),
}

qc_removed = {}

for c, (lo, hi) in lab_ranges.items():
    if c not in df.columns:
        continue
    bad = df[c].notna() & ~df[c].between(lo, hi)
    qc_removed[c] = int(bad.sum())
    if bad.any():
        df.loc[bad, c] = np.nan

print("Extreme/impossible lab values replaced by NaN:")
for c, n in qc_removed.items():
    print(f"  {c:20s}: {n}")

# ============================================================
# 6. SAVE FINAL PRE-MODEL DATASET
# ============================================================
print("\n[6/6] Luu final pre-model dataset...")

# Keep IDs/timestamps because they are needed for auditing/splitting,
# but they MUST NOT be fed into the model as predictors.
id_cols = ["subject_id", "hadm_id", "admittime", "dischtime", "cv_fold", "readmit_30d"]

# Final uniqueness check
assert not df[["subject_id", "hadm_id"]].duplicated().any()
assert df["readmit_30d"].isin([0, 1]).all()

# Verify patient is still in only one CV fold
if "cv_fold" in df.columns:
    folds_per_patient = df.groupby("subject_id")["cv_fold"].nunique()
    assert folds_per_patient.max() == 1, "DATA LEAKAGE: patient appears in >1 CV fold!"

# Sort for reproducibility
df = df.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)

# Missingness
missingness = (
    df.isna().mean().mul(100)
    .sort_values(ascending=False)
    .rename("missing_pct")
    .reset_index()
    .rename(columns={"index": "feature"})
)

df.to_csv(OUT_FINAL, index=False, compression="gzip")
missingness.to_csv(OUT_MISSING, index=False)

feature_cols = [c for c in df.columns if c not in id_cols]

summary_lines = [
    "MIMIC-IV v3.1 HF FINAL PRE-MODEL DATASET",
    "=" * 52,
    f"Patients: {df['subject_id'].nunique()}",
    f"Admissions: {len(df)}",
    f"Readmissions <=30d: {int(df['readmit_30d'].sum())}",
    f"Readmission rate: {100*df['readmit_30d'].mean():.2f}%",
    f"Predictor columns (before one-hot): {len(feature_cols)}",
    "",
    "Comorbidity prevalence:",
]
for c in comorb_cols:
    summary_lines.append(f"  {c}: {100*df[c].mean():.2f}%")

summary_lines += ["", "Medication prevalence:"]
for c in MED_PATTERNS:
    summary_lines.append(f"  {c}: {100*df[c].mean():.2f}%")

summary_lines += [
    "",
    "OMR feature coverage:",
    coverage.to_string(index=False),
    "",
    "Top 25 missing features:",
    missingness.head(25).to_string(index=False),
    "",
    "NOTE:",
    "- No global imputation/scaling/one-hot has been applied.",
    "- Fit Imputer/Scaler/Encoder on TRAIN only in the model pipeline.",
    "- subject_id/hadm_id/timestamps are for auditing and must NOT be model predictors.",
    "- Comorbidity and medication mappings are pragmatic project definitions.",
]

OUT_SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

print("\n=== DONE STEP 2 ===")
print("Final dataset :", OUT_FINAL)
print("Missingness   :", OUT_MISSING)
print("Summary       :", OUT_SUMMARY)
print("OMR coverage  :", OUT_OMR_COVERAGE)
print()
print("\n".join(summary_lines[:8]))
