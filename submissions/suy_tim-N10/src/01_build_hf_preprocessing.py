# 01_build_hf_preprocessing.py
# MIMIC-IV v3.1 - Heart Failure 30-day readmission
# Run locally on Windows. DO NOT upload raw MIMIC data to public repositories/AI services.

from pathlib import Path
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR = Path(r"D:\MIMIC_IV_3_1\hosp")
OUT_DIR = DATA_DIR.parent / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_PRESC = 500_000
CHUNK_LABS = 1_000_000
RANDOM_STATE = 42

FILES = {
    "patients": DATA_DIR / "patients.csv.gz",
    "admissions": DATA_DIR / "admissions.csv.gz",
    "diagnoses": DATA_DIR / "diagnoses_icd.csv.gz",
    "procedures": DATA_DIR / "procedures_icd.csv.gz",
    "prescriptions": DATA_DIR / "prescriptions.csv.gz",
    "d_labitems": DATA_DIR / "d_labitems.csv.gz",
    "labevents": DATA_DIR / "labevents.csv.gz",
}

missing = [str(p) for p in FILES.values() if not p.exists()]
if missing:
    raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

print("=== MIMIC-IV v3.1 HF preprocessing ===")
print("DATA_DIR:", DATA_DIR)
print("OUT_DIR :", OUT_DIR)

print("\n[1/9] Reading admissions + patients...")
admissions = pd.read_csv(
    FILES["admissions"],
    usecols=[
        "subject_id", "hadm_id", "admittime", "dischtime",
        "admission_type", "admission_location", "discharge_location",
        "insurance", "language", "marital_status", "race",
        "hospital_expire_flag"
    ],
    parse_dates=["admittime", "dischtime"],
)
patients = pd.read_csv(
    FILES["patients"],
    usecols=["subject_id", "gender", "anchor_age", "anchor_year"]
)
admissions = admissions.dropna(subset=["subject_id", "hadm_id", "admittime", "dischtime"]).copy()
admissions = admissions[admissions["dischtime"] >= admissions["admittime"]].copy()
admissions = admissions.merge(patients, on="subject_id", how="left")
admissions["age"] = admissions["anchor_age"] + admissions["admittime"].dt.year - admissions["anchor_year"]
admissions["los_days"] = (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 86400.0

print("[2/9] Selecting HF admissions (ICD-9 428* / ICD-10 I50*)...")
dx = pd.read_csv(FILES["diagnoses"], usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
dx["icd_code_norm"] = dx["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False).str.strip()
is_hf = (
    ((dx["icd_version"] == 9) & dx["icd_code_norm"].str.startswith("428")) |
    ((dx["icd_version"] == 10) & dx["icd_code_norm"].str.startswith("I50"))
)
hf_hadm_ids = set(dx.loc[is_hf, "hadm_id"].dropna().astype("int64").tolist())
print("HF hadm_id found:", len(hf_hadm_ids))

print("[3/9] Creating 30-day all-cause readmission label...")
all_adm = admissions.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)
all_adm["next_admittime"] = all_adm.groupby("subject_id")["admittime"].shift(-1)
all_adm["prev_dischtime"] = all_adm.groupby("subject_id")["dischtime"].shift(1)
all_adm["prior_admissions"] = all_adm.groupby("subject_id").cumcount()
gap_days = (all_adm["next_admittime"] - all_adm["dischtime"]).dt.total_seconds() / 86400.0
all_adm["readmit_30d"] = ((gap_days > 0) & (gap_days <= 30)).astype("int8")
all_adm["days_to_next_admission"] = gap_days
all_adm["days_since_last_discharge"] = (all_adm["admittime"] - all_adm["prev_dischtime"]).dt.total_seconds() / 86400.0

def add_prior_window_counts(df):
    out_30 = np.zeros(len(df), dtype=np.int32)
    out_90 = np.zeros(len(df), dtype=np.int32)
    out_365 = np.zeros(len(df), dtype=np.int32)
    for _, idx in df.groupby("subject_id", sort=False).groups.items():
        idx = np.asarray(list(idx))
        times_i = df.loc[idx, "admittime"].values.astype("datetime64[ns]").astype("int64")
        for days, out_arr in [(30, out_30), (90, out_90), (365, out_365)]:
            delta = np.int64(pd.Timedelta(days=days).value)
            left = np.searchsorted(times_i, times_i - delta, side="left")
            pos = np.arange(len(times_i), dtype=np.int64)
            out_arr[idx] = (pos - left).astype(np.int32)
    df["prior_admissions_30d"] = out_30
    df["prior_admissions_90d"] = out_90
    df["prior_admissions_1y"] = out_365
    return df

all_adm = add_prior_window_counts(all_adm)
cohort = all_adm[all_adm["hadm_id"].astype("int64").isin(hf_hadm_ids)].copy()
cohort = cohort[(cohort["age"] >= 18) & (cohort["hospital_expire_flag"] == 0) & (cohort["los_days"] >= 1.0)].copy()
cohort = cohort.sort_values(["subject_id", "admittime", "hadm_id"])
cohort["prior_hf_admissions"] = cohort.groupby("subject_id").cumcount()
print("Eligible HF patients   :", cohort["subject_id"].nunique())
print("Eligible HF admissions :", len(cohort))
print("Readmission <=30d      :", int(cohort["readmit_30d"].sum()))
print("Readmission rate       :", f'{100*cohort["readmit_30d"].mean():.2f}%')
cohort_hadm_ids = set(cohort["hadm_id"].astype("int64").tolist())

print("[4/9] Diagnosis features...")
dx_hf = dx[dx["hadm_id"].astype("int64").isin(cohort_hadm_ids)].copy()
diag_counts = dx_hf.groupby("hadm_id").agg(n_diagnoses=("icd_code_norm", "nunique")).reset_index()
cohort = cohort.merge(diag_counts, on="hadm_id", how="left")

print("[5/9] Procedure features...")
proc = pd.read_csv(FILES["procedures"], usecols=["hadm_id", "icd_code", "icd_version"])
proc = proc[proc["hadm_id"].astype("int64").isin(cohort_hadm_ids)].copy()
proc["icd_code_norm"] = proc["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False).str.strip()
proc_counts = proc.groupby("hadm_id").agg(n_procedures=("icd_code_norm", "nunique")).reset_index()
cohort = cohort.merge(proc_counts, on="hadm_id", how="left")
del proc, proc_counts
gc.collect()

print("[6/9] Prescription features by chunks...")
drug_pairs = []
for i, chunk in enumerate(pd.read_csv(FILES["prescriptions"], usecols=["hadm_id", "drug"], chunksize=CHUNK_PRESC)):
    chunk = chunk.dropna(subset=["hadm_id", "drug"])
    chunk = chunk[chunk["hadm_id"].astype("int64").isin(cohort_hadm_ids)]
    if not chunk.empty:
        chunk["drug_norm"] = chunk["drug"].astype(str).str.strip().str.lower()
        drug_pairs.append(chunk[["hadm_id", "drug_norm"]].drop_duplicates())
    if (i + 1) % 10 == 0:
        print(f"  prescription chunks: {i+1}")
if drug_pairs:
    drug_pairs = pd.concat(drug_pairs, ignore_index=True).drop_duplicates()
    drug_counts = drug_pairs.groupby("hadm_id")["drug_norm"].nunique().rename("n_unique_drugs").reset_index()
    cohort = cohort.merge(drug_counts, on="hadm_id", how="left")
    del drug_pairs, drug_counts
else:
    cohort["n_unique_drugs"] = np.nan
gc.collect()

print("[7/9] Selected lab features by chunks...")
d_lab = pd.read_csv(FILES["d_labitems"], usecols=["itemid", "label", "fluid"])
d_lab["label_norm"] = d_lab["label"].astype(str).str.strip().str.lower()
d_lab["fluid_norm"] = d_lab["fluid"].astype(str).str.strip().str.lower()
LAB_NAME_MAP = {
    "creatinine": "creatinine_last",
    "urea nitrogen": "bun_last",
    "sodium": "sodium_last",
    "potassium": "potassium_last",
    "hemoglobin": "hemoglobin_last",
    "white blood cells": "wbc_last",
    "platelet count": "platelet_last",
    "glucose": "glucose_last",
    "bicarbonate": "bicarbonate_last",
    "chloride": "chloride_last",
    "calcium, total": "calcium_last",
}
lab_dict = d_lab[d_lab["label_norm"].isin(LAB_NAME_MAP.keys()) & (d_lab["fluid_norm"] == "blood")][["itemid", "label_norm"]].copy()
lab_dict["feature_name"] = lab_dict["label_norm"].map(LAB_NAME_MAP)
selected_itemids = set(lab_dict["itemid"].astype("int64").tolist())
itemid_to_feature = dict(zip(lab_dict["itemid"].astype("int64"), lab_dict["feature_name"]))
print("Selected lab itemids:", len(selected_itemids))
print(lab_dict.sort_values(["feature_name", "itemid"]).to_string(index=False))

window = cohort[["hadm_id", "admittime", "dischtime"]].copy()
window["hadm_id"] = window["hadm_id"].astype("int64")
best_labs = pd.DataFrame(columns=["hadm_id", "itemid", "charttime", "valuenum"])
for i, chunk in enumerate(pd.read_csv(FILES["labevents"], usecols=["hadm_id", "itemid", "charttime", "valuenum"], chunksize=CHUNK_LABS, low_memory=False)):
    chunk = chunk.dropna(subset=["hadm_id", "itemid", "charttime", "valuenum"])
    if chunk.empty:
        continue
    chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
    chunk["itemid"] = chunk["itemid"].astype("int64")
    chunk = chunk[chunk["hadm_id"].isin(cohort_hadm_ids) & chunk["itemid"].isin(selected_itemids)]
    if not chunk.empty:
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime"])
        chunk = chunk.merge(window, on="hadm_id", how="inner")
        chunk = chunk[(chunk["charttime"] >= chunk["admittime"]) & (chunk["charttime"] <= chunk["dischtime"])]
        if not chunk.empty:
            current = chunk.sort_values("charttime").drop_duplicates(["hadm_id", "itemid"], keep="last")[["hadm_id", "itemid", "charttime", "valuenum"]]
            best_labs = pd.concat([best_labs, current], ignore_index=True)
            best_labs = best_labs.sort_values("charttime").drop_duplicates(["hadm_id", "itemid"], keep="last")
    if (i + 1) % 10 == 0:
        print(f"  labevents chunks: {i+1}; latest groups kept: {len(best_labs):,}")
if not best_labs.empty:
    best_labs["feature_name"] = best_labs["itemid"].astype("int64").map(itemid_to_feature)
    labs_wide = best_labs.pivot_table(index="hadm_id", columns="feature_name", values="valuenum", aggfunc="last").reset_index()
    labs_wide.columns.name = None
    cohort = cohort.merge(labs_wide, on="hadm_id", how="left")
del best_labs, d_lab, lab_dict
gc.collect()

print("[8/9] Cleaning final feature table...")
count_cols = ["n_diagnoses", "n_procedures", "n_unique_drugs", "prior_admissions", "prior_hf_admissions", "prior_admissions_30d", "prior_admissions_90d", "prior_admissions_1y"]
for c in count_cols:
    if c in cohort.columns:
        cohort[c] = cohort[c].fillna(0)

keep_cols = [
    "subject_id", "hadm_id", "admittime", "dischtime", "readmit_30d",
    "age", "gender", "los_days", "admission_type", "insurance", "race", "marital_status", "admission_location", "discharge_location",
    "prior_admissions", "prior_hf_admissions", "prior_admissions_30d", "prior_admissions_90d", "prior_admissions_1y", "days_since_last_discharge",
    "n_diagnoses", "n_procedures", "n_unique_drugs",
    "creatinine_last", "bun_last", "sodium_last", "potassium_last", "hemoglobin_last", "wbc_last", "platelet_last", "glucose_last", "bicarbonate_last", "chloride_last", "calcium_last",
]
keep_cols = [c for c in keep_cols if c in cohort.columns]
final = cohort[keep_cols].drop_duplicates(subset=["subject_id", "hadm_id"]).reset_index(drop=True)
missingness = final.isna().mean().mul(100).sort_values(ascending=False).rename("missing_pct").reset_index().rename(columns={"index": "feature"})

print("[9/9] Creating patient-level stratified CV folds...")
final["cv_fold"] = -1
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
X_dummy = np.zeros((len(final), 1))
y = final["readmit_30d"].astype(int).values
groups = final["subject_id"].values
for fold, (_, val_idx) in enumerate(sgkf.split(X_dummy, y, groups)):
    final.loc[val_idx, "cv_fold"] = fold
assert (final["cv_fold"] >= 0).all()
assert final.groupby("subject_id")["cv_fold"].nunique().max() == 1, "DATA LEAKAGE: patient appears in >1 fold!"

out_features = OUT_DIR / "hf_features_v1.csv.gz"
out_missing = OUT_DIR / "hf_missingness_v1.csv"
out_summary = OUT_DIR / "hf_summary_v1.txt"
final.to_csv(out_features, index=False, compression="gzip")
missingness.to_csv(out_missing, index=False)
summary = []
summary.append("MIMIC-IV v3.1 HF 30-day readmission preprocessing summary")
summary.append(f"Patients: {final['subject_id'].nunique()}")
summary.append(f"Admissions: {len(final)}")
summary.append(f"Readmissions <=30d: {int(final['readmit_30d'].sum())}")
summary.append(f"Readmission rate: {100*final['readmit_30d'].mean():.2f}%")
summary.append("")
summary.append("Fold distribution:")
summary.append(final.groupby("cv_fold")["readmit_30d"].agg(["count", "sum", "mean"]).to_string())
summary.append("")
summary.append("Top missingness:")
summary.append(missingness.head(20).to_string(index=False))
out_summary.write_text("\n".join(summary), encoding="utf-8")

print("\n=== DONE ===")
print("Feature file :", out_features)
print("Missingness  :", out_missing)
print("Summary      :", out_summary)
print("\n" + "\n".join(summary[:6]))
print("\nNOTE: no global imputation/scaling/one-hot has been applied.")
print("Those data-dependent steps must be fitted on TRAIN only in the model pipeline.")
