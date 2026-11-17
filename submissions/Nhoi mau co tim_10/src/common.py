"""Tiện ích dùng chung: nạp PTB-XL, chia tập, chuẩn hóa, chỉ số + CI bootstrap, ghi kết quả."""
import ast, json, os
import numpy as np, pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score, roc_curve)

SEED = 42
FS = 100
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_ANT, _INF = {"AMI", "ASMI", "ALMI"}, {"IMI", "ILMI", "IPMI", "IPLMI"}
META_COLS = ["y", "fold", "pid", "age", "sex", "loc"]


def _loc(mi_codes):
    s = set(mi_codes)
    if not s: return "NORM"
    return "truoc" if s & _ANT else "duoi" if s & _INF else "khac"


def _read_cache(path):
    z = np.load(path)
    return z["X"], pd.DataFrame({k: z[k] for k in META_COLS})


def load_ptbxl(root, cache):
    """Trả về X (N, T, 12) float32 và DataFrame meta (y, fold, pid, age, sex, loc).
    Nhãn: 1 = MI, 0 = NORM; bỏ các bản ghi thuộc lớp chẩn đoán khác (STTC, CD, HYP) hoặc đa nhãn lẫn."""
    if os.path.exists(cache):
        return _read_cache(cache)
    import wfdb
    db = pd.read_csv(f"{root}/ptbxl_database.csv", index_col="ecg_id")
    st = pd.read_csv(f"{root}/scp_statements.csv", index_col=0)
    st = st[st.diagnostic == 1]
    rows = []
    for ecg_id, r in db.iterrows():
        d = {k: v for k, v in ast.literal_eval(r.scp_codes).items() if k in st.index and v > 0}
        cls = {st.loc[k].diagnostic_class for k in d}
        if "MI" in cls: y = 1
        elif cls == {"NORM"}: y = 0
        else: continue
        mi = [k for k in d if st.loc[k].diagnostic_class == "MI"]
        age = r.age if r.age <= 120 else np.nan  # PTB-XL ẩn danh hóa tuổi > 89 thành 300
        rows.append((ecg_id, y, int(r.strat_fold), int(r.patient_id), age, int(r.sex), _loc(mi), r.filename_lr))
    meta = pd.DataFrame(rows, columns=["ecg_id", "y", "fold", "pid", "age", "sex", "loc", "file"])
    X = []
    for i, f in enumerate(meta.file):
        X.append(wfdb.rdsamp(f"{root}/{f}")[0])
        if i % 2000 == 0: print(f"  đọc {i}/{len(meta)}", flush=True)
    X = np.nan_to_num(np.stack(X).astype(np.float32))
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez_compressed(cache, X=X, **{k: meta[k].values.astype(str) if k == "loc" else meta[k].values for k in META_COLS})
    return X, meta[META_COLS]


def get_data(args):
    """--synthetic: dữ liệu mô phỏng để kiểm thử đường ống (KHÔNG dùng cho báo cáo)."""
    if getattr(args, "synthetic", False):
        from synthetic import make_synthetic
        return make_synthetic()
    return load_ptbxl(args.data, os.path.join(args.out, "cache.npz"))


def split_masks(meta):
    f = meta.fold.values
    return f <= 8, f == 9, f == 10          # train, val, test (strat_fold chính thức)


def check_no_patient_overlap(meta):
    tr, va, te = split_masks(meta)
    P = [set(meta.pid[m]) for m in (tr, va, te)]
    ov = {"train_val": len(P[0] & P[1]), "train_test": len(P[0] & P[2]), "val_test": len(P[1] & P[2])}
    assert sum(ov.values()) == 0, f"RÒ RỈ: bệnh nhân xuất hiện ở nhiều tập {ov}"
    return ov


def fit_norm(X_train):
    return X_train.mean((0, 1), keepdims=True), X_train.std((0, 1), keepdims=True) + 1e-6


def data_stats(meta):
    tr, va, te = split_masks(meta)
    out = {"n_total": int(len(meta)), "n_patients": int(meta.pid.nunique()),
           "prevalence_MI": float(meta.y.mean()), "patient_overlap": check_no_patient_overlap(meta)}
    for n, m in zip(["train", "val", "test"], [tr, va, te]):
        out[n] = {"n": int(m.sum()), "n_patients": int(meta.pid[m].nunique()), "n_MI": int(meta.y[m].sum()),
                  "MI_pct": float(100 * meta.y[m].mean())}
    out["MI_location_counts"] = {k: int(v) for k, v in meta[meta.y == 1]["loc"].value_counts().items()}
    out["age_mean_sd"] = [float(np.nanmean(meta.age)), float(np.nanstd(meta.age))]
    out["female_pct"] = float(100 * (meta.sex == 1).mean())
    return out


# ---------- chỉ số ----------
def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def point_metrics(y, p, thr):
    pred = p >= thr
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    d = lambda a, b: a / b if b else np.nan
    sens, ppv = d(tp, tp + fn), d(tp, tp + fp)
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p), "sens": sens,
            "spec": d(tn, tn + fp), "ppv": ppv, "npv": d(tn, tn + fn),
            "f1": d(2 * ppv * sens, ppv + sens) if ppv == ppv and sens == sens else np.nan,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def bootstrap_ci(y, p, thr, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); rows = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): rows.append(point_metrics(y[i], p[i], thr))
    keys = ["auroc", "auprc", "sens", "spec", "ppv", "npv", "f1"]
    return {k: [float(np.nanpercentile([r[k] for r in rows], q)) for q in (2.5, 97.5)] for k in keys}


def paired_auc_diff(y, p1, p2, n=2000, seed=SEED):
    """Hiệu AUROC (p1 - p2) trên cùng tập test, bootstrap theo cặp."""
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return {"diff": float(roc_auc_score(y, p1) - roc_auc_score(y, p2)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for k in range(bins):
        m = b == k
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def calibration(y, p, eps=1e-6):
    """Hiệu chuẩn theo TRIPOD+AI/PROBAST+AI: slope (hồi quy logistic y ~ logit p) và calibration-in-the-large (intercept với offset logit p)."""
    from scipy.optimize import brentq
    from sklearn.linear_model import LogisticRegression
    q = np.clip(p, eps, 1 - eps); z = np.log(q / (1 - q))
    slope = float(LogisticRegression(penalty=None, max_iter=1000).fit(z[:, None], y).coef_[0, 0])
    citl = float(brentq(lambda a: np.sum(y - 1 / (1 + np.exp(-(a + z)))), -15, 15))
    return {"cal_slope": slope, "cal_in_the_large": citl}


def calibration_ci(y, p, n=500, seed=SEED):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): v.append(calibration(y[i], p[i]))
    return {k: [float(np.percentile([r[k] for r in v], q)) for q in (2.5, 97.5)] for k in ("cal_slope", "cal_in_the_large")}


def finalize(name, y_val, p_val, y_te, p_te, te_idx, out):
    """Chọn ngưỡng Youden trên val → đánh giá test đúng 1 lần → ghi json + dự đoán."""
    os.makedirs(out, exist_ok=True)
    thr = youden_threshold(y_val, p_val)
    res = {"model": name, "threshold_from_val": thr, "val_auroc": float(roc_auc_score(y_val, p_val)),
           "n_test": int(len(y_te)), "n_test_MI": int(y_te.sum()),
           "test": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in point_metrics(y_te, p_te, thr).items()},
           "ci95": bootstrap_ci(y_te, p_te, thr), "brier": float(brier_score_loss(y_te, p_te)),
           "ece": ece(y_te, p_te), "calibration": {**calibration(y_te, p_te), "ci95": calibration_ci(y_te, p_te)}, "seed": SEED}
    json.dump(res, open(f"{out}/{name}_metrics.json", "w"), indent=2)
    np.savez(f"{out}/{name}_preds.npz", y_val=y_val, p_val=p_val, y_test=y_te, p_test=p_te, test_idx=te_idx)
    print(f"[{name}] test AUROC={res['test']['auroc']:.4f} {res['ci95']['auroc']}  Se={res['test']['sens']:.3f} Sp={res['test']['spec']:.3f}")
    return res
