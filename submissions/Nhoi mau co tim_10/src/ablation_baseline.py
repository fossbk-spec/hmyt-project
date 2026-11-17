"""Ablation cho baseline: bỏ từng nhóm đặc trưng (ST, Q, R, T, toàn cục) và đo ΔAUROC trên test (bootstrap cặp).
Chạy sau train_baseline.py:  python src/ablation_baseline.py --data data/ptb-xl"""
import argparse, json, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import SEED, get_data, paired_auc_diff, split_masks
from features import NAMES

GROUPS = {"ST": ("_st",), "Q": ("_q", "_qrs_min"), "R": ("_r",), "T": ("_t",), "toàn cục (HR, RR, số nhịp)": ("hr", "rr_std", "n_beats")}


def idx(keys): return [i for i, n in enumerate(NAMES) if any(n.endswith(k) or n == k for k in keys)]


def build(name, hp):
    if name == "lr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=float(hp.split("=")[1]), class_weight="balanced", max_iter=2000))
    d, l = hp.split(","); return HistGradientBoostingClassifier(max_depth=int(d.split("=")[1]), learning_rate=float(l.split("=")[1]),
                                                                 max_iter=300, class_weight="balanced", random_state=SEED)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", default="data/ptb-xl"); a.add_argument("--out", default="results"); a.add_argument("--synthetic", action="store_true")
    a = a.parse_args()
    _, meta = get_data(a); tr, va, te = split_masks(meta); y = meta.y.values
    F = np.load(f"{a.out}/features.npz")["F"]; res = {}
    for name in ("lr", "hgb"):
        hp = json.load(open(f"{a.out}/{name}_metrics.json"))["chosen_hparams"]
        def fit_pred(cols):
            m = build(name, hp).fit(F[tr][:, cols], y[tr]); return m.predict_proba(F[te][:, cols])[:, 1]
        full = list(range(F.shape[1])); p_full = fit_pred(full); rows = {}
        from sklearn.metrics import roc_auc_score
        rows["Đầy đủ (63 đặc trưng)"] = {"auroc": float(roc_auc_score(y[te], p_full)), "delta": None}
        for g, keys in GROUPS.items():
            cols = [i for i in full if i not in idx(keys)]
            rows[f"Bỏ nhóm {g}"] = {"auroc": float(roc_auc_score(y[te], (p := fit_pred(cols)))), "delta": paired_auc_diff(y[te], p, p_full)}
        cols = idx(GROUPS["ST"]); p = fit_pred(cols)
        rows["Chỉ nhóm ST"] = {"auroc": float(roc_auc_score(y[te], p)), "delta": paired_auc_diff(y[te], p, p_full)}
        res[name] = rows; print(name, {k: round(v["auroc"], 3) for k, v in rows.items()})
    json.dump(res, open(f"{a.out}/ablation.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
