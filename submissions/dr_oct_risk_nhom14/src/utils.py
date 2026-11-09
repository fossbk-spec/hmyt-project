"""Metrics, fairness metrics and threshold selection shared by all scripts."""
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def binary_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    pred = (p >= thr).astype(int)
    P = max(1, int((y == 1).sum()))
    N = max(1, int((y == 0).sum()))
    sens = float(((pred == 1) & (y == 1)).sum() / P)
    spec = float(((pred == 0) & (y == 0)).sum() / N)
    return dict(auc=safe_auc(y, p), bacc=(sens + spec) / 2, sens=sens, spec=spec, thr=float(thr))


def fairness_report(y, p, g, thr=0.5):
    """Group-wise AUC, ES-AUC, Mean/Max PSD (FairVision paper) and an approximate DEOdds.

    ES-AUC = AUC / (1 + sum_a |AUC - AUC_a|)   (Luo et al., Harvard-GF; matches FairVision tables)
    Mean PSD = std_a(AUC_a) / AUC ;  Max PSD = (max_a AUC_a - min_a AUC_a) / AUC
    DEOdds here = (max-min TPR across groups) + (max-min FPR across groups). This is OUR approximation:
    the paper does not give the exact formula, so do not compare absolute values with the paper.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    g = np.asarray(g)
    overall = safe_auc(y, p)
    aucs, tprs, fprs = {}, [], []
    for a in np.unique(g):
        m = g == a
        if m.sum() < 5 or len(np.unique(y[m])) < 2:
            continue
        aucs[int(a)] = safe_auc(y[m], p[m])
        mm = binary_metrics(y[m], p[m], thr)
        tprs.append(mm["sens"])
        fprs.append(1 - mm["spec"])
    if len(aucs) < 2:
        return {}
    v = np.array(list(aucs.values()))
    return dict(
        overall_auc=overall,
        group_auc=aucs,
        es_auc=float(overall / (1 + np.abs(v - overall).sum())),
        mean_psd=float(np.sqrt(((v - v.mean()) ** 2).mean()) / overall),
        max_psd=float((v.max() - v.min()) / overall),
        deodds_approx=float((max(tprs) - min(tprs)) + (max(fprs) - min(fprs))),
    )


def pick_threshold(y, p, strategy="youden", target_sens=0.9):
    """Choose a threshold on VALIDATION data only."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2 or strategy == "fixed":
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    if strategy == "youden":
        i = int(np.argmax(tpr - fpr))
    elif strategy == "sens":  # first (=highest) threshold that reaches the target sensitivity
        i = int(np.where(tpr >= target_sens)[0][0])
    else:
        raise ValueError(strategy)
    t = float(thr[i])
    return t if np.isfinite(t) else 1.0


def mean_std(dicts):
    """Aggregate a list of flat dicts of floats -> {key: (mean, std)}."""
    out = {}
    for k in dicts[0]:
        if isinstance(dicts[0][k], (int, float)):
            vals = np.array([d[k] for d in dicts], dtype=float)
            out[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
    return out


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
