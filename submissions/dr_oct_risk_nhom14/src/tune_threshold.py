"""Post-hoc threshold analysis from saved preds.npz (val used to choose, test used only to report).
  python tune_threshold.py runs/fv_vit/seed0/preds.npz --group_idx 0
Strategies: fixed 0.5, Youden, sensitivity>=target, and PER-GROUP Youden (analysis only: group-specific
thresholds raise ethical/regulatory questions - report them, do not assume they are deployable)."""
import argparse
import numpy as np
from utils import binary_metrics, fairness_report, pick_threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("preds"); p.add_argument("--group_idx", type=int, default=0)
    p.add_argument("--target_sens", type=float, nargs="+", default=[0.85, 0.90, 0.95])
    a = p.parse_args()
    d = np.load(a.preds, allow_pickle=True)
    yv, pv, yt, pt = d["yv"], d["pv"], d["yt"], d["pt"]
    gv, gt = d["gv"][:, a.group_idx], d["gt"][:, a.group_idx]
    def show(name, thr):
        m = binary_metrics(yt, pt, thr)
        f = fairness_report(yt, pt, gt, thr) if (gt >= 0).all() else {}
        print(f"{name:22s} thr={thr:.3f} sens={m['sens']:.3f} spec={m['spec']:.3f} bacc={m['bacc']:.3f}"
              + (f" deodds~={f['deodds_approx']:.3f}" if f else ""))
    show("fixed 0.5", 0.5)
    show("youden(val)", pick_threshold(yv, pv, "youden"))
    for s in a.target_sens: show(f"sens>={s}(val)", pick_threshold(yv, pv, "sens", s))
    if (gv >= 0).all():
        pred = np.zeros_like(yt)
        for g in np.unique(gt):
            thr = pick_threshold(yv[gv == g], pv[gv == g], "youden") if len(np.unique(yv[gv == g])) > 1 else 0.5
            pred[gt == g] = pt[gt == g] >= thr
        tp = ((pred == 1) & (yt == 1)).sum() / max(1, (yt == 1).sum()); tn = ((pred == 0) & (yt == 0)).sum() / max(1, (yt == 0).sum())
        print(f"{'per-group youden':22s} sens={tp:.3f} spec={tn:.3f} bacc={(tp + tn) / 2:.3f}")

if __name__ == "__main__":
    main()
