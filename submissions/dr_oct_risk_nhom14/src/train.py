"""Train / evaluate on FairVision-DR or OLIVES. Model selection and thresholds use VALIDATION only.

Paper-style baselines (FairVision paper, Sec. 5):
  3D OCT : python train.py --task fairvision --root R --mode 3d --epochs 50 --bs 2 --lr 5e-5
  2D     : python train.py --task fairvision --root R --mode 2d_mil --arch vit_b_16 --epochs 50 --bs 64 --lr 5e-4
  CNN    : python train.py --task fairvision --root R --mode 2d_mil --arch resnet50 --epochs 10 --bs 6 --lr 1e-4
OLIVES baseline (paper: ResNet-18 on OCT B-scans, DR vs DME, balanced accuracy):
  python train.py --task olives --csv C --root IMGS --arch resnet18 --first_last ...
Transfer: add --init_enc runs/xxx/seed0/encoder.pt [--freeze]   (2D models only)
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import CachedFV, FairVisionDR, OlivesBscan, group_split, load_olives_df
from models import build_model, make_loss
from utils import binary_metrics, dump_json, fairness_report, mean_std, pick_threshold, seed_everything


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["fairvision", "olives"], required=True)
    p.add_argument("--root", required=True, help="FairVision/DR root, or OLIVES image root")
    # fairvision
    p.add_argument("--oct_key", default="oct_bscans")
    p.add_argument("--label_key", default="dr_subtype")
    p.add_argument("--attrs", default="race,male,hispanic")
    p.add_argument("--slice_axis", type=int, default=0)
    p.add_argument("--slo_key", default="slo_fundus")
    p.add_argument("--cache", default=None, help="dir produced by cache_fv.py (fast path)")
    p.add_argument("--depth", type=int, default=64)
    p.add_argument("--k_slices", type=int, default=7)
    # olives
    p.add_argument("--csv")
    p.add_argument("--path_col", default="Path")
    p.add_argument("--label_col", default="Disease")
    p.add_argument("--pos_value", default=None)
    p.add_argument("--patient_col", default="Eye_ID")
    p.add_argument("--eye_col", default="Eye_ID")
    p.add_argument("--week_col", default="Week")
    p.add_argument("--first_last", action="store_true")
    # model
    p.add_argument("--mode", choices=["2d_mil", "3d", "slo"], default="2d_mil")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--pool", default="mean", choices=["mean", "max", "attn"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--init_enc")
    p.add_argument("--freeze", action="store_true", help="linear probe: freeze encoder")
    # optimisation
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--balance", choices=["none", "sampler", "posweight"], default="none")
    p.add_argument("--loss", choices=["ce", "fis"], default="ce")
    p.add_argument("--fis_c", type=float, default=0.5)
    p.add_argument("--fair_attr", default="race")
    p.add_argument("--drop_path", type=float, default=0.0)
    p.add_argument("--layer_decay", type=float, default=1.0, help="<1 only for timm:vit* (MAE fine-tuning)")
    p.add_argument("--warmup_epochs", type=float, default=0.0)
    p.add_argument("--imagenet_norm", action="store_true")
    p.add_argument("--log_every", type=int, default=0, help="print progress every N steps")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--channels_last", action="store_true")
    p.add_argument("--thr_strategy", default="youden", choices=["fixed", "youden", "sens"])
    p.add_argument("--target_sens", type=float, default=0.9)
    # misc
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_n", type=int, default=None, help="debug: limit samples per split")
    p.add_argument("--out", default="runs/exp")
    return p.parse_args(argv)


def build_data(a, seed):
    if a.task == "fairvision":
        attrs = a.attrs.split(",")
        if a.cache:
            return (CachedFV(a.cache, "train", True, a.max_n), CachedFV(a.cache, "val", False, a.max_n),
                    CachedFV(a.cache, "test", False, a.max_n), attrs)
        kw = dict(slo_key=a.slo_key, mode=a.mode, oct_key=a.oct_key, label_key=a.label_key, attrs=attrs, size=a.size, depth=a.depth,
                  k_slices=a.k_slices, slice_axis=a.slice_axis, max_n=a.max_n)
        return (FairVisionDR(a.root, "train", augment=True, **kw), FairVisionDR(a.root, "val", **kw),
                FairVisionDR(a.root, "test", **kw), attrs)
    df = load_olives_df(a.csv, a.path_col, a.label_col, a.patient_col, a.eye_col, a.week_col, a.first_last,
                        a.pos_value)
    tr, va, te = group_split(df, a.patient_col, seed=seed)
    if a.max_n:
        tr, va, te = tr.sample(min(a.max_n, len(tr)), random_state=0), va.head(a.max_n), te.head(a.max_n)
    return (OlivesBscan(tr, a.root, a.path_col, a.size, True), OlivesBscan(va, a.root, a.path_col, a.size),
            OlivesBscan(te, a.root, a.path_col, a.size), ["none"])


@torch.no_grad()
def predict(model, loader, dev, amp, cl):
    model.eval()
    ys, ps, gs = [], [], []
    for x, y, g in loader:
        x = x.to(dev, non_blocking=True)
        if cl and x.dim() == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev.type, enabled=amp):
            lo = model(x)
        ps.append(torch.sigmoid(lo.float()).cpu())
        ys.append(y)
        gs.append(g)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(gs).numpy()


def param_groups(model, lr, wd, layer_decay):
    """AdamW groups. With layer_decay<1 on a timm ViT: lr_i = lr * layer_decay**(n_layers+1-i) (MAE recipe),
    no weight decay on biases/norms/cls/pos tokens."""
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    blocks = [int(n.split(".")[2]) for n, _ in params if n.startswith("enc.blocks.")]
    if layer_decay >= 1.0 or not blocks:
        return [dict(params=[p for _, p in params], lr=lr, weight_decay=wd)]
    n_layers = max(blocks) + 1
    groups = {}
    for n, p in params:
        if n.startswith("enc.blocks."):
            lid = int(n.split(".")[2]) + 1
        elif n.startswith("enc.") and any(k in n for k in ("cls_token", "pos_embed", "patch_embed", "reg_token")):
            lid = 0
        else:
            lid = n_layers + 1
        no_wd = p.ndim == 1 or n.endswith(".bias") or "cls_token" in n or "pos_embed" in n
        g = groups.setdefault((lid, no_wd), dict(params=[], lr=lr * layer_decay ** (n_layers + 1 - lid),
                                                 weight_decay=0.0 if no_wd else wd))
        g["params"].append(p)
    print(f"layer-wise lr decay {layer_decay}: {len(groups)} groups, lr from {min(g['lr'] for g in groups.values()):.2e}"
          f" to {max(g['lr'] for g in groups.values()):.2e}")
    return list(groups.values())


def run_seed(a, seed, dev):
    seed_everything(seed)
    out = os.path.join(a.out, f"seed{seed}")
    os.makedirs(out, exist_ok=True)
    tr, va, te, attrs = build_data(a, seed)
    ytr = tr.labels()
    print(f"[seed {seed}] train/val/test = {len(tr)}/{len(va)}/{len(te)}  train positive rate={ytr.mean():.3f}")
    sampler = None
    pos_w = None
    if a.balance == "sampler":
        w = np.where(ytr == 1, 0.5 / max(1, (ytr == 1).sum()), 0.5 / max(1, (ytr == 0).sum()))
        sampler = WeightedRandomSampler(w, len(w), replacement=True)
    elif a.balance == "posweight":
        pos_w = torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32, device=dev)
    kw = dict(num_workers=a.workers, pin_memory=dev.type == "cuda")
    ltr = DataLoader(tr, a.bs, shuffle=sampler is None, sampler=sampler, drop_last=True, **kw)
    lva, lte = DataLoader(va, a.bs, **kw), DataLoader(te, a.bs, **kw)

    model = build_model(a).to(dev)
    if a.init_enc:
        sd = torch.load(a.init_enc, map_location=dev)["enc"]
        print("init encoder:", model.enc.load_state_dict(sd, strict=False))
    if a.freeze:
        for p_ in model.enc.parameters():
            p_.requires_grad = False
    if a.channels_last and a.mode == "2d_mil":
        model = model.to(memory_format=torch.channels_last)
    groups = param_groups(model, a.lr, a.wd, a.layer_decay)
    opt = torch.optim.AdamW(groups)
    steps = a.epochs * max(1, len(ltr))
    pct = min(0.5, max(0.02, a.warmup_epochs / a.epochs)) if a.warmup_epochs > 0 else 0.1
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps, pct_start=pct)
    scaler = torch.amp.GradScaler(enabled=a.amp and dev.type == "cuda")
    fis = make_loss(a, pos_w)
    fair_idx = attrs.index(a.fair_attr) if a.fair_attr in attrs else None
    best, best_ep = -1, -1
    for ep in range(a.epochs):
        model.train()
        if a.freeze:
            model.enc.eval()
        t0, tot = time.time(), 0.0
        for it, (x, y, g) in enumerate(ltr):
            x, y, g = x.to(dev), y.to(dev), g.to(dev)
            with torch.autocast(dev.type, enabled=a.amp):
                lo = model(x)
            lo = lo.float()
            if fis is not None:
                loss = fis(lo, y, g[:, fair_idx] if fair_idx is not None else None)
            else:
                loss = F.binary_cross_entropy_with_logits(lo, y, pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            if a.log_every and (it + 1) % a.log_every == 0:
                print(f"    step {it + 1}/{len(ltr)} loss={tot / (it + 1):.4f} {time.time() - t0:.0f}s", flush=True)
        yv, pv, _ = predict(model, lva, dev, a.amp, a.channels_last)
        m = binary_metrics(yv, pv)
        print(f"  ep{ep:02d} loss={tot / max(1, len(ltr)):.4f} val_auc={m['auc']:.4f} ({time.time() - t0:.0f}s)")
        if m["auc"] > best:
            best, best_ep = m["auc"], ep
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
            if hasattr(model, "enc"):
                torch.save({"enc": model.enc.state_dict(), "arch": a.arch}, os.path.join(out, "encoder.pt"))

    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=dev))
    yv, pv, gv = predict(model, lva, dev, a.amp, a.channels_last)
    yt, pt, gt = predict(model, lte, dev, a.amp, a.channels_last)
    np.savez(os.path.join(out, "preds.npz"), yv=yv, pv=pv, gv=gv, yt=yt, pt=pt, gt=gt, attrs=np.array(attrs))
    thr = pick_threshold(yv, pv, a.thr_strategy, a.target_sens)
    res = dict(seed=seed, best_epoch=best_ep, val_auc=best, test_at_0p5=binary_metrics(yt, pt, 0.5),
               test_at_val_thr=binary_metrics(yt, pt, thr))
    if a.task == "fairvision":
        res["fairness"] = {}
        for j, name in enumerate(attrs):
            if (gt[:, j] >= 0).all():
                res["fairness"][name] = fairness_report(yt, pt, gt[:, j], thr)
    dump_json(res, os.path.join(out, "results.json"))
    print(f"[seed {seed}] TEST auc={res['test_at_0p5']['auc']:.4f} bacc@thr={res['test_at_val_thr']['bacc']:.4f}")
    return res


def main(argv=None):
    a = get_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dump_json(vars(a), os.path.join(a.out, "args.json"))
    results = [run_seed(a, s, dev) for s in a.seeds]
    summ = dict(test_auc=mean_std([r["test_at_0p5"] for r in results])["auc"],
                test_bacc_at_val_thr=mean_std([r["test_at_val_thr"] for r in results])["bacc"],
                val_auc=mean_std([{"v": r["val_auc"]} for r in results])["v"])
    if a.task == "fairvision":
        for name in results[0].get("fairness", {}):
            rs = [r["fairness"][name] for r in results if r["fairness"].get(name)]
            if rs:
                summ[f"fair_{name}"] = {k: v for k, v in mean_std([{k: v for k, v in r.items()
                                        if isinstance(v, float)} for r in rs]).items()}
    dump_json(summ, os.path.join(a.out, "summary.json"))
    print("SUMMARY (mean, std):", summ)


if __name__ == "__main__":
    main()
