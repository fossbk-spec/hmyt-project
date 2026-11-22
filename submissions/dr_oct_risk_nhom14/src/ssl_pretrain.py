"""SimCLR-style self-supervised pre-training of a 2D encoder on B-scans (no labels used).
  python ssl_pretrain.py --task fairvision --root FairVision/DR --arch resnet18 --epochs 50 --out runs/ssl_fv
Then fine-tune/probe on the other dataset with: train.py ... --init_enc runs/ssl_fv/encoder.pt
OCT-specific choices: no flips of colour/hue (grey images); crops + blur + small brightness/contrast only."""
import argparse, os
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from data import FairVisionDR, OlivesBscan, group_split, load_olives_df
from models import _backbone
from utils import seed_everything


class TwoView(Dataset):
    def __init__(self, base, size):
        self.base = base
        self.tf = T.Compose([T.RandomResizedCrop(size, scale=(0.4, 1.0)), T.ColorJitter(0.3, 0.3),
                             T.GaussianBlur(5, (0.1, 1.5))])
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x = self.base[i][0]
        if x.dim() == 4: x = x[0]            # fairvision: one random B-scan (augment=True picks random slices)
        return self.tf(x), self.tf(x)


def nt_xent(z1, z2, tau=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e9)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["fairvision", "olives"], required=True)
    p.add_argument("--root", required=True); p.add_argument("--csv")
    p.add_argument("--path_col", default="Path"); p.add_argument("--label_col", default="Disease")
    p.add_argument("--patient_col", default="Eye_ID")
    p.add_argument("--oct_key", default="oct_bscans"); p.add_argument("--label_key", default="dr_subtype")
    p.add_argument("--arch", default="resnet18"); p.add_argument("--size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=50); p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_n", type=int, default=None); p.add_argument("--out", default="runs/ssl")
    a = p.parse_args()
    seed_everything(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.task == "fairvision":   # only TRAIN split volumes -> no test leakage
        base = FairVisionDR(a.root, "train", mode="2d_mil", oct_key=a.oct_key, label_key=a.label_key, attrs=(),
                            size=a.size, k_slices=1, augment=True, max_n=a.max_n)
    else:
        df = load_olives_df(a.csv, a.path_col, a.label_col, a.patient_col)
        base = OlivesBscan(group_split(df, a.patient_col, seed=0)[0], a.root, a.path_col, a.size)
    dl = DataLoader(TwoView(base, a.size), a.bs, shuffle=True, drop_last=True, num_workers=a.workers)
    enc, d = _backbone(a.arch, False, a.size)
    proj = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 128))
    enc, proj = enc.to(dev), proj.to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=a.lr, weight_decay=1e-4)
    os.makedirs(a.out, exist_ok=True)
    for ep in range(a.epochs):
        tot = 0.0
        for v1, v2 in dl:
            v1, v2 = v1.to(dev).repeat(1, 3, 1, 1), v2.to(dev).repeat(1, 3, 1, 1)
            loss = nt_xent(proj(enc(v1)), proj(enc(v2)))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"ep{ep:03d} ntxent={tot / max(1, len(dl)):.4f}")
        torch.save({"enc": enc.state_dict(), "arch": a.arch}, os.path.join(a.out, "encoder.pt"))

if __name__ == "__main__":
    main()
