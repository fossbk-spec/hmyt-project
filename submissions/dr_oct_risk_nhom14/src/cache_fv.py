"""Pre-process FairVision-DR ONCE into small uint8 arrays (fast training afterwards).
  python cache_fv.py --root /content/fv/DR --out /content/drive/MyDrive/oct/cache/3d_d64_s128 --mode 3d --depth 64 --size 128
  python cache_fv.py --root ... --out .../2d_k7_s224 --mode 2d_mil --k_slices 7 --size 224
  python cache_fv.py --root ... --out .../slo_s224   --mode slo --size 224
Use the SAME key arguments you found with `python data.py inspect_fairvision`."""
import argparse, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import FairVisionDR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["3d", "2d_mil", "slo"], required=True)
    p.add_argument("--oct_key", default="oct_bscans"); p.add_argument("--slo_key", default="slo_fundus")
    p.add_argument("--label_key", default="dr_subtype"); p.add_argument("--attrs", default="race,male,hispanic")
    p.add_argument("--size", type=int, default=224); p.add_argument("--depth", type=int, default=64)
    p.add_argument("--k_slices", type=int, default=7); p.add_argument("--slice_axis", type=int, default=0)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--max_n", type=int, default=None)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for split in ("train", "val", "test"):
        if os.path.exists(f"{a.out}/{split}_a.npy"):
            print(split, "already cached"); continue
        ds = FairVisionDR(a.root, split, a.mode, a.oct_key, a.label_key, a.attrs.split(","), a.size, a.depth,
                          a.k_slices, a.slice_axis, False, a.max_n, a.slo_key)
        N = len(ds); x0 = ds[0][0]
        X = np.lib.format.open_memmap(f"{a.out}/{split}_x.npy", "w+", np.uint8, (N, *x0.shape))
        Y = np.zeros(N, np.int64); A = np.zeros((N, len(a.attrs.split(","))), np.int64)
        for i, (x, y, at) in enumerate(DataLoader(ds, 1, num_workers=a.workers)):
            X[i] = (x[0] * 255).round().byte().numpy(); Y[i] = int(y[0]); A[i] = at[0].numpy()
            if i % 500 == 0: print(split, i, "/", N, flush=True)
        if not set(np.unique(Y)) <= {0, 1}:
            raise ValueError(f"labels are not binary: {np.unique(Y, return_counts=True)} - check --label_key / mapping")
        X.flush(); np.save(f"{a.out}/{split}_y.npy", Y)
        np.save(f"{a.out}/{split}_a.npy", A)          # written last = completion marker
        print(split, "done", X.shape, "positive rate", Y.mean().round(3))

if __name__ == "__main__":
    main()
