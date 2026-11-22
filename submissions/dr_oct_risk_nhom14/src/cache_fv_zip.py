"""Build ALL caches in ONE pass straight from dataset.zip (no 52 GB unzip, jpg/__MACOSX ignored).
Each npz is read once from the zip, then processed for every requested mode.
  python cache_fv_zip.py --zip /content/drive/MyDrive/OCT/FairVision_Dataset/dataset.zip --out /content/cache \
      --modes slo:224 3d:64x128 2d:7x224
mode specs:  slo:SIZE   |   3d:DEPTHxSIZE   |   2d:KxSIZE        (output dir name = mode spec, e.g. slo_224)
Resumable per split (a split is skipped if its *_a.npy exists). Bad zip members are listed, never silently dropped."""
import argparse, io, os, sys, zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from data import FairVisionDR, SPLIT_ALIASES


class ZipFV(FairVisionDR):
    def __init__(self, zip_path, names, mode, **kw):
        self.zip_path, self.files, self._zf = zip_path, names, None
        self.mode, self.oct_key, self.slo_key = mode, kw["oct_key"], kw["slo_key"]
        self.label_key, self.attrs = kw["label_key"], kw["attrs"]
        self.size, self.depth, self.k, self.axis, self.augment = kw["size"], kw.get("depth", 64), kw.get("k", 7), 0, False

    def _open(self, i):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
        return np.load(io.BytesIO(self._zf.read(self.files[i])), allow_pickle=True)


class Multi(Dataset):
    def __init__(self, dss): self.dss = dss
    def __len__(self): return len(self.dss[0])
    def __getitem__(self, i):
        d = self.dss[0]._open(i)
        return [ds.from_npz(d) for ds in self.dss]


def parse(spec):
    kind, v = spec.split(":")
    if kind == "slo": return dict(mode="slo", size=int(v)), f"slo_{v}"
    if kind == "3d": dp, sz = v.split("x"); return dict(mode="3d", depth=int(dp), size=int(sz)), f"3d_d{dp}_s{sz}"
    if kind == "2d": k, sz = v.split("x"); return dict(mode="2d_mil", k=int(k), size=int(sz)), f"2d_k{k}_s{sz}"
    raise ValueError(spec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True); p.add_argument("--out", required=True)
    p.add_argument("--modes", nargs="+", default=["slo:224", "3d:64x128"])
    p.add_argument("--oct_key", default="oct_bscans"); p.add_argument("--slo_key", default="slo_fundus")
    p.add_argument("--label_key", default="dr_subtype"); p.add_argument("--attrs", default="race,male,hispanic")
    p.add_argument("--workers", type=int, default=2); p.add_argument("--max_n", type=int, default=None)
    a = p.parse_args()
    attrs = a.attrs.split(",")
    zf = zipfile.ZipFile(a.zip)
    npz = sorted(n for n in zf.namelist() if n.endswith(".npz") and not n.startswith("__MACOSX") and "/._" not in n)
    print(len(npz), "npz members in zip")
    cfgs = [parse(m) for m in a.modes]
    for split in ("train", "val", "test"):
        names = [n for n in npz if n.split("/")[0].lower() in SPLIT_ALIASES[split]]
        if a.max_n: names = names[: a.max_n]
        print(f"{split}: {len(names)} files")
        if not names: sys.exit(f"no members for split {split}; zip top folders: {sorted({n.split('/')[0] for n in npz})}")
        if all(os.path.exists(f"{a.out}/{d}/{split}_a.npy") for _, d in cfgs):
            print(split, "already cached"); continue
        kw = dict(oct_key=a.oct_key, slo_key=a.slo_key, label_key=a.label_key, attrs=attrs)
        dss = [ZipFV(a.zip, names, **c, **kw) for c, _ in cfgs]
        Xs, Y, A, bad = [None] * len(cfgs), np.zeros(len(names), np.int64), np.zeros((len(names), len(attrs)), np.int64), []
        for i, item in enumerate(DataLoader(Multi(dss), batch_size=None, num_workers=a.workers, shuffle=False)):
            for j, (x, y, at) in enumerate(item):
                if Xs[j] is None:
                    os.makedirs(f"{a.out}/{cfgs[j][1]}", exist_ok=True)
                    Xs[j] = np.lib.format.open_memmap(f"{a.out}/{cfgs[j][1]}/{split}_x.npy", "w+", np.uint8, (len(names), *x.shape))
                Xs[j][i] = (x * 255).round().byte().numpy()
            Y[i] = int(item[0][1]); A[i] = item[0][2].numpy()
            if i % 500 == 0: print(split, i, "/", len(names), flush=True)
        if not set(np.unique(Y)) <= {0, 1}: sys.exit(f"labels not binary: {np.unique(Y, return_counts=True)}")
        for j, (_, d) in enumerate(cfgs):
            Xs[j].flush(); np.save(f"{a.out}/{d}/{split}_y.npy", Y)
            with open(f"{a.out}/{d}/{split}_names.txt", "w") as f: f.write("\n".join(names))
            np.save(f"{a.out}/{d}/{split}_a.npy", A)          # written last = completion marker
        print(f"{split} done: n={len(Y)} positive rate={Y.mean():.4f} | race counts={np.bincount(A[:, 0])} "
              f"male={np.bincount(A[:, 1])} hispanic={np.bincount(A[:, 2])}")

if __name__ == "__main__":
    main()
