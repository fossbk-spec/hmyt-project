"""Loaders for Harvard-FairVision (DR) and OLIVES.

IMPORTANT: the exact npz keys (FairVision) and csv column names (OLIVES) were NOT verified by the author
of this code (the datasets could not be downloaded in the sandbox). Run

    python data.py inspect_fairvision --root /path/FairVision/DR
    python data.py inspect_olives --csv /path/labels.csv

first and pass the right names through --oct_key/--label_key/--attrs or --path_col/--label_col/...
"""
import argparse
import glob
import hashlib
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

SPLIT_ALIASES = {"train": ("train", "training"), "val": ("val", "validation", "valid"), "test": ("test", "testing")}
STR_MAPS = {"asian": 0, "black": 1, "white": 2, "male": 1, "female": 0, "hispanic": 1, "non-hispanic": 0}


def _to_int(v):
    v = np.asarray(v).reshape(-1)[0]
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, (str, np.str_)):
        s = str(v).strip().lower()
        if s in STR_MAPS:
            return STR_MAPS[s]
        return int(float(s))
    return int(v)


def _label_from(v):
    """Binary DR label. FairVision card: dr_subtype is a 6-class string mapped to 0 = non-vision-threatening,
    1 = vision-threatening (severe NPDR or PDR). Numeric 0/1 values are passed through unchanged."""
    v = np.asarray(v).reshape(-1)[0]
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, (str, np.str_)):
        s = str(v).strip().lower()
        try:
            return int(float(s))
        except ValueError:
            pass
        return int("severe" in s or s == "pdr" or ("proliferative" in s and "non" not in s))
    return int(v)


def _unit(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-6)


# --------------------------------------------------------------------------- FairVision
def find_npz(root, split):
    allf = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
    alias_all = sum(SPLIT_ALIASES.values(), ())

    def parts(f):
        return [p.lower() for p in os.path.relpath(os.path.dirname(f), root).split(os.sep)]

    sel = [f for f in allf if any(p in SPLIT_ALIASES[split] for p in parts(f))]
    if sel:
        return sel
    if any(p in alias_all for f in allf for p in parts(f)):  # split folders exist but this one is empty
        return []
    # fallback: deterministic 60/10/30 split by file name (same proportions as the paper: 6000/1000/3000)
    def bucket(f):
        h = int(hashlib.md5(os.path.basename(f).encode()).hexdigest(), 16) % 100
        return "train" if h < 60 else "val" if h < 70 else "test"

    return [f for f in allf if bucket(f) == split]


class FairVisionDR(Dataset):
    """One item = one subject (one OCT volume). mode:
       '3d'     -> (1, depth, size, size) volume for a 3D CNN
       '2d_mil' -> (k_slices, 1, size, size) central B-scans, aggregated inside the model
    Returns (x, y, attrs) where attrs is a LongTensor over `attrs` keys (-1 if missing)."""

    def __init__(self, root, split, mode="2d_mil", oct_key="oct_bscans", label_key="dr_subtype",
                 attrs=("race", "male", "hispanic"), size=224, depth=64, k_slices=7, slice_axis=0,
                 augment=False, max_n=None, slo_key="slo_fundus"):
        self.slo_key = slo_key
        self.files = find_npz(root, split)
        if max_n:
            self.files = self.files[:max_n]
        if not self.files:
            raise FileNotFoundError(f"No npz found for split={split} under {root}")
        self.mode, self.oct_key, self.label_key, self.attrs = mode, oct_key, label_key, list(attrs)
        self.size, self.depth, self.k, self.axis, self.augment = size, depth, k_slices, slice_axis, augment

    def __len__(self):
        return len(self.files)

    def labels(self):  # lazy: reads only the label array of each npz
        return np.array([_label_from(np.load(f, allow_pickle=True)[self.label_key]) for f in self.files])

    def _open(self, i):
        return np.load(self.files[i], allow_pickle=True)

    def __getitem__(self, i):
        return self.from_npz(self._open(i))

    def from_npz(self, d):
        data_key = self.slo_key if self.mode == "slo" else self.oct_key
        if data_key not in d.files or self.label_key not in d.files:
            raise KeyError(f"keys {data_key}/{self.label_key} not in {d.files}; use data.py inspect_fairvision")
        if self.mode == "slo":  # 2D SLO fundus image -> (1,size,size)
            arr = np.squeeze(d[data_key]).astype(np.float32)
            if arr.ndim == 3:  # colour -> grey
                arr = arr.mean(axis=0 if arr.shape[0] in (1, 3) else -1)
            x = F.interpolate(torch.from_numpy(_unit(arr))[None, None], size=(self.size, self.size),
                              mode="bilinear", align_corners=False)[0]
            t = None
        else:
            vol = d[data_key]
            if self.axis != 0:
                vol = np.moveaxis(vol, self.axis, 0)
            t = torch.from_numpy(_unit(vol))  # (D,H,W)
        if self.mode == "slo":
            pass
        elif self.mode == "3d":
            x = F.interpolate(t[None, None], size=(self.depth, self.size, self.size), mode="trilinear",
                              align_corners=False)[0]
        else:
            D = t.shape[0]
            lo, hi = int(0.2 * D), max(int(0.8 * D), int(0.2 * D) + 1)
            if self.augment:
                idx = np.sort(np.random.choice(np.arange(lo, hi), size=min(self.k, hi - lo), replace=False))
            else:
                idx = np.linspace(lo, hi - 1, self.k).round().astype(int)
            x = F.interpolate(t[idx][:, None], size=(self.size, self.size), mode="bilinear", align_corners=False)
        if self.augment:
            x = (x * np.random.uniform(0.9, 1.1)).clamp(0, 1)
        y = _label_from(d[self.label_key])
        a = torch.tensor([_to_int(d[k]) if k in d.files else -1 for k in self.attrs], dtype=torch.long)
        return x, torch.tensor(y, dtype=torch.float32), a


class CachedFV(Dataset):
    """Reads arrays written by cache_fv.py (uint8 memmap). Fast: no npz decoding / resizing at train time."""

    def __init__(self, cache_dir, split, augment=False, max_n=None):
        self.x = np.load(os.path.join(cache_dir, f"{split}_x.npy"), mmap_mode="r")
        self.y = np.load(os.path.join(cache_dir, f"{split}_y.npy"))
        self.a = np.load(os.path.join(cache_dir, f"{split}_a.npy"))
        self.n = min(len(self.y), max_n) if max_n else len(self.y)
        self.augment = augment

    def __len__(self):
        return self.n

    def labels(self):
        return self.y[: self.n].astype(int)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.x[i])).float() / 255.0
        if self.augment:
            x = (x * np.random.uniform(0.9, 1.1)).clamp(0, 1)
        return x, torch.tensor(float(self.y[i])), torch.from_numpy(self.a[i].astype(np.int64))


def inspect_fairvision(root, n=200):
    files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
    print(f"{len(files)} npz files under {root}")
    for s in SPLIT_ALIASES:
        print(f"  split {s}: {len(find_npz(root, s))} files")
    if not files:
        return
    d = np.load(files[0], allow_pickle=True)
    print("keys of first file:")
    for k in d.files:
        a = d[k]
        print(f"  {k:20s} shape={a.shape} dtype={a.dtype}")
    for k in d.files:
        if d[k].size == 1:
            vals = [str(np.load(f, allow_pickle=True)[k].reshape(-1)[0]) for f in files[:n]]
            u, c = np.unique(vals, return_counts=True)
            if len(u) <= 12:
                print(f"  values of {k}: {dict(zip(u, c))}")


# --------------------------------------------------------------------------- OLIVES
def load_olives_df(csv, path_col, label_col, patient_col, eye_col=None, week_col=None, first_last=False,
                   pos_value=None):
    df = pd.read_csv(csv)
    for c in (path_col, label_col, patient_col):
        if c not in df.columns:
            raise KeyError(f"column {c} not in {list(df.columns)}; use data.py inspect_olives")
    if first_last and eye_col and week_col:  # paper: biomarkers only at first and last visit of each eye
        g = df.groupby(eye_col)[week_col]
        df = df[(df[week_col] == g.transform("min")) | (df[week_col] == g.transform("max"))]
    lab = df[label_col]
    if lab.dtype == object:
        classes = sorted(lab.unique())
        pv = pos_value or classes[-1]
        df = df.assign(_y=(lab == pv).astype(int))
        print(f"label '{label_col}': positive='{pv}' of {classes}")
    else:
        df = df.assign(_y=(lab > 0).astype(int))
    return df.reset_index(drop=True)


def group_split(df, group_col, ratios=(0.7, 0.1, 0.2), seed=0):
    """Split by patient/eye (never by image) so that B-scans of one eye never leak across splits."""
    rng = np.random.RandomState(seed)
    for _ in range(200):
        groups = df[group_col].unique()
        rng.shuffle(groups)
        n = len(groups)
        a, b = int(ratios[0] * n), int((ratios[0] + ratios[1]) * n)
        sets = [set(groups[:a]), set(groups[a:b]), set(groups[b:])]
        parts = [df[df[group_col].isin(s)] for s in sets]
        if all(p["_y"].nunique() == 2 for p in parts):
            return parts
    raise RuntimeError("could not find a split where every part contains both classes")


class OlivesBscan(Dataset):
    def __init__(self, df, img_root, path_col, size=224, augment=False):
        self.df, self.root, self.path_col, self.size = df.reset_index(drop=True), img_root, path_col, size
        self._index = None
        self.tf = T.Compose([T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                             T.ColorJitter(brightness=0.15, contrast=0.15)]) if augment else None

    def _resolve(self, p):
        p = str(p).replace("\\", "/").lstrip("/")
        full = os.path.join(self.root, p)
        if os.path.exists(full):
            return full
        if self._index is None:  # fall back to a basename index
            self._index = {os.path.basename(f): os.path.join(r, f) for r, _, fs in os.walk(self.root) for f in fs}
        return self._index[os.path.basename(p)]

    def labels(self):
        return self.df["_y"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(self._resolve(r[self.path_col])).convert("L").resize((self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]
        if self.tf is not None:
            x = self.tf(x)
        return x, torch.tensor(float(r["_y"])), torch.zeros(1, dtype=torch.long)


def inspect_olives(csv):
    df = pd.read_csv(csv)
    print(df.shape)
    print(df.head(3).T)
    for c in df.columns:
        if df[c].nunique() <= 12:
            print(f"{c}: {df[c].value_counts().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect_fairvision", "inspect_olives"])
    ap.add_argument("--root")
    ap.add_argument("--csv")
    a = ap.parse_args()
    inspect_fairvision(a.root) if a.cmd == "inspect_fairvision" else inspect_olives(a.csv)
