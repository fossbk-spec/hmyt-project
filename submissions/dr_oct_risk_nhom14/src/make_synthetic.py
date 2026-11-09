"""Create TINY fake data in the (assumed) formats, only to smoke-test the code:
  python make_synthetic.py --out synth
  -> synth/FairVision_DR/{Training,Validation,Test}/*.npz   synth/olives/imgs/*.png + synth/olives/labels.csv"""
import argparse, os
import numpy as np, pandas as pd
from PIL import Image

def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", default="synth"); p.add_argument("--n", type=int, default=60)
    a = p.parse_args(); rng = np.random.RandomState(0)
    for split, n in (("Training", a.n), ("Validation", a.n // 4), ("Test", a.n // 2)):
        d = f"{a.out}/FairVision_DR/{split}"; os.makedirs(d, exist_ok=True)
        for i in range(n):
            y = int(rng.rand() < 0.3); v = (rng.rand(32, 64, 64) * 120).astype(np.uint8)
            if y: v[:, 20:40, 20:40] += 80            # a bright "lesion" for positives
            np.savez(f"{d}/data_{split[:2]}{i:04d}.npz", oct_bscans=v, dr_subtype=y, race=rng.randint(3),
                     male=rng.randint(2), hispanic=int(rng.rand() < 0.1))
    os.makedirs(f"{a.out}/olives/imgs", exist_ok=True); rows = []
    for eye in range(24):
        dis = "DME" if eye % 2 else "DR"
        for wk in (0, 8, 16):
            for s in range(3):
                im = (rng.rand(64, 96) * 120).astype(np.uint8)
                if dis == "DME": im[20:40, 30:60] += 80
                fn = f"eye{eye}_w{wk}_s{s}.png"; Image.fromarray(im).save(f"{a.out}/olives/imgs/{fn}")
                rows.append(dict(Path=fn, Eye_ID=eye, Patient_ID=eye, Week=wk, BCVA=70, CST=300, Disease=dis))
    pd.DataFrame(rows).to_csv(f"{a.out}/olives/labels.csv", index=False)

if __name__ == "__main__":
    main()
