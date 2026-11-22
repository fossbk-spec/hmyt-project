"""Random search over hyper-parameters / preprocessing. Selection by VALIDATION AUC only (never test).
  python sweep.py --trials 12 --out runs/sweep_fv -- --task fairvision --root FairVision/DR --epochs 10
Everything after `--` is passed to train.py as the fixed part; the sampled part overrides/extends it."""
import argparse, csv, json, os, random, subprocess, sys

SPACE = dict(
    **{"--lr": [3e-5, 1e-4, 3e-4], "--wd": [1e-3, 1e-2, 5e-2], "--size": [160, 224, 256],
       "--k_slices": [3, 5, 7, 11], "--arch": ["resnet18", "resnet50", "efficientnet_b0", "densenet121"],
       "--pool": ["mean", "attn", "max"], "--balance": ["none", "sampler", "posweight"],
       "--loss": ["ce", "fis"], "--dropout": [0.0, 0.2]})

def main():
    argv = sys.argv[1:]
    fixed = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(); p.add_argument("--trials", type=int, default=10)
    p.add_argument("--out", default="runs/sweep"); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    rng = random.Random(a.seed); rows = []
    os.makedirs(a.out, exist_ok=True)
    for t in range(a.trials):
        cfg = {k: rng.choice(v) for k, v in SPACE.items()}
        cmd = [sys.executable, "train.py", *fixed, *[x for k, v in cfg.items() for x in (k, str(v))],
               "--out", f"{a.out}/t{t}", "--seeds", "0"]
        print("RUN", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode != 0: continue
        r = json.load(open(f"{a.out}/t{t}/seed0/results.json"))
        rows.append(dict(trial=t, val_auc=r["val_auc"], **cfg))
    rows.sort(key=lambda r: -r["val_auc"])
    with open(f"{a.out}/sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("BEST (by val AUC):", rows[0])

if __name__ == "__main__":
    main()
