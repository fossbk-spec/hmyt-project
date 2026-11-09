"""Params / FLOPs / latency benchmark with the optimisations that matter for OCT: fp16/bf16, channels_last,
torch.compile, fewer slices K, smaller input size.
  python latency.py --arch resnet18 --size 224 --k 7 --batch 1 8 --precision fp32 fp16 --channels_last --compile
Report latency PER SUBJECT (all K slices) because that is what a screening system pays."""
import argparse, json, time
import torch
from torch.utils.flop_counter import FlopCounterMode
from models import Classifier2D, Net3D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="resnet18"); p.add_argument("--mode", default="2d_mil")
    p.add_argument("--size", type=int, default=224); p.add_argument("--k", type=int, default=7)
    p.add_argument("--depth", type=int, default=64)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    p.add_argument("--precision", nargs="+", default=["fp32"], choices=["fp32", "fp16", "bf16"])
    p.add_argument("--channels_last", action="store_true"); p.add_argument("--compile", action="store_true")
    p.add_argument("--iters", type=int, default=30); p.add_argument("--out", default=None)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = (Net3D() if a.mode == "3d" else Classifier2D(a.arch, False, a.size)).to(dev).eval()
    if a.channels_last and a.mode != "3d": m = m.to(memory_format=torch.channels_last)
    if a.compile: m = torch.compile(m)
    nparam = sum(q.numel() for q in m.parameters()) / 1e6
    rows = []
    for prec in a.precision:
        if dev.type == "cpu" and prec == "fp16": continue
        dt = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[prec]
        for B in a.batch:
            x = torch.rand(B, 1, a.depth, a.size, a.size, device=dev) if a.mode == "3d" else \
                torch.rand(B, a.k, 1, a.size, a.size, device=dev)
            with torch.no_grad(), torch.autocast(dev.type, dtype=dt, enabled=dt is not None):
                for _ in range(5): m(x)
                if dev.type == "cuda": torch.cuda.synchronize()
                ts = []
                for _ in range(a.iters):
                    t0 = time.perf_counter(); m(x)
                    if dev.type == "cuda": torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
            ts.sort(); med = ts[len(ts) // 2]
            rows.append(dict(precision=prec, batch=B, ms_per_batch=round(med, 2), ms_per_subject=round(med / B, 2)))
    try:
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(torch.rand(1, 1, a.depth, a.size, a.size, device=dev) if a.mode == "3d" else
              torch.rand(1, a.k, 1, a.size, a.size, device=dev))
        gflops = fc.get_total_flops() / 1e9
    except Exception:
        gflops = float("nan")
    res = dict(arch=a.arch if a.mode != "3d" else "r3d_18", params_M=round(nparam, 2), GFLOPs_per_subject=round(gflops, 1),
               size=a.size, k=a.k, device=str(dev), rows=rows)
    print(json.dumps(res, indent=1))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
