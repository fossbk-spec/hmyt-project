"""CNN 1D end-to-end trên tín hiệu 12 chuyển đạo (100 Hz, 10 s).
Chạy: python src/train_cnn.py --data data/ptb-xl --epochs 30      (--synthetic để kiểm thử đường ống)
Chống rò rỉ: strat_fold chính thức; chuẩn hóa fit trên train; chọn epoch/ngưỡng trên val; test đánh giá 1 lần."""
import argparse, os, random
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from common import SEED, check_no_patient_overlap, fit_norm, finalize, get_data, split_masks


def blk(i, o, k):
    return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))


class Net(nn.Module):
    def __init__(self, c=12):
        super().__init__()
        self.blocks = nn.Sequential(blk(c, 32, 7), blk(32, 64, 7), blk(64, 128, 5), blk(128, 128, 5))
        self.pool, self.fc = nn.AdaptiveAvgPool1d(1), nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))

    def features(self, x): return self.blocks(x)                       # (B, 128, T/16) — dùng cho Grad-CAM
    def forward(self, x): return self.fc(self.pool(self.features(x)).flatten(1)).squeeze(-1)


@torch.no_grad()
def predict(m, X, dev, bs=256):
    m.eval()
    return np.concatenate([torch.sigmoid(m(torch.from_numpy(X[i:i + bs]).to(dev))).cpu().numpy() for i in range(0, len(X), bs)])


def prep(X, mu, sd): return np.ascontiguousarray(((X - mu) / sd).transpose(0, 2, 1), dtype=np.float32)   # (N, 12, T)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", default="data/ptb-xl"); a.add_argument("--out", default="results")
    a.add_argument("--epochs", type=int, default=30); a.add_argument("--seed", type=int, default=SEED)
    a.add_argument("--synthetic", action="store_true")
    a.add_argument("--no_aug", action="store_true", help="ablation: tắt tăng cường dữ liệu")
    a.add_argument("--name", default="cnn", help="tên mô hình (dùng cho ablation, vd cnn_noaug)")
    a = a.parse_args(); os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    X, meta = get_data(a); check_no_patient_overlap(meta)
    tr, va, te = split_masks(meta); y = meta.y.values
    mu, sd = fit_norm(X[tr]); Xn = prep(X, mu, sd)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("thiết bị:", dev)

    m = Net().to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    pw = torch.tensor((1 - y[tr].mean()) / y[tr].mean(), dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt, yt = torch.from_numpy(Xn[tr]), torch.from_numpy(y[tr].astype(np.float32))
    best, state = -1, None
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(len(Xt)); tot = 0
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]; xb = Xt[idx].to(dev); yb = yt[idx].to(dev)
            if not a.no_aug:
                xb = xb * (1 + 0.1 * (2 * torch.rand(len(xb), 1, 1, device=dev) - 1))      # tăng cường: biên độ ±10%
                xb = xb + 0.02 * torch.randn_like(xb)                                       # nhiễu Gaussian
                xb = torch.roll(xb, int(np.random.randint(-50, 51)), dims=2)                # dịch thời gian ±0.5 s
            opt.zero_grad(); loss = lossf(m(xb), yb); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        auc = roc_auc_score(y[va], predict(m, Xn[va], dev))
        print(f"epoch {ep + 1:02d}  loss={tot / len(Xt):.4f}  val AUROC={auc:.4f}", flush=True)
        if auc > best: best, state = auc, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    torch.save({"state": state, "mu": mu, "sd": sd}, f"{a.out}/{a.name}_model.pt")
    finalize(a.name, y[va], predict(m, Xn[va], dev), y[te], predict(m, Xn[te], dev), np.where(te)[0], a.out)


if __name__ == "__main__":
    main()
