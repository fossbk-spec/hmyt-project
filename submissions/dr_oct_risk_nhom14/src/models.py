"""Models and losses.

Classifier2D : any torchvision 2D backbone applied to every B-scan, features pooled over the K scans (MIL).
Net3D        : torchvision r3d_18 with a 1-channel stem (the paper uses a 3D ResNet from ACS convolutions;
               r3d_18 is a close, easy-to-run substitute - NOT identical).
FISLoss      : our reading of Fair Identity Scaling (Eq. 3-5 of the FairVision paper). The paper does not specify
               the OT estimator, the scaling of the weights or whether weights are detached; the choices are
               documented below and are assumptions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm


def _backbone(arch, pretrained, img_size, drop_path=0.0):
    w = "DEFAULT" if pretrained else None
    if arch.startswith("resnet"):
        m = getattr(tvm, arch)(weights=w)
        d = m.fc.in_features
        m.fc = nn.Identity()
    elif arch.startswith("densenet"):
        m = getattr(tvm, arch)(weights=w)
        d = m.classifier.in_features
        m.classifier = nn.Identity()
    elif arch.startswith("efficientnet"):
        m = getattr(tvm, arch)(weights=w)
        d = m.classifier[-1].in_features
        m.classifier = nn.Identity()
    elif arch.startswith("timm:"):  # e.g. timm:vit_base_patch16_224.mae  (needs `pip install timm`)
        import timm
        m = timm.create_model(arch[5:], pretrained=pretrained, num_classes=0, drop_path_rate=drop_path,
                              img_size=img_size)
        d = m.num_features
    elif arch == "vit_b_16":
        if pretrained and img_size != 224:
            raise ValueError("pretrained vit_b_16 needs --size 224")
        m = tvm.vit_b_16(weights=w) if pretrained else tvm.vit_b_16(weights=None, image_size=img_size)
        d = m.heads.head.in_features
        m.heads = nn.Identity()
    else:
        raise ValueError(arch)
    return m, d


class Classifier2D(nn.Module):
    def __init__(self, arch="resnet18", pretrained=False, img_size=224, pool="mean", dropout=0.0,
                 drop_path=0.0, imagenet_norm=False):
        super().__init__()
        self.enc, d = _backbone(arch, pretrained, img_size, drop_path)
        self.inorm = imagenet_norm
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.pool = pool
        if pool == "attn":
            self.att = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, 1)
        self.feat_dim = d

    def forward(self, x):  # x: (B,1,H,W) or (B,K,1,H,W)
        if x.dim() == 4:
            x = x[:, None]
        B, K = x.shape[:2]
        xx = x.flatten(0, 1).repeat(1, 3, 1, 1)
        if self.inorm:
            xx = (xx - self.mean) / self.std
        f = self.enc(xx).view(B, K, -1)
        if self.pool == "attn":
            f = (torch.softmax(self.att(f), 1) * f).sum(1)
        elif self.pool == "max":
            f = f.max(1).values
        else:
            f = f.mean(1)
        return self.head(self.drop(f)).squeeze(-1)


class Net3D(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        m = tvm.video.r3d_18(weights="DEFAULT" if pretrained else None)
        old = m.stem[0]
        new = nn.Conv3d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        if pretrained:
            new.weight.data = old.weight.data.sum(1, keepdim=True)
        m.stem[0] = new
        m.fc = nn.Linear(m.fc.in_features, 1)
        self.m = m

    def forward(self, x):  # (B,1,D,H,W)
        return self.m(x).squeeze(-1)


def build_model(args):
    if args.mode == "3d":
        return Net3D(args.pretrained)
    return Classifier2D(args.arch, args.pretrained, args.size, args.pool, args.dropout,
                        getattr(args, "drop_path", 0.0), getattr(args, "imagenet_norm", False))


def w1_1d(a, b, n_q=64):
    """1-D Wasserstein-1 between two empirical loss distributions via quantile matching."""
    q = torch.linspace(0, 1, n_q, device=a.device)
    return (torch.quantile(a, q) - torch.quantile(b, q)).abs().mean()


class FISLoss(nn.Module):
    """L = mean_i[ ((1-c)*sI_i + c*sG_i) * l_i ]   (Eq. 3)
    sI = softmax_batch(l) * B                      (Eq. 4; x B so the mean weight is 1 - ASSUMPTION)
    sG = softmax_groups(OT(l_batch, l_group)) * |A| (Eq. 5; OT = in-batch 1-D W1 - ASSUMPTION)
    Weights are detached (treated as constants), as the paper describes them as loss weights."""

    def __init__(self, c=0.5, pos_weight=None):
        super().__init__()
        self.c = c
        self.pos_weight = pos_weight

    def forward(self, logits, y, g=None):
        l = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=self.pos_weight)
        B = l.numel()
        with torch.no_grad():
            sI = torch.softmax(l, 0) * B
            sG = torch.ones_like(l)
            if g is not None and self.c > 0:
                groups = [a for a in torch.unique(g) if (g == a).sum() >= 2]
                if len(groups) >= 2:
                    ot = torch.stack([w1_1d(l, l[g == a]) for a in groups])
                    wg = torch.softmax(ot, 0) * len(groups)
                    sG = torch.ones_like(l)
                    for a, w in zip(groups, wg):
                        sG[g == a] = w
        return (((1 - self.c) * sI + self.c * sG) * l).mean()


def make_loss(args, pos_weight=None):
    if args.loss == "fis":
        return FISLoss(args.fis_c, pos_weight)
    return None  # plain BCE handled in train.py
