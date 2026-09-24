"""ICDR grader: ImageNet-pretrained CNN backbone + ordinal CORN head.

The CORN head (Shi, Cao & Raschka, 2023) predicts K-1 = 4 conditional
probabilities P(y > k | y > k-1). Chaining them gives cumulative
probabilities that are monotone by construction, so P(grade >= 2), the
referral probability, is a direct output rather than a sum of softmax
entries that can contradict each other.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 5

# RETFound (Zhou et al., Nature 2023): ViT-Large encoders pretrained on
# retinal photographs. The weights are gated on Hugging Face and are loaded
# from local files, so training never depends on the network. Used only as a
# teacher: a ViT does not import cleanly into MATLAB, so what ships is an
# EfficientNet distilled from it (train.py --teacher).
RETFOUND = {
    "retfound_mae": ("vit_large_patch16_224", "RETFound_mae_natureCFP.pth"),
    "retfound_dinov2": ("vit_large_patch14_dinov2", "RETFound_dinov2_meh.pth"),
}
RETFOUND_DIR = Path(os.environ.get("DRISHTI_MODELS", Path(__file__).resolve().parents[3] / "models")) / "retfound"


def _load_retfound(body: nn.Module, fname: str) -> None:
    from timm.layers import resample_abs_pos_embed
    ck = torch.load(RETFOUND_DIR / fname, map_location="cpu", weights_only=False)
    for key in ("model", "teacher", "state_dict"):
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            ck = ck[key]
            break
    sd = {}
    for k, v in ck.items():
        for pre in ("module.", "backbone.", "encoder."):
            if k.startswith(pre):
                k = k[len(pre):]
        if k.startswith(("decoder", "mask_token", "head.")):
            continue  # MAE decoder / pretraining head
        sd[k] = v
    own = body.state_dict()
    if "pos_embed" in sd and sd["pos_embed"].shape != own["pos_embed"].shape:
        n_pre = getattr(body, "num_prefix_tokens", 1)
        grid = body.patch_embed.grid_size
        sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], new_size=list(grid), num_prefix_tokens=n_pre)
    missing, unexpected = body.load_state_dict(sd, strict=False)
    matched = len(own) - len(missing)
    print(f"RETFound {fname}: {matched}/{len(own)} tensors loaded; "
          f"missing {missing[:4]}{'...' if len(missing) > 4 else ''}; unexpected {unexpected[:4]}")
    if matched < 0.9 * len(own):
        raise RuntimeError(f"RETFound checkpoint {fname} matched only {matched}/{len(own)} tensors")


class Grader(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, drop: float = 0.3,
                 img_size: int | None = None):
        super().__init__()
        self.backbone_name = backbone
        if backbone in RETFOUND:
            arch, fname = RETFOUND[backbone]
            self.body = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="",
                                          img_size=img_size or 224, drop_path_rate=0.2)
            if pretrained:
                _load_retfound(self.body, fname)
        else:
            self.body = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        feat = self.body.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(feat, N_CLASSES - 1)
        self.register_buffer("temperature", torch.ones(1))

    def features(self, x):
        f = self.body(x)  # (N, C, h, w) from a CNN: used by Grad-CAM
        if f.dim() == 3:  # ViT tokens (N, L, C): drop class/register tokens, lay out the patch grid
            f = f[:, getattr(self.body, "num_prefix_tokens", 1):]
            n, l, c = f.shape
            h = int(math.isqrt(l))
            f = f.transpose(1, 2).reshape(n, c, h, l // h)
        return f

    def logits_from_features(self, f):
        return self.head(self.drop(self.pool(f).flatten(1)))

    def forward(self, x):
        return self.logits_from_features(self.features(x))


def cumulative(logits: torch.Tensor) -> torch.Tensor:
    """P(y > k) for k = 0..3, monotone non-increasing in k.

    Written as an explicit chain of products rather than torch.cumprod,
    which has no ONNX export (and so could not reach MATLAB)."""
    s = torch.sigmoid(logits)
    cols = [s[:, :1]]
    for k in range(1, s.shape[1]):
        cols.append(cols[-1] * s[:, k:k + 1])
    return torch.cat(cols, dim=1)


def class_probs(logits: torch.Tensor) -> torch.Tensor:
    c = cumulative(logits)
    ones = torch.ones_like(c[:, :1])
    zeros = torch.zeros_like(c[:, :1])
    upper = torch.cat([ones, c], 1)  # P(y > k-1)
    lower = torch.cat([c, zeros], 1)  # P(y > k)
    return (upper - lower).clamp_min(0)


def referable_prob(logits: torch.Tensor) -> torch.Tensor:
    return cumulative(logits)[:, 1]  # P(y > 1) = P(grade >= 2)


def corn_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Task k is trained only on samples with y >= k (i.e. y > k-1)."""
    losses, count = 0.0, 0
    for k in range(N_CLASSES - 1):
        sel = y >= k
        if sel.sum() == 0:
            continue
        target = (y[sel] > k).float()
        losses = losses + F.binary_cross_entropy_with_logits(logits[sel, k], target, reduction="sum")
        count += int(sel.sum())
    return losses / max(count, 1)


def soft_corn_loss(logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """CORN against a teacher's class distribution instead of a hard grade.

    Task k models P(y > k | y > k-1). With hard labels it is trained on the
    samples that reach it (y >= k); with soft ones every sample contributes,
    weighted by the teacher's probability that it reaches task k, towards the
    teacher's conditional probability P(y > k) / P(y > k-1). A one-hot
    teacher reduces this exactly to corn_loss."""
    cum = torch.flip(torch.cumsum(torch.flip(probs, [1]), 1), [1])  # P(y >= k), k = 0..4
    reach, beyond = cum[:, :-1], cum[:, 1:]  # P(y > k-1), P(y > k) for k = 0..3
    target = (beyond / reach.clamp_min(1e-6)).clamp(0, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (reach * bce).sum() / reach.sum().clamp_min(1e-6)


class ExportWrapper(nn.Module):
    """Graph exported to ONNX / MATLAB: image -> (class probabilities, P(referable))."""

    def __init__(self, model: Grader):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x) / self.model.temperature
        return class_probs(logits), referable_prob(logits)
