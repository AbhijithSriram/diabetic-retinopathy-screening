"""Train the multi-label lesion + optic-disc U-Net.

One decoder, five independent sigmoid outputs (MA, HE, EX, SE, OD). The
"common lesions crowd out rare ones" failure comes from softmax competition
between classes; independent sigmoids plus a per-channel loss avoid it
without training a separate network per lesion.

Loss per channel: BCE + Tversky (alpha=0.3, beta=0.7, so missed lesion
pixels cost more than false alarms). Channels a dataset does not label
(optic disc on DDR) are masked out.

    python -m drishti.lesions.train --name seg_r34 --iters 8000
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import RUNS, SEED, SEG_CHANNELS
from .data import SegCrops, load_seg_manifest, precompute


def build_model(encoder: str = "resnet34", pretrained: bool = True):
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, classes=len(SEG_CHANNELS), decoder_attention_type="scse")


def tversky(prob, target, alpha=0.3, beta=0.7, eps=1.0):
    dims = (0, 2, 3)
    tp = (prob * target).sum(dims)
    fp = (prob * (1 - target)).sum(dims)
    fn = ((1 - prob) * target).sum(dims)
    return 1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)  # per channel


def seg_loss(logits, target, known):
    # known: (B, C) -> broadcast mask over pixels
    km = known[:, :, None, None]
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * km).sum((0, 2, 3)) / (km.sum((0, 2, 3)) * logits.shape[2] * logits.shape[3]).clamp_min(1)
    prob = torch.sigmoid(logits) * km
    tv = tversky(prob, target * km)
    ch_known = known.sum(0) > 0
    per_ch = (bce + tv)[ch_known]
    return per_ch.mean(), per_ch.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="seg_r34")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    df = load_seg_manifest()
    train_df = df[df.split == "train"]
    print(train_df.groupby("source").size().to_string())
    precompute(train_df)
    ds = SegCrops(train_df, length=args.iters * args.bs)
    dl = DataLoader(ds, args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True)

    model = build_model(args.encoder).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    model.train()
    t0, run, log = time.time(), None, []
    for it, (x, y, k) in enumerate(dl):
        x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
        y, k = y.cuda(non_blocking=True), k.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x)
        loss, per = seg_loss(lg.float(), y, k)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if (it + 1) % 250 == 0:
            rec = {"iter": it + 1, "loss": round(run, 4), "sec": round(time.time() - t0)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
        if it + 1 >= args.iters:
            break
    torch.save({"state": model.state_dict(), "encoder": args.encoder, "channels": SEG_CHANNELS}, out / "best.pt")
    (out / "log.json").write_text(json.dumps(log, indent=2))
    print("saved", out / "best.pt")


if __name__ == "__main__":
    main()
