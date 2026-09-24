"""Retinal vessel segmentation (PS module 2), trained on HRF.

HRF (Budai et al. 2013): 45 images, 15 healthy / 15 DR / 15 glaucoma, with
manual vessel masks. Split: images 01-10 of each group train (30), 11-15
test (15), so every group is represented in both. Images are cropped with
the same FOV preprocessing as the lesion model (LESION_SIZE frame), so the
vessel map lines up with the lesion map pixel for pixel.

Metrics are computed inside the FOV only; scoring the black corners too
would inflate specificity.

    python -m drishti.vessels.vessels train
    python -m drishti.vessels.vessels evaluate
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from ..config import LESION_SIZE, PROCESSED, RAW, RESULTS, RUNS, SEED
from ..data.datasets import to_tensor
from ..eval.metrics import wilson
from ..lesions.data import full_input
from ..preprocess import crop_square, fov_geometry, fov_mask, prepare_mask, read_rgb, write_image

CACHE = PROCESSED / "vessels1440"


def cache_hrf() -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    root = RAW / "vessels" / "hrf"
    recs = []
    for p in sorted((root / "images").glob("*.*")):
        stem = p.stem
        gt = next((root / "manual1").glob(stem + ".*"), None)
        if gt is None:
            continue
        num = int(stem.split("_")[0])
        split = "train" if num <= 10 else "test"
        out_i, out_m = CACHE / f"{stem}.jpg", CACHE / f"{stem}_v.png"
        if not out_m.exists():
            rgb = read_rgb(p)
            g = fov_geometry(fov_mask(rgb), LESION_SIZE)
            img = crop_square(rgb, g)
            fov = crop_square(fov_mask(rgb), g, interp=cv2.INTER_NEAREST)
            img[fov == 0] = 0
            m = cv2.imdecode(np.fromfile(str(gt), np.uint8), cv2.IMREAD_UNCHANGED)
            if m is None:
                from PIL import Image
                m = np.array(Image.open(gt))
            if m.ndim == 3:
                m = m.max(axis=2)
            v = prepare_mask(m > 0, g)
            write_image(out_i, img)
            cv2.imencode(".png", (v * 255).astype(np.uint8))[1].tofile(str(out_m))
        recs.append({"id": stem, "split": split, "image": str(out_i), "mask": str(out_m)})
    return recs


class Crops(Dataset):
    def __init__(self, recs, n, crop=384):
        import albumentations as A
        self.data = [(full_input(read_rgb(r["image"])), (cv2.imdecode(np.fromfile(r["mask"], np.uint8), 0) > 127).astype(np.float32)) for r in recs]
        self.n, self.c = n, crop
        self.aug = A.Compose([A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(), A.RandomBrightnessContrast(0.2, 0.2, p=0.7)])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng()
        img, m = self.data[int(rng.integers(len(self.data)))]
        H = img.shape[0]
        y0, x0 = rng.integers(0, H - self.c, size=2)
        o = self.aug(image=img[y0:y0 + self.c, x0:x0 + self.c], mask=m[y0:y0 + self.c, x0:x0 + self.c])
        return to_tensor(o["image"]), torch.from_numpy(o["mask"])[None]


def build(pretrained=True):
    return smp.Unet("resnet18", encoder_weights="imagenet" if pretrained else None, classes=1)


def train(iters=3000, bs=8):
    torch.manual_seed(SEED)
    recs = [r for r in cache_hrf() if r["split"] == "train"]
    dl = DataLoader(Crops(recs, iters * bs), bs, num_workers=0)
    model = build().cuda()
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=iters, pct_start=0.05)
    for it, (x, y) in enumerate(dl):
        x, y = x.cuda(), y.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x)
        lg = lg.float()
        p = torch.sigmoid(lg)
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
        loss = F.binary_cross_entropy_with_logits(lg, y) + dice
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % 500 == 0:
            print(json.dumps({"iter": it + 1, "loss": round(loss.item(), 4)}), flush=True)
    out = RUNS / "vessels_r18"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict()}, out / "best.pt")


def load(device="cuda"):
    m = build(pretrained=False)
    m.load_state_dict(torch.load(RUNS / "vessels_r18" / "best.pt", map_location=device, weights_only=False)["state"])
    return m.eval().to(device)


@torch.no_grad()
def predict(model, img_raw, device="cuda"):
    x = to_tensor(full_input(img_raw))[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = torch.sigmoid(model(x).float())[0, 0].cpu().numpy()
    return p * (img_raw.max(axis=2) > 8)


def evaluate():
    model = load()
    recs = [r for r in cache_hrf() if r["split"] == "test"]
    ys, ps = [], []
    tp = fn = tn = fp = 0
    for r in recs:
        img = read_rgb(r["image"])
        gt = cv2.imdecode(np.fromfile(r["mask"], np.uint8), 0) > 127
        fov = cv2.erode((img.max(axis=2) > 8).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        p = predict(model, img)
        y, s = gt[fov], p[fov]
        pred = s >= 0.5
        tp += int((y & pred).sum()); fn += int((y & ~pred).sum()); tn += int((~y & ~pred).sum()); fp += int((~y & pred).sum())
        idx = np.random.default_rng(0).choice(len(y), min(len(y), 200000), replace=False)
        ys.append(y[idx]); ps.append(s[idx])
    rep = {"dataset": "HRF test (images 11-15 of healthy / DR / glaucoma, 15 total)", "scale": f"FOV diameter {LESION_SIZE}px",
           "sensitivity": round(tp / (tp + fn), 4), "specificity": round(tn / (tn + fp), 4),
           "accuracy": round((tp + tn) / (tp + tn + fp + fn), 4),
           "auc": round(float(roc_auc_score(np.concatenate(ys), np.concatenate(ps))), 4)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "vessels_hrf.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "evaluate"])
    a = ap.parse_args()
    train() if a.cmd == "train" else evaluate()
