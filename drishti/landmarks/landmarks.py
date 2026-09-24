"""Optic disc and fovea centre localisation by heatmap regression.

A small U-Net predicts one Gaussian heatmap per landmark on the 512-px
grader crop; the peak (refined by a local weighted centroid) is the
estimate. Heatmap regression is generally more accurate than regressing
the two coordinates directly, and the peak height doubles as a confidence.
Trained on IDRiD localisation train (413), evaluated on IDRiD test (103)
in original pixels, the unit of the IDRiD challenge leaderboard.

    python -m drishti.landmarks.landmarks train --name lmk_r18
    python -m drishti.landmarks.landmarks evaluate --name lmk_r18
"""
from __future__ import annotations

import argparse
import json

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import PROCESSED, RESULTS, RUNS, SEED
from ..data import sources
from ..data.datasets import grader_input, to_tensor
from ..preprocess import Geometry, prepare, read_rgb, write_image

SIZE = 512
SIGMA = 6.0
CACHE = PROCESSED / "lmk512"


def build_model(pretrained=True):
    return smp.Unet("resnet18", encoder_weights="imagenet" if pretrained else None, classes=2)


def cache_localization() -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    df = sources.idrid_localization()
    recs = []
    for r in df.to_dict("records"):
        out = CACHE / f"{r['image_id']}.jpg"
        meta = CACHE / f"{r['image_id']}.json"
        if not meta.exists():
            rgb = read_rgb(r["path"])
            prep = prepare(rgb, SIZE)
            write_image(out, prep.image)
            meta.write_text(json.dumps(prep.geometry.as_dict()))
        g = Geometry(**json.loads(meta.read_text()))
        odx, ody = g.to_processed(r["od_x"], r["od_y"])
        fvx, fvy = g.to_processed(r["fv_x"], r["fv_y"])
        recs.append({**r, "cache": str(out), "geom": g.as_dict(), "kp": [(float(odx), float(ody)), (float(fvx), float(fvy))]})
    return recs


def heatmaps(kps, size=SIZE, sigma=SIGMA):
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    hm = np.zeros((2, size, size), np.float32)
    for i, (x, y) in enumerate(kps):
        if 0 <= x < size and 0 <= y < size:
            hm[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return hm


class LmkData(Dataset):
    def __init__(self, recs, train):
        self.recs, self.train = recs, train
        self.aug = A.Compose([
            A.Affine(rotate=(-20, 20), scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=0.9),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.8),
            A.HueSaturationValue(5, 12, 6, p=0.5),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=(3, 5))], p=0.2),
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)) if train else None

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = read_rgb(r["cache"])
        kps = r["kp"]
        if self.aug is not None:
            o = self.aug(image=img, keypoints=kps)
            img, kps = o["image"], [tuple(k) for k in o["keypoints"]]
        return to_tensor(grader_input(img)), torch.from_numpy(heatmaps(kps))


def peak(hm: np.ndarray) -> tuple[float, float, float]:
    """Argmax refined by a 7x7 weighted centroid; returns (x, y, height)."""
    y, x = np.unravel_index(int(np.argmax(hm)), hm.shape)
    h = float(hm[y, x])
    y0, y1, x0, x1 = max(0, y - 3), min(hm.shape[0], y + 4), max(0, x - 3), min(hm.shape[1], x + 4)
    w = np.clip(hm[y0:y1, x0:x1], 0, None)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    s = w.sum()
    return (float((w * xx).sum() / s), float((w * yy).sum() / s), h) if s > 0 else (float(x), float(y), h)


@torch.no_grad()
def predict(model, crop_raw: np.ndarray, device="cuda") -> dict:
    """crop_raw: 512-px FOV crop. Returns landmarks in that frame."""
    x = to_tensor(grader_input(crop_raw))[None].to(device)
    hm = torch.sigmoid(model(x).float())[0].cpu().numpy()
    od, fv = peak(hm[0]), peak(hm[1])
    return {"od": od, "fovea": fv}


def train(name, epochs=60, bs=12, lr=5e-4):
    torch.manual_seed(SEED)
    recs = cache_localization()
    tr = [r for r in recs if r["split"] == "train"]
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    dl = DataLoader(LmkData(tr, True), bs, shuffle=True, num_workers=3, drop_last=True, persistent_workers=True)
    model = build_model().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(dl), pct_start=0.1)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x)
            # positive pixels are rare: weight them up
            w = 1 + 50 * y
            loss = (F.binary_cross_entropy_with_logits(lg.float(), y, reduction="none") * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if ep % 10 == 9:
            print(json.dumps({"epoch": ep + 1, "loss": round(tot / len(dl), 5)}), flush=True)
    torch.save({"state": model.state_dict()}, out / "best.pt")


def load(name_or_path, device="cuda"):
    from pathlib import Path
    p = Path(name_or_path)
    if not p.suffix:
        p = RUNS / str(name_or_path) / "best.pt"
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["state"])
    return m.eval().to(device)


def evaluate(name):
    recs = [r for r in cache_localization() if r["split"] == "test"]
    model = load(name)
    d_od, d_fv = [], []
    for r in recs:
        pr = predict(model, read_rgb(r["cache"]))
        g = Geometry(**r["geom"])
        ox, oy = g.to_original(pr["od"][0], pr["od"][1])
        fx, fy = g.to_original(pr["fovea"][0], pr["fovea"][1])
        d_od.append(float(np.hypot(ox - r["od_x"], oy - r["od_y"])))
        d_fv.append(float(np.hypot(fx - r["fv_x"], fy - r["fv_y"])))
    rep = {"n": len(recs), "unit": "original IDRiD pixels (4288x2848)",
           "od_mean_ed": round(float(np.mean(d_od)), 1), "od_median_ed": round(float(np.median(d_od)), 1),
           "fovea_mean_ed": round(float(np.mean(d_fv)), 1), "fovea_median_ed": round(float(np.median(d_fv)), 1),
           "od_within_100px_pct": round(100 * float(np.mean(np.array(d_od) < 100)), 1),
           "fovea_within_100px_pct": round(100 * float(np.mean(np.array(d_fv) < 100)), 1),
           "fovea_within_200px_pct": round(100 * float(np.mean(np.array(d_fv) < 200)), 1)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"landmarks_{name}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "evaluate"])
    ap.add_argument("--name", default="lmk_r18")
    a = ap.parse_args()
    train(a.name) if a.cmd == "train" else evaluate(a.name)
