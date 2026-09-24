"""Grad-CAM for the referable-DR output, and its validation against
ophthalmologist lesion masks.

Target: log P(grade >= 2) = log sigmoid(l0) + log sigmoid(l1) under the CORN
head, i.e. "what made the model think this eye is referable".

Validation (the PS asks for "clinically validated Grad-CAM"): on IDRiD's 81
segmentation images, which the grader never saw, we measure how much of the
heat-map falls on expert-annotated lesions compared with chance:
- energy-in-lesion: share of Grad-CAM mass inside the (dilated) lesion mask
- lift: that share divided by the mask's share of the retina (1.0 = chance)
- pointing game: is the hottest pixel on an annotated lesion?
Saliency maps are known to localise imperfectly (Saporta et al., Nat Mach
Intell 2022; Arun et al., Radiol AI 2021), so this is reported as a
measurement, not assumed.

    python -m drishti.explain.gradcam --grader b0_512
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..config import GRADE_SIZE, RESULTS, RUNS
from ..data.datasets import grader_input, to_tensor
from ..eval.metrics import wilson
from ..grading.model import Grader


def load_grader(name_or_path, device="cuda") -> Grader:
    from pathlib import Path
    p = Path(name_or_path)
    if not p.suffix:
        p = RUNS / str(name_or_path) / "best.pt"
    ck = torch.load(p, map_location=device, weights_only=False)
    m = Grader(ck["backbone"], pretrained=False, img_size=ck.get("size"))
    m.graham = ck.get("graham", True)
    m.load_state_dict(ck["state"])
    return m.eval().to(device)


def gradcam(model: Grader, x: torch.Tensor) -> np.ndarray:
    """x: (1, 3, H, W). Returns an HxW map in [0, 1]."""
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        f = model.features(x).float()
        f.retain_grad()
        lg = model.logits_from_features(f) / model.temperature
        target = F.logsigmoid(lg[:, 0]) + F.logsigmoid(lg[:, 1])
        target.sum().backward()
    w = f.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * f).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def cam_for_crop(model, crop_raw: np.ndarray, device="cuda") -> np.ndarray:
    x = to_tensor(grader_input(crop_raw))[None].to(device)
    return gradcam(model, x)


def heat_overlay(crop_raw: np.ndarray, cam: np.ndarray, alpha=0.45) -> np.ndarray:
    hm = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    fov = (crop_raw.max(axis=2) > 8)[..., None]
    out = (crop_raw * (1 - alpha) + hm * alpha).astype(np.uint8)
    return np.where(fov, out, crop_raw)


def validate(grader_name: str, dilate_px: int = 10) -> dict:
    from ..lesions.data import load_seg_manifest, unpack
    from ..preprocess import read_rgb
    model = load_grader(grader_name)
    df = load_seg_manifest()
    rows = df[df.source == "idrid"].to_dict("records")
    energy, lift, areas, hits, n_lesion = [], [], [], 0, 0
    kern = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    for r in rows:
        img = cv2.resize(read_rgb(r["image"]), (GRADE_SIZE, GRADE_SIZE), interpolation=cv2.INTER_AREA)
        bits = cv2.imdecode(np.fromfile(r["mask"], np.uint8), cv2.IMREAD_UNCHANGED)
        les = unpack(bits)[:4].max(0)
        les = cv2.resize(les, (GRADE_SIZE, GRADE_SIZE), interpolation=cv2.INTER_AREA) > 0
        if les.sum() == 0:
            continue
        n_lesion += 1
        fov = img.max(axis=2) > 8
        mask = cv2.dilate(les.astype(np.uint8), kern).astype(bool) & fov
        cam = cam_for_crop(model, img) * fov
        e = float(cam[mask].sum() / max(cam.sum(), 1e-9))
        a = float(mask.sum() / fov.sum())
        energy.append(e)
        areas.append(a)
        lift.append(e / a if a > 0 else np.nan)
        y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
        hits += int(mask[y, x])
    energy, lift = np.array(energy), np.array(lift)
    rng = np.random.default_rng(0)
    boot = [np.nanmean(rng.choice(lift, len(lift))) for _ in range(2000)]
    rep = {
        "images_with_lesions": n_lesion, "dilation_px_at_512": dilate_px,
        "mean_energy_in_lesion": round(float(energy.mean()), 3),
        "mean_lift_over_chance": round(float(np.nanmean(lift)), 2),
        "lift_95ci": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)],
        "pointing_game": [round(v, 3) for v in wilson(hits, n_lesion)],
        "pointing_game_chance": round(float(np.mean(areas)), 3),
        "note": "IDRiD segmentation images (never seen by the grader); chance lift = 1.0",
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"gradcam_{grader_name}.json").write_text(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grader", default="b0_512")
    print(json.dumps(validate(ap.parse_args().grader), indent=2))
