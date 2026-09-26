"""Lesion inference: full-image probability maps -> countable, located lesions.

Connected components turn a probability map into objects with a position,
an area and a peak confidence. Microaneurysm centres are refined to
sub-pixel precision with a probability-weighted centroid, and reported in
original-image pixel coordinates via the preprocessing geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from ..config import LESION_SIZE, SEG_CHANNELS
from ..data.datasets import to_tensor
from ..preprocess import Geometry
from .data import full_input
from .train import build_model

# Default operating thresholds; `evaluate.py` re-derives them on DDR valid
# and writes runs/<name>/thresholds.json, which load_model() prefers.
DEFAULT_THR = {"MA": 0.5, "HE": 0.5, "EX": 0.5, "SE": 0.5, "OD": 0.5}
MIN_AREA = {"MA": 2, "HE": 6, "EX": 4, "SE": 20, "OD": 200}  # px at LESION_SIZE


@dataclass
class Lesion:
    kind: str
    x: float  # processed-image coordinates (LESION_SIZE frame), sub-pixel
    y: float
    area: int
    peak: float
    bbox: tuple = field(default_factory=tuple)


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = build_model(ck["encoder"], pretrained=False)
    m.load_state_dict(ck["state"])
    m.eval().to(device)
    thr = dict(DEFAULT_THR)
    tpath = ckpt_path.parent / "thresholds.json"
    if tpath.exists():
        import json
        thr.update(json.loads(tpath.read_text()))
    return m, thr


@torch.no_grad()
def predict_logits(model, img_raw: np.ndarray, device="cuda", tta: bool = True) -> np.ndarray:
    """img_raw: LESION_SIZE x LESION_SIZE RGB crop. Returns (5, H, W) logits
    (flip-TTA averaged in logit space); -30 outside the field of view."""
    x = to_tensor(full_input(img_raw))[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):  # bf16 on the GPU only
        lg = model(x).float()
        if tta:
            lg = (lg + torch.flip(model(torch.flip(x, [3])).float(), [3])) / 2
    lg = lg[0].cpu().numpy()
    fov = img_raw.max(axis=2) > 8
    lg[:, ~fov] = -30.0
    return lg


def predict_maps(model, img_raw: np.ndarray, device="cuda", tta: bool = True) -> np.ndarray:
    """Sigmoid probabilities (5, H, W), float64 so confident peaks stay distinct."""
    return 1.0 / (1.0 + np.exp(-predict_logits(model, img_raw, device, tta).astype(np.float64)))


def components(maps: np.ndarray, thr: dict, confident_only: bool = True) -> list[Lesion]:
    """Connected lesion components. The pixel threshold thr[kind] sets each
    lesion's extent; a component is kept only if its peak probability reaches
    thr["peak_" + kind], calibrated on DDR valid so that kept components are
    real lesions at least ~75% of the time (see evaluate.calibrate_peaks).
    Without that filter, faint false components make healthy eyes look
    diseased in the explanation."""
    out = []
    for c, kind in enumerate(SEG_CHANNELS[:4]):
        prob = maps[c]
        peak_min = thr.get(f"peak_{kind}", 0.0) if confident_only else 0.0
        binary = (prob >= thr[kind]).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < MIN_AREA[kind]:
                continue
            x, y, w, h = (int(v) for v in stats[i, :4])
            region = lab[y:y + h, x:x + w] == i
            pr = prob[y:y + h, x:x + w] * region
            if pr.max() < peak_min:
                continue
            wsum = pr.sum()
            yy, xx = np.mgrid[y:y + h, x:x + w]
            cx = float((pr * xx).sum() / wsum) if wsum > 0 else float(cent[i][0])
            cy = float((pr * yy).sum() / wsum) if wsum > 0 else float(cent[i][1])
            out.append(Lesion(kind, cx, cy, area, float(pr.max()), (x, y, w, h)))
    return out


def optic_disc(maps: np.ndarray, thr: float = 0.5):
    """Largest optic-disc component -> (x, y, radius) in LESION_SIZE frame."""
    binary = (maps[4] >= thr).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[i, cv2.CC_STAT_AREA]
    if area < MIN_AREA["OD"]:
        return None
    r = float(np.sqrt(area / np.pi))
    return float(cent[i][0]), float(cent[i][1]), r


def to_original(lesions: list[Lesion], g: Geometry) -> list[dict]:
    rows = []
    for L in lesions:
        ox, oy = g.to_original(L.x, L.y)
        rows.append({"kind": L.kind, "x": round(float(ox), 2), "y": round(float(oy), 2),
                     "area_px_orig": round(L.area / g.scale ** 2, 1), "peak": round(L.peak, 3)})
    return rows


def overlay(img_raw: np.ndarray, maps: np.ndarray, thr: dict) -> np.ndarray:
    """Colour overlay for reports: MA red, HE dark red, EX yellow, SE white.
    Only confident components (the ones the rule engine counts) are drawn."""
    colours = {"MA": (255, 40, 40), "HE": (150, 0, 0), "EX": (255, 220, 0), "SE": (230, 230, 255)}
    out = img_raw.copy()
    kept = components(maps, thr)
    for c, kind in enumerate(SEG_CHANNELS[:4]):
        m = np.zeros(maps.shape[1:], np.uint8)
        for L in kept:
            if L.kind == kind:
                x, y, w, h = L.bbox
                m[y:y + h, x:x + w] |= (maps[c][y:y + h, x:x + w] >= thr[kind]).astype(np.uint8)
        if kind == "MA":  # make tiny MAs visible as rings
            m = cv2.dilate(m, np.ones((5, 5), np.uint8)) - m
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, colours[kind], 2 if kind != "MA" else 1)
    return out
