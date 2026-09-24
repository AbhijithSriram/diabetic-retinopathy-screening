"""Image quality gate: measurable checks that each map to one instruction a
health worker can act on, plus a learned gradability score for failures
the rules cannot name (cataract, small pupil, eyelashes, reflections).

Rule metrics are computed on the 512-px FOV crop so thresholds do not
depend on the camera's resolution:
- field of view: is a round retina visible, and is it complete?
- exposure: median brightness of the retina, and clipped-pixel fraction
- illumination evenness: ratio of the darkest to brightest retinal sector
- focus: variance of the Laplacian of the green channel inside the retina,
  normalised by local contrast so dark-but-sharp images are not called blurry

The learned score (quality/train.py) is a small CNN trained to separate DDR's
"ungradable" images from gradable ones. Thresholds for both live in
configs/quality.json and are set on DDR valid, never on a test set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import REPO
from ..preprocess import fov_geometry, fov_mask, crop_square

CFG_PATH = REPO / "configs" / "quality.json"
DEFAULTS = {
    "min_fov_fraction": 0.45,  # retina area / crop area (a full circle is ~0.785)
    "dark_median": 45, "bright_median": 200,
    "max_clipped": 0.08,
    "min_evenness": 0.35,
    "min_focus": 0.012,
    "gradability_threshold": 0.5,
    "max_attempts": 3,
}

INSTRUCTIONS = {
    "no_retina": "No retina found. Check the lens is close to the eye and the patient's eye is open, then retake.",
    "off_centre": "The retina is cut off. Ask the patient to look straight at the fixation light and centre the camera on the pupil.",
    "too_dark": "Too dark. Dim the room lights so the pupil stays large, move the camera closer, and retake.",
    "too_bright": "Too bright or washed out. Reduce the flash / illumination and tilt slightly to avoid reflections.",
    "uneven": "One side is shadowed. Re-align the camera with the centre of the pupil and retake.",
    "blurry": "Blurry. Ask the patient not to blink or move, hold the camera steady, refocus, and retake.",
    "ungradable": "The picture is not clear enough to read (possible cataract or small pupil). Retake once; if it fails again the patient needs an in-person eye check.",
}


def load_cfg() -> dict:
    cfg = dict(DEFAULTS)
    if CFG_PATH.exists():
        cfg.update(json.loads(CFG_PATH.read_text()))
    return cfg


def metrics(rgb: np.ndarray) -> dict:
    mask_full = fov_mask(rgb)
    g = fov_geometry(mask_full, 512)
    crop = crop_square(rgb, g)
    fov = crop_square(mask_full, g, interp=cv2.INTER_NEAREST).astype(bool)
    fov_frac = float(fov.mean())
    if fov.sum() < 1000:
        return {"fov_fraction": fov_frac, "no_retina": True}
    # completeness: how much of the ideal inscribed circle is covered
    yy, xx = np.mgrid[:512, :512]
    circle = ((xx - 255.5) ** 2 + (yy - 255.5) ** 2) <= 250 ** 2
    completeness = float((fov & circle).sum() / circle.sum())
    lum = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    inner = cv2.erode(fov.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    vals = lum[inner]
    median = float(np.median(vals))
    clipped = float(np.mean((vals >= 250) | (vals <= 5)))
    # evenness: 3x3 sector medians inside the retina
    sectors = []
    for i in range(3):
        for j in range(3):
            sl = (slice(i * 170, (i + 1) * 170), slice(j * 170, (j + 1) * 170))
            v = lum[sl][inner[sl]]
            if v.size > 500:
                sectors.append(np.median(v))
    evenness = float(min(sectors) / max(max(sectors), 1.0)) if sectors else 0.0
    green = crop[:, :, 1].astype(np.float32)
    lap = cv2.Laplacian(green, cv2.CV_32F, ksize=3)
    local_sd = np.sqrt(np.maximum(cv2.GaussianBlur(green ** 2, (0, 0), 8) - cv2.GaussianBlur(green, (0, 0), 8) ** 2, 1.0))
    focus = float(np.var(lap[inner] / (local_sd[inner] + 4.0)) / 100.0)
    return {"fov_fraction": fov_frac, "completeness": completeness, "median": median, "clipped": clipped,
            "evenness": evenness, "focus": focus, "no_retina": False}


@dataclass
class QualityResult:
    passed: bool
    problems: list = field(default_factory=list)
    instructions: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    gradability: float | None = None


def check(rgb: np.ndarray, gradability: float | None = None, cfg: dict | None = None) -> QualityResult:
    """Pass/fail comes from framing (hard rules) plus the learned score.
    The exposure / evenness / focus rules explain *why* an image failed and
    which instruction to give; on their own they do not reject an image the
    learned model finds gradable, because hand-set thresholds are cruder."""
    cfg = cfg or load_cfg()
    m = metrics(rgb)
    hard, soft = [], []
    if m.get("no_retina"):
        hard.append("no_retina")
    else:
        if m["fov_fraction"] < cfg["min_fov_fraction"] or m["completeness"] < 0.6:
            hard.append("off_centre")
        if m["median"] < cfg["dark_median"]:
            soft.append("too_dark")
        elif m["median"] > cfg["bright_median"] or m["clipped"] > cfg["max_clipped"]:
            soft.append("too_bright")
        if m["evenness"] < cfg["min_evenness"]:
            soft.append("uneven")
        if m["focus"] < cfg["min_focus"]:
            soft.append("blurry")
    if gradability is None:
        passed = not hard and not soft
        probs = hard + soft
    else:
        passed = not hard and gradability >= cfg["gradability_threshold"]
        if passed:
            probs = []
        elif hard:
            probs = hard + soft
        else:
            probs = soft or ["ungradable"]
    return QualityResult(passed=passed, problems=probs, instructions=[INSTRUCTIONS[p] for p in probs], metrics=m, gradability=gradability)
