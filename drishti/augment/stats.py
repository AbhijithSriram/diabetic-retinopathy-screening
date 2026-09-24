"""Measurable signatures of a capture, for validating the camera simulator.

A simulator that only looks convincing is a liability: it lets us claim
robustness we have not earned. So the simulated captures are held to
measurements rather than to opinion. Each statistic below is something a
camera does to an image and a retina does not, computed inside the field of
view only, and normalised so it does not simply track exposure.

The test is then a distribution question, not a per-image one: do our
simulated captures of development data cover the range that real images
from cameras we never trained on actually occupy?
"""
from __future__ import annotations

import cv2
import numpy as np


def _disc(img: np.ndarray) -> np.ndarray:
    return (img.max(axis=2) > 10)


def capture_stats(img: np.ndarray) -> dict[str, float]:
    """Six numbers describing how an image was captured, not what it shows."""
    m = _disc(img)
    if m.sum() < 1000:
        return {}
    g = img[..., 1].astype(np.float32)
    gm = g[m]
    mean = float(gm.mean()) + 1e-6
    n = img.shape[0]
    c = (n - 1) / 2.0
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2) / c

    # sharpness: high-frequency energy relative to overall level. A soft lens,
    # a defocused shot and a small sensor all pull this down.
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    sharp = float(np.sqrt((lap[m] ** 2).mean())) / mean

    # noise: the robust spread of the residual after a 3x3 median, which
    # removes real structure but leaves sensor grain. MAD, so a few vessels
    # do not dominate.
    med = cv2.medianBlur(img[..., 1], 3).astype(np.float32)
    resid = (g - med)[m]
    noise = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826 / mean

    # vignetting: rim brightness against centre brightness
    inner, outer = m & (r < 0.35), m & (r > 0.80)
    vig = float(g[outer].mean() / (g[inner].mean() + 1e-6)) if outer.sum() > 200 and inner.sum() > 200 else np.nan

    # illumination non-uniformity: spread of the heavily blurred field
    small = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(m.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA) > 0.6
    illum = float(small[msmall].std() / (small[msmall].mean() + 1e-6)) if msmall.sum() > 20 else np.nan

    # colour balance, as channel ratios: immune to exposure, sensitive to
    # white balance, to the tone curve and to a yellowing lens
    rr = float(img[..., 0][m].mean() / mean)
    bb = float(img[..., 2][m].mean() / mean)

    # JPEG blockiness: energy on the 8-pixel grid against energy off it.
    # Around 1.0 means no block structure; heavy compression pushes it up.
    d = np.abs(np.diff(g, axis=1))
    valid = m[:, :-1] & m[:, 1:]
    cols = np.arange(d.shape[1])
    on = valid & (cols % 8 == 7)
    off = valid & (cols % 8 != 7)
    block = float(d[on].mean() / (d[off].mean() + 1e-6)) if on.sum() > 200 else np.nan

    return {"sharpness": sharp, "noise": noise, "vignetting": vig,
            "illum_cv": illum, "red_ratio": rr, "blue_ratio": bb, "blockiness": block}


KEYS = ["sharpness", "noise", "vignetting", "illum_cv", "red_ratio", "blue_ratio", "blockiness"]


def coverage(reference: np.ndarray, target: np.ndarray, lo: float = 5, hi: float = 95) -> float:
    """Share of `target` values falling inside the central range of
    `reference`. One number for "does this distribution reach that one"."""
    ok = np.isfinite(reference)
    if ok.sum() < 20:
        return float("nan")
    a, b = np.percentile(reference[ok], [lo, hi])
    t = target[np.isfinite(target)]
    return float(((t >= a) & (t <= b)).mean()) if t.size else float("nan")
