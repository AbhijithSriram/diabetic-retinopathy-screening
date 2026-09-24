"""The single preprocessing path used by training, evaluation, the app and
the MATLAB port (matlab/drishti_preprocess.m mirrors it step for step).

Steps:
1. find the circular field of view (FOV) of the fundus camera,
2. crop to it and pad to a square centred on the FOV, so every camera ends up
   with the retina filling the frame at the same scale,
3. resize to the stage's working size,
4. optionally apply local-average colour normalisation (Graham, Kaggle 2015),
   which removes most camera-specific colour cast and illumination gradient.

Every image, good or borderline, goes through the same steps. There is no
"enhance only if borderline" branch, because that would make the grader see
two differently distributed kinds of input.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache

import cv2
import numpy as np


@dataclass
class Geometry:
    """Maps between original-image and processed-image pixel coordinates."""

    x0: float  # top-left of the square crop in original coordinates (may be < 0)
    y0: float
    side: float  # side of the square crop in original pixels (= FOV diameter)
    out: int  # side of the processed image
    cx: float  # FOV centre, original coordinates
    cy: float
    radius: float  # FOV radius, original pixels

    @property
    def scale(self) -> float:
        return self.out / self.side

    def to_processed(self, x, y):
        return (np.asarray(x) - self.x0) * self.scale, (np.asarray(y) - self.y0) * self.scale

    def to_original(self, x, y):
        return np.asarray(x) / self.scale + self.x0, np.asarray(y) / self.scale + self.y0

    def as_dict(self) -> dict:
        return asdict(self)


def read_rgb(path) -> np.ndarray:
    """Read an image as RGB uint8. Handles non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(path, rgb: np.ndarray, quality: int = 95) -> None:
    ext = str(path).rsplit(".", 1)[-1].lower()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in ("jpg", "jpeg") else []
    ok, buf = cv2.imencode("." + ext, bgr, params)
    if not ok:
        raise ValueError(f"could not encode {path}")
    buf.tofile(str(path))


def fov_mask(rgb: np.ndarray) -> np.ndarray:
    """Binary mask of the camera's field of view.

    The background outside the FOV is near-black on every fundus camera, so a
    low threshold on the brightest channel, followed by keeping the largest
    component and filling holes, is robust across cameras. Works at reduced
    resolution for speed and scales the result back up.
    """
    h, w = rgb.shape[:2]
    f = 512.0 / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    bright = small.max(axis=2).astype(np.float32)
    thr = max(12.0, 0.08 * np.percentile(bright, 99))
    m = bright > thr
    # some cameras / exports put the retina on a white background instead of
    # black: if the image border is bright and grey, treat near-white as background
    b = max(2, small.shape[0] // 50)
    border = np.concatenate([small[:b].reshape(-1, 3), small[-b:].reshape(-1, 3), small[:, :b].reshape(-1, 3), small[:, -b:].reshape(-1, 3)])
    if np.median(border.min(axis=1)) > 200:
        lo = small.min(axis=2).astype(np.int16)
        spread = small.max(axis=2).astype(np.int16) - lo
        m &= ~((lo > 200) & (spread < 30))
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones((h, w), np.uint8)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == keep).astype(np.uint8)
    # fill holes (dark lesions or the dark fovea must stay inside the FOV):
    # flood the background from a padded border; anything not reached is a hole
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = pad.copy()
    cv2.floodFill(flood, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8), (0, 0), 2)
    m = (flood[1:-1, 1:-1] != 2).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def fov_geometry(mask: np.ndarray, out: int) -> Geometry:
    """Square crop centred on the FOV.

    Many 45-degree cameras clip the top and bottom of the circle, so the
    diameter is taken as the larger bbox side and the centre as the bbox
    centre (clipping is symmetric on the cameras in our datasets).
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        h, w = mask.shape
        side = float(max(h, w))
        return Geometry(x0=(w - side) / 2, y0=(h - side) / 2, side=side, out=out, cx=w / 2, cy=h / 2, radius=side / 2)
    x_min, x_max, y_min, y_max = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = float(max(x_max - x_min, y_max - y_min))
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    return Geometry(x0=cx - side / 2, y0=cy - side / 2, side=side, out=out, cx=cx, cy=cy, radius=side / 2)


def crop_square(img: np.ndarray, g: Geometry, interp=cv2.INTER_AREA) -> np.ndarray:
    """Crop the square described by g (padding with zeros) and resize."""
    s = g.scale
    m = np.array([[s, 0, -g.x0 * s], [0, s, -g.y0 * s]], dtype=np.float64)
    border = 0
    return cv2.warpAffine(img, m, (g.out, g.out), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


@lru_cache(maxsize=16)
def _circle(size: int, margin: float) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2.0
    r = size / 2.0 * (1.0 - margin)
    m = (((xx - c) ** 2 + (yy - c) ** 2) <= r * r).astype(np.uint8)
    m.setflags(write=False)
    return m


def circle_mask(size: int, margin: float = 0.0) -> np.ndarray:
    return _circle(int(size), float(margin))


def local_average(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with a large sigma, computed at 1/4 resolution.

    The blur keeps only frequencies far below the downsampled Nyquist limit,
    so blurring a 4x smaller image with sigma/4 and upsampling gives the
    same result (max difference ~1 grey level) about 16x faster. The MATLAB
    port does exactly the same."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 4.0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def graham_normalise(rgb: np.ndarray, fov: np.ndarray | None = None) -> np.ndarray:
    """Local-average colour subtraction: 4*I - 4*Gauss(I) + 128.

    sigma is 1/30 of the image side, as in Graham's Kaggle 2015 write-up. The
    region outside the FOV (and a thin rim, where the blur mixes in the black
    background) is set to 128, the neutral value.
    """
    size = rgb.shape[0]
    img = rgb.astype(np.float32)
    blur = local_average(img, size / 30.0)
    out = np.clip(4.0 * img - 4.0 * blur + 128.0, 0, 255).astype(np.uint8)
    k = 4 * max(1, size // 128) + 1  # same reach as the old repeated 5x5 erosions
    rim = circle_mask(size, margin=0.04) if fov is None else cv2.erode(fov, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    out[rim == 0] = 128
    return out


@dataclass
class Prepared:
    image: np.ndarray  # processed RGB, uint8, out x out
    fov: np.ndarray  # processed FOV mask, uint8 {0,1}
    geometry: Geometry


def prepare(rgb: np.ndarray, out: int, normalise: bool = False) -> Prepared:
    """FOV crop + square + resize (+ optional Graham normalisation)."""
    mask = fov_mask(rgb)
    g = fov_geometry(mask, out)
    img = crop_square(rgb, g)
    fov = crop_square(mask, g, interp=cv2.INTER_NEAREST)
    img[fov == 0] = 0
    if normalise:
        img = graham_normalise(img, fov)
    return Prepared(image=img, fov=fov, geometry=g)


def prepare_mask(mask: np.ndarray, g: Geometry) -> np.ndarray:
    """Apply the same crop to a label mask (nearest-free: area-then-threshold
    keeps tiny microaneurysm masks from vanishing when downscaling)."""
    m = (mask > 0).astype(np.float32)
    warped = crop_square(m, g, interp=cv2.INTER_AREA)
    return (warped > 0.25).astype(np.uint8)


def to_normalised(img: np.ndarray, fov: np.ndarray | None = None) -> np.ndarray:
    """Graham-normalise an already-cropped image (used on cached crops)."""
    if fov is None:
        fov = (img.max(axis=2) > 8).astype(np.uint8)
    return graham_normalise(img, fov)
