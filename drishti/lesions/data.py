"""Lesion segmentation data: IDRiD (54 train / 27 test) and DDR (383 train /
149 valid / 225 test) pixel masks, cached at LESION_SIZE.

Masks are stored as one uint8 PNG per image with one bit per channel
(bit 0 MA, 1 HE, 2 EX, 3 SE, 4 OD). `known` records which channels are
labelled for that image: DDR has no optic-disc masks, so its OD channel is
unknown and excluded from the loss (partial-label training).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import LESION_SIZE, PROCESSED, SEED, SEG_CHANNELS
from ..data import sources
from ..data.datasets import to_tensor
from ..preprocess import Geometry, crop_square, fov_mask, fov_geometry, graham_normalise, prepare_mask, read_rgb, write_image

CACHE = PROCESSED / f"seg{LESION_SIZE}"


def _read_mask(path: str, shape) -> np.ndarray:
    if path in ("", "EMPTY"):
        return np.zeros(shape[:2], np.uint8)
    m = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if m is None:  # some TIFF variants: fall back to PIL
        from PIL import Image
        m = np.array(Image.open(path))
    if m.ndim == 3:
        # colour channels only: one IDRiD mask (IDRiD_81_EX) is RGBA with an
        # opaque alpha, which would otherwise mark the whole image as lesion
        m = m[..., :3].max(axis=2)
    if m.shape[:2] != tuple(shape[:2]):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def _cache_one(rec: dict) -> dict:
    img_out = CACHE / f"{rec['image_id']}.jpg"
    msk_out = CACHE / f"{rec['image_id']}_mask.png"
    meta_out = CACHE / f"{rec['image_id']}.json"
    if img_out.exists() and msk_out.exists() and meta_out.exists():
        return json.loads(meta_out.read_text())
    rgb = read_rgb(rec["path"])
    g = fov_geometry(fov_mask(rgb), LESION_SIZE)
    img = crop_square(rgb, g)
    fov = crop_square(fov_mask(rgb), g, interp=cv2.INTER_NEAREST)
    img[fov == 0] = 0
    bits = np.zeros((LESION_SIZE, LESION_SIZE), np.uint8)
    known = []
    for i, ch in enumerate(SEG_CHANNELS):
        path = rec.get(ch, "")
        if ch == "OD" and path == "":
            known.append(0)
            continue
        known.append(1)
        m = prepare_mask(_read_mask(path, rgb.shape), g)
        bits |= (m << i).astype(np.uint8)
    write_image(img_out, img)
    cv2.imencode(".png", bits)[1].tofile(str(msk_out))
    meta = {"image_id": rec["image_id"], "source": rec["source"], "split": rec["split"], "image": str(img_out),
            "mask": str(msk_out), "known": known, "geometry": g.as_dict(), "orig_path": rec["path"]}
    meta_out.write_text(json.dumps(meta))
    return meta


def build_seg_manifest(workers: int = 8) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    recs = []
    for f in (sources.idrid_segmentation, sources.ddr_segmentation):
        d = f()
        if len(d):
            recs += d.to_dict("records")
    with ThreadPoolExecutor(workers) as ex:
        metas = list(ex.map(_cache_one, recs))
    df = pd.DataFrame(metas)
    df.to_json(PROCESSED / "seg_manifest.json", orient="records", indent=1)
    return df


def load_seg_manifest() -> pd.DataFrame:
    return pd.read_json(PROCESSED / "seg_manifest.json", orient="records")


def unpack(bits: np.ndarray) -> np.ndarray:
    return np.stack([(bits >> i) & 1 for i in range(len(SEG_CHANNELS))], 0).astype(np.float32)


NPY = PROCESSED / f"seg{LESION_SIZE}_npy"


def _npy(image_id: str, kind: str):
    return NPY / f"{image_id}_{kind}.npy"


def precompute(df: pd.DataFrame, workers: int = 8):
    """Write Graham-normalised images and mask bits as .npy for mmap reads."""
    NPY.mkdir(parents=True, exist_ok=True)

    def one(r):
        if not _npy(r["image_id"], "img").exists():
            np.save(_npy(r["image_id"], "img"), full_input(read_rgb(r["image"])))
            np.save(_npy(r["image_id"], "bits"), cv2.imdecode(np.fromfile(r["mask"], np.uint8), cv2.IMREAD_UNCHANGED))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, df.to_dict("records")))


class SegCrops(Dataset):
    """Random 512-px training crops, biased towards lesions so that rare,
    tiny classes (microaneurysms, soft exudates) are actually seen."""

    def __init__(self, df: pd.DataFrame, crop: int = 512, length: int = 4000, lesion_bias: float = 0.7):
        import albumentations as A
        self.rows = df.to_dict("records")
        self.crop, self.length, self.bias = crop, length, lesion_bias
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.7),
            A.HueSaturationValue(5, 10, 5, p=0.4),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.Defocus(radius=(1, 2))], p=0.15),
            A.ImageCompression(quality_range=(50, 95), p=0.3),
        ])
        self._cache = {}

    def __len__(self):
        return self.length

    def _load(self, i):
        # normalised image + mask as memory-mapped arrays (see precompute):
        # a crop reads only the rows it needs instead of decoding and
        # normalising the whole 1440 px image every time
        if i not in self._cache:
            r = self.rows[i]
            img = np.load(_npy(r["image_id"], "img"), mmap_mode="r")
            bits = np.load(_npy(r["image_id"], "bits"), mmap_mode="r")
            self._cache[i] = (img, bits, np.array(r["known"], np.float32))
        return self._cache[i]

    def _lesion_pixels(self, i, bits):
        """Per-channel lesion pixel coordinates, subsampled, computed once."""
        key = ("pix", i)
        if key not in self._cache:
            b = np.asarray(bits)
            out = []
            for k in range(5):
                ys, xs = np.nonzero((b >> k) & 1)
                if len(ys) > 4000:
                    sel = np.random.default_rng(i).choice(len(ys), 4000, replace=False)
                    ys, xs = ys[sel], xs[sel]
                out.append((ys, xs))
            self._cache[key] = out
        return self._cache[key]

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        i = int(rng.integers(len(self.rows)))
        img, bits, known = self._load(i)
        H = img.shape[0]
        c = self.crop
        pix = self._lesion_pixels(i, bits)
        present = [k for k in range(5) if len(pix[k][0])]
        if rng.random() < self.bias and present:
            # pick a lesion pixel, favouring rare channels
            ch_weights = np.array([4, 2, 1, 3, 0.5])
            w = ch_weights[present] / ch_weights[present].sum()
            k = int(rng.choice(present, p=w))
            ys, xs = pix[k]
            j = int(rng.integers(len(ys)))
            cy, cx = ys[j] + rng.integers(-c // 3, c // 3), xs[j] + rng.integers(-c // 3, c // 3)
        else:
            cy, cx = rng.integers(c // 2, H - c // 2, size=2)
        y0 = int(np.clip(cy - c // 2, 0, H - c))
        x0 = int(np.clip(cx - c // 2, 0, H - c))
        im = np.array(img[y0:y0 + c, x0:x0 + c])
        bm = np.array(bits[y0:y0 + c, x0:x0 + c])
        out = self.aug(image=im, mask=bm)
        x = to_tensor(out["image"])
        y = torch.from_numpy(unpack(out["mask"]))
        return x, y, torch.from_numpy(known)


def full_input(img: np.ndarray) -> np.ndarray:
    """Graham normalisation of the whole LESION_SIZE image (training crops are
    cut from the normalised image, so train and inference match exactly)."""
    fov = (img.max(axis=2) > 8).astype(np.uint8)
    return graham_normalise(img, fov)


if __name__ == "__main__":
    d = build_seg_manifest()
    print(d.groupby(["source", "split"]).size().to_string())
