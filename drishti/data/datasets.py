"""PyTorch datasets over the cached crops."""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..preprocess import circle_mask, graham_normalise, read_rgb

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def geometry_augment() -> A.Compose:
    """Framing: how the operator pointed the camera. This happens before the
    capture, so it is applied before the capture simulator too."""
    return A.Compose([
        A.Affine(rotate=(-180, 180), scale=(0.92, 1.08), translate_percent=(-0.03, 0.03), p=1.0),
        A.HorizontalFlip(p=0.5),
    ])


def field_augment(strong: bool = True) -> A.Compose:
    """Augmentations that imitate field capture: rotation and framing
    error, exposure and colour cast, defocus and motion blur, sensor noise
    and heavy JPEG compression from phone uploads."""
    t = [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
        A.RandomGamma(gamma_limit=(75, 135), p=0.5),
        A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=15, val_shift_limit=8, p=0.5),
    ]
    if strong:
        t += [
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=(3, 7)), A.Defocus(radius=(2, 4))], p=0.25),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.ImageCompression(quality_range=(35, 95), p=0.4),
        ]
    return A.Compose(t)


def colour_augment() -> A.Compose:
    """What is left of the photometric set once the capture simulator is
    doing the optics, the sensor and the ISP: only a per-channel gain, which
    the simulator's white balance does not fully span. Measured capture
    statistics say colour is where our training data differs most from
    cameras we have never seen (results/simulator_stats.json)."""
    return A.Compose([
        A.RGBShift(r_shift_limit=18, g_shift_limit=10, b_shift_limit=18, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
    ])


_SCALE = (1.0 / (255.0 * STD)).astype(np.float32)
_SHIFT = (-MEAN / STD).astype(np.float32)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
    return t.mul_(torch.from_numpy(_SCALE)[:, None, None]).add_(torch.from_numpy(_SHIFT)[:, None, None])


def grader_input(raw_crop: np.ndarray) -> np.ndarray:
    """The grader always sees the Graham-normalised crop."""
    fov = (raw_crop.max(axis=2) > 8).astype(np.uint8) & circle_mask(raw_crop.shape[0], margin=0.0)
    return graham_normalise(raw_crop, fov)


class GradingDataset(Dataset):
    """Cached crops, optionally re-imaged through a simulated cheap camera.

    `capture_p` is the share of training images pushed through
    `drishti.augment.capture`. It is a share rather than all of them because
    the grader still has to work on the good cameras it will also meet, and
    because the simulation costs about 85 ms an image. When it fires, the
    photometric albumentations are replaced by a colour-only set: the
    simulator already models blur, noise, compression and exposure, and
    stacking a second, unphysical copy of each on top only blurs the
    question of what helped.
    """

    def __init__(self, df, train: bool = False, size: int = 512, with_label: bool = True,
                 capture_p: float = 0.0, n_devices: int = 32, seed: int = 0, graham: bool = True,
                 soft: np.ndarray | None = None):
        # Prefer a cache built at this resolution; fall back to resizing the
        # 512 crops, which is honest only for going smaller, never larger.
        col = f"cache{size}"
        if col in df and (df[col].astype(str) != "").all():
            self.paths = df[col].tolist()
        else:
            self.paths = df["cache"].tolist()
        self.grades = df["grade"].to_numpy() if with_label else np.zeros(len(df), int)
        self.train = train
        self.size = size
        self.capture_p = capture_p if train else 0.0
        self.n_devices = n_devices
        self.seed = seed
        self.geo = geometry_augment() if train else None
        self.aug = field_augment() if train else None
        self.caug = colour_augment() if train else None
        self._rng = None
        self._devices = None
        # graham=False feeds the plain crop: for RETFound, which was pretrained
        # on photographs, not on Graham-normalised images
        self.graham = graham
        # soft: (N, 5) teacher class probabilities for distillation, or None
        self.soft = None if soft is None else torch.as_tensor(soft, dtype=torch.float32)

    def __len__(self):
        return len(self.paths)

    def _camera(self):
        """One device bank per worker process, drawn once and then reused, so
        a device's dust and white balance recur across many patients."""
        if self._devices is None:
            from ..augment import sample_profile
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng([self.seed, wid])
            self._devices = [sample_profile(self._rng) for _ in range(self.n_devices)]
        return self._rng, self._devices[self._rng.integers(len(self._devices))]

    def __getitem__(self, i):
        img = read_rgb(self.paths[i])
        if img.shape[0] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.geo is not None:
            img = self.geo(image=img)["image"]
        if self.capture_p and np.random.random() < self.capture_p:
            from ..augment import simulate_capture
            rng, prof = self._camera()
            img = simulate_capture(img, prof, rng)
            img = self.caug(image=img)["image"]
        elif self.aug is not None:
            img = self.aug(image=img)["image"]
        if self.graham:
            img = grader_input(img)
        if self.soft is not None:
            return to_tensor(img), int(self.grades[i]), self.soft[i]
        return to_tensor(img), int(self.grades[i])
