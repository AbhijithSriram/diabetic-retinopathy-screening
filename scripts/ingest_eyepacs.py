"""Turn the EyePACS archive into cached crops without ever unpacking it.

The archive is 7.8 GB and the machine has under 40 GB spare, so extracting
35,126 originals only to crop them and delete them again is a bad trade.
Each member is decoded from the zip in memory, cropped by the same
`preprocess.prepare` every other dataset goes through, and written straight
into the crop cache with its geometry sidecar --- the same files
`data.prepare` would have produced, so the rest of the pipeline cannot tell
the difference and no second preprocessing path exists.

    python scripts/ingest_eyepacs.py [--size 512] [--workers 5] [--limit 0]

Afterwards the zip can be deleted: `trainLabels.csv` is extracted alongside
it, and the adapter in `data/sources.py` reads the cache when the originals
are gone.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drishti.config import PROCESSED, RAW  # noqa: E402
from drishti.preprocess import prepare, write_image  # noqa: E402

ZIP = RAW / "eyepacs" / "diabetic-retinopathy-resized.zip"
# the uncropped copies: our own field-of-view detection and square crop are
# the single preprocessing path, and a mirror's cropping is not it
MEMBER_DIR = "resized_train/resized_train/"
_tls = local()


def _zip() -> zipfile.ZipFile:
    """One handle per worker thread: ZipFile is not safe to share."""
    if not hasattr(_tls, "z"):
        _tls.z = zipfile.ZipFile(ZIP)
    return _tls.z


def one(member: str, size: int, cache: Path) -> int:
    stem = Path(member).stem
    out = cache / f"eyepacs_{stem}.jpg"
    if out.exists():
        return 0
    try:
        buf = np.frombuffer(_zip().read(member), np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return -1
        rgb = bgr[..., ::-1]
        prep = prepare(rgb, size)
        write_image(out, prep.image)
        out.with_suffix(".json").write_text(json.dumps(
            prep.geometry.as_dict() | {"orig_h": rgb.shape[0], "orig_w": rgb.shape[1]}))
        return 1
    except Exception as e:
        print(f"  {stem}: {e}", flush=True)
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not ZIP.exists():
        sys.exit(f"missing {ZIP}")
    cache = PROCESSED / f"g{args.size}" / "eyepacs"
    cache.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP) as z:
        members = [n for n in z.namelist()
                   if n.startswith(MEMBER_DIR) and n.lower().endswith((".jpeg", ".jpg", ".png"))]
        labels = next((n for n in z.namelist() if n.endswith("trainLabels.csv")), None)
        if labels:
            (RAW / "eyepacs" / "trainLabels.csv").write_bytes(z.read(labels))
    if args.limit:
        members = members[:args.limit]
    print(f"{len(members)} images -> {cache}", flush=True)

    t0 = time.time()
    done = bad = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(lambda m: one(m, args.size, cache), members)):
            done += r == 1
            bad += r == -1
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(members)}  {rate:.0f}/s  eta {(len(members) - i) / rate / 60:.0f} min",
                      flush=True)
    print(f"{done} new crops, {bad} failed, {time.time() - t0:.0f}s")
    print(f"cache size: {sum(p.stat().st_size for p in cache.glob('*.jpg')) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
