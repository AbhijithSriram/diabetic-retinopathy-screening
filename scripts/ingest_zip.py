"""Turn a downloaded archive into cached crops without ever unpacking it.

EyePACS is 7.8 GB and ODIR another 1.7, on a machine with under 40 GB
spare. Extracting the originals only to crop them and delete them again is
a bad trade. Each member is decoded from the zip in memory, cropped by the
same `preprocess.prepare` every other dataset goes through, and written
straight into the crop cache with its geometry sidecar --- byte for byte
the files `data.prepare` would have produced, so the rest of the pipeline
cannot tell the difference and no second preprocessing path exists.

    python scripts/ingest_zip.py eyepacs [--size 512] [--workers 5]
    python scripts/ingest_zip.py odir

Afterwards the archive can be deleted: the label file is extracted beside
it, and the adapters in `data/sources.py` fall back to the cache when the
originals are gone.
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

DATASETS = {
    # For EyePACS the mirror also ships a pre-cropped copy; we take the
    # uncropped one, because our own field-of-view detection and square crop
    # are the single preprocessing path and a stranger's cropping is not it.
    "eyepacs": {
        "zip": RAW / "eyepacs" / "diabetic-retinopathy-resized.zip",
        "member_dir": "resized_train/resized_train/",
        "labels": ["trainLabels.csv"],
    },
    "odir": {
        "zip": RAW / "odir" / "ocular-disease-recognition-odir5k.zip",
        "member_dir": "ODIR-5K/ODIR-5K/Training Images/",
        "labels": ["full_df.csv"],
    },
}

_tls = local()


def _zip(path: Path) -> zipfile.ZipFile:
    """One handle per worker thread: ZipFile is not safe to share."""
    if not hasattr(_tls, "z"):
        _tls.z = zipfile.ZipFile(path)
    return _tls.z


def one(member: str, zpath: Path, source: str, size: int, cache: Path) -> int:
    stem = Path(member).stem
    out = cache / f"{source}_{stem}.jpg"
    if out.exists():
        return 0
    try:
        buf = np.frombuffer(_zip(zpath).read(member), np.uint8)
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
    ap.add_argument("source", choices=sorted(DATASETS))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    spec = DATASETS[args.source]
    zpath = spec["zip"]
    if not zpath.exists():
        sys.exit(f"missing {zpath}")
    cache = PROCESSED / f"g{args.size}" / args.source
    cache.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        members = [n for n in names
                   if n.startswith(spec["member_dir"]) and n.lower().endswith((".jpeg", ".jpg", ".png"))]
        for want in spec["labels"]:
            src = next((n for n in names if n.endswith(want)), None)
            if src:
                (zpath.parent / want).write_bytes(z.read(src))
    if args.limit:
        members = members[:args.limit]
    print(f"{len(members)} images -> {cache}", flush=True)

    t0 = time.time()
    done = bad = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(lambda m: one(m, zpath, args.source, args.size, cache), members)):
            done += r == 1
            bad += r == -1
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(members)}  {rate:.0f}/s  "
                      f"eta {(len(members) - i) / rate / 60:.0f} min", flush=True)
    print(f"{done} new crops, {bad} failed, {time.time() - t0:.0f}s")
    print(f"cache size: {sum(p.stat().st_size for p in cache.glob('*.jpg')) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
