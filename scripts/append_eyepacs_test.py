"""Add the EyePACS 2015 test half (~53,500 graded images) to the training pool
without touching a single existing manifest row.

    python scripts/append_eyepacs_test.py --labels <testLabels15.csv> --images <dir or .zip> [--workers 16]

Why append instead of re-running scripts/prepare_data.py: a rebuild re-derives
every split from scratch, needs every original (APTOS's are not on this
machine), and is exactly the operation COPY_ME.md warns can move images
between splits. Appending cannot move anything, and the script proves it: the
old rows are compared byte for byte before the new manifest is written.

What the new rows are:
  * source `eyepacs_test`, a name of its own, so a run pinned to
    `--sources ddr,aptos,eyepacs` trains on exactly what it did before and a
    new arm opts in with `--sources ddr,aptos,eyepacs,eyepacs_test`;
  * split `train` only. Like the rest of EyePACS these are one reader's grades
    (the competition's released solution file), fit for training and never
    for valB or any test;
  * cropped by the same `preprocess.prepare` at 512 and 768 as every other
    image, with the same retina signature for duplicate detection.

Leakage control: every new image is correlated against the retina signature
of every image already in the manifest. One that matches (r >= 0.80, the
threshold `prepare.near_duplicate_pairs` uses) anything outside train ---
valA, valB, test, ODIR, IDRiD, Messidor-2, excluded --- is added as
`excluded`, never as training data. Matches to existing training images are
harmless and kept.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drishti.config import PROCESSED  # noqa: E402
from drishti.data.prepare import retina_signature  # noqa: E402
from drishti.preprocess import prepare, write_image  # noqa: E402

SOURCE = "eyepacs_test"
MIN_CORR = 0.80
_tls = local()


def index_images(src: Path, prefix: str = "") -> dict[str, str]:
    """stem -> file path or zip member name, optionally only under `prefix`
    (the Kaggle bundle also holds the train half and APTOS, which we have)."""
    exts = (".jpeg", ".jpg", ".png", ".tif", ".tiff")
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            names = [n for n in z.namelist() if n.lower().endswith(exts) and n.startswith(prefix)]
    else:
        names = [str(p) for p in src.rglob("*") if p.suffix.lower() in exts and prefix in str(p)]
    out = {}
    for n in names:
        out.setdefault(Path(n).stem, n)
    return out


def read(src: Path, name: str) -> np.ndarray | None:
    if src.suffix.lower() == ".zip":
        if not hasattr(_tls, "z"):
            _tls.z = zipfile.ZipFile(src)
        buf = np.frombuffer(_tls.z.read(name), np.uint8)
    else:
        buf = np.fromfile(name, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return None if bgr is None else np.ascontiguousarray(bgr[..., ::-1])


def process(src: Path, stem: str, name: str) -> dict:
    iid = f"{SOURCE}_{stem}"
    c512 = PROCESSED / "g512" / SOURCE / f"{iid}.jpg"
    c768 = PROCESSED / "g768" / SOURCE / f"{iid}.jpg"
    rec = {"image_id": iid, "ok": 0}
    try:
        if c512.exists() and c768.exists() and c512.with_suffix(".json").exists():
            from drishti.preprocess import read_rgb
            img512 = read_rgb(c512)
            meta = json.loads(c512.with_suffix(".json").read_text())
        else:
            rgb = read(src, name)
            if rgb is None:
                return rec | {"error": "undecodable"}
            c512.parent.mkdir(parents=True, exist_ok=True)
            c768.parent.mkdir(parents=True, exist_ok=True)
            p512 = prepare(rgb, 512)
            p768 = prepare(rgb, 768)
            write_image(c512, p512.image)
            write_image(c768, p768.image)
            meta = p512.geometry.as_dict() | {"orig_h": rgb.shape[0], "orig_w": rgb.shape[1]}
            c512.with_suffix(".json").write_text(json.dumps(meta))
            c768.with_suffix(".json").write_text(json.dumps(
                p768.geometry.as_dict() | {"orig_h": rgb.shape[0], "orig_w": rgb.shape[1]}))
            img512 = p512.image
        h, thumb = retina_signature(img512)
        return rec | meta | {"ok": 1, "cache": str(c512), "cache768": str(c768),
                             "dhash": int(h), "_thumb": thumb}
    except Exception as e:
        return rec | {"error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="testLabels15.csv (columns image, level[, Usage])")
    ap.add_argument("--images", required=True, help="folder or .zip holding the test images")
    ap.add_argument("--prefix", default="", help="only images under this folder of the zip, e.g. 'resized test 15/'")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N labelled images only")
    ap.add_argument("--dry-run", action="store_true", help="crop and check, but do not write the manifest")
    a = ap.parse_args()

    man_path, thumbs_path = PROCESSED / "manifest.csv", PROCESSED / "thumbs.npy"
    man = pd.read_csv(man_path, low_memory=False, dtype=str, keep_default_na=False)
    thumbs = np.load(thumbs_path)
    if len(thumbs) != len(man):
        sys.exit(f"thumbs.npy has {len(thumbs)} rows, manifest {len(man)}: they must align")
    if (man.source == SOURCE).any():
        sys.exit(f"the manifest already has {SOURCE} rows; restore the backup before re-running")

    labels = pd.read_csv(a.labels)
    labels.columns = [c.strip().lower() for c in labels.columns]
    if not {"image", "level"} <= set(labels.columns):
        sys.exit(f"{a.labels}: expected columns image, level; got {list(labels.columns)}")
    src = Path(a.images)
    idx = index_images(src, a.prefix)
    labels["image"] = labels["image"].astype(str)
    found = labels[labels.image.isin(idx)]
    print(f"{len(labels)} labels, {len(idx)} images found, {len(found)} matched", flush=True)
    if a.limit:
        found = found.head(a.limit)
    clash = set(f"{SOURCE}_{s}" for s in found.image) & set(man.image_id)
    if clash:
        sys.exit(f"{len(clash)} image ids already in the manifest, e.g. {sorted(clash)[:3]}")
    # the 2015 competition split train and test by patient; confirm it
    train_patients = set(man.loc[man.source == "eyepacs", "exam_id"])
    new_patients = set("ep_" + found.image.str.rpartition("_")[0])
    shared = train_patients & new_patients
    print(f"patients shared with the EyePACS train half: {len(shared)}", flush=True)

    t0 = time.time()
    recs = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(lambda s: process(src, s, idx[s]), found.image)):
            recs.append(r)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(found)}  {rate:.0f}/s  eta {(len(found) - i) / rate / 60:.0f} min", flush=True)
    ok = [r for r in recs if r["ok"] == 1]
    print(f"{len(ok)} cropped, {len(recs) - len(ok)} failed, {time.time() - t0:.0f}s", flush=True)
    for r in [r for r in recs if r["ok"] != 1][:5]:
        print(f"  failed {r['image_id']}: {r.get('error')}")
    if not ok:
        sys.exit("nothing cropped")

    # leakage: correlate against every existing image outside train
    new_t = np.stack([r.pop("_thumb") for r in ok]).astype(np.float32)
    held = (man.split != "train").to_numpy()
    held_t = thumbs[held]
    held_ids = man.image_id.to_numpy()[held]
    best = np.full(len(new_t), -1.0, np.float32)
    best_j = np.zeros(len(new_t), int)
    for i0 in range(0, len(new_t), 4096):
        c = new_t[i0:i0 + 4096] @ held_t.T
        best[i0:i0 + 4096] = c.max(1)
        best_j[i0:i0 + 4096] = c.argmax(1)
    leak = best >= MIN_CORR
    print(f"near-duplicates of held-out images (excluded from training): {int(leak.sum())}", flush=True)
    for k in np.nonzero(leak)[0][:20]:
        print(f"  {ok[k]['image_id']} ~ {held_ids[best_j[k]]}  r={best[k]:.3f}")

    lab = found.set_index("image")
    next_group = int(pd.to_numeric(man.dup_group, errors="coerce").max()) + 1
    rows = []
    for k, r in enumerate(ok):
        stem = r["image_id"][len(SOURCE) + 1:]
        patient, _, eye = stem.rpartition("_")
        rows.append({
            "image_id": r["image_id"], "source": SOURCE, "path": f"{src}::{idx[stem]}", "role": "dev",
            "grade": int(lab.loc[stem, "level"]), "dme": -1, "gradable": 1,
            "exam_id": f"ep_{patient}", "eye": eye.lower() if eye.lower() in ("left", "right") else "",
            "orig_split": "test15",
            **{c: r.get(c, "") for c in ("x0", "y0", "side", "out", "cx", "cy", "radius", "orig_h", "orig_w")},
            "cache": r["cache"], "dhash": r["dhash"], "ok": 1, "dup_group": next_group + k,
            "has_dup": int(leak[k]), "label_conflict": 0, "group_key": r["image_id"], "ddr_series": "",
            "moved_for_leakage": int(leak[k]), "split": "excluded" if leak[k] else "train",
            "dup_of_dev": int(leak[k]), "cache768": r["cache768"],
        })
    new = pd.DataFrame(rows, columns=man.columns).astype(str)
    print(new.groupby(["split", "grade"]).size().unstack(fill_value=0).to_string(), flush=True)
    if a.dry_run:
        print("dry run: manifest not written")
        return

    stamp = time.strftime("%Y%m%d_%H%M")
    shutil.copy2(man_path, PROCESSED / f"manifest_before_{SOURCE}_{stamp}.csv")
    shutil.copy2(thumbs_path, PROCESSED / f"thumbs_before_{SOURCE}_{stamp}.npy")
    out = pd.concat([man, new], ignore_index=True)
    out.to_csv(man_path, index=False)
    np.save(thumbs_path, np.concatenate([thumbs, new_t]))
    # proof that nothing old moved: re-read and compare the old rows exactly
    back = pd.read_csv(man_path, low_memory=False, dtype=str, keep_default_na=False)
    assert back.iloc[:len(man)].equals(man), "existing manifest rows changed -- restore the backup"
    print(f"appended {len(new)} rows ({int((new.split == 'train').sum())} train); "
          f"existing {len(man)} rows verified unchanged; backup manifest_before_{SOURCE}_{stamp}.csv")


if __name__ == "__main__":
    main()
