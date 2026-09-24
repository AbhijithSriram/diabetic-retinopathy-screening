"""Add DeepDRiD's regular fundus photographs as a second model-selection set.

    python scripts/append_deepdrid.py --zip <deepdrid_github_snapshot.zip> [--workers 8]

DeepDRiD (Shanghai, ISBI 2020 challenge) grades every eye 0-4 by
ophthalmologists, with two photographs per eye (macula- and disc-centred).
It is added as split `devext2`: held out whole, never trained on, scored
freely when ranking candidates, never quoted as a result --- the same role
ODIR has, with better labels. ODIR's grades are read off free-text findings
and a second seed of the same recipe moved its AUC by 0.005, as much as the
differences being ranked, so a second, expert-graded selection set makes
model choice less of a coin toss.

Like `append_eyepacs_test.py`, this only appends: every existing manifest row
is verified byte-identical afterwards. An image whose retina signature
matches anything the models are fitted on (train, valA, valB) is flagged
`dup_of_dev = 1` so it can be left out of selection scores.
"""
from __future__ import annotations

import argparse
import io
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
from drishti.preprocess import prepare, read_rgb, write_image  # noqa: E402

SOURCE, SPLIT, MIN_CORR = "deepdrid", "devext2", 0.80
ROOT = "DeepDRiD-master/regular_fundus_images/"
_tls = local()


def labels(z: zipfile.ZipFile) -> pd.DataFrame:
    frames = []
    for part in ("regular-fundus-training/regular-fundus-training.csv",
                 "regular-fundus-validation/regular-fundus-validation.csv"):
        frames.append(pd.read_csv(io.BytesIO(z.read(ROOT + part))))
    d = pd.concat(frames, ignore_index=True)
    # one grade per image: the eye's own level (left or right column)
    d["grade"] = d.left_eye_DR_Level.fillna(d.right_eye_DR_Level)
    d["eye"] = np.where(d.left_eye_DR_Level.notna(), "left", "right")
    d = d[d.grade.notna()].copy()
    d["grade"] = d.grade.astype(int)
    # the challenge's own evaluation images, graded in spreadsheets
    try:
        ev = pd.read_excel(io.BytesIO(z.read(ROOT + "Online-Challenge1&2-Evaluation/Challenge1_labels.xlsx")))
        ev.columns = [c.strip() for c in ev.columns]
        idcol = next(c for c in ev.columns if "image" in c.lower())
        gcol = next(c for c in ev.columns if "dr" in c.lower() or "level" in c.lower() or "grade" in c.lower())
        ev = ev.rename(columns={idcol: "image_id", gcol: "grade"})[["image_id", "grade"]].dropna()
        ev["grade"] = ev.grade.astype(int)
        ev["patient_id"] = ev.image_id.astype(str).str.split("_").str[0]
        ev["eye"] = np.where(ev.image_id.astype(str).str.contains("_l"), "left", "right")
        ev["Overall quality"] = np.nan
        d = pd.concat([d, ev], ignore_index=True)
    except Exception as e:  # the spreadsheet layout is not guaranteed across snapshots
        print(f"challenge evaluation labels skipped: {e}")
    d["image_id"] = d.image_id.astype(str)
    return d.drop_duplicates("image_id")


def process(zpath: Path, member: str, iid: str) -> dict:
    c512 = PROCESSED / "g512" / SOURCE / f"{iid}.jpg"
    c768 = PROCESSED / "g768" / SOURCE / f"{iid}.jpg"
    rec = {"image_id": iid, "ok": 0}
    try:
        if c512.exists() and c768.exists() and c512.with_suffix(".json").exists():
            img512 = read_rgb(c512)
            meta = json.loads(c512.with_suffix(".json").read_text())
        else:
            if not hasattr(_tls, "z"):
                _tls.z = zipfile.ZipFile(zpath)
            bgr = cv2.imdecode(np.frombuffer(_tls.z.read(member), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return rec | {"error": "undecodable"}
            rgb = np.ascontiguousarray(bgr[..., ::-1])
            c512.parent.mkdir(parents=True, exist_ok=True)
            c768.parent.mkdir(parents=True, exist_ok=True)
            p512, p768 = prepare(rgb, 512), prepare(rgb, 768)
            write_image(c512, p512.image)
            write_image(c768, p768.image)
            meta = p512.geometry.as_dict() | {"orig_h": rgb.shape[0], "orig_w": rgb.shape[1]}
            c512.with_suffix(".json").write_text(json.dumps(meta))
            c768.with_suffix(".json").write_text(json.dumps(
                p768.geometry.as_dict() | {"orig_h": rgb.shape[0], "orig_w": rgb.shape[1]}))
            img512 = p512.image
        h, thumb = retina_signature(img512)
        return rec | meta | {"ok": 1, "cache": str(c512), "cache768": str(c768), "dhash": int(h), "_thumb": thumb}
    except Exception as e:
        return rec | {"error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    zpath = Path(a.zip)

    man_path, thumbs_path = PROCESSED / "manifest.csv", PROCESSED / "thumbs.npy"
    man = pd.read_csv(man_path, low_memory=False, dtype=str, keep_default_na=False)
    thumbs = np.load(thumbs_path)
    if len(thumbs) != len(man):
        sys.exit(f"thumbs.npy has {len(thumbs)} rows, manifest {len(man)}: they must align")
    if (man.source == SOURCE).any():
        sys.exit(f"the manifest already has {SOURCE} rows")

    with zipfile.ZipFile(zpath) as z:
        lab = labels(z)
        members = {Path(n).stem: n for n in z.namelist()
                   if n.startswith(ROOT) and n.lower().endswith((".jpg", ".jpeg", ".png"))}
    lab = lab[lab.image_id.isin(members)]
    print(f"{len(lab)} graded regular-fundus images; grades {lab.grade.value_counts().sort_index().to_dict()}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(lambda i: process(zpath, members[i], f"{SOURCE}_{i}"), lab.image_id))
    ok = [r for r in recs if r["ok"] == 1]
    print(f"{len(ok)} cropped, {len(recs) - len(ok)} failed, {time.time() - t0:.0f}s", flush=True)
    for r in [r for r in recs if r["ok"] != 1][:5]:
        print(f"  failed {r['image_id']}: {r.get('error')}")

    new_t = np.stack([r.pop("_thumb") for r in ok]).astype(np.float32)
    fit = man.split.isin(["train", "valA", "valB"]).to_numpy()
    c_best = np.full(len(new_t), -1.0, np.float32)
    for i0 in range(0, len(new_t), 4096):
        c_best[i0:i0 + 4096] = (new_t[i0:i0 + 4096] @ thumbs[fit].T).max(1)
    dup = c_best >= MIN_CORR
    print(f"near-duplicates of fitted-on images (flagged dup_of_dev): {int(dup.sum())}", flush=True)

    lab = lab.set_index("image_id")
    next_group = int(pd.to_numeric(man.dup_group, errors="coerce").max()) + 1
    rows = []
    for k, r in enumerate(ok):
        stem = r["image_id"][len(SOURCE) + 1:]
        L = lab.loc[stem]
        q = L.get("Overall quality")
        rows.append({
            "image_id": r["image_id"], "source": SOURCE, "path": f"{zpath}::{members[stem]}", "role": "dev",
            "grade": int(L.grade), "dme": -1, "gradable": 1,
            "exam_id": f"dd_{L.patient_id}", "eye": L.eye, "orig_split": "deepdrid",
            **{c: r.get(c, "") for c in ("x0", "y0", "side", "out", "cx", "cy", "radius", "orig_h", "orig_w")},
            "cache": r["cache"], "dhash": r["dhash"], "ok": 1, "dup_group": next_group + k,
            "has_dup": int(dup[k]), "label_conflict": 0, "group_key": r["image_id"], "ddr_series": "",
            "moved_for_leakage": 0, "split": SPLIT, "dup_of_dev": int(dup[k]), "cache768": r["cache768"],
        })
    new = pd.DataFrame(rows, columns=man.columns).astype(str)
    print(new.groupby("grade").size().to_dict())
    if a.dry_run:
        print("dry run: manifest not written")
        return
    stamp = time.strftime("%Y%m%d_%H%M")
    shutil.copy2(man_path, PROCESSED / f"manifest_before_{SOURCE}_{stamp}.csv")
    shutil.copy2(thumbs_path, PROCESSED / f"thumbs_before_{SOURCE}_{stamp}.npy")
    pd.concat([man, new], ignore_index=True).to_csv(man_path, index=False)
    np.save(thumbs_path, np.concatenate([thumbs, new_t]))
    back = pd.read_csv(man_path, low_memory=False, dtype=str, keep_default_na=False)
    assert back.iloc[:len(man)].equals(man), "existing manifest rows changed -- restore the backup"
    print(f"appended {len(new)} {SPLIT} rows; existing {len(man)} rows verified unchanged")


if __name__ == "__main__":
    main()
