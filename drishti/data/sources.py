"""Adapters that turn each public dataset's own layout into one table.

Every row is one image with the columns in COLUMNS. Adapters search by glob
instead of hard-coding paths, because the zip layouts differ slightly
between mirrors. An adapter returns an empty frame when its dataset is
missing, so the rest of the pipeline runs with whatever has been downloaded.

Grade conventions: ICDR 0-4; -1 means "no grade label". DDR's class 5
("ungradable") becomes grade -1, gradable 0.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RAW

COLUMNS = ["image_id", "source", "path", "role", "grade", "dme", "gradable", "exam_id", "eye", "orig_split"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c, v in [("grade", -1), ("dme", -1), ("gradable", 1)]:
        df[c] = df[c].fillna(v).astype(int)
    return df


def _index_images(root: Path) -> dict[str, Path]:
    """Map lower-case stem -> path for every image under root."""
    out = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            out.setdefault(p.stem.lower(), p)
    return out


def _source_path(source: str, stem: str, imgs: dict[str, Path]) -> str | None:
    """Where this image can be read from.

    EyePACS and ODIR are ingested straight from their archives into the crop
    cache (scripts/ingest_zip.py) and the archives are then deleted, so for
    those two the cached crop is the only copy left. `prepare` reuses a
    cached crop when it finds one, so pointing at it is enough for every
    later stage; nothing re-reads the original.
    """
    p = imgs.get(stem.lower())
    if p is not None:
        return str(p)
    from ..config import PROCESSED
    cached = PROCESSED / "g512" / source / f"{source}_{stem}.jpg"
    return str(cached) if cached.exists() else None


def _find(root: Path, pattern: str) -> list[Path]:
    rx = re.compile(pattern, re.I)
    return sorted(p for p in root.rglob("*") if rx.search(str(p).replace("\\", "/")))


# --------------------------------------------------------------------- APTOS
def aptos(root: Path = RAW / "aptos") -> pd.DataFrame:
    csvs = [p for p in root.rglob("train.csv")]
    if not csvs:
        return _frame([])
    labels = pd.read_csv(csvs[0])
    imgs = _index_images(root)
    rows = []
    for r in labels.itertuples():
        p = imgs.get(str(r.id_code).lower())
        if p is not None and "test" not in p.parent.name.lower():
            rows.append([f"aptos_{r.id_code}", "aptos", str(p), "dev", int(r.diagnosis), -1, 1, None, None, "train"])
    return _frame(rows)


# ----------------------------------------------------------------------- DDR
def ddr(root: Path = RAW / "ddr") -> pd.DataFrame:
    rows = []
    for split in ("train", "valid", "test"):
        txts = _find(root, rf"DR_grading/{split}\.txt$")
        if not txts:
            continue
        folder = txts[0].parent / split
        for line in txts[0].read_text().splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            name, lab = parts[0], int(parts[1])
            p = folder / name
            if not p.exists():
                continue
            grade, gradable = (lab, 1) if lab <= 4 else (-1, 0)
            rows.append([f"ddr_{Path(name).stem}", "ddr", str(p), "dev", grade, -1, gradable, None, None, split])
    return _frame(rows)


def ddr_segmentation(root: Path = RAW / "ddr") -> pd.DataFrame:
    """DDR lesion segmentation set: returns image path and one mask path per
    lesion (EX, HE, MA, SE). Masks are complete for all four lesions."""
    rows = []
    for split in ("train", "valid", "test"):
        imgs = _find(root, rf"lesion_segmentation/{split}/image/[^/]+\.(jpg|jpeg|png|tif)$")
        for p in imgs:
            rec = {"image_id": f"ddrseg_{p.stem}", "source": "ddr", "path": str(p), "split": split}
            # DDR names the folder "label" in train/test but "segmentation label" in valid
            label_dir = next((d for d in (p.parent.parent / "label", p.parent.parent / "segmentation label") if d.exists()), None)
            if label_dir is None:
                continue
            for les in ("MA", "HE", "EX", "SE"):
                cands = list((label_dir / les).glob(p.stem + ".*")) if (label_dir / les).exists() else []
                if not cands:
                    raise FileNotFoundError(f"DDR mask missing: {label_dir / les / p.stem}")
                rec[les] = str(cands[0])
            rec["OD"] = ""  # DDR has no optic-disc masks: loss is masked for this channel
            rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- IDRiD
def idrid_grading(root: Path = RAW / "idrid") -> pd.DataFrame:
    rows = []
    for split, key in (("train", "Training"), ("test", "Testing")):
        csvs = _find(root, rf"Disease Grading.*{key} Labels\.csv$")
        if not csvs:
            continue
        lab = pd.read_csv(csvs[0])
        lab.columns = [c.strip() for c in lab.columns]
        folder_key = "a. Training Set" if split == "train" else "b. Testing Set"
        imgs = {p.stem.lower(): p for p in _find(root, rf"Disease Grading/1\. Original Images/{re.escape(folder_key)}/[^/]+\.jpg$")}
        for r in lab.itertuples(index=False):
            name = str(r[0]).strip()
            if not name or name.lower() == "nan":
                continue
            p = imgs.get(name.lower())
            if p is None:
                continue
            grade, dme = int(r[1]), int(r[2])
            rows.append([f"idrid_{split}_{name}", "idrid", str(p), "external", grade, int(dme > 0), 1, None, None, split])
    return _frame(rows)


def idrid_dme_grade(root: Path = RAW / "idrid") -> dict[str, int]:
    """IDRiD 'Risk of macular edema' 0/1/2 keyed by image_id (for the DME rule)."""
    out = {}
    for split, key in (("train", "Training"), ("test", "Testing")):
        for c in _find(root, rf"Disease Grading.*{key} Labels\.csv$")[:1]:
            lab = pd.read_csv(c)
            for r in lab.itertuples(index=False):
                name = str(r[0]).strip()
                if name and name.lower() != "nan":
                    out[f"idrid_{split}_{name}"] = int(r[2])
    return out


def idrid_segmentation(root: Path = RAW / "idrid") -> pd.DataFrame:
    rows = []
    folders = {"MA": "1. Microaneurysms", "HE": "2. Haemorrhages", "EX": "3. Hard Exudates", "SE": "4. Soft Exudates", "OD": "5. Optic Disc"}
    for split, key in (("train", "a. Training Set"), ("test", "b. Testing Set")):
        imgs = _find(root, rf"Segmentation/1\. Original Images/{re.escape(key)}/[^/]+\.jpg$")
        for p in imgs:
            rec = {"image_id": f"idridseg_{p.stem}", "source": "idrid", "path": str(p), "split": split}
            for les, folder in folders.items():
                cands = _find(root, rf"Segmentation/2\. All Segmentation Groundtruths/{re.escape(key)}/{re.escape(folder)}/{p.stem}_{les}\.tif$")
                # IDRiD omits the file when a lesion type is absent: that is a
                # genuine negative (empty mask), not a missing label.
                rec[les] = str(cands[0]) if cands else "EMPTY"
            rows.append(rec)
    return pd.DataFrame(rows)


def idrid_localization(root: Path = RAW / "idrid") -> pd.DataFrame:
    """Optic disc and fovea centre coordinates (original pixel units)."""
    rows = []
    for split, key, folder in (("train", "Training", "a. Training Set"), ("test", "Testing", "b. Testing Set")):
        od = _find(root, rf"Optic Disc Center.*{key}.*\.csv$")
        fv = _find(root, rf"Fovea Center.*{key}.*\.csv$")
        if not od or not fv:
            continue

        def load(c):
            d = pd.read_csv(c)
            d = d.iloc[:, :3]
            d.columns = ["name", "x", "y"]
            return d.dropna(subset=["name"]).set_index("name")

        od_df, fv_df = load(od[0]), load(fv[0])
        imgs = {p.stem: p for p in _find(root, rf"Localization/1\. Original Images/{re.escape(folder)}/[^/]+\.jpg$")}
        for name, r in od_df.iterrows():
            name = str(name).strip()
            if name not in imgs or name not in fv_df.index:
                continue
            f = fv_df.loc[name]
            rows.append({"image_id": f"idridloc_{split}_{name}", "path": str(imgs[name]), "split": split,
                         "od_x": float(r.x), "od_y": float(r.y), "fv_x": float(f.x), "fv_y": float(f.y)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- Messidor-2
def messidor2(root: Path = RAW / "messidor2") -> pd.DataFrame:
    """Messidor-2 images with the Krause et al. (2018) adjudicated grades.

    Exam pairing comes from the ADCIS pairs file when present. Images the
    adjudication panel marked ungradable keep gradable=0 and are excluded
    from accuracy metrics, as in the published benchmarks.
    """
    grade_csv = [p for p in root.rglob("*.csv") if "messidor_data" in p.name.lower() or "adjudicated" in p.read_text(errors="ignore")[:500].lower()]
    if not grade_csv:
        return _frame([])
    g = pd.read_csv(grade_csv[0])
    g.columns = [c.strip().lower() for c in g.columns]
    imgs = _index_images(root)
    pairs = {}
    for c in root.rglob("*.csv"):
        if c == grade_csv[0]:
            continue
        try:
            d = pd.read_csv(c, sep=None, engine="python")
        except Exception:
            continue
        if d.shape[1] >= 2 and d.iloc[:, 0].astype(str).str.contains(r"\.(png|jpg|tif)", case=False).any():
            for i, r in d.iterrows():
                a, b = str(r.iloc[0]).strip(), str(r.iloc[1]).strip()
                pairs[Path(a).stem.lower()] = (i, "left")
                pairs[Path(b).stem.lower()] = (i, "right")
            break
    rows = []
    for r in g.itertuples(index=False):
        rec = r._asdict()
        name = str(rec["image_id"]).strip()
        p = imgs.get(Path(name).stem.lower())
        if p is None:
            continue
        gr = rec.get("adjudicated_dr_grade")
        grade = int(gr) if pd.notna(gr) else -1
        dme = rec.get("adjudicated_dme")
        dme = int(dme) if pd.notna(dme) else -1
        gradable = int(rec.get("adjudicated_gradable", 1)) if pd.notna(rec.get("adjudicated_gradable", 1)) else 1
        exam, eye = pairs.get(Path(name).stem.lower(), (None, None))
        rows.append([f"messidor2_{Path(name).stem}", "messidor2", str(p), "external", grade, dme, gradable,
                     f"m2exam_{exam}" if exam is not None else None, eye, "all"])
    return _frame(rows)


# ------------------------------------------------------------------- mBRSET
def mbrset(root: Path = RAW / "mbrset") -> pd.DataFrame:
    """mBRSET (PhysioNet, credentialed): handheld phone-based camera."""
    csvs = [p for p in root.rglob("*.csv") if "label" in p.name.lower() or "mbrset" in p.name.lower()]
    if not csvs:
        return _frame([])
    d = pd.read_csv(csvs[0])
    d.columns = [c.strip().lower() for c in d.columns]
    imgs = _index_images(root)
    idcol = next(c for c in d.columns if c in ("file", "image", "image_id", "filename"))
    grade_col = next((c for c in d.columns if "icdr" in c), None)
    dme_col = next((c for c in d.columns if "edema" in c or c == "dme"), None)
    qual_col = next((c for c in d.columns if "quality" in c), None)
    pat_col = next((c for c in d.columns if "patient" in c), None)
    rows = []
    for _, r in d.iterrows():
        p = imgs.get(Path(str(r[idcol])).stem.lower())
        if p is None:
            continue
        grade = int(r[grade_col]) if grade_col and pd.notna(r[grade_col]) else -1
        dme = int(r[dme_col]) if dme_col and pd.notna(r[dme_col]) else -1
        gradable = 1
        if qual_col and pd.notna(r[qual_col]):
            gradable = 0 if str(r[qual_col]).strip().lower() in ("unsatisfactory", "0", "no", "false") else 1
        rows.append([f"mbrset_{p.stem}", "mbrset", str(p), "external", grade, dme, gradable,
                     f"mb_{r[pat_col]}" if pat_col else None, None, "all"])
    return _frame(rows)


# ------------------------------------------------------------------ EyePACS
def eyepacs(root: Path = RAW / "eyepacs") -> pd.DataFrame:
    """EyePACS / Kaggle 2015: 35,126 training images from a US screening
    programme, on a wide mix of cameras and often poor quality. Exactly the
    diversity our development data lacks.

    Training only. The grades are a single reader's, with no adjudication
    and a known error rate, so they must never reach valB (which fits the
    conformal thresholds) or any test set: a noisy label there would
    corrupt a guarantee rather than just add a little training noise.
    `prepare.assign_splits` enforces that; this adapter only supplies rows.

    File names are `<patient>_<left|right>`, so unlike DDR and APTOS this
    source does carry eye pairing.
    """
    csvs = sorted(p for p in root.rglob("trainLabels*.csv"))
    if not csvs:
        return _frame([])
    # prefer the plain label file over any mirror's "cropped" variant
    csv = next((p for p in csvs if "crop" not in p.name.lower()), csvs[0])
    labels = pd.read_csv(csv)
    labels.columns = [c.strip().lower() for c in labels.columns]
    if "image" not in labels or "level" not in labels:
        return _frame([])
    imgs = _index_images(root)
    rows = []
    for r in labels.itertuples():
        name = str(r.image)
        p = _source_path("eyepacs", name, imgs)
        if p is None:
            continue
        patient, _, eye = name.rpartition("_")
        rows.append([f"eyepacs_{name}", "eyepacs", p, "dev", int(r.level), -1, 1,
                     f"ep_{patient}" if patient else None,
                     eye.lower() if eye.lower() in ("left", "right") else None, "train"])
    return _frame(rows)


# --------------------------------------------------------------------- ODIR
# ODIR-5K annotates every finding it sees per eye, in free text. The DR
# findings carry their ICDR severity in words, so they map to grades; the
# other retinal diseases do not, and an eye with only those findings is
# grade 0 for diabetic retinopathy, which is what we are grading.
ODIR_GRADE = [
    ("severe proliferative diabetic retinopathy", 4),
    ("proliferative diabetic retinopathy", 4),
    ("severe nonproliferative retinopathy", 3),
    ("severe non proliferative retinopathy", 3),
    ("moderate non proliferative retinopathy", 2),
    ("moderate nonproliferative retinopathy", 2),
    ("mild nonproliferative retinopathy", 1),
    ("mild non proliferative retinopathy", 1),
]
# phrases that say "diabetic retinopathy" without saying how much of it:
# no grade can be read off these, so the image is kept unlabelled
ODIR_AMBIGUOUS = ("suspected", "suspicious", "anterior segment")
ODIR_UNGRADABLE = ("low image quality", "optic disk photographically invisible",
                   "anterior segment image", "image offset", "no fundus image")


def _odir_grade(kw: str) -> int:
    k = kw.lower()
    for phrase, grade in ODIR_GRADE:
        if phrase in k:
            return grade
    if "diabetic retinopathy" in k or any(a in k for a in ODIR_AMBIGUOUS):
        return -1
    return 0


def odir(root: Path = RAW / "odir") -> pd.DataFrame:
    """ODIR-5K: 3,358 patients from Chinese hospitals on a mix of camera
    brands. Training only, like EyePACS: the grades are read off free-text
    findings rather than assigned on the ICDR scale, so they are good enough
    to learn from and not good enough to calibrate against.

    Unlike DDR and APTOS this source names its patients, so it is the only
    development data we have with both eyes of one visit.
    """
    csvs = [p for p in root.rglob("full_df.csv")]
    if not csvs:
        return _frame([])
    d = pd.read_csv(csvs[0])
    need = {"ID", "filename", "Left-Diagnostic Keywords", "Right-Diagnostic Keywords"}
    if not need <= set(d.columns):
        return _frame([])
    d = d.rename(columns={"Left-Diagnostic Keywords": "kw_left", "Right-Diagnostic Keywords": "kw_right"})
    imgs = _index_images(root)
    rows = []
    for r in d.itertuples(index=False):
        stem = Path(str(r.filename)).stem
        low = stem.lower()
        eye = "left" if low.endswith("_left") else "right" if low.endswith("_right") else None
        if eye is None:
            continue
        kw = str(r.kw_left if eye == "left" else r.kw_right)
        p = _source_path("odir", stem, imgs)
        if p is None:
            continue
        gradable = 0 if any(u in kw.lower() for u in ODIR_UNGRADABLE) else 1
        rows.append([f"odir_{stem}", "odir", p, "dev", _odir_grade(kw), -1, gradable,
                     f"od_{r.ID}", eye, "train"])
    return _frame(rows)


def all_grading_sources() -> pd.DataFrame:
    frames = [f() for f in (aptos, ddr, eyepacs, odir, idrid_grading, messidor2, mbrset)]
    frames = [f for f in frames if len(f)]
    if not frames:
        return _frame([])
    return pd.concat(frames, ignore_index=True)
