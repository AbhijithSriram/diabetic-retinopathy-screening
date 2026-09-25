"""Build the grading manifest: cache 512-px crops, find near-duplicates,
assign leakage-safe splits.

Split policy (fixed before any model is trained):
- Development sources (APTOS, DDR) are the only data the grader learns from.
  DDR is re-split by us (series x grade stratified, see assign_splits) because
  its official split confounds camera series with grade. APTOS is split
  70/15/15. Both split by duplicate group, so repeat photos of one eye never
  straddle two splits.
- The validation data is halved: valA for early stopping and temperature
  scaling, valB for conformal / threshold calibration. Keeping them apart
  preserves the exchangeability conformal guarantees rely on.
- External sources (IDRiD, Messidor-2, mBRSET) are never used for training
  or tuning. External images that near-duplicate a development image are
  flagged and excluded from external metrics.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import GRADE_SIZE, PROCESSED, SEED
from ..preprocess import prepare, read_rgb, write_image

CACHE = PROCESSED / f"g{GRADE_SIZE}"


def dhash(gray: np.ndarray, size: int = 8) -> np.uint64:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def retina_signature(img512: np.ndarray) -> tuple[np.uint64, np.ndarray]:
    """Duplicate-detection features from the centre of the retina.

    Whole-image hashes fail on fundus photos: every one is a bright disc on
    black with the same illumination falloff, so they all hash alike. Here we
    take the central 320 px (inside the field of view on every camera),
    remove illumination with a high-pass filter, and keep
    (a) a 64-bit DCT perceptual hash for fast candidate search and
    (b) a 32x32 zero-mean unit-norm thumbnail to confirm candidates by
        correlation (the vessel pattern differs between eyes).
    """
    g = cv2.cvtColor(img512, cv2.COLOR_RGB2GRAY).astype(np.float32)[96:416, 96:416]
    hp = g - cv2.GaussianBlur(g, (0, 0), 12)
    small = cv2.resize(hp, (32, 32), interpolation=cv2.INTER_AREA)
    d = cv2.dct(small)[:8, :8].flatten()[1:]
    bits = d > np.median(d)
    h = np.uint64(int("".join("1" if b else "0" for b in bits), 2))
    t = small - small.mean()
    t /= np.linalg.norm(t) + 1e-6
    return h, t.astype(np.float32).ravel()


def _process_one(row) -> dict:
    out = CACHE / row.source / f"{row.image_id}.jpg"
    rec = {"image_id": row.image_id}
    try:
        if out.exists():
            img = read_rgb(out)
            meta_path = out.with_suffix(".json")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        else:
            rgb = read_rgb(row.path)
            prep = prepare(rgb, GRADE_SIZE)
            out.parent.mkdir(parents=True, exist_ok=True)
            write_image(out, prep.image)
            img = prep.image
            meta = prep.geometry.as_dict() | {"orig_h": rgb.shape[0], "orig_w": rgb.shape[1]}
            out.with_suffix(".json").write_text(json.dumps(meta))
        h, thumb = retina_signature(img)
        rec.update(meta)
        rec["cache"] = str(out)
        rec["dhash"] = int(h)
        rec["_thumb"] = thumb
        rec["ok"] = 1
    except Exception as e:  # corrupt / unreadable file
        rec["ok"] = 0
        rec["error"] = str(e)[:200]
    return rec


def cache_crops(df: pd.DataFrame, workers: int = 12) -> tuple[pd.DataFrame, dict]:
    """Returns the manifest with geometry columns, and image_id -> thumbnail."""
    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(_process_one, df.itertuples(index=False), chunksize=16))
    thumbs = {r["image_id"]: r.pop("_thumb") for r in recs if "_thumb" in r}
    meta = pd.DataFrame(recs)
    return df.merge(meta, on="image_id", how="left"), thumbs


def near_duplicate_pairs(thumbs: np.ndarray, min_corr: float = 0.80) -> list[tuple[int, int]]:
    """All pairs whose high-pass retina thumbnails correlate >= min_corr.

    Checked by eye on a contact sheet (2026-09-19): pairs above ~0.8 were the
    same eye photographed again (repeat shots, re-exports); pairs below were
    different eyes. Exact all-pairs correlation, in blocks."""
    pairs = []
    for i0 in range(0, len(thumbs), 2048):
        c = thumbs[i0:i0 + 2048] @ thumbs.T
        ii, jj = np.nonzero(c >= min_corr)
        keep = ii + i0 < jj
        pairs += [(int(a), int(b)) for a, b in zip(ii[keep] + i0, jj[keep])]
    return pairs


def _groups(n: int, pairs) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return np.array([find(i) for i in range(n)])


def _stable_order(keys: list[str]) -> np.ndarray:
    """A deterministic shuffle keyed by content, not row order.

    Splits are assigned by sorting on blake2b(image id + seed), so adding a
    dataset later re-shuffles nothing: every image keeps the split it had.
    """
    import hashlib
    h = [int.from_bytes(hashlib.blake2b(f"{SEED}:{k}".encode(), digest_size=8).digest(), "big") for k in keys]
    return np.argsort(np.array(h, dtype=np.uint64), kind="stable")


def _quota_split(group_keys: list[str], fractions=(0.6, 0.1, 0.1), names=("train", "valA", "valB", "test")) -> dict:
    order = _stable_order(group_keys)
    n = len(order)
    cuts = np.floor(np.cumsum(fractions) * n).astype(int)
    out = {}
    for rank, idx in enumerate(order):
        k = int(np.searchsorted(cuts, rank, side="right"))
        out[group_keys[idx]] = names[min(k, len(names) - 1)]
    return out


def assign_splits(df: pd.DataFrame, thumbs: dict) -> pd.DataFrame:
    df = df.copy()
    t = np.stack([thumbs[i] for i in df.image_id])
    df["dup_group"] = _groups(len(df), near_duplicate_pairs(t))
    grp_size = df.groupby("dup_group")["image_id"].transform("size")
    df["has_dup"] = (grp_size > 1).astype(int)
    grades_in_group = df[df.grade >= 0].groupby("dup_group")["grade"].nunique()
    df["label_conflict"] = df["dup_group"].map(grades_in_group).fillna(1).gt(1).astype(int)

    split = pd.Series("", index=df.index, dtype=object)
    # DDR: our own split, NOT the official one. The official split confounds
    # image series with grade: every "timestamp"-named image (a distinct
    # camera / site) in train and valid is grade 0, while the official test
    # set holds ~1,000 referable images from that series. A model trained on
    # it can learn "this camera's look = healthy", and the official valid set
    # (series predicts the label perfectly) overstates accuracy. We pool
    # DDR, stratify by series x grade, and assign whole duplicate groups
    # (repeat photos of one eye) to one split: 60% train, 10% valA,
    # 10% valB, 20% test.
    # each duplicate group is named by its alphabetically first image id, so
    # the split of an image never depends on what else is in the manifest
    df["group_key"] = df.groupby("dup_group")["image_id"].transform("min")
    ddr = df.source == "ddr"
    df["ddr_series"] = np.where(ddr, np.where(df.image_id.str.startswith("ddr_007-"), "007", "timestamp"), "")
    grp = df[ddr].groupby("group_key").agg(series=("ddr_series", "first"), grade=("grade", "max"))
    assign_ddr = {}
    for _, g in grp.groupby(["series", "grade"]):
        assign_ddr.update(_quota_split(list(g.index)))
    split[ddr] = df.loc[ddr, "group_key"].map(assign_ddr)
    df["moved_for_leakage"] = 0

    # APTOS 70 / 7.5 / 7.5 / 15 by duplicate group, stratified by majority grade
    ap = df[df.source == "aptos"]
    if len(ap):
        gmaj = ap.groupby("group_key")["grade"].agg(lambda s: int(s.mode().iloc[0]))
        assign = {}
        for _, groups in gmaj.groupby(gmaj):
            assign.update(_quota_split(list(groups.index), fractions=(0.70, 0.075, 0.075)))
        split[df.source == "aptos"] = ap["group_key"].map(assign)

    # EyePACS is training material only. Its grades come from one reader with
    # no adjudication, so letting them into valB would mean fitting the
    # conformal thresholds --- the part of this system that carries a stated
    # guarantee --- against labels we do not trust. In train, the same noise
    # costs a little accuracy and nothing else.
    split[df.source == "eyepacs"] = "train"

    # ODIR is held out whole, as an internal cross-camera validation set.
    #
    # Until now every choice about generalisation had to be judged either on
    # in-distribution validation, which cannot see camera shift at all, or on
    # IDRiD and Messidor-2, which are the sets we report against and must not
    # tune on. ODIR is a third country on a different mix of camera brands,
    # so it gives us somewhere to compare candidate models honestly. Its
    # grades are read off free-text findings and are noisier than ours, which
    # is tolerable for ranking two models and is why it is never a headline
    # number.
    split[df.source == "odir"] = "devext"

    ext = df.role == "external"
    split[ext] = "external"
    df["split"] = split

    # Any image that duplicates something the model was fitted on is a
    # leakage risk, whichever set it sits in. "Fitted on" means train (the
    # weights), valA (early stopping) and valB (the conformal thresholds).
    #
    # This used to compare against role == "dev", which was right while the
    # only held-out sets were external. It is not right now: ODIR is held out
    # but carries role "dev", so under the old rule an ODIR image duplicating
    # a training image went unflagged, and ODIR is the set we choose models
    # on. 67 of its 6,392 images turned out to be in that position.
    fit = df.split.isin(["train", "valA", "valB"])
    fit_groups = set(df.loc[fit, "dup_group"])
    df["dup_of_dev"] = (~fit & df["dup_group"].isin(fit_groups)).astype(int)
    # duplicate groups whose copies disagree on grade are dropped from
    # validation and test (kept in train, where the noise only costs a little)
    bad = (df.label_conflict == 1) & df.split.isin(["valA", "valB", "test"])
    df.loc[bad, "split"] = "excluded"
    return df


def summarise(df: pd.DataFrame) -> str:
    lines = []
    t = df[df.ok == 1].pivot_table(index=["source", "split"], columns="grade", values="image_id", aggfunc="count", fill_value=0)
    lines.append(t.to_string())
    lines.append(f"\nnear-duplicate images: {int(df.has_dup.sum())}; DDR images moved to keep each eye in one split: "
                 f"{int(df['moved_for_leakage'].sum()) if 'moved_for_leakage' in df else 0}; external images duplicating dev: {int(df.dup_of_dev.sum())}; "
                 f"conflicting-label duplicates: {int(df.label_conflict.sum())}; unreadable: {int((df.ok != 1).sum())}")
    return "\n".join(lines)
