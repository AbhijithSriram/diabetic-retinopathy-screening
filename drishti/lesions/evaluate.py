"""Lesion evaluation and threshold selection.

1. Thresholds: per channel, the pixel-F1-optimal threshold on DDR *valid*
   (never on a test set). Written to runs/<name>/thresholds.json.
2. Pixel AUPR per lesion on IDRiD test (27 images, the IDRiD challenge test
   set) and DDR test (225), inside the field of view. Computed from
   1000-bin probability histograms so the full-resolution maps never need
   to be held in memory at once.
3. Image-level presence (is there any lesion of this type?), which is what
   the rule engine actually consumes, with Wilson intervals.

    python -m drishti.lesions.evaluate --name seg_r34
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from ..config import RESULTS, RUNS, SEG_CHANNELS
from ..eval.metrics import binary_report
from ..preprocess import read_rgb
from .data import load_seg_manifest, unpack
from .infer import MIN_AREA, components, load_model, predict_logits

# Histograms are taken on the LOGIT scale: the Tversky loss pushes confident
# pixels to probabilities within 1e-4 of 1, where equal-width probability bins
# would lump true and false positives together and understate AUPR.
BINS = 4000
LO, HI = -20.0, 20.0
EDGES = np.linspace(LO, HI, BINS + 1)


def _hist(logit, target, sel):
    x = np.clip(logit, LO, HI - 1e-6)
    pos = np.histogram(x[sel & (target > 0)], EDGES)[0]
    neg = np.histogram(x[sel & (target == 0)], EDGES)[0]
    return pos, neg


def aupr(pos, neg) -> float:
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    P = pos.sum()
    if P == 0:
        return float("nan")
    rec = tp / P
    prec = tp / np.maximum(tp + fp, 1)
    r_prev = np.concatenate([[0], rec[:-1]])
    return float(np.sum((rec - r_prev) * prec))


def best_f1_threshold(pos, neg) -> float:
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    fn = pos.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    k = int(np.argmax(f1))
    logit_thr = float(EDGES[BINS - 1 - k])  # lower edge of the last included bin
    return float(1.0 / (1.0 + np.exp(-logit_thr)))


def calibrate_peaks(model, rows, thr, min_precision=0.75, dilate=7):
    """Component-level calibration on development data (DDR valid).

    For each lesion type, every predicted component (pixel threshold thr)
    is a true positive if it touches a (dilated) annotated lesion of that
    type. Returns the smallest peak probability at which components are
    real at least `min_precision` of the time."""
    from .infer import components
    recs = {k: [] for k in SEG_CHANNELS[:4]}
    kern = np.ones((dilate, dilate), np.uint8)
    for r in rows:
        img = read_rgb(r["image"])
        tgt = unpack(cv2.imdecode(np.fromfile(r["mask"], np.uint8), cv2.IMREAD_UNCHANGED))
        maps = 1.0 / (1.0 + np.exp(-predict_logits(model, img).astype(np.float64)))
        for L in components(maps, thr, confident_only=False):
            c = SEG_CHANNELS.index(L.kind)
            x, y, w, h = L.bbox
            hit = cv2.dilate(tgt[c].astype(np.uint8), kern)[y:y + h, x:x + w].any()
            recs[L.kind].append((L.peak, bool(hit)))
    out, stats = {}, {}
    for k, v in recs.items():
        v = sorted(v, key=lambda t: -t[0])
        peaks = np.array([t[0] for t in v]); hits = np.array([t[1] for t in v], float)
        prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        ok = np.nonzero(prec >= min_precision)[0]
        # lowest peak whose cumulative precision (all components at or above it) is still >= target
        out[f"peak_{k}"] = float(peaks[ok.max()]) if len(ok) else 1.0
        stats[k] = {"components": len(v), "kept": int(ok.max() + 1) if len(ok) else 0, "precision_all": round(float(hits.mean()), 3) if len(v) else None}
    return out, stats


def run_split(model, rows, thr=None, collect_presence=True):
    hists = {ch: [np.zeros(BINS), np.zeros(BINS)] for ch in SEG_CHANNELS}
    presence = {ch: ([], []) for ch in SEG_CHANNELS[:4]}
    for r in rows:
        img = read_rgb(r["image"])
        bits = cv2.imdecode(np.fromfile(r["mask"], np.uint8), cv2.IMREAD_UNCHANGED)
        tgt = unpack(bits)
        logits = predict_logits(model, img)
        maps = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        fov = img.max(axis=2) > 8
        for c, ch in enumerate(SEG_CHANNELS):
            if not r["known"][c]:
                continue
            p, n = _hist(logits[c], tgt[c], fov)
            hists[ch][0] += p
            hists[ch][1] += n
        if thr is not None and collect_presence:
            found = {L.kind for L in components(maps, thr)}
            for c, ch in enumerate(SEG_CHANNELS[:4]):
                truth = tgt[c].sum() >= MIN_AREA[ch]
                presence[ch][0].append(truth)
                presence[ch][1].append(ch in found)
    return hists, presence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="seg_r34")
    args = ap.parse_args()
    run = RUNS / args.name
    model, _ = load_model(run / "best.pt")
    df = load_seg_manifest()

    valid = df[(df.source == "ddr") & (df.split == "valid")].to_dict("records")
    hv, _ = run_split(model, valid)
    thr = {ch: best_f1_threshold(*hv[ch]) for ch in SEG_CHANNELS[:4]}
    thr["OD"] = 0.5
    peaks, peak_stats = calibrate_peaks(model, valid, thr)
    thr.update(peaks)
    (run / "thresholds.json").write_text(json.dumps(thr, indent=2))
    print("thresholds (DDR valid, pixel-F1 + component peak for >=75% precision):", thr)
    print("component calibration:", peak_stats)

    report = {"thresholds": thr, "component_calibration": peak_stats, "note": "pixel AUPR inside the FOV at LESION_SIZE scale; IDRiD challenge scored at native resolution"}
    for name, sel in [("idrid_test", (df.source == "idrid") & (df.split == "test")),
                      ("ddr_test", (df.source == "ddr") & (df.split == "test"))]:
        rows = df[sel].to_dict("records")
        if not rows:
            continue
        h, pres = run_split(model, rows, thr)
        out = {"n_images": len(rows), "aupr": {ch: round(aupr(*h[ch]), 4) for ch in SEG_CHANNELS if h[ch][0].sum() > 0}}
        out["presence"] = {}
        for ch, (truth, pred) in pres.items():
            if len(truth):
                rep = binary_report(truth, pred)
                out["presence"][ch] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in rep.items()}
        report[name] = out
        print(name, json.dumps(out["aupr"]))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"lesions_{args.name}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
