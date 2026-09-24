"""Run the lesion model + landmarks + rule engine over external images, then
ask one question honestly: does the rule engine earn its place?

`extract` writes results/cache/rules_<seg>.csv (one row per image: lesion
counts, rule evidence grade, DME flag).

`ablation` compares, on each external set, at the frozen valB policy:
  A1  CNN, single threshold (refer / recall)
  A2  CNN bands (recall / review / refer)
  A3  A2 + lesion evidence >= grade 2 sends would-be recalls to review
  A4  A3 + hard exudate within 1 DD of fovea sends to review
  A2@A4-workload  CNN bands with lambda_low moved so review+refer equals A4's
The last row is the fair test (a matched-workload comparison):
if A4 does not miss fewer referable eyes than simply widening the CNN's own
review band to the same workload, the rules add explanation but not safety,
and we say so.

    python -m drishti.eval.rules_eval extract --seg seg_r34 --lmk lmk_r18
    python -m drishti.eval.rules_eval ablation --grader b0_512 --seg seg_r34
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from ..config import GRADE_SIZE, LESION_SIZE, PROCESSED, RESULTS, RUNS
from ..conformal import triage as T
from ..eval import metrics as M
from ..landmarks import landmarks as LM
from ..lesions.infer import components, load_model, optic_disc, predict_maps
from ..preprocess import prepare, read_rgb
from ..rules.icdr import assess

PCOLS = [f"p{k}" for k in range(5)]


def seg_overlap_ids() -> set:
    """IDRiD grading images that are the same photo as an image the lesion
    model trained on (different file names in IDRiD's sub-sets). Same
    retina-signature correlation test as the duplicate detector."""
    import cv2
    from ..data.prepare import retina_signature
    man = pd.read_csv(PROCESSED / "manifest.csv")
    thumbs = np.load(PROCESSED / "thumbs.npy")
    seg = pd.read_json(PROCESSED / "seg_manifest.json", orient="records")
    seg = seg[(seg.source == "idrid") & (seg.split == "train")]
    st = np.stack([retina_signature(cv2.resize(read_rgb(p), (GRADE_SIZE, GRADE_SIZE), interpolation=cv2.INTER_AREA))[1] for p in seg.image])
    idr = man.index[man.source == "idrid"].to_numpy()
    corr = thumbs[idr] @ st.T
    return set(man.image_id.to_numpy()[idr[(corr >= 0.8).any(axis=1)]])


def extract(seg_name: str, lmk_name: str):
    man = pd.read_csv(PROCESSED / "manifest.csv")
    ext = man[(man.role == "external") & (man.ok == 1)]
    seg_model, thr = load_model(RUNS / seg_name / "best.pt")
    lmk = LM.load(lmk_name)
    overlap = seg_overlap_ids() if (ext.source == "idrid").any() else set()
    rows = []
    for i, r in enumerate(ext.itertuples()):
        rgb = read_rgb(r.path)
        big = prepare(rgb, LESION_SIZE)
        maps = predict_maps(seg_model, big.image)
        les = components(maps, thr)
        od = optic_disc(maps, thr.get("OD", 0.5))
        small = read_rgb(r.cache)
        lm = LM.predict(lmk, small)
        s = LESION_SIZE / GRADE_SIZE
        fovea = (lm["fovea"][0] * s, lm["fovea"][1] * s)
        a = assess(les, od=od, fovea=fovea, fovea_conf=lm["fovea"][2])
        rows.append({"image_id": r.image_id, "source": r.source, "in_seg_train": int(r.image_id in overlap),
                     **{f"n_{k}": v for k, v in a["counts"].items()},
                     "evidence_grade": a["evidence_grade"], "dme_risk": a["dme_risk"],
                     "lesion_flag": a["lesion_flag"], "dme_flag": a["dme_flag"],
                     "fovea_conf": round(lm["fovea"][2], 3), "disc_source": a["disc_diameter_source"]})
        if i % 200 == 0:
            print(i, len(ext), flush=True)
    out = RESULTS / "cache"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"rules_{seg_name}.csv", index=False)
    print("wrote", out / f"rules_{seg_name}.csv")


def _outcomes(p, probs, lam_low, lam_high, lesion, dme, use_lesion, use_dme):
    out = np.where(p >= lam_high, "refer", np.where(p < lam_low, "recall", "review"))
    bump = np.zeros(len(p), bool)
    if use_lesion:
        bump |= lesion
    if use_dme:
        bump |= dme
    out = np.where((out == "recall") & bump, "review", out)
    return out


def _row(name, out, ref):
    missed = int(((out == "recall") & ref).sum())
    return {"config": name, "recall": round(float((out == "recall").mean()), 4), "review": round(float((out == "review").mean()), 4),
            "refer": round(float((out == "refer").mean()), 4), "referable_missed": missed, "referable": int(ref.sum()),
            "missed_rate_ci": [round(v, 4) for v in M.wilson(missed, int(ref.sum()))]}


def ablation(grader: str, seg_name: str):
    preds = pd.read_csv(RUNS / grader / "preds.csv")
    pol = T.TriagePolicy(**json.loads((RUNS / grader / "policy.json").read_text()))
    rules = pd.read_csv(RESULTS / "cache" / f"rules_{seg_name}.csv")
    report = {"policy": pol.as_dict()}
    for src in ("idrid", "messidor2", "mbrset"):
        d = preds[(preds.source == src) & (preds.grade >= 0) & (preds.gradable == 1) & (preds.dup_of_dev != 1)]
        d = d.merge(rules, on=["image_id", "source"], how="inner")
        if src == "idrid":
            d = d[d.in_seg_train == 0]
        if len(d) < 50:
            continue
        p, probs, ref = d.p_ref.to_numpy(), d[PCOLS].to_numpy(), d.grade.to_numpy() >= 2
        les, dme = d.lesion_flag.to_numpy(bool), d.dme_flag.to_numpy(bool)
        rows = []
        a1 = np.where(p >= pol.lambda_op, "refer", "recall")
        rows.append(_row("A1 CNN single threshold", a1, ref))
        rows.append(_row("A2 CNN bands", _outcomes(p, probs, pol.lambda_low, pol.lambda_high, les, dme, False, False), ref))
        rows.append(_row("A3 + lesion evidence", _outcomes(p, probs, pol.lambda_low, pol.lambda_high, les, dme, True, False), ref))
        a4 = _outcomes(p, probs, pol.lambda_low, pol.lambda_high, les, dme, True, True)
        rows.append(_row("A4 + DME rule", a4, ref))
        # matched workload: widen the CNN's review band until it sends as many to review/refer as A4
        target = float((a4 != "recall").mean())
        lam = float(np.quantile(p, 1 - target)) if target < 1 else 0.0
        rows.append(_row("A2 at A4's workload (CNN band widened)", _outcomes(p, probs, lam, pol.lambda_high, les, dme, False, False), ref))
        # evidence grade agreement (explanation fidelity)
        agree = float(np.mean((d.evidence_grade >= 2) == ref))
        report[src] = {"n": int(len(d)), "rows": rows, "rule_evidence_vs_truth_referable_accuracy": round(agree, 4),
                       "rule_evidence_referable": binary_block(ref, d.evidence_grade.to_numpy() >= 2)}
        print(src, len(d))
        for r in rows:
            print("   ", r)
    (RESULTS / f"ablation_{grader}_{seg_name}.json").write_text(json.dumps(report, indent=2))


def binary_block(y, pred):
    r = M.binary_report(y, pred)
    return {k: ([round(float(x), 4) for x in v] if isinstance(v, tuple) else v) for k, v in r.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "ablation"])
    ap.add_argument("--seg", default="seg_r34")
    ap.add_argument("--lmk", default="lmk_r18")
    ap.add_argument("--grader", default="b0_512")
    a = ap.parse_args()
    extract(a.seg, a.lmk) if a.cmd == "extract" else ablation(a.grader, a.seg)
