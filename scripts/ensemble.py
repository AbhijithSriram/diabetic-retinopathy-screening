"""Average several trained graders into one set of predictions.

Averaging different seeds, backbones and resolutions is the cheapest
reliable gain left, and it steadies the operating point across cameras,
which is our actual weakness. The ensemble is assembled at the level of
`preds.csv`, not by loading models together, so that every downstream
stage --- conformal calibration, the triage policy, the evaluation, the
report --- runs on it unchanged and without knowing it is an ensemble.

    python scripts/ensemble.py --runs b0_all,v2s_512,b0_768 --name ens3

Class probabilities are averaged rather than logits. The CORN head's
logits are conditional (P(y > k | y > k-1)), so their average has no
meaning; the class distribution they imply does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drishti.config import RUNS  # noqa: E402

PCOLS = [f"p{k}" for k in range(5)]
KEYS = ["image_id", "source", "split", "grade", "dme", "gradable", "exam_id", "eye", "dup_of_dev"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run names")
    ap.add_argument("--name", required=True, help="name of the ensemble run directory")
    ap.add_argument("--weights", default="", help="comma-separated, default equal")
    args = ap.parse_args()

    names = [n.strip() for n in args.runs.split(",")]
    frames = []
    for n in names:
        p = RUNS / n / "preds.csv"
        if not p.exists():
            sys.exit(f"missing {p}")
        frames.append(pd.read_csv(p).set_index("image_id").sort_index())

    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    if len(common) < len(frames[0]):
        print(f"note: {len(frames[0]) - len(common)} images are not scored by every member; "
              f"keeping the {len(common)} they share")
    frames = [f.loc[common] for f in frames]

    w = [float(x) for x in args.weights.split(",")] if args.weights else [1.0] * len(frames)
    if len(w) != len(frames):
        sys.exit("--weights must have one entry per run")
    tot = sum(w)

    out = frames[0][[c for c in KEYS if c != "image_id"]].copy()
    for c in PCOLS:
        out[c] = sum(wi * f[c] for wi, f in zip(w, frames)) / tot
    # keep the probabilities a distribution, and derive the referral
    # probability from them so the two can never disagree
    s = out[PCOLS].sum(axis=1)
    for c in PCOLS:
        out[c] = out[c] / s
    out["p_ref"] = out[["p2", "p3", "p4"]].sum(axis=1)

    d = RUNS / args.name
    d.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(d / "preds.csv", index=False)
    (d / "args.json").write_text(json.dumps(
        {"ensemble_of": names, "weights": w, "n_images": int(len(out))}, indent=2))
    print(f"wrote {d / 'preds.csv'} ({len(out)} images from {len(names)} runs)")
    print("next: python -m drishti.eval.grader_eval --grader " + args.name)


if __name__ == "__main__":
    main()
