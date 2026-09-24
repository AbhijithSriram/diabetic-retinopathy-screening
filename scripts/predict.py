"""Score a trained grader on splits it has not been scored on yet.

`train.py` writes `preds.csv` for whatever splits existed in the manifest
when it ran. When a dataset is added afterwards --- ODIR arrived after four
runs were already trained --- the earlier runs have no predictions on it,
and comparing candidates on the new set is impossible until they do.

This loads `runs/<name>/best.pt`, scores the requested splits with the same
fitted temperature training used, and merges the rows into the existing
`preds.csv`. Rows already present are left alone unless `--refresh` is
given, so re-running this is cheap and cannot quietly change a number that
has already been reported.

`--tta flip` (the default) reproduces exactly what training wrote.
`--tta dihedral` averages all eight orientations, which is worth a little
accuracy but makes the run incomparable with one scored the other way --- so
it is all runs in a comparison or none of them.

    python scripts/predict.py --name v2s_512 --splits devext
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drishti.config import PROCESSED, RUNS  # noqa: E402
from drishti.grading.model import Grader, class_probs, referable_prob  # noqa: E402
from drishti.grading.train import make_loader, predict_logits  # noqa: E402

KEEP = ["image_id", "source", "split", "grade", "dme", "gradable", "exam_id", "eye", "dup_of_dev"]


@torch.no_grad()
def predict_dihedral_tempered(model, loader, device, temp: float = 1.0):
    """Average over the eight rotations and reflections of the square.

    A fundus crop has no canonical orientation --- training already rotates
    through a full turn --- so all eight are equally valid views of the same
    eye, and averaging them cancels part of the model's orientation noise.
    Eight times the inference cost, which is nothing next to training.

    Averaging happens on the class distribution, not on the CORN logits:
    those are conditional probabilities and their mean has no meaning.
    """
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        acc = None
        for k in range(4):
            xr = torch.rot90(x, k, dims=[2, 3])
            for flip in (False, True):
                xi = torch.flip(xr, dims=[3]) if flip else xr
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = class_probs(model(xi).float() / temp)
                acc = p if acc is None else acc + p
        out.append((acc / 8).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--splits", default="devext", help="comma-separated manifest splits")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--refresh", action="store_true", help="re-score rows that are already present")
    ap.add_argument("--tta", choices=["flip", "dihedral"], default="flip",
                    help="flip matches what training wrote; dihedral averages all eight "
                         "orientations and must be applied to every run being compared")
    args = ap.parse_args()

    run = RUNS / args.name
    ck = torch.load(run / "best.pt", map_location="cuda", weights_only=False)
    size = int(ck.get("size", 512))
    temp = float(ck.get("temperature", 1.0))
    model = Grader(ck.get("backbone", "efficientnet_b0"), pretrained=False, img_size=size)
    model.load_state_dict(ck["state"])
    model = model.to("cuda").to(memory_format=torch.channels_last).eval()
    print(f"{args.name}: {ck.get('backbone')} at {size} px, temperature {temp:.2f}")

    df = pd.read_csv(PROCESSED / "manifest.csv")
    df = df[(df.ok == 1) & df.split.isin([s.strip() for s in args.splits.split(",")])]
    if df.empty:
        sys.exit(f"no manifest rows in splits {args.splits}")

    old = pd.read_csv(run / "preds.csv") if (run / "preds.csv").exists() else None
    if old is not None and not args.refresh:
        df = df[~df.image_id.isin(set(old.image_id))]
        if df.empty:
            print("every requested row is already scored; nothing to do")
            return
    print(f"scoring {len(df)} images")

    col = f"cache{size}"
    if size != 512 and not (col in df and (df[col].astype(str) != "").all()):
        sys.exit(f"no {col} column for these rows: run scripts/cache_size.py --size {size} first")

    loader = make_loader(df.assign(grade=0), False, args.bs, 0, size, graham=ck.get("graham", True))
    res = df[[c for c in KEEP if c in df]].copy()
    for c in KEEP:
        if c not in res:
            res[c] = None

    if args.tta == "dihedral":
        # temperature divides each view's logits before they become
        # probabilities, so it cannot be applied after the average
        probs = predict_dihedral_tempered(model, loader, "cuda", temp)
        for k in range(5):
            res[f"p{k}"] = probs[:, k]
        res["p_ref"] = probs[:, 2:].sum(axis=1)
        for k in range(4):
            res[f"logit{k}"] = np.nan          # no single set of logits to record
    else:
        lg = predict_logits(model, loader, "cuda")
        t = torch.tensor(lg) / temp
        probs = class_probs(t).numpy()
        for k in range(5):
            res[f"p{k}"] = probs[:, k]
        res["p_ref"] = referable_prob(t).numpy()
        for k in range(4):
            res[f"logit{k}"] = lg[:, k]

    if old is not None:
        old = old[~old.image_id.isin(set(res.image_id))]
        res = pd.concat([old, res], ignore_index=True)
    res.to_csv(run / "preds.csv", index=False)
    print(f"wrote {run / 'preds.csv'} ({len(res)} rows total)")


if __name__ == "__main__":
    main()
