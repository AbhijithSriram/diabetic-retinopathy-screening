"""Rank candidate graders without looking at a reported test set.

The question "which of these models should we ship" has to be answered
somewhere, and answering it on IDRiD or Messidor-2 would turn those from
external tests into validation sets --- quietly, and irreversibly, because
you cannot un-see a number. So candidates are ranked here on:

  valA / valB   in-distribution, the same cameras as training
  DDR / APTOS test   in-distribution, held out
  ODIR (devext)      a different country and camera mix, never trained on

ODIR is the column that matters. Its labels are read off free-text findings
and are noisier than ours, which depresses every model's score by roughly
the same amount --- fine for ranking, not for quoting.

    python scripts/compare_runs.py --runs b0_final,v2s_512,b0_768,b0_sim,b0_all

Writes results/candidates.json so the ranking that drove the decision is on
record, not just in somebody's memory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drishti.config import RESULTS, RUNS  # noqa: E402
from drishti.eval.metrics import auc_ci, ece, expected_grade, qwk  # noqa: E402

PCOLS = [f"p{k}" for k in range(5)]
SETS = {
    "valA": lambda d: d.split == "valA",
    "valB": lambda d: d.split == "valB",
    "DDR test": lambda d: (d.source == "ddr") & (d.split == "test"),
    "APTOS test": lambda d: (d.source == "aptos") & (d.split == "test"),
    "ODIR (cross-camera)": lambda d: d.split == "devext",
}


def block(d: pd.DataFrame) -> dict | None:
    d = d[(d.grade >= 0) & (d.gradable == 1)]
    if "dup_of_dev" in d:
        d = d[d.dup_of_dev != 1]
    if len(d) < 50 or d.grade.nunique() < 2:
        return None
    y = d.grade.to_numpy()
    ref = y >= 2
    groups = d.exam_id.fillna(pd.Series(d.image_id.values, index=d.index)).to_numpy()
    a, lo, hi = auc_ci(ref, d.p_ref.to_numpy(), groups=groups, n_boot=500)
    return {"n": int(len(d)), "prevalence": round(float(ref.mean()), 3),
            "auc": round(float(a), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "qwk": round(qwk(y, expected_grade(d[PCOLS].to_numpy())), 4),
            "ece": round(ece(ref, d.p_ref.to_numpy()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="candidates.json")
    args = ap.parse_args()

    report = {}
    for name in [n.strip() for n in args.runs.split(",")]:
        p = RUNS / name / "preds.csv"
        if not p.exists():
            print(f"{name:12s} no preds.csv yet, skipping")
            continue
        d = pd.read_csv(p)
        cfg = {}
        ap_ = RUNS / name / "args.json"
        if ap_.exists():
            a = json.loads(ap_.read_text())
            cfg = {k: a.get(k) for k in ("backbone", "size", "sources", "capture_p", "epochs")
                   if a.get(k) is not None}
        report[name] = {"config": cfg, "sets": {}}
        for label, sel in SETS.items():
            b = block(d[sel(d)])
            if b:
                report[name]["sets"][label] = b

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(report, indent=2))

    cols = list(SETS)
    print(f"\n{'run':14s}" + "".join(f"{c[:19]:>21s}" for c in cols))
    for name, r in report.items():
        row = f"{name:14s}"
        for c in cols:
            b = r["sets"].get(c)
            row += f"{(f'{b['auc']:.4f}' if b else '-'):>21s}"
        print(row)
    print("\nreferable-DR AUC. ODIR is the only column that says anything about a "
          "camera the model has not seen.")
    best = max((r for r in report.items() if "ODIR (cross-camera)" in r[1]["sets"]),
               key=lambda kv: kv[1]["sets"]["ODIR (cross-camera)"]["auc"], default=None)
    if best:
        b = best[1]["sets"]["ODIR (cross-camera)"]
        print(f"best on ODIR: {best[0]}  AUC {b['auc']:.4f} ({b['ci'][0]:.4f}-{b['ci'][1]:.4f}), n={b['n']}")
    print(f"wrote {RESULTS / args.out}")


if __name__ == "__main__":
    main()
