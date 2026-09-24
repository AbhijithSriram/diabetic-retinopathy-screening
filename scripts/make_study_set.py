"""Pick a grade-stratified reader-study set from external public images.

Messidor-2 (adjudicated grades) is preferred; IDRiD fills in if it is
missing. Writes app/study_set.csv (image_id, path, grade, dme, source).

    python scripts/make_study_set.py [--n 60]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drishti.config import PROCESSED, REPO, SEED  # noqa: E402

QUOTA = {0: 0.25, 1: 0.15, 2: 0.25, 3: 0.15, 4: 0.20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    m = pd.read_csv(PROCESSED / "manifest.csv")
    pool = m[(m.role == "external") & (m.grade >= 0) & (m.gradable == 1) & (m.dup_of_dev != 1)]
    src = "messidor2" if (pool.source == "messidor2").sum() >= a.n else "idrid"
    pool = pool[pool.source == src]
    picks = []
    for g, frac in QUOTA.items():
        sub = pool[pool.grade == g]
        picks.append(sub.sample(min(len(sub), round(frac * a.n)), random_state=SEED))
    out = pd.concat(picks)[["image_id", "path", "grade", "dme", "source"]]
    dest = REPO / "app" / "study_set.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest}: {len(out)} images from {src}", out.grade.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
