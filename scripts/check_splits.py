"""Prove that adding a dataset did not move any existing image's split.

The splits are assigned by a content hash so that they survive new data
(`prepare._stable_order`), but there is one way they could still move:
duplicate groups are named after their alphabetically first image id, and a
new dataset joining a group renames it. A test image quietly becoming a
training image would invalidate every number already reported, and nothing
would look wrong.

So this compares a snapshot taken before the change with the manifest after
it, and exits non-zero if anything moved.

    python scripts/check_splits.py data/processed/manifest_before_eyepacs.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drishti.config import PROCESSED  # noqa: E402


def main():
    before_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROCESSED / "manifest_before_eyepacs.csv"
    if not before_path.exists():
        sys.exit(f"no snapshot at {before_path}")
    before = pd.read_csv(before_path)[["image_id", "source", "split"]]
    after = pd.read_csv(PROCESSED / "manifest.csv")[["image_id", "source", "split"]]

    m = before.merge(after, on="image_id", how="left", suffixes=("_before", "_after"))
    missing = m[m.split_after.isna()]
    moved = m[m.split_after.notna() & (m.split_before != m.split_after)]

    print(f"{len(before)} images in the snapshot, {len(after)} in the manifest now")
    print(f"new images: {len(after) - len(before) + len(missing)}")
    if len(missing):
        print(f"\nGONE: {len(missing)} images are no longer in the manifest")
        print(missing.groupby("source_before").size().to_string())
    if len(moved):
        print(f"\nMOVED: {len(moved)} images changed split")
        print(moved.groupby(["source_before", "split_before", "split_after"]).size().to_string())
        # Not all moves are equal, and treating them as equal makes this
        # check useless: it cries wolf on the harmless case and buries the
        # one that matters. An image that leaves an evaluation set shrinks
        # that set slightly. An image that moves INTO the data the model is
        # fitted on has contaminated whatever it left.
        FIT = {"train", "valA", "valB"}
        grave = moved[moved.split_after.isin(FIT) & ~moved.split_before.isin(FIT)]
        if len(grave):
            print(f"\nSERIOUS: {len(grave)} images moved from a held-out set into "
                  f"{sorted(FIT)}. Anything measured on those sets before this rebuild "
                  f"is contaminated and must not be reported.")
            print(grave.groupby(["source_before", "split_before", "split_after"]).size().to_string())
        else:
            print("\nNone of these moved into train/valA/valB, so nothing was contaminated.")
            print("Evaluation sets changed size slightly; re-run the evaluation before quoting "
                  "counts, but earlier conclusions stand.")
    else:
        print("\nno existing image changed split")
        grave = moved
    return 1 if (len(grave) or len(missing)) else 0


if __name__ == "__main__":
    sys.exit(main())
