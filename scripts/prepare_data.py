"""Build data/processed/manifest.csv from whatever datasets are downloaded.

Usage:  python scripts/prepare_data.py [--workers 12]
Re-running is cheap: cached crops are reused.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drishti.config import PROCESSED  # noqa: E402
from drishti.data import prepare, sources  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    df = sources.all_grading_sources()
    if df.empty:
        sys.exit("no datasets found under data/raw")
    print(df.groupby(["source", "orig_split"]).size().to_string())
    df, thumbs = prepare.cache_crops(df, workers=args.workers)
    bad = df[df.ok != 1]
    if len(bad):
        print(f"{len(bad)} unreadable images, e.g. {bad.path.iloc[0]}: {bad.error.iloc[0]}")
    df = df[df.ok == 1].reset_index(drop=True)
    df = prepare.assign_splits(df, thumbs)
    import numpy as np
    np.save(PROCESSED / "thumbs.npy", np.stack([thumbs[i] for i in df.image_id]))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "manifest.csv"
    # Keep any extra-resolution cache columns a previous run of
    # scripts/cache_size.py added. They are keyed by image id, not by row
    # order, so adding a dataset here does not invalidate them --- and
    # silently dropping them would break training at that resolution with a
    # confusing "no such column" hours later.
    if out.exists():
        import pandas as pd_
        prev = pd_.read_csv(out)
        extra = [c for c in prev.columns if c.startswith("cache") and c != "cache" and c not in df.columns]
        if extra:
            df = df.merge(prev[["image_id"] + extra], on="image_id", how="left")
            for c in extra:
                df[c] = df[c].fillna("")
            print(f"carried over {extra} for {int((df[extra[0]] != '').sum())} images")
    df.to_csv(out, index=False)
    print(prepare.summarise(df))
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
