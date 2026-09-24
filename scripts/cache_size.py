"""Build a second crop cache at a different resolution.

Microaneurysms are a handful of pixels across. At 512 px over the field of
view they are one or two pixels, which is at the edge of what a convolution
can see at all; the lesion model already runs at 1440 for that reason. This
builds the same crops the manifest already describes, at a larger size, and
records them in a new column so the grader can be trained at either
resolution from the same splits.

The crop geometry is identical to the one in the manifest, so nothing about
the splits, the duplicate groups or the leakage controls changes.

    python scripts/cache_size.py --size 768 [--workers 6] [--sources ddr,aptos]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drishti.config import PROCESSED                       # noqa: E402
from drishti.preprocess import prepare, read_rgb, write_image  # noqa: E402


def build(row, size: int, cache: Path) -> tuple[str, str]:
    out = cache / row.source / f"{row.image_id}.jpg"
    if out.exists():
        return row.image_id, str(out)
    try:
        rgb = read_rgb(row.path)
        prep = prepare(rgb, size)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_image(out, prep.image)
        return row.image_id, str(out)
    except Exception as e:
        print(f"  {row.image_id}: {e}")
        return row.image_id, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sources", default="", help="comma-separated; default is every source")
    args = ap.parse_args()

    man = PROCESSED / "manifest.csv"
    df = pd.read_csv(man)
    todo = df[df.ok == 1]
    if args.sources:
        todo = todo[todo.source.isin(args.sources.split(","))]
    cache = PROCESSED / f"g{args.size}"
    print(f"{len(todo)} images -> {cache}")

    done = {}
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (iid, path) in enumerate(ex.map(lambda r: build(r, args.size, cache),
                                               todo.itertuples(index=False))):
            done[iid] = path
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)

    col = f"cache{args.size}"
    # Merge, never overwrite. With --sources this run only knows about some
    # of the manifest; assigning the mapping wholesale would blank the column
    # for every other dataset and silently un-cache work done earlier.
    found = df.image_id.map(done)
    df[col] = found.fillna(df[col]) if col in df else found.fillna("")
    df[col] = df[col].fillna("")
    df.to_csv(man, index=False)
    n = int((df[col] != "").sum())
    sample = df[df[col] != ""][col].iloc[0]
    print(f"{n} crops cached; manifest column {col} written")
    print("size check:", cv2.imread(sample).shape)


if __name__ == "__main__":
    main()
