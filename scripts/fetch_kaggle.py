"""Download the Kaggle-hosted pieces (needs the user's own Kaggle API token
and, for APTOS, the competition rules accepted on kaggle.com by that user).

- APTOS 2019: only train.csv and train_images/ are kept (the competition's
  test labels were never released, so its test images are useless here).
- Messidor-2 adjudicated grades (Krause et al. 2018): messidor_data.csv.

    python scripts/fetch_kaggle.py [--aptos] [--messidor]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drishti.config import RAW  # noqa: E402

KAGGLE = [shutil.which("kaggle") or "kaggle"]  # the Kaggle CLI on PATH


def run(args):
    print(" ".join(args))
    subprocess.run(args, check=True)


def aptos():
    dest = RAW / "aptos"
    dest.mkdir(parents=True, exist_ok=True)
    run(KAGGLE + ["competitions", "download", "-c", "aptos2019-blindness-detection", "-p", str(dest)])
    z = dest / "aptos2019-blindness-detection.zip"
    with zipfile.ZipFile(z) as f:
        keep = [n for n in f.namelist() if n == "train.csv" or n.startswith("train_images/")]
        for n in keep:
            f.extract(n, dest)
    z.unlink()
    print(f"APTOS: {len(keep) - 1} training images in {dest}")


def messidor():
    dest = RAW / "messidor2"
    dest.mkdir(parents=True, exist_ok=True)
    run(KAGGLE + ["datasets", "download", "-d", "google-brain/messidor2-dr-grades", "-p", str(dest), "--unzip"])
    print("Messidor-2 grades:", [p.name for p in dest.glob("*.csv")])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aptos", action="store_true")
    ap.add_argument("--messidor", action="store_true")
    a = ap.parse_args()
    if not (a.aptos or a.messidor):
        a.aptos = a.messidor = True
    if a.messidor:
        messidor()
    if a.aptos:
        aptos()
