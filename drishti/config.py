"""Paths and constants shared by every stage.

Datasets live outside the repository. DRISHTI_DATA overrides the default
location, which is ../data relative to the repository root.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DRISHTI_DATA", REPO.parent / "data")).resolve()
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RUNS = Path(os.environ.get("DRISHTI_RUNS", REPO / "runs")).resolve()
RESULTS = REPO / "results"

# Grader input: the field of view is cropped, squared and resized to this.
GRADE_SIZE = 512
# Lesion / landmark scale: field-of-view diameter in pixels. Messidor-2 and
# most DDR images are natively 1000-1500 px across the field of view, so
# 1440 keeps microaneurysms at 2-7 px without upsampling most test images.
LESION_SIZE = 1440

ICDR_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
REFERABLE_GRADE = 2  # ICDR grade 2 or worse is referable

LESIONS = ["MA", "HE", "EX", "SE"]  # microaneurysm, haemorrhage, hard / soft exudate
SEG_CHANNELS = LESIONS + ["OD"]  # optic disc is a fifth, partially labelled channel

SEED = 20260919
