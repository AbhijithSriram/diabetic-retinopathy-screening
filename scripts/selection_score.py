"""Score runs on ODIR (devext) and valA only -- the selection sets. Never
reads IDRiD / Messidor-2 rows. Appends to overnight/odir_scores.txt.
    python selection_score.py run1 run2 ...
"""
import sys
import sys, datetime
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drishti.config import RESULTS, RUNS  # noqa: E402
out = RESULTS / "selection_scores.txt"
for n in sys.argv[1:]:
    f = RUNS / n / "preds.csv"
    if not f.exists():
        line = f"{n}: no preds.csv"
    else:
        d = pd.read_csv(f)
        d = d[d.split.isin(["devext", "devext2", "valA"])]
        r = []
        for s, label in [("valA", "valA"), ("devext", "ODIR"), ("devext2", "DeepDRiD")]:
            x = d[(d.split == s) & (d.grade >= 0)]
            if s == "devext2":
                x = x[x.dup_of_dev != 1]
            if len(x) and x.grade.ge(2).nunique() == 2:
                r.append(f"{label} {roc_auc_score(x.grade >= 2, x.p_ref):.4f} (n={len(x)})")
        line = f"{n}: " + "  ".join(r)
    line = f"[{datetime.datetime.now():%m-%d %H:%M}] {line}"
    print(line)
    with out.open("a") as fh:
        fh.write(line + "\n")
