"""Leave-one-dataset-out report: referable-DR AUC of each fold on every
development-side held-out set. Never reads IDRiD / Messidor-2 rows.

Folds (all EfficientNet-B0 at 512, same recipe and seed):
  lodo_all          ddr + aptos + eyepacs
  lodo_no_ddr       aptos + eyepacs      -> DDR test is an unseen hospital system
  lodo_no_aptos     ddr + eyepacs        -> APTOS test is an unseen clinic
  lodo_no_eyepacs   ddr + aptos          -> measures what EyePACS adds everywhere
"""
from pathlib import Path

import sys
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drishti.config import RESULTS, RUNS  # noqa: E402
FOLDS = ["lodo_all", "lodo_no_ddr", "lodo_no_aptos", "lodo_no_eyepacs"]
SETS = {
    "DDR test": lambda d: (d.source == "ddr") & (d.split == "test"),
    "APTOS test": lambda d: (d.source == "aptos") & (d.split == "test"),
    "ODIR": lambda d: d.split == "devext",
    "DeepDRiD": lambda d: (d.split == "devext2") & (d.dup_of_dev != 1),
}
rows = []
for f in FOLDS:
    p = RUNS / f / "preds.csv"
    if not p.exists():
        continue
    d = pd.read_csv(p)
    d = d[(d.grade >= 0) & ~d.source.isin(["idrid", "messidor2"])]
    r = {"fold": f}
    for name, sel in SETS.items():
        x = d[sel(d)]
        r[name] = roc_auc_score(x.grade >= 2, x.p_ref) if len(x) and (x.grade >= 2).nunique() == 2 else float("nan")
    rows.append(r)
t = pd.DataFrame(rows).set_index("fold")
if "lodo_all" in t.index:
    delta = (t - t.loc["lodo_all"]).add_suffix(" Δ")
    t = pd.concat([t, delta], axis=1)
txt = t.to_string(float_format=lambda v: f"{v:+.4f}" if abs(v) < 0.5 else f"{v:.4f}")
print(txt)
(RESULTS / "lodo_report.txt").write_text(txt + "\n")
