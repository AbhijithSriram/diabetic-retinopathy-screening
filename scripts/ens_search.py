"""Rank ensembles on ODIR (devext) only. Never touches IDRiD / Messidor-2 rows.

Combines members exactly as scripts/ensemble.py does (mean of class
probabilities, renormalised, P(ref) = p2+p3+p4). For each combination of
2-4 members reports valA AUC, ODIR AUC with a bootstrap CI, and whether every
member is an EfficientNet (all operators built-in in MATLAB; ConvNeXt needs
custom Transpose/Reshape layers).

    python ens_search.py [run ...]      default: every finished grader run below
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drishti.config import RESULTS, RUNS  # noqa: E402
OUT = RESULTS / "ensemble_search.txt"
DEFAULT = ["v2s_all", "b0_768_all", "cnx_all_wd", "b0_all",
           "v2s_e15", "cnx_e10", "b0_768_e10", "v2s768_e8", "cnxs_e8"]
P = [f"p{k}" for k in range(5)]


def load(n):
    f = RUNS / n / "preds.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = d[d.split.isin(["valA", "devext", "devext2"]) & (d.grade >= 0)]
    d = d[~((d.split == "devext2") & (d.dup_of_dev == 1))].set_index("image_id").sort_index()
    return d


def backbone(n):
    a = json.loads((RUNS / n / "args.json").read_text())
    return a.get("backbone", "")


def auc_ci(y, p, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    a = roc_auc_score(y, p)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            bs.append(roc_auc_score(y[i], p[i]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    names = sys.argv[1:] or DEFAULT
    runs = {n: d for n in names if (d := load(n)) is not None}
    common = None
    for d in runs.values():
        common = d.index if common is None else common.intersection(d.index)
    runs = {n: d.loc[common] for n, d in runs.items()}
    ref = next(iter(runs.values()))
    va, od = (ref.split == "valA").to_numpy(), (ref.split == "devext").to_numpy()
    dd = (ref.split == "devext2").to_numpy()
    y = (ref.grade >= 2).to_numpy()
    rows = []
    for k in range(1, min(4, len(runs)) + 1):
        for combo in itertools.combinations(runs, k):
            pr = sum(runs[n][P].to_numpy() for n in combo) / k
            pr = pr / pr.sum(1, keepdims=True)
            pref = pr[:, 2:].sum(1)
            a_od, lo, hi = auc_ci(y[od], pref[od], n_boot=200 if k > 1 else 300)
            a_dd = roc_auc_score(y[dd], pref[dd]) if dd.sum() and y[dd].min() != y[dd].max() else np.nan
            rows.append({"members": "+".join(combo), "k": k,
                         "matlab_builtin": all("convnext" not in backbone(n) and "retfound" not in backbone(n)
                                               for n in combo),
                         "valA": roc_auc_score(y[va], pref[va]), "ODIR": a_od, "ODIR_lo": lo, "ODIR_hi": hi,
                         "DeepDRiD": a_dd, "mean_sel": np.nanmean([a_od, a_dd])})
    # rank on the mean of the two selection sets when DeepDRiD is scored
    key = "mean_sel" if dd.sum() else "ODIR"
    t = pd.DataFrame(rows).sort_values(key, ascending=False)
    txt = (f"{len(runs)} runs, {int(od.sum())} ODIR / {int(dd.sum())} DeepDRiD / {int(va.sum())} valA images; "
           f"ranked by {key}\n"
           + t.head(40).to_string(index=False, float_format=lambda v: f"{v:.4f}")
           + "\n\nBest all-EfficientNet (fully built-in MATLAB) ensembles:\n"
           + t[t.matlab_builtin].head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}")
           + "\n\nSingle models:\n"
           + t[t.k == 1].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    OUT.write_text(txt + "\n")
    t.to_csv(OUT.with_suffix(".csv"), index=False)
    print(txt)


if __name__ == "__main__":
    main()
