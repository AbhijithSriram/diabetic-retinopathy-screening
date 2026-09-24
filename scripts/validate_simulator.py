"""Does the capture simulator produce images that look, by measurement, like
images from cameras we did not train on?

The claim we want to be able to defend is narrow and testable: applying the
simulator to development data (DDR, APTOS: desktop mydriatic cameras) widens
the distribution of capture statistics so that it covers the external sets
(IDRiD, Messidor-2), which our clean development data does not.

    python scripts/validate_simulator.py [--n 400]

Writes results/simulator_stats.json and a contact sheet under
results/figures/. Nothing here touches labels, so it costs no test-set look.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drishti.augment import sample_profile, simulate_capture          # noqa: E402
from drishti.augment.stats import KEYS, capture_stats, coverage       # noqa: E402
from drishti.config import PROCESSED, RESULTS                         # noqa: E402
from drishti.preprocess import read_rgb                               # noqa: E402


def collect(paths, fn=None, rng=None, n_devices=24):
    """Statistics over a list of cached crops, optionally re-imaged first.
    Devices are drawn once and reused, as they would be in a real programme."""
    devices = [sample_profile(rng) for _ in range(n_devices)] if fn else []
    rows = []
    for i, p in enumerate(paths):
        img = read_rgb(p)
        if img.shape[0] != 512:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        if fn:
            img = fn(img, devices[i % len(devices)])
        s = capture_stats(img)
        if s:
            rows.append(s)
    return pd.DataFrame(rows)


def domain_auc(a: pd.DataFrame, b: pd.DataFrame, seed=0) -> float:
    """Cross-validated AUC of a classifier separating two groups on capture
    statistics alone. 0.5 means indistinguishable; 1.0 means a model could
    read the source off the image without looking at the retina."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    x = pd.concat([a[KEYS], b[KEYS]]).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median())
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    if len(np.unique(y)) < 2 or len(x) < 40:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(clf, x, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def contact_sheet(paths, out: Path, rng, cols=6):
    """One row per camera family, so the simulation can be eyeballed as well
    as measured. The first column is the untouched image."""
    kinds = ["desktop", "aged_desktop", "handheld", "phone_lens"]
    tiles = []
    for k in kinds:
        prof = sample_profile(rng, k)
        row = []
        for p in paths[:cols]:
            img = read_rgb(p)
            if img.shape[0] != 512:
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            row.append(cv2.resize(simulate_capture(img, prof, rng), (256, 256)))
        tiles.append(np.hstack(row))
    orig = []
    for p in paths[:cols]:
        img = read_rgb(p)
        if img.shape[0] != 512:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        orig.append(cv2.resize(img, (256, 256)))
    sheet = np.vstack([np.hstack(orig)] + tiles)
    for i, name in enumerate(["original"] + kinds):
        cv2.putText(sheet, name, (8, 24 + i * 256), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="images sampled per group")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    m = pd.read_csv(PROCESSED / "manifest.csv")
    m = m[m.ok == 1]

    def pick(sel):
        d = m[sel]
        return d.sample(min(args.n, len(d)), random_state=args.seed).cache.tolist()

    dev_paths = pick(m.role == "dev")
    groups = {
        "dev real (DDR + APTOS)": collect(dev_paths),
        "dev simulated": collect(dev_paths, fn=lambda i, p: simulate_capture(i, p, rng), rng=rng),
        "IDRiD (external)": collect(pick(m.source == "idrid")),
        "Messidor-2 (external)": collect(pick(m.source == "messidor2")),
    }

    report = {"n_per_group": {k: int(len(v)) for k, v in groups.items()}, "summary": {}, "coverage": {}}
    for key in KEYS:
        report["summary"][key] = {
            name: {"p5": round(float(np.nanpercentile(df[key], 5)), 4),
                   "median": round(float(np.nanmedian(df[key])), 4),
                   "p95": round(float(np.nanpercentile(df[key], 95)), 4)}
            for name, df in groups.items()}

    # Coverage saturates: the development data is already wide in each of
    # these statistics taken alone. The sharper question is joint, and it has
    # to be asked about the pool the grader will actually be trained on ---
    # real images and simulated ones together, since simulation is applied to
    # a fraction of each epoch, not to all of it. Comparing external data
    # against the simulated images alone answers the wrong question: those
    # are deliberately degraded, so of course they separate.
    #
    # An AUC near 0.5 means a classifier cannot tell an external image from
    # the training pool on capture signature alone. Lower is better, and the
    # number to beat is the one for the real pool on its own.
    pool = pd.concat([groups["dev real (DDR + APTOS)"], groups["dev simulated"]], ignore_index=True)
    report["separability_auc"] = {"note": "AUC of a 5-fold cross-validated classifier on capture "
                                          "statistics; 0.5 = indistinguishable, lower is better"}
    for ext in ("IDRiD (external)", "Messidor-2 (external)"):
        report["separability_auc"][ext] = {
            "training_pool_real_only": round(domain_auc(groups["dev real (DDR + APTOS)"], groups[ext]), 3),
            "training_pool_with_simulation": round(domain_auc(pool, groups[ext]), 3)}
    # a high value here is the point, not a fault: it says the simulator
    # actually changes the image rather than passing it through
    report["separability_auc"]["real_vs_simulated (should be high)"] = round(
        domain_auc(groups["dev real (DDR + APTOS)"], groups["dev simulated"]), 3)

    # and the marginal question: what share of each external set's images sit
    # inside the range our training data occupies, before and after simulation
    for ext in ("IDRiD (external)", "Messidor-2 (external)"):
        cov = {}
        for key in KEYS:
            t = groups[ext][key].to_numpy()
            cov[key] = {
                "covered_by_dev_real": round(coverage(groups["dev real (DDR + APTOS)"][key].to_numpy(), t), 3),
                "covered_by_dev_simulated": round(coverage(groups["dev simulated"][key].to_numpy(), t), 3)}
        mean_real = np.nanmean([v["covered_by_dev_real"] for v in cov.values()])
        mean_sim = np.nanmean([v["covered_by_dev_simulated"] for v in cov.values()])
        cov["mean_over_statistics"] = {"covered_by_dev_real": round(float(mean_real), 3),
                                       "covered_by_dev_simulated": round(float(mean_sim), 3)}
        report["coverage"][ext] = cov

    sheet = contact_sheet(dev_paths, RESULTS / "figures" / "capture_simulator.jpg", rng)
    report["contact_sheet"] = str(sheet.relative_to(RESULTS.parent))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "simulator_stats.json").write_text(json.dumps(report, indent=2))

    print(f"{'statistic':12s} " + "".join(f"{n[:18]:>20s}" for n in groups))
    for key in KEYS:
        print(f"{key:12s} " + "".join(f"{report['summary'][key][n]['median']:20.3f}" for n in groups))
    print()
    for ext, cov in report["coverage"].items():
        mm = cov["mean_over_statistics"]
        sep = report["separability_auc"][ext]
        print(f"{ext:24s} coverage {mm['covered_by_dev_real']:.2f} -> {mm['covered_by_dev_simulated']:.2f}"
              f"   separability AUC {sep['training_pool_real_only']:.3f} -> "
              f"{sep['training_pool_with_simulation']:.3f}  (0.5 = indistinguishable)")
    print(f"{'real vs simulated':24s} separability AUC "
          f"{report['separability_auc']['real_vs_simulated (should be high)']:.3f} (high is correct here)")
    print("wrote", RESULTS / "simulator_stats.json", "and", sheet)


if __name__ == "__main__":
    main()
