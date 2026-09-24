"""Metrics with uncertainty. Every proportion gets a Wilson 95% interval;
AUC gets a percentile bootstrap interval (resampled by exam when exam ids
exist, since two eyes of one patient are not independent)."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

Z = 1.959963984540054


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fmt_ci(p, lo, hi, pct=True) -> str:
    if p != p:
        return "n/a"
    if pct:
        return f"{100 * p:.1f}% ({100 * lo:.1f}-{100 * hi:.1f})"
    return f"{p:.3f} ({lo:.3f}-{hi:.3f})"


def binary_report(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    sens = wilson(tp, tp + fn)
    spec = wilson(tn, tn + fp)
    npv = wilson(tn, tn + fn)
    ppv = wilson(tp, tp + fp)
    return {"n": len(y_true), "positives": tp + fn, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": sens, "specificity": spec, "npv": npv, "ppv": ppv}


def auc_ci(y_true, score, groups=None, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, float)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")
    auc = roc_auc_score(y_true, score)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inv == g)[0] for g in range(len(uniq))]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        boots.append(roc_auc_score(yt, score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def qwk(y_true, y_pred) -> float:
    return float(cohen_kappa_score(np.asarray(y_true), np.asarray(y_pred), weights="quadratic"))


def ece(y_true, prob, bins: int = 10) -> float:
    """Expected calibration error of a binary probability."""
    y_true = np.asarray(y_true).astype(float)
    prob = np.asarray(prob, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if sel.any():
            total += sel.mean() * abs(prob[sel].mean() - y_true[sel].mean())
    return float(total)


def expected_grade(probs: np.ndarray) -> np.ndarray:
    """Point prediction for QWK: the median of the ordinal distribution."""
    cdf = np.cumsum(probs, axis=1)
    return (cdf < 0.5).sum(axis=1).clip(0, probs.shape[1] - 1)
