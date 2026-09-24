"""Uncertainty and triage.

Two tools, each with a finite-sample guarantee under exchangeability between
calibration and deployment data (the assumption that breaks when the camera
or population changes, which is why `recalibrate_domain` exists):

1. Ordinal conformal prediction sets: a contiguous range of ICDR grades
   ({1,2}, never {0,3}) that contains the true grade with probability
   >= 1 - alpha (Lu et al., MICCAI 2022, "ordinal APS"-style construction).
2. Conformal risk control (Angelopoulos et al., ICLR 2024) on the quantity
   that matters clinically: the fraction of referable patients the system
   would send home. `lambda_low` is chosen so that
   E[ P(p_ref < lambda_low | referable) ] <= alpha_miss.

Triage outcomes: recall (routine re-screen), review (30-second tele-review
by an ophthalmologist), refer (direct referral), recapture (image problem).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

K = 5


# ------------------------------------------------------------ ordinal sets
def _nested_intervals(p: np.ndarray) -> list[tuple[int, int, float]]:
    """Grow an interval from the mode, adding whichever neighbour has more
    mass. Returns the nested sequence of (lo, hi, mass)."""
    lo = hi = int(np.argmax(p))
    mass = float(p[lo])
    seq = [(lo, hi, mass)]
    while lo > 0 or hi < len(p) - 1:
        left = p[lo - 1] if lo > 0 else -1.0
        right = p[hi + 1] if hi < len(p) - 1 else -1.0
        if right > left:
            hi += 1
            mass += float(p[hi])
        else:
            lo -= 1
            mass += float(p[lo])
        seq.append((lo, hi, mass))
    return seq


def ordinal_score(p: np.ndarray, y: int) -> float:
    """Mass of the first nested interval that contains y."""
    for lo, hi, mass in _nested_intervals(p):
        if lo <= y <= hi:
            return mass
    return 1.0


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return 1.0 + 1e-9
    return float(np.sort(scores)[k - 1])


def calibrate_sets(probs: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> float:
    scores = np.array([ordinal_score(p, int(t)) for p, t in zip(probs, y)])
    return conformal_quantile(scores, alpha)


def prediction_set(p: np.ndarray, qhat: float) -> tuple[int, int]:
    """Largest nested interval with mass <= qhat (always at least the mode)."""
    seq = _nested_intervals(p)
    best = seq[0]
    for lo, hi, mass in seq:
        if mass <= qhat + 1e-12:
            best = (lo, hi, mass)
        else:
            break
    return best[0], best[1]


# ------------------------------------------------- conformal risk control
def crc_lambda_low(p_ref: np.ndarray, referable: np.ndarray, alpha_miss: float) -> float:
    """Largest threshold such that, with the CRC finite-sample correction,
    the expected fraction of referable cases below it is <= alpha_miss.

    With n referable calibration cases, CRC allows at most
    k = floor(alpha*(n+1) - 1) of them below the threshold. If k < 0
    (fewer than 1/alpha - 1 referable cases) nothing may be auto-recalled.
    """
    s = np.sort(np.asarray(p_ref)[np.asarray(referable).astype(bool)])
    n = len(s)
    k = math.floor(alpha_miss * (n + 1) - 1)
    if n == 0 or k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(s[k])  # strictly-below count is at most k


def crc_lambda_high(p_ref: np.ndarray, referable: np.ndarray, beta_false: float) -> float:
    """Smallest threshold such that the expected fraction of non-referable
    cases at or above it is <= beta_false (direct referrals of healthy eyes)."""
    s = np.sort(np.asarray(p_ref)[~np.asarray(referable).astype(bool)])[::-1]
    m = len(s)
    k = math.floor(beta_false * (m + 1) - 1)
    if m == 0 or k < 0:
        return 1.0 + 1e-9
    if k >= m:
        return 0.0
    return float(np.nextafter(s[k], 1.0))  # above the (k+1)-th largest


# ----------------------------------------------------------------- policy
@dataclass
class TriagePolicy:
    lambda_low: float  # below: auto-recall (if no lesion red flag)
    lambda_high: float  # at/above: direct referral
    lambda_op: float  # single operating point used for the headline sens/spec
    qhat: float  # ordinal-set threshold
    alpha_miss: float
    beta_false: float
    alpha_set: float
    calibrated_on: str = ""
    n_cal: int = 0

    def as_dict(self):
        return asdict(self)


def fit_policy(p_ref, grades, probs, alpha_miss=0.03, beta_false=0.02, alpha_set=0.10, sens_target=0.95, calibrated_on="") -> TriagePolicy:
    ref = np.asarray(grades) >= 2
    return TriagePolicy(
        lambda_low=crc_lambda_low(p_ref, ref, alpha_miss),
        lambda_high=crc_lambda_high(p_ref, ref, beta_false),
        lambda_op=crc_lambda_low(p_ref, ref, 1 - sens_target),
        qhat=calibrate_sets(probs, grades, alpha_set),
        alpha_miss=alpha_miss, beta_false=beta_false, alpha_set=alpha_set,
        calibrated_on=calibrated_on, n_cal=int(len(p_ref)),
    )


def triage(p_ref: float, probs: np.ndarray, policy: TriagePolicy, lesion_flag: bool = False, dme_flag: bool = False) -> dict:
    lo, hi = prediction_set(probs, policy.qhat)
    urgent = hi >= 3
    if p_ref >= policy.lambda_high:
        outcome = "refer"
    elif p_ref < policy.lambda_low and not (lesion_flag or dme_flag):
        outcome = "recall"
    else:
        outcome = "review"
    reasons = []
    if outcome == "review":
        if p_ref >= policy.lambda_low:
            reasons.append("model uncertain about referable disease")
        if lesion_flag:
            reasons.append("lesions found that indicate grade 2 or worse")
        if dme_flag:
            reasons.append("hard exudates close to the fovea (possible macular oedema)")
    return {"outcome": outcome, "set": [lo, hi], "urgent": bool(urgent), "reasons": reasons}


def recalibrate_domain(p_ref: np.ndarray, grades: np.ndarray, alpha_miss: float) -> float:
    """Domain-specific (Mondrian) lambda_low from audit labels collected on
    one camera / site. Audit labels must include a random sample of
    auto-recalled cases, not only reviewed ones, or the estimate is biased."""
    return crc_lambda_low(p_ref, np.asarray(grades) >= 2, alpha_miss)
