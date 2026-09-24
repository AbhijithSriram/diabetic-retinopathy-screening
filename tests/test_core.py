"""Unit tests that need no dataset: preprocessing geometry, conformal
guarantees on synthetic data, rule engine, metrics, CORN head."""
import numpy as np
import pytest
import torch

from drishti.conformal import triage as T
from drishti.eval.metrics import wilson, binary_report, qwk, expected_grade
from drishti.grading.model import class_probs, corn_loss, referable_prob, cumulative
from drishti.preprocess import prepare, fov_mask


def synthetic_fundus(h=900, w=1200, r=400, clip=0):
    img = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    circle = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 <= r * r
    img[circle] = (160, 80, 40)
    if clip:
        img[:clip] = 0
        img[h - clip:] = 0
    return img


def test_fov_crop_is_centred_and_square():
    img = synthetic_fundus()
    p = prepare(img, 256)
    assert p.image.shape == (256, 256, 3)
    assert abs(p.geometry.side - 800) <= 4
    # round trip of a coordinate
    x, y = p.geometry.to_processed(600, 450)
    ox, oy = p.geometry.to_original(x, y)
    assert abs(ox - 600) < 1e-6 and abs(oy - 450) < 1e-6
    assert abs(x - 128) < 2 and abs(y - 128) < 2


def test_fov_handles_clipped_circle():
    img = synthetic_fundus(clip=120)  # top and bottom of the circle cut off
    p = prepare(img, 256)
    assert abs(p.geometry.side - 800) <= 4  # diameter still from the width
    assert fov_mask(img).mean() > 0.3


def test_corn_probabilities_are_valid():
    lg = torch.randn(64, 4) * 3
    p = class_probs(lg)
    assert torch.allclose(p.sum(1), torch.ones(64), atol=1e-5)
    assert (p >= 0).all()
    c = cumulative(lg)
    assert (c[:, 1:] <= c[:, :-1] + 1e-7).all()
    assert torch.allclose(referable_prob(lg), p[:, 2:].sum(1), atol=1e-5)
    y = torch.randint(0, 5, (64,))
    assert corn_loss(lg, y).item() > 0


def _synthetic_probs(n, rng, noise=1.0):
    y = rng.integers(0, 5, n)
    logits = -((np.arange(5)[None] - (y[:, None] + rng.normal(0, noise, (n, 1)))) ** 2)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True), y


def test_ordinal_sets_cover_and_are_contiguous():
    rng = np.random.default_rng(0)
    cal_p, cal_y = _synthetic_probs(2000, rng)
    q = T.calibrate_sets(cal_p, cal_y, alpha=0.1)
    te_p, te_y = _synthetic_probs(5000, rng)
    cover = np.mean([lo <= y <= hi for (lo, hi), y in ((T.prediction_set(p, q), y) for p, y in zip(te_p, te_y))])
    assert cover >= 0.88  # 1 - alpha minus sampling slack


def test_crc_controls_missed_referable_rate():
    rng = np.random.default_rng(1)
    fnrs = []
    for rep in range(200):
        y = rng.random(600) < 0.3
        s = np.clip(rng.normal(np.where(y, 0.7, 0.3), 0.2), 0, 1)
        lam = T.crc_lambda_low(s[:300], y[:300], alpha_miss=0.05)
        te_s, te_y = s[300:], y[300:]
        fnrs.append(np.mean(te_s[te_y] < lam))
    assert np.mean(fnrs) <= 0.055  # guarantee is on the expectation


def test_crc_refuses_when_too_few_referable():
    assert T.crc_lambda_low(np.array([0.9, 0.8]), np.array([1, 1]), 0.05) == 0.0


def test_triage_outcomes():
    pol = T.TriagePolicy(lambda_low=0.2, lambda_high=0.8, lambda_op=0.3, qhat=0.9, alpha_miss=0.03, beta_false=0.02, alpha_set=0.1)
    p_healthy = np.array([0.9, 0.08, 0.01, 0.005, 0.005])
    assert T.triage(0.05, p_healthy, pol)["outcome"] == "recall"
    assert T.triage(0.05, p_healthy, pol, lesion_flag=True)["outcome"] == "review"
    assert T.triage(0.5, p_healthy, pol)["outcome"] == "review"
    assert T.triage(0.95, np.array([0, 0.02, 0.3, 0.4, 0.28]), pol)["outcome"] == "refer"


def test_wilson_and_report():
    p, lo, hi = wilson(61, 64)
    assert 0.86 < lo < 0.88 and 0.98 < hi < 0.99
    r = binary_report([1, 1, 0, 0], [1, 0, 0, 1])
    assert r["tp"] == 1 and r["fn"] == 1 and r["tn"] == 1 and r["fp"] == 1


def test_expected_grade_and_qwk():
    probs = np.eye(5)
    assert (expected_grade(probs) == np.arange(5)).all()
    assert qwk([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == pytest.approx(1.0)


def test_rule_engine():
    from drishti.lesions.infer import Lesion
    from drishti.rules.icdr import assess
    c = 720.0
    assert assess([])["evidence_grade"] == 0
    ma = [Lesion("MA", c + 50, c + 10, 4, 0.9)]
    assert assess(ma)["evidence_grade"] == 1
    ex_near = ma + [Lesion("EX", c + 20, c, 30, 0.9)]
    r = assess(ex_near, od=(c + 500, c, 90), fovea=(c, c), fovea_conf=0.9)
    assert r["evidence_grade"] == 2 and r["dme_flag"] and r["lesion_flag"]
    ex_far = ma + [Lesion("EX", c - 600, c - 100, 30, 0.9)]
    r2 = assess(ex_far, od=(c + 500, c, 90), fovea=(c, c), fovea_conf=0.9)
    assert r2["dme_risk"] == 1 and not r2["dme_flag"]
