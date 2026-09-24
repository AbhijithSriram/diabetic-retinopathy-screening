"""Unit tests for the cheap-camera simulator. No dataset needed.

These exist because the first version of this module had a bug that no
visual check would have caught: OpenCV returns a 2-D array for a
single-channel input, so dividing by a (h, w, 1) mask broadcast an (h, w)
numerator into an (h, w, h) array --- silently wrong and 100x slower. The
shape assertions below are the guard against that class of mistake.
"""
import numpy as np
import pytest

from drishti.augment import PROFILE_BANK, sample_profile, simulate_capture
from drishti.augment.capture import PROFILE_MIX, ShotParams, _disc_kernel, _masked_blur
from drishti.augment.stats import KEYS, capture_stats


def synthetic_crop(n=256, texture=True):
    """A disc of retina-coloured texture on black, like a cached crop."""
    rng = np.random.default_rng(0)
    img = np.zeros((n, n, 3), np.uint8)
    base = rng.normal(0, 12, (n, n, 1)) if texture else np.zeros((n, n, 1))
    img[:] = np.clip(base + np.array([150, 75, 40]), 0, 255).astype(np.uint8)
    c = (n - 1) / 2
    yy, xx = np.ogrid[:n, :n]
    img[((xx - c) ** 2 + (yy - c) ** 2) > c * c] = 0
    return img


@pytest.mark.parametrize("shape", [(32, 32), (32, 32, 1), (32, 32, 3)])
def test_masked_blur_preserves_shape(shape):
    """The (h, w, 1) case is the one that used to broadcast into (h, w, h)."""
    lin = np.random.rand(*shape).astype(np.float32)
    mask = np.ones((32, 32), np.float32)
    out = _masked_blur(lin, mask, 1.1)
    assert out.shape[:2] == (32, 32)
    assert out.ndim <= 3 and out.size <= lin.size


def test_masked_blur_does_not_bleed_the_surround_inward():
    """Normalised convolution: a blur next to the black rim must not darken
    the retina, which plain blurring would."""
    lin = np.ones((64, 64), np.float32)
    mask = np.zeros((64, 64), np.float32)
    mask[:, 20:] = 1.0                      # everything left of x=20 is outside
    out = _masked_blur(lin, mask, 2.0)
    assert out[32, 25] == pytest.approx(1.0, abs=1e-3)


def test_disc_kernel_is_normalised_and_round():
    k = _disc_kernel(3.0)
    assert k.sum() == pytest.approx(1.0, abs=1e-6)
    assert k[0, 0] < k[len(k) // 2, len(k) // 2]     # corners outside the disc


def test_profile_mix_matches_the_bank():
    assert set(PROFILE_MIX) == set(PROFILE_BANK)
    assert sum(PROFILE_MIX.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("kind", sorted(PROFILE_BANK))
def test_simulate_runs_for_every_family(kind):
    img = synthetic_crop()
    prof = sample_profile(np.random.default_rng(1), kind)
    out = simulate_capture(img, prof, np.random.default_rng(2))
    assert out.shape == img.shape and out.dtype == np.uint8


def test_field_of_view_is_preserved():
    """Nothing may light up outside the original disc: a blur that leaked
    across the rim would hand the grader a give-away halo."""
    img = synthetic_crop()
    prof = sample_profile(np.random.default_rng(3), "phone_lens")
    out = simulate_capture(img, prof, np.random.default_rng(4))
    outside = (img.max(axis=2) <= 10) & (out.max(axis=2) > 0)
    assert outside.sum() == 0


def test_same_device_and_shot_give_the_same_image():
    img = synthetic_crop()
    prof = sample_profile(np.random.default_rng(5), "handheld")
    shot = ShotParams(pupil_radius=1.2, haze=0.2, motion_px=2.0)
    a = simulate_capture(img, prof, np.random.default_rng(6), shot)
    b = simulate_capture(img, prof, np.random.default_rng(6), shot)
    assert np.array_equal(a, b)


def test_brightness_is_preserved_not_inflated():
    """Auto-exposure aims at the scene's own highlight level, so a simulated
    capture must not be systematically brighter than what it came from."""
    img = synthetic_crop()
    disc = img.max(axis=2) > 10
    levels = []
    for i in range(6):
        prof = sample_profile(np.random.default_rng(100 + i))
        out = simulate_capture(img, prof, np.random.default_rng(200 + i))
        levels.append(out[..., 1][disc].mean())
    ratio = np.mean(levels) / img[..., 1][disc].mean()
    assert 0.6 < ratio < 1.4, f"mean level moved by {ratio:.2f}x"


def test_the_simulator_actually_degrades():
    """The opposite failure: a pipeline that quietly passes the image
    through would look fine in every test above."""
    img = synthetic_crop()
    prof = sample_profile(np.random.default_rng(7), "phone_lens")
    out = simulate_capture(img, prof, np.random.default_rng(8))
    assert capture_stats(out)["sharpness"] < capture_stats(img)["sharpness"]


def test_capture_stats_are_finite_on_a_plain_crop():
    s = capture_stats(synthetic_crop())
    assert set(s) == set(KEYS)
    assert all(np.isfinite(v) for v in s.values())
