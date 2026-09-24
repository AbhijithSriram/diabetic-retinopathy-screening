"""A physically ordered model of a cheap handheld fundus camera.

Our development images come from desktop mydriatic cameras (DDR, APTOS).
Rural screening uses handheld or phone-mounted cameras through a small,
often undilated pupil. That gap is the largest single reason the grader
loses specificity on a camera it has never seen, and we cannot buy the
missing data. So we manufacture it: take a clean capture and push it back
through the imaging chain of a worse camera.

The stage order follows the physics, not convenience. Each stage acts on
the state the previous one leaves behind:

    retina reflectance
      -> illumination      ring flash through the pupil: uneven field
      -> ocular media      cataract: forward scatter (veiling glare), yellowing
      -> optics            defocus, chromatic aberration, motion, pupil vignette
      -> sensor            limited resolution, shot noise, read noise
      -> ISP               white balance, exposure, tone curve, sharpening, JPEG

Two things separate this from the usual bag of augmentation tricks.

Photometry happens in linear light. Stored fundus images are gamma encoded.
Blurring, scattering and adding photon noise in that space is wrong by a
power of about 2.2, and that is exactly the sort of error that makes
synthetic data look plausible and behave badly. Every stage here runs on
linearised intensity; the tone curve is re-applied at the end.

A camera is a persistent object, not a per-image random draw. A real
deployment has a handful of devices, each with its own dust, its own focus
error, its own white balance, reused across thousands of patients.
CameraProfile holds those fixed traits; only alignment, exposure and hand
motion are resampled per shot. A grader trained against a bank of such
devices has to learn disease rather than device.

The field of view is respected throughout: every convolution is a
normalised convolution over the retinal disc, blur(I*M) / blur(M), so the
black surround is never smeared into the retina and the rim does not pick
up a dark halo that no real camera produces.

    from drishti.augment import sample_profile, simulate_capture
    out = simulate_capture(crop, sample_profile(rng), rng)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

SRGB_GAMMA = 2.2


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_LIN_LUT = np.power(np.arange(256, dtype=np.float32) / 255.0, SRGB_GAMMA).astype(np.float32)


def _to_linear(img_u8: np.ndarray) -> np.ndarray:
    return _LIN_LUT[img_u8]


# Tone curve as a lookup, indexed by sqrt(intensity) rather than intensity.
# The square root spends the table where fundus images actually live --- the
# dark end --- so 4096 entries land within one grey level of the exact curve,
# at a third of the cost of a pow() over 786k floats.
_SRGB_N = 4096
_TONE_LUTS: dict[int, np.ndarray] = {}


def _tone_lut(extra_gamma: float = 1.0) -> np.ndarray:
    """Tone curve plus the device's own gamma deviation, in one table."""
    key = int(round(extra_gamma * 100))
    if key not in _TONE_LUTS:
        x = (np.arange(_SRGB_N + 1) / _SRGB_N) ** 2          # undo the sqrt index
        y = (x ** (1.0 / SRGB_GAMMA)) ** (key / 100.0)
        _TONE_LUTS[key] = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return _TONE_LUTS[key]


def _to_srgb_u8(lin: np.ndarray, extra_gamma: float = 1.0) -> np.ndarray:
    """Clip, apply the tone curve and quantise, in one pass over the array."""
    np.clip(lin, 0.0, 1.0, out=lin)
    np.sqrt(lin, out=lin)
    lin *= _SRGB_N
    return _tone_lut(extra_gamma)[lin.astype(np.uint16)]


def _wide_blur(lin: np.ndarray, mask: np.ndarray, sigma_frac: float, ds: int = 8) -> np.ndarray:
    """Normalised blur with a very wide kernel, done at reduced resolution.
    A sigma of a tenth of the image is what downsampling exists for; at full
    size OpenCV builds a 367-tap kernel and the stage costs 300 ms."""
    n = lin.shape[0]
    m = max(n // ds, 32)
    small = cv2.resize(lin * mask[..., None], (m, m), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(mask, (m, m), interpolation=cv2.INTER_AREA)
    sg = max(sigma_frac * m, 0.8)
    k = int(2 * round(3 * sg) + 1)
    num = cv2.GaussianBlur(small, (k, k), sg, borderType=cv2.BORDER_REPLICATE)
    den = cv2.GaussianBlur(msmall, (k, k), sg, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(num / _like(np.maximum(den, 1e-4), num), (n, n), interpolation=cv2.INTER_LINEAR)


def _fov_mask(img_u8: np.ndarray) -> np.ndarray:
    """The retinal disc. Cached crops are already square and centred, but a
    few have a flat top or bottom, so measure rather than assume."""
    m = (img_u8.max(axis=2) > 10).astype(np.float32)
    if m.mean() < 0.15:  # unexpectedly dark: fall back to the inscribed circle
        h = img_u8.shape[0]
        y, x = np.ogrid[:h, :h]
        c = (h - 1) / 2.0
        m = (((x - c) ** 2 + (y - c) ** 2) <= c * c).astype(np.float32)
    return m


def _like(m: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Broadcast a 2-D mask to match `ref`. OpenCV silently returns a 2-D
    array for a single-channel input, so the shape has to be read off the
    result rather than assumed --- getting this wrong broadcasts (h, w)
    against (h, w, 1) into an (h, w, h) array, which is both wrong and slow."""
    return m if ref.ndim == 2 else m[..., None]


def _masked_blur(lin: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that does not bleed the black surround into the retina."""
    if sigma <= 0.05:
        return lin
    k = int(2 * round(3 * sigma) + 1)
    num = cv2.GaussianBlur(lin * _like(mask, lin), (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
    den = cv2.GaussianBlur(mask, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
    return num / _like(np.maximum(den, 1e-4), num)


def _masked_filter(lin: np.ndarray, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    num = cv2.filter2D(lin * _like(mask, lin), -1, kernel, borderType=cv2.BORDER_REPLICATE)
    den = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return num / _like(np.maximum(den, 1e-4), num)


def _disc_kernel(radius: float) -> np.ndarray:
    """Defocus is a disc, not a Gaussian: an out-of-focus point takes the
    shape of the aperture. Using a Gaussian is the commonest way synthetic
    blur fails to resemble real blur."""
    r = max(radius, 0.5)
    k = int(2 * math.ceil(r) + 1)
    c = (k - 1) / 2.0
    y, x = np.ogrid[:k, :k]
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2)
    ker = np.clip(r + 0.5 - d, 0.0, 1.0).astype(np.float32)  # antialiased edge
    return ker / ker.sum()


def _motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    n = max(int(round(length)), 1)
    if n < 2:
        return np.ones((1, 1), np.float32)
    k = np.zeros((n, n), np.float32)
    cv2.line(k, (0, n // 2), (n - 1, n // 2), 1.0, 1)
    m = cv2.getRotationMatrix2D(((n - 1) / 2.0, (n - 1) / 2.0), angle_deg, 1.0)
    k = cv2.warpAffine(k, m, (n, n))
    s = float(k.sum())
    return k / s if s > 0 else np.ones((1, 1), np.float32)


def _radial_scale(chan: np.ndarray, factor: float) -> np.ndarray:
    """Lateral chromatic aberration: the lens magnifies each wavelength
    slightly differently, so colours separate towards the rim."""
    if abs(factor - 1.0) < 1e-5:
        return chan
    h, w = chan.shape
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), 0.0, factor)
    return cv2.warpAffine(chan, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# the camera
# --------------------------------------------------------------------------

@dataclass
class CameraProfile:
    """Traits of one physical device, fixed across every patient it images."""
    name: str = "generic"
    # optics
    defocus_px: float = 1.2          # disc PSF radius, quoted at 512 px across the field
    lca_gain: float = 1.0            # lateral chromatic aberration multiplier
    blue_blur_px: float = 0.8        # longitudinal CA: the blue focus sits ~2 D short
    red_blur_px: float = 0.3
    vignette: float = 0.35           # rim falloff, 0 = none
    # illumination
    illum_amp: float = 0.35          # strength of the ring-flash hot spot
    illum_sigma: float = 0.45        # its width, as a fraction of the field radius
    # sensor
    sensor_px: int = 400             # effective resolution across the field
    well_depth: float = 900.0        # photons at full well; lower is noisier
    read_noise: float = 0.006        # linear units
    dust: tuple = ()                 # (x, y, r, strength), field-relative
    # ISP
    wb: tuple = (1.0, 1.0, 1.0)      # white-balance gains, R G B
    tone_gamma: float = 1.0          # deviation from sRGB in the tone curve
    sharpen: float = 0.4             # unsharp-mask amount
    jpeg_q: int = 70
    # the ocular media this device is typically used through
    media_yellow: float = 0.0        # 0 clear, 1 dense brunescent cataract

    def __post_init__(self):
        self.dust = tuple(self.dust)
        self._fields: dict[int, np.ndarray] = {}


PROFILE_BANK: dict[str, dict] = {
    # Parameters are not guesses: they are set so that the statistics of the
    # simulated captures bracket the statistics of the real external sets.
    # scripts/validate_simulator.py is the measurement that fixed them, and
    # re-running it is what should happen if any number here is touched.

    # A good desktop mydriatic camera: what our training data already is.
    # Kept in the bank so "barely simulated" stays a member of the set.
    "desktop": dict(defocus_px=0.6, lca_gain=0.4, blue_blur_px=0.3, red_blur_px=0.1,
                    vignette=0.10, illum_amp=0.15, sensor_px=512, well_depth=6000,
                    read_noise=0.0015, sharpen=0.15, jpeg_q=92),
    # Handheld non-mydriatic: small pupil, hand-held, strong flash gradient.
    "handheld": dict(defocus_px=1.8, lca_gain=1.3, blue_blur_px=1.1, red_blur_px=0.4,
                     vignette=0.28, illum_amp=0.35, sensor_px=380, well_depth=1400,
                     read_noise=0.004, sharpen=0.40, jpeg_q=78),
    # A phone behind a 20 D lens: worst optics, heaviest ISP, hardest JPEG.
    "phone_lens": dict(defocus_px=2.0, lca_gain=1.8, blue_blur_px=1.4, red_blur_px=0.5,
                       vignette=0.35, illum_amp=0.42, sensor_px=340, well_depth=800,
                       read_noise=0.007, sharpen=0.60, jpeg_q=65),
    # An old desktop camera in a district hospital: clean optics, tired sensor.
    "aged_desktop": dict(defocus_px=1.0, lca_gain=0.7, blue_blur_px=0.6, red_blur_px=0.2,
                         vignette=0.20, illum_amp=0.25, sensor_px=430, well_depth=1200,
                         read_noise=0.005, sharpen=0.22, jpeg_q=82),
}


# How often each family is drawn. Weighted, not uniform: a quarter of the
# bank being the worst phone optics would put a tenth of every epoch beyond
# the point where the image's own grade is still visible, which is label
# noise dressed up as augmentation. Keyed by name so that reordering the
# bank cannot silently reassign the weights.
PROFILE_MIX: dict[str, float] = {
    "desktop": 0.30, "aged_desktop": 0.30, "handheld": 0.28, "phone_lens": 0.12,
}
assert set(PROFILE_MIX) == set(PROFILE_BANK) and abs(sum(PROFILE_MIX.values()) - 1) < 1e-9


def sample_profile(rng: np.random.Generator, kind: str | None = None) -> CameraProfile:
    """Draw one device. `kind` picks a family; its parameters are then
    jittered, because two cameras of the same model are not identical."""
    if kind is None:
        names = list(PROFILE_MIX)
        kind = str(rng.choice(names, p=[PROFILE_MIX[k] for k in names]))
    base = dict(PROFILE_BANK[kind])

    def j(v, f):
        return float(v * rng.uniform(1 - f, 1 + f))

    p = CameraProfile(
        name=kind,
        defocus_px=j(base["defocus_px"], 0.35),
        lca_gain=j(base["lca_gain"], 0.4),
        blue_blur_px=j(base["blue_blur_px"], 0.4),
        red_blur_px=j(base["red_blur_px"], 0.4),
        vignette=float(np.clip(j(base["vignette"], 0.3), 0.0, 0.8)),
        illum_amp=float(np.clip(j(base["illum_amp"], 0.4), 0.0, 0.8)),
        illum_sigma=float(rng.uniform(0.30, 0.65)),
        sensor_px=int(j(base["sensor_px"], 0.12)),
        well_depth=j(base["well_depth"], 0.4),
        read_noise=j(base["read_noise"], 0.5),
        # White balance is the widest knob in the bank. Real cameras differ
        # here more than anywhere else --- IDRiD and Messidor-2 sit well to the
        # red of our development data --- and it costs nothing to span it.
        wb=((float(rng.uniform(0.95, 1.12)), 1.0, float(rng.uniform(0.88, 1.10)))
            if kind == "desktop" else
            (float(rng.uniform(0.92, 1.34)), 1.0, float(rng.uniform(0.50, 1.14)))),
        tone_gamma=j(1.0, 0.10),
        sharpen=j(base["sharpen"], 0.4),
        jpeg_q=int(np.clip(j(base["jpeg_q"], 0.18), 25, 97)),
        media_yellow=float(rng.uniform(0.0, 0.15) if kind == "desktop" else rng.uniform(0.0, 0.65)),
    )
    # dust and smears on the objective: few, small, fixed for the life of the device
    n = int(rng.integers(0, 3 if kind == "desktop" else 7))
    # Softer and wider than a first guess: sharp specks survive Graham
    # normalisation as crisp dots, which is a fixed pattern a network can key
    # on to recognise "this image was simulated".
    p.dust = tuple((float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)),
                    float(rng.uniform(0.02, 0.06)), float(rng.uniform(0.08, 0.30)))
                   for _ in range(n))
    return p


@dataclass
class ShotParams:
    """What changes between two patients on the same device."""
    pupil_offset: tuple = (0.0, 0.0)   # misalignment, in field radii
    pupil_radius: float = 1.3          # below 1 clips the field: a small-pupil crescent
    illum_offset: tuple = (0.0, 0.0)
    haze: float = 0.0                  # forward scatter: cataract, or a dirty lens
    motion_px: float = 0.0
    motion_angle: float = 0.0
    exposure: float = 1.0      # photons collected: sets how noisy the frame is
    ae_error: float = 1.0      # how far the metering lands from the right level


def sample_shot(rng: np.random.Generator, profile: CameraProfile) -> ShotParams:
    hard = profile.name in ("handheld", "phone_lens")
    return ShotParams(
        pupil_offset=(float(rng.normal(0, 0.10 if hard else 0.04)),
                      float(rng.normal(0, 0.10 if hard else 0.04))),
        pupil_radius=float(rng.uniform(0.80, 1.35) if hard else rng.uniform(1.05, 1.40)),
        illum_offset=(float(rng.normal(0, 0.25)), float(rng.normal(0, 0.25))),
        haze=float(np.clip(rng.gamma(1.6, 0.14 if hard else 0.06), 0, 0.75)),
        motion_px=float(abs(rng.normal(0, 1.6 if hard else 0.4))),
        motion_angle=float(rng.uniform(0, 180)),
        exposure=float(np.exp(rng.normal(0, 0.45 if hard else 0.20))),
        ae_error=float(np.exp(rng.normal(0, 0.18 if hard else 0.08))),
    )


_GRID: dict[int, tuple] = {}


def _grid(n: int) -> tuple:
    """Field-relative coordinates: (nx, ny, r), each -1..1 across the disc."""
    if n not in _GRID:
        c = (n - 1) / 2.0
        y, x = np.ogrid[:n, :n]
        nx = ((x - c) / c).astype(np.float32)
        ny = ((y - c) / c).astype(np.float32)
        _GRID[n] = (nx, ny, np.sqrt(nx ** 2 + ny ** 2).astype(np.float32))
    return _GRID[n]


def _device_field(profile: CameraProfile, n: int) -> np.ndarray:
    """Vignetting and objective dust: fixed for the life of the device, so
    built once and reused across every patient it images."""
    cache = profile._fields
    if n not in cache:
        nx, ny, r = _grid(n)
        f = np.clip(1.0 - profile.vignette * r ** 2, 0.05, 1.0).astype(np.float32)
        for dx, dy, dr, amp in profile.dust:
            dd = (nx - dx) ** 2 + (ny - dy) ** 2
            f *= (1.0 - amp * np.exp(-dd / (dr * dr))).astype(np.float32)
        if len(cache) > 4:
            cache.clear()
        cache[n] = f
    return cache[n]


def _coarse_field(n: int, build, ds: int = 4) -> np.ndarray:
    """Evaluate a smooth per-shot field at reduced resolution and upsample.
    Illumination and the pupil aperture vary over tens of pixels; there is
    nothing in them worth computing at full size."""
    m = max(n // ds, 48)
    nx, ny, r = _grid(m)
    return cv2.resize(build(nx, ny, r), (n, n), interpolation=cv2.INTER_LINEAR)


def simulate_capture(crop_u8: np.ndarray, profile: CameraProfile,
                     rng: np.random.Generator, shot: ShotParams | None = None) -> np.ndarray:
    """Re-image a clean fundus crop through `profile`. Returns uint8 RGB of
    the same size, with the field of view preserved."""
    n = crop_u8.shape[0]
    s = n / 512.0                       # profile lengths are quoted at 512 px
    shot = shot or sample_shot(rng, profile)
    mask = _fov_mask(crop_u8)
    lin = _to_linear(crop_u8)
    # What the auto-exposure will aim for: the scene's own highlight level.
    # A camera that meters correctly reproduces the brightness of what it is
    # pointed at, so the simulation must change the character of the capture,
    # not its overall level --- otherwise every simulated image is uniformly
    # brighter than every real one and the grader learns that instead.
    ref = lin[::5, ::5][mask[::5, ::5] > 0.5]
    target = float(np.percentile(ref, 99.0)) if ref.size else 0.9

    # 1. illumination (ring flash through the pupil, off centre) and the
    #    yellowing of the ocular media. Both are pre-optics multiplications,
    #    so they fold into a single pass: one is a field, one a colour.
    icx, icy = shot.illum_offset
    two_s2 = 2 * profile.illum_sigma ** 2
    hot = _coarse_field(n, lambda nx, ny, r: np.exp(-((nx - icx) ** 2 + (ny - icy) ** 2) / two_s2))
    hot = (1.0 + profile.illum_amp * hot) / (1.0 + profile.illum_amp)
    t = profile.media_yellow            # Lambert-Beer: blue is absorbed most
    lin *= hot[..., None] * np.array([1.0, 1.0 - 0.10 * t, 1.0 - 0.32 * t], np.float32)

    # 2. forward scatter through a clouded lens. It veils contrast, which is
    #    what makes a cataract photo look washed out rather than dark.
    if shot.haze > 0.01:
        veil = _wide_blur(lin, mask, 0.12)
        cv2.addWeighted(lin, 1 - shot.haze, veil, shot.haze, 0.0, dst=lin)

    # 3. optics
    if shot.motion_px * s > 1.2:
        lin = _masked_filter(lin, mask, _motion_kernel(shot.motion_px * s, shot.motion_angle))
    if profile.defocus_px * s > 0.6:
        lin = _masked_filter(lin, mask, _disc_kernel(profile.defocus_px * s))
    if profile.blue_blur_px * s > 0.2:  # longitudinal chromatic aberration
        lin[..., 2] = _masked_blur(lin[..., 2], mask, profile.blue_blur_px * s)
    if profile.red_blur_px * s > 0.2:
        lin[..., 0] = _masked_blur(lin[..., 0], mask, profile.red_blur_px * s)
    if profile.lca_gain > 0.05:         # lateral chromatic aberration
        lin[..., 0] = _radial_scale(lin[..., 0], 1.0 + 0.0016 * profile.lca_gain)
        lin[..., 2] = _radial_scale(lin[..., 2], 1.0 - 0.0016 * profile.lca_gain)

    # vignetting and objective dust (fixed for this device), then the pupil.
    # An undilated pupil clips the field off centre: the crescent shadow that
    # every handheld operator recognises.
    px, py = shot.pupil_offset
    pupil = _coarse_field(n, lambda nx, ny, r: np.clip(
        (shot.pupil_radius - np.sqrt((nx - px) ** 2 + (ny - py) ** 2)) / 0.18, 0.0, 1.0))
    lin *= (_device_field(profile, n) * pupil)[..., None]

    # 4. sensor: limited resolution (optics plus demosaic), then photon statistics
    sp = max(int(profile.sensor_px * s), 64)
    if sp < n:
        small = cv2.resize(lin, (sp, sp), interpolation=cv2.INTER_AREA)
        lin = cv2.resize(small, (n, n), interpolation=cv2.INTER_LINEAR)
    # Exposure and white balance are both per-channel constants here, so they
    # ride along on one multiply. Exposure is set before the sensor because it
    # is what decides the photon count: an underexposed frame that the ISP
    # later brightens is a noisy frame, which is the whole point of modelling
    # it rather than just scaling the output.
    lin *= shot.exposure * np.asarray(profile.wb, np.float32)
    # Shot noise and read noise are independent, so they add in variance and
    # one normal draw carries both. The Gaussian approximation to the photon
    # count is exact to well under a grey level at these well depths, and
    # about eight times faster than np.random.poisson. Intensity is
    # non-negative here: nothing before this stage can make it otherwise.
    wd = max(profile.well_depth, 20.0)
    sd = lin * (1.0 / wd)
    sd += profile.read_noise ** 2
    np.sqrt(sd, out=sd)
    sd *= rng.standard_normal(lin.shape, dtype=np.float32)
    lin += sd
    np.clip(lin, 0.0, None, out=lin)

    # 5. ISP: auto-exposure measured on a subsample of the disc, then the
    #    tone curve, the sharpening halo that gives a cheap ISP away, and JPEG
    sub = lin[::5, ::5][mask[::5, ::5] > 0.5]
    hi = float(np.percentile(sub, 99.0)) if sub.size else 1.0
    if hi > 1e-4:
        lin *= min(target * shot.ae_error / hi, 40.0)
    u8 = _to_srgb_u8(lin, profile.tone_gamma)
    if profile.sharpen > 0.05:
        blur = cv2.GaussianBlur(u8, (0, 0), 1.2 * s)
        u8 = cv2.addWeighted(u8, 1 + profile.sharpen, blur, -profile.sharpen, 0)
    if profile.jpeg_q < 96:
        ok, buf = cv2.imencode(".jpg", u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(profile.jpeg_q)])
        if ok:
            u8 = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]

    u8 *= mask.astype(np.uint8)[..., None]
    return u8
