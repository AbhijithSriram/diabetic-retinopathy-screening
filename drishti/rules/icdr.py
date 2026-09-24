"""ICDR rule engine over detected lesions: explanation, DME risk, and a
triage signal. It is deliberately not a second grader with a veto.

What single-field fundus photos plus our lesion detector can assess:
- grade 0 vs 1 vs >=2: no lesions / microaneurysms only / any haemorrhage or
  exudate (ICDR: "more than just microaneurysms" is moderate NPDR);
- one of the three 4-2-1 criteria for severe NPDR: >20 intraretinal
  haemorrhages in each of four quadrants. Venous beading and IRMA are not
  detected, so severe NPDR can be supported but never ruled out;
- macular oedema risk, IDRiD definition: hard exudates within one disc
  diameter (DD) of the fovea centre = risk 2, elsewhere = risk 1.
What it cannot assess: neovascularisation (no reliable public pixel labels),
so proliferative DR is left to the CNN and the reviewer.

All coordinates are in the LESION_SIZE frame, where the field of view is a
circle of radius LESION_SIZE/2 centred in the image.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..config import LESION_SIZE

NAMES = {"MA": "microaneurysm", "HE": "haemorrhage", "EX": "hard exudate", "SE": "soft exudate (cotton-wool spot)"}
DD_FALLBACK_FRACTION = 1 / 7.5  # disc diameter / FOV diameter for a 45-degree field


def disc_diameter(od) -> tuple[float, str]:
    fov_d = float(LESION_SIZE)
    if od is not None:
        d = 2 * od[2]
        if fov_d / 12 <= d <= fov_d / 4:
            return d, "measured"
    return fov_d * DD_FALLBACK_FRACTION, "assumed (disc not segmented)"


def quadrant(x, y, fx, fy, nasal_sign: int) -> str:
    vert = "superior" if y < fy else "inferior"
    horiz = "nasal" if np.sign(x - fx) == nasal_sign else "temporal"
    return f"{vert}-{horiz}"


def assess(lesions: list, od=None, fovea=None, fovea_conf: float = 1.0) -> dict:
    c = LESION_SIZE / 2.0
    fx, fy = (fovea[0], fovea[1]) if fovea is not None and fovea_conf >= 0.2 else (c, c)
    fovea_src = "detected" if fovea is not None and fovea_conf >= 0.2 else "assumed at image centre"
    nasal_sign = int(np.sign(od[0] - fx)) if od is not None else 1
    dd, dd_src = disc_diameter(od)

    counts = Counter(L.kind for L in lesions)
    quads = {q: Counter() for q in ("superior-nasal", "superior-temporal", "inferior-nasal", "inferior-temporal")}
    for L in lesions:
        quads[quadrant(L.x, L.y, fx, fy, nasal_sign)][L.kind] += 1

    trace = []
    if not lesions:
        evidence = 0
        trace.append("No lesions detected.")
    elif set(counts) == {"MA"}:
        evidence = 1
        trace.append(f"Only microaneurysms ({counts['MA']}) -> consistent with mild NPDR (ICDR grade 1).")
    else:
        evidence = 2
        found = ", ".join(f"{counts[k]} {NAMES[k]}{'s' if counts[k] != 1 else ''}" for k in ("HE", "EX", "SE") if counts[k])
        trace.append(f"More than microaneurysms alone ({found}) -> at least moderate NPDR (ICDR grade 2), referable.")
    he_all_quads = all(quads[q]["HE"] > 20 for q in quads)
    if he_all_quads:
        evidence = 3
        trace.append(">20 haemorrhages in all four quadrants -> meets the haemorrhage criterion of the 4-2-1 rule (severe NPDR).")
    trace.append("Not assessed from this photo: venous beading, IRMA, neovascularisation.")

    ex = [L for L in lesions if L.kind == "EX"]
    if ex:
        dist = min(float(np.hypot(L.x - fx, L.y - fy)) for L in ex) / dd
        dme_risk = 2 if dist <= 1.0 else 1
        trace.append(f"Nearest hard exudate is {dist:.1f} disc diameters from the fovea"
                     + (" -> within 1 DD: possible macular oedema." if dme_risk == 2 else "."))
    else:
        dist, dme_risk = None, 0

    return {
        "counts": {k: int(counts.get(k, 0)) for k in NAMES},
        "quadrants": {q: dict(v) for q, v in quads.items()},
        "evidence_grade": evidence,
        "dme_risk": dme_risk,
        "nearest_exudate_dd": None if dist is None else round(dist, 2),
        "lesion_flag": evidence >= 2,
        "dme_flag": dme_risk == 2,
        "fovea": {"x": round(float(fx), 1), "y": round(float(fy), 1), "source": fovea_src},
        "disc_diameter_px": round(dd, 1), "disc_diameter_source": dd_src,
        "trace": trace,
    }
