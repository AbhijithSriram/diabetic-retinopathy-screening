"""Screening through MATLAB: the web app's backend when the PyTorch
checkpoints are not on this machine but the MATLAB export is.

MATLAB (matlab/drishti_screen.m, via the MATLAB Engine for Python) does
everything numerical: the quality rules, the grader ensemble and its Grad-CAM,
lesion maps, landmarks and the ICDR rule engine, all from the ONNX files in
runs/onnx_matlab. Python then applies the same triage policy and patient
messages as drishti/pipeline.py, so the two backends return the same result
shape and differ only in who ran the networks. The learned gradability
model is in the export as quality.onnx (scripts/export_matlab.py
--only-quality adds it to an older export); without it MATLAB falls back to
the rules, as the Python gate does when that model is missing.

The engine takes ~10 s to start and the first photograph ~90 s (ONNX import);
both happen once, in a background thread started by warm_up().
"""
from __future__ import annotations

import base64
import json
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import ICDR_NAMES, REPO, RUNS
from .conformal import triage as T
from .quality import gate
from .report.messages import worker_message

ONNX_DIR = RUNS / "onnx_matlab"
MATLAB_DIR = REPO / "matlab"
TRIAGE_CFG = REPO / "configs" / "triage.json"


def available() -> bool:
    if not (ONNX_DIR / "manifest.json").exists():
        return False
    try:
        import matlab.engine  # noqa: F401
    except ImportError:
        return False
    return True


def _b64(rgb: np.ndarray, size: int = 512) -> str:
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA) if rgb.shape[0] != size else rgb
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _py(v):
    """MATLAB Engine values -> plain Python (numbers, lists, dicts)."""
    if isinstance(v, dict):
        return {k: _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if type(v).__module__.startswith("matlab"):
        a = np.asarray(v)
        return a.item() if a.size == 1 else a.tolist()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


class MatlabScreener:
    backend = "matlab"

    def __init__(self):
        self.man = json.loads((ONNX_DIR / "manifest.json").read_text())
        self.policy = T.TriagePolicy(**self.man["policy"])
        self.site_lambdas: dict[str, float] = {}
        self.missing = [] if "quality" in self.man else ["gradability"]
        self._eng = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._started = False
        self._start_once = threading.Lock()
        self.status = "starting MATLAB"

    # ------------------------------------------------------------ engine
    def warm_up(self):
        """Start MATLAB and import every network on a sample, off the request path.
        Idempotent: only the first call starts anything."""
        with self._start_once:
            if self._started:
                return
            self._started = True
        threading.Thread(target=self._start, daemon=True, name="matlab-warm-up").start()

    def _start(self):
        try:
            import matlab.engine
            with self._lock:
                t0 = time.time()
                eng = matlab.engine.start_matlab()
                eng.cd(str(MATLAB_DIR), nargout=0)
                self.status = "loading networks"
                sample = next((REPO / "app" / "samples").glob("*.jpg"), None)
                if sample is not None:
                    eng.drishti_screen(str(sample), str(ONNX_DIR), True, nargout=1)
                self._eng = eng
                self.status = f"ready ({time.time() - t0:.0f} s to start)"
        except BaseException as e:  # surfaced on the next screen() call
            self._error = e
            self.status = f"failed: {e}"
        finally:
            self._ready.set()

    def _engine(self):
        self.warm_up()
        self._ready.wait()
        if self._error is not None:
            raise RuntimeError(f"MATLAB backend unavailable: {self._error}")
        return self._eng

    def policy_for(self, site: str | None) -> T.TriagePolicy:
        """Site / camera-specific lambda_low once audit labels exist (Mondrian)."""
        if site and site in self.site_lambdas:
            d = self.policy.as_dict() | {"lambda_low": self.site_lambdas[site], "calibrated_on": f"site {site} audit"}
            return T.TriagePolicy(**d)
        return self.policy

    # ------------------------------------------------------------ screen
    def screen(self, rgb: np.ndarray, attempt: int = 1, site: str | None = None, lang: str = "en", with_images: bool = True) -> dict:
        t0 = time.time()
        eng = self._engine()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "photo.png"  # lossless hand-over; MATLAB reads the file
            cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            with self._lock:  # one MATLAB session serves one request at a time
                m = eng.drishti_screen(str(p), str(ONNX_DIR), with_images, nargout=1)
        images = {k: np.asarray(v, dtype=np.uint8) for k, v in dict(m.pop("images", {})).items()}
        m = _py(m)

        q = m["quality"]
        problems = q["problems"] if isinstance(q["problems"], list) else [q["problems"]]
        instructions = q["instructions"] if isinstance(q["instructions"], list) else [q["instructions"]]
        res = {"attempt": attempt, "site": site, "missing_models": self.missing, "backend": "matlab",
               "quality": {"passed": bool(q["passed"]), "problems": problems, "instructions": instructions,
                           "gradability": None if q.get("gradability") is None else round(q["gradability"], 3),
                           "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in q["metrics"].items()}}}
        if with_images:
            res["images"] = {"photo": _b64(images["photo"])}
        if not res["quality"]["passed"]:
            if attempt >= gate.load_cfg()["max_attempts"]:
                res.update(outcome="refer", reason="ungradable", urgent=False)
            else:
                res.update(outcome="recapture", reason="quality")
            res["message"] = worker_message(res, lang)
            res["seconds"] = round(time.time() - t0, 2)
            return res

        probs = np.asarray(m["probs"], dtype=float).ravel()
        probs = probs / probs.sum()
        p_ref = float(probs[2:].sum())
        res["grade"] = {"probs": [round(float(p), 4) for p in probs], "p_referable": round(p_ref, 4),
                        "most_likely": ICDR_NAMES[int(np.argmax(probs))],
                        "members_p_referable": [round(float(x), 4) for x in np.ravel(m["member_p_referable"])]}
        if with_images:
            res["images"]["gradcam"] = _b64(images["gradcam"])
            res["images"]["lesions"] = _b64(images["lesions"], 768)

        rules = m["rules"]
        # MATLAB field names cannot hold hyphens; match drishti/rules/icdr.py
        rules["quadrants"] = {k.replace("_", "-"): {kk: int(vv) for kk, vv in v.items()} for k, v in rules["quadrants"].items()}
        rules["counts"] = {k: int(v) for k, v in rules["counts"].items()}
        for k in ("lesion_flag", "dme_flag"):
            rules[k] = bool(rules[k])
        for k in ("evidence_grade", "dme_risk"):
            rules[k] = int(rules[k])
        rules["trace"] = rules["trace"] if isinstance(rules["trace"], list) else [rules["trace"]]
        res["rules"] = rules
        kinds = m["lesion_kinds"]
        rows = np.asarray(m["lesions"], dtype=float).reshape(-1, 5)
        res["lesions"] = [{"kind": kinds[int(k) - 1], "x": round(x, 2), "y": round(y, 2),
                           "area_px_orig": round(a, 1), "peak": round(pk, 3)} for x, y, a, pk, k in rows[:400]]

        # --- triage: the same policy and flags as drishti/pipeline.py
        pol = self.policy_for(site)
        tcfg = json.loads(TRIAGE_CFG.read_text()) if TRIAGE_CFG.exists() else {"use_lesion_flag": False, "use_dme_flag": True}
        tri = T.triage(p_ref, probs, pol,
                       lesion_flag=bool(tcfg["use_lesion_flag"] and rules["lesion_flag"]),
                       dme_flag=bool(tcfg["use_dme_flag"] and rules["dme_flag"]))
        urgent = tri["urgent"] or rules["dme_flag"]
        res.update(outcome=tri["outcome"], urgent=urgent, reasons=tri["reasons"],
                   grade_set=[ICDR_NAMES[tri["set"][0]], ICDR_NAMES[tri["set"][1]]] if tri["set"][0] != tri["set"][1] else [ICDR_NAMES[tri["set"][0]]],
                   policy={"lambda_low": pol.lambda_low, "lambda_high": pol.lambda_high, "calibrated_on": pol.calibrated_on})
        res["message"] = worker_message(res, lang)
        res["seconds"] = round(time.time() - t0, 2)
        return res
