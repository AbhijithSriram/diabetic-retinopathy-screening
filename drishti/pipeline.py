"""End-to-end screening of one photograph.

    quality gate -> grader (+ prediction set, Grad-CAM) -> lesions + landmarks
    -> rule engine -> triage policy -> plain-language messages

Any model that has not been trained yet is skipped and the result says so,
so the app works (with reduced output) at every stage of the build.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import GRADE_SIZE, LESION_SIZE, RUNS, ICDR_NAMES
from .conformal import triage as T
from .data.datasets import grader_input, to_tensor
from .grading.model import class_probs
from .preprocess import prepare
from .quality import gate
from .report.messages import worker_message

DEFAULT_MODELS = {"grader": "b0_512", "seg": "seg_r34", "lmk": "lmk_r18", "qual": "qual_b0"}
MODELS_CFG = Path(__file__).resolve().parent.parent / "configs" / "models.json"  # which run of each model the app serves
TRIAGE_CFG = Path(__file__).resolve().parent.parent / "configs" / "triage.json"


def _b64(rgb: np.ndarray, size: int = 512) -> str:
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA) if rgb.shape[0] != size else rgb
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


class Screener:
    def __init__(self, models: dict | None = None, device: str | None = None):
        chosen = json.loads(MODELS_CFG.read_text()) if MODELS_CFG.exists() else {}
        self.names = {**DEFAULT_MODELS, **chosen, **(models or {})}
        # device_count as well: with CUDA_VISIBLE_DEVICES set to "" torch reports
        # CUDA available but has no device, and loading onto it would fail
        self.device = device or ("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
        self.grader = self.seg = self.lmk = self.qual = None
        self.members = []  # (model, input size) per ensemble member
        self.seg_thr, self.policy, self.site_lambdas = None, None, {}
        self.missing = []
        self._load()

    def _load(self):
        n = self.names
        # "grader" names either one run or an ensemble made by scripts/ensemble.py,
        # whose args.json lists its members; the policy is the named run's either way
        gdir = RUNS / n["grader"]
        members = [n["grader"]]
        if (gdir / "args.json").exists():
            members = json.loads((gdir / "args.json").read_text()).get("ensemble_of", members)
        paths = [RUNS / m / "best.pt" for m in members]
        if all(p.exists() for p in paths):
            from .explain.gradcam import load_grader
            for p in paths:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                self.members.append((load_grader(p, self.device), int(ck.get("size", GRADE_SIZE))))
            self.grader = self.members[0][0]  # Grad-CAM comes from the first member
            pol = gdir / "policy.json"
            if pol.exists():
                self.policy = T.TriagePolicy(**json.loads(pol.read_text()))
        else:
            self.missing.append("grader")
        p = RUNS / n["seg"] / "best.pt"
        if p.exists():
            from .lesions.infer import load_model
            self.seg, self.seg_thr = load_model(p, self.device)
        else:
            self.missing.append("lesions")
        p = RUNS / n["lmk"] / "best.pt"
        if p.exists():
            from .landmarks import landmarks as LM
            self.lmk = LM.load(p, self.device)
        else:
            self.missing.append("landmarks")
        p = RUNS / n["qual"] / "best.pt"
        if p.exists():
            from .quality.train import build
            self.qual = build(pretrained=False)
            self.qual.load_state_dict(torch.load(p, map_location=self.device, weights_only=False)["state"])
            self.qual.eval().to(self.device)
        else:
            self.missing.append("gradability")
        if self.policy is None:
            self.missing.append("triage policy")

    def policy_for(self, site: str | None) -> T.TriagePolicy | None:
        """Site / camera-specific lambda_low once audit labels exist (Mondrian)."""
        if self.policy is None:
            return None
        if site and site in self.site_lambdas:
            d = self.policy.as_dict() | {"lambda_low": self.site_lambdas[site], "calibrated_on": f"site {site} audit"}
            return T.TriagePolicy(**d)
        return self.policy

    @torch.no_grad()
    def _gradability(self, crop512: np.ndarray) -> float | None:
        if self.qual is None:
            return None
        from .quality.train import SIZE
        x = to_tensor(cv2.resize(crop512, (SIZE, SIZE), interpolation=cv2.INTER_AREA))[None].to(self.device)
        return float(torch.sigmoid(self.qual(x).float())[0, 0])

    def screen(self, rgb: np.ndarray, attempt: int = 1, site: str | None = None, lang: str = "en", with_images: bool = True) -> dict:
        t0 = time.time()
        small = prepare(rgb, GRADE_SIZE)
        qcfg = gate.load_cfg()
        grad = self._gradability(small.image)
        q = gate.check(rgb, grad, qcfg)
        res = {"attempt": attempt, "site": site, "missing_models": self.missing,
               "quality": {"passed": q.passed, "problems": q.problems, "instructions": q.instructions,
                           "gradability": None if grad is None else round(grad, 3),
                           "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.metrics.items()}}}
        if with_images:
            res["images"] = {"photo": _b64(small.image)}
        if not q.passed:
            if attempt >= qcfg["max_attempts"]:
                res.update(outcome="refer", reason="ungradable", urgent=False)
            else:
                res.update(outcome="recapture", reason="quality")
            res["message"] = worker_message(res, lang)
            res["seconds"] = round(time.time() - t0, 2)
            return res

        # --- grader
        if self.grader is not None:
            # as scripts/ensemble.py: each member at its own resolution with flip
            # TTA on temperature-scaled logits, then the mean of class probabilities
            crops = {GRADE_SIZE: small.image}
            member_probs = []
            for model, size in self.members:
                if size not in crops:
                    crops[size] = prepare(rgb, size).image
                x = to_tensor(grader_input(crops[size]))[None].to(self.device)
                with torch.no_grad():
                    lg = model(x).float()
                    lg = (lg + model(torch.flip(x, [3])).float()) / 2 / model.temperature
                member_probs.append(class_probs(lg)[0].cpu().numpy())
            probs = np.mean(member_probs, axis=0)
            probs = probs / probs.sum()
            p_ref = float(probs[2:].sum())
            res["grade"] = {"probs": [round(float(p), 4) for p in probs], "p_referable": round(p_ref, 4),
                            "most_likely": ICDR_NAMES[int(np.argmax(probs))]}
            if with_images:
                from .explain.gradcam import cam_for_crop, heat_overlay
                cam = cam_for_crop(self.grader, small.image, self.device)
                res["images"]["gradcam"] = _b64(heat_overlay(small.image, cam))
        # --- lesions, landmarks, rules
        rules = None
        if self.seg is not None:
            from .lesions.infer import components, optic_disc, overlay, predict_maps, to_original
            from .rules.icdr import assess
            big = prepare(rgb, LESION_SIZE)
            maps = predict_maps(self.seg, big.image, self.device)
            les = components(maps, self.seg_thr)
            od = optic_disc(maps, self.seg_thr.get("OD", 0.5))
            fovea, fconf = None, 0.0
            if self.lmk is not None:
                from .landmarks import landmarks as LM
                lm = LM.predict(self.lmk, small.image, self.device)
                s = LESION_SIZE / GRADE_SIZE
                fovea, fconf = (lm["fovea"][0] * s, lm["fovea"][1] * s), lm["fovea"][2]
            rules = assess(les, od=od, fovea=fovea, fovea_conf=fconf)
            res["rules"] = rules
            res["lesions"] = to_original(les, big.geometry)[:400]
            if with_images:
                ov = overlay(big.image, maps, self.seg_thr)
                if fovea is not None:
                    cv2.circle(ov, (int(fovea[0]), int(fovea[1])), int(rules["disc_diameter_px"]), (0, 255, 255), 2)
                if od is not None:
                    cv2.circle(ov, (int(od[0]), int(od[1])), int(od[2]), (0, 200, 0), 2)
                res["images"]["lesions"] = _b64(ov, 768)
        # --- triage
        pol = self.policy_for(site)
        if pol is not None and "grade" in res:
            tcfg = json.loads(TRIAGE_CFG.read_text()) if TRIAGE_CFG.exists() else {"use_lesion_flag": False, "use_dme_flag": True}
            tri = T.triage(res["grade"]["p_referable"], np.array(res["grade"]["probs"]), pol,
                           lesion_flag=bool(tcfg["use_lesion_flag"] and rules and rules["lesion_flag"]),
                           dme_flag=bool(tcfg["use_dme_flag"] and rules and rules["dme_flag"]))
            urgent = tri["urgent"] or bool(rules and rules["dme_flag"])
            res.update(outcome=tri["outcome"], urgent=urgent, reasons=tri["reasons"],
                       grade_set=[ICDR_NAMES[tri["set"][0]], ICDR_NAMES[tri["set"][1]]] if tri["set"][0] != tri["set"][1] else [ICDR_NAMES[tri["set"][0]]],
                       policy={"lambda_low": pol.lambda_low, "lambda_high": pol.lambda_high, "calibrated_on": pol.calibrated_on})
        else:
            res.update(outcome="review", urgent=False, reasons=["grader or policy not available: human review"])
        res["message"] = worker_message(res, lang)
        res["seconds"] = round(time.time() - t0, 2)
        return res
