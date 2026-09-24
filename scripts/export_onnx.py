"""Export the trained models to ONNX for MATLAB (importNetworkFromONNX) and
check that onnxruntime reproduces PyTorch's outputs.

Outputs (runs/onnx/):
  grader.onnx     input  image (1,3,512,512), ImageNet-normalised Graham crop
                  output probs (1,5), p_ref (1,1), features (1,C,16,16), logits (1,4)
                  (features + logits let MATLAB compute Grad-CAM itself)
  lesions.onnx    input  image (1,3,H,W) normalised Graham crop at 1440
                  output maps (1,5,H,W) sigmoid probabilities (MA,HE,EX,SE,OD)
  landmarks.onnx  input  image (1,3,512,512); output heatmaps (1,2,512,512)
  manifest.json   temperature, thresholds, triage policy, normalisation constants

    python scripts/export_onnx.py [--grader b0_512 --seg seg_r34 --lmk lmk_r18]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drishti.config import RUNS  # noqa: E402
from drishti.grading.model import class_probs, referable_prob  # noqa: E402


class GraderONNX(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        f = self.m.features(x)
        lg = self.m.logits_from_features(f) / self.m.temperature
        return class_probs(lg), referable_prob(lg)[:, None], f, lg


class SigmoidWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.sigmoid(self.m(x))


def export(model, dummy, path, names_out, dynamic_hw=False):
    model.eval()
    axes = {"image": {2: "H", 3: "W"}} if dynamic_hw else None
    torch.onnx.export(model, dummy, str(path), input_names=["image"], output_names=names_out,
                      opset_version=17, dynamic_axes=axes, do_constant_folding=True)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy)
    ref = ref if isinstance(ref, tuple) else (ref,)
    got = sess.run(None, {"image": dummy.numpy()})
    worst = max(float(np.abs(a.numpy() - b).max()) for a, b in zip(ref, got))
    print(f"{path.name}: max |torch - onnxruntime| = {worst:.2e}")
    assert worst < 1e-3, "ONNX parity check failed"
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grader", default="b0_512")
    ap.add_argument("--seg", default="seg_r34")
    ap.add_argument("--lmk", default="lmk_r18")
    a = ap.parse_args()
    out = RUNS / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "grade_size": 512, "lesion_size": 1440,
                "graham": "4*I - 4*GaussianBlur(I, sigma=size/30) + 128, outside FOV = 128", "parity": {}}

    gp = RUNS / a.grader / "best.pt"
    if gp.exists():
        from drishti.explain.gradcam import load_grader
        g = load_grader(gp, "cpu")
        manifest["temperature"] = float(g.temperature)
        manifest["parity"]["grader"] = export(GraderONNX(g), torch.randn(1, 3, 512, 512), out / "grader.onnx",
                                              ["probs", "p_ref", "features", "logits"])
        pol = RUNS / a.grader / "policy.json"
        if pol.exists():
            manifest["policy"] = json.loads(pol.read_text())
    sp = RUNS / a.seg / "best.pt"
    if sp.exists():
        from drishti.lesions.infer import load_model
        s, thr = load_model(sp, "cpu")
        manifest["lesion_thresholds"] = thr
        manifest["parity"]["lesions"] = export(SigmoidWrap(s), torch.randn(1, 3, 512, 512), out / "lesions.onnx", ["maps"], dynamic_hw=True)
    lp = RUNS / a.lmk / "best.pt"
    if lp.exists():
        from drishti.landmarks.landmarks import load
        manifest["parity"]["landmarks"] = export(SigmoidWrap(load(lp, "cpu")), torch.randn(1, 3, 512, 512), out / "landmarks.onnx", ["heatmaps"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
