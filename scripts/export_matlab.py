"""Export the grader ensemble, lesion and landmark models for MATLAB.

    python scripts/export_matlab.py --graders v2s_all,b0_768_all,cnx_all_wd --ens ens_vrc3b

`export_onnx.py` wrote one grader whose graph carried the CORN arithmetic
(Slice, Gather, Clip) and EfficientNet's squeeze-excite means (ReduceMean),
and a lesion model with a dynamic input size (Shape -> Gather -> Resize).
Those are exactly the operators MATLAB's `importNetworkFromONNX` is least
likely to map onto built-in layers. This exporter keeps each graph to plain
network layers:

  * every grader has a fixed input size and two outputs, `logits` (the four
    CORN logits, temperature already folded into the last layer) and
    `features` (the last feature map, for Grad-CAM);
  * squeeze-excite means are exported as GlobalAveragePool;
  * the lesion and landmark nets are exported at a fixed size and simplified,
    so no shape arithmetic survives;
  * CORN probabilities, flip test-time augmentation and the ensemble average
    are done in MATLAB (`matlab/grade_ensemble.m`), a few lines each.

Checks, all of which must pass or the export fails:
  1. per-model parity, PyTorch vs onnxruntime, on real crops, as error
     relative to each output's scale (onnxruntime's fused CPU kernels alone
     move B0-768 logits by ~3e-3 against an fp64 reference, while the graph
     run unfused is within 6e-4 -- the error is in the runtime, not the file);
  2. end-to-end: the MATLAB recipe replayed in NumPy over the ONNX outputs
     reproduces the ensemble's `preds.csv` P(referable) on valA images (the
     training predictions ran under bf16 autocast, so agreement is to ~1e-2,
     not 1e-5);
  3. an operator audit, written to the manifest, listing any operator that
     is not on the list MATLAB converts to built-in layers.

Output: runs/onnx_matlab/ (grader_<run>.onnx, lesions.onnx, landmarks.onnx,
manifest.json).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drishti.config import LESION_SIZE, PROCESSED, RUNS  # noqa: E402
from drishti.data.datasets import grader_input, to_tensor  # noqa: E402
from drishti.preprocess import read_rgb  # noqa: E402

# Operators importNetworkFromONNX maps to built-in layers (R2024b+ for
# LayerNormalization, R2025a+ for Gelu). Anything else becomes an
# auto-generated custom layer: usually fine, but it is flagged.
BUILTIN = {"Conv", "BatchNormalization", "Relu", "Sigmoid", "Mul", "Add", "Sub", "Div",
           "GlobalAveragePool", "MaxPool", "AveragePool", "Flatten", "Gemm", "MatMul",
           "Concat", "Resize", "LayerNormalization", "Gelu", "Softmax", "Clip", "Constant"}


def _se_forward(self, x):
    x_se = F.adaptive_avg_pool2d(x, 1)  # GlobalAveragePool, not ReduceMean
    x_se = self.conv_reduce(x_se)
    x_se = self.act1(x_se)
    x_se = self.conv_expand(x_se)
    return x * self.gate(x_se)


def matlab_friendly(model: nn.Module) -> nn.Module:
    from timm.models._efficientnet_blocks import SqueezeExcite
    for m in model.modules():
        if isinstance(m, SqueezeExcite):
            m.forward = types.MethodType(_se_forward, m)
    return model


class GraderExport(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g
        t = float(g.temperature)
        with torch.no_grad():  # logits / T, folded into the last layer
            g.head.weight.div_(t)
            g.head.bias.div_(t)
            g.temperature.fill_(1.0)

    def forward(self, x):
        f = self.g.features(x)
        return self.g.logits_from_features(f), f


class SigmoidWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.sigmoid(self.m(x))


def op_audit(path: Path) -> dict:
    import onnx
    m = onnx.load(str(path))
    ops = collections.Counter(n.op_type for n in m.graph.node)
    return {"opset": max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
            "ops": dict(ops), "not_builtin": sorted(set(ops) - BUILTIN)}


def export(model, size, path, outputs, opset):
    model.eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(model, dummy, str(path), input_names=["image"], output_names=outputs,
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    import onnxslim
    import onnx
    slim = onnxslim.slim(onnx.load(str(path)))
    onnx.save(slim, str(path))


def ort_session(path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def parity(model, sess, xs) -> float:
    worst = 0.0
    with torch.no_grad():
        for x in xs:
            ref = model(x)
            ref = ref if isinstance(ref, tuple) else (ref,)
            got = sess.run(None, {"image": x.numpy()})
            # relative to each output's scale: a feature map with values near 80
            # cannot be held to the same absolute error as a probability
            worst = max(worst, max(float(np.abs(a.numpy() - b).max() / max(1.0, float(np.abs(a.numpy()).max())))
                                   for a, b in zip(ref, got)))
    return worst


def corn_probs_np(logits: np.ndarray) -> np.ndarray:
    """The MATLAB recipe, in NumPy: chained sigmoids -> class probabilities."""
    s = 1 / (1 + np.exp(-logits))
    c = np.cumprod(s, axis=1)
    upper = np.concatenate([np.ones((len(c), 1)), c], 1)
    lower = np.concatenate([c, np.zeros((len(c), 1))], 1)
    return np.clip(upper - lower, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graders", required=True, help="comma-separated run names (ensemble members)")
    ap.add_argument("--ens", default="", help="ensemble run holding policy.json and preds.csv, for the end-to-end check")
    ap.add_argument("--seg", default="seg_r34")
    ap.add_argument("--lmk", default="lmk_r18")
    ap.add_argument("--out", default=str(RUNS / "onnx_matlab"))
    ap.add_argument("--n-check", type=int, default=24, help="valA images for the end-to-end check")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)

    from drishti.explain.gradcam import load_grader
    man = pd.read_csv(PROCESSED / "manifest.csv", low_memory=False)
    manifest = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "lesion_size": LESION_SIZE,
                "graham": "4*I - 4*GaussianBlur(I, sigma=size/30) + 128, outside FOV = 128",
                "ensemble": "mean over members of CORN class probabilities; each member averages its "
                            "logits over the image and its horizontal flip; P(referable) = p2+p3+p4",
                "graders": [], "audit": {}, "parity": {}}

    names = [n.strip() for n in a.graders.split(",")]
    check = man[(man.ok == 1) & (man.split == "valA")].sample(a.n_check, random_state=0)
    member_probs = []
    for n in names:
        ck = torch.load(RUNS / n / "best.pt", map_location="cpu", weights_only=False)
        size = int(ck.get("size", 512))
        g = matlab_friendly(load_grader(n, "cpu"))
        t = float(g.temperature)
        wrap = GraderExport(g).eval()
        # ConvNeXt's GELU needs opset 20 to stay a single Gelu op (MATLAB geluLayer);
        # EfficientNets have none, so they stay on 17 for older MATLAB releases.
        opset = 20 if "convnext" in ck["backbone"] else 17
        path = out / f"grader_{n}.onnx"
        try:
            export(wrap, size, path, ["logits", "features"], opset)
        except Exception as e:  # legacy exporter may not reach opset 20
            print(f"{n}: opset {opset} failed ({e.__class__.__name__}); retrying at 18")
            opset = 18
            export(wrap, size, path, ["logits", "features"], opset)
        sess = ort_session(path)
        col = "cache768" if size == 768 else "cache"
        xs = [to_tensor(grader_input(read_rgb(p)))[None] for p in check[col]]
        manifest["parity"][path.name] = parity(wrap, sess, xs[:4])
        manifest["audit"][path.name] = op_audit(path)
        # the MATLAB recipe: flip TTA on logits, then CORN class probabilities
        lg = np.stack([(sess.run(None, {"image": x.numpy()})[0][0]
                        + sess.run(None, {"image": torch.flip(x, dims=[3]).numpy()})[0][0]) / 2 for x in xs])
        member_probs.append(corn_probs_np(lg))
        manifest["graders"].append({"file": path.name, "run": n, "backbone": ck["backbone"], "size": size,
                                    "temperature_folded": t, "opset": opset})
        print(f"{path.name}: size {size}, opset {opset}, parity {manifest['parity'][path.name]:.2e}, "
              f"not built-in: {manifest['audit'][path.name]['not_builtin']}", flush=True)
        assert manifest["parity"][path.name] < 1e-3, "PyTorch vs onnxruntime parity failed"

    if a.ens:
        preds = pd.read_csv(RUNS / a.ens / "preds.csv").set_index("image_id")
        p_ref = np.mean(member_probs, axis=0)[:, 2:].sum(1)
        ref = preds.loc[check.image_id, "p_ref"].to_numpy()
        diff = float(np.abs(p_ref - ref).max())
        manifest["parity"]["ensemble_vs_preds_csv"] = diff
        print(f"end-to-end: MATLAB recipe over ONNX vs {a.ens}/preds.csv, max |dP(referable)| = {diff:.4f} "
              f"over {len(ref)} valA images")
        assert diff < 0.03, "ensemble recipe does not reproduce preds.csv"
        pol = RUNS / a.ens / "policy.json"
        if pol.exists():
            manifest["policy"] = json.loads(pol.read_text())
            manifest["policy_from"] = a.ens

    # reference outputs on the app's sample photos, from the original files
    # through the Python preprocessing, for matlab/check_export.m to compare
    from drishti.preprocess import prepare
    from drishti.config import REPO
    refs = []
    for f in sorted((REPO / "app" / "samples").glob("*.jpg")):
        rgb = read_rgb(f)
        probs = []
        for gi in manifest["graders"]:
            sess = ort_session(out / gi["file"])
            x = to_tensor(grader_input(prepare(rgb, gi["size"]).image))[None]
            lg = (sess.run(None, {"image": x.numpy()})[0][0]
                  + sess.run(None, {"image": torch.flip(x, dims=[3]).numpy()})[0][0]) / 2
            probs.append(corn_probs_np(lg[None])[0])
        pm = np.mean(probs, axis=0)
        refs.append({"file": f.name, "p_referable": float(pm[2:].sum()), "probs": [float(v) for v in pm]})
        print(f"reference {f.name}: P(referable) {pm[2:].sum():.4f}")
    manifest["reference"] = refs

    sp = RUNS / a.seg / "best.pt"
    if sp.exists():
        from drishti.lesions.infer import load_model
        s, thr = load_model(sp, "cpu")
        manifest["lesion_thresholds"] = thr
        # logits, not probabilities: Python averages the flip TTA in logit space and
        # pins everything outside the field of view to -30 before the sigmoid
        w = matlab_friendly(s).eval()
        path = out / "lesions.onnx"
        export(w, LESION_SIZE, path, ["logits"], 17)
        manifest["parity"][path.name] = parity(w, ort_session(path), [torch.randn(1, 3, LESION_SIZE, LESION_SIZE)])
        manifest["audit"][path.name] = op_audit(path)
        print(f"{path.name}: parity {manifest['parity'][path.name]:.2e}, "
              f"not built-in: {manifest['audit'][path.name]['not_builtin']}", flush=True)
    lp = RUNS / a.lmk / "best.pt"
    if lp.exists():
        from drishti.landmarks.landmarks import load
        w = matlab_friendly(SigmoidWrap(load(lp, "cpu"))).eval()
        path = out / "landmarks.onnx"
        export(w, 512, path, ["heatmaps"], 17)
        manifest["parity"][path.name] = parity(w, ort_session(path), [torch.randn(1, 3, 512, 512)])
        manifest["audit"][path.name] = op_audit(path)
        print(f"{path.name}: parity {manifest['parity'][path.name]:.2e}, "
              f"not built-in: {manifest['audit'][path.name]['not_builtin']}", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
