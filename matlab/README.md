# MATLAB / Simulink

These files are the MathWorks side of the project. **None of them has been run
yet**: no build machine so far has had a MATLAB licence. Run them on a licensed
machine (**R2025a or later** for the ConvNeXt member's `Gelu` layer; R2024b is
enough for an all-EfficientNet ensemble) before the demo, and fix anything that
breaks. `check_export.m` is the first thing to run: it says in one line whether
MATLAB reproduces the Python numbers.

| File | What it does | Toolboxes |
|---|---|---|
| `drishti_preprocess.m` | FOV crop + square + resize (mirrors `drishti/preprocess.py` `prepare`) | Image Processing |
| `grader_input.m` | Graham normalisation of a crop, exactly as the networks were trained (mirrors `grader_input` / lesion `full_input`) | Image Processing |
| `normalise_imagenet.m` | ImageNet mean/std normalisation | none |
| `grade_ensemble.m` | imports each `grader_<run>.onnx`, flip-averages its logits, forms CORN class probabilities, averages across members -> P(ICDR grade), P(referable) | Deep Learning (+ ONNX converter), Image Processing |
| `check_export.m` | runs `grade_ensemble` on `app/samples/*.jpg` and compares with the Python reference values stored in `manifest.json`; prints PASS/CHECK | as above |
| `quality_gate.m` | focus / exposure / evenness / framing checks with retake instructions | Image Processing |
| `run_pipeline.m` | quality gate -> grader ensemble + Grad-CAM -> lesion maps (flip-TTA in logit space, FOV mask, calibrated peak filter) -> connected components (weighted centroids = sub-pixel MA centres) -> rule engine -> triage policy -> figure | Deep Learning (+ ONNX converter), Image Processing |
| `rule_engine.m` | ICDR evidence, quadrant counts, DME risk (mirrors `drishti/rules/icdr.py`) | none |
| `build_simulink_model.m` | builds `drishti_district.slx`, a SimEvents model of the district workflow, from `sim/params.json` | Simulink, SimEvents |
| `district_sim.m` | the same capacity model stepped day by day in base MATLAB (cross-check) | none |

Steps:

1. In Python: `python scripts/export_matlab.py --graders <members> --ens <ensemble run>`
   (for the current final model: `--graders v2s_all,b0_768_all,cnx_all_wd --ens ens_vrc3b`).
   This writes `runs/onnx_matlab/`: one `grader_<run>.onnx` per ensemble member,
   `lesions.onnx`, `landmarks.onnx` and `manifest.json` (thresholds, triage policy,
   operator audit, parity figures, reference outputs). Before writing, it
   checks every network against onnxruntime and replays this folder's
   ensemble recipe in NumPy against the ensemble's recorded predictions.
   The graphs are kept to built-in MATLAB layers: fixed input sizes, no shape
   arithmetic, squeeze-excite means as global average pooling, and the CORN
   arithmetic moved out of the network into `grade_ensemble.m`. The one
   exception is ConvNeXt's channel LayerNorm, which exports as
   Transpose + LayerNormalization; MATLAB imports the Transpose as an
   auto-generated custom layer.
2. In MATLAB, from this folder:
   ```matlab
   T = check_export();             % PASS expected; differences ~0.01
   r = run_pipeline('../app/samples/hrf_01_dr.jpg');
   mdl = build_simulink_model();   % prints any block parameter the release rejects
   out = sim(mdl);
   S = district_sim([], 2, 0.6, 0.40);
   ```
3. If `build_simulink_model` prints `set_param ... rejected`, run
   `get_param('drishti_district/<Block>','DialogParameters')` to find the
   parameter name your release uses, and edit it in the script.
