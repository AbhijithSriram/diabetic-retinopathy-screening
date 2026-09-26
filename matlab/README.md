# MATLAB / Simulink

The MathWorks side of the project. **Verified end to end on MATLAB R2026b
(academic licence) on 25 September 2026**: the grader ensemble reproduces the
Python reference, the full pipeline grades a photograph, and the SimEvents
district model runs and agrees with the base-MATLAB capacity model.

Requires **R2024b or later**. Every network in the final model imports as
built-in layers, so no custom-layer support is needed. Toolboxes: Image
Processing, Deep Learning (+ Converter for ONNX Model Format), and for the
district model Simulink + SimEvents.

`check_export.m` is the first thing to run: it says in one line whether MATLAB
reproduces the Python numbers.

| File | What it does | Toolboxes |
|---|---|---|
| `drishti_preprocess.m` | FOV crop + square + resize (mirrors `drishti/preprocess.py` `prepare`) | Image Processing |
| `grader_input.m` | Graham normalisation of a crop, exactly as the networks were trained | Image Processing |
| `normalise_imagenet.m` | ImageNet mean/std normalisation | none |
| `grade_ensemble.m` | imports each `grader_<run>.onnx`, flip-averages its logits, forms CORN class probabilities, averages across members | Deep Learning (+ ONNX), Image Processing |
| `check_export.m` | runs `grade_ensemble` on `app/samples/*.jpg` and compares with the Python reference values in `manifest.json`; prints PASS/CHECK | as above |
| `quality_gate.m` | focus / exposure / evenness / framing checks with retake instructions | Image Processing |
| `drishti_screen.m` | quality gate -> grader ensemble + Grad-CAM -> lesion maps -> connected components -> landmarks -> rule engine, with report images; networks cached per session. The web app's MATLAB backend calls this | Deep Learning (+ ONNX), Image Processing |
| `run_pipeline.m` | `drishti_screen` -> triage policy (switches from `configs/triage.json`) -> figure | as above |
| `gradability.m` | learned gradability score from `quality.onnx` (added to an existing export with `scripts/export_matlab.py --only-quality`); `check_export` compares it with Python on the samples and on degraded copies | Deep Learning (+ ONNX), Image Processing |
| `resize_area.m` | bit-exact port of OpenCV's `INTER_AREA` downscaling, ties rounded to even; `imresize` is close on sharp photos but moved the gradability score by up to 0.48 on blurred ones | none |
| `on_device.m` | runs inference on the GPU when `canUseGPU()` (2-4 s a photograph instead of ~9 s); `DRISHTI_MATLAB_CPU=1` forces the CPU | Parallel Computing (optional) |
| `rule_engine.m` | ICDR evidence, quadrant counts, DME risk (mirrors `drishti/rules/icdr.py`) | none |
| `build_simulink_model.m` | builds `drishti_district.slx`, a SimEvents model of the district workflow, from `sim/params.json` | Simulink, SimEvents |
| `district_report.m` | turns a `sim()` result into the capacity numbers module 5 asks for | none |
| `district_sim.m` | the same capacity model stepped day by day in base MATLAB (cross-check) | none |
| `probe_simevents.m` | prints the SimEvents block parameter names this release accepts | Simulink, SimEvents |
| `probe_simevents_where.m` | locates the SimEvents block library if `load_system` cannot find it | Simulink |

## Running it

```matlab
% 1. in Python, export the models
%    python scripts/export_matlab.py --graders v2s_d2,v2s768_d2,b0_768_d2 --ens ens_d3
%    (or download the release asset and unzip it into runs/onnx_matlab and app/samples)

% 2. in MATLAB, from this folder
T = check_export();                                  % PASS expected
r = run_pipeline('../app/samples/hrf_01_dr.jpg');    % grade + lesions + Grad-CAM

% 3. the district model
mdl = build_simulink_model();
out = sim(mdl);
T2  = district_report(out);
S   = district_sim([], 2, 0.6, 0.40);                % base-MATLAB cross-check
```

## Verified results, R2026b, 25 Sep 2026

`check_export` on the six sample photographs: **PASS**, largest difference from
the Python reference **0.0049**, and all six land on the same triage decision
as Python.

`run_pipeline` on `hrf_01_dr.jpg`: P(referable) 0.956 (Python 0.951), outcome
`refer` in both. Lesions MA 56 / HE 2 / EX 10 / SE 0 against Python's
53 / 2 / 10 / 0 -- the three extra microaneurysms are blobs sitting on the
detection threshold, which is where a sub-grey-level preprocessing difference
shows up. Fovea (270.99, 263.98) against Python's (271, 262).

District model, 90 working days: 18,685 recalled, 5,660 recalled after review,
6,811 lost to follow-up, **4,741 treated**. Reviewer utilisation 24%, capture
utilisation 4.6%. Scaled to a year that is 32,089 referrals and 13,170
attended, against `district_sim`'s 31,794 and 12,875 -- agreement within 1-3%.
Little's law holds on both queues (hospital: 2.195/h x 1.5075 h = 3.31 against
a reported average length of 3.3094).

## Where MATLAB and OpenCV/Simulink disagree

Everything below was found by running the code against a Python reference.
**None of it raised an error.** Each one silently produced a plausible wrong
number, which is why `check_export` exists and why the district model is
cross-checked against `district_sim` rather than trusted on inspection.

| Where | The trap | What it cost |
|---|---|---|
| `local_average` in `grader_input.m`, `drishti_preprocess.m` | `cv2.GaussianBlur` with `ksize=0` on a float32 image truncates the Gaussian at **4 sigma**; `imgaussfilt` defaults to **2 sigma** | ~2 grey levels on 28% of pixels after Graham's 4x gain; 0.035 of P(referable) on one sample |
| `quality_gate.m` | `cv2.Laplacian(ksize=3)` is built from Sobel second derivatives, giving `[2 0 2; 0 -8 0; 2 0 2]`. The textbook four-neighbour kernel is `ksize=1` | \|lap\| 2.45x too small, focus (a variance) 6x too small, so a sharp photograph was reported as blurry and the pipeline refused to grade it |
| borders | OpenCV defaults to `BORDER_REFLECT_101`, `imfilter`/`imgaussfilt` to `replicate` | small, but free to fix with `'symmetric'` |
| variance | `numpy.var` normalises by N, MATLAB's `var` by N-1 | negligible here, but use `var(x, 1)` to match |
| SimEvents library name | It is **`sldelib`** from R2025b or so, not `simevents`; `simevents` is now only a function that opens the library | `load_system('simevents')` fails outright |
| `Entity Server` | the constant service time is `ServiceTimeValue`; there is no `ServiceTime` | would have been skipped silently, leaving all three servers on their default service time |
| `Entity Input Switch` | has **no actions at all** in this release | the attribute the switch reads could not be computed there; it is now set in the Entity Generator |
| `AttributeName` / `AttributeInitialValue` | validated against each other, so neither can be set alone | both must go in one `set_param` call |
| SimEvents actions | compiled, not interpreted, so every expression needs a provably fixed size and cannot call an ordinary function file | `find(u <= c, 1)` types as `[1 x :?]` and is rejected; `triangular_draw` had to be inlined |
| statistic ports | statistics come **before** the entity output port, and logging an entity port yields a struct rather than a number | an off-by-one that mislabelled utilisation as average wait -- and would have read as a plausible number |

If `build_simulink_model` prints `set_param ... rejected`, run
`probe_simevents` to get the parameter names this release accepts.
