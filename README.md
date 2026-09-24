# DrishtiTrust

Explainable, uncertainty-aware diabetic retinopathy screening for rural India,
built for Smart India Hackathon 2026, problem statement **SIH26038**
(MathWorks): *Explainable AI for Diabetic Retinopathy Screening in Rural India*.

> Research prototype. Not a medical device. Every patient-facing screen says so.

**Results:** [`results/VALIDATION.md`](results/VALIDATION.md), generated from
the JSON results files, with confidence intervals. Final model `ens_e3`:
an average of three EfficientNets (V2-S at 512, V2-S at 768, B0 at 768)
trained on DDR + APTOS + EyePACS, **selected on ODIR, committed to in writing
and scored once on the external sets** ([docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
§15). Every network in it imports into MATLAB as built-in layers
([matlab/README.md](matlab/README.md)).

| | Result |
|---|---|
| Referable DR, DDR held-out test (2,503) | AUC 0.989; sensitivity 95.1%, specificity 93.7% |
| Referable DR, APTOS held-out test (534) | AUC 0.987; sensitivity 96.7%, specificity 91.0% |
| Referable DR, ODIR held out (6,228; China, mixed cameras, never trained on) | AUC 0.946 |
| Referable DR, IDRiD external (516) | AUC 0.964; sensitivity 92.9%, specificity 79.8% |
| **Referable DR, Messidor-2 external (1,744), frozen threshold** | **AUC 0.967; sensitivity 91.7% (88.8–93.9), specificity 86.0% (84.0–87.8) — meets the PS target by point estimate** |
| Messidor-2 at exam level, frozen threshold, vs IDx-DR (96.8% / 87.0%) | AUC 0.965; sensitivity 93.7% (90.2–96.0), specificity 80.1% (76.8–83.1) |
| Messidor-2 exam-level curve, if set to IDx-DR's 96.8% sensitivity | specificity 69.5% (curve comparison, not a deployable threshold) |
| Messidor-2 exam-level curve, if set to 90% sensitivity | specificity 87.6% (curve comparison, not a deployable threshold) |
| Grad-CAM vs ophthalmologist lesion masks (IDRiD, 81) | 2.3× chance energy on lesions; hottest pixel on a lesion 88% of the time |
| Lesions, IDRiD test (mean AUPR of 4) | 0.592 (2018 challenge winners 0.647) |
| Optic disc / fovea, IDRiD test | 13 px / 32 px mean error (4288-px images) |
| Vessels, HRF test | AUC 0.980 |
| Ungradable-image detection, DDR test | AUC 0.993 |

**The honest headline:** with its threshold fixed in advance on our own
calibration data, the grader meets the problem statement's >90% / >85% target
on held-out internal data and, by point estimate, on Messidor-2, a camera and
country it has never seen: **91.7% sensitivity, 86.0% specificity** per image.
It misses the specificity target on IDRiD (79.8%) and at Messidor-2 exam level
(80.1%), where two eyes give two chances of a false alarm. It does **not** beat
IDx-DR: their published operating point sits above our curve, and at their
96.8% sensitivity we give up 17.5 points of specificity (22 with the previous
model, `ens_vrc3b`; see §15 for what improved and what did not).

The "if set to" rows are points on the ROC curve, read off the test set to
compare our curve with IDx-DR's. They describe what the model *can*
distinguish; they are not thresholds anyone could have chosen in advance, and
they are not claimed as achieved performance.

What moved it was data and training, not ideas. Adding EyePACS was worth +0.037 AUC on a
camera the model had never seen; training each network properly on a 32 GB
workstation instead of an 8 GB laptop +0.005–0.008 per network; a larger backbone +0.013, higher resolution
+0.016 on its own and +0.003 once the data was there. Three ideas that sounded
better — combining both eyes, re-tuning the threshold from unlabelled photos,
and a physically-grounded cheap-camera simulator — were built, measured and
rejected. All of it is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), including
a register of every look at a sealed test set.

## What it does

```
photo ─► quality gate ──fail──► specific retake instruction (3rd failure ⇒ in-person referral)
          │pass
          ▼
   ordinal CNN grader ─► P(referable), 90% plausible grade range, Grad-CAM
   lesion U-Net + landmarks ─► counted, located lesions ─► ICDR rule engine
          │
          ▼
   triage policy (conformal risk control, fitted once on calibration data)
     recall   : confident no referable DR ─► re-screen in 12 months
     review   : uncertain, or lesions / macular-oedema signs ─► 30-second tele-review
     refer    : confident referable ─► referral-completion tracker
```

| PS 26038 module | Where |
|---|---|
| 1. Quality assessment and enhancement, recapture guidance | `drishti/quality/`, `drishti/preprocess.py` |
| 2. Structure segmentation: optic disc and fovea, vessels, MA, HE, EX, SE | `drishti/landmarks/`, `drishti/vessels/`, `drishti/lesions/` |
| 3. ICDR 0-4 grading, >90% sensitivity / >85% specificity for referable DR | `drishti/grading/`, `drishti/eval/grader_eval.py` |
| 4. Grad-CAM, lesion evidence tied to ICDR criteria, calibrated confidence, reports for review in under 30 s | `drishti/explain/`, `drishti/rules/`, `drishti/conformal/`, `app/` (review screen + reader study) |
| 5. Simulink workflow simulation for 100k+ patients a year | `sim/district.py` (SimPy), `matlab/build_simulink_model.m` (SimEvents), `matlab/district_sim.m` |

Neovascularisation is the one structure we do not segment. The only public
pixel labels are FGADR's 49 masks, which are available on request. It is left
to the grader and the reviewer; see [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Design lines we defend

- **A human stays in the loop.** Uncertain cases go to a 30-second
  ophthalmologist tele-review, not straight to a hospital referral. The
  reader-study mode (`/study`) measures review time directly.
- **The rule engine explains; it does not overrule.** Lesion evidence can send
  a would-be "recall" to review. Its value is tested against simply widening
  the CNN's own review band to the same workload
  (`drishti/eval/rules_eval.py`).
- **The guarantee is on the number that matters.** Conformal risk control
  bounds the share of referable eyes that are auto-recalled, rather than
  only the coverage of a 5-class set.
- **New cameras get their own calibration.** A small random audit of
  auto-recalled cases keeps collecting labels, and each camera or site gets
  its own threshold once there are enough of them (Mondrian conformal).
  Until then the site's cases go to review.
- **One preprocessing path.** Every image, borderline or not, gets the same
  crop and normalisation, in Python and in MATLAB.
- **No free-text generation.** Patient messages are fixed templates
  (English, Hindi and Tamil) filled with the model's outputs.

## Running it

Tested on Linux with an RTX 5000 Ada (32 GB); the first builds ran on Windows
with an 8 GB RTX 4060. Datasets and trained weights are not in this
repository (see [docs/DATASETS.md](docs/DATASETS.md)). The exported MATLAB
models and the sample photographs are published as a release asset:
[v1.0-ens_e3](../../releases/tag/v1.0-ens_e3) (`drishtitrust_matlab_models_ens_e3.zip`,
with placement instructions inside).

```bash
python -m venv --system-site-packages .venv    # needs PyTorch with CUDA already installed
.venv/bin/pip install -r requirements.txt       # .venv/Scripts/pip on Windows

# data: crops, duplicate detection, splits -> data/processed/manifest.csv
python scripts/prepare_data.py
python scripts/append_eyepacs_test.py --labels testLabels15.csv --images <zip> --prefix "resized test 15/"
python scripts/append_deepdrid.py --zip deepdrid_github_snapshot.zip   # second selection set

# graders: the three members of the final ensemble ens_e3
python -m drishti.grading.train --name v2s_e15    --backbone tf_efficientnetv2_s --size 512 --epochs 15 --bs 32 --lr 4e-4 --ema 0.9998 --sources ddr,aptos,eyepacs
python -m drishti.grading.train --name b0_768_e10 --backbone efficientnet_b0     --size 768 --epochs 10 --bs 24 --lr 4e-4 --ema 0.9998 --sources ddr,aptos,eyepacs
python -m drishti.grading.train --name v2s768_e8  --backbone tf_efficientnetv2_s --size 768 --epochs 8  --bs 24 --lr 3e-4 --ema 0.9998 --sources ddr,aptos,eyepacs
python scripts/ensemble.py --runs v2s_e15,b0_768_e10,v2s768_e8 --name ens_e3

# model selection: ODIR + DeepDRiD only, never the sealed sets
python scripts/selection_score.py v2s_e15 b0_768_e10 v2s768_e8
python scripts/ens_search.py v2s_e15 b0_768_e10 v2s768_e8
python scripts/lodo_report.py                   # leave-one-dataset-out generalisation table
python -m drishti.eval.grader_eval --grader ens_e3 --policy-only   # fit the triage policy on valB, no test set scored

# other modules
python -m drishti.lesions.data && python -m drishti.lesions.train --name seg_r34 && python -m drishti.lesions.evaluate --name seg_r34
python -m drishti.landmarks.landmarks train && python -m drishti.landmarks.landmarks evaluate
python -m drishti.vessels.vessels train && python -m drishti.vessels.vessels evaluate
python -m drishti.quality.train

# the one disclosed evaluation of a committed model (logged in docs/EXPERIMENTS.md)
python -m drishti.eval.grader_eval --grader ens_e3 --rules seg_r34
python scripts/make_report.py --grader ens_e3 --seg seg_r34        # results/VALIDATION.md

# MATLAB export (built-in-layer ONNX graphs + parity checks), simulation, app
python scripts/export_matlab.py --graders v2s_e15,b0_768_e10,v2s768_e8 --ens ens_e3
python -m sim.district --sweep --eval eval_ens_e3.json
python -m uvicorn app.main:app --port 8026      # web app
```

The app serves the runs named in `configs/models.json` (an ensemble name
serves all its members); triage switches are in `configs/triage.json`,
referral time frames in `configs/referral.json`.

MATLAB: see [matlab/README.md](matlab/README.md) — start with
`check_export()`. Data sources and leakage controls:
[docs/DATASETS.md](docs/DATASETS.md). Experiment history and every look at a
sealed test set: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
