# DrishtiTrust explained in plain words

Written 20 September 2026 for the team; numbers updated 24 September for the final model `ens_d3`. No jargon without a definition.
Numbers here come from `results/VALIDATION.md`, which is generated from the
raw result files.

---

## 1. The problem

Diabetes damages the small blood vessels at the back of the eye (the retina).
Caught early, treatment saves sight. Caught late, the blindness is permanent.
India does not have enough eye doctors to examine every diabetic every year.

So: a health worker takes a photograph of the retina with a cheap camera, and
software decides who needs to see a specialist.

## 2. What we built

A photo goes in one end, a decision comes out the other.

1. **Photo-quality check** — is the picture readable at all (dark, blurry,
   off-centre)? If not, the worker gets a specific instruction ("too dark, dim
   the room lights and move closer") and retakes it. After three failures the
   patient is referred for an in-person exam, because an unreadable photo is
   not a healthy patient.
2. **The grader** — a neural network that reads the photo and outputs a score
   from 0 to 1: how likely this eye has disease serious enough to need a
   specialist.
3. **The lesion finder** — a second network that outlines the actual damage
   (tiny vessel bulges, bleeds, fatty deposits, cotton-wool spots) and counts
   it. It also locates the optic disc and the fovea, the centre of vision, so
   damage can be described by position.
4. **The rule engine** — turns those counts into the standard clinical wording
   ("more than microaneurysms alone, so at least moderate disease"), and
   measures how close the fatty deposits are to the centre of vision.
5. **The decision logic** — converts the score into one of four actions:
   re-screen next year, retake the photo, send for a 30-second remote look by
   an eye doctor, or refer to hospital.
6. **The app** — the health worker's screen, the doctor's review screen with a
   stopwatch, a referral tracker that chases patients who don't turn up, a
   dashboard, and a mode for timing real ophthalmologists.

Plus a simulation of a whole district (100,000 patients a year) and MATLAB
versions of the code, because MathWorks sponsors this problem statement.

## 3. The vocabulary

- **Sensitivity** — of the people who *have* the disease, the share we catch.
  Missing them is the dangerous error. "Seek the sick."
- **Specificity** — of the people who are *healthy*, the share we correctly
  leave alone. Failing here drags well people to hospital. "Spare the healthy."
- **Threshold / cutoff** — the model outputs a score; we pick the line above
  which we act. A lower line catches more disease and raises more false alarms.
  The two above always trade off against each other.
- **AUC** — how well the model *ranks* sick above healthy, independent of where
  the line is drawn. 1.0 perfect, 0.5 a coin toss. Ours is 0.95–0.99 depending on the camera.
- **Image level vs exam level** — one photo, versus one patient visit (two
  eyes). A visit is referred if either eye looks bad, so a visit gives two
  chances for a false alarm; exam-level specificity is always the lower number.
- **External test set** — photos from a hospital and camera the model never saw
  while learning. The only honest test.
- **Calibration / per-site recalibration** — moving the cutoff for one clinic's
  camera using a few hundred locally graded photos. It adapts the alarm level,
  it does not retrain the model.
- **Prediction set** — instead of one grade, the range of grades that is
  plausible for this photo ("mild to moderate"). A wide range means "unsure".
- **IDx-DR** — a commercial, FDA-approved system that does this job
  autonomously. It published results on Messidor-2, one of our test sets, so
  it is a fair yardstick: **96.8% sensitivity, 87.0% specificity**.

## 4. The datasets

**Learnt from:**
- **DDR** — hospitals across China.
- **APTOS** — Aravind Eye Hospital, rural Tamil Nadu. This is the set the
  problem statement itself attaches.
- **EyePACS** — about 88,700 photos from a US screening programme with many
  different cameras (both halves of the 2015 Kaggle competition).

**Used only to choose between models (never learnt from, never quoted as results):**
- **ODIR** — 6,300 photos, Chinese hospitals, mixed cameras; grades read from
  doctors' free-text notes, so noisy.
- **DeepDRiD** — 2,000 photos from Shanghai graded by ophthalmologists.

**Tested on (never trained on, looked at once per final model):**
- **IDRiD** — 516 photos, one Indian clinic in Nanded, Kowa camera. Some have
  every lesion outlined by hand, which is how we checked the lesion finder and
  the heat map.
- **Messidor-2** — 1,748 photos / 874 patient visits, French hospitals, Topcon
  camera. Chosen because IDx-DR published on exactly these images.

**Also:** HRF (45 photos with every vessel traced) for the vessel model.

## 5. What the numbers say

| Where | Of 100 who need a specialist, we catch | Of 100 who are healthy, we correctly leave alone |
|---|---|---|
| Held-out photos from the sources we trained on | 95–96 | 92–93 |
| IDRiD (unseen Indian clinic), per photo | 93 | 87 |
| Messidor-2 (unseen French clinic), per photo | 95 | 90 |
| Messidor-2, per patient visit | 96 | 87 |
| **IDx-DR on those same visits** | **97** | **87** |

The problem statement asks for more than 90 caught and more than 85 left
alone. **We meet that on both cameras we have never seen.** On Messidor-2,
per patient visit, we land on the commercial system's published numbers:
96.3% / 87.4% against its 96.8% / 87.0%, and a ranking score (AUC, see
below) of 0.982 against 0.980.

**Is that "beating IDx-DR"?** No: it is a tie. With 874 visits, the
uncertainty on each of those numbers is about ±3 points, so neither system is
measurably better. And IDx-DR also checks photo quality and looks at both
eyes together in ways our number does not. A tie with an FDA-authorised
device, on a camera we never trained on, with the cutoff fixed before we
looked, is the honest claim.

**What AUC means, in one line:** take one eye that needs a specialist and one
that does not; AUC is how often the model gives the sick eye the higher
score. 0.5 is a coin toss, 1.0 is perfect.

**How it got here** (Messidor-2, per patient visit, AUC):

| Model | When | AUC |
|---|---|---|
| First honest model | 19 Sep, laptop | 0.939 |
| + EyePACS, bigger networks, ensemble | 21 Sep, laptop | 0.959 |
| Trained properly on a 32 GB GPU | 24 Sep, workstation | 0.965 |
| **+ the other half of EyePACS** | **24 Sep, workstation** | **0.982** |

Nearly all of it came from data: more photographs from more cameras. Ideas
that sounded clever (combining both eyes, re-tuning the cutoff from unlabelled
photos, a cheap-camera simulator, a retinal "foundation model" called
RETFound) were built, measured and rejected.

## 6. The three findings that matter

1. **A trap inside a public dataset.** DDR's official split is arranged so that
   every photo from one of its two cameras is labelled healthy. A model trained
   on it can learn "this camera means healthy" instead of learning disease. Our
   first model scored a beautiful 0.987 because of that. We re-split the data
   and retrained; the honest number was lower. Any team using DDR's official
   split is probably fooling itself.

2. **The "two AIs cross-checking each other" design is harmful.** We built it
   and measured it. On 1,744 unseen photos, letting the lesion rules overrule
   the grader missed **7** patients out of 457 who needed care. Simply making
   the grader itself more cautious, at exactly the same review workload, missed
   **4**. So the cross-check costs safety at equal cost. In this build the rules
   explain the decision; they do not make it.

3. **The heat map is honest, and we can prove it.** The "why" picture puts 2.3
   times more heat on doctor-marked damage than chance would, and its hottest
   point lands on real damage 88% of the time (chance: 29%). Most teams show a
   heat map and assert it means something; this one is measured against expert
   pixel annotations.

Supporting results: lesion finding averages 0.59 (the 2018 IDRiD challenge
winners got 0.65); the optic disc is located within 13 pixels and the fovea
within 32 pixels on 4288-pixel images; vessels reach 0.980 AUC; unreadable
photos are detected at 0.993 AUC.

## 7. The conclusion

**The system works, and on public data it now matches a certified
commercial device.** It is still not deployable: it has not been tested on a
real Indian phone camera, and no ophthalmologist has yet timed the 30-second
review.

**What to say in the pitch:**

- We meet the problem statement's target on two cameras and countries the
  model never saw, with the cutoff fixed in advance.
- On IDx-DR's own test photographs we tie IDx-DR. We say "tie", because
  that is what the uncertainty allows.
- Uncertain cases go to a 30-second remote look by an eye doctor instead of
  a forced yes/no. On the 78% of photographs the system decides by itself, it
  catches 98.5% of the disease.
- Every number was obtained without peeking: models were chosen on two
  practice sets, the choice was written down before the real test, and every
  look at the real test is logged (`docs/EXPERIMENTS.md`).

## 8. What is not done

- MATLAB and Simulink scripts are written but have never been run (no licence
  on any build machine so far). `matlab/check_export.m` is the first test.
- No test on a real Indian phone-camera dataset.
- No measurement of a real ophthalmologist's review time — that is what the
  Aravind contact is for (`docs/READER_STUDY.md`).
- Referral-completion numbers in the simulation are assumptions, not measurements.

## 9. Where things are

```
SIH26038/
├── data/raw/          DDR, APTOS, EyePACS, IDRiD, Messidor-2, DeepDRiD, HRF (not in git)
├── data/processed/    cached crops + manifest.csv (the split table)
└── drishtitrust/      github.com/AbhijithSriram/diabetic-retinopathy-screening
    ├── drishti/       the pipeline (quality, grading, lesions, landmarks,
    │                  vessels, rules, conformal triage, explain, report)
    ├── app/           the web app (field, review, referrals, dashboard, study)
    ├── sim/           district simulation
    ├── matlab/        MATLAB + Simulink scripts (untested)
    ├── runs/          trained models + ONNX exports (not in git; MATLAB
    │                  models are a GitHub release)
    ├── results/       VALIDATION.md and the raw JSON behind it
    └── docs/          DATASETS, EXPERIMENTS, LIMITATIONS, READER_STUDY, IDEAS, this file
```

Read `results/VALIDATION.md` first. Every number in it is generated by
`scripts/make_report.py` from the JSON results; none is typed by hand.
