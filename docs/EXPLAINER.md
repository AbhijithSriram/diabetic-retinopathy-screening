# DrishtiTrust explained in plain words

Written 20 September 2026, for the team. No jargon without a definition.
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
  the line is drawn. 1.0 perfect, 0.5 a coin toss. Ours is 0.94–0.98.
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
- **DDR** — 13,673 photos, hospitals across China.
- **APTOS** — 3,662 photos from Aravind Eye Hospital, rural Tamil Nadu. This is
  the set the problem statement itself attaches.

**Tested on (never trained on):**
- **IDRiD** — 516 photos, one Indian clinic in Nanded, Kowa camera. Some have
  every lesion outlined by hand, which is how we checked the lesion finder and
  the heat map.
- **Messidor-2** — 1,748 photos / 874 patient visits, French hospitals, Topcon
  camera. Chosen because IDx-DR published on exactly these images.

**Also:** HRF (45 photos with every vessel traced) for the vessel model.

## 5. What the numbers say

| Where | Of 100 who need a specialist, we catch | Of 100 who are healthy, we wrongly send |
|---|---|---|
| Held-out photos from the sources we trained on | 94–96 | 10–14 |
| IDRiD (unseen Indian clinic) | 93 | 26 |
| Messidor-2, per photo (unseen French clinic) | 94 | 31 |
| Messidor-2, per patient visit | 97 | **43** |
| IDx-DR on those same visits | 97 | 13 |

We find the disease as well as the commercial system and we panic about healthy
people about three times as often.

**Is that just a badly placed cutoff?** No. Cheating by picking the best
possible cutoff using the Messidor-2 answers themselves, the choices are:

| Cutoff | Sensitivity | Specificity |
|---|---|---|
| ours, frozen in advance | 96.7% | 57.0% |
| best possible at that sensitivity | 97.0% | 52.3% |
| best possible at 90% sensitivity | 90.0% | 75.0% |
| best possible at IDx-DR's specificity | 84.8% | 87.0% |

No cutoff reaches 96.8 / 87.0. **The model itself is weaker**, not just the
dial. (On IDRiD the dial did matter: the frozen cutoff was much worse than the
best available one.)

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

**The system works and it is not deployable.** Both are true.

On data resembling its training it meets the problem statement's targets
(>90% sensitivity, >85% specificity). On a camera it has never seen it still
catches the disease but over-refers the healthy. Because the ranking quality
holds up, the model is not blind — it cannot tell where "normal" sits on an
unfamiliar camera, and it is genuinely weaker than a certified commercial
device.

**What to say in the pitch**, which is stronger than a fake 95%:

- Every other team quotes one number from one dataset. We show what happens on
  a camera the model never saw, which is exactly where real deployments fail.
- Our design already answers it: uncertain cases go to a 30-second remote look
  by an eye doctor instead of a bus ride to a hospital, so a false alarm costs
  seconds rather than a lost day's wages.
- The honest claim: *catches 95+% of disease; on a new camera roughly a third
  of healthy patients get a 30-second tele-review, falling as the site
  accumulates a few hundred locally graded photos.*

## 8. What is not done

- MATLAB and Simulink scripts are written but have never been run (no licence
  on the build machine).
- No test on a real Indian phone-camera dataset.
- No measurement of a real ophthalmologist's review time — that is what the
  Aravind contact is for (`docs/READER_STUDY.md`).
- Referral-completion numbers in the simulation are assumptions, not measurements.

## 9. Where things are

```
C:\Users\abhij\SIH\26038\
├── data\raw\          DDR, APTOS, IDRiD, Messidor-2, HRF (not in git)
├── data\processed\    cached crops + manifest.csv (the split table)
└── drishtitrust\
    ├── drishti\       the pipeline (quality, grading, lesions, landmarks,
    │                  vessels, rules, conformal triage, explain, report)
    ├── app\           the web app (field, review, referrals, dashboard, study)
    ├── sim\           district simulation
    ├── matlab\        MATLAB + Simulink scripts (untested)
    ├── runs\          trained models + ONNX exports
    ├── results\       VALIDATION.md and the raw JSON behind it
    └── docs\          DATASETS, LIMITATIONS, READER_STUDY, this file, IDEAS
```

Read `results/VALIDATION.md` first. Every number in it is generated by
`scripts/make_report.py` from the JSON results; none is typed by hand.
