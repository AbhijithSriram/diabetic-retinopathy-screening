# SIH submission — deck content and how to claim the results

Drafted 25 September 2026, when the problem statement stood at 245 of 500
submissions and was filling fast. **The deadline is to submit, not to
finish.** If the portal allows editing after filing, file early and refine.

---

## What to claim, and what not to

This section matters more than the slides. Every number below is in
`results/VALIDATION.md`; the judgement about how to present them is not
recorded anywhere else.

**Claim this — it is accurate and defensible:**

> Matches IDx-DR, the first FDA-authorised autonomous AI diagnostic, on its
> own published benchmark: **96.3% sensitivity / 87.4% specificity against
> 96.8% / 87.0%**, on the same photographs, with our threshold fixed in
> advance and the sealed test set opened once.

IDx-DR's published figures sit inside our confidence intervals (sensitivity
93.3-98.0, specificity 84.5-89.8), so "statistically indistinguishable" is
the rigorous phrasing.

**Do not claim a breakthrough, a world first, or novelty in the accuracy.**
Gulshan et al. reached AUC 0.990 on Messidor-2 in 2016; we are at 0.982, so
we sit slightly *below* a result from a decade ago. Matching IDx-DR's
Messidor-2 numbers has been done repeatedly in the literature. The method is
the well-trodden one — ImageNet-pretrained CNNs, ensembled, trained mostly on
EyePACS — and the proof that it is not a breakthrough is that it took about
twelve hours of training on one laptop GPU once the data was in place. An
evaluator who knows the field will think of Gulshan, and the overclaim will
cost more credibility than the claim is worth.

**What is genuinely distinctive, and is where novelty should be claimed:**

1. **Per-site conformal recalibration.** A stated safety guarantee (at most
   3% of referable patients auto-cleared), measured breaking on an unseen
   camera (6.6%), and repaired with about 200 locally graded photographs.
   Guarantees tested to destruction and then fixed are rare here.
2. **The DDR split discovery** — a flaw in a widely used public dataset that
   inflates the numbers other people report, verifiable in five minutes.
3. **Three plausible ideas built, measured and rejected**, including our own
   camera simulator. The negative results are why the positive ones can be
   trusted.
4. **A validated heat map** rather than an asserted one.

**Context worth having ready if challenged:** IDx-DR's own prospective FDA
trial reached 87.2% / 90.7% in real primary-care clinics, about ten points
below its Messidor-2 sensitivity. Curated benchmarks flatter everyone,
including us. We have not been tested prospectively either.

---

## Slide 1 — Proposed Solution

**DrishtiTrust — DR screening that explains itself and knows when to ask for
help.**

*What it is.* A MATLAB-based retinal image pipeline for rural primary health
centres. A health worker photographs the retina; the system checks the image
is usable, grades it on ICDR 0-4, outlines the lesions, explains its
reasoning, and sorts the patient into **re-screen next year / 30-second
tele-review / refer to hospital** — then tracks whether referred patients
arrive.

*How it addresses the problem.*
- About 80% of photographs are decided autonomously. A district screening
  100,000 patients a year needs two ophthalmologists at two hours a day
  (24% utilised), not a reading room.
- A quality gate rejects unusable photographs with a specific instruction;
  after three failures the patient is referred in person, because an
  unreadable photograph is not a healthy patient.
- Every decision ships with a heat map, the lesions behind it, the ICDR
  criterion in clinical wording, and a plausible-grade range.
- It does not stop at "refer": a completion engine chases patients through
  family and the ASHA worker.

*Innovation and uniqueness.* The four points listed above under "What is
genuinely distinctive", plus the IDx-DR comparison worded as above.

## Slide 2 — Technical Approach

*Technologies.* MATLAB R2024b+ (Image Processing, Deep Learning + ONNX
Converter, Computer Vision, Statistics & ML, Simulink + SimEvents); Python /
PyTorch for training; ONNX as the bridge; FastAPI app. Runs on CPU.

*Pipeline.* quality gate -> preprocessing -> {3-CNN ensemble grader (CORN
ordinal, ICDR 0-4) | lesion U-Net at 1440 px with sub-pixel centroids |
landmarks + vessels} -> ICDR rule engine -> Grad-CAM -> conformal triage ->
app and referral tracker -> Simulink district model.

*Methodology.*
- Trained on ~99,000 images (DDR + APTOS + both halves of EyePACS). **ODIR
  (6,392) and DeepDRiD (2,000) held out solely to choose between models;
  IDRiD and Messidor-2 sealed** and opened once, with every look recorded.
- Ensemble: EfficientNetV2-S at 512 and 768, EfficientNet-B0 at 768.
- Calibration: temperature scaling plus conformal prediction sets.
- **MATLAB deployment verified, not assumed:** `check_export.m` compares
  MATLAB against the Python reference — PASS, max difference 0.005,
  identical triage decisions.
- Simulink cross-checked against an independent base-MATLAB model: agreement
  within 1-3%, and Little's law holds on both queues.

## Slide 3 — Feasibility and Viability

*Feasibility — a working prototype, not a proposal.* Quality AUC 0.993;
lesions 0.592 mean AP (2018 IDRiD challenge winners 0.647); optic disc /
fovea 13 px / 32 px; vessels AUC 0.980; **grading on Messidor-2, an unseen
camera: 95.2% / 89.6%, AUC 0.982**; IDRiD 93.2% / 87.1%. The brief's target
is >90% / >85%, met on every external set with the threshold fixed in
advance.

*Risks and mitigations.* Camera shift (measured; per-site recalibration from
~200 local labels restores the guarantee). Label-noise ceiling (public labels
are single-grader; drives the decision to defer rather than overclaim). No
public Indian handheld data (mBRSET credentialing under way).
Neovascularisation not segmented (FGADR requested; proliferative DR still
caught at grading level). Referral non-attendance — 19,000 of 32,000 annual
referrals, which the completion engine targets. Rural connectivity — runs
offline on CPU, bandwidth modelled explicitly in Simulink.

## Slide 4 — Impact and Benefits

77 million diabetic adults, ~18% with DR, 1 ophthalmologist per 100,000 rural
people.

*Capacity, quantified by the Simulink model rather than estimated:* 100,000
screenings a year need two ophthalmologists at two hours a day, 24% utilised,
against 100,000 images read manually. About 80% decided without a specialist.
Each uncertain case costs a 30-second tele-review.

*Clinical.* Early detection prevents up to 90% of vision loss. Sensitivity
comparable to an FDA-authorised device, with a *bounded* miss rate. A human
stays in the loop by design.

*Economic.* A false alarm costs 30 seconds of a doctor's time, not a
patient's lost day's wages and bus fare. Cameras run at 4.6% utilisation —
reading capacity is the constraint, and no new hardware is needed.

*Social.* Screening reaches the village. The referral gap is surfaced and
tracked. Patient messages are fixed templates in English, Hindi and Tamil —
no generated text, so the system cannot invent medical advice.

## Slide 5 — Research and References

Abramoff 2016 IOVS (the IDx-DR comparison); Abramoff 2018 npj Digital
Medicine (FDA pivotal trial); Gulshan 2016 JAMA; Krause 2018 Ophthalmology
(the adjudicated Messidor-2 grades); Shi, Cao & Raschka 2023 (CORN);
Angelopoulos & Bates (conformal prediction); Graham 2015 (illumination
normalisation); Zhou 2023 Nature (RETFound, future work).

Datasets: DDR, APTOS 2019, EyePACS 2015, ODIR-5K, DeepDRiD, IDRiD,
Messidor-2, HRF — full terms in `DATASETS.md`.

Our own record: `results/VALIDATION.md` (every figure with confidence
intervals, generated from raw files); `docs/EXPERIMENTS.md` (14 experiments,
three rejected ideas, and the register of every look at a sealed test set);
`docs/TECHNICAL_BRIEFING.md`; `matlab/README.md`.

---

## If there is room for one more line

The DDR split discovery. It is the single item most likely to make an
evaluator sit up, it is verifiable in five minutes, and it reframes our
lower-looking numbers as the honest ones.
