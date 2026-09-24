# DrishtiTrust — A Technical Briefing

*Smart India Hackathon 2026 · Problem Statement SIH26038 (MathWorks)*
*Explainable AI for Diabetic Retinopathy Screening in Rural India*

This document explains the whole project from first principles. It assumes no
background in medicine or machine learning. Every technical term is defined
the first time it appears, and the definitions build on each other, so it is
best read in order. By the end, a reader should be able to judge for
themselves how good the results are, where they are weak, and why.

---

## Contents

1. [The one-page version](#1-the-one-page-version)
2. [The problem](#2-the-problem)
3. [The vocabulary you need](#3-the-vocabulary-you-need)
4. [The yardstick: IDx-DR](#4-the-yardstick-idx-dr)
5. [The data](#5-the-data)
6. [The pipeline, stage by stage](#6-the-pipeline-stage-by-stage)
7. [How we kept the numbers honest](#7-how-we-kept-the-numbers-honest)
8. [What we did: the experimental story](#8-what-we-did-the-experimental-story)
9. [Final results](#9-final-results)
10. [How impressive is this, really?](#10-how-impressive-is-this-really)
11. [Limitations](#11-limitations)
12. [How much further it can go, and the plan](#12-how-much-further-it-can-go-and-the-plan)
13. [Appendix: glossary and file map](#13-appendix)

---

## 1. The one-page version

**The problem.** Diabetes quietly damages the small blood vessels at the back
of the eye. Caught early, treatment preserves sight; caught late, the loss is
permanent. India has one of the largest diabetic populations in the world and
far too few eye specialists to examine everyone every year. The answer the
problem statement asks for: a health worker photographs the eye with a cheap
camera, and software decides who needs a specialist — while explaining its
reasoning well enough that a doctor can check it in under 30 seconds.

**What we built.** A complete screening system: it checks whether a photo is
usable, grades the disease on the standard five-point clinical scale, outlines
the individual lesions, explains its decision with a heat map that we
verified against expert annotations, attaches a statistical guarantee to its
own safety, and sorts every patient into *clear*, *doctor review* or *refer* —
then tracks whether referred patients actually reach a hospital. A district
simulation, a MATLAB/Simulink port, and a web app complete it.

**The headline result.** The benchmark we compare against is **IDx-DR**, the
first AI system ever authorised by the US FDA to diagnose a disease without a
doctor. On the same public test photographs IDx-DR was evaluated on, taken on
a camera in a country our model has never seen, and with the decision
threshold fixed *in advance*:

| | Sensitivity (catches disease) | Specificity (spares the healthy) | AUC |
|---|---|---|---|
| **Problem statement target** | > 90% | > 85% | – |
| **DrishtiTrust, per photograph** | **95.2%** | **89.6%** | 0.982 |
| **DrishtiTrust, per patient visit** | **96.3%** | **87.4%** | 0.982 |
| IDx-DR, per patient visit (published) | 96.8% | 87.0% | 0.980 |

**We meet the problem statement's target on both unseen cameras, and per
patient visit we land on IDx-DR's published operating point.** That is a
statistical tie with a certified commercial device, not a win: with 874
visits every figure carries about ±3 points of uncertainty, and IDx-DR also
judges photo quality and both eyes together in ways our number does not. We
got here from 57% specificity at that sensitivity five days earlier, and we
know precisely which changes bought it --- almost all of it was data.

**What makes this different from other entries.** Not the model — every team
has access to the same models. What is unusual is the discipline around it:
a trap we discovered in a widely used public dataset, safety guarantees that
we measured breaking on a new camera and then repaired, explanations we
checked against specialists' own markings, three plausible ideas we built,
tested and *rejected*, and a written register of every time anyone looked at
the sealed test data.

---

## 2. The problem

### 2.1 Diabetic retinopathy

The **retina** is the light-sensitive layer at the back of the eye — the
film in the camera. It is fed by a dense network of tiny blood vessels.
Years of high blood sugar weaken the walls of those vessels. This damage is
called **diabetic retinopathy**, usually shortened to **DR**.

It progresses in stages, and the early stages cause no symptoms at all:

- **Microaneurysms** — tiny bulges in weakened vessel walls, the earliest
  visible sign. A few pixels across in a photograph.
- **Haemorrhages** — small bleeds where vessels leak.
- **Hard exudates** — yellow fatty deposits left behind by leaking fluid.
- **Cotton-wool spots** — pale fluffy patches where nerve tissue has been
  starved of blood.
- **Neovascularisation** — in the most advanced stage, the eye grows fragile
  new vessels that bleed easily and can detach the retina.

A separate but related danger is **diabetic macular oedema** (**DME**):
swelling at the **macula**, the small central area responsible for sharp
reading vision. The **fovea** is the very centre of the macula.

The cruelty of DR is the timing. By the time a patient notices blurred
vision, the damage is often advanced. Treatment — laser, injections — works
best before symptoms appear. The only way to catch it in time is to look at
every diabetic patient's retina regularly.

### 2.2 Why India needs automated screening

The ICMR-INDIAB study (*Lancet Diabetes & Endocrinology*, 2023) estimated
that around **101 million** Indians live with diabetes. Every one of them
should have an annual eye examination. India has on the order of twenty
thousand ophthalmologists, concentrated in cities. The arithmetic does not
work.

A photograph of the retina — a **fundus photograph** — can be taken by a
trained health worker in a village in a couple of minutes. The bottleneck is
*reading* it. If software can reliably decide which photographs show disease
that needs a specialist, scarce specialists spend their time on the patients
who need them.

There is a second, less discussed bottleneck. Studies of Indian screening
programmes have repeatedly found that a large share of patients told to "go
to the district hospital" never arrive — distance, cost and lost wages get in
the way. A screening programme that stops at "refer" has solved half the
problem. *(Population statistics in this section should be cited from their
primary sources before being used in a pitch.)*

### 2.3 What the problem statement asks for

SIH26038 specifies five modules. We built all five:

| Module | What it asks | Where it lives |
|---|---|---|
| 1. Quality assessment | Detect unusable photos, tell the operator how to retake | `drishti/quality/` |
| 2. Structure segmentation | Find the optic disc, fovea, vessels and each lesion type, down to microaneurysms | `drishti/landmarks/`, `vessels/`, `lesions/` |
| 3. Grading | Grade DR on the international five-point scale; **>90% sensitivity and >85% specificity for referable DR** | `drishti/grading/` |
| 4. Explainability | Heat maps, lesion evidence tied to clinical criteria, calibrated confidence, reports a doctor can verify **in under 30 seconds** | `drishti/explain/`, `rules/`, `conformal/`, `app/` |
| 5. Simulation | A Simulink model of a district screening 100,000+ patients a year, including review capacity | `sim/`, `matlab/` |

---

## 3. The vocabulary you need

This section is the foundation for everything after it.

### 3.1 The grading scale

Doctors grade DR on the **International Clinical Diabetic Retinopathy
scale**, **ICDR** for short:

| Grade | Name | Meaning |
|---|---|---|
| 0 | No DR | Healthy |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than just microaneurysms |
| 3 | Severe | Extensive bleeding and damage |
| 4 | Proliferative | New vessels growing — sight-threatening |

A patient with **grade 2 or worse** needs a specialist. This is called
**referable DR**, and it is the thing the whole system exists to detect.
Grade 0 and 1 patients can safely be re-screened next year.

### 3.2 Sensitivity and specificity

Any screening test makes two kinds of mistake, and they are not equally bad.

- **Sensitivity** — of the people who *have* referable disease, what fraction
  does the test catch? Missing a sick person is the dangerous error: they go
  home reassured and lose their sight. *Seek the sick.*
- **Specificity** — of the people who are *healthy*, what fraction does the
  test correctly leave alone? Failing here sends a healthy person on a long,
  expensive trip to hospital for nothing. *Spare the healthy.*

**A worked example.** Screen 1,000 diabetic patients, of whom 100 truly need
a specialist and 900 do not. A test with 90% sensitivity and 85% specificity:

- catches **90** of the 100 sick patients and misses **10**;
- correctly clears **765** of the 900 healthy ones, and wrongly refers **135**.

That is the problem statement's minimum bar.

### 3.3 The threshold, and why the two numbers fight

Our model does not answer "yes" or "no". It outputs a score between 0 and 1
— how likely this eye is to have referable disease. We then choose a
**threshold**: above it, we act.

Lower the threshold and you catch more disease, but you also sound more
false alarms. Raise it and the reverse happens. **Sensitivity and
specificity always trade against each other through the threshold.** Quoting
either number alone is meaningless; quoting both at a threshold you chose
*after* seeing the answers is cheating. This point matters enormously later.

### 3.4 The ROC curve and AUC

Because the threshold is a dial, a model is better described by what happens
at *every* setting of the dial. Plot sensitivity against the false-alarm rate
as the threshold sweeps from 0 to 1 and you get a **ROC curve** (Receiver
Operating Characteristic — the name is a relic of Second World War radar).

The **AUC** — area under that curve — condenses it into one number, and it
has a beautifully simple meaning:

> **Pick one sick patient and one healthy patient at random. The AUC is the
> probability that the model gives the sick one the more worrying score.**

An AUC of 1.0 is perfect ranking. 0.5 is a coin toss. AUC does not depend on
where the threshold is set, which makes it the fairest single number for
comparing two models. It says how well the model *can* separate the groups;
the threshold decides how that ability is spent.

### 3.5 Image level and exam level

A clinic visit produces two photographs, one per eye. The patient is
referred if *either* eye shows referable disease.

- **Image level** — each photograph scored on its own.
- **Exam level** — each patient visit scored as a whole.

Exam-level specificity is always harder, because a healthy patient has two
chances for a false alarm, one per eye. IDx-DR reports exam level; much of
the research literature reports image level. We report both, and we are
careful never to compare one with the other.

### 3.6 Training, validation, test — and why you only look once

A machine-learning model learns from examples. The standard discipline
splits the data three ways, and a student preparing for an exam is the right
analogy:

- **Training set** — the textbook. The model learns from it.
- **Validation set** — practice papers. Used to make decisions: when to stop
  training, which settings are best.
- **Test set** — the real exam, sealed until the end. Used **once**, to
  measure how good the final model is.

The reason the test set must stay sealed: every time you look at a score on
it and change something in response, you are quietly tuning to that exam.
Do it enough times and your final score measures how well you memorised
this particular paper, not how well you know the subject. This is the
single most common way impressive-looking AI results turn out to be false.

**Data leakage** is when test data sneaks into training by accident — the
same photograph appearing twice, say. It is the equivalent of seeing the
exam paper beforehand, and it inflates results.

### 3.7 External validation and domain shift

A model trained on photos from one camera, in one hospital, often performs
worse on photos from a different camera — different colours, sharpness,
lighting. This is **domain shift**; for us, specifically **camera shift**.
It is like a person who learnt to read one family's handwriting struggling
with a stranger's.

**External validation** means testing on data from somewhere the model has
never seen: a different hospital, a different country, a different camera.
It is the only honest test of whether a model will work in the field, and it
is exactly where real deployments fail. A result on data from the same
source as training is called **in-distribution**, and it is almost always
flattering.

### 3.8 Neural networks, CNNs and backbones

A **neural network** is a mathematical function with millions of adjustable
numbers, tuned automatically so that its output matches the right answers on
training examples. A **convolutional neural network** (**CNN**) is the kind
built for images: it scans the picture with small learned filters, first
detecting edges, then textures, then shapes, then objects — here, lesions.

The **backbone** is the main body of the network that turns an image into a
compact description. Standard backbones have names — *EfficientNet*,
*ConvNeXt* — and come **pretrained** on millions of everyday photographs, so
they already know what edges and textures look like before seeing a single
retina. We **fine-tune** them: continue training on retinal images.

### 3.9 Calibration and ensembles

A model is **calibrated** if its confidence is honest — if, among all the
photos where it says "80% likely referable", about 80% really are. Raw
networks tend to be overconfident. **Temperature scaling** is a standard fix
that softens the confidence using a held-out set.

An **ensemble** is several models whose outputs are averaged. Different
models make different mistakes; averaging cancels some of them out. It is a
committee instead of a single expert.

---

## 4. The yardstick: IDx-DR

To know whether a result is good, you need something to compare it with.
We chose the strongest honest comparison available.

**IDx-DR** (now marketed as LumineticsCore) received FDA authorisation in
April 2018 — the first AI system authorised to make a diagnosis in any field
of medicine without a clinician interpreting the result. It is the gold
standard for *autonomous* DR screening.

Crucially, its developers published its performance on **Messidor-2**, a
public set of retinal photographs from French hospitals (Abràmoff et al.,
*Investigative Ophthalmology & Visual Science*, 2016):

> **IDx-DR on Messidor-2, exam level: 96.8% sensitivity, 87.0% specificity.**

Because Messidor-2 is public, we can score our system on *exactly the same
photographs* with *exactly the same definition* of referable disease. That
makes this a genuinely like-for-like comparison, which is rare.

Two pieces of context keep the comparison honest in both directions:

- **Messidor-2 is a best case, even for IDx-DR.** In its prospective FDA
  trial in real primary-care clinics (Abràmoff et al., *npj Digital
  Medicine*, 2018), IDx-DR reached **87.2% sensitivity and 90.7%
  specificity** — about ten points lower on sensitivity than on the curated
  public set. Real clinics are harder than public datasets. The same will be
  true of us, and we have not yet been tested prospectively.
- **IDx-DR is not allowed to say "I'm not sure".** As an autonomous device,
  it must commit to an answer. Our system is designed to defer uncertain
  cases to a doctor. That is both a limitation of the comparison and, as
  section 9.4 shows, an advantage of our design.

---

## 5. The data

No patient data from any hospital was used. Everything below is public.

| Dataset | Where from | Images | How we use it |
|---|---|---|---|
| **DDR** | Hospitals across China | 13,673 | Training, validation, internal test |
| **APTOS 2019** | Aravind Eye Hospital, rural Tamil Nadu — the set attached to the problem statement | 3,662 | Training, validation, internal test |
| **EyePACS** | A US screening programme, many camera types | 35,126 | **Training only** |
| **ODIR-5K** | Chinese hospitals, mixed camera brands | 6,392 | **Model selection only — never trained on** |
| **IDRiD** | An eye clinic in Nanded, India | 516 | **Sealed external test** |
| **Messidor-2** | French hospitals | 1,748 (874 visits) | **Sealed external test** — the IDx-DR comparison |
| HRF | German university hospital | 45 | Blood-vessel model |

About **45,000 images** were used for training. Three roles need explaining.

**Why EyePACS is training-only.** Each EyePACS image was graded by one
person, with no second opinion and a known error rate. That noise costs a
little accuracy in training. But our safety thresholds are fitted on a
validation set, and fitting a *guarantee* against untrustworthy labels would
make the guarantee untrustworthy. So EyePACS is forbidden from ever reaching
the validation data, enforced in code.

**Why ODIR is held out entirely.** Section 7.4 explains this, and it turned
out to be one of the most important decisions in the project.

**Why Messidor-2 and IDRiD are sealed.** They are the exam. Messidor-2 uses
labels adjudicated by several retina specialists (Krause et al., 2018), which
makes it cleaner than anything we train on.

---

## 6. The pipeline, stage by stage

```
  fundus photograph
        │
        ▼
 ┌──────────────────┐   unusable    ┌─────────────────────────────────┐
 │ 1. QUALITY GATE  │──────────────►│ specific retake instruction      │
 └──────────────────┘               │ ("too dark: move closer")        │
        │ usable                    │ 3rd failure → in-person referral │
        ▼                           └─────────────────────────────────┘
 ┌──────────────────┐
 │ 2. PREPROCESSING │  crop to the retina, normalise lighting
 └──────────────────┘
        │
        ├──────────────────────┬─────────────────────────┐
        ▼                      ▼                         ▼
 ┌──────────────┐    ┌───────────────────┐    ┌───────────────────┐
 │ 3. GRADER    │    │ 4. LESION FINDER  │    │ 4. LANDMARKS +    │
 │ ensemble of  │    │ outlines each     │    │    VESSELS        │
 │ 3 networks   │    │ lesion, per pixel │    │ optic disc, fovea │
 └──────────────┘    └───────────────────┘    └───────────────────┘
        │                      │                         │
        │                      └──────────┬──────────────┘
        │                                 ▼
        │                      ┌───────────────────────┐
        │                      │ 5. CLINICAL RULES     │
        │                      │ counts → ICDR wording │
        │                      └───────────────────────┘
        ▼                                 │
 ┌──────────────────┐                     │
 │ 6. EXPLANATION   │ heat map (verified) │
 └──────────────────┘                     │
        │                                 │
        ▼                                 ▼
 ┌─────────────────────────────────────────────────┐
 │ 7. UNCERTAINTY + TRIAGE                          │
 │ statistical guarantee → CLEAR / REVIEW / REFER   │
 └─────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────┐
 │ 8. APP: field screen · 30-second review screen · │
 │    referral tracker · dashboard                  │
 └─────────────────────────────────────────────────┘
        │
        ▼
   9. DISTRICT SIMULATION (Python + Simulink)
```

### Stage 1 — The quality gate

A blurry, dark or badly framed photo is not a healthy patient; it is an
unanswered question. The gate combines measurable rules (brightness,
contrast, focus, how much of the retina is visible) with a small network
trained to recognise ungradable images.

When a photo fails, the operator gets a *specific* instruction — "too dark:
dim the room lights and move closer" — rather than "error". After three
failures, the patient is referred for an in-person examination, because
declaring an unreadable photo "healthy" would be the most dangerous possible
default.

> **Result:** detects ungradable photos with **AUC 0.993** — 97.9%
> sensitivity, 96.9% specificity on 2,739 held-out DDR images.

### Stage 2 — Preprocessing

Every image, from every camera, goes through one identical path:

1. **Field-of-view detection.** Find the circular retinal area and discard
   the black border, including the white-bordered photographs some DDR
   cameras produce.
2. **Crop and square.** Centre the retina in a square image.
3. **Graham normalisation.** Subtract a heavily blurred copy of the image
   from itself. This cancels slow lighting differences between cameras and
   makes small high-contrast structures — exactly what microaneurysms are —
   stand out. It is named after Ben Graham, who used it to win the 2015
   Kaggle DR competition.

There is only one preprocessing path, in both Python and MATLAB. A system
that treats borderline images differently from clean ones creates a second,
untested code path at exactly the point where accuracy matters most.

### Stage 3 — The grader

The grader is the heart of the system: it looks at the photo and outputs the
ICDR grade and the probability of referable disease.

**The ordinal head (CORN).** A naive classifier treats grades 0–4 as five
unrelated categories, as if "moderate" were no closer to "severe" than to
"healthy". That wastes information and can produce contradictions — a model
that thinks grade 3 is likely but grade 2 is not.

Instead we use a **CORN** head (Conditional Ordinal Regression for Neural
networks; Shi, Cao & Raschka, 2023). It asks a chain of questions:

> Is it worse than grade 0? · Given that, is it worse than grade 1? · …

Chaining the answers produces probabilities that are *guaranteed* to respect
the order of the scale. The probability of referable disease — P(grade ≥ 2)
— falls straight out of the chain instead of being stitched together
afterwards. We rewrote the chain as explicit multiplications because the
standard implementation cannot be exported to ONNX, the format that carries
models into MATLAB.

**The final grader is an ensemble of three networks**, each trained on all
45,000 training images:

| Member | Backbone | Resolution | Why it is in the committee |
|---|---|---|---|
| `v2s_all` | EfficientNetV2-S | 512 px | Strongest single model |
| `b0_768_all` | EfficientNet-B0 | 768 px | Higher resolution sees smaller lesions |
| `cnx_all_wd` | ConvNeXt-Tiny | 512 px | A different architecture family — makes *different* mistakes |

How that particular committee was chosen, and why the third member is there
despite being the weakest of the three alone, is one of the more interesting
findings in the project (section 8.6).

### Stage 4 — Lesions, landmarks and vessels

A grade alone is not an explanation. A doctor wants to see *what* the model
saw.

**The lesion finder** is a **U-Net** — a network that labels every single
pixel rather than the whole image — run at high resolution (1,440 pixels
across the retina) so that microaneurysms remain several pixels wide. It
outlines microaneurysms, haemorrhages, hard exudates and cotton-wool spots
separately.

Raw per-pixel output is noisy: on first measurement, only 52–62% of the
blobs it drew were real lesions. An explanation that is wrong half the time
is worse than none, so each detected blob now has to pass a confidence
threshold calibrated to keep **at least 75% precision** — three in four
outlined lesions are real.

**Landmarks.** A separate network locates the optic disc and the fovea, so
that lesions can be described by *where* they are. Exudates near the fovea
are a sign of macular oedema, the most sight-threatening complication.

**Vessels.** A third network traces the blood vessels.

> **Results:**
> - Lesions: mean **0.592** average precision across the four types on the
>   IDRiD test set. The winners of the 2018 IDRiD international challenge
>   scored 0.647.
> - Optic disc located within **13 pixels** on average, fovea within
>   **32 pixels**, on 4,288-pixel-wide images. 99% of optic discs within 100
>   pixels.
> - Vessels: **AUC 0.980** on the HRF test set.

### Stage 5 — The clinical rule engine

The rule engine turns lesion counts and positions into the language the ICDR
scale itself uses: "more than microaneurysms alone, therefore at least
moderate", "hard exudates within one disc-diameter of the fovea".

**A measured design decision.** An obvious idea — and several competing
teams have built it — is to have the rule engine and the neural network
*cross-check each other*, and escalate whenever they disagree. It sounds
safer: two opinions instead of one.

We built it and measured it, on the first version of the grader. On
Messidor-2's 1,744 photographs, of which 457 needed a specialist, letting the
rules overrule the network **missed 7** patients. Simply making the network itself more cautious, tuned to send
*exactly the same number* of patients to review, **missed 4**. At equal cost,
the cross-check is less safe. The rules can escalate a case to review, but
they never overrule the grader. They explain; they do not decide.

### Stage 6 — Explanation: the heat map, and proof it means something

**Grad-CAM** (Gradient-weighted Class Activation Mapping) produces a heat
map showing which regions of the photo most influenced the network's
decision. Almost every team in this competition will show one.

Almost none will show that it is *right*. We did. IDRiD includes 81 images
where ophthalmologists outlined every lesion by hand, and our grader never
saw them. We measured whether the heat lands on the lesions:

> - The heat map places **2.3 times** more of its weight on specialist-marked
>   lesions than chance would (95% confidence interval 2.1–2.5).
> - Its single hottest point lands on a real lesion **88%** of the time,
>   against **29%** by chance. (This measurement is called the *pointing
>   game*.)

A heat map asserted to be meaningful is decoration. A heat map measured
against expert annotation is evidence.

### Stage 7 — Uncertainty and triage: a guarantee, not a guess

This is the most technically advanced part of the system, and the part most
directly aimed at patient safety.

**The idea.** Instead of a yes/no answer, every patient is sorted into one
of three actions:

| Action | When | What happens |
|---|---|---|
| **Clear** | Confident there is no referable disease | Re-screen in 12 months |
| **Review** | Uncertain, or lesion evidence of macular oedema | A 30-second remote look by an ophthalmologist |
| **Refer** | Confident referable disease | Hospital referral, tracked to completion |

Two thresholds separate the three bands. The question is how to set them so
the result is *safe*, not merely *accurate on average*.

**Conformal risk control.** **Conformal prediction** is a branch of
statistics that turns a model's raw scores into mathematical guarantees
without needing to trust the model itself. We use a variant called
**conformal risk control** to set the "clear" threshold so that:

> **At most 3% of patients who truly need a specialist are ever
> auto-cleared.**

That guarantee is fitted once, on a held-out calibration set, and then
frozen. We also produce a **prediction set** for each patient — the range of
grades that is plausible, such as "mild to moderate" — which covers the true
grade 90% of the time on familiar data. A wide range is itself a signal:
"I'm not sure".

**The catch — and we measured it.** Conformal guarantees rest on one
assumption: *future patients resemble the calibration patients*.
Statisticians call this **exchangeability**. On a new camera, it is false.
So the guarantee should break there, and it does:

| Where | Needed a specialist but auto-cleared | 95% interval | Verdict against ≤ 3% |
|---|---|---|---|
| Held-out APTOS (familiar cameras) | 0.9% | 0.3–3.4% | holds |
| Held-out DDR (familiar cameras) | 3.3% | 2.4–4.5% | at target, within sampling error |
| IDRiD (unseen Indian camera) | 3.7% | 2.1–6.4% | borderline |
| **Messidor-2 (unseen French camera)** | **6.6%** | **4.6–9.2%** | **broken** — the whole interval is above 3% |

The guarantee is a statement about the *expected* rate, so a familiar-camera
figure of 3.3% with an interval spanning 3% is consistent with it. Messidor-2
is not: even the most optimistic end of its interval is above the target.

**The repair: per-site recalibration.** Each new site keeps a small random
audit: a sample of its auto-cleared patients is also sent for review,
producing locally graded examples. Once enough exist, that site gets its own
recalibrated threshold. (Technically, *Mondrian* conformal prediction —
separate guarantees per group.) Until then, its uncertain cases go to review
by default. On Messidor-2:

| Local audit labels | Needed a specialist but auto-cleared | Share of photos auto-cleared |
|---|---|---|
| 0 (frozen threshold) | 6.6% — guarantee broken | 59% |
| 50 | 0.0% | 0% — safe only by deciding nothing |
| 200 | 2.0% — restored | 41% |
| 300 | 2.5% — restored | 46% |

With 200–300 locally graded photos, the guarantee holds again *on average*
while the system still clears 40–46% of patients by itself. Two honest
caveats: with only 50 labels, safety is bought by refusing to auto-clear
anyone; and "on average" is the precise claim — in one random audit out of
ten, the miss rate still exceeds 3%.

The point for a judge is not that the guarantee is perfect. It is that **we
know exactly when it breaks, why, by how much, and what it costs to fix** —
which is what distinguishes an engineered safety property from a slide.

### Stage 8 — The app and the referral tracker

A web application (FastAPI) provides:

- **The field screen** for the health worker: capture, quality feedback,
  result.
- **The review screen** for the ophthalmologist: the photo, the grade, the
  plausible range, the outlined lesions, the heat map and the ICDR findings,
  with a stopwatch built in to measure the problem statement's 30-second
  requirement.
- **The referral tracker**: a referred patient who has not attended is
  escalated — patient, then family, then the village health worker
  (*ASHA*). This addresses the half of the problem most systems ignore.
- **A dashboard**, and **a reader-study mode** that times real
  ophthalmologists grading photos (section 12).
- Patient messages come from fixed templates in English, Hindi and Tamil.
  No free-text generation, so the system cannot invent medical advice.

### Stage 9 — The district simulation

The problem statement asks for a Simulink model of a district screening
100,000+ patients a year. We built a discrete-event simulation (SimPy in
Python, SimEvents in Simulink) that models camera sessions, ungradable
retakes, review queues with finite ophthalmologist capacity, referral
attendance and follow-up, using the measured triage rates from stage 7. It
answers the planner's question: *how many reviewer-hours does this district
need, and where does the queue form?*

---

## 7. How we kept the numbers honest

This section is the one we would most like a judge to read.

### 7.1 A trap inside a public dataset

DDR, one of our training sets, comes with an **official** split into
training, validation and test portions. Its images come from two distinct
cameras, identifiable by filename. In the official split, **every image from
one camera in the training and validation portions is graded healthy**.

A network trained on that split can learn "this camera's look means
healthy" instead of learning to recognise disease. Our first model did
exactly that: it scored a superb **0.987 AUC** on the official validation
set — but only about **0.92** when measured within each camera separately.
The difference was the camera, not the disease.

We re-split DDR ourselves, balancing camera and grade across every portion,
and retrained. The honest number was lower. **Any team using DDR's official
split is at risk of reporting an inflated result**, and this is verifiable
by anyone in five minutes from the filenames.

### 7.2 Duplicate photographs

Public datasets contain the same eye photographed twice, or the same image
re-saved. If one copy lands in training and the other in testing, the test
is contaminated.

Our first duplicate detector flagged 11,594 of 14,189 images — absurdly many,
because to a crude image fingerprint every retina looks like "a bright
circle on black". We rebuilt it to compare the fine vessel pattern at the
centre of the retina, verified it by eye on sample sheets, and it found
**496** genuine duplicates. Whole duplicate groups are now assigned to a
single split.

When EyePACS and ODIR were added, a check comparing the before-and-after
splits confirmed that **no image moved from a test set into training**, and
flagged 86 ODIR images that resembled training images; those are excluded
from ODIR's scores.

### 7.3 Every look at the test sets is written down

`docs/EXPERIMENTS.md` ends with a register of **every occasion** anyone
looked at IDRiD or Messidor-2 results, with the date, what was looked at and
why. There are nine entries. A reviewer can check that no decision was made
by tuning to the exam.

### 7.4 The referee: ODIR

The central difficulty: to improve a model you must compare versions, and
comparing on the sealed test sets would unseal them.

Our validation set cannot help either — it comes from the *same* cameras as
training, so it cannot see camera shift at all. We held out **ODIR** — a
third country, a mix of camera brands, never trained on — purely as a
referee for choosing between models. It is scored freely and **never quoted
as a result**.

This turned out to matter far more than expected. On in-distribution
validation, a bigger network and higher resolution looked like rounding
errors (+0.002 and +0.004 AUC). On ODIR, they were worth **+0.013** and
**+0.016** — three to four times more. **A validation set drawn from your
training cameras cannot rank changes whose whole purpose is to survive a
change of camera.** Without ODIR we would have discarded two of our best
improvements.

### 7.5 Committing before looking

For the final model, we fixed the choice on ODIR, stated in writing that we
would report whatever Messidor-2 said — good or bad — and only then looked.
In research this is called **pre-registration**. It is the difference
between reporting a result and choosing one.

### 7.6 We correct our own mistakes in writing

The experiment log records the errors, not just the successes: an earlier
over-claim that the problem-statement target was met at the exam level,
based on a threshold read off the test set rather than fixed in advance, is
recorded and corrected in place (`EXPERIMENTS.md` §14). The figures in this
briefing are the corrected ones.

---

## 8. What we did: the experimental story

### 8.1 Where we started

The first complete system used one network (EfficientNet-B0) trained on
about 11,000 images from DDR and APTOS. On Messidor-2, per patient visit, it
**matched IDx-DR's sensitivity (96.7%) but reached only 57.0% specificity**
against their 87.0%. It found the disease; it panicked about healthy people
three times as often.

### 8.2 The diagnosis: the model, not the dial

The obvious suspicion was a badly placed threshold. We tested that directly
by computing the best specificity *any* threshold could have achieved at
IDx-DR's sensitivity: **52.3%**. No setting of the dial reached their
numbers. The model itself was weaker at separating sick from healthy — a
ranking problem, not a threshold problem.

We then checked whether the comparison itself was unfair to us. IDx-DR's
definition of referable includes macular oedema, which our grader does not
predict directly. If many referrals were oedema-only, we could never match
them. Of 270 referable Messidor-2 visits, **268 were referable by grade and
only 2 by oedema alone**. No excuse there: the gap was real.

### 8.3 Three good ideas that failed

We had a list of ideas for closing the gap. We tested them before believing
them:

**Combining both eyes intelligently.** Today, the worse eye decides. We
tried four smarter ways of combining the two scores. The best improved AUC by
0.003 — and at our operating point, specificity by two tenths of a point.
**Rejected.**

**Self-tuning on a new camera from unlabelled photos.** The idea: take 50
photos from a new clinic, have nobody grade them, and reset the threshold
from the spread of scores. It is an attractive pitch line. Tested on IDRiD,
it dropped sensitivity from **92.9% to 74.6%** — it would have missed one
sick patient in four. The method silently assumes every site has the same
amount of disease; IDRiD had far more. **Rejected as a hazard.**

*Something useful fell out of that failure:* the same mathematics estimates
how much disease a site has, from 50 ungraded photos, with no clinician
time — within a point on IDRiD (63% estimated, 63% true). That is exactly the
number a district planner needs, and the simulation previously had to
assume it.

**A physically accurate simulator of cheap cameras.** Our training photos
came from good hospital cameras; the field uses cheap handheld ones. We
built software that re-photographs a good image as a cheap camera would:
ring-flash glare, cataract haze, out-of-focus optics, colour fringing, dust,
sensor noise, heavy compression — modelled in physical light units, with each
virtual camera keeping its own dust and colour cast across thousands of
patients, as a real device would. It was adapted from an automotive camera
simulator a team member built for a separate project.

It was also measured. Its own validation found two faults no visual
inspection had caught, and one real insight: **the largest measurable
difference between our training photos and every unseen camera is colour,
not blur** — now fixed with a one-line adjustment. But training with the
simulator did not improve performance on unseen cameras. **Rejected as a
training method**, kept as a stress-testing tool.

**Why negative results matter here.** Each of these would have made a good
slide. Each was wrong. A team that has not tested its ideas cannot tell which
of its claims are true, and neither can its judges.

### 8.4 The lever that worked: data

We added **35,126 photographs from EyePACS**, a US screening programme using
many different cameras. The official download required accepting
competition terms; the same images were available as a plain public
dataset, decoded straight from the archive into our pipeline so that 35,000
originals never had to be stored at once.

The effect, measured on the ODIR referee:

| | Familiar cameras | **Unseen cameras** | Gap |
|---|---|---|---|
| Before EyePACS | 0.977 | **0.888** | −0.089 |
| After EyePACS | 0.976 | **0.926** | −0.050 |

**+0.037 AUC on unseen cameras — more than every other change combined.**
Notice the familiar-camera number did not improve at all. What the extra
photos bought was *generalisation*, the one thing we were short of. The gap
between familiar and unseen cameras nearly halved.

### 8.5 Capacity and resolution — and how they overlap

| Change | On small data | On full data |
|---|---|---|
| Bigger network (EfficientNetV2-S) | +0.013 | combined with data: +0.044 total |
| Higher resolution (768 px) | +0.016 | **+0.003** |

Resolution was worth a lot on the small dataset and almost nothing once the
data was there. The two levers were partly buying the same thing — robustness
to new cameras — and data buys more of it. That result alone saves a week of
work chasing even higher resolutions.

### 8.6 How to choose an ensemble

Averaging models helped, but *which* models to average turned out to follow
a rule nobody on the team expected.

First surprise: adding two strong models trained only on DDR and APTOS to
the committee **improved** every familiar-camera score and **worsened** the
unseen-camera score. They pulled the average back towards the narrow domain
they had learnt.

Second surprise: a ConvNeXt network was the **weakest** of the full-data
models on its own, yet swapping it in for a *stronger* EfficientNet
**improved** the committee. Three EfficientNets agree with each other too
much to be worth three votes; a different architecture family makes
different mistakes.

> **The rule: pick ensemble members for how differently they err, not for
> how well they score alone.**

ConvNeXt's first run used training settings tuned for EfficientNets.
Correcting one of them (weight decay) lifted it from 0.914 to 0.921 on ODIR —
still the weakest model alone (the EfficientNet it replaced scores 0.926),
and still the member that makes the committee best. The final committee is
`v2s_all` + `b0_768_all` + `cnx_all_wd`.

### 8.7 The whole night, on the referee

| Model | Unseen-camera AUC (ODIR) |
|---|---|
| Starting point | 0.888 |
| + 35k EyePACS images | 0.926 |
| + bigger network | 0.932 |
| Ensemble of full-data models | 0.937 |
| **Final committee, diverse architectures** | **0.940** |

All of it on one laptop GPU with 8 GB of memory, in about twelve hours.

---

## 9. Final results

Final model: `ens_d3`, three EfficientNets (V2-S at 512, V2-S at 768, B0 at
768) trained on DDR + APTOS + both halves of EyePACS. Chosen on two held-out
selection sets (ODIR and DeepDRiD) by a rule written down before the last
candidate was scored, committed to in writing, then scored once on the
sealed sets. Every figure uses the threshold **fixed in advance** on our own
calibration data, unless explicitly marked otherwise. Ranges in brackets are
95% confidence intervals.

### 9.1 Detecting referable DR

| Test set | What it is | AUC | Sensitivity | Specificity | PS target (>90 / >85)? |
|---|---|---|---|---|---|
| DDR test | Familiar cameras, held out | 0.986 | 95.1% | 93.4% | ✅ |
| APTOS test | Familiar cameras, held out | 0.986 | 96.2% | 91.9% | ✅ |
| **Messidor-2, per photo** | **Unseen French camera** | **0.982** | **95.2%** (92.8–96.8) | **89.6%** (87.8–91.1) | ✅ |
| **Messidor-2, per visit** | Unseen, IDx-DR's measure | **0.982** | **96.3%** (93.3–98.0) | **87.4%** (84.5–89.8) | ✅ |
| **IDRiD, per photo** | Unseen Indian camera | **0.976** | **93.2%** (89.9–95.5) | **87.1%** (81.6–91.1) | ✅ (point estimate) |

**Reading this honestly.** On Messidor-2 both numbers clear the target with
their whole intervals, per photograph. On IDRiD both point estimates clear
it; the intervals are wide because the set is small (516 photos), so a
stricter reading calls IDRiD "met, probably".

### 9.2 Against IDx-DR

| | IDx-DR (Abramoff 2016) | DrishtiTrust `ens_d3` |
|---|---|---|
| Sensitivity, per visit | 96.8% (93.3–98.8) | 96.3% (93.3–98.0) |
| Specificity, per visit | 87.0% (84.2–89.4) | 87.4% (84.5–89.8) |
| AUC, per visit | 0.980 (0.968–0.992) | 0.982 (0.974–0.988) |

Read off our curve (curve comparisons, not thresholds we could have set in
advance): at IDx-DR's 96.8% sensitivity our specificity is 86.9%; at its
87.0% specificity our sensitivity is 96.7%.

**The two systems are statistically indistinguishable on these
photographs.** Our operating point was fixed on our own calibration data
before Messidor-2 was scored, so this is not a tuned result. What we do not
claim: that we beat it, or that we would match it prospectively in clinics,
where IDx-DR's own sensitivity dropped by about ten points.

### 9.3 Progress, on the sealed set

| Model | Date | Messidor-2 AUC per visit | Specificity at 96.8% sensitivity |
|---|---|---|---|
| First honest model (`b0_final`) | 19 Sep | 0.939 | 52.3% |
| Laptop final (`ens_vrc3b`) | 21 Sep | 0.959 | 64.9% |
| Properly trained (`ens_e3`) | 24 Sep | 0.965 | 69.5% |
| **+ EyePACS test half (`ens_d3`)** | **24 Sep** | **0.982** | **86.9%** |

### 9.4 The deferral advantage

On Messidor-2 photographs, the frozen triage policy sorts patients as
follows:

| | All photos | Of those needing a specialist | Of healthy photos |
|---|---|---|---|
| Cleared automatically | 57.6% | 1.1% | 77.7% |
| Sent for 30-second review | 22.4% | 26.0% | 21.1% |
| Referred automatically | 20.0% | 72.9% | 1.2% |

So the system decides **about 78% of photographs by itself**. On those, it
catches **98.5%** of disease and correctly clears **98.5%** of healthy eyes.
The other 22% reach a doctor, who looks for 30 seconds instead of a patient
travelling for a day. The share of referable photographs auto-cleared (1.1%,
5 of 457) is inside the 3% conformal guarantee on this unseen camera; with
the previous model it was 6.6%. The "doctor resolves the review cases
correctly" part is an assumption until the reader study measures it.

### 9.5 The other modules

| Module | Result | Reference point |
|---|---|---|
| Ungradable-photo detection | AUC 0.993 | — |
| Lesion segmentation (4 types) | mean average precision 0.592 | 2018 IDRiD challenge winners: 0.647 |
| Optic disc / fovea | 13 px / 32 px mean error on 4,288-px images | — |
| Blood vessels | AUC 0.980 | — |
| Heat map vs expert markings | 2.3× chance; hottest point on a lesion 88% (chance 29%) | Rarely measured at all |

(The heat-map figures were measured on the earlier single model `b0_final`.)

---

## 10. How impressive is this, really?

Honest calibration, in both directions.

### 10.1 What is genuinely strong

- **External validation with confidence intervals, against a named
  commercial benchmark on identical photographs.** Most competition entries
  report one number on data from the same source they trained on. Our
  familiar-camera numbers (0.985–0.986 AUC) would look as good as anyone's.
  We lead with the harder number instead, because it is the one that predicts
  field performance.
- **The problem statement's accuracy target is met on two unseen cameras,**
  with a threshold fixed in advance, and per patient visit the result ties
  IDx-DR's published operating point on IDx-DR's own test photographs.
- **A dataset flaw that likely inflates other teams' results**, discovered,
  documented and verifiable in minutes.
- **An explainability claim backed by measurement** against specialists'
  annotations.
- **A safety guarantee whose failure mode is measured and repaired**, rather
  than asserted.
- **Measured negative results.** Three good ideas killed by evidence is
  the strongest possible signal that the positive results are real too.
- **Complete scope**: all five problem-statement modules, an app, a
  referral tracker, a district simulation and a MATLAB path.

### 10.2 How it compares with published research

The landmark study in this field (Gulshan et al., *JAMA*, 2016, Google)
reached AUC **0.990** on Messidor-2. It used about **128,000** training
images, each graded by **three to seven** ophthalmologists who had to agree.

We reached **0.982** per photograph with about **99,000** images, each
graded by **one** person. The remaining gap is mostly label quality, which
cannot be fixed with public data (section 12.1).

### 10.3 What it is not

- It does not beat IDx-DR; it ties it, on one public set of photographs.
- It has not been tested prospectively on patients, which is where even
  IDx-DR's sensitivity fell by about ten points.
- It has not been tested on a real Indian handheld camera (section 11).
- It is a research prototype, not a medical device.

### 10.4 The honest summary

Judged as a hackathon system: unusually rigorous and unusually complete,
with accuracy that clears the brief on two unseen cameras. Judged against a
certified commercial device on the same retrospective photographs: level,
within the uncertainty. Judged as a medical device: untested where it
matters most, on Indian handheld cameras and with real reviewers.

---

## 11. Limitations

1. **No test on real Indian field hardware.** The one public handheld
   dataset (mBRSET, Brazil) requires credentialed access we have not yet
   obtained. Every "unseen camera" result here is on desktop hospital
   cameras.
2. **Single-grader training labels.** EyePACS, DDR and APTOS are graded by
   one reader per image. This caps achievable accuracy.
3. **Retrospective only.** Every figure is on archived photographs; IDx-DR's
   own sensitivity fell by about ten points in its prospective trial.
4. **The safety guarantee needs local calibration** on every new camera —
   200–300 locally graded photos — before it can be relied on, even though it
   held on Messidor-2.
5. **The 30-second review claim is unmeasured.** The timing tool is built;
   the reader study has not yet happened.
6. **The MATLAB and Simulink ports have not been executed** — no build
   machine has had a MATLAB licence. The exported networks are checked
   against PyTorch and use only built-in MATLAB layers, and the MATLAB
   ensemble recipe is replayed in Python against the final predictions;
   `matlab/check_export.m` is the first thing to run on a licensed machine.
7. **Neovascularisation is not segmented**; the only public pixel labels
   (FGADR, 49 images) are available on request only.
8. **Referral-attendance figures in the simulation are assumptions**, not
   measurements.
9. **The heat-map validation was done on an earlier single model** and has
   not been repeated on `ens_d3`.

---

## 12. How much further it can go, and the plan

### 12.1 Where the ceiling is

Five days ago this section estimated that matching IDx-DR needed an AUC of
about 0.98 and that public data would plausibly stop near 0.97. The final
model reached 0.982 on Messidor-2. The lever was not computation or a
cleverer model: it was the second half of EyePACS, 53,572 more photographs
from many more cameras. Label quality is still the ceiling above that ---
our teachers are single graders, IDx-DR's and Gulshan et al.'s were panels
of specialists --- and no amount of computation removes it.

### 12.2 What more computation bought

Moving from an 8 GB laptop GPU to a 32 GB workstation let every network
train to convergence (10–15 passes instead of 3–7), which was worth about
+0.005–0.008 AUC per network on the selection sets, and made RETFound
fine-tuning possible. RETFound, the retinal foundation model, then lost to a
properly trained EfficientNet on both selection sets and was rejected
(`docs/EXPERIMENTS.md` §16).

### 12.3 More data

| Dataset | Size | Why it matters | Access |
|---|---|---|---|
| **mBRSET** | 5,164 | **The only public handheld/smartphone DR set** — tests our actual weakness | PhysioNet credentialed access |
| BRSET | 16,266 | Brazilian hospital cameras, more diversity | Same PhysioNet credentialing |
| FGADR | 2,842 | Pixel-level lesion annotations, including neovascularisation | Request to authors |

### 12.4 The plan, in priority order

1. **Run the MATLAB path on a licensed machine** (`check_export.m`, then
   `run_pipeline.m` and the Simulink model), and fix whatever breaks.
2. **Start PhysioNet credentialing** and validate on mBRSET: the first real
   handheld-camera number.
3. **Run the reader study** with a practising ophthalmologist.
4. **Repeat the heat-map validation** on the final ensemble.

### 12.5 The things no amount of computation can buy

The most valuable remaining steps are not technical:

- **A reader study.** Forty public-dataset photographs, shown to a practising
  ophthalmologist in our review screen, with every decision timed. It is the
  only way to evidence the problem statement's 30-second requirement, and it
  needs about 25 minutes of one specialist's time. It uses only public
  images, so it involves no patient data and no ethics approval. The protocol
  is in `docs/READER_STUDY.md`, and the app's `/study` mode records it.
- **Real field images from an Indian eye hospital**, through that hospital's
  ethics committee. Slow, but even a few hundred handheld photographs would
  give the only genuinely field-relevant number in the competition.
- **Referral-attendance data** from a real programme, to replace the
  simulation's assumptions with measurements.

No competing team can generate these by asking an AI assistant. They are
the part of this project that is genuinely difficult to copy.

---

## 13. Appendix

### 13.1 Glossary

| Term | Meaning |
|---|---|
| **AUC** | Probability the model ranks a random sick patient above a random healthy one. 1.0 perfect, 0.5 chance. |
| **Calibration** | Whether stated confidence matches reality ("80% sure" is right 80% of the time). |
| **CNN** | Convolutional neural network — the standard architecture for images. |
| **Conformal prediction / risk control** | Statistical methods that turn model scores into guarantees, valid when future data resembles calibration data. |
| **CORN** | An output design that respects the order of the 0–4 grade scale. |
| **DME** | Diabetic macular oedema — swelling at the centre of vision. |
| **Domain / camera shift** | Worse performance on data from a camera or site unlike the training data. |
| **DR** | Diabetic retinopathy. |
| **Ensemble** | Several models averaged, to cancel individual mistakes. |
| **Exam level** | Scored per patient visit (two eyes), not per photograph. |
| **Exchangeability** | The assumption, underlying conformal guarantees, that future cases resemble past ones. |
| **External validation** | Testing on data from a site the model has never seen. |
| **Fovea** | The centre of the macula; responsible for the sharpest vision. |
| **Fundus photograph** | A photograph of the back of the eye. |
| **Graham normalisation** | Subtracting a blurred copy of an image to remove lighting differences. |
| **Grad-CAM** | A heat map showing which image regions drove a decision. |
| **ICDR scale** | The international five-point DR grading scale (0–4). |
| **In-distribution** | Data from the same sources as training. Usually flattering. |
| **Leakage** | Test data accidentally present in training. Inflates results. |
| **Mondrian conformal** | Separate conformal guarantees per group — here, per camera or site. |
| **Pointing game** | How often a heat map's hottest point lands on a real lesion. |
| **Prediction set** | The range of grades judged plausible, e.g. "mild to moderate". |
| **Pre-registration** | Committing to what will be reported before seeing the result. |
| **Referable DR** | ICDR grade 2 or worse — needs a specialist. |
| **RETFound** | A retinal foundation model pretrained on ~1.6 million images. |
| **ROC curve** | Sensitivity against false-alarm rate at every threshold. |
| **Sensitivity** | Share of sick patients caught. |
| **Specificity** | Share of healthy patients correctly left alone. |
| **Temperature scaling** | A standard correction for overconfident networks. |
| **Threshold** | The score above which the system acts. |
| **U-Net** | A network that labels every pixel of an image. |

### 13.2 Where things are

```
drishtitrust/
├── README.md                  headline results and how to run everything
├── docs/
│   ├── TECHNICAL_BRIEFING.md  this document
│   ├── EXPLAINER.md           plain-language project summary
│   ├── EXPERIMENTS.md         every experiment, result, and look at test data
│   ├── IDEAS.md               the improvement plan, annotated with outcomes
│   ├── DATASETS.md            data sources, splits and leakage controls
│   ├── LIMITATIONS.md
│   └── READER_STUDY.md        protocol for timing real ophthalmologists
├── results/VALIDATION.md      full results with confidence intervals,
│                              generated from raw result files — none typed by hand
├── drishti/                   the pipeline
│   ├── quality/  preprocess.py  grading/  lesions/  landmarks/  vessels/
│   ├── rules/  explain/  conformal/  eval/
│   └── augment/               the cheap-camera simulator
├── app/                       web app: field, review, referrals, dashboard, study
├── sim/                       district simulation (Python)
├── matlab/                    MATLAB and Simulink ports
├── scripts/                   data ingest, training queue, ensembling, reports
└── tests/                     28 automated tests
```

### 13.3 Key references

- Abràmoff MD et al. *Improved automated detection of diabetic retinopathy
  on a publicly available dataset through integration of deep learning.*
  Investigative Ophthalmology & Visual Science, 2016.
- Abràmoff MD et al. *Pivotal trial of an autonomous AI-based diagnostic
  system for detection of diabetic retinopathy in primary care offices.*
  npj Digital Medicine, 2018.
- Gulshan V et al. *Development and validation of a deep learning algorithm
  for detection of diabetic retinopathy in retinal fundus photographs.*
  JAMA, 2016.
- Krause J et al. *Grader variability and the importance of reference
  standards for evaluating machine learning models for diabetic
  retinopathy.* Ophthalmology, 2018.
- Zhou Y et al. *A foundation model for generalizable disease detection from
  retinal images.* Nature, 2023.
- Shi X, Cao W, Raschka S. *Deep neural networks for rank-consistent ordinal
  regression based on conditional probabilities.* Pattern Analysis and
  Applications, 2023.
- Anjana RM et al. *Metabolic non-communicable disease health report of
  India: the ICMR-INDIAB national cross-sectional study (ICMR-INDIAB-17).*
  Lancet Diabetes & Endocrinology, 2023.
