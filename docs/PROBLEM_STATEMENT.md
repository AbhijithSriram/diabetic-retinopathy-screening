# SIH26038 — the problem statement, and where we stand against it

The brief verbatim, then an honest compliance audit. Kept in the repository
because every other document here describes what we built; this one is the
only record of what we were *asked* to build.

---

## The brief

**Title:** Explainable AI for Diabetic Retinopathy Screening in Rural India
**Organisation:** MathWorks

### Background

> India has over 77 million diabetic adults — the second highest globally.
> Diabetic Retinopathy (DR) affects ~18% of this population and is a leading
> cause of preventable blindness. Early screening can prevent 90% of vision
> loss, but India has only ~1 ophthalmologist per 100,000 rural population,
> making mass manual screening infeasible. Existing AI solutions function as
> black boxes, lack clinical validation rigor, and fail with variable image
> quality from portable fundus cameras in field conditions. A robust,
> explainable, and validated screening system is essential for deployment in
> primary healthcare centres across rural India.

### Description

> Design a MATLAB-based retinal image analysis pipeline for automated DR
> screening addressing real-world deployment challenges:
>
> 1. **Image Quality Assessment and Enhancement:** Automatically evaluate
>    fundus images for adequacy (focus, illumination, field of view). Apply
>    adaptive enhancement (CLAHE, illumination normalization, denoising) for
>    borderline images; reject ungradeable ones with recapture feedback.
> 2. **Retinal Structure Segmentation:** Extract clinically relevant
>    structures — optic disc/fovea localization, vessel segmentation,
>    microaneurysm detection, exudate segmentation, hemorrhage
>    classification, and neovascularization detection.
> 3. **DR Severity Grading:** Classify using the International Clinical DR
>    severity scale (Levels 0-4) with clinically acceptable sensitivity
>    (>90%) and specificity (>85%) for referable DR (Level 2+).
> 4. **Explainability Module:** Implement Grad-CAM attention maps,
>    lesion-level evidence correlated with clinical criteria, calibrated
>    confidence scores, and automated annotated reports — enabling
>    ophthalmologist validation in under 30 seconds for a human-in-the-loop
>    workflow.
> 5. **Simulink Workflow Simulation:** Model the telemedicine screening
>    pipeline in Simulink — image acquisition rates, bandwidth constraints,
>    processing throughput, and review capacity — to optimize resource
>    allocation for district-level programs serving 100,000+ patients
>    annually.
>
> This problem demands clinical validation rigor, sub-pixel microaneurysm
> detection, and clinically meaningful explainability.
>
> **Tools:** Image Processing Toolbox, Computer Vision Toolbox, Deep Learning
> Toolbox, Medical Imaging Toolbox, Simulink, Statistics and Machine Learning
> Toolbox
>
> **Expected Solution:** A working prototype demonstrating: DR classification
> with >90% sensitivity and >85% specificity for referable DR; explainable
> Grad-CAM outputs rated as clinically useful; a Simulink model optimizing
> screening resource allocation; and validation against published benchmarks
> showing the integrated pipeline outperforms any single technique approach.

---

## Compliance audit (25 September 2026)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Adequacy: focus, illumination, field of view | met | quality gate, AUC 0.993 |
| 1 | Reject ungradeable with recapture feedback | met | specific instructions; 3rd failure to in-person referral |
| 1 | **Adaptive enhancement: CLAHE, denoising, for borderline images** | **NOT MET** | illumination normalisation (Graham) is applied to every image; there is no CLAHE, no denoising and no borderline branch |
| 2 | Optic disc / fovea localisation | met | 13 px / 32 px mean error on 4288-px images |
| 2 | Vessel segmentation | met | AUC 0.980 on HRF |
| 2 | Microaneurysm detection, sub-pixel | met | intensity-weighted centroids at 1440 px |
| 2 | Exudate segmentation, haemorrhage classification | met | 4-class lesion U-Net, 0.592 mean AP |
| 2 | **Neovascularization detection** | **NOT MET** | not segmented. `rules/icdr.py` states it. Only public pixel labels are FGADR's 49 masks, available on request. Proliferative DR is still caught at the grading level (ICDR 4) and always escalated |
| 3 | ICDR 0-4 grading | met | CORN ordinal head |
| 3 | >90% sensitivity, >85% specificity for referable | **met on every set** | Messidor-2 95.2/89.6, IDRiD 93.2/87.1, DDR 95.1/93.4, APTOS 96.2/91.9 — threshold fixed in advance |
| 4 | Grad-CAM attention maps | exceeded | validated against expert masks: 2.3x chance energy, 88% pointing game |
| 4 | Lesion evidence correlated with clinical criteria | met | ICDR rule engine with a written trace |
| 4 | Calibrated confidence scores | met | temperature scaling, conformal prediction sets, ECE reported |
| 4 | Automated annotated reports | met | review screen and report module |
| 4 | **Validation in under 30 seconds** | **UNMEASURED** | timing built into the app's `/study` mode; the reader study has not been run |
| 5 | Acquisition rates, bandwidth, throughput, review capacity | met | SimEvents model, verified R2026b |
| 5 | 100,000+ patients annually | met | 2 reviewers at 2 h/day, 24% utilised |
| 5 | **Optimise resource allocation** | partial | parameters can be swept (`district_sim`, `sim/district.py --sweep`); no sweep has been run and presented |
| E | Validation against published benchmarks | met | matches IDx-DR on Messidor-2: 96.3/87.4 against 96.8/87.0 |
| E | **Integrated pipeline outperforms any single technique** | **DATA EXISTS, NOT FRAMED** | see below |
| E | Grad-CAM **rated** as clinically useful | **UNMEASURED** | needs the reader study |

### The four gaps, in priority order

1. **Reader study.** Blocks two explicit requirements (30-second validation,
   Grad-CAM "rated as clinically useful"). Needs an ophthalmologist for about
   25 minutes; protocol in `READER_STUDY.md`, set in `app/study_set.csv`,
   photographs are public so no ethics approval is involved. Longest lead
   time of anything remaining — book it first.
2. **CLAHE / denoising branch** for borderline images. Half a day. Apply to
   images the quality gate flags borderline and *measure* whether grading
   improves. A neutral result is still reportable and the box is named
   explicitly in the brief.
3. **"Outperforms any single technique."** The evidence exists and is not
   assembled. Note that one part of it cuts against us and should be
   presented anyway: on Messidor-2 the lesion-rule integration missed 7
   referable patients where simply widening the CNN's own uncertainty band to
   the same review workload missed 4. The ensemble *is* a multi-technique
   result that does win. Lead with the negative; it makes the rest credible.
4. **Neovascularisation.** Cannot be closed without pixel labels. Request
   FGADR; until then state the limitation and show grade-4 detection as the
   mitigation.

### Note on statistics

The brief's own figures are **77 million** diabetic adults and **~18%** DR
prevalence. `TECHNICAL_BRIEFING.md` cites 101 million from ICMR-INDIAB 2023.
Both are defensible — 77M is the older IDF figure — but **quote the brief's
numbers in any submission.** Arguing with the premise gains nothing.
