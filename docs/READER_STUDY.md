# Reader study: can an ophthalmologist validate an AI-screened case in under 30 seconds?

A one-page protocol to share with a partner eye hospital.

## Why

Problem statement SIH26038 asks for reports that let "ophthalmologist
validation [happen] in under 30 seconds for a human-in-the-loop workflow".
No dataset can show that; only real ophthalmologists using the real screen can.

## What the reader does

- Opens the DrishtiTrust review screen on a laptop (we bring it; nothing is installed).
- Sees 40 retina photographs, one at a time, in random order. Each comes with
  the AI's grade probabilities, the plausible grade range, the lesion outlines,
  the Grad-CAM heat map and a short list of ICDR findings.
- For each photo, presses a key for the ICDR grade (0–4), ticks "macular oedema"
  if present, or marks the photo ungradable.
- Total time: about 20–25 minutes.

## What is recorded

- The time from the photo appearing to the decision (milliseconds).
- The grade chosen.
- Nothing about the reader except the initials they type.

## Images

Only **public research datasets** are used: Messidor-2 (France), which comes
with adjudicated grades from three retina specialists, or IDRiD (India). **No
patient data from the hospital is used or collected**, so no ethics approval or
data-sharing agreement is needed for this step.

## What we report

- Median and 90th-percentile seconds per case; share decided in under 30 s.
- Agreement with the reference grades (quadratic weighted kappa); sensitivity and
  specificity for referable DR, with 95% confidence intervals.
- Optional second arm: the same reader grades 20 other photos with the AI panel
  hidden, which measures whether the AI summary speeds reading up or slows it down.

## What we would like to ask for, beyond the study

- The realistic figures behind our district simulation: patients per camp day,
  grader turnaround, and what share of referred patients reach the hospital.
- A check of the Tamil and Hindi patient messages by someone who trains field staff.

## Running it

`python -m uvicorn app.main:app --port 8026`, then open `http://localhost:8026/study`.
Run `python scripts/make_study_set.py` once beforehand to choose the 60
candidate photos (grade-stratified). Results appear at
`/study-results/<session>` and in `app/drishti.db`.
