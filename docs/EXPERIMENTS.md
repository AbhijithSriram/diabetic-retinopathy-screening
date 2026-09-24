# Experiment log

What was tried, what it measured, and what it cost. Negative results are
kept: they are the reason we are not spending the remaining weeks on ideas
that sound good and do nothing, and they are the part of this repository
another team cannot get by asking an assistant for suggestions.

Test-set discipline: everything is fitted on development data, and candidates
are ranked on ODIR, which is held out from training and never quoted as a
result. Every look at IDRiD or Messidor-2 is recorded in the register at the
bottom of this file — a count is not repeated here, because a count in two
places goes stale in one of them. No threshold, no architecture, no
augmentation setting and no ensemble has been chosen by its score on either
sealed set.

---

## 20 September 2026

### 1. Both-eye fusion — no gain at our operating point. Rejected.

A visit is referred if either eye scores high, which is a `max` over two
eyes and doubles the chance of a false alarm. The obvious fix is to combine
the two scores properly instead. Exam-level results on Messidor-2's 874
visits:

| Rule | Exam AUC | Specificity at 96.7% sensitivity | Specificity at 90% sensitivity |
|---|---|---|---|
| max (what we do today) | 0.9394 | 52.3% | 75.0% |
| mean of probabilities | 0.9386 | 53.3% | 79.1% |
| mean of log-odds | 0.9392 | 50.3% | 74.3% |
| noisy-OR | 0.9420 | 52.5% | 78.6% |

Noisy-OR is the best rule and it buys 0.003 AUC. At the sensitivity we
actually operate at, every rule lands within a point of the others. The
several free points of specificity this was expected to yield exist only in
the middle of the curve, at around 90% sensitivity, where we do not sit.

Worth knowing separately: there is no development data to fit a two-eye
combiner on. DDR and APTOS are single images with no patient identifiers, so
Messidor-2 is the only set with eye pairs, and fitting a combiner on it would
be tuning on the test set. A parameter-free rule is the only honest option
here regardless of what it buys.

### 2. Unsupervised threshold transfer — unsafe. Rejected.

The pitch was to be able to say *"a new camera needs 50 unlabelled photos,
not 200 graded ones"*. Two ways to move the threshold using unlabelled
images from the new camera:

**Rank transfer.** Our frozen threshold sits at the 51st percentile of
development scores; put the new camera's threshold at its own 51st
percentile.

| | Sensitivity | Specificity |
|---|---|---|
| IDRiD, frozen threshold | 92.9% | 73.6% |
| IDRiD, rank transfer | **74.6%** | 94.3% |
| Messidor-2, frozen threshold | 94.3% | 69.5% |
| Messidor-2, rank transfer | 94.7% | 67.4% |

It collapses sensitivity on IDRiD, because rank transfer silently assumes the
new site has the same prevalence of disease as our development data. IDRiD's
is 62.6%, ours is 43.9%, so matching ranks pushes the threshold far too high.
In a screening tool, an unsupervised method that can quietly drop sensitivity
by 18 points is not a method, it is a hazard.

**Distribution-matched transfer.** Model the target's score histogram as a
mixture of our own two class-conditional score distributions under an
unknown affine shift in log-odds and an unknown prevalence; fit the three
parameters to the unlabelled scores; move the threshold by the fitted shift.
This is prevalence-aware, and it is safe — but it only slides along the same
ROC curve (IDRiD 94.4%/71.0%, Messidor-2 96.3%/56.9%). No free specificity,
because the fitted scale is below 1: the new camera's scores are *compressed*,
which is what worse discrimination looks like, not a shifted threshold.

### 3. Prevalence estimation from unlabelled photos — works. Keep.

The mixture fit above is a poor threshold-setter and a good prevalence
estimator, which is worth more than what it was built for:

| Site | True referable prevalence | Estimated, 50 unlabelled photos | All photos |
|---|---|---|---|
| IDRiD | 0.626 | 0.631 | 0.635 |
| Messidor-2 | 0.262 | 0.321 | 0.309 |

No labels, no grading, no clinician time. A district can be told roughly how
much disease it has before anyone reads a single image, which is exactly the
number the Simulink resource model needs and currently assumes.

### 4. Capture simulator — built, calibrated, honest about what it does not do.

`drishti/augment/capture.py`: illumination, ocular media, optics, sensor and
ISP, in that order, in linear light, with cameras as persistent objects
rather than per-image random draws. Ported in spirit from the automotive
display-capture simulator in `Desktop/Visteon/OCV/capture_simulator.py`.

`scripts/validate_simulator.py` measures seven capture statistics and asks
whether simulation moves our training pool towards cameras we have never
seen. Medians after calibration:

| Statistic | dev real | dev simulated | IDRiD | Messidor-2 |
|---|---|---|---|---|
| sharpness | 0.446 | 0.440 | 0.528 | 0.378 |
| noise | 0.018 | 0.025 | 0.033 | 0.016 |
| vignetting | 0.844 | 0.715 | 0.724 | 0.902 |
| illumination CV | 0.251 | 0.297 | 0.255 | 0.195 |
| red ratio | 1.759 | 1.825 | **2.083** | **2.146** |
| blue ratio | 0.416 | 0.422 | **0.259** | 0.336 |
| blockiness | 1.099 | 1.386 | 1.015 | 1.171 |

The blur, noise and vignetting statistics land where they should. Two things
the measurement says that guesswork would not have:

- **The largest measurable difference between our training data and both
  external sets is colour, not blur.** Both external sets are markedly
  redder than anything in DDR or APTOS. That is a one-line augmentation, and
  it is now in `colour_augment()`.
- **Simulation does not make the training pool statistically closer to the
  external sets.** A classifier separating external images from the training
  pool on these seven statistics scores 0.909 on IDRiD with real data alone
  and 0.932 with simulated images added; Messidor-2 is unchanged at 0.922.
  Coverage of the external range is ~0.95 either way, because development
  data is already wide in each statistic taken alone.

So the simulator is not justified by these proxies. Whether it helps has to
be settled downstream, by training with and without it — which is the only
test that was ever going to count. Two calibration faults the contact sheet
caught and the statistics did not: dust specks sharp enough to survive
Graham normalisation as fixed dots a network could key on, and a phone-lens
profile degraded past the point where the image's own grade is still
visible, which is label noise wearing an augmentation's clothes. Both fixed.

Cost: 85 ms an image, so it is applied to a share of each epoch
(`--capture-p`), not all of it.

### 5. Is macular oedema a structural ceiling? No. Hypothesis closed.

IDx-DR's referable definition is "moderate NPDR or worse **and/or** macular
oedema". Our grader predicts the ICDR grade only; the oedema signal comes
from the lesion rule engine, which we have already shown should not overrule
it. If a meaningful share of Messidor-2's referable exams were referable
*only* through oedema, a grade-only model could never reach IDx-DR's
sensitivity and the whole comparison would be unfair to us.

Of 874 exams, 270 are referable: 268 by grade and **2** by oedema alone. The
ceiling on sensitivity for a grade-only model is therefore 99.3%, well above
the 96.8% we are chasing. Oedema is not the problem, and the gap is
discrimination on grade.

(This inspects the distribution of Messidor-2's labels, not our scores
against them. It is recorded below for completeness, but it cannot have
tuned anything: no model output was involved.)

### 6. Datasets added: EyePACS and ODIR-5K

EyePACS: 35,126 images, all matched to their labels, 17,563 patients with
both eyes. Referable prevalence 19.6%, against 44% in DDR and APTOS --- so it
also moves our training prevalence towards what a screening programme
actually sees. Training only; see `DATASETS.md` for why it may never reach
`valB`.

ODIR-5K: held out whole as an internal cross-camera validation set, which is
what lets candidate models be compared without spending looks at IDRiD or
Messidor-2.

### 7. Resolution at 768 px: the test was confounded, and the run peaked at epoch 1

`b0_768` was warm-started from `b0_final`, a 512-px model, to save GPU time.
Its best checkpoint is **epoch 1 of 9** (valA AUC 0.9797, against 0.9765 for
its own starting point and 0.9796 for a larger backbone at 512). Every
later epoch was worse.

Two things follow, and the second matters more than the first.

The cheap reading is that resolution buys about as much as capacity does,
and not much: +0.003 either way, in distribution.

The honest reading is that **this is not a clean test of resolution.** A
model warm-started from a 512-px solution and converging in one epoch has
been nudged, not retrained; it may be sitting in the basin its parent found
rather than in one that uses the extra detail. The result is evidence about
*this recipe*, not about 768 px. A clean test trains from ImageNet weights
at 768 and costs about 90 minutes.

That it peaked at epoch 1 is also a straightforward waste: eight of nine
epochs went to overfitting. Warm-started runs at a new resolution should be
given 2-3 epochs, not 9.

### 8. The data lever, measured. This is the finding of the day.

Five candidates ranked on ODIR --- held out whole, a different country and a
different mix of camera brands, never trained on.

**Adding EyePACS is worth +0.037 AUC on a camera the model has never seen.**
For scale, everything else measured today moved the in-distribution number by
about 0.003, and two ideas moved it by nothing at all.

The shape of the result matters as much as the size. `b0_all` is *not* better
in distribution --- on valA it is marginally worse (0.9760 against 0.9769).
What the extra images bought is generalisation, which is exactly what we
lacked:

| | in-distribution | unseen camera | gap |
|---|---|---|---|
| `b0_final` | 0.9769 | 0.8882 | -0.089 |
| `b0_all` | 0.9760 | 0.9255 | -0.050 |

The generalisation gap nearly halves. This retires the question the whole
brainstorm was circling: the problem was never a missing idea, a combination
rule or a threshold. It was that the model had seen ~11,000 images from two
sources. Both rejected ideas were attempts to get for free what only data
buys.

Two cautions before anyone quotes it. ODIR's labels are read off free-text
findings and are noisier than ours, so 0.9255 ranks models and does not belong
on a slide. And whether this transfers to Messidor-2 is unknown until a final
candidate is scored there once; proportional transfer is an assumption.

### 9. The full cross-camera ranking, and what it overturns

| Run | What is different | valA | DDR test | **ODIR (unseen cameras)** | vs baseline |
|---|---|---|---|---|---|
| `b0_sim` | + capture simulator | 0.9769 | 0.9781 | 0.8843 | **-0.004** |
| `b0_final` | the model we had | 0.9769 | 0.9786 | 0.8882 | --- |
| `v2s_512` | bigger backbone | 0.9792 | 0.9831 | 0.9007 | +0.013 |
| `b0_768` | 768 px | 0.9809 | 0.9784 | 0.9042 | +0.016 |
| `b0_all` | + 35,126 EyePACS images | 0.9760 | 0.9803 | 0.9255 | **+0.037** |
| `ens2` | b0_all + v2s_512 | 0.9811 | 0.9857 | 0.9255 | +0.037 |
| `ens3` | b0_all + b0_768 + v2s_512 | **0.9838** | **0.9861** | **0.9278** | **+0.040** |

**The capture simulator is rejected.** 0.8843 against 0.8882: no benefit, and
the difference sits inside run-to-run noise, so the honest verdict is "no
measurable gain", not "harmful". This was the largest engineering investment
of the day, and its own validation had already warned us --- simulated images
did not move the training pool closer to the external sets on measured capture
statistics (section 4), and the downstream test agrees. The simulator stays in
the repository as an artefact for stress-testing and demonstration. It does
not get to claim an accuracy improvement.

**In-distribution measurement systematically under-reports what helps
generalisation, and today it nearly cost us two good levers.** Backbone and
resolution both looked like rounding errors on valA (+0.002, +0.004) and were
described that way at the time. On an unseen camera they are worth +0.013 and
+0.016: three to four times more. The inference in section 7, that resolution
"disappointed", was drawn from the in-distribution number and is withdrawn ---
768 px is the second-best single lever measured. The general lesson is that a
validation set drawn from the training cameras cannot rank interventions whose
whole purpose is to survive a change of camera. ODIR should have existed from
the start.

**Ensembling is a cheap win in distribution and a thin one across cameras.**
`ens3` is best on every column, but note where the gain lands: +0.008 on valA
over its best member, +0.002 on ODIR. Averaging models cancels variance; it
does not cancel a domain shift all three members share. Free, worth taking,
not a substitute for data.

Current best candidate: `ens3`. No Messidor-2 or IDRiD number was consulted to
reach that ordering.

### 10. Combining the levers, and a rule for what belongs in an ensemble

`v2s_all` --- the backbone that won on capacity, trained on all 46,000 images:

| Run | Trained on | valA | **ODIR (unseen cameras)** |
|---|---|---|---|
| `b0_final` | DDR + APTOS | 0.9769 | 0.8882 |
| `v2s_512` | DDR + APTOS, bigger backbone | 0.9792 | 0.9007 |
| `b0_all` | + EyePACS | 0.9760 | 0.9255 |
| `v2s_all` | + EyePACS, bigger backbone | 0.9795 | **0.9321** |

The two levers combine, slightly sub-additively: data alone was +0.037,
backbone alone +0.013, together +0.044 rather than +0.050. `v2s_all` is the
best single model on both axes at once, which none of the earlier candidates
managed.

**Ensembles, and the rule that falls out of them:**

| Ensemble | Members | valA | **ODIR** |
|---|---|---|---|
| `ens3` | b0_all + b0_768 + v2s_512 | 0.9838 | 0.9278 |
| `ens_mix3` | b0_all + v2s_all + v2s_512 | 0.9830 | 0.9327 |
| `ens_all4` | all four | **0.9846** | 0.9340 |
| `ens_data2` | **b0_all + v2s_all only** | 0.9814 | **0.9355** |

The best in-distribution ensemble and the best cross-camera ensemble are not
the same ensemble, and they disagree in an instructive direction. Adding the
two DDR+APTOS-only models raises every in-distribution column and *lowers*
ODIR. They pull the average back towards the narrow domain they were trained
on; in distribution that reads as useful variance reduction, on a new camera
it reads as bias.

So: **ensemble members should be selected for the diversity of their training
data, not for their individual scores.** `v2s_512` and `b0_768` are strong
models --- they beat `b0_all` on DDR test --- and they still make the ensemble
worse where it matters.

Honest limits on this ranking. Six combinations were tried and the best kept,
which is selection, even though ODIR is the set reserved for exactly that. The
confidence intervals of the top three overlap heavily (`ens_data2`
0.9268-0.9447, `v2s_all` 0.9231-0.9415), so the ordering *within* the top
group is not established --- what is established is the gap between everything
trained on EyePACS and everything not.

Cumulative: **0.8882 to 0.9355 on an unseen camera, +0.047**, entirely from
data, capacity and an ensembling rule. No threshold was moved and no sealed
set was consulted.

### 11. Resolution on top of enough data buys almost nothing

`b0_768_all` --- the same 46,000 images at 768 px --- scores 0.9289 on ODIR,
against `b0_all`'s 0.9255 at 512 px. **+0.003.**

On the small two-source training set the same change was worth +0.016
(section 9). So the two levers overlap: much of what extra resolution was
buying was generalisation, and once the data supplies that, resolution has
little left to add. Worth knowing before anyone spends a week on 1024 px.

The final all-data ensemble, `ens_data3` (b0_all + v2s_all + b0_768_all),
reaches **ODIR 0.9370 (0.9286-0.9463)**, the best of everything tried.

| | ODIR |
|---|---|
| `b0_final`, where the night started | 0.8882 |
| `ens_data3`, where it ended | **0.9370** |

### 12. The disclosed look: the final candidate against IDx-DR

`ens_data3` was chosen entirely on ODIR. It was then scored once on the
sealed sets. This is that look, and it is the whole reason for the
discipline in the sections above.

**Messidor-2, exam level, IDx-DR's definition of referable (874 exams):**

| | exam AUC | sensitivity | specificity |
|---|---|---|---|
| `b0_final` (frozen policy) | 0.9394 | 96.7% | 57.0% |
| `ens_data3` (frozen policy) | **0.9528** | 93.7% (90.1-96.0) | **75.5% (71.9-78.8)** |
| IDx-DR, Abramoff 2016 | ~0.98 implied | 96.8% | 87.0% |

The two frozen operating points are not comparable directly --- the policy is
refitted on valB for each model and landed in a different place on the curve,
trading 3 points of sensitivity for 18 of specificity. The curve is the honest
comparison:

| Matched at | `b0_final` | `ens_data3` |
|---|---|---|
| IDx-DR's 96.8% sensitivity -> our specificity | 52.3% | **57.6%** |
| IDx-DR's 87.0% specificity -> our sensitivity | 84.8% | **89.6%** |
| The PS's 90% sensitivity -> our specificity | 75.0% | **84.3%** |

**We have not beaten IDx-DR.** Their published point still sits above our
curve: at their sensitivity we give up 29 points of specificity. What changed
in one night is that roughly a third of the gap closed, and the problem
statement's own target --- >90% sensitivity with >85% specificity --- went from
far away to 90.0% / 84.3% at exam level, short by seven tenths of a point of
specificity.

Other sealed-set numbers for `ens_data3`, all improved:

| Set | AUC | sensitivity | specificity |
|---|---|---|---|
| Messidor-2, per image | 0.957 (was 0.940) | 91.9% | 82.6% (was 69.5%) |
| IDRiD, per image | 0.962 (was 0.936) | 91.9% | 76.2% (was 73.6%) |
| DDR test (internal) | 0.985 (was 0.979) | 95.5% | 91.3% (was 86.3%) |

None of this was obtained by tuning on these sets. The ranking that produced
`ens_data3` used ODIR alone; this evaluation was run once, afterwards, and is
recorded below.

### 13. A weaker model from a different family beats a stronger one from the same family

`cnx_all` --- ConvNeXt-Tiny on the same 46,000 images --- is the **worst** of
the four all-data models on its own, and it makes the ensemble **better**:

| Model alone | ODIR |
|---|---|
| `cnx_all` | 0.9144 |
| `b0_all` | 0.9255 |
| `b0_768_all` | 0.9289 |
| `v2s_all` | 0.9321 |

| Ensemble | Members | valA | **ODIR** |
|---|---|---|---|
| `ens_data3` | b0_all + v2s_all + b0_768_all | 0.9817 | 0.9370 |
| `ens_data4` | all four | 0.9825 | 0.9388 |
| `ens_vrc3` | **v2s_all + b0_768_all + cnx_all** | 0.9829 | **0.9393** |

Dropping `b0_all` (0.9255) in favour of `cnx_all` (0.9144) *improves* the
ensemble. So section 10's rule was close but not quite right. It is not only
training-data diversity that matters --- it is error decorrelation, and an
architecture from a different family supplies more of that than a stronger
sibling of a model already in the mix. Three EfficientNets agree with each
other too much to be worth three votes.

A caveat on the recipe: `cnx_all` was trained with the schedule tuned for
EfficientNets (weight decay 1e-4, short one-cycle). ConvNeXt normally wants
around 0.05. Its 0.9144 is a statement about our hyperparameters, not about
the architecture, and a properly tuned run would likely be stronger --- which
would make the ensemble better still.

**No second look was spent on this.** `ens_vrc3` beats the already-evaluated
`ens_data3` by 0.0023 on ODIR, with confidence intervals that overlap almost
entirely (0.9307-0.9483 against 0.9286-0.9463), after roughly ten ensemble
combinations were tried on that set. The expected true difference is close to
zero. Scoring it on Messidor-2 and then keeping whichever of the two looked
better there is precisely the tuning this register exists to prevent, so the
reported model remains `ens_data3` and section 12 stands.

If the team later wants to ship `ens_vrc3`, the honest way to do it is to
commit to it *before* looking, evaluate once, and report that number whatever
it says.

## 21 September 2026

### 14. Final model: the PS target is met per image on Messidor-2, with the threshold fixed in advance

Fixing the ConvNeXt recipe --- weight decay 0.05 instead of the 1e-4 that
suits the EfficientNets --- moved it from 0.9144 to **0.9213** on ODIR, which
confirms section 13's suspicion that the first run measured our
hyperparameters rather than the architecture.

The ensemble that wins on ODIR is `ens_vrc3b` = `v2s_all` + `b0_768_all` +
`cnx_all_wd`: **ODIR 0.9404 (0.9318-0.9495)**.

It was committed to *before* being scored on the sealed sets, on the
undertaking that whatever came out would be reported. This is that number.

**Messidor-2, exam level, IDx-DR's definition of referable (874 exams):**

| | exam AUC | at IDx-DR's 87.0% spec | at the PS's 90% sens |
|---|---|---|---|
| `b0_final` (20 Sep, start) | 0.9394 | sens 84.8% | spec 75.0% |
| `ens_data3` | 0.9528 | sens 89.6% | spec 84.3% |
| **`ens_vrc3b` (final)** | **0.9590** | **sens 92.2%** | **spec 88.6%** |
| IDx-DR, Abramoff 2016 | ~0.98 implied | sens 96.8% | spec 87.0% |

**The problem statement requires >90% sensitivity and >85% specificity for
referable DR. With the threshold fixed in advance on valB, `ens_vrc3b` reaches
91.5% (88.5-93.7) / 87.6% (85.7-89.3) per image on Messidor-2 --- a camera and
country it has never seen.** Both point estimates clear the target; the
specificity interval clears 85% entirely, the sensitivity interval's lower
bound does not clear 90%. At exam level the frozen threshold gives 93.3% /
82.5%, short on specificity.

> **Correction, 22 Sep.** An earlier version of this section, and of the
> README, said the target was met "at 90.0% / 88.6% at exam level". That pair
> is a point on the exam-level ROC curve, found by searching Messidor-2's own
> labels for the threshold giving 90% sensitivity. It is a fair way to compare
> two curves at a matched sensitivity, which is how the table above uses it.
> It is not a threshold anyone could have set without the test labels, so it
> cannot be claimed as achieved performance. The frozen-threshold figures in
> the paragraph above are the claim.

Per-image numbers, all external, all improved:

| Set | AUC | sensitivity | specificity |
|---|---|---|---|
| Messidor-2 | 0.963 (was 0.940) | 91.5% | **87.6%** (was 69.5%) |
| IDRiD | 0.958 (was 0.936) | 93.5% | 73.6% |
| DDR test (internal) | 0.986 (was 0.979) | 94.7% | 93.2% (was 86.3%) |

**IDx-DR is still not beaten.** Their published point sits above our curve: at
their 96.8% sensitivity we give up 22 points of specificity (64.9% against
87.0%). What is now true, and was not yesterday, is that the *problem
statement's* bar is cleared on an unseen camera, and that the remaining gap to
a certified commercial device is 22 points rather than 35.

Nothing here was selected on a sealed set. The ordering came from ODIR; this
evaluation ran once, after the choice was fixed, and is logged below.

---

## 23–24 September 2026 (Linux workstation, RTX 5000 Ada 32 GB)

### 15. The models were undertrained: training them properly is worth as much as the whole ensemble was

The move to a 32 GB workstation removed the laptop's limits (3 data-loader
workers, batch 10–14, 3–7 epochs). The same three architectures were
retrained on the same 45,196 training images, same sources, same seed, with
one changed recipe: 10–15 epochs, batch 24–32, 16 loader workers, and an
exponential moving average of the weights (decay 0.9998, which is what is
validated and kept). The manifest was carried over byte for byte (paths
rewritten only), so every split is the one behind sections 1–14;
`check_splits.py` confirms nothing moved into train, valA or valB.

Ranked on ODIR, as always; no sealed set was consulted for anything below.

| Model | laptop run | ODIR | retrained | ODIR | gain |
|---|---|---|---|---|---|
| EfficientNetV2-S, 512 | `v2s_all` (7 ep) | 0.9327 | `v2s_e15` (best ep 10 of 15) | **0.9403** | +0.008 |
| ConvNeXt-Tiny, 512 | `cnx_all_wd` (6 ep) | 0.9219 | `cnx_e10` (best ep 10 of 10) | 0.9293 | +0.007 |
| EfficientNet-B0, 768 | `b0_768_all` (3 ep) | 0.9293 | `b0_768_e10` (best ep 6 of 10) | 0.9340 | +0.005 |

**One properly trained EfficientNetV2-S (0.9403) is as good on ODIR as the
whole three-network ensemble it was part of (`ens_vrc3b`, 0.9408).**

Longer is not better without limit: `v2s_e15` peaked at epoch 10 and its
valA AUC then fell from 0.982 to 0.977 while training loss kept falling ---
the network starting to memorise single-grader EyePACS labels. Best-epoch
selection on valA keeps the peak, and the remaining runs were shortened to
8–10 epochs.

Ensembles of the retrained models (ODIR, 95% bootstrap interval):

| Ensemble | ODIR |
|---|---|
| `ens_vrc3b` (laptop final) | 0.9408 |
| `v2s_e15` + `cnx_e10` + `b0_768_e10` | 0.9447 (0.937–0.953) |
| `v2s_e15` + `b0_768_e10` (all EfficientNet) | 0.9437 (0.935–0.952) |

Ensembling now adds much less than it did (+0.004 over the best single
model, against +0.008 on the laptop): better-trained members agree more.
Dropping ConvNeXt costs 0.001, far inside the interval --- which matters
because of the next point.

**MATLAB decides between near-equals.** `scripts/export_matlab.py` audits
every exported graph against the operators MATLAB's `importNetworkFromONNX`
converts to built-in layers. The EfficientNets, the lesion U-Net and the
landmark net are entirely built-in. ConvNeXt's channel LayerNorm exports as
Transpose + LayerNormalization, which MATLAB can only import through
auto-generated custom layers, and its GELU needs R2025a. Where two candidates
are statistically indistinguishable on ODIR, the one that runs natively in
the sponsor's toolchain wins.

The two larger candidates:

| Model | ODIR |
|---|---|
| `v2s768_e8`, EfficientNetV2-S at 768 (best ep 6 of 8) | **0.9415**, best single model |
| `cnxs_e8`, ConvNeXt-Small at 512 (best ep 8 of 8) | 0.9323 |

Final ranking (ODIR, 95% bootstrap interval), full table in
`overnight/ens_search.txt` on the workstation:

| Ensemble | ODIR |
|---|---|
| `v2s_e15` + `v2s768_e8` + `cnxs_e8` (best overall) | 0.9476 (0.940–0.956) |
| **`ens_e3` = `v2s_e15` + `b0_768_e10` + `v2s768_e8`** (best all-EfficientNet) | **0.9464 (0.938–0.954)** |
| `ens_vrc3b` (laptop final, section 14) | 0.9408 (0.932–0.950) |

**Selection rule, written down at 04:12 on 24 Sep, before `cnxs_e8` had a
score:** the final model is the highest-ODIR ensemble whose members are all
EfficientNets; one containing ConvNeXt replaces it only if it beats it on
ODIR by more than 0.005. The best ConvNeXt ensemble leads by 0.0012, so the
rule selects **`ens_e3`**. Its triage policy was fitted on valB alone
(`grader_eval.py --policy-only`, new, which stops before any test set is
scored), and it is exported for MATLAB in `runs/onnx_matlab_ens_e3/`, every
network built-in, the MATLAB recipe replayed against its predictions to 0.004.

**Commitment, 05:12 on 24 Sep, before the look below:** `ens_e3` is the
candidate to replace `ens_vrc3b`. It is scored once on IDRiD and Messidor-2
and the result is reported here whatever it says, including if it is worse
than section 14.

**The look (05:14, one run of `grader_eval --grader ens_e3 --rules seg_r34`):**

| | `ens_vrc3b` (section 14) | `ens_e3` |
|---|---|---|
| Messidor-2 per image, AUC | 0.963 | **0.967** (0.957–0.976) |
| Messidor-2 per image, frozen threshold | 91.5% / 87.6% | 91.7% (88.8–93.9) / 86.0% (84.0–87.8) |
| Messidor-2 exam level, AUC | 0.959 | **0.965** (0.953–0.976) |
| Messidor-2 exam level, frozen threshold | 93.3% / 82.5% | 93.7% / 80.1% |
| exam-level curve at IDx-DR's 96.8% sensitivity → specificity | 64.9% | **69.5%** |
| exam-level curve at IDx-DR's 87.0% specificity → sensitivity | 92.2% | 90.7% |
| exam-level curve at the PS's 90% sensitivity → specificity | 88.6% | 87.6% |
| IDRiD, AUC | 0.958 | **0.964** (0.950–0.976) |
| IDRiD, frozen threshold | 93.5% / 73.6% | 92.9% / **79.8%** |
| DDR test, AUC | 0.986 | **0.989** |

(The curve points are read off Messidor-2's own labels to compare curves, as
in section 14; they are not achievable operating points. The method
reproduces section 14's 92.2% / 88.6% / 64.9% for `ens_vrc3b` exactly.)

What it says, without spin:

- **Ranking improved everywhere.** AUC is up on every held-out and external
  set, as ODIR predicted it would be.
- **The curve changed shape.** At the high-sensitivity end, where a
  screening programme operates and where IDx-DR sits, specificity rose from
  64.9% to 69.5%. In the middle of the curve it fell by about a point.
- **At the frozen threshold, Messidor-2 specificity fell 1.6 points**
  (87.6% → 86.0%) while IDRiD's rose 6.2 (73.6% → 79.8%). The per-image
  PS target is still met on Messidor-2 by point estimate (91.7% / 86.0%);
  the specificity interval now dips below 85% where section 14's did not.
- **IDx-DR is still not beaten.** At their 96.8% sensitivity we give up
  17.5 points of specificity (69.5% against 87.0%), down from 22.

`ens_e3` replaces `ens_vrc3b` as the final model. It was chosen on ODIR and
on MATLAB compatibility before this look; reverting to `ens_vrc3b` because
its frozen Messidor-2 specificity is 1.6 points higher would be choosing a
model by its score on the sealed set.

## Looks at held-out data

ODIR is deliberately absent from this register. It is the designated
model-selection set: held out from training, scored freely, and never quoted
as a result. That is the whole point of having it --- every ranking decision
on 20 September was made there so that the two sets below stayed sealed.

| Date | Set | What was looked at | Why |
|---|---|---|---|
| 19 Sep | IDRiD | frozen-policy sens/spec, recalibration curve | first external evaluation |
| 19 Sep | IDRiD | oracle ROC points | to test whether the threshold or the model was at fault |
| 19 Sep | Messidor-2 | exam-level sens/spec vs IDx-DR, oracle ROC | headline external comparison |
| 20 Sep | Messidor-2 | exam-level AUC under four fusion rules | experiment 1 |
| 20 Sep | IDRiD, Messidor-2 | sens/spec under two threshold-transfer rules | experiment 2 |
| 20 Sep | IDRiD, Messidor-2 | capture statistics only, no labels | experiment 4, costs no label look |
| 20 Sep | Messidor-2 | label composition only (how many exams are oedema-only) | experiment 5, no model output involved |
| 21 Sep | IDRiD, Messidor-2 | **full evaluation of the final candidate `ens_data3`**, plus its ROC at matched operating points | experiment 12: one look, on a model selected entirely on ODIR |
| 21 Sep | IDRiD, Messidor-2 | **full evaluation of `ens_vrc3b`**, the final model | experiment 14: committed to on ODIR before looking, reported as it came out |
| 24 Sep | IDRiD, Messidor-2 | **full evaluation of `ens_e3`**, plus its exam-level ROC at matched operating points (and `ens_vrc3b`'s, recomputed to check the method) | experiment 15: selected on ODIR by a rule written before the last candidate was scored, committed to in writing, scored once, reported as it came out |

Nothing in this file was selected by its score on a held-out set. Experiments
1 and 2 were rejected on those sets, which is the safe direction: we are
declining to adopt things, not tuning towards them.
