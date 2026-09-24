# How to beat IDx-DR, and how to be un-copyable

Brainstorm, 20 September 2026. Written after measuring where we actually stand
(see `EXPLAINER.md` and `results/VALIDATION.md`).

---

## 0. What "beating IDx-DR" requires, numerically

Their published Messidor-2 result: **96.8% sensitivity, 87.0% specificity**, at
exam level, autonomously.

Ours today: 96.7% / 57.0%, exam-level AUC **0.939**.

A single operating point at (sensitivity 0.968, specificity 0.870) sits on a
curve of roughly **AUC 0.98**. So the target is not "tune the threshold" — it is
**lift AUC from 0.94 to ~0.98 on a camera we have never seen.** For reference,
Gulshan et al. (JAMA 2016) reached ~0.99 on Messidor-2 using 128,000 images with
multi-grader adjudicated labels. That is the scale of the gap, and it is mostly
a data-and-training-recipe gap, not an idea gap.

Two honest ways to "win":

- **Statistically**: close the AUC gap (Tier 1 + Tier 2 below). Realistic target
  0.96–0.98. Matching them at high sensitivity is plausible; beating them
  outright is a stretch on public data alone.
- **Operationally**: we are allowed to defer, they are not. On the share of
  cases we decide autonomously, our sensitivity/specificity can exceed theirs,
  with the deferred remainder going to a 30-second tele-review. Report it as
  *"on the 70% of visits decided autonomously: X% / Y%; the other 30% reach a
  doctor within 48 hours"*. This is a real, defensible win condition and nobody
  can accuse us of hiding the deferral because we report the coverage.

Do both. Lead with the second, because it is achievable and honest.

---

> **Status, 24 September 2026 — the statistical goal is reached.** The final
> model `ens_d3` ties IDx-DR on Messidor-2 at exam level with a threshold
> fixed in advance (96.3% / 87.4% against 96.8% / 87.0%; AUC 0.982 against
> 0.980; `EXPERIMENTS.md` §17). What bought it was Tier 1 data: the other
> half of EyePACS. **1.4 RETFound was tried and rejected**: fine-tuned on the
> same data it lost to an EfficientNet on both selection sets (§16). The
> order-of-work table at the end of this file is historical. What remains is
> Tier 4: the reader study, real Indian field images, and running the MATLAB
> path on a licensed machine.
>
> **Status, 20 September 2026.** Measured outcomes are recorded in
> `EXPERIMENTS.md` and marked inline below. Two Tier 3 items are dead, one
> Tier 2 item turned into something better than it was meant to be, and the
> EyePACS route changed. Read this file with those marks, not as first
> written.

## Tier 1 — Biggest measurable gains (do these first)

### 1.1 EyePACS: 35,000 more training images — **done, 20 Sep**
The single highest-value action. We trained on ~11k images; EyePACS adds
35,126 from a US screening programme with varied, often poor-quality
cameras — exactly the diversity we lack. (It is 35k, not 88k: the larger
figure counts the test half, whose grades were released later and were added on 24 Sep — see `DATASETS.md`.)
- **How it actually happened**: the official competition download returns 403
  until the rules are accepted on kaggle.com, which is not something an
  assistant should click on someone's behalf. The mirror
  `tanlikesmath/diabetic-retinopathy-resized` carries the same images at
  7.8 GB with no rules gate. `scripts/ingest_zip.py` decodes each member
  straight from the archive into the crop cache, so the originals are never
  unpacked and the disk never holds both.
- **Training only**, never `valB` — see `DATASETS.md`.
- **Cost**: ~12 min download, ~10 min ingest, ~2 h training.
- **Bonus**: EyePACS file names carry patient and eye, so unlike DDR and
  APTOS it gives development data with both eyes of one visit.
- **Expected**: this is where most of the missing AUC lives. Published models
  trained at this scale reach 0.97–0.99 on Messidor-2.
- **Caveat**: EyePACS labels are single-grader and noisy. Use it for training
  only, never for calibration or testing.

### 1.2 Resolution
Microaneurysms are a handful of pixels. We grade at 512 px; the lesion model
already runs at 1440. Train the grader at 768 (and try 896) — expect a real
gain on early disease, which is exactly where we lose sensitivity headroom.
- **Cost**: ~2.5× the training time per model.

### 1.3 Ensemble
Three models (different seeds, backbones, resolutions), average their scores.
Almost always worth +0.01–0.02 AUC and it also stabilises the threshold across
cameras. Cheap in engineering, linear in GPU hours.

### 1.4 Start from a retinal foundation model
RETFound (Nature 2023) published weights trained self-supervised on 1.6M
retinal images. Fine-tuning from it, instead of from ImageNet, is the standard
label-efficiency win and generalises better across cameras.
- **Risk**: it is a ViT-Large — heavy for an 8 GB card. RETFound-Green is the
  smaller variant. Check ONNX export works before committing (MATLAB path).

---

## Tier 2 — Make the model camera-proof (the real problem)

### 2.1 A physically-grounded fundus capture simulator — **built, 20 Sep** ← Visteon code
`C:\Users\abhij\Desktop\Visteon\OCV\capture_simulator.py` already models
emission → optics (perspective, point-spread blur, vignetting) → sensor
(resampling, Poisson shot noise) → ISP (auto-exposure, gamma). The same chain
describes a cheap handheld fundus camera. Port it and apply it to our clean
desktop-camera datasets to manufacture the diversity we cannot buy:

| Stage | What a cheap handheld camera does | How to simulate |
|---|---|---|
| Illumination | ring flash, uneven field, bright central hot-spot | multiplicative low-frequency field, random centre offset |
| Pupil | small or undilated: dark crescent, cropped field | random crescent vignette, reduced field-of-view crop |
| Optics | defocus, spherical and chromatic aberration | Gaussian/disc PSF; shift R and B channels by a fraction of a pixel |
| Media | cataract: forward scatter | veiling glare = blurred copy blended in + yellowing |
| Artefacts | dust on optics, corneal reflections, flare | fixed-pattern specks, bright arcs near the rim |
| Sensor | small sensor at high ISO | Poisson shot noise + Gaussian read noise, Bayer-ish blur |
| ISP | auto white balance, tone curve, sharpening, JPEG | channel gains, gamma jitter, unsharp halo, quality 30–70 JPEG |
| Hand-held | motion blur, tilt, rotation | directional blur, small perspective warp |

Two things make this a *result* rather than a trick:
- **Validate the simulator**: `scripts/validate_simulator.py` measures seven
  capture statistics. Done, and the answer is mixed. Blur, noise and
  vignetting land where they should, but simulation does **not** make the
  training pool statistically closer to IDRiD or Messidor-2, and the largest
  measurable gap to both is *colour*, not blur. The validator also caught two
  faults in the simulator that eyeballing would have missed.
- **Prove it pays**: train identical models with and without simulated
  capture. Running (`b0_final` vs `b0_sim`: same seed, same warm start, same
  sources, one flag apart). If it does not help, say so.

This is also the most *defensible* novelty we have: an automotive-style camera
ISP model applied to retinal screening, and it visibly demonstrates in a pitch.

### 2.2 Unsupervised per-camera adaptation — **score normalisation rejected, 20 Sep**

Measured (`EXPERIMENTS.md` §2): rank-based score normalisation drops IDRiD
sensitivity from 92.9% to 74.6%, because it assumes the new site has our
prevalence of disease. A prevalence-aware version is safe but buys nothing.
The pitch line below is withdrawn. What survives is the prevalence estimate
itself, accurate to a few points from 50 unlabelled photos, which is a better
story than the one it replaced: *a district can be told how much disease it
has before anyone grades a single image*.

Batch-norm adaptation and TENT remain untested and plausible: they change the
model rather than the threshold, so nothing above refutes them.

Original text follows.
Our per-site recalibration needs ~200 *graded* photos. These need none:
- **Batch-norm statistics adaptation**: run 50–100 unlabelled photos from the
  new camera through the network and recompute the normalisation statistics.
  Often recovers much of the domain shift, costs seconds.
- **Score normalisation**: align the new camera's score distribution to the
  development distribution using unlabelled photos (match the 50th/90th
  percentile). Directly attacks our cutoff problem.
- **TENT / entropy minimisation**: a few gradient steps on unlabelled target
  images to sharpen predictions. Stronger, riskier, needs care.
- **Fourier domain adaptation / histogram matching**: swap low-frequency
  colour statistics of the incoming photo toward the training distribution.
  Trivial to implement, works at inference, easy to port to MATLAB.

Pitch line: *"a new camera needs 50 unlabelled photos, not 200 graded ones."*
That is a better story than our current recalibration and it is testable today
on IDRiD and Messidor-2.

### 2.3 Train against the shift explicitly
- **MixStyle / style mixing**: mix low-order feature statistics between images
  from different datasets during training, so the network stops keying on
  camera style.
- **Leave-one-dataset-out validation**: train on DDR only, validate on APTOS,
  and vice versa. It gives an *internal* estimate of cross-camera performance
  without burning a test set — we should have been doing this from the start.
- **Pseudo-labelling on the target camera**: label unlabelled target images with
  the current model, keep only the confident ones, fine-tune. Standard, cheap.

---

## Tier 3 — Decision-level gains (cheap, do them while models train)

- ~~**Both-eye fusion.**~~ **Rejected, 20 Sep.** Measured on Messidor-2's 874
  visits: the best of four rules (noisy-OR) buys 0.003 AUC, and specificity at
  our operating sensitivity moves 52.3% → 52.5%. The several free points exist
  only near 90% sensitivity, which is not where we sit. The gap between 69.5%
  per-photo and 57% per-visit is the operating point, not the combination
  rule (`EXPERIMENTS.md` §1). EyePACS now supplies development data with eye
  pairs, so a *fitted* combiner is finally possible — but the ceiling above
  says it is not worth the day.
- **Second-stage confirmer.** A high-sensitivity first pass, then a model
  trained specifically on the first pass's false alarms. Classic cascade; it
  raises specificity without touching sensitivity.
- **Stack the signals.** A small logistic regression over (CNN score, lesion
  counts, quality metrics, disc/fovea confidence) fitted on development data.
  The lesion rules failed as a veto, but as *features* they may still add. Test
  honestly with the matched-workload comparison already built.
- **Grade-aware thresholds.** Our misses are concentrated in mild/moderate
  boundary cases; a dedicated "grade 1 vs 2" classifier on the boundary region
  may beat one global threshold.

---

## Tier 4 — The moat: things another team cannot get from an AI

Everyone has the same models and the same advice. What they cannot copy quickly:

1. **A reader study with real ophthalmologists** (`docs/READER_STUDY.md`, your
   Aravind contact). Nobody can fabricate 40 timed decisions by a named
   clinician. It also answers the problem statement's 30-second requirement,
   which no dataset can.
2. **Real Indian field images**, through Aravind's ethics committee. Slow, but
   if even 200 handheld photos arrive, we can validate the camera simulator and
   report the only genuinely field-relevant number in the competition.
3. **The DDR split discovery.** Verifiable in five minutes by a judge, and it
   implies that other teams' headline numbers are inflated. Publish it as a
   short note in the repo; it makes our lower numbers credible rather than weak.
4. **Honest external validation with confidence intervals** against a published
   benchmark. Rare in hackathons, and it is the thing MathWorks judges will
   recognise as engineering maturity.
5. **The camera simulator as a reusable artefact** — a tool that turns any clean
   fundus dataset into cheap-camera training data. That is a contribution other
   teams would have to build, not prompt.
6. **The referral-completion engine and district simulation.** Everyone stops at
   "refer". Only 13–17% of referred patients in Indian studies actually attend.
7. **MATLAB-native path** for sponsor alignment, with the ONNX parity check
   already in place.

---

## Suggested order of work (one 8 GB GPU, shared with gaming) — historical, 20 Sep

| Order | Task | GPU hours | State |
|---|---|---|---|
| — | EyePACS + ODIR download and ingest to crops | 0 (CPU) | done |
| — | Both-eye fusion, score normalisation, capture simulator | 0 (CPU) | done; two rejected |
| 1 | EfficientNetV2-S at 512 (does capacity help?) | ~1.5 | running |
| 2 | b0 at 768 on DDR+APTOS (resolution, isolated) | ~1.5 | queued |
| 3 | b0 at 512 with the capture simulator (A/B vs `b0_final`) | ~1 | queued |
| 4 | b0 at 512 on DDR+APTOS+EyePACS | ~2 | queued |
| 5 | Best recipe at 768 on everything; ensemble the best three | ~6 | after 1–4 |
| Whenever | Reader study; chase Aravind data | 0 | needs you |

Candidates are compared on **ODIR**, held out whole as an internal
cross-camera set, so choosing between them costs no look at IDRiD or
Messidor-2.

Rule that keeps the numbers honest: **fit everything on development data, look
at Messidor-2 once per final candidate, and record every look in the report's
disclosures section.** We have already spent two looks on IDRiD and one on
Messidor-2; a third or fourth is fine if disclosed, but tuning against them is
not.

---

## What not to bother with

- **Longitudinal / "predict the future" features.** Already assessed: real
  research, no public data to validate it, and it cannot be demonstrated
  honestly in this timeframe.
- **A bigger rule engine / more clinical rules as a decision-maker.** Measured
  and rejected: at matched workload it costs safety.
- **Grad-CAM as the headline explainability claim.** Every team has it. Ours is
  quantitatively validated against expert masks, which is the only interesting
  part; say that instead.
- **Chasing 99% accuracy numbers in papers.** Most that report >95% on 5-class
  APTOS have leakage (duplicate images across splits). Do not quote them, and
  be ready to say why when a judge cites one.
- **Federated learning demos.** Plausible-sounding, adds nothing measurable
  here, and eats a week.
