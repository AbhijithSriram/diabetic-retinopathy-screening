# Datasets

No image or label file is committed to this repository. `scripts/prepare_data.py`
reads whatever is present under `../data/raw/` (override with `DRISHTI_DATA`).

| Dataset | Role here | Images | Labels used | Source | Licence / terms |
|---|---|---|---|---|---|
| DDR (OIA-DDR) | grader train / valid / internal test; quality model; lesion segmentation | 13,673 graded (incl. ungradable class); 757 with lesion masks | ICDR 0-4 + ungradable; MA/HE/EX/SE pixel masks | github.com/nkicsl/DDR-dataset (Google Drive) | research use; cite Li et al., Inf. Sci. 2019 |
| APTOS 2019 | grader train / valid / internal test (attached to the PS) | 3,662 | ICDR 0-4 | Kaggle competition `aptos2019-blindness-detection` | Kaggle competition rules (accepted by the team lead) |
| IDRiD | **external test for the grader (never trained on)**; lesion + optic-disc masks; optic disc / fovea landmarks | 516 graded; 81 segmented; 516 localised | ICDR 0-4, DME risk 0-2; pixel masks; OD and fovea centres | IEEE DataPort (via a CC-BY Zenodo re-upload, MD5-checked) | CC BY 4.0 |
| Messidor-2 | **external test (sealed)**; comparison with IDx-DR 2016 | 1,748 (874 exams) | Krause et al. 2018 adjudicated grade, DME, gradability | images: ADCIS (licence accepted by the team lead); grades: Kaggle `google-brain/messidor2-dr-grades` | non-commercial research; no redistribution |
| EyePACS (Kaggle DR 2015) | grader **training only** | 35,126 | ICDR 0-4, single reader | Kaggle dataset `tanlikesmath/diabetic-retinopathy-resized` (a resized mirror of the competition data) | competition data, research use |
| ODIR-5K | **held-out internal cross-camera validation**; never trained on | 6,392 (3,358 patients) | ICDR grade read from per-eye free-text findings | Kaggle dataset `andrewmvd/ocular-disease-recognition-odir5k` | CC BY 4.0 (per the uploader) |
| HRF | vessel segmentation train / test | 45 | manual vessel masks | FAU Erlangen | free for research, cite Budai et al. 2013 |
| mBRSET (optional) | external test on a handheld phone-based camera | 5,164 | ICDR, DME, quality | PhysioNet (credentialed access) | PhysioNet Credentialed Health Data Licence 1.5.0 |

## Finding: DDR's official split confounds camera series with grade

DDR images come in two naming series: `007-xxxx-yyy` (8,250 images) and
timestamp names (5,424), which differ in resolution and look (a different
camera or site). In the **official** split:

| official split | timestamp series | 007 series |
|---|---|---|
| train | 1,258 images, **all grade 0** | all grades, 575 ungradable |
| valid | 1,253 images, **all grade 0** | **no grade 0** |
| test | grades 0–3, 957 of them grade 2 | no grade 0 |

A grader trained on the official split can learn "this camera's look =
healthy". The official validation set also lets the series alone predict the
label, which inflates validation accuracy. Our first grader (trained on the
official split) scored validation AUC 0.987, but AUC was about 0.92 within
either series, and sensitivity on the official test set's timestamp series
was 72%.

We therefore **pool DDR and re-split it** (series × grade stratified, whole
duplicate groups, 60/10/10/20). Internal DDR results in this repository are
on our split, not the official one, and are not comparable with papers that
use the official split.

## Why EyePACS and ODIR are treated differently from DDR and APTOS

Both were added on 20 September 2026 and neither is allowed to influence a
threshold or a reported number.

**EyePACS is training material only.** Its 35,126 grades come from one reader
per image with no adjudication, and the competition's own documentation
acknowledges the error rate. Noisy labels in `train` cost a little accuracy.
The same labels in `valB` would mean fitting the conformal thresholds --- the
part of this system that carries a stated guarantee about missed referable
disease --- against grades we do not trust. `prepare.assign_splits` forces
every EyePACS row to `train` so this cannot happen by accident.

Its images are ingested straight from the archive into the crop cache
(`scripts/ingest_zip.py`) rather than unpacked, because the machine does not
have room for 35,000 originals and their crops at once. They go through the
same `preprocess.prepare` as every other image; there is no second
preprocessing path.

**ODIR-5K is held out whole**, as an internal cross-camera validation set
(`split == "devext"`). Until it arrived, every judgement about generalisation
had to be made either on in-distribution validation, which cannot see camera
shift at all, or on IDRiD and Messidor-2, which are the sets we report
against and must not tune on. ODIR is a third country on a mixture of camera
brands, so it is somewhere to compare two candidate models honestly.

Its grades are read off free-text findings ("moderate non proliferative
retinopathy" to grade 2, and so on; `sources.ODIR_GRADE`), with anything
ambiguous --- "diabetic retinopathy" with no severity, "suspected" --- left
unlabelled. An eye whose findings list other diseases but no DR is grade 0,
which is what the annotation means. That is good enough to rank two models
and not good enough to quote, so ODIR never appears as a headline number.

## Leakage controls

- The grader learns only from DDR train, APTOS train and EyePACS. IDRiD,
  Messidor-2 and mBRSET are external, and ODIR is held out: no training, no
  threshold, no early stopping uses any of them.
- Original Messidor (1,058 of Messidor-2's images) is **not** used anywhere,
  including pretraining, because it is inside Messidor-2.
- Near-duplicates are found with a 64-bit difference hash on the FOV crop.
  APTOS is split by duplicate group; external images that duplicate a
  development image are excluded from external metrics; duplicate groups
  with conflicting grades are dropped from validation and test.
- The lesion model trains on IDRiD segmentation images. Any IDRiD grading
  image that is the same photo (hash match) is excluded when the lesion rules
  are evaluated on IDRiD (`rules_eval.py`, `in_seg_train`).
- Validation data is halved: valA for early stopping and temperature, valB
  for conformal calibration, so the calibration set was not used for model
  selection.

## Added 24 September 2026

| Dataset | Images | Role | Why |
|---|---|---|---|
| EyePACS 2015 test half (source `eyepacs_test`) | 53,576 (53,572 used) | training only | The competition's released solution grades, single reader like the train half. Different patients from the train half (checked). Appended by `scripts/append_eyepacs_test.py`; 4 images that matched an ODIR or DDR held-out image were added as `excluded`. |
| DeepDRiD, regular fundus (split `devext2`) | 2,000 | second model-selection set, never trained on | Shanghai, 0–4 grades per eye by ophthalmologists, two photographs per eye. ODIR's grades are read off free text and a second training seed moved its AUC by 0.005, as much as the differences being ranked; a second, expert-graded selection set reduces that noise. None of its images duplicates a training image. |

Both were **appended**: every existing manifest row is compared byte for byte
after writing, and `check_splits.py` still reports only the 15 known moves
to `excluded`, none into train, valA or valB.

**Not used, and why:**
- **RFMiD** comes from the same Nanded clinic as IDRiD, one of our external
  test sets. Training on it would make IDRiD no longer unseen. It also has
  only a yes/no DR label.
- **FLAIR** and **UrFound** (retinal foundation models) were pretrained on
  image–text pairs built from 37 public datasets, including IDRiD, Messidor,
  ODIR, DeepDRiD, DDR and HRF with their labels. Using them would leak our
  test and selection sets.
- **RETFound-Green** was pretrained, without labels, on all of DDR, ODIR and
  part of AIROGS. It is usable, but its ODIR and DDR-test scores would be
  inflated, so it may only be judged on DeepDRiD.
- **RETFound-MAE** (Moorfields + Kaggle EyePACS) is the clean choice and is
  the teacher for distillation. It is a ViT and does not import cleanly into
  MATLAB, so what ships is an EfficientNet trained on its soft labels.
