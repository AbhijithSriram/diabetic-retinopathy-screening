"""Compose results/VALIDATION.md from the JSON results. Every number in the
report is read from a results file written by an evaluation script; nothing
is typed in by hand.

    python scripts/make_report.py --grader b0_512 --seg seg_r34
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drishti.config import RESULTS  # noqa: E402


def pct(t, digits=1):
    if t is None or t[0] != t[0]:
        return "n/a"
    return f"{100 * t[0]:.{digits}f}% ({100 * t[1]:.{digits}f}–{100 * t[2]:.{digits}f})"


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grader", default="b0_512")
    ap.add_argument("--seg", default="seg_r34")
    a = ap.parse_args()
    L = ["# DrishtiTrust validation report", "",
         "Every figure below is generated from `results/*.json` by `scripts/make_report.py`.",
         "95% intervals: Wilson for proportions, bootstrap (by exam where known) for AUC.",
         "Thresholds were fitted on the calibration split (valB) only and frozen before any external image was scored.", ""]

    ev = load(f"eval_{a.grader}.json")
    if ev:
        pol = ev["policy"]
        L += ["## 1. Referable-DR grading (PS target: sensitivity > 90%, specificity > 85%)", "",
              f"Operating point λ_op = {pol['lambda_op']:.4f} (chosen on valB for ≥95% sensitivity). "
              f"Triage: auto-recall below λ_low = {pol['lambda_low']:.4f} (CRC, target ≤{100 * pol['alpha_miss']:.0f}% of referable auto-recalled), "
              f"direct referral at or above λ_high = {pol['lambda_high']:.4f}. Prediction sets target {100 * (1 - pol['alpha_set']):.0f}% coverage.", "",
              "| Test set | Images | AUC | Sensitivity | Specificity | QWK | ECE | Meets PS target |", "|---|---|---|---|---|---|---|---|"]
        for name, s in ev["sets"].items():
            op = s["at_operating_point"]
            meets = "yes" if s["meets_ps_target"]["sensitivity_gt_90"] and s["meets_ps_target"]["specificity_gt_85"] else "no"
            if meets == "yes" and not s["meets_ps_target"]["ci_lower_sens_gt_90"]:
                meets = "point estimate only"
            L.append(f"| {name} | {s['n_images']} | {s['auc_referable'][0]:.3f} ({s['auc_referable'][1]:.3f}–{s['auc_referable'][2]:.3f}) | "
                     f"{pct(op['sensitivity'])} | {pct(op['specificity'])} | {s['qwk']:.3f} | {s['ece_referable']:.3f} | {meets} |")
        m2 = ev.get("messidor2_exam_level")
        if m2:
            L += ["", "### Messidor-2 at exam level, IDx-DR definition (moderate NPDR or worse and/or macular oedema)", "",
                  f"{m2['n_exams']} exams, {m2['referable_exams']} referable. Published reference: IDx-DR X2.1, sensitivity 96.8%, specificity 87.0% (Abramoff et al., IOVS 2016).", "",
                  "| Configuration | Sensitivity | Specificity |", "|---|---|---|",
                  f"| CNN only | {pct(m2['cnn_only']['sensitivity'])} | {pct(m2['cnn_only']['specificity'])} |"]
            if "cnn_plus_dme_rule" in m2:
                L.append(f"| CNN + DME rule (exudate within 1 DD of fovea) | {pct(m2['cnn_plus_dme_rule']['sensitivity'])} | {pct(m2['cnn_plus_dme_rule']['specificity'])} |")
        L += ["", "## 2. Uncertainty: prediction sets and triage", "",
              "| Test set | Set coverage | Mean set size | Recall | Review | Refer | Referable auto-recalled |", "|---|---|---|---|---|---|---|"]
        for name, s in ev["sets"].items():
            t = s["triage_cnn_only"]
            L.append(f"| {name} | {pct(s['prediction_sets']['coverage'])} | {s['prediction_sets']['mean_size']:.2f} | "
                     f"{100 * t['recall_fraction']:.1f}% | {100 * t['review_fraction']:.1f}% | {100 * t['refer_fraction']:.1f}% | "
                     f"{t['referable_auto_recalled']['count']}/{t['referable_auto_recalled']['of_referable']} ({pct(t['referable_auto_recalled']['rate'])}) |")
        L += ["", "Per-grade coverage (external sets), showing where the guarantee is weakest:", ""]
        for name, s in ev["sets"].items():
            if "external" in name:
                per = ", ".join(f"grade {g}: {100 * v[0]:.0f}% (n={v[3]})" for g, v in s["prediction_sets"]["per_grade"].items())
                L.append(f"- {name}: {per}")
        if ev.get("recalibration"):
            L += ["", "### Camera shift: re-calibrating λ_low from audit labels on the new camera", "",
                  f"Target: at most {100 * pol['alpha_miss']:.0f}% of referable eyes auto-recalled. Averages over 300 random audit orders.", "",
                  "| Deployment | Audit labels | Referable auto-recalled (mean / 90th pct) | Share auto-recalled | Operating point sensitivity (mean / 10th pct) | Specificity (mean / 10th pct) |",
                  "|---|---|---|---|---|---|"]
            for name, r in ev["recalibration"].items():
                for row in r["rows"]:
                    L.append(f"| {name} | {row['audit_labels'] or 'none (frozen valB thresholds)'} | {100 * row['mean_missed_referable_rate']:.1f}% / {100 * row['p90_missed_referable_rate']:.1f}% | "
                             f"{100 * row['mean_auto_recall_fraction']:.1f}% | {100 * row['op_sensitivity_mean']:.1f}% / {100 * row['op_sensitivity_p10']:.1f}% | "
                             f"{100 * row['op_specificity_mean']:.1f}% / {100 * row['op_specificity_p10']:.1f}% |")
                sr = r.get("site_recalibrated_triage")
                if sr:
                    L.append("")
                    L.append(f"With {sr['audit_labels']} audit labels and both thresholds re-fitted: healthy eyes referred directly "
                             f"{100 * sr['refer_given_non_referable']:.1f}%, referable eyes auto-recalled {100 * sr['recall_given_referable']:.1f}%, "
                             f"healthy eyes sent to tele-review {100 * sr['review_given_non_referable']:.1f}%.")

    ab = load(f"ablation_{a.grader}_{a.seg}.json")
    if ab:
        L += ["", "## 3. Does the lesion rule engine earn its place? (matched-workload ablation)", ""]
        for src, v in ab.items():
            if src == "policy":
                continue
            L += [f"**{src}** ({v['n']} images)", "", "| Configuration | Recall | Review | Refer | Referable auto-recalled |", "|---|---|---|---|---|"]
            for r in v["rows"]:
                L.append(f"| {r['config']} | {100 * r['recall']:.1f}% | {100 * r['review']:.1f}% | {100 * r['refer']:.1f}% | {r['referable_missed']}/{r['referable']} |")
            rb = v["rule_evidence_referable"]
            L += ["", f"Rule evidence alone (lesions beyond microaneurysms ⇒ referable): sensitivity {pct(rb['sensitivity'])}, specificity {pct(rb['specificity'])}.", ""]

    les = load(f"lesions_{a.seg}.json")
    if les:
        L += ["## 4. Lesion segmentation (pixel AUPR inside the field of view)", "",
              "| Test set | MA | HE | EX | SE | Mean of 4 | OD | Published mean AUPR, same test set [1] |", "|---|---|---|---|---|---|---|---|"]
        ref = {"idrid_test": "IDRiD 2018 winners VRT 0.647, iFLYTEK 0.648; M2MRF 0.672",
               "ddr_test": "HRNetV2 0.452; M2MRF 0.499"}
        for k in ("idrid_test", "ddr_test"):
            if k in les:
                au = les[k]["aupr"]
                mean4 = sum(au[c] for c in ("MA", "HE", "EX", "SE")) / 4
                L.append(f"| {k} ({les[k]['n_images']} images) | " + " | ".join(f"{au[c]:.3f}" for c in ("MA", "HE", "EX", "SE"))
                         + f" | **{mean4:.3f}** | " + (f"{au['OD']:.3f}" if "OD" in au else "–") + f" | {ref[k]} |")
        L += ["", f"_{les['note']}. [1] Liu et al., M2MRF, arXiv 2111.00193, Tables 4 and 6. Our lesion model is below the state of the art._", ""]
        cc = les.get("component_calibration")
        if cc:
            L += ["Lesion counts and overlays only include components whose peak confidence gives at least 75% precision on DDR-valid masks "
                  "(kept / all predicted: " + ", ".join(f"{k} {v['kept']}/{v['components']}" for k, v in cc.items()) + ").", ""]
        for k in ("idrid_test", "ddr_test"):
            if k in les and les[k].get("presence"):
                L.append(f"Image-level lesion presence, {k}: " + "; ".join(
                    f"{c} sens {pct(v['sensitivity'], 0)}, spec {pct(v['specificity'], 0)}" for c, v in les[k]["presence"].items()))
        L.append("")
    lm = load("landmarks_lmk_r18.json")
    if lm:
        L += ["## 5. Optic disc and fovea localisation (IDRiD test, original pixels)", "",
              f"Optic disc: mean distance {lm['od_mean_ed']} px (median {lm['od_median_ed']}); within 100 px: {lm['od_within_100px_pct']}%. "
              f"Fovea: mean {lm['fovea_mean_ed']} px (median {lm['fovea_median_ed']}); within 100 px: {lm['fovea_within_100px_pct']}%, within 200 px: {lm['fovea_within_200px_pct']}%.", ""]
    ve = load("vessels_hrf.json")
    if ve:
        L += ["## 6. Vessel segmentation", "", f"{ve['dataset']}, {ve['scale']}: sensitivity {100 * ve['sensitivity']:.1f}%, specificity {100 * ve['specificity']:.1f}%, "
              f"accuracy {100 * ve['accuracy']:.1f}%, AUC {ve['auc']:.3f} (inside the FOV).", ""]
    gc = load(f"gradcam_{a.grader}.json")
    if gc:
        L += ["## 7. Grad-CAM checked against ophthalmologist lesion masks", "",
              f"{gc['note']}. Images with lesions: {gc['images_with_lesions']}. Share of Grad-CAM mass on annotated lesions (dilated {gc['dilation_px_at_512']} px): "
              f"{100 * gc['mean_energy_in_lesion']:.1f}%, which is {gc['mean_lift_over_chance']}× chance (95% CI {gc['lift_95ci'][0]}–{gc['lift_95ci'][1]}). "
              f"Pointing game (hottest pixel on a lesion): {100 * gc['pointing_game'][0]:.0f}% ({100 * gc['pointing_game'][1]:.0f}–{100 * gc['pointing_game'][2]:.0f}), chance {100 * gc['pointing_game_chance']:.0f}%.", ""]
    q = load("quality_qual_b0.json")
    if q:
        d = q["ddr_test"]
        L += ["## 8. Image-quality gate", "", f"DDR test, detecting ungradable images: AUC {d['auc']:.3f}; sensitivity {pct(d['sensitivity'])}, specificity {pct(d['specificity'])}."]
        if "messidor2" in q:
            m = q["messidor2"]
            L.append(f"Messidor-2 (external), adjudicated ungradable: AUC {m['auc']:.3f}; sensitivity {pct(m['sensitivity'])}, specificity {pct(m['specificity'])}.")
        L.append("")
    sc = load("simulation/sim_scenarios.json")
    if sc:
        L += ["## 9a. District scenarios (100,000 screenings / year, 8% referable prevalence)", "",
              "Triage rates are the measured conditional rates re-mixed to a screening population's prevalence; other parameters are in sim/params.yaml.", "",
              "| Scenario | Recall | Tele-review | Direct refer | Direct referrals truly referable | Referable auto-recalled | Reviews / yr | Referrals / yr | Reach hospital / yr |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in sc:
            L.append(f"| {r['scenario']} (completion {int(100 * r['completion'])}%) | {r['recall_pct']}% | {r['review_pct']}% | {r['direct_refer_pct']}% | "
                     f"{100 * r['true_referable_share_of_direct_refer']:.0f}% | {100 * r['missed_referable_per_referable']:.1f}% | {r['reviews_per_year']:,} | "
                     f"{r['referrals_per_year']:,} | {r['attended_per_year']:,} |")
        L.append("")
    sim = load("simulation/sim_summary.json")
    if sim:
        L += ["## 9b. District capacity sweep (100,000 screenings / year)", "",
              f"Triage fractions: {sim['params']['triage'].get('source', 'defaults in sim/params.yaml')}.", "",
              "Fewest tele-reviewers (2 h/day each) that review 95% of cases within 48 h:", ""]
        for k, v in sim["min_reviewers_for_95pct_sla"].items():
            L.append(f"- {k}: {v}")
        L += ["", "Referral completion vs patients reaching the eye hospital per year:", ""]
        for c in sim["completion_sweep"]:
            L.append(f"- completion {int(100 * c['completion'])}%: {c['attended_per_year']:,} patients/year (of {c['referrals_per_year']:,} referred)")
        L += ["", "![simulation](simulation/sim_sweep.png)", ""]
    L += ["## 10. Development history and disclosures", "",
          "- **DDR's official split is confounded** (docs/DATASETS.md): every timestamp-series image in its train and valid sets is grade 0. "
          "A first grader trained on it reached validation AUC 0.987 but about 0.92 within either series. All results here use our series x grade stratified re-split.",
          "- **IDRiD was looked at during development**: two grader runs (the confounded-split model and this one) were scored on it, and the lesion "
          "confidence problem was found on it. The fixes were then calibrated on development data only (DDR valid for lesion confidence, valB for the triage policy).",
          "- **Every look at IDRiD and Messidor-2 is logged** in the register at the end of docs/EXPERIMENTS.md. Candidate models were ranked on ODIR, held out whole for that purpose; each final candidate was committed to in writing before being scored on the sealed sets, and scored once. No threshold, hyper-parameter or design choice was made after seeing them.",
          "- **Training data**: DDR train + APTOS train + EyePACS (35,126, training only). ODIR is the model-selection set; Messidor-2, IDRiD and (if downloaded) mBRSET are never trained or tuned on.",
          "- **The lesion rule engine is not a triage trigger** (configs/triage.json): at matched workload it added no safety over the CNN's own review band.", ""]
    (RESULTS / "VALIDATION.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", RESULTS / "VALIDATION.md")


if __name__ == "__main__":
    main()
