"""Validation of the grader and the triage policy.

Reads runs/<grader>/preds.csv (predictions made once, after training) and,
if present, results/cache/rules_<seg>.csv (lesion-rule features for the
external images). Nothing here trains or tunes on a test set: the policy is
fitted on valB only, then frozen.

Reported per test set:
- referable-DR AUC (bootstrap CI, resampled by exam when exams are known)
- sensitivity / specificity at the frozen operating point (Wilson CIs)
- quadratic weighted kappa, expected calibration error
- ordinal prediction-set coverage (overall and per grade) and size
- triage split (recall / review / refer) and referable cases auto-recalled
- Messidor-2 at exam level with the IDx-DR definition of referable
  (moderate NPDR or worse and/or macular oedema), for comparison with
  Abramoff et al. 2016 (96.8% / 87.0%)
- recalibration under camera shift: Mondrian CRC with N audit labels

    python -m drishti.eval.grader_eval --grader b0_512 [--rules seg_r34]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ..config import RESULTS, RUNS
from ..conformal import triage as T
from . import metrics as M

PCOLS = [f"p{k}" for k in range(5)]
IDX_DR_2016 = {"sensitivity": 0.968, "specificity": 0.870, "source": "Abramoff et al., IOVS 2016, IDx-DR X2.1 on Messidor-2 (exam level)"}


def _ci(t):
    return [round(float(v), 4) for v in t]


def binary_block(y, pred):
    r = M.binary_report(y, pred)
    return {k: (_ci(v) if isinstance(v, tuple) else v) for k, v in r.items()}


def set_block(probs, y, qhat):
    sets = np.array([T.prediction_set(p, qhat) for p in probs])
    covered = (sets[:, 0] <= y) & (y <= sets[:, 1])
    size = sets[:, 1] - sets[:, 0] + 1
    per = {}
    for g in range(5):
        sel = y == g
        if sel.any():
            per[str(g)] = _ci(M.wilson(int(covered[sel].sum()), int(sel.sum()))) + [int(sel.sum())]
    return {"coverage": _ci(M.wilson(int(covered.sum()), len(y))), "mean_size": round(float(size.mean()), 3),
            "size_hist": {str(k): int((size == k).sum()) for k in range(1, 6)}, "per_grade": per}


def triage_block(df, pol, rules=None):
    ref = df.grade.to_numpy() >= 2
    out = {}
    lesion = np.zeros(len(df), bool)
    dme = np.zeros(len(df), bool)
    if rules is not None:
        lesion = df.image_id.map(rules.set_index("image_id")["lesion_flag"]).fillna(False).to_numpy(bool)
        dme = df.image_id.map(rules.set_index("image_id")["dme_flag"]).fillna(False).to_numpy(bool)
    outcomes = []
    for p, pr, lf, dm in zip(df.p_ref.to_numpy(), df[PCOLS].to_numpy(), lesion, dme):
        outcomes.append(T.triage(p, pr, pol, lesion_flag=bool(lf), dme_flag=bool(dm))["outcome"])
    outcomes = np.array(outcomes)
    n = len(df)
    for o in ("recall", "review", "refer"):
        out[f"{o}_fraction"] = round(float((outcomes == o).mean()), 4)
        # conditional rates let the district simulation re-mix to a real
        # screening population's prevalence instead of the dataset's
        out[f"{o}_given_referable"] = round(float((outcomes[ref] == o).mean()), 4) if ref.any() else None
        out[f"{o}_given_non_referable"] = round(float((outcomes[~ref] == o).mean()), 4) if (~ref).any() else None
    missed = int(((outcomes == "recall") & ref).sum())
    out["referable_auto_recalled"] = {"count": missed, "of_referable": int(ref.sum()),
                                      "rate": _ci(M.wilson(missed, int(ref.sum())))}
    fr = int(((outcomes == "refer") & ~ref).sum())
    out["non_referable_direct_referred"] = {"count": fr, "of_non_referable": int((~ref).sum()),
                                            "rate": _ci(M.wilson(fr, int((~ref).sum())))}
    out["n"] = n
    return out, outcomes


def evaluate_set(name, df, pol, rules=None):
    df = df[(df.grade >= 0) & (df.gradable == 1)]
    if "dup_of_dev" in df:
        df = df[df.dup_of_dev != 1]
    if len(df) == 0:
        return None
    y = df.grade.to_numpy()
    ref = y >= 2
    probs = df[PCOLS].to_numpy()
    groups = df.exam_id.fillna(df.image_id).to_numpy()
    res = {"n_images": int(len(df)), "grade_counts": {str(g): int((y == g).sum()) for g in range(5)}}
    res["auc_referable"] = _ci(M.auc_ci(ref, df.p_ref, groups=groups))
    res["at_operating_point"] = binary_block(ref, df.p_ref.to_numpy() >= pol.lambda_op)
    res["meets_ps_target"] = {"sensitivity_gt_90": res["at_operating_point"]["sensitivity"][0] > 0.90,
                              "specificity_gt_85": res["at_operating_point"]["specificity"][0] > 0.85,
                              "ci_lower_sens_gt_90": res["at_operating_point"]["sensitivity"][1] > 0.90}
    res["qwk"] = round(M.qwk(y, M.expected_grade(probs)), 4)
    res["ece_referable"] = round(M.ece(ref, df.p_ref), 4)
    res["prediction_sets"] = set_block(probs, y, pol.qhat)
    tri, _ = triage_block(df, pol)
    res["triage_cnn_only"] = tri
    if rules is not None and df.image_id.isin(rules.image_id).any():
        tri_r, _ = triage_block(df, pol, rules)
        res["triage_with_rules"] = tri_r
    return res


def messidor_exam_level(df, pol, rules=None):
    """IDx-DR comparison: exam = two eyes; referable = any eye grade >= 2 or DME.
    An exam with an ungradable eye is scored on its gradable eye."""
    df = df[(df.source == "messidor2") & (df.gradable == 1) & (df.grade >= 0) & df.exam_id.notna()]
    if df.empty:
        return None
    dme_rule = None
    if rules is not None:
        dme_rule = df.image_id.map(rules.set_index("image_id")["dme_flag"]).fillna(False)
    g = df.assign(ref=(df.grade >= 2) | (df.dme == 1), dme_rule=dme_rule if dme_rule is not None else False)
    ex = g.groupby("exam_id").agg(ref=("ref", "max"), p=("p_ref", "max"), dme_rule=("dme_rule", "max"))
    out = {"n_exams": int(len(ex)), "referable_exams": int(ex.ref.sum()), "reference": IDX_DR_2016}
    out["auc_cnn"] = _ci(M.auc_ci(ex.ref, ex.p))
    out["cnn_only"] = binary_block(ex.ref, ex.p >= pol.lambda_op)
    if rules is not None:
        out["cnn_plus_dme_rule"] = binary_block(ex.ref, (ex.p >= pol.lambda_op) | ex.dme_rule.astype(bool))
    return out


def recalibration_demo(df, pol, alpha_miss, sens_target=0.95, n_labels=(0, 50, 100, 200, 300), repeats=300, seed=0):
    """Deploy on a new camera: first N cases get audit labels (random sample,
    including would-be recalls), lambda_low is re-fitted by CRC on those, and
    the remaining cases are scored. N=0 means the valB lambda is used."""
    df = df[(df.grade >= 0) & (df.gradable == 1)]
    if "dup_of_dev" in df:
        df = df[df.dup_of_dev != 1]
    p = df.p_ref.to_numpy()
    ref = df.grade.to_numpy() >= 2
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_labels:
        if n >= len(df) - 50:
            continue
        fnr, recall_frac, sens, spec = [], [], [], []
        for _ in range(repeats):
            idx = rng.permutation(len(df))
            cal, test = idx[:n], idx[n:]
            if n == 0:
                lam, lam_op = pol.lambda_low, pol.lambda_op
            else:
                lam = T.recalibrate_domain(p[cal], np.where(ref[cal], 2, 0), alpha_miss)
                # site operating point: CRC for >= (1 - (1 - sens target)) sensitivity on the audit sample;
                # falls back to the frozen point if too few referable cases were audited
                lam_op = T.crc_lambda_low(p[cal], ref[cal], 1 - sens_target) if ref[cal].sum() >= 1 / (1 - sens_target) else pol.lambda_op
            tr = ref[test]
            fnr.append(float(np.mean(p[test][tr] < lam)) if tr.any() else 0.0)
            recall_frac.append(float(np.mean(p[test] < lam)))
            sens.append(float(np.mean(p[test][tr] >= lam_op)) if tr.any() else np.nan)
            spec.append(float(np.mean(p[test][~tr] < lam_op)) if (~tr).any() else np.nan)
        rows.append({"audit_labels": n, "mean_missed_referable_rate": round(float(np.mean(fnr)), 4),
                     "p90_missed_referable_rate": round(float(np.percentile(fnr, 90)), 4),
                     "mean_auto_recall_fraction": round(float(np.mean(recall_frac)), 4),
                     "op_sensitivity_mean": round(float(np.nanmean(sens)), 4),
                     "op_sensitivity_p10": round(float(np.nanpercentile(sens, 10)), 4),
                     "op_specificity_mean": round(float(np.nanmean(spec)), 4),
                     "op_specificity_p10": round(float(np.nanpercentile(spec, 10)), 4)})
    # conditional triage rates under the site-recalibrated policy (N = 200 audit
    # labels, lambda_low and lambda_high both re-fitted), for the district simulation
    n = min(200, len(df) - 60)
    acc = {f"{o}_given_{c}": [] for o in ("recall", "review", "refer") for c in ("referable", "non_referable")}
    for _ in range(repeats):
        idx = rng.permutation(len(df))
        cal, test = idx[:n], idx[n:]
        lo = T.recalibrate_domain(p[cal], np.where(ref[cal], 2, 0), alpha_miss)
        hi = max(lo, T.crc_lambda_high(p[cal], ref[cal], pol.beta_false))
        out = np.where(p[test] >= hi, "refer", np.where(p[test] < lo, "recall", "review"))
        tr = ref[test]
        for o in ("recall", "review", "refer"):
            acc[f"{o}_given_referable"].append(float(np.mean(out[tr] == o)))
            acc[f"{o}_given_non_referable"].append(float(np.mean(out[~tr] == o)))
    site = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    site["audit_labels"] = n
    return {"alpha_miss_target": alpha_miss, "sens_target": sens_target, "rows": rows, "site_recalibrated_triage": site}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grader", default="b0_512")
    ap.add_argument("--rules", default="")
    ap.add_argument("--alpha-miss", type=float, default=0.03)
    ap.add_argument("--beta-false", type=float, default=0.02)
    ap.add_argument("--alpha-set", type=float, default=0.10)
    ap.add_argument("--sens-target", type=float, default=0.95)
    ap.add_argument("--policy-only", action="store_true",
                    help="fit and write the triage policy on valB, then stop before any test set is "
                         "scored: prepares a candidate for export without spending a look")
    args = ap.parse_args()

    preds = pd.read_csv(RUNS / args.grader / "preds.csv")
    rules = None
    if args.rules:
        rp = RESULTS / "cache" / f"rules_{args.rules}.csv"
        rules = pd.read_csv(rp) if rp.exists() else None

    cal = preds[(preds.split == "valB") & (preds.grade >= 0) & (preds.gradable == 1)]
    pol = T.fit_policy(cal.p_ref.to_numpy(), cal.grade.to_numpy(), cal[PCOLS].to_numpy(),
                       alpha_miss=args.alpha_miss, beta_false=args.beta_false, alpha_set=args.alpha_set,
                       sens_target=args.sens_target, calibrated_on="valB (DDR valid half + APTOS 7.5%)")
    (RUNS / args.grader / "policy.json").write_text(json.dumps(pol.as_dict(), indent=2))
    print("policy:", json.dumps(pol.as_dict()))
    if args.policy_only:
        return

    report = {"grader": args.grader, "policy": pol.as_dict(), "sets": {}}
    tests = {
        "valB (calibration, for reference)": cal,
        "DDR test (internal)": preds[(preds.source == "ddr") & (preds.split == "test")],
        "APTOS test (internal)": preds[(preds.source == "aptos") & (preds.split == "test")],
        # held out whole, never trained on: our cross-camera signal for
        # choosing between candidates without spending a look at IDRiD or
        # Messidor-2. Noisier labels, so it ranks models, it does not rate them.
        "ODIR (held-out, China, mixed cameras)": preds[preds.source == "odir"],
        "IDRiD all 516 (external, India, Kowa)": preds[preds.source == "idrid"],
        "Messidor-2 (external, France, Topcon)": preds[preds.source == "messidor2"],
        "mBRSET (external, Brazil, handheld)": preds[preds.source == "mbrset"],
    }
    for name, d in tests.items():
        r = evaluate_set(name, d, pol, rules)
        if r is not None:
            report["sets"][name] = r
            op = r["at_operating_point"]
            print(f"{name:42s} n={r['n_images']:5d} AUC={r['auc_referable'][0]:.3f} "
                  f"sens={op['sensitivity'][0]:.3f} ({op['sensitivity'][1]:.3f}-{op['sensitivity'][2]:.3f}) "
                  f"spec={op['specificity'][0]:.3f} ({op['specificity'][1]:.3f}-{op['specificity'][2]:.3f}) "
                  f"QWK={r['qwk']:.3f} cover={r['prediction_sets']['coverage'][0]:.3f}")
    m2 = messidor_exam_level(preds, pol, rules)
    if m2:
        report["messidor2_exam_level"] = m2
    report["recalibration"] = {}
    for name in ("IDRiD all 516 (external, India, Kowa)", "Messidor-2 (external, France, Topcon)", "mBRSET (external, Brazil, handheld)"):
        d = tests[name]
        if len(d[(d.grade >= 0)]) > 150:
            report["recalibration"][name] = recalibration_demo(d, pol, args.alpha_miss, args.sens_target)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"eval_{args.grader}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
