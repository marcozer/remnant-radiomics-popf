#!/usr/bin/env python3
"""Comparison with published preoperative clinical scores and radioclinical refits, in the patients with complete score
covariates (main pancreatic duct diameter and pancreatic neck thickness).

MPD: diameter on the preoperative CT at the isthmus (reviewed; 1 mm when the duct is not resolved on CT; operative report when the
review was unsure); the operative-report
MPD is a sensitivity analysis (sensitivity_operative_mpd.csv).
Published equations, no refit: preoperative D-FRS (De Pastena 2023) and the 2025 DISPAIR-FRS research equation with
neck transection assumed for all patients. Refitted elastic nets (same Optuna tuning as the final model): k-rad + MPD
diameter and neck thickness; k-rad + DISPAIR-FRS covariates (neck thickness, MPD diameter, age with the published cubic
spline terms, male sex). Clinical-only refits of the D-FRS and DISPAIR-FRS covariates (rD-FRS, rDISPAIR-FRS) as unpenalized
logistic regressions (maximum likelihood, no tuning; odds ratios per unit with Wald 95% CIs in clinical_refit_coefficients.csv).
k-rad (the signature model) is the final model (fitted on all patients). Apparent AUCs with simple-bootstrap 95% CIs, calibration-in-the-large,
calibration slope and quantile calibration groups; DeLong tests on these apparent predictions are descriptive only
(k-rad and the refits are scored on the patients they were fitted to). Bootstrap .632+ (2,000 class-stratified samples of
the complete-case patients; ComBat and Optuna refitted in bag for every fitted model, k-rad included, with the signature
features fixed; published scores applied unchanged) and weighted ROC curves. Model comparisons: paired differences of the
.632+ AUC between k-rad and each other model over the same bootstrap samples (mean, 95% range, two-sided bootstrap p)."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402
import models  # noqa: E402

OUT = common.RESULTS / "clinical_comparison"
OPTUNA_SEEDS = (7, 11, 13, 17, 19)
APPARENT_SEED = 20260715
DISPAIR = ["neck_thickness", "mpd", "age", "age_spline_41", "age_spline_65", "age_spline_77", "male"]
RAD, SIGNATURE = common.model_name(), common.signature()
REFITS = {f"{RAD} + MPD/thickness": SIGNATURE + ["mpd", "neck_thickness"],
          f"{RAD} + DISPAIR covariates": SIGNATURE + DISPAIR,
          "rD-FRS": ["mpd", "neck_thickness"],
          "rDISPAIR-FRS": DISPAIR}
PUBLISHED = ("D-FRS (published)", "DISPAIR-FRS 2025 (published)")
LINEAR = ("rD-FRS", "rDISPAIR-FRS")  # clinical-only refits: unpenalized logistic regression
_D = {}


def complete_case():
    d = common.load()
    c = d.clinical
    keep = c[["mpd", "neck_thickness", "age", "male"]].notna().all(axis=1).to_numpy()
    covariates = pd.DataFrame({"mpd": c.mpd, "neck_thickness": c.neck_thickness, "age": c.age, "male": c.male.astype(float)}, index=d.ids)
    for knot in (41, 65, 77):
        covariates[f"age_spline_{knot}"] = np.maximum(covariates.age - knot, 0.0) ** 3
    return d, keep, covariates


def refit(name, X, y, trials):
    """Elastic net tuned as the final model (best of the Optuna seeds) or, for the clinical-only refits, the unpenalized logistic
    regression; returns the estimator and its tuning record."""
    if name in LINEAR:
        return models.Logistic(X, y), {"method": "unpenalized logistic regression"}
    fits = [models.optuna_en(X, y, np.arange(len(y)), seed=s, trials=trials, repeats=10) + (s,) for s in OPTUNA_SEEDS]
    est, params, cv, seed = max(fits, key=lambda f: f[2])
    return est, {"optuna_seed": seed, **params, "cv_auc_5x10": cv}


def coefficient_rows(model, source, est, features):
    """Coefficients of a clinical-only refit on the original scale (per mm, per year, male vs female) with Wald 95% CIs."""
    rows = []
    for j, term in enumerate(features + ["intercept"]):
        b, se = float(est.coef[j]), float(est.se[j])
        rows.append({"model": model, "mpd_source": source, "term": term, "coefficient": b, "se": se,
                     "odds_ratio": np.exp(b) if term != "intercept" else np.nan,
                     "ci_low": np.exp(b - 1.959964 * se) if term != "intercept" else np.nan,
                     "ci_high": np.exp(b + 1.959964 * se) if term != "intercept" else np.nan})
    return rows


def operative_sensitivity(d, cov, harmonized, rows, y, rad_pred, trials, coefficients):
    """Sensitivity analysis: the operative-report MPD (clinical.csv mpd_operative) in place of the CT MPD at the isthmus; published
    equations and refits as in the main analysis; apparent AUCs (95% CI, 5,000 bootstrap samples) and DeLong tests against the
    signature model. The coefficients of the clinical-only refits are appended to `coefficients`."""
    cov = cov.copy()
    cov["mpd"] = d.clinical.mpd_operative.to_numpy(float)
    preds = {}
    for name, features in REFITS.items():
        X = matrix(harmonized, cov, rows, features)
        est, _ = refit(name, X, y, trials)
        preds[name] = est.predict_proba(X)
        if name in LINEAR:
            coefficients += coefficient_rows(name, "operative-report MPD", est, features)
    preds["D-FRS (published)"], preds["DISPAIR-FRS 2025 (published)"] = published(cov.iloc[rows])
    out = []
    for position, (name, p) in enumerate(preds.items()):
        low, high = models.bootstrap_auc_interval(y, p, 5000, APPARENT_SEED + 2000 + position)
        out.append({"model": name, "auc_apparent": float(roc_auc_score(y, p)), "ci_low": low, "ci_high": high,
                    "delong_p_vs_signature": float(models.delong_test(y, rad_pred, p)[3])})
    pd.DataFrame(out).to_csv(OUT / "sensitivity_operative_mpd.csv", index=False)
    return out


def published(cov):
    dfrs = expit(-4.211 + 0.388 * cov.mpd + 0.131 * cov.neck_thickness)
    dispair = expit(-3.587 + 0.115 * cov.neck_thickness + 0.000932 * cov.age - 1.316e-5 * cov.age_spline_41 + 3.948e-5 * cov.age_spline_65
                    - 2.632e-5 * cov.age_spline_77 + 0.894 - 0.0407 * cov.mpd + 0.230 * cov.male)
    return dfrs.to_numpy(), dispair.to_numpy()


def matrix(harmonized, covariates, rows, features):
    frame = pd.concat([harmonized.reset_index(drop=True), covariates.reset_index(drop=True)], axis=1)
    return frame.iloc[rows][features].to_numpy(float)


def init_worker():
    import numba

    numba.set_num_threads(1)
    d, keep, cov = complete_case()
    _D.update(d=d, rows=np.flatnonzero(keep), cov=cov)


def job(b, attempt, in_bag, oob, trials):
    """in_bag / oob index the complete-case patients."""
    d, rows, cov = _D["d"], _D["rows"], _D["cov"]
    y = d.y[rows]
    tr, te = common.combat(d.radiomics, d.batch, rows[in_bag], rows[oob])
    out = []
    for name, features in {RAD: SIGNATURE, **REFITS}.items():
        Xin = pd.concat([tr.reset_index(drop=True), cov.iloc[rows[in_bag]].reset_index(drop=True)], axis=1)[features].to_numpy(float)
        Xoob = pd.concat([te.reset_index(drop=True), cov.iloc[rows[oob]].reset_index(drop=True)], axis=1)[features].to_numpy(float)
        if name in LINEAR:
            est, params = models.Logistic(Xin, y[in_bag]), {"C": np.nan, "l1_ratio": np.nan}
        else:
            est, params, _ = models.optuna_en(Xin, y[in_bag], in_bag, seed=1000 + b, trials=trials, repeats=3)
        p_in, p_oob = est.predict_proba(Xin), est.predict_proba(Xoob)
        r = models.auc_632plus(y[in_bag], p_in, y[oob], p_oob)
        out.append(({"model": name, "replicate": b, "attempt": attempt, "C": params["C"], "l1_ratio": params["l1_ratio"], **r},
                    models.weighted_roc(y[in_bag], p_in, y[oob], p_oob, r["weight_632plus"])))
    for name, score in zip(PUBLISHED, published(cov.iloc[rows])):
        r = models.auc_632plus(y[in_bag], score[in_bag], y[oob], score[oob])
        out.append(({"model": name, "replicate": b, "attempt": attempt, "C": np.nan, "l1_ratio": np.nan, **r},
                    models.weighted_roc(y[in_bag], score[in_bag], y[oob], score[oob], r["weight_632plus"])))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d, keep, cov = complete_case()
    rows = np.flatnonzero(keep)
    y = d.y[rows]
    harmonized = common.harmonize_all(d.radiomics, d.batch)[SIGNATURE]
    preds = {RAD: common.final_predictions(d)[rows]}
    tuning, coefficients = {}, []
    for name, features in REFITS.items():
        X = matrix(harmonized, cov, rows, features)
        est, tuning[name] = refit(name, X, y, args.trials)
        preds[name] = est.predict_proba(X)
        if name in LINEAR:
            coefficients += coefficient_rows(name, "CT MPD", est, features)
    dfrs, dispair = published(cov.iloc[rows])
    preds["D-FRS (published)"], preds["DISPAIR-FRS 2025 (published)"] = dfrs, dispair

    figure_models = [RAD, f"{RAD} + MPD/thickness", f"{RAD} + DISPAIR covariates", "D-FRS (published)", "DISPAIR-FRS 2025 (published)"]
    metrics, bins, curves = [], [], []
    for position, name in enumerate(figure_models + ["rD-FRS", "rDISPAIR-FRS"]):
        p = preds[name]
        seed = APPARENT_SEED + (100 + position if position < 5 else 900 + position - 4)
        low, high = models.bootstrap_auc_interval(y, p, 5000, seed)
        intercept, slope = models.calibration_parameters(y, p)
        metrics.append({"model": name, "n": int(len(y)), "events": int(y.sum()), "auc_apparent": float(roc_auc_score(y, p)), "ci_low": low,
                        "ci_high": high, "calibration_intercept": intercept, "calibration_slope": slope})
        fpr, tpr, _ = roc_curve(y, p)
        curves.append(pd.DataFrame({"model": name, "fpr": fpr, "tpr": tpr}))
        if name in figure_models[:3]:
            bins.append(models.calibration_bins(y, p).assign(model=name))
    delong = []
    names = figure_models + ["rD-FRS", "rDISPAIR-FRS"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            auc_a, auc_b, z, pval = models.delong_test(y, preds[a], preds[b])
            delong.append({"reference": a, "alternative": b, "auc_reference": auc_a, "auc_alternative": auc_b, "z": z, "p_value": pval})
    pd.DataFrame(metrics).to_csv(OUT / "apparent_metrics.csv", index=False)
    pd.DataFrame(delong).to_csv(OUT / "delong_apparent.csv", index=False)
    pd.concat(bins).to_csv(OUT / "calibration_groups.csv", index=False)
    pd.concat(curves).to_csv(OUT / "apparent_roc.csv", index=False)
    operative_sensitivity(d, cov, harmonized, rows, y, preds[RAD], args.trials, coefficients)
    pd.DataFrame(coefficients).to_csv(OUT / "clinical_refit_coefficients.csv", index=False)

    samples = models.paired_samples(y, args.bootstraps, models.BOOTSTRAP_SEED)
    started, reps, sums = time.time(), [], {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        futures = [pool.submit(job, b, attempt, ib, ob, args.trials) for b, (attempt, ib, ob) in enumerate(samples)]
        for n, f in enumerate(as_completed(futures), 1):
            for row, curve in f.result():
                reps.append(row)
                sums[row["model"]] = sums.get(row["model"], 0) + curve
            if n % 250 == 0:
                print(f"{n}/{len(futures)} bootstraps ({(time.time() - started) / 60:.1f} min)", flush=True)
    reps = pd.DataFrame(reps).sort_values(["model", "replicate"])
    reps.to_csv(OUT / "bootstrap_replicates.csv", index=False)
    roc = pd.DataFrame({"fpr": np.r_[0.0, models.FPR_GRID], **{m: np.r_[0.0, s / len(samples)] for m, s in sums.items()}})
    roc.to_csv(OUT / "bootstrap_weighted_roc.csv", index=False)
    apparent = {m["model"]: m["auc_apparent"] for m in metrics}
    validation = []
    for name in [RAD, *REFITS, *PUBLISHED]:
        g = reps[reps.model == name]
        validation.append({"model": name, "auc_apparent": apparent[name], "auc_632plus": float(g.auc_632plus.mean()),
                           "ci_low": float(g.auc_632plus.quantile(0.025)), "ci_high": float(g.auc_632plus.quantile(0.975)),
                           "mean_oob_auc": float(g.auc_oob.mean()), "weighted_roc_area": float(np.trapz(roc[name], roc.fpr))})
    pd.DataFrame(validation).to_csv(OUT / "bootstrap_632plus.csv", index=False)
    pivot = reps.pivot(index="replicate", columns="model", values="auc_632plus")
    comparisons = []
    for other in [*REFITS, *PUBLISHED]:
        diff = pivot[RAD] - pivot[other]
        comparisons.append({"reference": RAD, "comparator": other, "difference_632plus": float(diff.mean()),
                            "range_low": float(diff.quantile(0.025)), "range_high": float(diff.quantile(0.975)),
                            "bootstrap_p": float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))), "replicates": int(len(diff))})
    pd.DataFrame(comparisons).to_csv(OUT / "paired_632plus_differences.csv", index=False)
    summary = {"complete_case_patients": int(len(y)), "complete_case_events": int(y.sum()), "excluded_missing_covariates": int((~keep).sum()),
               "excluded_events": int(d.y[~keep].sum()), "refit_tuning": tuning, "bootstraps": args.bootstraps,
               "minutes": round((time.time() - started) / 60, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))
    print(pd.DataFrame(metrics).round(3).to_string(index=False))
    print(pd.DataFrame(validation).round(3).to_string(index=False))
    print(pd.DataFrame(comparisons).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
