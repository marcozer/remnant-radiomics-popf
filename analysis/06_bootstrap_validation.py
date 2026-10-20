#!/usr/bin/env python3
"""Bootstrap validation of the signature (k-rad): 2,000 class-stratified bootstrap samples (primary series, seed
20260721) and 2,000 more (replication series, seed 20261001). In every sample, ComBat, scaling and the tuning of the
elastic net are refitted on the in-bag patients exactly as for the final model (models.tune_final: best of five Optuna
runs, 100 trials each, patient-grouped 5-fold CV x 10); the panel is fixed.
Per sample: .632+ AUC, .632+-weighted in-bag/out-of-bag ROC curve, calibration-slope optimism (Harrell), and constrained
MCC cutpoints chosen in bag (rule-out: sensitivity >= 90%; rule-in: specificity >= 90%) applied unchanged out of bag.
Cutpoints are also chosen on the out-of-bag predictions pooled over the 2,000 primary samples (each patient predicted
only by models that did not see it): the same rule, applied to predictions for unseen patients. Risk strata apply each set
of cutpoints to the final model."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402
import models  # noqa: E402

OUT = common.RESULTS / "bootstrap_validation"
STRATEGIES = ("Rule-out", "Rule-in")
_D = {}


def init_worker():
    import numba

    numba.set_num_threads(1)
    _D["d"] = common.load()


def job(series, b, attempt, in_bag, oob, trials):
    d = _D["d"]
    y = d.y
    h = common.Harmonizer()
    tr = h.fit_transform(d.radiomics.iloc[in_bag], d.batch[in_bag])[common.signature()].to_numpy(float)
    te = h.transform(d.radiomics.iloc[oob], d.batch[oob])[common.signature()].to_numpy(float)
    full = h.transform(d.radiomics, d.batch)[common.signature()].to_numpy(float)
    est, params, _, _ = models.tune_final(tr, y[in_bag], in_bag, trials=trials)
    p_in, p_oob, p_full = est.predict_proba(tr), est.predict_proba(te), est.predict_proba(full)
    r = models.auc_632plus(y[in_bag], p_in, y[oob], p_oob)
    row = {"series": series, "replicate": b, "attempt": attempt, "n_oob": len(oob), "events_oob": int(y[oob].sum()),
           "C": params["C"], "l1_ratio": params["l1_ratio"], **r,
           "slope_in_bag": models.calibration_parameters(y[in_bag], p_in)[1], "slope_original": models.calibration_parameters(y, p_full)[1]}
    for strategy, threshold in zip(STRATEGIES, models.constrained_cutpoints(y[in_bag], p_in)):
        row[f"{strategy}_threshold"] = threshold
        for k, v in models.threshold_metrics(y[in_bag], p_in, threshold).items():
            if k in ("sensitivity", "specificity", "mcc"):
                row[f"{strategy}_in_bag_{k}"] = v
        for k, v in models.threshold_metrics(y[oob], p_oob, threshold).items():
            if k in models.METRICS:
                row[f"{strategy}_oob_{k}"] = v
    return row, models.weighted_roc(y[in_bag], p_in, y[oob], p_oob, r["weight_632plus"]), (oob, p_oob)


def interval(values):
    values = pd.Series(values).dropna()
    return [float(values.quantile(0.025)), float(values.quantile(0.975))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = common.load()
    p = common.final_predictions(d)
    y = d.y
    series = {"primary": models.paired_samples(y, args.bootstraps, models.BOOTSTRAP_SEED),
              "replication": models.paired_samples(y, args.bootstraps, models.SECOND_SEED)}
    started, rows, curves, pooled = time.time(), [], {k: np.zeros_like(models.FPR_GRID) for k in series}, []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        futures = [pool.submit(job, s, b, attempt, ib, ob, args.trials) for s, samples in series.items() for b, (attempt, ib, ob) in enumerate(samples)]
        for n, f in enumerate(as_completed(futures), 1):
            row, curve, (oob, p_oob) = f.result()
            rows.append(row)
            curves[row["series"]] += curve
            if row["series"] == "primary":
                pooled.append((row["replicate"], oob, p_oob))
            if n % 500 == 0:
                print(f"{n}/{len(futures)} bootstraps ({(time.time() - started) / 60:.1f} min)", flush=True)
    reps = pd.DataFrame(rows).sort_values(["series", "replicate"])
    reps.to_csv(OUT / "replicates.csv", index=False)
    roc = pd.DataFrame({"fpr": np.r_[0.0, models.FPR_GRID]})
    for s, samples in series.items():
        roc[f"tpr_{s}"] = np.r_[0.0, curves[s] / len(samples)]
    roc.to_csv(OUT / "weighted_roc.csv", index=False)

    summary = {"patients": int(len(y)), "events": int(y.sum()), "bootstraps_per_series": args.bootstraps, "optuna_trials": args.trials}
    for s in series:
        g = reps[reps.series == s]
        area = float(np.trapz(roc[f"tpr_{s}"], roc.fpr))
        summary[f"auc_632plus_{s}"] = {"auc": float(g.auc_632plus.mean()), "ci": interval(g.auc_632plus), "mean_oob_auc": float(g.auc_oob.mean()),
                                       "mean_in_bag_auc": float(g.auc_in_bag.mean()), "weighted_roc_area": area}
        assert abs(area - g.auc_632plus.mean()) < 2e-4
    summary["auc_632plus_pooled"] = {"auc": float(reps.auc_632plus.mean()), "ci": interval(reps.auc_632plus), "mean_oob_auc": float(reps.auc_oob.mean())}
    g = reps[reps.series == "primary"]
    final_intercept, final_slope = models.calibration_parameters(y, p)
    corrected = final_slope - (g.slope_in_bag - g.slope_original)
    summary["calibration"] = {"apparent_intercept": final_intercept, "apparent_slope": final_slope,
                              "optimism_corrected_slope": float(corrected.mean()), "optimism_corrected_slope_ci": interval(corrected)}

    full = models.constrained_cutpoints(y, p)
    medians = np.array([g[f"{s}_threshold"].median() for s in STRATEGIES])
    pooled.sort(key=lambda item: item[0])
    y_pool = np.concatenate([y[oob] for _, oob, _ in pooled])
    p_pool = np.concatenate([p_oob for _, _, p_oob in pooled])
    out_of_bag = models.constrained_cutpoints_grid(y_pool, p_pool)
    cut = {"full_cohort": dict(zip(STRATEGIES, full.tolist())), "bootstrap_median": dict(zip(STRATEGIES, medians.tolist())),
           "oob_pooled": dict(zip(STRATEGIES, out_of_bag.tolist())), "bootstrap_threshold_ci": {s: interval(g[f"{s}_threshold"]) for s in STRATEGIES},
           "oob_pooled_derivation_metrics": {s: models.threshold_metrics(y_pool, p_pool, t) for s, t in zip(STRATEGIES, out_of_bag)},
           "oob_pooled_predictions": int(len(p_pool))}
    for name, (low, high) in (("full_cohort", full), ("bootstrap_median", medians), ("oob_pooled", out_of_bag)):
        groups = models.risk_groups(p, low, high)
        cut[f"strata_{name}"] = [{"group": label, "n": int((groups == k).sum()), "events": int(y[groups == k].sum()),
                                  "p_min": float(p[groups == k].min()), "p_max": float(p[groups == k].max())}
                                 for k, label in enumerate(("Low", "Intermediate", "High"))]
        cut[f"apparent_metrics_{name}"] = {s: models.threshold_metrics(y, p, t) for s, t in zip(STRATEGIES, (low, high))}
    cut["oob"] = {s: {m: {"mean": float(g[f"{s}_oob_{m}"].mean()), "median": float(g[f"{s}_oob_{m}"].median()), "ci": interval(g[f"{s}_oob_{m}"])}
                      for m in models.METRICS} for s in STRATEGIES}
    cut["in_bag_target_met"] = {"Rule-out": float((g["Rule-out_in_bag_sensitivity"] >= 0.9).mean()),
                                "Rule-in": float((g["Rule-in_in_bag_specificity"] >= 0.9).mean())}
    summary["cutpoints"] = cut
    summary["minutes"] = round((time.time() - started) / 60, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cutpoints"}, indent=1))
    print(json.dumps({k: cut[k] for k in ("full_cohort", "bootstrap_median", "oob_pooled", "strata_full_cohort", "strata_bootstrap_median",
                                          "strata_oob_pooled")}, indent=1))


if __name__ == "__main__":
    main()
