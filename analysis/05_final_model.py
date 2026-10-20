#!/usr/bin/env python3
"""Final model on the signature (k-rad, signature of 03: the k most stable Stabl features, k chosen by inner cross-validation): ComBat by reconstruction protocol on all patients, elastic net tuned by Optuna (TPE, 100 trials,
patient-grouped stratified 5-fold CV repeated 10 times) with five sampler seeds, the seed with the best CV AUC kept,
refitted on all patients. Apparent AUC (simple bootstrap 95% CI), calibration intercept/slope, Brier score, coefficients."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402
import models  # noqa: E402

OUT = common.RESULTS / "final_model"
OPTUNA_SEEDS = models.FINAL_SEEDS


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = common.load()
    X = common.harmonize_all(d.radiomics, d.batch)[common.signature()].to_numpy(float)
    everyone = np.arange(len(d.y))
    fits = []
    for seed in OPTUNA_SEEDS:
        est, params, cv = models.optuna_en(X, d.y, everyone, seed=seed, trials=100, repeats=10)
        fits.append({"seed": seed, "params": params, "cv_auc": cv, "apparent_auc": float(roc_auc_score(d.y, est.predict_proba(X))), "est": est})
    best = max(fits, key=lambda f: f["cv_auc"])
    est = best["est"]
    p = est.predict_proba(X)
    intercept, slope = models.calibration_parameters(d.y, p)
    low, high = models.bootstrap_auc_interval(d.y, p, 5000, 20260815)
    spec = {"signature": common.signature(), "name": common.model_name(), "optuna_seed": best["seed"], "C": best["params"]["C"], "l1_ratio": best["params"]["l1_ratio"],
            "cv_auc_5x10": best["cv_auc"], "intercept": float(est.coef[-1]), "coefficients": est.coef[:-1].tolist(),
            "predictor_mean": est.mean.tolist(), "predictor_sd": est.scale.tolist()}
    summary = {"patients": int(len(d.y)), "events": int(d.y.sum()), **{k: spec[k] for k in ("optuna_seed", "C", "l1_ratio", "cv_auc_5x10")},
               "apparent_auc": float(roc_auc_score(d.y, p)), "apparent_auc_ci": [low, high], "calibration_intercept": intercept,
               "calibration_slope": slope, "brier": float(brier_score_loss(d.y, p)),
               "optuna_seeds": [{k: f[k] for k in ("seed", "params", "cv_auc", "apparent_auc")} for f in fits]}
    (OUT / "model.json").write_text(json.dumps(spec, indent=2) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
