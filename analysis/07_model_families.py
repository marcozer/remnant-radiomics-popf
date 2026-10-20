#!/usr/bin/env python3
"""Model families on the signature (k-rad), bootstrap .632+ on the 2,000 primary bootstrap samples (paired).
In every sample ComBat is refitted in bag and each family is tuned in bag (fixed grids, patient-grouped stratified
3-fold CV, mean AUC) then refitted; the elastic-net row is the primary analysis (Optuna in bag, 06_bootstrap_validation).
Apparent AUCs: each family tuned (stratified 3-fold CV) and fitted on all patients."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402
import models  # noqa: E402

OUT = common.RESULTS / "model_families"
MODEL_SEED = 20260616
_D = {}


def factories():
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def scaled(estimator):
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])

    return {
        "L2-LR": (lambda p, s: scaled(LogisticRegression(penalty="l2", C=float(p["C"]), solver="lbfgs", max_iter=10000, random_state=s)),
                  list(ParameterGrid({"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}))),
        "SVM": (lambda p, s: scaled(SVC(C=float(p["C"]), gamma=p["gamma"], kernel="rbf", probability=True, random_state=s)),
                list(ParameterGrid({"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.1, 1.0]}))),
        "RF": (lambda p, s: RandomForestClassifier(n_estimators=300, max_depth=p["max_depth"], min_samples_leaf=int(p["min_samples_leaf"]),
                                                   random_state=s, n_jobs=1),
               list(ParameterGrid({"max_depth": [2, 3, 5, None], "min_samples_leaf": [2, 5, 10]}))),
        "XGB": (lambda p, s: xgb.XGBClassifier(n_estimators=200, max_depth=int(p["max_depth"]), learning_rate=float(p["learning_rate"]), subsample=0.8,
                                               colsample_bytree=0.8, eval_metric="logloss", random_state=s, n_jobs=1),
                list(ParameterGrid({"max_depth": [1, 2, 3], "learning_rate": [0.03, 0.1]}))),
        "LGBM": (lambda p, s: lgb.LGBMClassifier(n_estimators=200, num_leaves=int(p["num_leaves"]), min_child_samples=int(p["min_child_samples"]),
                                                 learning_rate=0.05, verbosity=-1, random_state=s, n_jobs=1),
                 list(ParameterGrid({"num_leaves": [3, 7, 15], "min_child_samples": [10, 20, 30]}))),
    }


def tune(factory, grid, X, y, groups, seed):
    """Patient-grouped stratified 3-fold CV (mean AUC; first best grid point kept), then refit on all rows."""
    folds = [(a, b) for a, b in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=MODEL_SEED).split(X, y, groups)
             if np.unique(y[a]).size == 2 and np.unique(y[b]).size == 2]
    best, best_auc = None, -math.inf
    for params in grid:
        auc = float(np.mean([roc_auc_score(y[b], factory(params, MODEL_SEED).fit(X[a], y[a]).predict_proba(X[b])[:, 1]) for a, b in folds]))
        if auc > best_auc:
            best, best_auc = params, auc
    return factory(best, seed).fit(X, y), best, best_auc


def init_worker():
    warnings.filterwarnings("ignore")
    _D["d"] = common.load()
    _D["f"] = factories()


def job(b, attempt, in_bag, oob):
    d, y = _D["d"], _D["d"].y
    tr, te = common.combat(d.radiomics, d.batch, in_bag, oob)
    Xin, Xoob = tr[common.signature()].to_numpy(float), te[common.signature()].to_numpy(float)
    rows = []
    for name, (factory, grid) in _D["f"].items():
        est, params, _ = tune(factory, grid, Xin, y[in_bag], in_bag, MODEL_SEED + attempt + 1000)
        r = models.auc_632plus(y[in_bag], est.predict_proba(Xin)[:, 1], y[oob], est.predict_proba(Xoob)[:, 1])
        rows.append({"model": name, "replicate": b, "attempt": attempt, "params": json.dumps(params, sort_keys=True, default=str), **r})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    d = common.load()
    y = d.y
    samples = models.paired_samples(y, args.bootstraps, models.BOOTSTRAP_SEED)
    started, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        futures = [pool.submit(job, b, attempt, ib, ob) for b, (attempt, ib, ob) in enumerate(samples)]
        for n, f in enumerate(as_completed(futures), 1):
            rows += f.result()
            if n % 250 == 0:
                print(f"{n}/{len(futures)} bootstraps ({(time.time() - started) / 60:.1f} min)", flush=True)
    reps = pd.DataFrame(rows)
    primary = pd.read_csv(common.RESULTS / "bootstrap_validation/replicates.csv", float_precision="round_trip").query("series == 'primary'")
    primary = primary[primary.replicate < args.bootstraps]
    reps = pd.concat([reps, primary.assign(model="EN")[["model", "replicate", "attempt", "auc_in_bag", "auc_oob", "relative_overfit",
                                                         "weight_632plus", "auc_632plus"]]], ignore_index=True)
    reps.sort_values(["model", "replicate"]).to_csv(OUT / "replicates.csv", index=False)
    H = common.harmonize_all(d.radiomics, d.batch)[common.signature()].to_numpy(float)
    apparent = {"EN": float(roc_auc_score(y, common.final_predictions(d)))}
    tuned = {}
    for name, (factory, grid) in factories().items():
        from sklearn.model_selection import StratifiedKFold

        folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=MODEL_SEED).split(H, y))
        scores = [np.mean([roc_auc_score(y[b], factory(p, MODEL_SEED).fit(H[a], y[a]).predict_proba(H[b])[:, 1]) for a, b in folds]) for p in grid]
        params = grid[int(np.argmax(scores))]
        apparent[name] = float(roc_auc_score(y, factory(params, MODEL_SEED).fit(H, y).predict_proba(H)[:, 1]))
        tuned[name] = params
    pivot = reps.pivot(index="replicate", columns="model", values="auc_632plus")
    table = []
    for name in ["EN", "L2-LR", "SVM", "RF", "XGB", "LGBM"]:
        diff = pivot[name] - pivot["EN"]
        table.append({"model": name, "apparent_auc": apparent[name], "auc_632plus": float(pivot[name].mean()),
                      "ci_low": float(pivot[name].quantile(0.025)), "ci_high": float(pivot[name].quantile(0.975)),
                      "mean_oob_auc": float(reps[reps.model == name].auc_oob.mean()), "delta_vs_en": float(diff.mean()),
                      "delta_ci": [float(diff.quantile(0.025)), float(diff.quantile(0.975))],
                      "full_cohort_hyperparameters": tuned.get(name, "final model")})
    pd.DataFrame(table).to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"bootstraps": args.bootstraps, "models": table, "minutes": round((time.time() - started) / 60, 1)},
                                                 indent=2, default=str) + "\n")
    print(pd.DataFrame(table).drop(columns=["delta_ci", "full_cohort_hyperparameters"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
