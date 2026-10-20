#!/usr/bin/env python3
"""Sensitivity analysis of the tuning criterion of the final model (k-rad): nested cross-validation (outer stratified 5-fold
x 20 repeats, ComBat refitted on each outer training fold, tuning on the outer training fold only) of the elastic net tuned
by Optuna on the cross-validated AUC (models.tune_final, the final model), the elastic net tuned on the cross-validated
log-loss (l1_ratio 0, 0.1, 0.25, 0.5, 0.75 or 1; models.tune_lr) and the L2-penalized logistic regression tuned on the
log-loss (l1_ratio 0). Per repeat (out-of-fold predictions pooled over its 5 folds): log-loss, Brier score, AUC and
calibration slope; paired differences with the elastic net tuned on the AUC (mean, SD over repeats, share of repeats with
a lower log-loss)."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402
import models  # noqa: E402

OUT = common.RESULTS / "model_choice"
OUTER_SEED = 20261008
EN_L1 = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
REFERENCE = "EN (Optuna, AUC)"
_D = {}


def init_worker():
    import numba

    numba.set_num_threads(1)
    _D["d"] = common.load()


def job(k, train, test):
    d = _D["d"]
    tr, te = common.combat(d.radiomics, d.batch, train, test)
    Xtr, Xte, ytr = tr[common.signature()].to_numpy(float), te[common.signature()].to_numpy(float), d.y[train]
    groups = np.arange(len(train))
    out = {}
    est, params, _ = models.tune_lr(Xtr, ytr, groups)
    out["L2-LR (log-loss)"] = (est.predict_proba(Xte).tolist(), params)
    est, params, _, _ = models.tune_final(Xtr, ytr, groups)
    out[REFERENCE] = (est.predict_proba(Xte).tolist(), params)
    est, params, _ = models.tune_lr(Xtr, ytr, groups, l1_ratios=EN_L1)
    out["EN (log-loss)"] = (est.predict_proba(Xte).tolist(), params)
    return k, test, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    y = common.load().y
    outer = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=OUTER_SEED).split(np.zeros(len(y)), y))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        res = list(pool.map(job, range(len(outer)), [o[0] for o in outer], [o[1] for o in outer]))
    names = list(res[0][2])
    per_repeat = {n: [] for n in names}
    for r in range(args.repeats):
        ks = range(5 * r, 5 * r + 5)
        idx = np.concatenate([res[k][1] for k in ks])
        for n in names:
            p = np.concatenate([res[k][2][n][0] for k in ks])
            per_repeat[n].append({"log_loss": models.log_loss(y[idx], p), "brier": float(np.mean((p - y[idx]) ** 2)),
                                  "auc": float(roc_auc_score(y[idx], p)), "slope": models.calibration_parameters(y[idx], p)[1]})
    reference = np.array([m["log_loss"] for m in per_repeat[REFERENCE]])
    table = {}
    for n in names:
        rows = per_repeat[n]
        ll = np.array([m["log_loss"] for m in rows])
        params = [res[k][2][n][1] for k in range(len(res))]
        Cs, l1s = np.array([p["C"] for p in params]), np.array([p["l1_ratio"] for p in params])
        table[n] = {**{m: float(np.mean([row[m] for row in rows])) for m in ("log_loss", "brier", "auc", "slope")},
                    "auc_sd": float(np.std([row["auc"] for row in rows], ddof=1)),
                    "log_loss_difference_vs_reference": float(np.mean(ll - reference)), "log_loss_difference_sd": float(np.std(ll - reference, ddof=1)),
                    "share_of_repeats_with_lower_log_loss": float(np.mean(ll < reference)),
                    "C_median": float(np.median(Cs)), "C_range": [float(Cs.min()), float(Cs.max())],
                    "l1_ratio_median": float(np.median(l1s)), "folds_with_l1_ratio_0": float(np.mean(l1s == 0))}
    summary = {"design": f"outer stratified 5-fold x {args.repeats} (seed {OUTER_SEED}), ComBat and tuning in the outer training folds",
               "reference": REFERENCE, "models": table, "minutes": round((time.time() - started) / 60, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for n, row in table.items():
        print(f"{n:22s} log-loss {row['log_loss']:.4f} (diff {row['log_loss_difference_vs_reference']:+.4f}, lower in "
              f"{100 * row['share_of_repeats_with_lower_log_loss']:.0f}% of repeats) | Brier {row['brier']:.4f} | AUC {row['auc']:.4f} | "
              f"slope {row['slope']:.2f} | C {row['C_median']:.3g} | l1 {row['l1_ratio_median']:.2f}")


if __name__ == "__main__":
    main()
