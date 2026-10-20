"""Model fitting and validation helpers: Optuna elastic net, bootstrap samplers, .632+, calibration, DeLong, cutpoints."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

import fast_stabl as fs

BOOTSTRAP_SEED = 20260721  # primary series of class-stratified bootstrap samples
SECOND_SEED = 20261001  # replication series
FPR_GRID = np.linspace(0, 1, 8193)


# ----------------------------------------------------------------------------- elastic net
def optuna_en(X, y, groups, seed, trials=100, repeats=3, splits=5):
    """Elastic net (C, l1_ratio) tuned by Optuna TPE on a patient-grouped stratified 5-fold CV repeated `repeats` times
    (mean fold AUC), refitted on all rows. Groups keep bootstrap duplicates on one side of each split."""
    import optuna
    from sklearn.model_selection import StratifiedGroupKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X = np.asarray(X, float)
    prepared = []
    for r in range(repeats):
        for a, b in StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed + r).split(X, y, groups):
            if np.unique(y[a]).size < 2 or np.unique(y[b]).size < 2:
                continue
            m, s = fs._standardize(X[a])
            prepared.append(((X[a] - m) / s, y[a], (X[b] - m) / s, y[b]))

    def objective(trial):
        C, l1 = trial.suggest_float("C", 1e-3, 1e2, log=True), trial.suggest_float("l1_ratio", 0.0, 1.0)
        return float(np.mean([roc_auc_score(yb, fs.predict(fs.fit_logistic(Xa, ya, [C], l1)[0], Xb)) for Xa, ya, Xb, yb in prepared]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=15))
    study.enqueue_trial({"C": 1.0, "l1_ratio": 0.5})
    study.optimize(objective, n_trials=trials)
    return fit_en(X, y, study.best_params["C"], study.best_params["l1_ratio"]), study.best_params, float(study.best_value)


FINAL_SEEDS = (7, 11, 13, 17, 19)  # Optuna sampler seeds of the final model (05)


def tune_final(X, y, groups, trials=100):
    """Elastic net tuned exactly as the final model: one Optuna run per seed in FINAL_SEEDS (patient-grouped stratified
    5-fold CV repeated 10 times), the run with the best CV AUC kept. Returns (estimator, params, cv_auc, seed)."""
    fits = [optuna_en(X, y, groups, seed=s, trials=trials, repeats=10) + (s,) for s in FINAL_SEEDS]
    return max(fits, key=lambda f: f[2])


LR_C = np.logspace(-3, 3, 31)  # C grid of the log-loss tuning (sensitivity analysis, 05b_model_choice)
LR_SEED = 20261007


def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune_lr(X, y, groups, repeats=10, splits=5, seed=LR_SEED, l1_ratios=(0.0,)):
    """Penalized logistic regression tuned on the log-loss (sensitivity analysis of the tuning criterion): C on LR_C (and
    l1_ratio on `l1_ratios`) chosen by the lowest mean fold log-loss over patient-grouped stratified 5-fold CV repeated
    `repeats` times, refitted on all rows. Default: L2 penalty only. Returns (estimator, params, cv_logloss)."""
    from sklearn.model_selection import StratifiedGroupKFold

    X = np.asarray(X, float)
    loss, folds = np.zeros((len(l1_ratios), len(LR_C))), 0
    for r in range(repeats):
        for a, b in StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed + r).split(X, y, groups):
            if np.unique(y[a]).size < 2 or np.unique(y[b]).size < 2:
                continue
            m, s = fs._standardize(X[a])
            for i, l1 in enumerate(l1_ratios):
                coefs = fs.fit_logistic((X[a] - m) / s, y[a], LR_C, l1)
                with np.errstate(over="ignore"):  # extreme C of the grid: probabilities of 0 or 1, clipped in log_loss
                    loss[i] += [log_loss(y[b], fs.predict(c, (X[b] - m) / s)) for c in coefs]
            folds += 1
    i, j = np.unravel_index(int(np.argmin(loss)), loss.shape)
    C, l1 = float(LR_C[j]), float(l1_ratios[i])
    return fit_en(X, y, C, l1), {"C": C, "l1_ratio": l1}, float(loss[i, j] / folds)


def fit_en(X, y, C, l1_ratio):
    m, s = fs._standardize(np.asarray(X, float))
    return fs.Scaled(m, s, fs.fit_logistic((X - m) / s, y, [C], l1_ratio)[0])


class Logistic:
    """Unpenalized logistic regression (maximum likelihood): the clinical-only refits of 08. Fitted on standardized predictors;
    `coef` holds the coefficients on the original scale with the intercept last (as fast_stabl) and `se` their Wald standard
    errors from the observed information. Same predict_proba interface as the elastic nets."""

    def __init__(self, X, y):
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, float)
        m, s = fs._standardize(X)
        est = LogisticRegression(penalty=None, max_iter=10000, tol=1e-10).fit((X - m) / s, y)
        b = est.coef_[0] / s
        self.coef = np.r_[b, est.intercept_[0] - np.dot(b, m)]
        p = self.predict_proba(X)
        design = np.column_stack([X, np.ones(len(X))])
        information = (design * (p * (1 - p))[:, None]).T @ design
        self.se = np.sqrt(np.diag(np.linalg.pinv(information)))

    def predict_proba(self, X):
        return fs.predict(self.coef, np.asarray(X, float))


# ----------------------------------------------------------------------------- bootstrap
def stratified_bootstrap_indices(y, rng):
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    in_bag = np.concatenate([rng.choice(negative, size=len(negative), replace=True), rng.choice(positive, size=len(positive), replace=True)])
    rng.shuffle(in_bag)
    represented = np.zeros(len(y), dtype=bool)
    represented[np.unique(in_bag)] = True
    return in_bag, np.flatnonzero(~represented)


def paired_samples(y, n, seed=BOOTSTRAP_SEED):
    """Class-stratified bootstrap samples with >= 10 out-of-bag patients and both classes on each side."""
    rng = np.random.default_rng(seed)
    out, attempt = [], 0
    while len(out) < n:
        in_bag, oob = stratified_bootstrap_indices(y, rng)
        if len(oob) >= 10 and np.unique(y[in_bag]).size == 2 and np.unique(y[oob]).size == 2:
            out.append((attempt, in_bag, oob))
        attempt += 1
    return out


def auc_632plus(y_in, p_in, y_oob, p_oob):
    """.632+ estimator (Efron and Tibshirani, 1997) for the AUC: error = 1 - AUC, no-information error 0.5; the out-of-bag
    error is capped at the no-information error before the relative overfitting rate R and the weight w = .632/(1 - .368R)."""
    auc_in, auc_oob = float(roc_auc_score(y_in, p_in)), float(roc_auc_score(y_oob, p_oob))
    err_in, err_oob = 1.0 - auc_in, min(1.0 - auc_oob, 0.5)
    r = (err_oob - err_in) / (0.5 - err_in) if err_oob > err_in and err_in < 0.5 else 0.0
    w = 0.632 / (1.0 - 0.368 * r)
    return {"auc_in_bag": auc_in, "auc_oob": auc_oob, "relative_overfit": r, "weight_632plus": w,
            "auc_632plus": 1.0 - ((1.0 - w) * err_in + w * err_oob)}


def weighted_roc(y_in, p_in, y_oob, p_oob, weight):
    """.632+-weighted mean of the in-bag and out-of-bag ROC curves on a common FPR grid."""
    curve = np.zeros_like(FPR_GRID)
    for target, prob, fraction in ((y_in, p_in, 1.0 - weight), (y_oob, p_oob, weight)):
        fpr, tpr, _ = roc_curve(target, prob, drop_intermediate=False)
        curve += fraction * np.interp(FPR_GRID, fpr, tpr)
    return curve


def bootstrap_auc_interval(y, p, n, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if np.unique(y[s]).size == 2:
            values.append(roc_auc_score(y[s], p[s]))
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


# ----------------------------------------------------------------------------- calibration
def calibration_parameters(y, p):
    """Calibration-in-the-large (intercept a of y ~ a + offset(logit p)) and calibration slope (b of y ~ a + b logit p)."""
    from scipy.optimize import brentq
    from sklearn.linear_model import LogisticRegression

    lp = logit(np.clip(p, 1e-8, 1 - 1e-8))
    slope = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000).fit(lp.reshape(-1, 1), y).coef_[0, 0]
    in_the_large = brentq(lambda a: float(np.sum(y - 1.0 / (1.0 + np.exp(-(a + lp))))), -30.0, 30.0, xtol=1e-12)
    return float(in_the_large), float(slope)


def calibration_bins(y, p, n_bins=6):
    """Quantile groups: mean predicted probability, observed rate and 95% Wilson interval."""
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame.p, q=n_bins, labels=False, duplicates="drop")
    rows, z = [], 1.959963984540054
    for b, g in frame.groupby("bin", observed=True):
        n, e = len(g), int(g.y.sum())
        obs = e / n
        den = 1 + z ** 2 / n
        center = (obs + z ** 2 / (2 * n)) / den
        half = z * math.sqrt(obs * (1 - obs) / n + z ** 2 / (4 * n ** 2)) / den
        rows.append({"bin": int(b), "n": n, "events": e, "mean_predicted": float(g.p.mean()), "observed": obs,
                     "observed_ci_low": max(0.0, center - half), "observed_ci_high": min(1.0, center + half)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- DeLong
def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    ranks = np.zeros(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x))
    out[order] = ranks
    return out


def delong_test(y, a, b):
    """DeLong test for two correlated AUCs; returns (auc_a, auc_b, z, p)."""
    order = np.argsort(-np.asarray(y))
    m = int(np.sum(y))
    preds = np.vstack([np.asarray(a, float)[order], np.asarray(b, float)[order]])
    n = preds.shape[1] - m
    tx = np.array([_midrank(r[:m]) for r in preds])
    ty = np.array([_midrank(r[m:]) for r in preds])
    tz = np.array([_midrank(r) for r in preds])
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01, v10 = (tz[:, :m] - tx) / n, 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    zval = (aucs[0] - aucs[1]) / math.sqrt(var) if var > 0 else float("nan")
    return float(aucs[0]), float(aucs[1]), float(zval), float(2 * norm.sf(abs(zval))) if var > 0 else float("nan")


# ----------------------------------------------------------------------------- cutpoints
METRICS = ["sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "f1", "mcc"]


def metric_arrays(y, p, thresholds):
    thresholds = np.atleast_1d(thresholds).astype(float)
    classified = p[:, None] >= thresholds[None, :]
    tp = (classified & (y[:, None] == 1)).sum(0)
    fp = (classified & (y[:, None] == 0)).sum(0)
    fn = int(y.sum()) - tp
    tn = int((y == 0).sum()) - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        den = np.sqrt((tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn))
        return {"threshold": thresholds, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "sensitivity": sens, "specificity": spec,
                "ppv": np.divide(tp, tp + fp, out=np.full(len(tp), np.nan), where=(tp + fp) > 0),
                "npv": np.divide(tn, tn + fn, out=np.full(len(tp), np.nan), where=(tn + fn) > 0),
                "balanced_accuracy": (sens + spec) / 2,
                "f1": np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp)), where=(2 * tp + fp + fn) > 0),
                "mcc": np.divide(tp * tn - fp * fn, den, out=np.zeros(len(tp)), where=den > 0)}


def threshold_metrics(y, p, threshold):
    return {k: float(v[0]) for k, v in metric_arrays(y, p, [threshold]).items()}


def constrained_cutpoints(y, p, target=0.90):
    """Rule-out: maximal MCC with sensitivity >= target; rule-in: maximal MCC with specificity >= target (candidates: 0, the
    predicted probabilities, 1; ties broken by balanced accuracy, the constrained metric, then distance to 0.5)."""
    candidates = np.unique(np.r_[0.0, p, 1.0])
    table = metric_arrays(y, p, candidates)
    out = []
    for constrained in ("sensitivity", "specificity"):
        eligible = np.flatnonzero(table[constrained] >= target)
        keys = (-np.abs(candidates[eligible] - 0.5), table[constrained][eligible], table["balanced_accuracy"][eligible], table["mcc"][eligible])
        out.append(float(candidates[eligible[np.lexsort(keys)[-1]]]))
    return np.array(out)


def constrained_cutpoints_grid(y, p, step=0.0005, target=0.90):
    """constrained_cutpoints for large pooled samples (out-of-bag predictions pooled over bootstrap samples): the same
    rule and tie-breaks, with candidate thresholds on a fixed grid and counts from sorted predictions."""
    grid = np.round(np.arange(0.0, 1.0 + step / 2, step), 6)
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    tp = len(pos) - np.searchsorted(pos, grid, side="left")
    fp = len(neg) - np.searchsorted(neg, grid, side="left")
    fn, tn = len(pos) - tp, len(neg) - fp
    sens, spec = tp / len(pos), tn / len(neg)
    den = np.sqrt((tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp.astype(float) * tn - fp.astype(float) * fn, den, out=np.zeros(len(grid)), where=den > 0)
    table = {"sensitivity": sens, "specificity": spec, "balanced_accuracy": (sens + spec) / 2, "mcc": mcc}
    out = []
    for constrained in ("sensitivity", "specificity"):
        eligible = np.flatnonzero(table[constrained] >= target)
        keys = (-np.abs(grid[eligible] - 0.5), table[constrained][eligible], table["balanced_accuracy"][eligible], table["mcc"][eligible])
        out.append(float(grid[eligible[np.lexsort(keys)[-1]]]))
    return np.array(out)


def risk_groups(p, low, high):
    return np.where(p < low, 0, np.where(p >= high, 2, 1))
