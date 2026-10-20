"""Stabl and penalized logistic regression compiled with numba (checked against the official Stabl 1.0.0).

For each run seed (42 + 1000 k): automatic C grid linspace(min_C, 100 min_C, 30); random-permutation decoys (as many
as real features); 1,000 class-balanced subsamples of 75% of the patients without replacement; for every C and
subsample an L1 logistic regression (class-balanced, penalized intercept, as liblinear) with selection |coef| >= 1e-5;
stability score = max over C of the selection frequency. weakness w gives the randomized LASSO (column weights
~ U[w, 1], new draw per subsample). The penalized logistic path is solved by a proximal-Newton coordinate descent
compiled with numba. The same solver fits the elastic-net models (scikit-learn LogisticRegression parametrization).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

import stabl_official

SELECTION_THRESHOLD = 1e-5
FAST = {"reassoc", "contract", "arcp", "nsz"}


# ----------------------------------------------------------------------------- compiled solver
@njit(cache=True, inline="always", fastmath=FAST)
def _sigmoid(t):
    if t >= 0.0:
        e = np.exp(-t)
        return 1.0 / (1.0 + e)
    e = np.exp(t)
    return e / (1.0 + e)


@njit(cache=True, inline="always", fastmath=FAST)
def _log1pexp(t):
    if t > 0.0:
        return t + np.log1p(np.exp(-t))
    return np.log1p(np.exp(t))


@njit(cache=True, fastmath=FAST)
def logistic_path(XT, y, cw, lambdas, l1_ratio, penalize_last, tol, coef_out):
    """Penalized logistic regression path.

    For each lambda (in the given order, warm-started), minimizes
        sum_i cw_i * logloss_i(w) + lambda * sum_j pen_j * (l1_ratio * |w_j| + (1 - l1_ratio) / 2 * w_j^2)
    XT is (p, n): feature-major, last row = constant 1 (intercept). pen_last = 1 if penalize_last
    (liblinear) else 0 (saga). lambda = 1 / C gives the scikit-learn solutions. Stops when the largest
    violation of the optimality conditions is <= tol * max(lambda, 1).
    """
    p, n = XT.shape
    w = np.zeros(p)
    eta = np.zeros(n)
    g = np.empty(n)
    h = np.empty(n)
    G = np.empty(p)
    a = np.empty(p)
    d = np.zeros(p)
    Xd = np.zeros(n)
    work = np.empty(p, np.int64)
    pen = np.ones(p)
    if not penalize_last:
        pen[p - 1] = 0.0
    for l in range(lambdas.shape[0]):
        lam = lambdas[l]
        lam1 = lam * l1_ratio
        lam2 = lam * (1.0 - l1_ratio)
        stop = tol * (lam if lam > 1.0 else 1.0)
        for newton in range(200):
            loss = 0.0
            for i in range(n):
                pi = _sigmoid(eta[i])
                g[i] = cw[i] * (pi - y[i])
                hi = cw[i] * pi * (1.0 - pi)
                h[i] = hi if hi > 1e-12 else 1e-12
                loss += cw[i] * (_log1pexp(eta[i]) - y[i] * eta[i])
            viol = 0.0
            nw = 0
            for j in range(p):
                s = 0.0
                for i in range(n):
                    s += XT[j, i] * g[i]
                s += lam2 * pen[j] * w[j]
                G[j] = s
                l1j = lam1 * pen[j]
                wj = w[j]
                if wj > 0.0:
                    v = abs(s + l1j)
                elif wj < 0.0:
                    v = abs(s - l1j)
                else:
                    v = abs(s) - l1j
                    if v < 0.0:
                        v = 0.0
                if v > viol:
                    viol = v
                if wj != 0.0 or abs(s) > l1j:
                    work[nw] = j
                    nw += 1
            if viol <= stop:
                break
            for k in range(nw):
                j = work[k]
                s = 0.0
                for i in range(n):
                    s += h[i] * XT[j, i] * XT[j, i]
                a[j] = s + lam2 * pen[j]
                d[j] = 0.0
            for i in range(n):
                Xd[i] = 0.0
            inner_stop = 0.05 * viol
            for sweep in range(2000):
                maxchange = 0.0
                for k in range(nw):
                    j = work[k]
                    s = 0.0
                    for i in range(n):
                        s += h[i] * XT[j, i] * Xd[i]
                    b = G[j] + s + lam2 * pen[j] * d[j]
                    z = w[j] + d[j]
                    u = a[j] * z - b
                    l1j = lam1 * pen[j]
                    if u > l1j:
                        znew = (u - l1j) / a[j]
                    elif u < -l1j:
                        znew = (u + l1j) / a[j]
                    else:
                        znew = 0.0
                    delta = znew - z
                    if delta != 0.0:
                        d[j] += delta
                        for i in range(n):
                            Xd[i] += delta * XT[j, i]
                        change = abs(delta) * a[j]
                        if change > maxchange:
                            maxchange = change
                if maxchange <= inner_stop:
                    break
            # Armijo line search on the exact objective (newGLMNET rule, sigma = 0.01)
            smooth_pen = 0.0
            l1_pen = 0.0
            dec = 0.0
            for k in range(nw):
                j = work[k]
                smooth_pen += 0.5 * lam2 * pen[j] * w[j] * w[j]
                l1_pen += lam1 * pen[j] * abs(w[j])
                dec += G[j] * d[j] + lam1 * pen[j] * (abs(w[j] + d[j]) - abs(w[j]))
            f0 = loss + smooth_pen + l1_pen
            alpha = 1.0
            for ls in range(60):
                lt = 0.0
                for i in range(n):
                    e = eta[i] + alpha * Xd[i]
                    lt += cw[i] * (_log1pexp(e) - y[i] * e)
                pt = 0.0
                for k in range(nw):
                    j = work[k]
                    wn = w[j] + alpha * d[j]
                    pt += pen[j] * (lam1 * abs(wn) + 0.5 * lam2 * wn * wn)
                if lt + pt <= f0 + 0.01 * alpha * dec or alpha < 1e-10:
                    break
                alpha *= 0.5
            for k in range(nw):
                j = work[k]
                w[j] += alpha * d[j]
            for i in range(n):
                eta[i] += alpha * Xd[i]
        for j in range(p):
            coef_out[l, j] = w[j]


@njit(parallel=True, cache=True, fastmath=FAST)
def _bootstrap_selection(XaugT, y, boot, lambdas, l1_ratios, penalize_intercept, tol, threshold, weights):
    """Selection indicators over subsamples. XaugT is (P, n) feature-major without the intercept row;
    boot is (B, m) subsample indices; one path per l1 ratio (1.0 = LASSO), stacked on the lambda axis.
    penalize_intercept: True penalizes the intercept as liblinear does (L1 Stabl).
    weights (B, P): column scaling per subsample (randomized LASSO, Meinshausen and Bühlmann); ones = plain Stabl."""
    P, n = XaugT.shape
    B, m = boot.shape
    L = lambdas.shape[0]
    R = l1_ratios.shape[0]
    sel = np.zeros((B, R * L, P), np.uint8)
    for b in prange(B):
        XT = np.empty((P + 1, m))
        yy = np.empty(m)
        npos = 0.0
        for i in range(m):
            yy[i] = y[boot[b, i]]
            npos += yy[i]
        for j in range(P):
            wj = weights[b, j]
            for i in range(m):
                XT[j, i] = XaugT[j, boot[b, i]] * wj
        for i in range(m):
            XT[P, i] = 1.0
        w1 = m / (2.0 * npos)
        w0 = m / (2.0 * (m - npos))
        cw = np.empty(m)
        for i in range(m):
            cw[i] = w1 if yy[i] == 1.0 else w0
        coefs = np.empty((L, P + 1))
        for r in range(R):
            logistic_path(XT, yy, cw, lambdas, l1_ratios[r], penalize_intercept, tol, coefs)
            for l in range(L):
                for j in range(P):
                    if abs(coefs[l, j]) >= threshold:
                        sel[b, r * L + l, j] = 1
    return sel


# ----------------------------------------------------------------------------- Stabl
def stabl_run(X, y, seed, n_bootstraps=1000, sample_fraction=0.75, n_lambda=30, tol=1e-4, weakness=None):
    """One Stabl run on standardized X; returns the max-over-C stability scores of the real and decoy features."""
    y = np.asarray(y, float)
    C = stabl_official.auto_c_grid(X, y.astype(int), n_lambda)
    Xaug = stabl_official.random_permutation_features(X, seed)
    m = int(np.floor(sample_fraction * len(y)))
    boot = np.asarray(stabl_official.bootstrap_generator(n_bootstraps, y.astype(int), m, False, "balanced", seed), np.int64)
    weights = np.ones((len(boot), Xaug.shape[1]))
    if weakness is not None:
        weights = np.random.default_rng(seed + 7).uniform(weakness, 1.0, size=weights.shape)
    sel = _bootstrap_selection(np.ascontiguousarray(Xaug.T), y, boot, 1.0 / C, np.array([1.0]), True, tol, SELECTION_THRESHOLD, weights)
    freq = sel.mean(axis=0)
    p = X.shape[1]
    return freq[:, :p].max(axis=0), freq[:, p:].max(axis=0)


def run_seeds(n_runs, base_seed=42):
    return [base_seed + k * 1000 for k in range(n_runs)]


# ----------------------------------------------------------------------------- models
def _design(X):
    return np.ascontiguousarray(np.vstack([X.T, np.ones(X.shape[0])]))


def fit_logistic(X, y, C_values, l1_ratio, tol=1e-7):
    """scikit-learn LogisticRegression(penalty='elasticnet', C, l1_ratio, class_weight=None) coefficients
    for each C (intercept unpenalized); X must already be standardized. Returns (len(C), p + 1)."""
    C_values = np.asarray(C_values, float)
    order = np.argsort(C_values)
    coefs = np.empty((len(C_values), X.shape[1] + 1))
    out = np.empty_like(coefs)
    logistic_path(_design(X), np.asarray(y, float), np.ones(len(y)), 1.0 / C_values[order], float(l1_ratio), False, tol, out)
    coefs[order] = out
    return coefs


def predict(coef, X):
    eta = X @ coef[:-1] + coef[-1]
    return 1.0 / (1.0 + np.exp(-eta))


class Scaled:
    def __init__(self, mean, scale, coef):
        self.mean, self.scale, self.coef = mean, scale, coef

    def predict_proba(self, X):
        return predict(self.coef, (np.asarray(X, float) - self.mean) / self.scale)


def _standardize(X):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale
