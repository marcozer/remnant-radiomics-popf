"""Panel size chosen by inner cross-validation: the number k of most stable Stabl features whose elastic net (tuned by
Optuna, as the final model, with a lighter budget) has the highest pooled inner AUC; ComBat, prefilter, Stabl and the
Optuna tuning are refitted on every inner training part (the inner test part never reaches the ranking or the tuning).
Used on all patients (03_feature_selection) and inside every outer training fold (09_nested_cv)."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

import common
import fast_stabl as fs
import models

KS = tuple(range(3, 11))
INNER_SPLITS, INNER_RUNS = 5, 5
INNER_TRIALS, INNER_TUNING_REPEATS = 50, 2  # Optuna budget of the inner elastic nets: trials, repeats of the grouped 5-fold tuning CV
_W = {}


def ranking(frame, y, runs):
    """Mean Stabl stability of the prefiltered features of `frame` (ComBat-harmonized training rows)."""
    kept = common.prefilter(frame, correlation=0.80)
    X = common.standardize(frame[kept].to_numpy(float))
    real = np.vstack([fs.stabl_run(X, y, s, n_bootstraps=1000, sample_fraction=0.75, weakness=0.5)[0] for s in fs.run_seeds(runs)])
    return dict(zip(kept, real.mean(axis=0).tolist()))


def top_panel(stability, k):
    """The k features with the highest stability, in ranking-table order."""
    names = list(stability)
    return [names[i] for i in np.sort(np.argsort(-np.array([stability[f] for f in names]), kind="stable")[:k])]


def _split(d, rows, a, b, seed, runs, ks):
    """Out-of-split probabilities of the top-k Optuna elastic nets for one inner split (same tuning seed for every k)."""
    tr, te = common.combat(d.radiomics, d.batch, rows[a], rows[b])
    stability = ranking(tr, d.y[rows[a]], runs)
    out = {}
    for k in ks:
        panel = top_panel(stability, k)
        est, _, _ = models.optuna_en(tr[panel].to_numpy(float), d.y[rows[a]], rows[a], seed=seed, trials=INNER_TRIALS, repeats=INNER_TUNING_REPEATS)
        out[k] = est.predict_proba(te[panel].to_numpy(float))
    return out


def _init(threads):
    import numba

    numba.set_num_threads(threads)
    _W["d"] = common.load()


def _job(task):
    return _split(_W["d"], *task)


def inner_auc(d, rows, seed, repeats, runs=INNER_RUNS, ks=KS, workers=1):
    """Pooled inner AUC of the top-k Optuna elastic net for each k, mean over `repeats` repeats of a stratified 5-fold split
    of `rows`; ComBat, prefilter, Stabl and the tuning refitted on every inner training part. `workers` > 1 runs the inner
    splits in parallel processes (one numba thread each)."""
    rows = np.asarray(rows)
    pred = {k: np.full((repeats, len(rows)), np.nan) for k in ks}
    splits = list(RepeatedStratifiedKFold(n_splits=INNER_SPLITS, n_repeats=repeats, random_state=seed).split(np.zeros(len(rows)), d.y[rows]))
    tasks = [(rows, a, b, int(np.random.SeedSequence([seed, j]).generate_state(1)[0] % 2**30), runs, ks) for j, (a, b) in enumerate(splits)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init, initargs=(1,)) as pool:
            results = list(pool.map(_job, tasks))
    else:
        results = [_split(d, *task) for task in tasks]
    for j, ((_, b), out) in enumerate(zip(splits, results)):
        for k in ks:
            pred[k][j // INNER_SPLITS, b] = out[k]
    return {k: float(np.mean([roc_auc_score(d.y[rows], pred[k][r]) for r in range(repeats)])) for k in ks}


def choose(auc):
    """k with the highest inner AUC; the smallest k on ties."""
    best = max(auc.values())
    return min(int(k) for k, v in auc.items() if v == best)
