#!/usr/bin/env python3
"""Nested cross-validation of the whole selection-and-modeling pipeline. Outer folds: stratified 10-fold repeated 10 times
(fold seed 42). In every training fold: ComBat by reconstruction protocol, prefilter |rho| 0.80, z-score, Stabl exactly as on
all patients (03: randomized LASSO w 0.5, random permutations, 50 runs x 1,000 subsamples), features ranked by mean
stability. Rules applied to the fold's ranking:
  TOPk        the k most stable features, k = 3..10, elastic net tuned as the final model (models.tune_final: best of five
              Optuna runs, 100 trials, grouped 5-fold x 10). TOP5 (the size of the final signature) is the reported nested
              estimate of the selected model; the other sizes are a sensitivity analysis.
  INNER-K     stricter sensitivity analysis: the panel size k is also re-chosen inside each training fold by inner
              cross-validation (src/panel_size.py: stratified 5-fold x 2 repeats of the training fold; ComBat, prefilter,
              Stabl (5 runs) and an Optuna-tuned elastic net on the top-k features, k = 3..10, refitted on every inner
              training part; highest pooled inner AUC, smallest k on ties); the k most stable features of the fold's
              50-run ranking; elastic net tuned as the final model.
Every stage is cached in results/nested_cv/, keyed to the data checksums and the settings: the outer rankings
(folds_<K>fold.json), the inner choice of k (innerk_optuna_<K>fold.json), and the elastic nets of INNER-K
(innerk_optuna_final_<K>fold.json) and of the fixed sizes (topk_final_<K>fold.json).
Estimates: pooled out-of-fold AUC per repeat (mean, SD; 95% CI of the mean from 2,000 class-stratified patient bootstrap
samples). Reselection: for each feature of the full-cohort signature, the share of training folds whose panel contains it. Fold records hold row positions only."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from collections import Counter
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402
import fast_stabl as fs  # noqa: E402
import models  # noqa: E402
import panel_size  # noqa: E402

OUT = common.RESULTS / "nested_cv"
SEED, TOPK = 42, range(3, 11)
_D = {}


def init_worker(threads):
    import numba

    numba.set_num_threads(threads)
    _D["d"] = common.load()


top_panel = panel_size.top_panel


def job(splits, seed, k, train, test, runs):
    """Outer training fold: ComBat, prefilter and Stabl; features ranked by mean stability."""
    started = time.time()
    d, y = _D["d"], _D["d"].y
    tr, _ = common.combat(d.radiomics, d.batch, train, test)
    kept = common.prefilter(tr, correlation=0.80)
    X = common.standardize(tr[kept].to_numpy(float))
    real = np.vstack([fs.stabl_run(X, y[train], s, n_bootstraps=1000, sample_fraction=0.75, weakness=0.5)[0] for s in fs.run_seeds(runs)])
    stability = dict(zip(kept, real.mean(axis=0).tolist()))
    return {"splits": splits, "seed": seed, "k": k, "test": test.tolist(), "stability": stability,
            "panels": {f"TOP{n}": top_panel(stability, n) for n in TOPK}, "prob": {}, "params": {}, "seconds": time.time() - started}


def inner_job(splits, seed, k, test, stability, inner_repeats, inner_runs):
    """INNER-K rule for one outer fold: k chosen by inner cross-validation of the training fold; the panel is the top k of
    the stored outer ranking."""
    started = time.time()
    d = _D["d"]
    test = np.asarray(test)
    train = np.setdiff1d(np.arange(len(d.y)), test)
    auc = panel_size.inner_auc(d, train, seed=int(np.random.SeedSequence([seed, k]).generate_state(1)[0]), repeats=inner_repeats, runs=inner_runs)
    k_star = panel_size.choose(auc)
    return {"splits": splits, "seed": seed, "k": k, "inner_auc": {str(n): v for n, v in auc.items()}, "k_star": k_star,
            "panel": top_panel(stability, k_star), "seconds": time.time() - started}


def topk_job(splits, seed, k, test, stability, trials):
    """Fixed panel sizes k = 3..10 (top k of the stored outer ranking), elastic net tuned as the final model."""
    d = _D["d"]
    test = np.asarray(test)
    train = np.setdiff1d(np.arange(len(d.y)), test)
    tr, te = common.combat(d.radiomics, d.batch, train, test)
    out = {"splits": splits, "seed": seed, "k": k, "prob": {}, "params": {}}
    for n in TOPK:
        panel = top_panel(stability, n)
        est, params, _, _ = models.tune_final(tr[panel].to_numpy(float), d.y[train], train, trials=trials)
        out["prob"][f"TOP{n}"], out["params"][f"TOP{n}"] = est.predict_proba(te[panel].to_numpy(float)).tolist(), params
    return out


def final_job(splits, seed, k, test, panel, trials):
    """INNER-K rule: the panel chosen by inner cross-validation of the training fold (innerk cache), elastic net tuned as
    the final model on the training fold and applied to the test fold."""
    d = _D["d"]
    test = np.asarray(test)
    train = np.setdiff1d(np.arange(len(d.y)), test)
    tr, te = common.combat(d.radiomics, d.batch, train, test)
    est, params, _, _ = models.tune_final(tr[panel].to_numpy(float), d.y[train], train, trials=trials)
    return {"splits": splits, "seed": seed, "k": k, "panel": panel, "prob": est.predict_proba(te[panel].to_numpy(float)).tolist(), "params": params}


REPORTED = ("INNER-K", *(f"TOP{n}" for n in TOPK))


def summarize(records, y, splits, repeats, seeds):
    rng = np.random.default_rng(20260928)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    resamples = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(2000)]
    table = {}
    for method in REPORTED:
        pooled, by_seed, stack = [], {}, []
        for seed in seeds:
            for rep in range(repeats):
                rs = [r for r in records if r["seed"] == seed and r["k"] // splits == rep]
                idx = np.concatenate([r["test"] for r in rs])
                auc = float(roc_auc_score(y[idx], np.concatenate([r["prob"][method] for r in rs])))
                pooled.append(auc)
                by_seed.setdefault(str(seed), []).append(auc)
                row = np.full(len(y), np.nan)
                for r in rs:
                    row[r["test"]] = r["prob"][method]
                stack.append(row)
        full = np.vstack(stack)  # one complete out-of-fold prediction vector per repeat
        mean_ci = np.quantile([np.mean([roc_auc_score(y[i], row[i]) for row in full]) for i in resamples], [0.025, 0.975]).tolist()
        table[method] = {"auc_mean": float(np.mean(pooled)), "auc_mean_ci": mean_ci, "auc_sd": float(np.std(pooled, ddof=1)), "auc_range": [float(min(pooled)), float(max(pooled))],
                         "by_fold_seed": {s: float(np.mean(v)) for s, v in by_seed.items()}, "per_repeat": pooled}
    signature = common.signature()
    selection = {}
    for rule in ("INNER-K", f"TOP{len(signature)}"):
        counts = Counter(f for r in records for f in r["panels"][rule])
        sizes = [len(r["panels"][rule]) for r in records]
        selection[rule] = {"signature_feature_frequency": {f: counts.get(f, 0) / len(records) for f in signature},
                           "folds_with_whole_signature": float(np.mean([set(signature) <= set(r["panels"][rule]) for r in records])),
                           "panel_size": {"median": float(np.median(sizes)), "q1": float(np.quantile(sizes, 0.25)), "q3": float(np.quantile(sizes, 0.75)),
                                          "min": int(min(sizes)), "max": int(max(sizes)), "empty_folds": int(sum(n == 0 for n in sizes))},
                           "most_selected": [(f, c / len(records)) for f, c in counts.most_common(12)]}
    if "INNER-K" in records[0]["panels"]:
        selection["INNER-K"]["k_chosen"] = {str(n): c for n, c in sorted(Counter(r["k_star"] for r in records).items())}
    return {"folds": len(records), "signature": signature, "methods": table, "selection": selection}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, nargs="+", default=[10])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    parser.add_argument("--runs", type=int, default=common.STABL_RUNS)
    parser.add_argument("--inner-repeats", type=int, default=2)
    parser.add_argument("--inner-runs", type=int, default=panel_size.INNER_RUNS)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_NUM_THREADS"] = str(args.threads)  # the (spawned) workers size their numba pool from this variable
    y = common.load().y
    started = time.time()
    for splits in args.splits:
        cache, stamp = OUT / f"folds_{splits}fold.json", OUT / f"folds_{splits}fold.sha256"
        inputs = "".join(line for line in (common.DATA / "SHA256SUMS").read_text().splitlines(True) if line.split()[-1] in ("cohort.csv", "radiomics.csv"))
        inputs += f"stabl runs {args.runs}, outer rankings\n"
        if cache.exists() and stamp.exists() and stamp.read_text() != inputs:
            cache.unlink()  # folds of other data or settings
        records = json.loads(cache.read_text()) if cache.exists() else []
        stamp.write_text(inputs)
        done = {(r["seed"], r["k"]) for r in records}
        jobs = [(splits, seed, k, tr, te) for seed in args.seeds
                for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=splits, n_repeats=args.repeats, random_state=seed).split(np.zeros(len(y)), y))
                if (seed, k) not in done]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(args.threads,)) as pool:
            futures = [pool.submit(job, *j, args.runs) for j in jobs]
            for n, f in enumerate(as_completed(futures), 1):
                records.append(f.result())
                if n % 10 == 0 or n == len(futures):
                    cache.write_text(json.dumps(records) + "\n")
                    print(f"{splits}-fold: {len(records)}/{len(args.seeds) * splits * args.repeats} folds ({(time.time() - started) / 60:.1f} min)", flush=True)
        records = [r for r in records if r["seed"] in args.seeds and r["k"] < splits * args.repeats]
        icache, istamp = OUT / f"innerk_optuna_{splits}fold.json", OUT / f"innerk_optuna_{splits}fold.sha256"
        idesign = inputs + (f"inner k: stratified {panel_size.INNER_SPLITS}-fold x {args.inner_repeats} repeats, stabl runs {args.inner_runs}, "
                            f"k {panel_size.KS[0]}..{panel_size.KS[-1]}, inner elastic net optuna {panel_size.INNER_TRIALS} trials x "
                            f"{panel_size.INNER_TUNING_REPEATS} cv repeats\n")
        if icache.exists() and istamp.exists() and istamp.read_text() != idesign:
            icache.unlink()  # INNER-K folds of other data or settings
        inner = json.loads(icache.read_text()) if icache.exists() else []
        istamp.write_text(idesign)
        done = {(r["seed"], r["k"]) for r in inner}
        todo = [r for r in records if (r["seed"], r["k"]) not in done]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(args.threads,)) as pool:
            futures = [pool.submit(inner_job, splits, r["seed"], r["k"], r["test"], r["stability"], args.inner_repeats, args.inner_runs)
                       for r in todo]
            for n, f in enumerate(as_completed(futures), 1):
                inner.append(f.result())
                if n % 10 == 0 or n == len(futures):
                    icache.write_text(json.dumps(inner) + "\n")
                    print(f"{splits}-fold INNER-K: {len(inner)}/{len(records)} folds ({(time.time() - started) / 60:.1f} min)", flush=True)
        by_fold = {(r["seed"], r["k"]): r for r in inner}
        for r in records:
            x = by_fold[(r["seed"], r["k"])]
            r["panels"]["INNER-K"], r["k_star"] = x["panel"], x["k_star"]
        fcache, fstamp = OUT / f"innerk_optuna_final_{splits}fold.json", OUT / f"innerk_optuna_final_{splits}fold.sha256"
        fdesign = idesign + f"outer elastic net tuned as the final model (seeds {models.FINAL_SEEDS}, 5-fold x 10), optuna trials {args.trials}\n"
        if fcache.exists() and fstamp.exists() and fstamp.read_text() != fdesign:
            fcache.unlink()  # final-tuning folds of other data or settings
        final = json.loads(fcache.read_text()) if fcache.exists() else []
        fstamp.write_text(fdesign)
        done = {(r["seed"], r["k"]) for r in final}
        todo = [r for r in records if (r["seed"], r["k"]) not in done]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(args.threads,)) as pool:
            futures = [pool.submit(final_job, splits, r["seed"], r["k"], r["test"], r["panels"]["INNER-K"], args.trials) for r in todo]
            for n, f in enumerate(as_completed(futures), 1):
                final.append(f.result())
                if n % 50 == 0 or n == len(futures):
                    fcache.write_text(json.dumps(final) + "\n")
                    print(f"{splits}-fold INNER-K final tuning: {len(final)}/{len(records)} folds ({(time.time() - started) / 60:.1f} min)", flush=True)
        by_fold = {(r["seed"], r["k"]): r for r in final}
        for r in records:
            x = by_fold[(r["seed"], r["k"])]
            if x["panel"] != r["panels"]["INNER-K"]:
                raise RuntimeError("cached final-tuning panel differs from the INNER-K panel")
            r["prob"]["INNER-K"], r["params"]["INNER-K"] = x["prob"], x["params"]
        tcache, tstamp = OUT / f"topk_final_{splits}fold.json", OUT / f"topk_final_{splits}fold.sha256"
        tdesign = inputs + f"fixed top-k {TOPK.start}..{TOPK.stop - 1}, tuning as the final model (seeds {models.FINAL_SEEDS}, 5-fold x 10), optuna trials {args.trials}\n"
        if tcache.exists() and tstamp.exists() and tstamp.read_text() != tdesign:
            tcache.unlink()  # fixed-size folds of other data or settings
        topk = json.loads(tcache.read_text()) if tcache.exists() else []
        tstamp.write_text(tdesign)
        done = {(r["seed"], r["k"]) for r in topk}
        todo = [r for r in records if (r["seed"], r["k"]) not in done]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(args.threads,)) as pool:
            futures = [pool.submit(topk_job, splits, r["seed"], r["k"], r["test"], r["stability"], args.trials) for r in todo]
            for n, f in enumerate(as_completed(futures), 1):
                topk.append(f.result())
                if n % 20 == 0 or n == len(futures):
                    tcache.write_text(json.dumps(topk) + "\n")
                    print(f"{splits}-fold TOPk final tuning: {len(topk)}/{len(records)} folds ({(time.time() - started) / 60:.1f} min)", flush=True)
        by_fold = {(r["seed"], r["k"]): r for r in topk}
        for r in records:
            r["prob"].update(by_fold[(r["seed"], r["k"])]["prob"])
        summary = {"design": f"{splits}-fold x {args.repeats} repeats x fold seeds {list(args.seeds)}", "stabl_runs": args.runs, "patients": int(len(y)),
                   "events": int(y.sum()), **summarize(records, y, splits, args.repeats, args.seeds)}
        (OUT / f"summary_{splits}fold.json").write_text(json.dumps(summary, indent=2) + "\n")
        for method, row in summary["methods"].items():
            print(f"[{splits}-fold] {method:12s} AUC {row['auc_mean']:.3f} (SD {row['auc_sd']:.3f}; seeds {row['by_fold_seed']}) | "
                  f"95% CI {row['auc_mean_ci'][0]:.3f}-{row['auc_mean_ci'][1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
