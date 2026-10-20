#!/usr/bin/env python3
"""Full-cohort feature selection: ComBat by reconstruction protocol, outcome-blind prefilter (|rho| 0.80), z-score, Stabl
(randomized LASSO w 0.5, random-permutation decoys, 1,000 subsamples of 75%) over 50 runs, features ranked by mean
stability. The number of features k is chosen by inner cross-validation on all patients (src/panel_size.py: stratified
5-fold x 10 repeats; ComBat, prefilter, Stabl (5 runs) and an Optuna-tuned elastic net on the top-k features, k = 3..10,
refitted on every inner training part; highest pooled inner AUC, smallest k on ties). Signature (k-rad): the k most stable
features.
The same rule is repeated inside every outer training fold of the nested cross-validation (09_nested_cv.py)."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402
import fast_stabl as fs  # noqa: E402
import panel_size  # noqa: E402

OUT = common.RESULTS / "feature_selection"
RUNS = common.STABL_RUNS
INNER_SEED, INNER_REPEATS = 2026, 10


def main():
    import numba

    numba.set_num_threads(min(int(os.environ.get("THREADS", "12")), numba.config.NUMBA_NUM_THREADS))
    OUT.mkdir(parents=True, exist_ok=True)
    d = common.load()
    harmonized = common.harmonize_all(d.radiomics, d.batch)
    kept = common.prefilter(harmonized, correlation=0.80)
    X = common.standardize(harmonized[kept].to_numpy(float))
    real = np.vstack([fs.stabl_run(X, d.y, seed, n_bootstraps=1000, sample_fraction=0.75, weakness=0.5)[0] for seed in fs.run_seeds(RUNS)])
    stability = real.mean(axis=0)
    inner = panel_size.inner_auc(d, np.arange(len(d.y)), seed=INNER_SEED, repeats=INNER_REPEATS, workers=int(os.environ.get("THREADS", "12")))
    k_star = panel_size.choose(inner)
    signature = panel_size.top_panel(dict(zip(kept, stability.tolist())), k_star)
    table = pd.DataFrame({"feature": kept, "family": [common.family(f) for f in kept], "display_name": [common.short_name(f) for f in kept],
                          "mean_stability": stability})
    table = table.sort_values("mean_stability", ascending=False, kind="stable")
    table.to_csv(OUT / "stabl_stability.csv", index=False)
    count = lambda features: pd.Series([common.family(f) for f in features]).value_counts().reindex(["Original", "LoG-filtered", "Wavelet-filtered"], fill_value=0).astype(int).to_dict()  # noqa: E731
    summary = {"patients": int(len(d.y)), "events": int(d.y.sum()), "stabl_runs": RUNS, "subsamples_per_run": 1000,
               "extracted": int(d.radiomics.shape[1]), "after_prefilter": len(kept), "signature": signature, "signature_size": k_star,
               "signature_rule": "the k most stable features (mean stability over the STABL runs), k chosen by inner cross-validation",
               "inner_cv": {"best_k": k_star, "inner_auc": {str(k): v for k, v in inner.items()}, "splits": panel_size.INNER_SPLITS,
                            "repeats": INNER_REPEATS, "stabl_runs": panel_size.INNER_RUNS, "optuna_trials": panel_size.INNER_TRIALS,
                            "optuna_cv_repeats": panel_size.INNER_TUNING_REPEATS,
                            "design": f"stratified {panel_size.INNER_SPLITS}-fold x {INNER_REPEATS} repeats (seed {INNER_SEED}); ComBat, prefilter "
                                      f"and STABL ({panel_size.INNER_RUNS} runs x 1,000 subsamples) refitted on each inner training part; "
                                      f"elastic net tuned by Optuna ({panel_size.INNER_TRIALS} trials, grouped 5-fold CV x {panel_size.INNER_TUNING_REPEATS}) "
                                      f"on the top-k features, k = {panel_size.KS[0]}..{panel_size.KS[-1]}; "
                                      f"highest pooled inner AUC, smallest k on ties"},
               "by_family": {"extracted": count(d.radiomics.columns), "after_prefilter": count(kept), "signature": count(signature)}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_family"}, indent=1))
    print(table.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
