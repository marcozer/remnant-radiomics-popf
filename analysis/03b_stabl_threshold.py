#!/usr/bin/env python3
"""Why the signature does not use Stabl's own threshold. The official reliability threshold of Stabl 1.0.0
(_compute_FDPplus / _get_support_mask, explore off: FDP+(t) = (1 + #artificial > t) / max(1, #real > t) on the grid
0, 0.01, ..., 0.99; cutoff 1 when the minimum exceeds 1) applied to
  (a) the 50 Stabl runs of 03_feature_selection.py (same seeds and settings; their mean stability is checked against
      stabl_stability.csv), and
  (b) the Stabl 1.0.0 package defaults: uniform subsamples of 50% without replacement, plain L1 logistic regression
      (class-balanced), random-permutation artificial features, C = 0.01..1 (10 values); 10 seeds.
Reported per setting: runs with an empty panel, panel sizes, minimum FDP+, median stability of the real and the
artificial features, selection count of each feature. No outcome model is fitted."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402
import fast_stabl as fs  # noqa: E402
import stabl_official as so  # noqa: E402

OUT = common.RESULTS / "feature_selection"
GRID = np.arange(0.0, 1.0, 0.01)
DEFAULT_SEEDS = 10


def official(real, decoy):
    fdp = np.array([(1 + np.sum(decoy > t)) / max(1, np.sum(real > t)) for t in GRID])
    cutoff = 1.0 if fdp.min() > 1 else float(min(GRID[np.argmin(fdp)], 1))
    return cutoff, float(fdp.min()), real > cutoff


def defaults_run(X, y, seed):
    Xaug = so.random_permutation_features(X, seed)
    boot = np.asarray(so.bootstrap_generator(1000, y, int(np.floor(0.5 * len(y))), False, None, seed), np.int64)
    sel = fs._bootstrap_selection(np.ascontiguousarray(Xaug.T), y.astype(float), boot, 1.0 / np.linspace(0.01, 1, 10), np.array([1.0]), True,
                                  1e-4, fs.SELECTION_THRESHOLD, np.ones((len(boot), Xaug.shape[1])))
    freq = sel.mean(axis=0)
    p = X.shape[1]
    return freq[:, :p].max(axis=0), freq[:, p:].max(axis=0)


def summarize(kept, runs):
    rows = [official(real, decoy) + (real, decoy) for real, decoy in runs]
    sizes = [int(mask.sum()) for _, _, mask, _, _ in rows]
    counts = Counter(kept[i] for _, _, mask, _, _ in rows for i in np.flatnonzero(mask))
    return {"runs": len(rows), "empty_panels": sum(s == 0 for s in sizes), "panel_sizes": sizes,
            "min_fdp_plus_median": float(np.median([f for _, f, _, _, _ in rows])),
            "min_fdp_plus_range": [float(min(f for _, f, _, _, _ in rows)), float(max(f for _, f, _, _, _ in rows))],
            "median_stability_real": float(np.median([np.median(r) for _, _, _, r, _ in rows])),
            "median_stability_artificial": float(np.median([np.median(a) for _, _, _, _, a in rows])),
            "max_artificial_range": [float(min(a.max() for _, _, _, _, a in rows)), float(max(a.max() for _, _, _, _, a in rows))],
            "selection_count": dict(counts.most_common())}


def main():
    import numba

    numba.set_num_threads(min(int(os.environ.get("THREADS", "12")), numba.config.NUMBA_NUM_THREADS))
    d = common.load()
    harmonized = common.harmonize_all(d.radiomics, d.batch)
    kept = common.prefilter(harmonized, correlation=0.80)
    X = common.standardize(harmonized[kept].to_numpy(float))
    pipeline = [fs.stabl_run(X, d.y, seed, n_bootstraps=1000, sample_fraction=0.75, weakness=0.5) for seed in fs.run_seeds(common.STABL_RUNS)]
    stored = pd.read_csv(OUT / "stabl_stability.csv", float_precision="round_trip").set_index("feature").loc[kept, "mean_stability"].to_numpy()
    if np.max(np.abs(np.mean([r for r, _ in pipeline], axis=0) - stored)) > 1e-12:
        raise RuntimeError("Stabl runs differ from 03_feature_selection.py")
    defaults = [defaults_run(X, d.y, seed) for seed in fs.run_seeds(DEFAULT_SEEDS)]
    summary = {"patients": int(len(d.y)), "events": int(d.y.sum()), "features": len(kept),
               "pipeline_settings": summarize(kept, pipeline), "stabl_package_defaults": summarize(kept, defaults)}
    (OUT / "stabl_threshold.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name in ("pipeline_settings", "stabl_package_defaults"):
        s = summary[name]
        print(f"{name}: empty {s['empty_panels']}/{s['runs']}, sizes {sorted(Counter(s['panel_sizes']).items())}, min FDP+ median "
              f"{s['min_fdp_plus_median']:.2f}, median stability real {s['median_stability_real']:.2f} vs artificial "
              f"{s['median_stability_artificial']:.2f}; {s['selection_count']}")


if __name__ == "__main__":
    main()
