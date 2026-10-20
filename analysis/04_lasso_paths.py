#!/usr/bin/env python3
"""LASSO (L1-penalized logistic regression) coefficient paths of the prefiltered features: full cohort, ComBat by protocol,
z-scored; the base learner of Stabl (liblinear, balanced class weights). C spans the range of the automatic Stabl grid,
from l1_min_c (no feature selected) to 100 x l1_min_c, 200 log-spaced values; lambda = 1/C. Coefficients are log-odds per
standard deviation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402

OUT = common.RESULTS / "feature_selection"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = common.load()
    harmonized = common.harmonize_all(d.radiomics, d.batch)
    kept = common.prefilter(harmonized, correlation=0.80)
    assert len(kept) == json.loads((OUT / "summary.json").read_text())["after_prefilter"]
    X = common.standardize(harmonized[kept].to_numpy(float))
    Cs = l1_min_c(X, d.y, loss="log") * np.logspace(0, 2, 200)
    est = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", tol=1e-8, max_iter=int(1e6), random_state=0)
    coefs = np.array([est.set_params(C=C).fit(X, d.y).coef_.ravel() for C in Cs])
    table = pd.DataFrame(coefs[::-1], columns=kept)
    table.insert(0, "lambda", 1.0 / Cs[::-1])
    table.to_csv(OUT / "lasso_paths.csv", index=False)
    print(f"{len(kept)} features x {len(Cs)} values of C; lambda {table['lambda'].min():.3g} to {table['lambda'].max():.3g}")


if __name__ == "__main__":
    main()
