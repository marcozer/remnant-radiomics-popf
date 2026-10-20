"""Pieces of the official Stabl package used by src/fast_stabl.py (subsample generator, random-permutation
decoys, automatic C grid), copied from https://github.com/gregbellan/Stabl (commit 1d07f85, stabl/stabl.py and
stabl/utils.py) so that subsamples and decoys are identical to the official implementation.

The Clear BSD License

Copyright (c) 2023, SurgeCare
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors may be used
  to endorse or promote products derived from this software without specific
  prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS
LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import l1_min_c
from sklearn.utils.class_weight import compute_sample_weight


def auto_c_grid(X, y, n_lambda=30):
    """utils.auto_mode_lambda_grid(X, y, "classification", None, n_lambda)["C"]."""
    min_c = l1_min_c(X, y, loss="log")
    return np.linspace(min_c, min_c * 100, n_lambda)


def random_permutation_features(X, seed):
    """Stabl._make_artificial_features(X, "random_permutation", nb_noise=X.shape[1], random_state=seed)."""
    rng = np.random.default_rng(seed=seed)
    artificial = X.copy()
    indices = rng.choice(a=artificial.shape[1], size=X.shape[1], replace=False)
    artificial = artificial[:, indices]
    for i in range(artificial.shape[1]):
        rng.shuffle(artificial[:, i])
    return np.concatenate([X, artificial], axis=1)


def classic_bootstrap(y, n_subsamples, replace=True, class_weight=None, rng=None):
    n_samples = y.shape[0]
    if class_weight is not None:
        weights = compute_sample_weight(class_weight, y)
        probabilities = weights / weights.sum()
    else:
        probabilities = None
    sampled = rng.choice(a=n_samples, size=n_subsamples, replace=replace, p=probabilities)
    if len(np.unique(y[sampled])) < 2:
        sampled = classic_bootstrap(y, n_subsamples, replace=replace, class_weight=class_weight, rng=rng)
    return sampled


def bootstrap_generator(n_bootstraps, y, n_subsamples, replace, class_weight, random_state):
    """stabl._bootstrap_generator with classic_bootstrap."""
    rng = np.random.RandomState(random_state)
    return [classic_bootstrap(y=y, n_subsamples=n_subsamples, replace=replace, class_weight=class_weight, rng=rng)
            for _ in range(n_bootstraps)]
