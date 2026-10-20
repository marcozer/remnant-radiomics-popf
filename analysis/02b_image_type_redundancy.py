"""Outcome-blind redundancy of the six excluded image types against the 758 analyzed features.

ComBat by reconstruction protocol on all patients (each block separately, as in the analysis), then, for every excluded
feature, the largest |Spearman rho| with an analyzed feature; and the number of candidates left by the prefilter
(|rho| 0.80) with and without the excluded types. Features constant in the cohort or in the ComBat reference protocol are
left out. Writes results/image_type_checks/redundancy.csv."""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402

TYPES = ["square", "squareroot", "logarithm", "exponential", "gradient", "lbp-2D"]
OUT = common.RESULTS / "image_type_checks"


def main():
    d = common.load()
    extra = pd.read_csv(common.DATA / "excluded_image_types.csv", index_col="patient_id", float_precision="round_trip").loc[d.ids]
    reference = d.batch == common.REFERENCE_BATCH
    informative = (extra.std() > 0) & (extra[reference].std() > 0)
    extra = extra.loc[:, informative]
    kept = common.harmonize_all(d.radiomics, d.batch)
    extra = common.harmonize_all(extra, d.batch)
    rho = np.abs(spearmanr(np.c_[extra.to_numpy(float), kept.to_numpy(float)])[0])[: extra.shape[1], extra.shape[1]:]
    table = pd.DataFrame({"feature": extra.columns, "image_type": [c.split("_")[0] for c in extra.columns],
                          "max_abs_rho_with_analysed": np.nanmax(rho, axis=1),
                          "closest_analysed_feature": kept.columns[np.nanargmax(rho, axis=1)]})
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "redundancy.csv", index=False)
    print(f"excluded features: {int(informative.size)} extracted, {int((~informative).sum())} constant, {extra.shape[1]} informative")
    r = table.max_abs_rho_with_analysed
    print(f"largest |rho| with an analyzed feature: > 0.80 for {np.mean(r > 0.80):.0%}, > 0.90 for {np.mean(r > 0.90):.0%}, median {r.median():.2f}")
    for t in TYPES:
        rt = table.loc[table.image_type == t, "max_abs_rho_with_analysed"]
        print(f"  {t:12s} {len(rt):3d} features: > 0.80 for {np.mean(rt > 0.80):4.0%}, median {rt.median():.2f}")
    alone = common.prefilter(kept, correlation=0.80)
    both = common.prefilter(pd.concat([kept, extra], axis=1), correlation=0.80)
    added = [f for f in both if f in extra.columns]
    print(f"prefilter |rho| 0.80: analyzed features {len(alone)}; with the excluded image types {len(both)} "
          f"({len(both) - len(added)} analyzed + {len(added)} excluded-type features)")


if __name__ == "__main__":
    main()
