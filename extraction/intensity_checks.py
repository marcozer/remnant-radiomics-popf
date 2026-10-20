"""Outcome-blind checks of the original and the six excluded image types, as PyRadiomics computes them
(results/image_type_checks/intensity_checks.csv).

For each patient, on the cropped CT resampled with the extraction settings (1 mm in-plane, B-spline):
  scaling value   largest absolute intensity of the image; it scales the Square, SquareRoot, Logarithm and Exponential
                  transforms. scaling_from_air tells whether it comes from air (negative extreme) or dense tissue
                  (bone, contrast)
  gray levels     number of gray levels inside the head-isthmus mask after discretization (bin width 25; LBP2D codes: 1)
                  and share of mask voxels in the fullest bin, per image type
Input: the cases CSV of extract_features.py (patient_id, ct, mask). CT images are not shared.

  python3 extraction/intensity_checks.py cases.csv results/image_type_checks/intensity_checks.csv
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from extract_features import HERE, prepare

TYPES = ["Original", "Square", "SquareRoot", "Logarithm", "Exponential", "Gradient", "LBP2D"]


def one(patient_id, ct_path, mask_path):
    import SimpleITK as sitk
    import yaml
    from radiomics import imageoperations

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    settings = yaml.safe_load((HERE / "radiomics_params.yaml").read_text())["setting"]
    image, roi = prepare(ct_path, mask_path)
    image, roi = imageoperations.resampleImage(image, roi, **settings)
    array, inside = sitk.GetArrayFromImage(image).astype(float), sitk.GetArrayFromImage(roi) > 0
    out = {"patient_id": patient_id, "scaling_value_hu": float(np.abs(array).max()), "scaling_from_air": bool(-array.min() >= array.max())}
    for t in TYPES:
        derived = image if t == "Original" else next(getattr(imageoperations, f"get{t}Image")(image, roi, **settings))[0]
        x = sitk.GetArrayFromImage(derived).astype(float)[inside]
        width = 1 if t == "LBP2D" else 25
        edges = np.arange(x.min() - x.min() % width, x.max() + width * 2, width)
        counts = np.bincount(np.digitize(x, edges))
        out[f"{t}_gray_levels"] = int((counts > 0).sum())
        out[f"{t}_fullest_bin_share"] = float(counts.max() / counts.sum())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cases = pd.read_csv(args.cases, dtype=str)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, cases["patient_id"], cases["ct"], cases["mask"]))
    table = pd.DataFrame(rows).set_index("patient_id")
    table.to_csv(args.output)
    summary(table)


def summary(table):
    s = table.scaling_value_hu
    print(f"{len(table)} patients; scaling value: median {s.median():.0f} HU (IQR {s.quantile(.25):.0f}-{s.quantile(.75):.0f}), "
          f"range {s.min():.0f}-{s.max():.0f}; from air in {table.scaling_from_air.mean():.0%}, from bone or contrast in {1 - table.scaling_from_air.mean():.0%}")
    for t in TYPES:
        levels, top = table[f"{t}_gray_levels"], table[f"{t}_fullest_bin_share"]
        print(f"  {t:12s} gray levels median {levels.median():3.0f} (IQR {levels.quantile(.25):.0f}-{levels.quantile(.75):.0f}), "
              f"<= 3 levels in {np.mean(levels <= 3):4.0%} of patients; fullest bin {top.median():.1%}")


if __name__ == "__main__":
    main()
