"""Radiomic features of the head-isthmus region, as in data/radiomics.csv.

Input: a CSV with columns patient_id, ct, mask (NIfTI paths; mask = head-isthmus region, any non-zero label).
Mask preparation, then PyRadiomics with radiomics_params.yaml:
  1. binarize the mask; a mask on the CT grid takes the CT geometry, any other mask is resampled onto the CT grid
     (nearest neighbor);
  2. keep the largest connected component;
  3. crop CT and mask to the bounding box of the mask plus a 40 mm margin.
With --params extraction/radiomics_params_excluded_types.yaml the same code extracts the six excluded image types
(data/excluded_image_types.csv). CT images are not shared; this script documents and reproduces the extraction.

  python3 extraction/extract_features.py cases.csv features.csv [--params extraction/radiomics_params.yaml]
"""

import argparse
import contextlib
import io
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MARGIN_MM = 40.0


def largest_component(mask):
    import SimpleITK as sitk

    array = sitk.GetArrayFromImage(sitk.ConnectedComponent(mask))
    sizes = np.bincount(array.ravel())[1:]
    keep = sitk.GetImageFromArray((array == (np.argmax(sizes) + 1)).astype(np.uint8))
    keep.CopyInformation(mask)
    return keep


def prepare(ct_path, mask_path):
    """CT and head-isthmus mask cropped to the mask bounding box plus MARGIN_MM."""
    import SimpleITK as sitk

    ct = sitk.ReadImage(str(ct_path))
    mask = sitk.Cast(sitk.ReadImage(str(mask_path)) > 0, sitk.sitkUInt8)
    if mask.GetSize() == ct.GetSize():
        mask.CopyInformation(ct)
    else:
        mask = sitk.Resample(mask, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    mask = largest_component(mask)
    array = sitk.GetArrayFromImage(mask)
    idx = np.argwhere(array > 0)
    margin = np.ceil(MARGIN_MM / np.array(ct.GetSpacing())[::-1]).astype(int)
    lower, upper = np.maximum(idx.min(axis=0) - margin, 0), np.minimum(idx.max(axis=0) + margin + 1, array.shape)
    size, index = [int(v) for v in (upper - lower)[::-1]], [int(v) for v in lower[::-1]]
    return sitk.RegionOfInterest(ct, size, index), sitk.RegionOfInterest(mask, size, index)


def extract(patient_id, ct_path, mask_path, params):
    import SimpleITK as sitk
    from radiomics import featureextractor

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    image, roi = prepare(ct_path, mask_path)
    extractor = featureextractor.RadiomicsFeatureExtractor(str(params))
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        values = {k: float(np.asarray(v).item()) for k, v in extractor.execute(image, roi).items() if not k.startswith("diagnostics_")}
    return patient_id, values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--params", type=Path, default=HERE / "radiomics_params.yaml")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cases = pd.read_csv(args.cases, dtype=str)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = dict(pool.map(extract, cases["patient_id"], cases["ct"], cases["mask"], [args.params] * len(cases)))
    table = pd.DataFrame.from_dict(rows, orient="index").loc[cases["patient_id"]]
    table.index.name = "patient_id"
    table.to_csv(args.output)
    print(f"{len(table)} patients x {table.shape[1]} features -> {args.output}")


if __name__ == "__main__":
    main()
