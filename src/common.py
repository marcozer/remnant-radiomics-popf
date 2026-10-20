"""Input data, ComBat harmonization, outcome-blind prefilter and display constants."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import io
import json
import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
RESULTS = Path(os.environ.get("RESULTS_DIR", REPO / "results"))
REFERENCE_BATCH = "GE_STANDARD_std"
# Stabl on all patients (03) and inside every training fold (09): randomized LASSO, 1,000 subsamples of 75% per run, features
# ranked by mean stability over STABL_RUNS runs; the number of features is chosen by inner cross-validation (src/panel_size.py).
STABL_RUNS = 50
NORD = {"ink": "#2E3440", "slate": "#4C566A", "grid": "#D8DEE9", "light": "#ECEFF4", "blue": "#5E81AC", "frost": "#81A1C1",
        "cyan": "#88C0D0", "teal": "#8FBCBB", "green": "#A3BE8C", "yellow": "#EBCB8B", "orange": "#D08770", "red": "#BF616A",
        "purple": "#B48EAD", "paper": "#FFFFFF"}

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


@functools.lru_cache(maxsize=None)
def signature():
    """Radiomic signature (03_feature_selection.py): the k most stable Stabl features on all patients, k chosen by inner
    cross-validation on all patients."""
    summary = json.loads((RESULTS / "feature_selection/summary.json").read_text())
    panel = summary["signature"]
    if not panel:
        raise RuntimeError("empty signature")
    return list(panel)


def model_name():
    """k-rad, k = number of signature features."""
    return f"{len(signature())}-rad"


def signature_words():
    return NUMBER_WORDS.get(len(signature()), str(len(signature())))


def display():
    """Model names as in the manuscript (keys: names used in the result files)."""
    rad = model_name()
    return {rad: rad, f"{rad} + MPD/thickness": f"{rad} + D-FRS", f"{rad} + DISPAIR covariates": f"{rad} + DISPAIR-FRS",
            "D-FRS (published)": "D-FRS (published)", "DISPAIR-FRS 2025 (published)": "DISPAIR-FRS (published)",
            "rD-FRS": "rD-FRS", "rDISPAIR-FRS": "rDISPAIR-FRS"}


def verify_checksums():
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"data/{name} differs from data/SHA256SUMS")


def load():
    """Cohort: patients in analysis order (row order defines folds and bootstrap samples)."""
    verify_checksums()
    cohort = pd.read_csv(DATA / "cohort.csv", float_precision="round_trip")
    radiomics = pd.read_csv(DATA / "radiomics.csv", index_col="patient_id", float_precision="round_trip")
    if list(radiomics.index) != list(cohort.patient_id):
        raise RuntimeError("radiomics and cohort rows differ")
    clinical = pd.read_csv(DATA / "clinical.csv", index_col="patient_id", float_precision="round_trip").loc[cohort.patient_id]
    scanners = pd.read_csv(DATA / "scanners.csv", index_col="patient_id", float_precision="round_trip").loc[cohort.patient_id]
    return SimpleNamespace(ids=cohort.patient_id.to_numpy(), y=cohort.cr_popf.to_numpy().astype(int), cohort=cohort, radiomics=radiomics,
                           batch=cohort.combat_batch.to_numpy().astype(str), clinical=clinical, scanners=scanners)


class Harmonizer:
    """neuroCombat (Fortin et al.) by reconstruction protocol with a reference batch; shape features are not harmonized.
    Estimates come from the training rows only; other rows are mapped with neuroCombatFromTraining."""

    def __init__(self, reference=REFERENCE_BATCH):
        self.reference = reference

    def fit_transform(self, frame, batch):
        if not hasattr(np, "int"):
            np.int = int  # neuroCombat 0.2.12 uses np.int when ref_batch is set
        from neuroCombat import neuroCombat

        self.columns = [c for c in frame.columns if not c.startswith("original_shape")]
        self.code = {level: i for i, level in enumerate(sorted(set(batch)))}
        covars = pd.DataFrame({"batch": [self.code[b] for b in batch]})
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = neuroCombat(dat=frame[self.columns].to_numpy(float).T, covars=covars, batch_col="batch",
                                 ref_batch=self.code[self.reference], mean_only=False)
        self.estimates = result["estimates"]
        out = frame.copy()
        values = result["data"].T
        out[self.columns] = np.where(np.isfinite(values), values, frame[self.columns].to_numpy(float))
        return out

    def transform(self, frame, batch):
        from neuroCombat import neuroCombatFromTraining

        out = frame.copy()
        seen = np.array([b in self.code for b in batch], dtype=bool)
        if not seen.all():
            warnings.warn(f"{int((~seen).sum())} rows from protocols absent from the training rows are left unharmonized")
        if seen.any():
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = neuroCombatFromTraining(dat=frame.loc[seen, self.columns].to_numpy(float).T,
                                                 batch=np.array([self.code[b] for b in batch[seen]]).astype(str),
                                                 estimates=self.estimates)["data"].T
            original = frame.loc[seen, self.columns].to_numpy(float)
            out.loc[seen, self.columns] = np.where(np.isfinite(values), values, original)
        return out


def combat(frame, batch, train, test):
    """ComBat fitted on the rows `train`, applied to the rows `test`."""
    h = Harmonizer()
    return h.fit_transform(frame.iloc[train], batch[train]), h.transform(frame.iloc[test], batch[test])


def harmonize_all(frame, batch):
    everyone = np.arange(len(frame))
    return combat(frame, batch, everyone, np.array([], int))[0]


def prefilter(frame, correlation=0.80, variance=0.01, missing=0.3):
    """Outcome-blind filter: missing fraction <= 0.3, variance > 0.01, then for each Spearman |rho| > threshold pair
    (row-major order) the lower-variance member is dropped unless one member is already dropped."""
    from scipy.stats import spearmanr

    names = np.asarray(frame.columns)
    X = frame.to_numpy(float)
    keep = np.isnan(X).mean(axis=0) <= missing
    X, names = X[:, keep], names[keep]
    median = np.nanmedian(X, axis=0)
    filled = np.where(np.isnan(X), median, X)
    keep = filled.var(axis=0) > variance
    X, names, filled = X[:, keep], names[keep], filled[:, keep]
    corr = np.abs(spearmanr(filled)[0])
    ii, jj = np.where(np.triu(corr > correlation, k=1))
    var = np.nanvar(X, axis=0)
    removed = set()
    for i, j in zip(ii.tolist(), jj.tolist()):
        if i not in removed and j not in removed:
            removed.add(i if var[i] < var[j] else j)
    return [n for k, n in enumerate(names) if k not in removed]


def standardize(X):
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (X - mean) / scale


def final_model():
    return json.loads((RESULTS / "final_model/model.json").read_text())


def final_predictions(d, spec=None):
    """Apparent probabilities of the final model (ComBat fitted on all patients)."""
    spec = spec or final_model()
    X = harmonize_all(d.radiomics, d.batch)[spec["signature"]].to_numpy(float)
    eta = ((X - np.array(spec["predictor_mean"])) / np.array(spec["predictor_sd"])) @ np.array(spec["coefficients"]) + spec["intercept"]
    return 1.0 / (1.0 + np.exp(-eta))


def family(feature):
    return "Original" if feature.startswith("original_") else "LoG-filtered" if feature.startswith("log-sigma") else "Wavelet-filtered"


def cohort_flow():
    """Flow chart of the cohort (data/flow.json, counts only): records, duplicate records, exclusions by group and reason."""
    verify_checksums()
    flow = json.loads((DATA / "flow.json").read_text())
    if flow["screened"] - sum(sum(g.values()) for g in flow["excluded"].values()) != flow["patients"]:
        raise RuntimeError("flow does not add up")
    return flow


CUTPOINTS = "oob_pooled"  # reported rule-out/rule-in cutpoints: chosen on pooled out-of-bag predictions (06); bootstrap_median is a sensitivity analysis

EXCLUDED_IMAGE_TYPES = {"square": "Square", "squareroot": "Square root", "logarithm": "Logarithm", "exponential": "Exponential",
                        "gradient": "Gradient magnitude", "lbp-2D": "Local binary patterns"}


def excluded_image_types():
    """Feature counts of the six image types excluded on technical grounds (data/excluded_image_types.csv, README)."""
    columns = pd.read_csv(DATA / "excluded_image_types.csv", nrows=0).columns.drop("patient_id")
    counts = pd.Series([c.split("_")[0] for c in columns]).value_counts()
    return {name: int(counts[key]) for key, name in EXCLUDED_IMAGE_TYPES.items()}


CLASS = {"firstorder": "first-order", "glcm": "GLCM", "gldm": "GLDM", "glrlm": "GLRLM", "glszm": "GLSZM", "ngtdm": "NGTDM", "shape": "shape"}
WORDS = {"MCC": "MCC", "Maximum2DDiameterColumn": "maximum 2D diameter (column)", "Maximum2DDiameterRow": "maximum 2D diameter (row)",
         "Maximum2DDiameterSlice": "maximum 2D diameter (slice)", "Maximum3DDiameter": "maximum 3D diameter", "Idmn": "IDMN", "Idn": "IDN",
         "Imc1": "IMC1", "Imc2": "IMC2", "Idm": "IDM", "Id": "ID", "10Percentile": "10th percentile", "90Percentile": "90th percentile"}


def short_name(feature):
    """Display name, e.g. log-sigma-7-0-mm-3D_ngtdm_Busyness -> LoG 7 mm NGTDM busyness."""
    prefix, klass, name = feature.split("_", 2)
    if prefix.startswith("log-sigma"):
        head = f"LoG {prefix.split('-')[2]} mm"
    elif prefix.startswith("wavelet"):
        head = f"Wavelet {prefix.split('-')[1]}"
    else:
        head = "Original" if klass != "shape" else "Shape"
    words = WORDS.get(name) or re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name).lower()
    for old, new in (("gray level", "gray-level"), ("non uniformity", "non-uniformity"), ("run length", "run-length")):
        words = words.replace(old, new)
    return f"{head} {CLASS[klass] + ' ' if klass != 'shape' else ''}{words}".replace("  ", " ")
