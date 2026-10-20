#!/usr/bin/env python3
"""Risk groups of the reported cutpoints in left-out patients (Supplementary Table 4) and sensitivity analyses of the cutpoints,
on the 2,000 primary bootstrap samples of 06. In every
sample ComBat and the elastic net are refitted in bag with the hyperparameters 06 tuned in that sample (06 and 10 tune
identically; the in-bag cutpoints of 06 and the nested out-of-bag rule of 10 are checked to be reproduced exactly).
1. Cutpoints chosen on the study patients' own predictions (constrained MCC on the final model's predictions, 06 full_cohort)
   and the reported cutpoints (pooled out-of-bag predictions): value in the study cohort; optimism-corrected value (value in
   the study cohort minus the mean, over the samples, of the in-bag value minus the value on the original patients, the
   cutpoints being re-chosen in every sample by the same rule: in bag for the first rule, on the pooled out-of-bag predictions
   of 500 inner bootstrap refits (as in 10) for the reported rule); value in the patients left out of each sample.
2. Targets of 80%, 85% and 95% instead of 90% (rule-out sensitivity, rule-in specificity) for the reported rule: cutpoints on
   the out-of-bag predictions pooled over the 2,000 samples, risk groups of the final model, and the nested validation of 10
   (500 inner samples, same seeds) for every target."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402
import models  # noqa: E402

OUT = common.RESULTS / "cutpoint_sensitivity"
STRATEGIES = ("Rule-out", "Rule-in")
TARGET_METRIC = {"Rule-out": "sensitivity", "Rule-in": "specificity"}
TARGETS = (0.80, 0.85, 0.90, 0.95)
REPORTED = 0.90
INNER_SEED, INNER = 20260923, 500  # inner samples of 10_nested_cutpoints
GROUPS = ("low", "intermediate", "high")
SHOW = ("sensitivity", "specificity", "ppv", "npv", "mcc")
_D = {}


def init_worker():
    import numba

    numba.set_num_threads(1)
    _D["d"] = common.load()
    _D["sig"] = common.signature()


def metrics(y, p, threshold):
    return {k: v for k, v in models.threshold_metrics(y, p, threshold).items() if k in models.METRICS}


def job(b, in_bag, oob, C, l1_ratio):
    d, sig = _D["d"], _D["sig"]
    y = d.y
    h = common.Harmonizer()
    tr = h.fit_transform(d.radiomics.iloc[in_bag], d.batch[in_bag])[sig].to_numpy(float)
    te = h.transform(d.radiomics.iloc[oob], d.batch[oob])[sig].to_numpy(float)
    full = h.transform(d.radiomics, d.batch)[sig].to_numpy(float)
    yin, yoob = y[in_bag], y[oob]
    est = models.fit_en(tr, yin, C, l1_ratio)
    p_in, p_oob, p_full = est.predict_proba(tr), est.predict_proba(te), est.predict_proba(full)
    rng = np.random.default_rng(np.random.SeedSequence([INNER_SEED, b]))
    inner_y, inner_p = [], []
    for _ in range(INNER):
        s, _ = models.stratified_bootstrap_indices(yin, rng)
        model = models.fit_en(tr[s], yin[s], C, l1_ratio)
        unseen = np.flatnonzero(~np.isin(in_bag, in_bag[s]))  # outer in-bag patients with no copy in the inner sample, once each
        unseen = unseen[np.unique(in_bag[unseen], return_index=True)[1]]
        inner_y.append(yin[unseen])
        inner_p.append(model.predict_proba(tr[unseen]))
    iy, ip = np.concatenate(inner_y), np.concatenate(inner_p)
    rules = {"own_predictions": models.constrained_cutpoints(yin, p_in)}
    rules |= {f"oob_pooled_{t:.2f}": models.constrained_cutpoints_grid(iy, ip, target=t) for t in TARGETS}
    rows = []
    for rule, cut in rules.items():
        groups = models.risk_groups(p_oob, *cut)
        row = {"replicate": b, "rule": rule, "n_oob": len(oob), "events_oob": int(yoob.sum())}
        for k, g in enumerate(GROUPS):
            row[f"{g}_n"], row[f"{g}_events"] = int((groups == k).sum()), int(yoob[groups == k].sum())
        corrected = rule in ("own_predictions", f"oob_pooled_{REPORTED:.2f}")  # optimism correction: the own-predictions and reported rules
        for strategy, threshold in zip(STRATEGIES, cut):
            row[f"{strategy}_threshold"] = float(threshold)
            for where, (yy, pp) in (("in_bag", (yin, p_in)), ("original", (y, p_full)), ("oob", (yoob, p_oob))):
                if corrected or where == "oob":
                    row.update({f"{strategy}_{where}_{k}": v for k, v in metrics(yy, pp, threshold).items()})
        rows.append(row)
    return rows, oob, p_oob


def summary(values):
    v = pd.Series(values, dtype=float).dropna()
    return {"median": float(v.median()), "ci": [float(v.quantile(0.025)), float(v.quantile(0.975))], "mean": float(v.mean()), "samples": int(len(v))}


def strata(y, p, cut):
    groups = models.risk_groups(p, *cut)
    return [{"group": g.capitalize(), "n": int((groups == k).sum()), "events": int(y[groups == k].sum())} for k, g in enumerate(GROUPS)]


def left_out_groups(g):
    """Per risk group in the left-out patients: share of the patients, CR-POPF rate, and share of all CR-POPF (low-risk group: fistulas
    missed by the rule-out threshold, 1 - sensitivity; high-risk group: fistulas captured by the rule-in threshold, its sensitivity)."""
    return {k: {"share": summary(g[f"{k}_n"] / g.n_oob), "rate": summary(g[f"{k}_events"] / g[f"{k}_n"].where(g[f"{k}_n"] > 0)),
                "share_of_cr_popf": summary(g[f"{k}_events"] / g.events_oob)} for k in GROUPS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    d = common.load()
    y, p = d.y, common.final_predictions(d)
    boot = json.loads((common.RESULTS / "bootstrap_validation/summary.json").read_text())["cutpoints"]
    reps = pd.read_csv(common.RESULTS / "bootstrap_validation/replicates.csv", float_precision="round_trip")
    reps = reps[reps.series == "primary"].set_index("replicate").sort_index()
    outer = pd.read_csv(common.RESULTS / "nested_cutpoints/outer_replicates.csv", float_precision="round_trip")
    samples = models.paired_samples(y, len(reps), models.BOOTSTRAP_SEED)
    assert [a for a, _, _ in samples] == reps.attempt.tolist(), "bootstrap samples differ from 06"
    rows, pooled = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        for r, oob, p_oob in pool.map(job, range(len(samples)), [s[1] for s in samples], [s[2] for s in samples], reps.C.tolist(),
                                      reps.l1_ratio.tolist(), chunksize=10):
            rows += r
            pooled.append((oob, p_oob))
    reps_out = pd.DataFrame(rows)
    reps_out.to_csv(OUT / "replicates.csv", index=False)
    y_pool, p_pool = np.concatenate([y[o] for o, _ in pooled]), np.concatenate([q for _, q in pooled])

    own = reps_out[reps_out.rule == "own_predictions"].set_index("replicate").sort_index()
    reported = reps_out[reps_out.rule == f"oob_pooled_{REPORTED:.2f}"].set_index("replicate").sort_index()
    for s in STRATEGIES:  # the models of 06 and 10
        assert np.array_equal(own[f"{s}_threshold"].to_numpy(), reps[f"{s}_threshold"].to_numpy()), "in-bag cutpoints differ from 06"
        nested = outer[outer.strategy == s].set_index("replicate").sort_index()
        assert np.array_equal(reported[f"{s}_threshold"].to_numpy(), nested.oob_rule_threshold.to_numpy()), "nested cutpoints differ from 10"
        for m in models.METRICS:
            assert np.array_equal(reported[f"{s}_oob_{m}"].to_numpy(), nested[f"oob_rule_oob_{m}"].to_numpy(), equal_nan=True), "nested metrics differ from 10"
    full = {t: models.constrained_cutpoints_grid(y_pool, p_pool, target=t) for t in TARGETS}
    own_cut = models.constrained_cutpoints(y, p)
    assert len(p_pool) == boot["oob_pooled_predictions"] and full[REPORTED].tolist() == [boot["oob_pooled"][s] for s in STRATEGIES]
    assert own_cut.tolist() == [boot["full_cohort"][s] for s in STRATEGIES]

    rules = {}
    for name, cut, g in (("oob_pooled", full[REPORTED], reported), ("own_predictions", own_cut, own)):
        entry = {"thresholds": dict(zip(STRATEGIES, map(float, cut))), "study_cohort_strata": strata(y, p, cut),
                 "left_out_groups": left_out_groups(g)}
        for s, t in zip(STRATEGIES, cut):
            cohort = metrics(y, p, t)
            entry[s] = {"rechosen_threshold": summary(g[f"{s}_threshold"]), "target_met": float((g[f"{s}_oob_{TARGET_METRIC[s]}"] >= REPORTED).mean())}
            for m in SHOW:
                optimism = g[f"{s}_in_bag_{m}"] - g[f"{s}_original_{m}"]
                entry[s][m] = {"study_cohort": cohort[m], "optimism": float(optimism.mean()), "optimism_corrected": float(cohort[m] - optimism.mean()),
                               "left_out": summary(g[f"{s}_oob_{m}"])}
        rules[name] = entry
    targets = {}
    for t in TARGETS:
        g = reps_out[reps_out.rule == f"oob_pooled_{t:.2f}"]
        entry = {"thresholds": dict(zip(STRATEGIES, map(float, full[t]))), "study_cohort_strata": strata(y, p, full[t]),
                 "pooled_out_of_bag": {s: metrics(y_pool, p_pool, c) for s, c in zip(STRATEGIES, full[t])}, "left_out_groups": left_out_groups(g)}
        for s, c in zip(STRATEGIES, full[t]):
            entry[s] = {"study_cohort": metrics(y, p, c), "target_met": float((g[f"{s}_oob_{TARGET_METRIC[s]}"] >= t).mean()),
                        "rechosen_threshold": summary(g[f"{s}_threshold"]), **{m: summary(g[f"{s}_oob_{m}"]) for m in SHOW}}
        targets[f"{t:.2f}"] = entry
    grid = np.round(np.arange(0.0, 1.0 + 0.00025, 0.0005), 6)
    table = models.metric_arrays(y_pool, p_pool, grid)
    best = int(np.argmax(table["mcc"]))
    out = {"samples": len(samples), "inner_samples": INNER, "reported_target": REPORTED, "targets_evaluated": list(TARGETS),
           "left_out_patients": {"n": summary(reported.n_oob), "events": summary(reported.events_oob)},
           "unconstrained_mcc_maximum": {k: float(table[k][best]) for k in ("threshold", "mcc", "sensitivity", "specificity")},
           "rules": rules, "targets": targets, "minutes": round((time.time() - started) / 60, 1)}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    for name, e in rules.items():
        print(f"{name}: thresholds {e['thresholds']} | rule-out sensitivity cohort {e['Rule-out']['sensitivity']['study_cohort']:.3f}, corrected "
              f"{e['Rule-out']['sensitivity']['optimism_corrected']:.3f}, left out {e['Rule-out']['sensitivity']['left_out']['median']:.3f} "
              f"(target met {e['Rule-out']['target_met']:.4f}) | rule-in specificity corrected {e['Rule-in']['specificity']['optimism_corrected']:.3f}, "
              f"left out {e['Rule-in']['specificity']['left_out']['median']:.3f} (target met {e['Rule-in']['target_met']:.4f})")
    for t, e in targets.items():
        print(f"target {t}: thresholds {e['thresholds']['Rule-out']:.4f} / {e['thresholds']['Rule-in']:.4f} | strata {[(s['n'], s['events']) for s in e['study_cohort_strata']]} "
              f"| target met {e['Rule-out']['target_met']:.3f} / {e['Rule-in']['target_met']:.3f}")
    print(f"{out['minutes']} min")


if __name__ == "__main__":
    main()
