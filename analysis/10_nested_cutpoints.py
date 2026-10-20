#!/usr/bin/env python3
"""Nested bootstrap validation of the cutpoint rule (constrained MCC on pooled out-of-bag predictions). Outer samples: the
2,000 primary bootstrap samples of 06; in each, the model is refitted in bag as in 06_bootstrap_validation (ComBat + elastic
net tuned as the final model). In each of 500 inner class-stratified bootstrap samples of the outer in-bag patients, the
elastic net is refitted with the outer hyperparameters on the outer-harmonized features and predicts its own out-of-bag
patients (outer in-bag patients with no copy in the inner sample, each once); the constrained-MCC cutpoints chosen on these
pooled inner out-of-bag predictions are applied unchanged to the outer out-of-bag patients.
Sensitivity analyses, evaluated the same way: the medians of the constrained-MCC cutpoints chosen in each inner sample, and
cutpoints chosen once on the outer in-bag predictions. In the first 50 outer samples, 2,000 inner samples check that the
medians are stable at 500."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402
import models  # noqa: E402

OUT = common.RESULTS / "nested_cutpoints"
INNER_SEED = 20260923
STRATEGIES = ("Rule-out", "Rule-in")
_D = {}


def init_worker():
    import numba

    numba.set_num_threads(1)
    _D["d"] = common.load()


def job(b, attempt, in_bag, oob, inner, draws, trials):
    d, y = _D["d"], _D["d"].y
    tr, te = common.combat(d.radiomics, d.batch, in_bag, oob)
    Xin, Xoob = tr[common.signature()].to_numpy(float), te[common.signature()].to_numpy(float)
    yin, yoob = y[in_bag], y[oob]
    est, params, _, _ = models.tune_final(Xin, yin, in_bag, trials=trials)
    p_in, p_oob = est.predict_proba(Xin), est.predict_proba(Xoob)
    single = models.constrained_cutpoints(yin, p_in)
    rng = np.random.default_rng(np.random.SeedSequence([INNER_SEED, b]))
    selected, inner_y, inner_p = np.empty((draws, 2)), [], []
    for i in range(draws):
        s, _ = models.stratified_bootstrap_indices(yin, rng)
        model = models.fit_en(Xin[s], yin[s], params["C"], params["l1_ratio"])
        selected[i] = models.constrained_cutpoints(yin[s], model.predict_proba(Xin[s]))
        if i < inner:
            # out of bag = outer in-bag patients with no copy in the inner sample, one position per patient
            unseen = np.flatnonzero(~np.isin(in_bag, in_bag[s]))
            unseen = unseen[np.unique(in_bag[unseen], return_index=True)[1]]
            inner_y.append(yin[unseen])
            inner_p.append(model.predict_proba(Xin[unseen]))
    medians = {n: np.median(selected[:n], axis=0) for n in sorted({100, 250, inner, draws}) if n <= draws}
    median = medians[inner]
    out_of_bag = models.constrained_cutpoints_grid(np.concatenate(inner_y), np.concatenate(inner_p))
    groups = models.risk_groups(p_oob, *median)
    groups_oob_rule = models.risk_groups(p_oob, *out_of_bag)
    rows = []
    for j, strategy in enumerate(STRATEGIES):
        row = {"replicate": b, "strategy": strategy, "n_oob": len(oob), "events_oob": int(yoob.sum()), "median_threshold": float(median[j]),
               "single_threshold": float(single[j]), "oob_rule_threshold": float(out_of_bag[j]),
               **{f"median_inner_{n}": float(m[j]) for n, m in medians.items()}}
        row.update({f"oob_{k}": v for k, v in models.threshold_metrics(yoob, p_oob, median[j]).items() if k in models.METRICS})
        row.update({f"single_oob_{k}": v for k, v in models.threshold_metrics(yoob, p_oob, single[j]).items() if k in models.METRICS})
        row.update({f"oob_rule_oob_{k}": v for k, v in models.threshold_metrics(yoob, p_oob, out_of_bag[j]).items() if k in models.METRICS})
        row.update({f"median_in_bag_{k}": v for k, v in models.threshold_metrics(yin, p_in, median[j]).items() if k in ("sensitivity", "specificity")})
        rows.append(row)
    strata = [{"replicate": b, "rule": rule, "group": g, "n": int((gr == k).sum()), "events": int(yoob[gr == k].sum())}
              for rule, gr in (("median", groups), ("oob_pooled", groups_oob_rule)) for k, g in enumerate(("Low", "Intermediate", "High"))]
    return rows, strata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=int, default=2000)
    parser.add_argument("--inner", type=int, default=500)
    parser.add_argument("--confirmation-outer", type=int, default=50)
    parser.add_argument("--confirmation-inner", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    y = common.load().y
    samples = models.paired_samples(y, args.outer, models.BOOTSTRAP_SEED)
    started, rows, strata = time.time(), [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        futures = [pool.submit(job, b, attempt, ib, ob, args.inner, args.confirmation_inner if b < args.confirmation_outer else args.inner, args.trials)
                   for b, (attempt, ib, ob) in enumerate(samples)]
        for n, f in enumerate(as_completed(futures), 1):
            r, s = f.result()
            rows += r
            strata += s
            if n % 250 == 0:
                print(f"{n}/{len(futures)} outer samples ({(time.time() - started) / 60:.1f} min)", flush=True)
    rows = pd.DataFrame(rows).sort_values(["replicate", "strategy"])
    rows.to_csv(OUT / "outer_replicates.csv", index=False)
    strata = pd.DataFrame(strata)
    strata.to_csv(OUT / "outer_strata.csv", index=False)
    summary = {"outer": args.outer, "inner": args.inner, "confirmation": [args.confirmation_outer, args.confirmation_inner], "strategies": {}}
    for strategy in STRATEGIES:
        g = rows[rows.strategy == strategy]
        target = "sensitivity" if strategy == "Rule-out" else "specificity"
        entry = {"median_threshold": {"median": float(g.median_threshold.median()), "ci": g.median_threshold.quantile([0.025, 0.975]).tolist()},
                 "oob_rule_threshold": {"median": float(g.oob_rule_threshold.median()), "ci": g.oob_rule_threshold.quantile([0.025, 0.975]).tolist()},
                 "in_bag_target_met": float((g[f"median_in_bag_{target}"] >= 0.9).mean()), "oob_target_met": float((g[f"oob_{target}"] >= 0.9).mean()),
                 "oob_rule_target_met": float((g[f"oob_rule_oob_{target}"] >= 0.9).mean())}
        for method, prefix in (("nested_median", "oob_"), ("nested_oob_rule", "oob_rule_oob_"), ("single_constrained_mcc", "single_oob_")):
            entry[method] = {m: {"mean": float(g[prefix + m].mean()), "median": float(g[prefix + m].median()),
                                 "ci": g[prefix + m].dropna().quantile([0.025, 0.975]).tolist()} for m in models.METRICS}
        conf = g[g.replicate < args.confirmation_outer]
        entry[f"inner_{args.inner}_vs_{args.confirmation_inner}_median_abs_difference"] = float((conf[f"median_inner_{args.confirmation_inner}"] - conf[f"median_inner_{args.inner}"]).abs().median())
        summary["strategies"][strategy] = entry
    for rule, key in (("median", "pooled_oob_strata"), ("oob_pooled", "pooled_oob_strata_oob_rule")):
        oob_groups = strata[strata.rule == rule].groupby("group")[["n", "events"]].sum()
        summary[key] = {g: {"n": int(oob_groups.loc[g, "n"]), "events": int(oob_groups.loc[g, "events"]),
                            "rate": float(oob_groups.loc[g, "events"] / oob_groups.loc[g, "n"])} for g in ("Low", "Intermediate", "High")}
    summary["minutes"] = round((time.time() - started) / 60, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1)[:3000])


if __name__ == "__main__":
    main()
