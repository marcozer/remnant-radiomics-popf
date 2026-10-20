#!/usr/bin/env python3
"""Table 2 (elastic-net specification of the signature model, k-rad), Supplementary Table 4 (the rule-out/rule-in risk groups of the final model,
reported cutpoints common.CUTPOINTS, in the study cohort and in the left-out patients of the nested bootstrap validation, 10b) and the
numbers reported in the manuscript text (results/key_numbers.json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402
import models  # noqa: E402

R = common.RESULTS
OUT = R / "tables"


def load(path):
    return json.loads((R / path).read_text())


def pct(x):
    return f"{100 * x:.1f}%"


def median_interval(s):
    return f"{100 * s['median']:.1f}% ({100 * s['ci'][0]:.1f}–{100 * s['ci'][1]:.1f})"


def risk_group_table(cut, sens):
    """Supplementary Table 4: the three risk groups of the reported cutpoints in the study cohort (final model) and in the patients left
    out of the bootstrap samples (nested validation, 10b): share of the patients, CR-POPF rate, and share of all CR-POPF (low-risk group:
    fistulas missed by the rule-out threshold; high-risk group: fistulas captured by the rule-in threshold)."""
    low, high = cut[common.CUTPOINTS]["Rule-out"], cut[common.CUTPOINTS]["Rule-in"]
    strata = cut[f"strata_{common.CUTPOINTS}"]
    n, e = sum(s["n"] for s in strata), sum(s["events"] for s in strata)
    left = sens["rules"][common.CUTPOINTS]["left_out_groups"]
    groups = ("low", "intermediate", "high")
    rows = [(f"Study cohort ({n} patients)", "", "", ""),
            ("Patients, n (%)", *[f"{s['n']} ({pct(s['n'] / n)})" for s in strata]),
            ("CR-POPF, n (% of the group)", *[f"{s['events']} ({pct(s['events'] / s['n'])})" for s in strata]),
            ("Share of all CR-POPF", *[pct(s["events"] / e) for s in strata]),
            ("Left-out patients, median (95% interval)", "", "", ""),
            ("Patients", *[median_interval(left[g]["share"]) for g in groups]),
            ("CR-POPF rate in the group", *[median_interval(left[g]["rate"]) for g in groups]),
            ("Share of all CR-POPF", *[median_interval(left[g]["share_of_cr_popf"]) for g in groups])]
    return pd.DataFrame(rows, columns=["Risk group", f"Low risk, probability <{low:.4f}", "Intermediate risk", f"High risk, probability ≥{high:.4f}"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec, final, boot = common.final_model(), load("final_model/summary.json"), load("bootstrap_validation/summary.json")
    cal = boot["calibration"]
    rows = [("Intercept", f"{spec['intercept']:.6f}", "-", "-")]
    rows += [(f, f"{c:.6f}", f"{m:.6g}", f"{s:.6g}") for f, c, m, s in zip(spec["signature"], spec["coefficients"], spec["predictor_mean"], spec["predictor_sd"])]
    rows += [("Calibration-in-the-large (apparent)", f"{round(cal['apparent_intercept'], 3) + 0.0:.3f}", "-", "-"),  # + 0.0: no "-0.000"
             ("Calibration slope (apparent)", f"{cal['apparent_slope']:.3f}", "-", "-"),
             ("Calibration slope (bootstrap optimism-corrected, 95% range)",
              f"{cal['optimism_corrected_slope']:.3f} ({cal['optimism_corrected_slope_ci'][0]:.3f}–{cal['optimism_corrected_slope_ci'][1]:.3f})", "-", "-")]
    pd.DataFrame(rows, columns=["Term", "Coefficient", "Predictor mean", "Predictor SD"]).to_csv(OUT / "table2.csv", index=False)

    cut = boot["cutpoints"]
    low, high = cut[common.CUTPOINTS]["Rule-out"], cut[common.CUTPOINTS]["Rule-in"]
    cut_sens = load("cutpoint_sensitivity/summary.json")
    risk_group_table(cut, cut_sens).to_csv(OUT / "supplementary_table4.csv", index=False)

    d = common.load()
    p = common.final_predictions(d)
    groups = models.risk_groups(p, low, high)
    grades = {g: {k: int(v) for k, v in d.cohort.popf_grade[(groups == j) & (d.y == 1)].value_counts().sort_index().items()}
              for j, g in enumerate(("Low", "Intermediate", "High"))}
    volume = d.radiomics["original_shape_VoxelVolume"] / 1000.0
    sel = load("feature_selection/summary.json")
    fam = pd.read_csv(R / "model_families/summary.csv", float_precision="round_trip").set_index("model")
    clin = pd.read_csv(R / "clinical_comparison/apparent_metrics.csv", float_precision="round_trip").set_index("model")
    paired = pd.read_csv(R / "clinical_comparison/paired_632plus_differences.csv", float_precision="round_trip").set_index("comparator")
    val = pd.read_csv(R / "clinical_comparison/bootstrap_632plus.csv", float_precision="round_trip").set_index("model")
    cs = load("clinical_comparison/summary.json")
    sens = pd.read_csv(R / "clinical_comparison/sensitivity_operative_mpd.csv", float_precision="round_trip").set_index("model")
    nested_summary, nmed = load("nested_cv/summary_10fold.json"), load("nested_cutpoints/summary.json")
    choice = load("model_choice/summary.json")

    def nested(s):
        out = {"design": s["design"], "stabl_runs": s["stabl_runs"]}
        for m in ["INNER-K", *(f"TOP{k}" for k in range(3, 11))]:
            r = s["methods"][m]
            out[m] = {"auc_mean": r["auc_mean"], "auc_mean_ci": r["auc_mean_ci"], "auc_sd": r["auc_sd"], "auc_range": r["auc_range"], "by_fold_seed": r["by_fold_seed"]}
        for rule in ("INNER-K", f"TOP{len(common.signature())}"):
            out[f"{rule} signature-feature reselection"] = s["selection"][rule]["signature_feature_frequency"]
            out[f"{rule} folds with the whole signature"] = s["selection"][rule]["folds_with_whole_signature"]
            out[f"{rule} panel size"] = s["selection"][rule]["panel_size"]
        out["INNER-K k chosen"] = s["selection"]["INNER-K"]["k_chosen"]
        return out

    harmonized = common.harmonize_all(d.radiomics, d.batch)
    low_variance_kept = common.prefilter(harmonized, correlation=np.inf)  # missingness and variance steps only
    candidates = common.prefilter(harmonized, correlation=0.80)
    assert len(candidates) == sel["after_prefilter"]
    prefilter = {"rule": "missing fraction <= 0.3; unstandardized variance > 0.01; for each Spearman |rho| > 0.80 pair the lower-variance member is "
                         "removed (features are kept unmodified, never merged)",
                 "after_variance": len(low_variance_kept), "removed_low_variance": int(d.radiomics.shape[1]) - len(low_variance_kept),
                 "removed_correlated": len(low_variance_kept) - len(candidates),
                 "candidates_by_class": {c: sum(f.split("_")[1] == c for f in candidates) for c in ("firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm")},
                 "candidates_by_image_type": {t: sum(common.family(f) == t for f in candidates) for t in ("Original", "LoG-filtered", "Wavelet-filtered")}}

    oob = cut["oob"]
    key = {
        "cohort": {"patients": int(len(d.y)), "cr_popf": int(d.y.sum())},
        "head_isthmus_volume_ml": {"median": float(volume.median()), "q1": float(volume.quantile(.25)), "q3": float(volume.quantile(.75))},
        "features": {"extracted_all_image_types": sel["extracted"] + sum(common.excluded_image_types().values()),
                     "excluded_image_types": common.excluded_image_types()}
                    | {k: sel[k] for k in ("extracted", "after_prefilter", "stabl_runs")}
                    | {"model": common.model_name(), "signature": common.signature(), "signature_rule": sel["signature_rule"], "inner_cv": sel["inner_cv"],
                       "prefilter": prefilter},
        "final_model": {"C": spec["C"], "l1_ratio": spec["l1_ratio"], "apparent_auc": final["apparent_auc"], "apparent_auc_ci": final["apparent_auc_ci"],
                        "brier": final["brier"], **cal},
        "auc_632plus": boot["auc_632plus_primary"] | {"replication_series": boot["auc_632plus_replication"]["auc"], "pooled_both_series": boot["auc_632plus_pooled"]["auc"]},
        "model_families": {m: {"auc_632plus": float(fam.loc[m, "auc_632plus"]), "ci": [float(fam.loc[m, "ci_low"]), float(fam.loc[m, "ci_high"])]} for m in fam.index},
        "clinical_comparison": {"patients": cs["complete_case_patients"], "events": cs["complete_case_events"],
                                "apparent_auc": {m: [float(clin.loc[m, "auc_apparent"]), float(clin.loc[m, "ci_low"]), float(clin.loc[m, "ci_high"])] for m in clin.index},
                                "auc_632plus_difference_vs_signature_model": {m: {"difference": float(r.difference_632plus), "range": [float(r.range_low), float(r.range_high)],
                                                                       "bootstrap_p": float(r.bootstrap_p)} for m, r in paired.iterrows()},
                                "auc_632plus": {m: [float(val.loc[m, "auc_632plus"]), float(val.loc[m, "ci_low"]), float(val.loc[m, "ci_high"])] for m in val.index},
                                "sensitivity_operative_mpd": {m: {"auc_apparent": [float(r.auc_apparent), float(r.ci_low), float(r.ci_high)],
                                                                  "delong_p_vs_signature": float(r.delong_p_vs_signature)} for m, r in sens.iterrows()}},
        "cutpoints": {"full_cohort": cut["full_cohort"], "strata_full_cohort": cut["strata_full_cohort"], "bootstrap_median": cut["bootstrap_median"],
                      "strata_bootstrap_median": cut["strata_bootstrap_median"],
                      "per_replicate_in_bag_cutpoints_out_of_bag": {"Rule-out": {m: oob["Rule-out"][m] for m in ("sensitivity", "npv", "specificity", "ppv")},
                                                                    "Rule-in": {m: oob["Rule-in"][m] for m in ("specificity", "ppv", "sensitivity", "npv")}},
                      "nested_median_rule": {s: {m: nmed["strategies"][s]["nested_median"][m] for m in ("sensitivity", "specificity", "ppv", "npv")}
                                             for s in ("Rule-out", "Rule-in")},
                      "reported_rule": common.CUTPOINTS, "oob_pooled": cut["oob_pooled"], "strata_oob_pooled": cut["strata_oob_pooled"],
                      "nested_oob_rule": {s: {m: nmed["strategies"][s]["nested_oob_rule"][m] for m in ("sensitivity", "specificity", "ppv", "npv")}
                                          for s in ("Rule-out", "Rule-in")},
                      "nested_target_met": {s: {"oob_pooled": nmed["strategies"][s]["oob_rule_target_met"], "bootstrap_median": nmed["strategies"][s]["oob_target_met"]}
                                            for s in ("Rule-out", "Rule-in")},
                      "nested_oob_strata": {"oob_pooled": nmed["pooled_oob_strata_oob_rule"], "bootstrap_median": nmed["pooled_oob_strata"]},
                      "strata_oob_pooled_grades": grades, "low_risk_cr_popf_probabilities": sorted(p[(groups == 0) & (d.y == 1)].tolist()),
                      "nested_oob_rule_rechosen_thresholds": {s: cut_sens["rules"]["oob_pooled"][s]["rechosen_threshold"] for s in ("Rule-out", "Rule-in")},
                      "nested_oob_rule_left_out_groups": cut_sens["rules"]["oob_pooled"]["left_out_groups"],
                      "nested_left_out_patients": cut_sens["left_out_patients"]},
        "cutpoint_sensitivity": {k: cut_sens[k] for k in ("samples", "inner_samples", "reported_target", "targets_evaluated", "unconstrained_mcc_maximum",
                                                      "rules", "targets")},
        "nested_cv": nested(nested_summary),
        "tuning_criterion_sensitivity": {"design": choice["design"], "reference": choice["reference"], "models": choice["models"]},
    }
    (R / "key_numbers.json").write_text(json.dumps(key, indent=2, default=float) + "\n")
    print(pd.read_csv(OUT / "table2.csv").to_string(index=False))
    print(pd.read_csv(OUT / "supplementary_table4.csv", keep_default_na=False).to_string(index=False))


if __name__ == "__main__":
    main()
