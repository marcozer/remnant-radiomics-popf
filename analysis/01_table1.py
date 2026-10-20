#!/usr/bin/env python3
"""Table 1: patient, operative, pathologic and postoperative characteristics by CR-POPF status. Median [IQR] and n (%) with
available-case denominators; Mann-Whitney U and Fisher exact tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402

OUT = common.RESULTS / "tables"
ROWS = [
    ("Age, years", "c", "age"), ("Male sex", "b", "male"), ("BMI, kg/m2", "c", "bmi"), ("Diabetes", "b", "diabetes"), ("Alcohol use", "b", "alcohol"),
    ("Tobacco use", "b", "tobacco"), ("ASA score ≥3", "b", "asa3"), ("Lesion size, mm", "c", "lesion_size"), ("Neck thickness, mm", "c", "neck_thickness"),
    ("MPD diameter, mm", "c", "mpd"),
    ("Lesion location", None, None),
    ("Head involvement", "b", "lesion_head"), ("Body involvement", "b", "lesion_body"), ("Tail involvement", "b", "lesion_tail"),
    ("Isthmus involvement", "b", "lesion_isthmus"),
    ("Operative Data", None, None),
    ("Minimally invasive approach", "b", "mis"), ("Operative duration, min", "c", "op_time"),
    ("Blood loss, mL", "c", "blood_loss"), ("Transfusion required", "b", "transfusion"), ("Soft pancreas", "b", "soft"),
    ("Pancreatic transection", None, None),
    ("Stapler closure", "b", "stapler"), ("Wirsung suture", "b", "wirsung_suture"), ("Drainage placed", "b", "drain"),
    ("Spleen preserved", "b", "spleen_preserved"), ("Splenic vessels preserved", "b", "vessels_preserved"),
    ("Intraoperative somatostatin", "b", "somatostatin"),
    ("Pathology", None, None),
    ("Adenocarcinoma", "p", "Adenocarcinoma"), ("Mucinous Cystadenoma", "p", "Mucinous Cystadenoma"), ("Chronic Pancreatitis", "p", "Chronic Pancreatitis"),
    ("IPMN", "p", "IPMN"), ("SPT", "p", "SPT"), ("Neuroendocrine Tumor", "p", "Neuroendocrine Tumor"), ("Other", "p", "Other"),
    ("Postoperative Outcomes", None, None),
    ("Mortality", "b", "mortality"), ("Clavien-Dindo grade ≥III", "b", "clavien3"), ("Any POPF (biochemical leak or grade B/C)", "b", "any_popf"),
    ("CR-POPF (grade B/C)", "b", "cr"), ("POPF-related collection", "b", "popf_collection"), ("Antibiotic therapy", "b", "antibiotics"),
    ("Collection requiring drainage", "b", "collection_drained"), ("Bedside drainage", "b", "drain_bedside"), ("Endoscopic drainage", "b", "drain_endoscopic"),
    ("Radiological drainage", "b", "drain_radiological"), ("Reoperation / surgical drainage", "b", "drain_surgical"),
    ("Prolonged drainage", "b", "prolonged_drainage"),
]


def fmt_cont(s):
    s = s.dropna()
    return f"{s.median():.1f} [{s.quantile(.25):.1f}–{s.quantile(.75):.1f}]"


def fmt_bin(s):
    s = s.dropna().astype(bool)
    return f"{int(s.sum())} ({100 * s.mean():.1f}%)"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def row(df, kind, col):
    s = df[col]
    a, b = s[~df.cr].dropna(), s[df.cr].dropna()
    if kind == "c":
        return [fmt_cont(s), fmt_cont(a), fmt_cont(b), fmt_p(mannwhitneyu(a, b, alternative="two-sided").pvalue), int(s.isna().sum())]
    a, b = a.astype(bool), b.astype(bool)
    p = fisher_exact([[int(b.sum()), int((~b).sum())], [int(a.sum()), int((~a).sum())]])[1]
    return [fmt_bin(s), fmt_bin(a), fmt_bin(b), fmt_p(p), int(s.isna().sum())]


def build(df):
    out = []
    for label, kind, col in ROWS:
        if kind is None:
            out.append([label, "", "", "", "", ""])
        elif kind == "p":
            out.append([label] + row(df.assign(_p=(df.pathology == col).astype(float)), "b", "_p"))
        else:
            out.append([label] + row(df, kind, col))
    n, n0, n1 = len(df), int((~df.cr).sum()), int(df.cr.sum())
    return pd.DataFrame(out, columns=["Patient Characteristics", f"Cohort (n={n})", f"No CR-POPF (n={n0})", f"CR-POPF (n={n1})", "P value", "Missing"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = common.load()
    df = d.clinical.copy()
    df["cr"] = d.y.astype(bool)
    table = build(df)
    table.to_csv(OUT / "table1.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
