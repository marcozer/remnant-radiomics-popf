#!/usr/bin/env python3
"""Supplementary Table 1: CT scanner characteristics (DICOM metadata of the analyzed examinations)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402

OUT = common.RESULTS / "tables"
MANUFACTURER = {"GE MEDICAL SYSTEMS": "GE Medical Systems", "SIEMENS": "Siemens Healthineers", "Philips": "Philips Healthcare",
                "PHILIPS": "Philips Healthcare", "TOSHIBA": "Canon / Toshiba", "CANON MEDICAL SYSTEMS": "Canon / Toshiba", "CANON_MEC": "Canon / Toshiba"}
GROUPS = [  # (manufacturer, display, models)
    ("GE Medical Systems", "Revolution HD", ["Revolution HD"]),
    ("GE Medical Systems", "Revolution Apex", ["Revolution Apex"]),
    ("GE Medical Systems", "Revolution EVO / CT / Frontier / Maxima", ["Revolution EVO", "Revolution CT", "Revolution Frontier", "Revolution Maxima"]),
    ("GE Medical Systems", "Optima CT540 / CT660", ["Optima CT540", "Optima CT660"]),
    ("GE Medical Systems", "Discovery CT750 HD / RT", ["Discovery CT750 HD", "Discovery RT"]),
    ("Siemens Healthineers", "SOMATOM go.All / go.Top", ["SOMATOM go.All", "SOMATOM go.Top"]),
    ("Siemens Healthineers", "SOMATOM Definition AS / AS+ / Edge", ["SOMATOM Definition AS", "SOMATOM Definition AS+", "SOMATOM Definition Edge"]),
    ("Siemens Healthineers", "SOMATOM Drive / Force / X.cite", ["SOMATOM Drive", "SOMATOM Force", "SOMATOM X.cite"]),
    ("Siemens Healthineers", "SOMATOM Scope / Perspective / Emotion 16", ["SOMATOM Scope", "SOMATOM Perspective", "Emotion 16"]),
    ("Philips Healthcare", "Ingenuity CT", ["Ingenuity CT"]),
    ("Philips Healthcare", "Incisive CT", ["Incisive CT"]),
    ("Canon / Toshiba", "Aquilion Series (ONE, PRIME, Prime SP, Serve SP, Lightning)",
     ["Aquilion ONE", "Aquilion PRIME", "Aquilion Prime SP", "Aquilion Serve SP", "Aquilion Lightning"]),
]


def span(values, digits, exact=False):
    """min–max with `digits` decimals; exact=True adds the decimal a value needs (slice thickness 0.625 mm)."""
    values = values.dropna()
    low, high = values.min(), values.max()
    fmt = lambda v: f"{v:.{digits + 1}f}" if exact and round(v, digits) != v else f"{v:.{digits}f}"  # noqa: E731
    return fmt(low) if low == high else f"{fmt(low)}–{fmt(high)}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = common.load().scanners
    s["maker"] = s.manufacturer.map(lambda m: MANUFACTURER.get(m, MANUFACTURER.get(str(m).upper(), m)))
    rows, used = [], set()
    for maker, display, names in GROUPS:
        g = s[(s.maker == maker) & s.model.isin(names)]
        used |= set(g.index)
        rows.append({"Manufacturer": maker, "Model": display, "No. of Scans": len(g), "kVp Range": span(g.kvp, 0),
                     "Exposure Time (ms)": span(g.exposure_time_ms, 0), "Tube Current (mA)": span(g.tube_current_ma, 0),
                     "Slice Thickness (mm)": span(g.slice_thickness_mm, 2, exact=True), "Pixel Spacing (mm)": span(g.pixel_spacing_mm, 3)})
    if len(used) != len(s):
        raise RuntimeError(f"unassigned scanner models: {sorted(set(s.drop(index=list(used)).model))}")
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "supplementary_table1.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
