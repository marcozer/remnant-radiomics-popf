#!/usr/bin/env python3
"""Manuscript figures (SVG with editable text, PNG 300 dpi): Figure 1 (cohort and workflow), Figure 3 (model families and
ROC), Figure 4 (comparison with clinical scores), Figure 5 (risk-group heatmap), Supplementary Figures 1-3.
Figure 2 (segmentation examples on CT images) is not part of this repository."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402
import models  # noqa: E402

N = common.NORD
R = common.RESULTS
OUT = R / "figures"
FEATURE_COLORS = [N["blue"], N["red"], N["yellow"], N["green"], N["cyan"], N["purple"], N["orange"], N["teal"], N["slate"], N["frost"]]


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 13,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "text.color": N["ink"], "axes.labelcolor": N["ink"],
                         "axes.edgecolor": N["slate"], "xtick.color": N["ink"], "ytick.color": N["ink"], "figure.facecolor": "white",
                         "axes.facecolor": "white", "svg.hashsalt": "remnant-radiomics-popf"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white", metadata={"Date": None})
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def axis_style(ax, grid=True):
    if grid:
        ax.grid(color=N["grid"], alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(N["slate"])
        spine.set_linewidth(1.2)


def panel(ax, label, x=-0.12, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=22, fontweight="bold", color=N["ink"])


def load_json(path):
    return json.loads((R / path).read_text())


# ----------------------------------------------------------------------------- Figure 1
def box(ax, cx, top, w, title, lines=(), fill="#ECEFF4", edge="#6B7A8F", title_size=15, line_size=13, lw=2.2, pad=11, min_h=0):
    """Rounded box in point coordinates, height from its text; returns the bottom y."""
    heights = [1.32 * title_size] + [1.32 * line_size] * len(lines)
    h = max(min_h, 2 * pad + sum(heights))
    bottom = top - h
    ax.add_patch(FancyBboxPatch((cx - w / 2, bottom), w, h, boxstyle="round,pad=0,rounding_size=12", facecolor=fill, edgecolor=edge, linewidth=lw))
    y = top - (h - sum(heights)) / 2
    for k, (text, lh) in enumerate(zip([title, *lines], heights)):
        ax.text(cx, y - lh / 2, text, ha="center", va="center", fontsize=title_size if k == 0 else line_size,
                fontweight="bold" if k == 0 else "normal", color=N["ink"])
        y -= lh
    return bottom


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#4C566A", lw=2.2, mutation_scale=18, shrinkA=0, shrinkB=0))


def line(ax, xs, ys):
    ax.plot(xs, ys, color="#4C566A", lw=2.2, solid_capstyle="butt")


def figure1():
    f = common.cohort_flow()
    sel = load_json("feature_selection/summary.json")
    full = sel["extracted"] + sum(common.excluded_image_types().values())
    W, H, cx = 760, 1100, 225
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    top = H - 12
    b1 = box(ax, cx, top, 410, "Distal pancreatectomy with available CT",
             [f"{f['period']}: {f['records']} records, {f['duplicates']} duplicates", f"n={f['screened']} patients"])
    y = b1 - 26
    for group in ("clinical", "imaging"):
        items = f["excluded"][group]
        order = sorted(items, key=lambda k: (-items[k], k))
        bottom = box(ax, 590, y, 300, f"Excluded, {group} (n={sum(items.values())})", [f"{k} (n={items[k]})" for k in order],
                     fill="white", edge="#9AA5B8", title_size=13, line_size=12, lw=1.6)
        arrow(ax, cx, (y + bottom) / 2, 440, (y + bottom) / 2)
        y = bottom - 14
    an_top = y - 12
    line(ax, [cx, cx], [b1, an_top + 0.5])
    arrow(ax, cx, an_top + 14, cx, an_top)
    steps = [("Patients analyzed", [f"(n={f['patients']})"]),
             ("TotalSegmentator pancreas mask", ["Head-isthmus delineation", f"Reviewed in all {f['patients']} patients"]),
             ("PyRadiomics extraction", [f"{full:,} features, 9 image types"]),
             ("Outcome-blind preprocessing", [f"Technical exclusion of 6 image types: {sel['extracted']:,} features",
                                              "ComBat harmonization, refitted on each training set",
                                              f"Unsupervised filtering: {sel['after_prefilter']} features"]),
             ("Stabl selection", [f"{common.signature_words().capitalize()}-feature signature ({common.model_name()})",
                                  f"{len(common.signature())} most stable features of {sel['stabl_runs']} runs",
                                  "number chosen by inner cross-validation"])]
    y = an_top
    for k, (title, lines) in enumerate(steps):
        bottom = box(ax, cx, y, 410, title, lines, title_size=15, line_size=13.5)
        y = bottom - 20
        if k + 1 < len(steps):
            arrow(ax, cx, bottom, cx, y)
    split = bottom - 22
    left, right = 115, 345
    line(ax, [cx, cx], [bottom, split])
    line(ax, [left, right], [split, split])
    ax.plot([cx], [split], "o", color="#4C566A", markersize=5)
    arrow(ax, left, split, left, split - 20)
    arrow(ax, right, split, right, split - 20)
    n, events = f["patients"], f["cr_popf"]
    box(ax, left, split - 20, 210, "No CR-POPF", [f"{n - events} ({100 * (n - events) / n:.1f}%)", ""], title_size=16, line_size=14)
    last = box(ax, right, split - 20, 230, "CR-POPF", [f"{events} ({100 * events / n:.1f}%)", f"Grade B: {f['grade_b']} | Grade C: {f['grade_c']}"],
               fill="#DFE5EE", edge="#5E81AC", title_size=16, line_size=14)
    ax.set_ylim(last - 10, H)
    fig.set_size_inches(W / 72, (H - last + 10) / 72)
    save(fig, "Figure_1_Study_Cohort_and_Workflow")


# ----------------------------------------------------------------------------- Figure 3
def figure3():
    fam = pd.read_csv(R / "model_families/summary.csv", float_precision="round_trip").set_index("model")
    boot = load_json("bootstrap_validation/summary.json")
    final = load_json("final_model/summary.json")
    order = ["EN"] + fam.drop(index="EN").sort_values("auc_632plus", ascending=False).index.tolist()
    d = common.load()
    p = common.final_predictions(d)
    roc = pd.read_csv(R / "bootstrap_validation/weighted_roc.csv", float_precision="round_trip")
    primary = boot["auc_632plus_primary"]
    fig = plt.figure(figsize=(16.0, 7.6))
    ax, cax, ax_roc = fig.add_axes([0.075, 0.13, 0.30, 0.79]), fig.add_axes([0.395, 0.13, 0.013, 0.79]), fig.add_axes([0.52, 0.13, 0.45, 0.79])
    cmap = LinearSegmentedColormap.from_list("nord_auc", ["#F2F4F8", "#88C0D0", "#5E81AC", "#2E4A72"])
    values = fam.loc[order, "auc_632plus"].to_numpy()
    if "EN" in fam.index:
        values[0] = primary["auc"]
    image = ax.imshow(values[:, None], cmap=cmap, vmin=0.55, vmax=0.85, aspect="auto")
    for i, name in enumerate(order):
        low, high = (primary["ci"] if name == "EN" else (fam.loc[name, "ci_low"], fam.loc[name, "ci_high"]))
        ax.text(0, i, f"{values[i]:.3f}\n({low:.3f}-{high:.3f})", ha="center", va="center", fontsize=16, fontweight="bold",
                color="white" if values[i] >= 0.70 else N["ink"])
    for i in range(1, len(order)):
        ax.axhline(i - 0.5, color="white", linewidth=0.8, alpha=0.6)
    ax.set_yticks(range(len(order)), order, fontsize=17, fontweight="bold")
    ax.set_xticks([0], ["Corrected AUC\n(95% range)"], fontsize=16)
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_color(N["slate"])
        spine.set_linewidth(1.2)
    cb = fig.colorbar(image, cax=cax, ticks=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    cb.ax.set_title("AUC", fontsize=15, pad=10)
    cb.outline.set_color(N["slate"])
    cb.ax.tick_params(labelsize=13)
    panel(ax, "A", x=-0.18, y=1.08)
    fpr, tpr, _ = roc_curve(d.y, p)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#6B7A8F", linewidth=1.5)
    ax_roc.plot(fpr, tpr, color=N["blue"], linewidth=3.2, label=f"Study cohort, AUC {final['apparent_auc']:.3f}")
    ax_roc.plot(roc.fpr, roc.tpr_primary, color=N["orange"], linewidth=2.4, linestyle=(0, (5, 2.5)),
                label=f"After .632+ optimism correction\nAUC {primary['auc']:.3f} (95% range, {primary['ci'][0]:.3f}-{primary['ci'][1]:.3f})")
    ax_roc.set(xlim=(0, 1), ylim=(0, 1.02))
    ax_roc.set_xlabel("1 - specificity", fontsize=19)
    ax_roc.set_ylabel("Sensitivity", fontsize=19)
    ax_roc.tick_params(labelsize=17)
    ax_roc.set_aspect("equal", adjustable="box")
    leg = ax_roc.legend(loc="lower right", fontsize=13, frameon=True, edgecolor=N["grid"])
    for t in leg.get_texts():
        t.set_fontweight("bold")
    axis_style(ax_roc)
    panel(ax_roc, "B", x=-0.2, y=1.08)
    save(fig, "Figure_3_Model_Families_and_ROC")


# ----------------------------------------------------------------------------- Figure 4
def figure4():
    rad = common.model_name()
    # model labels as in the manuscript
    LABEL = {rad: rad, f"{rad} + MPD/thickness": f"{rad} + D-FRS", f"{rad} + DISPAIR covariates": f"{rad} + DISPAIR-FRS",
             "D-FRS (published)": "Preoperative D-FRS (published)", "DISPAIR-FRS 2025 (published)": "DISPAIR-FRS 2025 (published)"}
    FIG4 = [(rad, N["blue"], "-"), (f"{rad} + MPD/thickness", N["orange"], "-"), (f"{rad} + DISPAIR covariates", N["purple"], "-"),
            ("D-FRS (published)", N["green"], (0, (5, 2))), ("DISPAIR-FRS 2025 (published)", N["red"], (0, (5, 2, 1.5, 2)))]
    metrics = pd.read_csv(R / "clinical_comparison/apparent_metrics.csv", float_precision="round_trip").set_index("model")
    curves = pd.read_csv(R / "clinical_comparison/apparent_roc.csv", float_precision="round_trip")
    bins = pd.read_csv(R / "clinical_comparison/calibration_groups.csv", float_precision="round_trip")
    summary = load_json("clinical_comparison/summary.json")
    fig = plt.figure(figsize=(16.5, 8.8))
    ax_roc = fig.add_axes([0.05, 0.33, 0.33, 0.60])
    ax_cal = fig.add_axes([0.45, 0.33, 0.42, 0.60])
    handles = []
    for name, color, ls in FIG4:
        c = curves[curves.model == name]
        row = metrics.loc[name]
        label = f"{LABEL[name]} | {row.auc_apparent:.3f} [{row.ci_low:.3f}-{row.ci_high:.3f}]"
        (h,) = ax_roc.plot(c.fpr, c.tpr, color=color, linestyle=ls, linewidth=2.6, label=label)
        handles.append(h)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color=N["slate"], linewidth=1.2)
    ax_roc.set(xlim=(0, 1), ylim=(0, 1.0), xlabel="False-positive rate (1 - specificity)", ylabel="True-positive rate (sensitivity)")
    ax_roc.set_title("ROC curves (study cohort)", fontsize=16, fontweight="bold")
    axis_style(ax_roc)
    panel(ax_roc, "A", x=-0.18, y=1.08)
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color=N["slate"], linewidth=1.3, label="Ideal")
    for name, color, _ in FIG4[:3]:
        g = bins[bins.model == name]
        ax_cal.errorbar(g.mean_predicted, g.observed, yerr=[g.observed - g.observed_ci_low, g.observed_ci_high - g.observed], color=color, marker="o",
                        markersize=8, markerfacecolor="white", markeredgewidth=2, linewidth=2.4, capsize=3, label=LABEL[name])
    top = float(max(bins.observed_ci_high.max(), bins.mean_predicted.max())) + 0.05
    ax_cal.set(xlim=(-0.02, min(1.0, bins.mean_predicted.max() + 0.12)), ylim=(-0.02, min(1.0, top)), xlabel="Mean predicted probability",
               ylabel="Observed CR-POPF rate")
    ax_cal.set_title("Calibration of the refitted models (study cohort)", fontsize=16, fontweight="bold")
    ax_cal.legend(loc="upper left", fontsize=10, frameon=False)
    axis_style(ax_cal)
    panel(ax_cal, "B", x=-0.12, y=1.08)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.46, 0.0), ncol=1, frameon=False, fontsize=13,
               title="AUC [95% bootstrap CI], study cohort",
               title_fontsize=14, handlelength=3.2)
    save(fig, "Figure_4_Clinical_Score_Comparison")


# ----------------------------------------------------------------------------- Figure 5
def figure5():
    boot = load_json("bootstrap_validation/summary.json")
    d = common.load()
    spec = common.final_model()
    p = common.final_predictions(d, spec)
    X = common.harmonize_all(d.radiomics, d.batch)[spec["signature"]].to_numpy(float)
    z = (X - np.array(spec["predictor_mean"])) / np.array(spec["predictor_sd"])
    low, high = boot["cutpoints"][common.CUTPOINTS]["Rule-out"], boot["cutpoints"][common.CUTPOINTS]["Rule-in"]
    groups = models.risk_groups(p, low, high)
    y = d.y
    fig = plt.figure(figsize=(16.4, 8.15))
    x0, x1 = 0.260, 0.895
    width = x1 - x0
    ax_group = fig.add_axes([x0, .765, width, .135])
    ax_outcome = fig.add_axes([x0, .680, width, .059])
    ax_prob = fig.add_axes([x0, .595, width, .059])
    ax_heat = fig.add_axes([x0, .080, width, .465])
    feature_map = LinearSegmentedColormap.from_list("nord_z", [N["blue"], "white", N["red"]])
    prob_map = LinearSegmentedColormap.from_list("nord_p", ["white", N["cyan"], N["blue"]])
    outcome_map = ListedColormap([N["grid"], N["red"]])
    fills = [N["cyan"], N["yellow"], N["orange"]]
    limits = [f"probability < {low:.4f}", f"probability {low:.4f} to < {high:.4f}", f"probability ≥ {high:.4f}"]
    for slot, label in enumerate(("Low risk", "Intermediate risk", "High risk")):
        idx = np.flatnonzero(groups == slot)
        order = idx[np.argsort(p[idx], kind="stable")]
        n, events = len(order), int(y[order].sum())
        edges = np.linspace(slot, slot + 1, n + 1)
        ax_group.axvspan(slot, slot + 1, color=fills[slot], alpha=.72 if slot != 2 else .82)
        ax_group.text(slot + .5, .76, label, ha="center", va="center", fontsize=17, fontweight="bold")
        ax_group.text(slot + .5, .48, limits[slot], ha="center", va="center", fontsize=13.5)
        ax_group.text(slot + .5, .19, f"{events}/{n} CR-POPF ({100 * events / n:.1f}%)", ha="center", va="center", fontsize=15.5, fontweight="bold")
        ax_outcome.pcolormesh(edges, [0, 1], y[order][None, :], cmap=outcome_map, vmin=0, vmax=1, shading="flat")
        pim = ax_prob.pcolormesh(edges, [0, 1], p[order][None, :], cmap=prob_map, vmin=0, vmax=1, shading="flat")
        him = ax_heat.pcolormesh(edges, np.arange(len(spec["signature"]) + 1), z[order].T, cmap=feature_map, vmin=-2.5, vmax=2.5, shading="flat")
    for a in (ax_group, ax_outcome, ax_prob, ax_heat):
        a.set_xlim(0, 3)
        a.set_xticks([])
        for spine in a.spines.values():
            spine.set_visible(False)
        for sep in (1, 2):
            a.axvline(sep, color="white", linewidth=3.3)
    for a, label in ((ax_group, "Risk group"), (ax_outcome, "Observed outcome"), (ax_prob, "Predicted probability")):
        a.set_ylim(0, 1)
        a.set_yticks([.5], [label], fontsize=19)
        a.tick_params(axis="y", length=0, pad=15)
    k = len(spec["signature"])
    ax_heat.set_ylim(k, 0)
    ax_heat.set_yticks(np.arange(k) + .5, [common.short_name(f) for f in spec["signature"]], fontsize=15)
    ax_heat.tick_params(axis="y", length=0, pad=8)
    ax_heat.set_ylabel("Radiomic feature", fontsize=19, labelpad=14)
    cax = fig.add_axes([.920, .595, .019, .144])
    cb = fig.colorbar(pim, cax=cax, ticks=[0, .5, 1])
    cb.ax.set_title("Probability", fontsize=14, pad=9)
    cb.ax.tick_params(labelsize=14)
    hax = fig.add_axes([.920, .080, .019, .465])
    hb = fig.colorbar(him, cax=hax, ticks=[-2, -1, 0, 1, 2])
    hb.set_label("Feature z score", fontsize=17, labelpad=9)
    hb.ax.tick_params(labelsize=15)
    fig.legend(handles=[Patch(facecolor=N["grid"], label="No CR-POPF"), Patch(facecolor=N["red"], label="CR-POPF")], loc="upper left",
               bbox_to_anchor=(0.085, 0.69), ncol=2, frameon=False, fontsize=12, handlelength=1.2, columnspacing=1.0)
    save(fig, "Figure_5_Risk_Group_Heatmap")


# ----------------------------------------------------------------------------- Supplementary Figure 1
def supplementary1():
    sel = load_json("feature_selection/summary.json")
    ranking = pd.read_csv(R / "feature_selection/stabl_stability.csv", float_precision="round_trip")
    top = ranking.sort_values("mean_stability", ascending=False, kind="stable").head(12)
    signature = common.signature()
    fig = plt.figure(figsize=(16.0, 7.6))
    ax_flow = fig.add_axes([0.0, 0.05, 0.235, 0.88])
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    full = sel["extracted"] + sum(common.excluded_image_types().values())
    stages = [("Radiomic extraction", f"{full:,} radiomics", "#EFE7EE", "#E2D3DD"),
              ("Prefiltering: technical\nexclusion of 6 image types", f"{sel['extracted']:,} radiomics", "#ECEFF4", "#D8DEE9"),
              ("Prefiltering: unsupervised\nfiltering", f"{sel['after_prefilter']} radiomics", "#E5F0F5", "#CDE2EB"),
              (f"Stabl ranking\nacross {sel['stabl_runs']} runs", f"{sel['after_prefilter']} radiomics ranked", "#F3E3DF", "#ECCFC9"),
              ("Number chosen by inner\ncross-validation", f"{len(signature)} radiomics", "#DFE5EE", "#CBD4E2")]
    ys = [0.92, 0.715, 0.51, 0.305, 0.10]
    for (title, sub, fill, edge), yy in zip(stages, ys):
        ax_flow.add_patch(FancyBboxPatch((0.06, yy - 0.075), 0.88, 0.15, boxstyle="round,pad=0,rounding_size=0.025", facecolor=fill, edgecolor=edge,
                                         linewidth=2))
        two = "\n" in title
        ax_flow.text(0.50, yy + (0.028 if two else 0.023), title, ha="center", va="center", fontsize=13, fontweight="bold")
        ax_flow.text(0.50, yy - (0.035 if two else 0.023), sub, ha="center", va="center", fontsize=13, fontweight="bold")
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax_flow, 0.50, a - 0.08, 0.50, b + 0.08)
    ax_flow.text(0.0, 1.04, "A", fontsize=22, fontweight="bold", transform=ax_flow.transAxes, va="top")

    ax = fig.add_axes([0.445, 0.11, 0.255, 0.82])
    pos = np.arange(len(top))[::-1]
    selected = top.feature.isin(signature).to_numpy()
    ax.barh(pos, top.mean_stability, color=np.where(selected, N["blue"], "#D8DEE9"), edgecolor=np.where(selected, "#4C6A92", "#4C566A"),
            linewidth=0.8, height=0.8, alpha=0.95)
    for yv, v in zip(pos, top.mean_stability):
        ax.text(v + 0.012, yv, f"{v:.2f}", va="center", fontsize=11, bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none"})
    ax.set_yticks(pos, [common.short_name(f) for f in top.feature], fontsize=11)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(f"Mean stability across {sel['stabl_runs']} Stabl runs", fontsize=14, labelpad=10)
    ax.grid(axis="x", color=N["grid"], alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(-0.80, 1.06, "B", fontsize=22, fontweight="bold", transform=ax.transAxes, va="top")

    ax = fig.add_axes([0.785, 0.60, 0.195, 0.31])
    inner = sel["inner_cv"]
    ks = sorted(int(k) for k in inner["inner_auc"])
    auc = [inner["inner_auc"][str(k)] for k in ks]
    best = inner["best_k"]
    ax.plot(ks, auc, color=N["slate"], linewidth=1.6, marker="o", markersize=6, markerfacecolor="white", markeredgecolor=N["slate"], zorder=2)
    ax.plot([best], [inner["inner_auc"][str(best)]], marker="o", markersize=11, color=N["blue"], markeredgecolor="#4C6A92", zorder=3)
    ax.annotate(f"{best} features", (best, inner["inner_auc"][str(best)]), xytext=(0, 12), textcoords="offset points", ha="center",
                fontsize=11, fontweight="bold", color="#4C6A92")
    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
    ax.set_ylim(min(auc) - 0.012, max(auc) + 0.03)
    ax.set_xlabel("Number of features", fontsize=13, labelpad=6)
    ax.set_ylabel("Inner cross-validated AUC", fontsize=12, labelpad=8)
    ax.grid(color=N["grid"], alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(-0.36, 1.18, "C", fontsize=22, fontweight="bold", transform=ax.transAxes, va="top")

    # D: strict nested cross-validation (the whole pipeline rebuilt in every training fold): number of training folds selecting
    # each signature feature (order of panel B) and the most selected other feature
    nested = load_json("nested_cv/summary_10fold.json")
    rule = f"TOP{len(signature)}"
    method, selection = nested["methods"][rule], nested["selection"][rule]
    folds = int(nested["design"].split("-fold")[0]) * len(method["per_repeat"])
    counts = {f: round(v * folds) for f, v in selection["most_selected"]}
    ordered = [f for f in top.feature if f in signature]
    others = [(f, c) for f, c in counts.items() if f not in signature]
    rows = [(common.short_name(f), counts[f], True) for f in ordered] + [(f"Any other feature (n={len(others)})", max(c for _, c in others), False)]
    ax = fig.add_axes([0.785, 0.11, 0.195, 0.31])
    pos = np.arange(len(rows))[::-1]
    for yv, (label, count, in_signature) in zip(pos, rows):
        ax.barh(yv, count, color=N["blue"] if in_signature else "#D8DEE9", edgecolor="#4C6A92" if in_signature else "#4C566A", linewidth=0.8, height=0.78, alpha=0.95)
        if in_signature:
            ax.text(2, yv, label, va="center", ha="left", fontsize=9.5, color="white")
            ax.text(count - 2, yv, f"{count}", va="center", ha="right", fontsize=9.5, color="white", fontweight="bold")
        else:
            ax.text(count + 2, yv, f"{label}: ≤{count}", va="center", ha="left", fontsize=9.5, color=N["ink"])
    ax.set_yticks([])
    ax.set_xlim(0, folds)
    ax.set_xticks([0, 25, 50, 75, folds])
    ax.set_xlabel(f"Training folds selecting the feature (of {folds})", fontsize=12, labelpad=6)
    ax.grid(axis="x", color=N["grid"], alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.text(-0.36, 1.18, "D", fontsize=22, fontweight="bold", transform=ax.transAxes, va="top")

    save(fig, "Supplementary_Figure_1_Stabl_Selection")


# ----------------------------------------------------------------------------- Supplementary Figure 2
def supplementary2():
    sel = load_json("feature_selection/summary.json")
    paths = pd.read_csv(R / "feature_selection/lasso_paths.csv", float_precision="round_trip")
    fig = plt.figure(figsize=(16.4, 6.6))
    ax_e = fig.add_axes([0.055, 0.20, 0.11, 0.72])
    ax_a = fig.add_axes([0.225, 0.20, 0.19, 0.72])
    ax_b = fig.add_axes([0.48, 0.20, 0.50, 0.72])
    colors = {"Original": "#7E9CBF", "LoG-filtered": N["red"], "Wavelet-filtered": N["orange"]}
    excluded = sum(common.excluded_image_types().values())
    kept = np.zeros(2)
    for fam, color in colors.items():
        v = sel["by_family"]["extracted"][fam]
        ax_e.bar(range(2), [v, v], bottom=kept, color=color, edgecolor=N["slate"], linewidth=0.9, width=0.75)
        kept += v
    ax_e.bar([0], [excluded], bottom=kept[:1], color=N["light"], edgecolor=N["slate"], linewidth=0.9, width=0.75, hatch="///")
    totals = [kept[0] + excluded, kept[1]]
    for i, total in enumerate(totals):
        ax_e.text(i, total + max(totals) * 0.03, f"{int(total):,}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax_e.set_xticks(range(2), ["All 9\nimage types", "Prefiltering:\ntechnical\nexclusion"], fontsize=11)
    ax_e.set_ylim(0, max(totals) * 1.15)
    ax_e.set_ylabel("Number of features")
    axis_style(ax_e)
    panel(ax_e, "A", x=-0.62, y=1.1)
    fig.add_artist(FancyArrowPatch((0.172, 0.56), (0.203, 0.56), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=18, color="#4C566A", lw=2.0))
    stages = [("after_prefilter", "Prefiltering:\nunsupervised\nfilter"), ("signature", "Stabl\nsignature")]
    by_family = dict(sel["by_family"], signature={fam: sum(common.family(f) == fam for f in common.signature()) for fam in colors})
    bottoms = np.zeros(len(stages))
    for fam, color in colors.items():
        values = np.array([by_family[k][fam] for k, _ in stages], float)
        ax_a.bar(range(len(stages)), values, bottom=bottoms, color=color, edgecolor=N["slate"], linewidth=0.9, width=0.7, label=fam)
        bottoms += values
    for i, total in enumerate(bottoms):
        ax_a.text(i, total + max(bottoms) * 0.04, f"{int(total)}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax_a.set_xticks(range(len(stages)), [s for _, s in stages], fontsize=11)
    ax_a.set_xlim(-0.6, len(stages) - 0.4)
    ax_a.set_ylim(0, max(bottoms) * 1.18)
    fig.text(0.235, 0.955, "Progressive radiomics feature reduction", ha="center", va="center", fontsize=15, fontweight="bold")
    handles, labels = ax_a.get_legend_handles_labels()
    handles.append(Patch(facecolor=N["light"], edgecolor=N["slate"], hatch="///"))
    labels.append("Excluded image types (6)")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.235, -0.04), ncol=2, frameon=False, fontsize=10.5)
    axis_style(ax_a)
    lam = paths["lambda"].to_numpy()
    for f in paths.columns[1:]:
        if f not in common.signature():
            ax_b.plot(lam, paths[f], color="#C9D1DE", linewidth=0.6, alpha=0.45, zorder=1)
    handles = []
    for f, color in zip(common.signature(), FEATURE_COLORS):
        (h,) = ax_b.plot(lam, paths[f], color=color, linewidth=2.6, alpha=0.95, zorder=3, label=common.short_name(f))
        handles.append(h)
    ax_b.axhline(0, color=N["slate"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax_b.set_xscale("log")
    ax_b.set_xlim(lam.min(), lam.max())
    lim = np.ceil(np.abs(paths.iloc[:, 1:].to_numpy()).max() * 5) / 5
    ax_b.set_ylim(-lim, lim)
    ax_b.set_xlabel("Regularization parameter, lambda")
    ax_b.set_ylabel("Standardized coefficient")
    ax_b.set_title("LASSO coefficient paths (Stabl panel highlighted)", fontsize=15, fontweight="bold")
    axis_style(ax_b)
    panel(ax_b, "B", x=-0.08, y=1.1)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.70, -0.04), ncol=2, frameon=False, fontsize=10.5)
    save(fig, "Supplementary_Figure_2_Feature_Reduction_and_LASSO")


# ----------------------------------------------------------------------------- Supplementary Figure 3
def supplementary3():
    rad = common.model_name()
    SUPP3 = [(k, common.display()[k]) for k in (rad, f"{rad} + MPD/thickness", f"{rad} + DISPAIR covariates", "rD-FRS", "rDISPAIR-FRS")]
    SUPP3_TITLES = dict(SUPP3)
    SUPP3_A = dict(SUPP3, **{"rD-FRS": "D-FRS refit", "rDISPAIR-FRS": "DISPAIR-FRS refit"})  # panel A labels
    val = pd.read_csv(R / "clinical_comparison/bootstrap_632plus.csv", float_precision="round_trip").set_index("model")
    curves = pd.read_csv(R / "clinical_comparison/apparent_roc.csv", float_precision="round_trip")
    boot = pd.read_csv(R / "clinical_comparison/bootstrap_weighted_roc.csv", float_precision="round_trip")
    summary = load_json("clinical_comparison/summary.json")
    fig = plt.figure(figsize=(16.4, 10.4))
    gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.42, left=0.11, right=0.98, top=0.93, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0])
    pos = np.arange(len(SUPP3))[::-1]
    for (name, label), yv in zip(SUPP3, pos):
        r = val.loc[name]
        ax.barh(yv, r.auc_632plus - 0.3, left=0.3, color=N["blue"] if name == rad else N["cyan"], alpha=0.85, height=0.55)
        # 95% range of the replicates, drawn on its own (the mean .632+ of a near-null refit can lie outside its range)
        ax.errorbar((r.ci_low + r.ci_high) / 2, yv, xerr=(r.ci_high - r.ci_low) / 2, color=N["slate"], capsize=3, linewidth=1.3)
        ax.plot(r.auc_apparent, yv, "o", markerfacecolor="white", markeredgecolor=N["orange"], markeredgewidth=1.8, markersize=8)
        ax.text(r.auc_632plus, yv + 0.38, f"{r.auc_632plus:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axvline(0.5, color=N["slate"], linestyle="--", linewidth=1)
    ax.set_yticks(pos, [SUPP3_A[name] for name, _ in SUPP3], fontsize=11)
    ax.set_xlim(0.3, 0.95)
    ax.set_xlabel("AUC")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=N["orange"], markeredgewidth=1.8,
                              markersize=8, label="AUC, study cohort")], loc="lower right", frameon=False, fontsize=11)
    ax.text(-0.35, 1.12, "A", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    slots = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for (name, _), (r, c), letter in zip(SUPP3, slots, "BCDEF"):
        a = fig.add_subplot(gs[r, c])
        cv = curves[curves.model == name]
        a.plot(cv.fpr, cv.tpr, color="#E8C3BC", linewidth=2.6, label=f"AUC (study cohort) {val.loc[name, 'auc_apparent']:.3f}")
        a.plot(boot.fpr, boot[name], color=N["blue"], linewidth=2.4, label=f"Bootstrap .632+ AUC {val.loc[name, 'auc_632plus']:.3f}")
        a.plot([0, 1], [0, 1], linestyle="--", color="#9AA5B8", linewidth=1)
        a.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 - specificity", ylabel="Sensitivity")
        a.set_title(SUPP3_TITLES[name], fontsize=13, fontweight="bold")
        a.set_aspect("equal", adjustable="box")
        a.legend(loc="lower right", frameon=False, fontsize=9.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.text(-0.3, 1.12, letter, transform=a.transAxes, fontsize=16, fontweight="bold", va="top")
    if summary["excluded_missing_covariates"]:
        fig.text(0.5, 0.005, f"Complete-case set: n={summary['complete_case_patients']}, {summary['complete_case_events']} CR-POPF",
                 ha="center", fontsize=11, color=N["slate"])
    save(fig, "Supplementary_Figure_3_Model_Validation")


def main():
    style()
    which = sys.argv[1:] or ["1", "3", "4", "5", "s1", "s2", "s3"]
    table = {"1": figure1, "3": figure3, "4": figure4, "5": figure5, "s1": supplementary1, "s2": supplementary2, "s3": supplementary3}
    for key in which:
        table[key]()
        print("figure", key, "done")


if __name__ == "__main__":
    main()
