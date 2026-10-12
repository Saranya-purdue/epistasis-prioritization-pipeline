#!/usr/bin/env python3
from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

parser = argparse.ArgumentParser(description="Plot Benchmark B manuscript figures.")
parser.add_argument("--base", required=True, help="Benchmark B base directory.")
parser.add_argument("--input", default=None, help="Optional benchmarkB_master_metrics.csv.")
parser.add_argument("--outdir", default=None, help="Optional figure output directory.")
args = parser.parse_args()

BENCHB = Path(args.base).expanduser().resolve()
SUMMARY_DIR = BENCHB / "benchmark_B_summary"
INPUT = Path(args.input).expanduser().resolve() if args.input else SUMMARY_DIR / "benchmarkB_master_metrics.csv"
OUT = Path(args.outdir).expanduser().resolve() if args.outdir else SUMMARY_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "Full 10k -> RF-SHAP": "#35618F",
    "Weighted VAE 2k -> RF-SHAP": "#7AA6C2",
    "Full 10k -> RF-SHAP-Taylor": "#B55D3F",
    "Weighted VAE 2k -> RF-SHAP-Taylor": "#D99A7C",
    "Full 10k -> MatrixEpistasis": "#6D6D6D",
    "Full 10k -> PLINK": "#B5B5B5",
}

ARCH_COLORS = {
    "Joint dominant": "#3B6FB6",
    "Joint recessive": "#B65C4A",
    "Multiplicative": "#6A8E3A",
    "Exponential": "#8D68A8",
}

PIPELINE_ORDER = [
    "Full 10k -> RF-SHAP",
    "Weighted VAE 2k -> RF-SHAP",
    "Full 10k -> RF-SHAP-Taylor",
    "Weighted VAE 2k -> RF-SHAP-Taylor",
    "Full 10k -> MatrixEpistasis",
    "Full 10k -> PLINK",
]

RF_PIPELINES = [
    "Full 10k -> RF-SHAP",
    "Weighted VAE 2k -> RF-SHAP",
    "Full 10k -> RF-SHAP-Taylor",
    "Weighted VAE 2k -> RF-SHAP-Taylor",
]

ARCH_ORDER = ["Joint dominant", "Joint recessive", "Multiplicative", "Exponential"]
PAIR_COUNTS = [10, 50, 100]

SHORT_LABEL = {
    "Full 10k -> RF-SHAP": "RF-SHAP\nFull 10k",
    "Weighted VAE 2k -> RF-SHAP": "RF-SHAP\nVAE 2k",
    "Full 10k -> RF-SHAP-Taylor": "Taylor\nFull 10k",
    "Weighted VAE 2k -> RF-SHAP-Taylor": "Taylor\nVAE 2k",
    "Full 10k -> MatrixEpistasis": "Matrix-\nEpistasis",
    "Full 10k -> PLINK": "PLINK",
}

df = pd.read_csv(INPUT)
required = {
    "Pair_count", "Architecture", "Pipeline",
    "PreRF_Pair_availability", "Pair_availability",
    "Recall_at_T", "Recall_at_5T", "Recall_at_10T",
    "Conditional_Recall_at_5T", "Normalized_AP",
    "Median_rank_percentile", "Runtime_seconds",
    "Peak_memory_MB", "N_input_snps", "Candidate_pair_space",
}
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Pair_count"] = pd.to_numeric(df["Pair_count"], errors="raise")
df["Peak_memory_GB"] = pd.to_numeric(df["Peak_memory_MB"], errors="coerce") / 1024.0
df["Runtime_hours"] = pd.to_numeric(df["Runtime_seconds"], errors="coerce") / 3600.0

N_TOTAL = 10_000
N_SELECTED = 2_000
RANDOM_PAIR_EXPECTED = (N_SELECTED / N_TOTAL) * ((N_SELECTED - 1) / (N_TOTAL - 1))

means = (
    df.groupby(["Pair_count", "Pipeline"], as_index=False)
      .agg(
          mean_preRF=("PreRF_Pair_availability", "mean"),
          mean_pair_avail=("Pair_availability", "mean"),
          mean_recall_T=("Recall_at_T", "mean"),
          mean_recall_5T=("Recall_at_5T", "mean"),
          mean_recall_10T=("Recall_at_10T", "mean"),
          mean_cond_5T=("Conditional_Recall_at_5T", "mean"),
          mean_nAP=("Normalized_AP", "mean"),
          mean_memory_GB=("Peak_memory_GB", "mean"),
          mean_runtime_h=("Runtime_hours", "mean"),
      )
)
means.to_csv(OUT / "benchmarkB_descriptive_plot_means.csv", index=False)

def save_both(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

def add_panel_label(ax, label):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")

def metric_architecture_figure(metric, ylabel, stem, ylim=None, pipelines=None):
    if pipelines is None:
        pipelines = PIPELINE_ORDER

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, arch in zip(axes, ARCH_ORDER):
        sub = df[df["Architecture"] == arch]
        for pipeline in pipelines:
            m = sub[sub["Pipeline"] == pipeline].sort_values("Pair_count")
            if len(m) == 0:
                continue
            ax.plot(
                m["Pair_count"], m[metric].astype(float),
                marker="o", ms=5, lw=1.8,
                color=COLORS[pipeline],
                label=SHORT_LABEL[pipeline].replace("\n", " "),
            )
        ax.set_title(arch)
        ax.set_xticks(PAIR_COUNTS)
        ax.set_xlabel("Number of true epistatic pairs")
        ax.grid(axis="y", alpha=0.18)
        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_both(fig, stem)

# ============================================================
# MAIN FIGURE
# ============================================================

fig = plt.figure(figsize=(13.8, 10.5))
gs = fig.add_gridspec(
    2, 2, left=0.08, right=0.98, top=0.96, bottom=0.09,
    wspace=0.30, hspace=0.36
)

# A. Search-space compression
axA = fig.add_subplot(gs[0, 0])
add_panel_label(axA, "A")
pair_spaces = {"Full 10k": math.comb(10_000, 2), "VAE 2k": math.comb(2_000, 2)}
labels = list(pair_spaces.keys())
vals = np.array(list(pair_spaces.values()), dtype=float)
bars = axA.bar(labels, vals, width=0.58, color=["#5E5E5E", "#8BAFC5"])
axA.set_yscale("log")
axA.set_ylabel("Theoretical SNP-pair search space")
axA.set_title("Search-space compression before interaction screening")
axA.grid(axis="y", alpha=0.18)

for bar, val in zip(bars, vals):
    axA.text(bar.get_x() + bar.get_width()/2, val * 1.16, f"{int(val):,}",
             ha="center", va="bottom", fontsize=9)

reduction = 1 - vals[1] / vals[0]
axA.text(0.5, 0.16, f"{reduction*100:.1f}% reduction",
         transform=axA.transAxes, ha="center", va="center",
         fontsize=11, fontweight="bold")
axA.text(0.5, 0.07, "10,000 → 2,000 SNPs",
         transform=axA.transAxes, ha="center", va="center", fontsize=9)

# B. Enrichment after weighted-VAE selection
axB = fig.add_subplot(gs[0, 1])
add_panel_label(axB, "B")
weighted = df[df["Pipeline"] == "Weighted VAE 2k -> RF-SHAP"].copy()
offsets = dict(zip(ARCH_ORDER, np.linspace(-0.12, 0.12, len(ARCH_ORDER))))

for i, n in enumerate(PAIR_COUNTS):
    sub = weighted[weighted["Pair_count"] == n]
    for arch in ARCH_ORDER:
        a = sub[sub["Architecture"] == arch]
        if len(a) == 0:
            continue
        axB.scatter(
            i + offsets[arch],
            float(a.iloc[0]["PreRF_Pair_availability"]),
            s=38, color=ARCH_COLORS[arch], alpha=0.80,
            edgecolor="white", linewidth=0.45, zorder=4
        )

    mean_val = sub["PreRF_Pair_availability"].mean()
    axB.scatter(i, mean_val, s=92, marker="D",
                facecolor="white", edgecolor="black",
                linewidth=1.3, zorder=6)
    axB.text(i, mean_val + 0.035, f"{mean_val*100:.1f}%",
             ha="center", va="bottom", fontsize=8.5)

axB.axhline(RANDOM_PAIR_EXPECTED, linestyle="--", linewidth=1.5,
            color="black", alpha=0.7)
axB.text(2.38, RANDOM_PAIR_EXPECTED + 0.008,
         f"Random 2k expectation ≈ {RANDOM_PAIR_EXPECTED*100:.1f}%",
         ha="right", va="bottom", fontsize=8.5)
axB.set_xticks(range(len(PAIR_COUNTS)))
axB.set_xticklabels(PAIR_COUNTS)
axB.set_xlabel("Number of true epistatic pairs")
axB.set_ylabel("True-pair availability after VAE selection")
axB.set_ylim(0, max(0.58, weighted["PreRF_Pair_availability"].max() + 0.08))
axB.set_title("Phenotype-weighted VAE enriches true interaction pairs")
axB.grid(axis="y", alpha=0.18)

arch_handles = [
    Line2D([0], [0], marker="o", linestyle="none",
           markerfacecolor=ARCH_COLORS[a], markeredgecolor="white",
           markersize=7, label=a)
    for a in ARCH_ORDER
]
mean_handle = Line2D([0], [0], marker="D", linestyle="none",
                     markerfacecolor="white", markeredgecolor="black",
                     markersize=7, label="Descriptive mean")
axB.legend(handles=arch_handles + [mean_handle], frameon=False,
           loc="upper right", fontsize=7.8)

# C. End-to-end Recall@5T
axC = fig.add_subplot(gs[1, 0])
add_panel_label(axC, "C")
recall_pipelines = RF_PIPELINES
x = np.arange(len(PAIR_COUNTS), dtype=float)
line_styles = {
    "Full 10k -> RF-SHAP": "-",
    "Weighted VAE 2k -> RF-SHAP": "--",
    "Full 10k -> RF-SHAP-Taylor": "-",
    "Weighted VAE 2k -> RF-SHAP-Taylor": "--",
}
offset_map = {
    "Full 10k -> RF-SHAP": -0.15,
    "Weighted VAE 2k -> RF-SHAP": -0.05,
    "Full 10k -> RF-SHAP-Taylor": 0.05,
    "Weighted VAE 2k -> RF-SHAP-Taylor": 0.15,
}

for pipeline in recall_pipelines:
    subp = df[df["Pipeline"] == pipeline]
    ymean = []
    for i, n in enumerate(PAIR_COUNTS):
        sub = subp[subp["Pair_count"] == n]
        ymean.append(sub["Recall_at_5T"].mean())
        for arch in ARCH_ORDER:
            a = sub[sub["Architecture"] == arch]
            if len(a) == 0:
                continue
            axC.scatter(
                i + offset_map[pipeline],
                float(a.iloc[0]["Recall_at_5T"]),
                s=24, color=COLORS[pipeline], alpha=0.33,
                edgecolor="none", zorder=2
            )
    axC.plot(
        x, ymean, marker="o", ms=6, lw=2,
        linestyle=line_styles[pipeline],
        color=COLORS[pipeline],
        label=SHORT_LABEL[pipeline].replace("\n", " "),
        zorder=5
    )

axC.set_xticks(x)
axC.set_xticklabels(PAIR_COUNTS)
axC.set_xlabel("Number of true epistatic pairs")
axC.set_ylabel("End-to-end Recall@5T")
axC.set_ylim(-0.02, 1.02)
axC.set_title("Recovery loss increases with interaction complexity")
axC.grid(axis="y", alpha=0.18)
axC.legend(frameon=False, loc="upper right", fontsize=7.7)
axC.text(
    0.02, 0.02,
    "Points: individual EpiGEN architectures\nLines: descriptive means across architectures",
    transform=axC.transAxes, fontsize=7.6, va="bottom"
)

# D. Peak memory
axD = fig.add_subplot(gs[1, 1])
add_panel_label(axD, "D")
memory_groups = [
    ("RF-SHAP", "Full 10k -> RF-SHAP", "Weighted VAE 2k -> RF-SHAP"),
    ("RF-SHAP-Taylor", "Full 10k -> RF-SHAP-Taylor",
     "Weighted VAE 2k -> RF-SHAP-Taylor"),
]
bar_width = 0.30
group_x = np.arange(len(memory_groups))

for gi, (method_label, full_pipe, weighted_pipe) in enumerate(memory_groups):
    full = df[df["Pipeline"] == full_pipe]["Peak_memory_GB"].dropna()
    red = df[df["Pipeline"] == weighted_pipe]["Peak_memory_GB"].dropna()
    full_mean = full.mean()
    red_mean = red.mean()

    bx1 = gi - bar_width/2
    bx2 = gi + bar_width/2

    axD.bar(bx1, full_mean, width=bar_width, color=COLORS[full_pipe], alpha=0.90)
    axD.bar(bx2, red_mean, width=bar_width, color=COLORS[weighted_pipe], alpha=0.90)

    jitter = np.linspace(-0.055, 0.055, len(full))
    axD.scatter(np.full(len(full), bx1) + jitter, full, s=16,
                color="black", alpha=0.35, zorder=4)
    jitter = np.linspace(-0.055, 0.055, len(red))
    axD.scatter(np.full(len(red), bx2) + jitter, red, s=16,
                color="black", alpha=0.35, zorder=4)

    mem_reduction = 1 - red_mean / full_mean
    ytop = max(full.max(), red.max())
    axD.text(gi, ytop + 0.55, f"{mem_reduction*100:.0f}% lower",
             ha="center", va="bottom", fontsize=9, fontweight="bold")

axD.set_xticks(group_x)
axD.set_xticklabels(["RF-SHAP", "RF-SHAP-Taylor"])
axD.set_ylabel("Peak memory (GB)")
axD.set_title("VAE-reduced input lowers downstream RF memory")
axD.grid(axis="y", alpha=0.18)
axD.legend(
    handles=[
        Patch(facecolor="#5E5E5E", label="Full 10k"),
        Patch(facecolor="#A4BED0", label="Weighted VAE 2k"),
    ],
    frameon=False, loc="upper right"
)
axD.text(
    0.02, 0.02,
    "Bars: descriptive mean across 12 EpiGEN conditions\nPoints: individual conditions",
    transform=axD.transAxes, fontsize=7.6, va="bottom"
)

fig.suptitle(
    "End-to-end EpiGEN benchmark of VAE-guided epistasis prioritization",
    y=0.995, fontsize=14
)
save_both(fig, "Figure_main_BenchmarkB_end_to_end")

# ============================================================
# SUPPLEMENT S1: weighted-VAE funnel
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharey=True)
weighted_methods = [
    ("Weighted VAE 2k -> RF-SHAP", "RF-SHAP"),
    ("Weighted VAE 2k -> RF-SHAP-Taylor", "RF-SHAP-Taylor"),
]
stage_cols = ["PreRF_Pair_availability", "Pair_availability", "Recall_at_5T"]
stage_labels = ["After VAE\n2k selection", "After RF\ntop-500", "Recovered\n@5T"]

for row, (pipeline, method_label) in enumerate(weighted_methods):
    p = df[df["Pipeline"] == pipeline]
    for col, n in enumerate(PAIR_COUNTS):
        ax = axes[row, col]
        sub = p[p["Pair_count"] == n]

        for arch in ARCH_ORDER:
            a = sub[sub["Architecture"] == arch]
            if len(a) == 0:
                continue
            vals = [float(a.iloc[0][c]) for c in stage_cols]
            ax.plot(range(3), vals, marker="o", lw=1.4, alpha=0.75,
                    color=ARCH_COLORS[arch], label=arch)

        mean_vals = [sub[c].mean() for c in stage_cols]
        ax.plot(range(3), mean_vals, marker="D", ms=6, lw=2.5,
                color="black", label="Descriptive mean", zorder=6)

        ax.set_xticks(range(3))
        ax.set_xticklabels(stage_labels)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.18)

        if row == 0:
            ax.set_title(f"{n} true pairs")
        if col == 0:
            ax.set_ylabel(f"{method_label}\nFraction of all true pairs")

handles = [
    Line2D([0], [0], color=ARCH_COLORS[a], marker="o", label=a)
    for a in ARCH_ORDER
]
handles.append(Line2D([0], [0], color="black", marker="D",
                      lw=2.5, label="Descriptive mean"))
fig.legend(handles=handles, loc="upper center", ncol=5,
           frameon=False, bbox_to_anchor=(0.5, 1.01))
fig.suptitle(
    "Weighted-VAE recovery funnel distinguishes selection, screening, and ranking losses",
    y=1.04, fontsize=13
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
save_both(fig, "FigureS1_weighted_VAE_recovery_funnel")

# ============================================================
# SUPPLEMENT S2-S7
# ============================================================

metric_architecture_figure(
    "Recall_at_T", "Recall@T",
    "FigureS2_architecture_Recall_at_T", (-0.02, 1.02)
)

metric_architecture_figure(
    "Recall_at_5T", "Recall@5T",
    "FigureS3_architecture_Recall_at_5T", (-0.02, 1.02)
)

metric_architecture_figure(
    "Recall_at_10T", "Recall@10T",
    "FigureS4_architecture_Recall_at_10T", (-0.02, 1.02)
)

metric_architecture_figure(
    "Normalized_AP", "Normalized average precision",
    "FigureS5_architecture_normalized_AP", (-0.03, 1.02)
)

metric_architecture_figure(
    "Pair_availability",
    "True-pair availability after method-specific screening",
    "FigureS6_architecture_pair_availability",
    (-0.02, 1.02),
    pipelines=RF_PIPELINES,
)

metric_architecture_figure(
    "Conditional_Recall_at_5T",
    "Conditional Recall@5T",
    "FigureS7_architecture_conditional_Recall_at_5T",
    (-0.02, 1.02),
    pipelines=PIPELINE_ORDER,
)

# ============================================================
# SUPPLEMENT S8: runtime + memory
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.5))
xpos = np.arange(len(PIPELINE_ORDER))

for i, pipeline in enumerate(PIPELINE_ORDER):
    sub = df[df["Pipeline"] == pipeline]

    vals = sub["Runtime_hours"].dropna().to_numpy()
    jitter = np.linspace(-0.08, 0.08, len(vals))
    ax1.scatter(np.full(len(vals), i) + jitter, vals, s=24,
                color=COLORS[pipeline], alpha=0.55, edgecolor="none")
    ax1.scatter(i, np.mean(vals), marker="D", s=62,
                facecolor="white", edgecolor="black", linewidth=1.1, zorder=5)

    vals2 = sub["Peak_memory_GB"].dropna().to_numpy()
    jitter = np.linspace(-0.08, 0.08, len(vals2))
    ax2.scatter(np.full(len(vals2), i) + jitter, vals2, s=24,
                color=COLORS[pipeline], alpha=0.55, edgecolor="none")
    ax2.scatter(i, np.mean(vals2), marker="D", s=62,
                facecolor="white", edgecolor="black", linewidth=1.1, zorder=5)

ax1.set_yscale("log")
ax1.set_ylabel("Downstream runtime (hours)")
ax1.set_title("Runtime")
ax1.grid(axis="y", alpha=0.18)

ax2.set_yscale("log")
ax2.set_ylabel("Peak memory (GB)")
ax2.set_title("Peak memory")
ax2.grid(axis="y", alpha=0.18)

for ax in (ax1, ax2):
    ax.set_xticks(xpos)
    ax.set_xticklabels([SHORT_LABEL[p] for p in PIPELINE_ORDER],
                       rotation=35, ha="right")

ax1.text(
    0.02, 0.02,
    "Colored points: individual EpiGEN conditions\nDiamonds: descriptive mean",
    transform=ax1.transAxes, fontsize=8, va="bottom"
)

fig.suptitle("Computational requirements across Benchmark B pipelines", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
save_both(fig, "FigureS8_runtime_and_memory")

# ============================================================
# SUPPLEMENT S9: rank percentile
# ============================================================

metric_architecture_figure(
    "Median_rank_percentile",
    "Median true-pair rank percentile (%)",
    "FigureS9_architecture_median_rank_percentile",
    (-2, 102),
)

print("=" * 72)
print("Benchmark B plots complete")
print("Input:", INPUT)
print("Output:", OUT)
print("Main: Figure_main_BenchmarkB_end_to_end.pdf/png")
print("Supplement: FigureS1-FigureS9")
print(
    f"Random 2k expected pair retention: "
    f"{RANDOM_PAIR_EXPECTED:.6f} ({RANDOM_PAIR_EXPECTED*100:.2f}%)"
)
