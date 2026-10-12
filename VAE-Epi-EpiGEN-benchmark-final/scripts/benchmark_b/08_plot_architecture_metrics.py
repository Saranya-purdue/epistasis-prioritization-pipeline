#!/usr/bin/env python3
from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

parser = argparse.ArgumentParser(description="Plot architecture-stratified Benchmark B metrics.")
parser.add_argument("--base", required=True, help="Benchmark B base directory.")
parser.add_argument("--input", default=None, help="Optional benchmarkB_master_metrics.csv.")
parser.add_argument("--outdir", default=None, help="Optional figure output directory.")
args = parser.parse_args()

BENCHB = Path(args.base).expanduser().resolve()
SUMMARY_DIR = BENCHB / "benchmark_B_summary"
INPUT = Path(args.input).expanduser().resolve() if args.input else SUMMARY_DIR / "benchmarkB_master_metrics.csv"
OUT = Path(args.outdir).expanduser().resolve() if args.outdir else SUMMARY_DIR / "figures_architecture"
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
def save_both(fig, stem, show=True):
    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        OUT / f"{stem}.png",
        dpi=600,
        bbox_inches="tight"
    )

    if show:
        plt.show()

    plt.close(fig)

def add_panel_label(ax, label):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes,
            fontsize=14, va="top", ha="left")

def metric_architecture_figure(metric, ylabel, stem, ylim=None, pipelines=None):
    if pipelines is None:
        pipelines = PIPELINE_ORDER

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
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

fig = plt.figure(figsize=(8,6))
gs = fig.add_gridspec(
    2, 2, left=0.08, right=0.98, top=0.96, bottom=0.09,
    wspace=0.30, hspace=0.36
)

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
