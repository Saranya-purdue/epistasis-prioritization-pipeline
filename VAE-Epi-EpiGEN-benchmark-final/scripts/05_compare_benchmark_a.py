#!/usr/bin/env python3

from pathlib import Path
import argparse
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description="Summarize EpiGEN Benchmark A outputs.")
parser.add_argument("--base", required=True, help="Benchmark A base directory.")
parser.add_argument("--outdir", default=None, help="Optional output directory.")
args = parser.parse_args()

BASE = Path(args.base).expanduser().resolve()
OUT = Path(args.outdir).expanduser().resolve() if args.outdir else BASE / "benchmark_comparison_10_50_100"
OUT.mkdir(parents=True, exist_ok=True)

PAIR_COUNTS = [10, 50, 100]

SCENARIOS = [
    "joint_dominant",
    "joint_recessive",
    "multiplicative",
    "exponential",
]

SCENARIO_LABELS = {
    "joint_dominant": "Joint dominant",
    "joint_recessive": "Joint recessive",
    "multiplicative": "Multiplicative",
    "exponential": "Exponential",
}

N_SNPS = 2000

N_PAIR_UNIVERSE = (
    N_SNPS * (N_SNPS - 1) // 2
)

print("Pair universe:", N_PAIR_UNIVERSE)


# ============================================================
# METHODS
#
# Tree-path methods are optional. If files do not exist they
# are simply skipped.
# ============================================================

METHOD_PATTERNS = {

    "PLINK": [
        "plink/plink_epistasis.epi.qt",
        "plink_epistasis.epi.qt",
    ],

    "MatrixEpistasis": [
        "matrixepistasis/matrixepistasis_all_pairs.tsv.gz",
        "matrixepistasis_all_pairs.tsv.gz",
    ],

    "RF-SHAP": [
        "rf_shap/rf_rfshap_pair_scores.tsv.gz",
        "rf_shap_pairs/rf_rfshap_pair_scores.tsv.gz",
        "rf_rfshap_pair_scores.tsv.gz",
    ],

    "RF-SHAP-Taylor": [
        "rf_shap_taylor/rf_shap_guided_taylor_pair_scores.tsv.gz",
        "rf_shap_guided_taylor_top500/"
        "rf_shap_guided_taylor_pair_scores.tsv.gz",
        "rf_shap_guided_taylor_pair_scores.tsv.gz",
    ],

    # Optional baseline pair rankings
    "RF-path": [
        "tree_ml_baselines/rf/"
        "rf_path_pair_scores_top100000.tsv.gz",
    ],

    "ExtraTrees-path": [
        "tree_ml_baselines/extratrees/"
        "extratrees_path_pair_scores_top100000.tsv.gz",
    ],
}


# ============================================================
# SNP normalization
# ============================================================

def clean_snp(x):

    x = str(x).strip()

    m = re.search(
        r"rs\d+",
        x,
        flags=re.IGNORECASE
    )

    if m:
        return m.group(0).lower()

    return x.lower()


def canonical_pair(a, b):

    a = clean_snp(a)
    b = clean_snp(b)

    return tuple(
        sorted([a, b])
    )


# ============================================================
# Resolve truth
# ============================================================

def find_truth_file(n_pairs, scenario):

    model_name = scenario.replace("_", "-")

    candidates = []

    if n_pairs == 10:

        candidates.extend([
            BASE / "causal_pairs.csv",
            BASE / "causal_pairs_10.csv",

            BASE /
            scenario /
            f"{model_name}_true_pairs.csv",

            BASE /
            "pairs_10" /
            f"causal_pairs_10.csv",

            BASE /
            "pairs_10" /
            scenario /
            f"{model_name}_true_pairs.csv",
        ])

    else:

        candidates.extend([
            BASE /
            f"pairs_{n_pairs}" /
            f"causal_pairs_{n_pairs}.csv",

            BASE /
            f"pairs_{n_pairs}" /
            scenario /
            f"{model_name}_true_pairs.csv",
        ])

    for f in candidates:

        if f.exists():
            return f

    raise FileNotFoundError(
        f"Truth file not found for "
        f"{n_pairs} pairs / {scenario}"
    )


def load_truth(n_pairs, scenario):

    f = find_truth_file(
        n_pairs,
        scenario
    )

    df = pd.read_csv(f)

    if not {
        "SNP1",
        "SNP2"
    }.issubset(df.columns):

        raise ValueError(
            f"{f} does not contain SNP1/SNP2."
        )

    truth_pairs = {
        canonical_pair(a, b)
        for a, b in zip(
            df["SNP1"],
            df["SNP2"]
        )
    }

    causal_snps = set()

    for a, b in truth_pairs:

        causal_snps.add(a)
        causal_snps.add(b)

    return (
        truth_pairs,
        causal_snps,
        f
    )


# ============================================================
# Locate result directory
# ============================================================

def result_root(n_pairs, scenario):

    candidates = []

    if n_pairs == 10:

        candidates.extend([
            BASE / "results" / scenario,
            BASE / "results" / "pairs_10" / scenario,
            BASE / "pairs_10" / scenario,
            BASE / scenario,
        ])

    else:

        candidates.extend([
            BASE / "results" / f"pairs_{n_pairs}" / scenario,
            BASE / f"pairs_{n_pairs}" / scenario,
        ])

    for p in candidates:

        if p.exists():
            return p

    return candidates[0]


# ============================================================
# Find method output
# ============================================================

def find_method_file(root, patterns):

    for rel in patterns:

        f = root / rel

        if f.exists():
            return f

    # Recursive fallback by final filename
    for rel in patterns:

        filename = Path(rel).name

        hits = list(
            root.rglob(filename)
        )

        if hits:
            return hits[0]

    return None


# ============================================================
# Read output file
# ============================================================

def read_table(path):

    path = Path(path)

    if path.name.endswith(
        ".epi.qt"
    ):

        return pd.read_csv(
            path,
            sep=r"\s+",
            engine="python"
        )

    return pd.read_csv(
        path,
        sep="\t",
        compression="infer"
    )


# ============================================================
# Detect SNP columns
# ============================================================

def find_snp_columns(df):

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    candidates = [
        ("snp1", "snp2"),
        ("snp_a", "snp_b"),
        ("snpa", "snpb"),
        ("variant1", "variant2"),
        ("variant_a", "variant_b"),
    ]

    for a, b in candidates:

        if a in lower and b in lower:

            return (
                lower[a],
                lower[b]
            )

    raise ValueError(
        "Could not identify pair columns.\n"
        f"Columns = {df.columns.tolist()}"
    )


# ============================================================
# Find p-value column
# ============================================================

def find_p_column(df):

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in [
        "pvalue",
        "p_value",
        "p",
        "pval",
    ]:

        if name in lower:
            return lower[name]

    return None


# ============================================================
# Detect ranking score
# ============================================================

def choose_ranking(df, method):

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    # --------------------------------------------------------
    # Statistical methods: smaller p = better
    # --------------------------------------------------------

    if method in {
        "PLINK",
        "MatrixEpistasis"
    }:

        pcol = find_p_column(df)

        if pcol is None:

            raise ValueError(
                f"No p-value found for {method}"
            )

        return pcol, True


    # --------------------------------------------------------
    # Explicit rank column
    # --------------------------------------------------------

    for name in [
        "rank",
        "pair_rank"
    ]:

        if name in lower:

            return lower[name], True


    # --------------------------------------------------------
    # Higher interaction score = better
    # --------------------------------------------------------

    preferred = [

        "mean_abs_shap_interaction",
        "mean_abs_interaction",

        "shap_interaction_strength",
        "shap_interaction",

        "interaction_strength",

        "mean_abs_taylor",
        "taylor_score",
        "taylor_interaction",

        "pair_score",
        "interaction_score",

        "cooccurrence_score",
        "cooccurrence",

        "importance",
        "score",
    ]

    for name in preferred:

        if name in lower:

            return lower[name], False


    # --------------------------------------------------------
    # Generic numeric score fallback
    # --------------------------------------------------------

    exclude = {
        "snp1",
        "snp2",
        "snpa",
        "snpb",
        "chr1",
        "chr2",
        "is_true",
        "is_true_pair",
    }

    for c in df.columns:

        cname = str(c).lower()

        if cname in exclude:
            continue

        if any(
            key in cname
            for key in [
                "score",
                "strength",
                "importance",
                "interaction",
                "cooccur",
            ]
        ):

            numeric = pd.to_numeric(
                df[c],
                errors="coerce"
            )

            if numeric.notna().sum() > 0:

                return c, False


    warnings.warn(
        f"No ranking score detected for {method}. "
        "Assuming file is already sorted."
    )

    return None, False


# ============================================================
# Create ranked canonical pair list
# ============================================================

def rank_method_file(
    path,
    method
):

    df = read_table(
        path
    )

    s1_col, s2_col = (
        find_snp_columns(
            df
        )
    )

    rank_col, ascending = (
        choose_ranking(
            df,
            method
        )
    )

    tmp = pd.DataFrame({

        "snp1": (
            df[s1_col]
            .map(clean_snp)
        ),

        "snp2": (
            df[s2_col]
            .map(clean_snp)
        ),
    })

    tmp["pair"] = [
        tuple(
            sorted([a, b])
        )
        for a, b in zip(
            tmp["snp1"],
            tmp["snp2"]
        )
    ]

    # Remove self-pairs
    tmp = tmp[
        tmp["snp1"]
        !=
        tmp["snp2"]
    ].copy()


    if rank_col is not None:

        tmp["ranking_value"] = (
            pd.to_numeric(
                df.loc[
                    tmp.index,
                    rank_col
                ],
                errors="coerce"
            )
        )

        tmp = tmp.dropna(
            subset=["ranking_value"]
        )

        tmp = tmp.sort_values(
            "ranking_value",
            ascending=ascending,
            kind="mergesort"
        )


    # Keep best occurrence if duplicated
    tmp = tmp.drop_duplicates(
        "pair",
        keep="first"
    ).reset_index(
        drop=True
    )

    ranked_pairs = (
        tmp["pair"]
        .tolist()
    )

    scored_snps = set(
        tmp["snp1"]
    ) | set(
        tmp["snp2"]
    )


    # --------------------------------------------------------
    # Save p-values where available for significance analysis
    # --------------------------------------------------------

    pcol = find_p_column(
        df
    )

    pair_to_p = None

    if pcol is not None:

        ptmp = pd.DataFrame({

            "pair": [
                canonical_pair(a, b)
                for a, b in zip(
                    df[s1_col],
                    df[s2_col]
                )
            ],

            "p": pd.to_numeric(
                df[pcol],
                errors="coerce"
            ),
        })

        ptmp = (
            ptmp
            .dropna(subset=["p"])
            .sort_values("p")
            .drop_duplicates(
                "pair",
                keep="first"
            )
        )

        pair_to_p = dict(
            zip(
                ptmp["pair"],
                ptmp["p"]
            )
        )

    return (
        ranked_pairs,
        scored_snps,
        pair_to_p,
        rank_col
    )


# ============================================================
# Ranking metrics
# ============================================================

def average_precision_prefix(
    ranked_pairs,
    truth_pairs
):

    """
    Conservative end-to-end AP.

    True pairs not present in the reported/scored list
    contribute zero to AP.
    """

    T = len(
        truth_pairs
    )

    hits = np.fromiter(
        (
            pair in truth_pairs
            for pair in ranked_pairs
        ),
        dtype=bool,
        count=len(ranked_pairs)
    )

    if not hits.any():
        return 0.0

    cumulative_tp = np.cumsum(
        hits
    )

    hit_positions = (
        np.flatnonzero(
            hits
        )
        + 1
    )

    precisions = (
        cumulative_tp[
            hit_positions - 1
        ]
        /
        hit_positions
    )

    return float(
        precisions.sum()
        /
        T
    )


def evaluate_ranking(
    ranked_pairs,
    scored_snps,
    truth_pairs,
    causal_snps
):

    T = len(
        truth_pairs
    )

    # --------------------------------------------------------
    # Screening metrics
    # --------------------------------------------------------

    causal_snp_found = (
        causal_snps
        &
        scored_snps
    )

    snp_coverage = (
        len(causal_snp_found)
        /
        len(causal_snps)
    )


    available_truth = {

        pair
        for pair in truth_pairs

        if (
            pair[0] in scored_snps
            and
            pair[1] in scored_snps
        )
    }

    n_available = len(
        available_truth
    )

    pair_availability = (
        n_available
        /
        T
    )


    # --------------------------------------------------------
    # AP and normalized AP
    # --------------------------------------------------------

    ap = average_precision_prefix(
        ranked_pairs,
        truth_pairs
    )

    prevalence = (
        T /
        N_PAIR_UNIVERSE
    )

    normalized_ap = (
        (ap - prevalence)
        /
        (1.0 - prevalence)
    )


    # --------------------------------------------------------
    # True pair ranks
    # Missing pair -> rank = universe + 1
    # --------------------------------------------------------

    rank_lookup = {
        pair: i + 1
        for i, pair
        in enumerate(
            ranked_pairs
        )
    }

    true_ranks = np.array([
        rank_lookup.get(
            pair,
            N_PAIR_UNIVERSE + 1
        )
        for pair
        in truth_pairs
    ])

    median_rank = float(
        np.median(
            true_ranks
        )
    )

    mean_rank = float(
        np.mean(
            true_ranks
        )
    )

    median_rank_percentile = (
        100.0
        *
        median_rank
        /
        N_PAIR_UNIVERSE
    )


    # --------------------------------------------------------
    # Top-K metrics
    # --------------------------------------------------------

    output = {

        "N_true_pairs":
            T,

        "N_causal_snps":
            len(causal_snps),

        "N_ranked_pairs":
            len(ranked_pairs),

        "N_scored_snps":
            len(scored_snps),

        "Causal_SNP_coverage":
            snp_coverage,

        "N_true_pairs_available":
            n_available,

        "Pair_availability":
            pair_availability,

        "AP":
            ap,

        "Random_AP":
            prevalence,

        "Normalized_AP":
            normalized_ap,

        "Median_true_pair_rank":
            median_rank,

        "Mean_true_pair_rank":
            mean_rank,

        "Median_rank_percentile":
            median_rank_percentile,
    }


    for multiplier in [
        1,
        5,
        10
    ]:

        K = (
            T * multiplier
        )

        top = ranked_pairs[
            :min(
                K,
                len(ranked_pairs)
            )
        ]

        tp = sum(
            pair in truth_pairs
            for pair in top
        )

        recall = (
            tp / T
        )

        # Requested K remains denominator.
        # This penalizes a method if it reports < K pairs.
        precision = (
            tp / K
        )

        if (
            precision + recall
        ) > 0:

            f1 = (
                2
                *
                precision
                *
                recall
                /
                (
                    precision
                    +
                    recall
                )
            )

        else:

            f1 = 0.0


        if n_available > 0:

            conditional_recall = (
                tp /
                n_available
            )

        else:

            conditional_recall = (
                np.nan
            )


        label = (
            "T"
            if multiplier == 1
            else f"{multiplier}T"
        )

        output[
            f"K_{label}"
        ] = K

        output[
            f"TP_at_{label}"
        ] = tp

        output[
            f"Recall_at_{label}"
        ] = recall

        output[
            f"Precision_at_{label}"
        ] = precision

        output[
            f"F1_at_{label}"
        ] = f1

        output[
            f"Conditional_Recall_at_{label}"
        ] = conditional_recall


    return output


# ============================================================
# Bonferroni metrics for PLINK / MatrixEpistasis
# ============================================================

def evaluate_significance(
    pair_to_p,
    truth_pairs
):

    threshold = (
        0.05
        /
        N_PAIR_UNIVERSE
    )

    sig_pairs = {
        pair
        for pair, p
        in pair_to_p.items()
        if p <= threshold
    }

    tp = len(
        sig_pairs
        &
        truth_pairs
    )

    fp = len(
        sig_pairs
        -
        truth_pairs
    )

    fn = (
        len(truth_pairs)
        -
        tp
    )

    sensitivity = (
        tp
        /
        len(truth_pairs)
    )

    if len(sig_pairs) > 0:

        precision = (
            tp
            /
            len(sig_pairs)
        )

        fdr = (
            fp
            /
            len(sig_pairs)
        )

    else:

        precision = np.nan
        fdr = np.nan

    return {

        "Bonferroni_threshold":
            threshold,

        "N_significant":
            len(sig_pairs),

        "TP_significant":
            tp,

        "FP_significant":
            fp,

        "FN_significant":
            fn,

        "Sensitivity":
            sensitivity,

        "Precision":
            precision,

        "Empirical_FDR":
            fdr,
    }


# ============================================================
# RUN ALL CONDITIONS
# ============================================================

rows = []
sig_rows = []
manifest = []


for n_pairs in PAIR_COUNTS:

    for scenario in SCENARIOS:

        truth_pairs, causal_snps, truth_file = (
            load_truth(
                n_pairs,
                scenario
            )
        )

        if len(
            truth_pairs
        ) != n_pairs:

            warnings.warn(
                f"{n_pairs}/{scenario}: "
                f"truth has {len(truth_pairs)} pairs."
            )

        root = result_root(
            n_pairs,
            scenario
        )

        print()
        print(
            "======================================================"
        )
        print(
            f"{n_pairs} pairs | {scenario}"
        )
        print(
            "Result root:",
            root
        )
        print(
            "Truth:",
            truth_file
        )
        print(
            "======================================================"
        )


        for method, patterns in (
            METHOD_PATTERNS.items()
        ):

            f = find_method_file(
                root,
                patterns
            )

            if f is None:

                print(
                    f"  MISSING: {method}"
                )

                continue


            print(
                f"  Reading {method}: {f}"
            )

            try:

                ranked_pairs, scored_snps, pair_to_p, rank_col = (
                    rank_method_file(
                        f,
                        method
                    )
                )

            except Exception as e:

                warnings.warn(
                    f"Failed {method} "
                    f"{n_pairs}/{scenario}: {e}"
                )

                continue


            metrics = evaluate_ranking(

                ranked_pairs=
                    ranked_pairs,

                scored_snps=
                    scored_snps,

                truth_pairs=
                    truth_pairs,

                causal_snps=
                    causal_snps,
            )


            metrics.update({

                "Pair_count":
                    n_pairs,

                "Scenario":
                    scenario,

                "Architecture":
                    SCENARIO_LABELS[
                        scenario
                    ],

                "Method":
                    method,
            })

            rows.append(
                metrics
            )


            manifest.append({

                "Pair_count":
                    n_pairs,

                "Scenario":
                    scenario,

                "Method":
                    method,

                "File":
                    str(f),

                "Ranking_column":
                    rank_col,

                "N_rows_ranked":
                    len(ranked_pairs),
            })


            if (
                pair_to_p is not None
                and
                method in {
                    "PLINK",
                    "MatrixEpistasis"
                }
            ):

                sig = (
                    evaluate_significance(
                        pair_to_p,
                        truth_pairs
                    )
                )

                sig.update({

                    "Pair_count":
                        n_pairs,

                    "Scenario":
                        scenario,

                    "Architecture":
                        SCENARIO_LABELS[
                            scenario
                        ],

                    "Method":
                        method,
                })

                sig_rows.append(
                    sig
                )


# ============================================================
# SAVE MASTER TABLE
# ============================================================

metrics_df = pd.DataFrame(
    rows
)

metrics_df = metrics_df[
    [
        "Pair_count",
        "Scenario",
        "Architecture",
        "Method",

        "N_true_pairs",
        "N_causal_snps",

        "N_ranked_pairs",
        "N_scored_snps",

        "Causal_SNP_coverage",
        "N_true_pairs_available",
        "Pair_availability",

        "TP_at_T",
        "Recall_at_T",

        "TP_at_5T",
        "Recall_at_5T",
        "Precision_at_5T",

        "TP_at_10T",
        "Recall_at_10T",
        "Precision_at_10T",

        "Conditional_Recall_at_T",
        "Conditional_Recall_at_5T",
        "Conditional_Recall_at_10T",

        "AP",
        "Random_AP",
        "Normalized_AP",

        "Median_true_pair_rank",
        "Mean_true_pair_rank",
        "Median_rank_percentile",
    ]
]


metrics_df.to_csv(
    OUT /
    "epigen_benchmark_all_metrics.csv",
    index=False
)


pd.DataFrame(
    manifest
).to_csv(
    OUT /
    "method_file_manifest.csv",
    index=False
)


if sig_rows:

    pd.DataFrame(
        sig_rows
    ).to_csv(
        OUT /
        "plink_matrix_significance_metrics.csv",
        index=False
    )


# ============================================================
# PRIMARY MANUSCRIPT TABLE
# ============================================================

primary_cols = [

    "Pair_count",
    "Architecture",
    "Method",

    "Causal_SNP_coverage",
    "Pair_availability",

    "Recall_at_T",
    "Recall_at_5T",
    "Recall_at_10T",

    "Normalized_AP",

    "Median_rank_percentile",
]


primary = metrics_df[
    primary_cols
].copy()


for c in [
    "Causal_SNP_coverage",
    "Pair_availability",
    "Recall_at_T",
    "Recall_at_5T",
    "Recall_at_10T",
    "Normalized_AP",
]:

    primary[c] = (
        primary[c]
        .round(4)
    )


primary[
    "Median_rank_percentile"
] = (
    primary[
        "Median_rank_percentile"
    ]
    .round(3)
)


primary.to_csv(
    OUT /
    "Table_primary_method_comparison.csv",
    index=False
)


# ============================================================
# PIVOT TABLE: Recall@T
# ============================================================

pivot = metrics_df.pivot_table(

    index=[
        "Pair_count",
        "Architecture"
    ],

    columns="Method",

    values="Recall_at_T"
)

pivot.to_csv(
    OUT /
    "Table_recall_at_T_pivot.csv"
)


# ============================================================
# DESCRIPTIVE MEAN ACROSS ARCHITECTURES
#
# Important:
# architectures are NOT statistical replicates.
# ============================================================

summary = (
    metrics_df
    .groupby(
        [
            "Pair_count",
            "Method"
        ],
        as_index=False
    )
    .agg(

        Mean_Recall_T=(
            "Recall_at_T",
            "mean"
        ),

        Mean_Recall_5T=(
            "Recall_at_5T",
            "mean"
        ),

        Mean_Recall_10T=(
            "Recall_at_10T",
            "mean"
        ),

        Mean_Normalized_AP=(
            "Normalized_AP",
            "mean"
        ),

        Mean_Pair_Availability=(
            "Pair_availability",
            "mean"
        ),
    )
)


summary.to_csv(
    OUT /
    "Table_descriptive_mean_across_architectures.csv",
    index=False
)


# ============================================================
# PLOTTING FUNCTION
# ============================================================

def architecture_line_plot(
    df,
    metric,
    ylabel,
    filename,
    ylim=None
):

    architectures = [
        "Joint dominant",
        "Joint recessive",
        "Multiplicative",
        "Exponential",
    ]

    methods = list(
        df["Method"]
        .drop_duplicates()
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for ax, arch in zip(
        axes,
        architectures
    ):

        sub = df[
            df["Architecture"]
            ==
            arch
        ]

        for method in methods:

            m = (
                sub[
                    sub["Method"]
                    ==
                    method
                ]
                .sort_values(
                    "Pair_count"
                )
            )

            if len(m) == 0:
                continue

            ax.plot(
                m["Pair_count"],
                m[metric],
                marker="o",
                linewidth=2,
                label=method
            )

        ax.set_title(
            arch
        )

        ax.set_xticks(
            [10, 50, 100]
        )

        ax.set_xlabel(
            "Number of true epistatic pairs"
        )

        ax.grid(
            alpha=0.25
        )

        if ylim is not None:
            ax.set_ylim(
                *ylim
            )


    axes[0].set_ylabel(
        ylabel
    )

    axes[2].set_ylabel(
        ylabel
    )


    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.93
        ]
    )

    fig.savefig(
        OUT / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# FIGURE 1 — PRIMARY: Recall@T
# ============================================================

architecture_line_plot(

    metrics_df,

    metric="Recall_at_T",

    ylabel="Recall at T",

    filename=
        "Figure1_recall_at_T.png",

    ylim=(0, 1)
)


# ============================================================
# FIGURE 2 — Recall@5T
# ============================================================

architecture_line_plot(

    metrics_df,

    metric="Recall_at_5T",

    ylabel="Recall at 5T",

    filename=
        "Figure2_recall_at_5T.png",

    ylim=(0, 1)
)


# ============================================================
# FIGURE 3 — Normalized AP
# ============================================================

architecture_line_plot(

    metrics_df,

    metric="Normalized_AP",

    ylabel="Normalized average precision",

    filename=
        "Figure3_normalized_AP.png",

    ylim=None
)


# ============================================================
# FIGURE 4 — Pair availability
# ============================================================

architecture_line_plot(

    metrics_df,

    metric="Pair_availability",

    ylabel="True-pair availability",

    filename=
        "Figure4_pair_availability.png",

    ylim=(0, 1)
)


# ============================================================
# FIGURE 5 — Median true-pair rank percentile
# Lower is better
# ============================================================

architecture_line_plot(

    metrics_df,

    metric="Median_rank_percentile",

    ylabel="Median true-pair rank percentile (%)",

    filename=
        "Figure5_median_rank_percentile.png",

    ylim=None
)


print()
print(
    "======================================================"
)
print(
    "Benchmark comparison complete."
)
print(
    "Output:"
)
print(
    OUT
)
print(
    "======================================================"
)

print()
print(
    primary.to_string(
        index=False
    )
)