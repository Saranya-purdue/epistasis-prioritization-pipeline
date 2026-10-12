#!/usr/bin/env python3

from pathlib import Path
import re
import argparse

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

_cli = argparse.ArgumentParser(description="Random 2,000-SNP retention control for Benchmark B.")
_cli.add_argument("--base", required=True, help="Benchmark B root directory.")
_cli.add_argument("--out", default=None, help="Output CSV; defaults under <base>/summary_random.")
_cli.add_argument("--seed", type=int, default=42)
_cli_args = _cli.parse_args()
BASE = Path(_cli_args.base)
FULL_GENO = BASE / "data/genotype_benchmark_10000.csv"
OUT = Path(_cli_args.out) if _cli_args.out else BASE / "summary_random/random_2000_retention.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

PAIR_COUNTS = [
    10,
    50,
    100
]

SCENARIOS = [
    "joint_dominant",
    "joint_recessive",
    "multiplicative",
    "exponential",
]

N_SELECT = 2000

N_REPS = 1000

SEED = _cli_args.seed


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(x):

    x = str(x).strip()

    m = re.search(
        r"rs\d+",
        x,
        flags=re.I
    )

    if m:
        return m.group(0).lower()

    return x.lower()


# ============================================================
# LOAD 10,000 SNP UNIVERSE
#
# Header only -> very low memory
# ============================================================

print(
    "Reading full genotype header..."
)

header = pd.read_csv(
    FULL_GENO,
    nrows=0
)

# first column = sample ID
variant_columns = list(
    header.columns[1:]
)

variant_norm = np.array(
    [
        normalize(x)
        for x in variant_columns
    ],
    dtype=object
)


if len(variant_norm) != 10000:

    print(
        "WARNING: expected 10,000 SNPs, found:",
        len(variant_norm)
    )


if len(set(variant_norm)) != len(variant_norm):

    raise ValueError(
        "Duplicate normalized SNP IDs detected "
        "in 10k genotype universe."
    )


print(
    "Full SNP universe:",
    len(variant_norm)
)


# ============================================================
# THEORETICAL RANDOM EXPECTATION
# ============================================================

N_TOTAL = len(
    variant_norm
)

expected_snp_recall = (
    N_SELECT /
    N_TOTAL
)

expected_pair_recall = (
    (N_SELECT / N_TOTAL)
    *
    ((N_SELECT - 1) / (N_TOTAL - 1))
)


print()
print(
    "Theoretical random SNP retention:",
    expected_snp_recall
)

print(
    "Theoretical random complete-pair retention:",
    expected_pair_recall
)


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(
    SEED
)


# ============================================================
# LOOP OVER 12 CONDITIONS
# ============================================================

summary_rows = []


for n_pairs in PAIR_COUNTS:

    for scenario in SCENARIOS:

        print()
        print(
            "=" * 80
        )

        print(
            f"{n_pairs} pairs | {scenario}"
        )

        print(
            "=" * 80
        )


        # ----------------------------------------------------
        # Truth
        # ----------------------------------------------------

        truth_file = (
            BASE /
            "inputs" /
            "full_10000" /
            f"pairs_{n_pairs}" /
            scenario /
            "truth_all_pairs.tsv"
        )


        # fallback
        if not truth_file.exists():

            truth_file = (
                BASE /
                "inputs" /
                "vae_2000" /
                f"pairs_{n_pairs}" /
                scenario /
                "truth_all_pairs.tsv"
            )


        truth = pd.read_csv(
            truth_file,
            sep="\t"
        )


        lower = {
            str(c).lower(): c
            for c in truth.columns
        }


        if (
            "snp1" in lower
            and
            "snp2" in lower
        ):

            c1 = lower["snp1"]
            c2 = lower["snp2"]

        else:

            c1 = truth.columns[0]
            c2 = truth.columns[1]


        s1 = np.array(
            truth[c1]
            .map(normalize),
            dtype=object
        )

        s2 = np.array(
            truth[c2]
            .map(normalize),
            dtype=object
        )


        causal_snps = set(
            s1
        ) | set(
            s2
        )


        # Check truth exists in the 10k universe
        missing_truth = (
            causal_snps -
            set(variant_norm)
        )


        if missing_truth:

            raise ValueError(
                f"{len(missing_truth)} causal SNPs "
                f"not found in full 10k genotype."
            )


        # ----------------------------------------------------
        # Random selection repetitions
        # ----------------------------------------------------

        snp_recalls = []

        pair_recalls = []

        n_causal_retained_list = []

        n_pairs_retained_list = []


        for rep in range(
            N_REPS
        ):

            selected_idx = rng.choice(
                N_TOTAL,
                size=N_SELECT,
                replace=False
            )


            selected = set(
                variant_norm[
                    selected_idx
                ]
            )


            # causal SNP retention
            n_causal_retained = len(
                causal_snps &
                selected
            )


            snp_recall = (
                n_causal_retained /
                len(causal_snps)
            )


            # complete pair retention
            pair_available = np.array(
                [
                    (
                        a in selected
                        and
                        b in selected
                    )
                    for a, b
                    in zip(
                        s1,
                        s2
                    )
                ],
                dtype=bool
            )


            n_pairs_retained = int(
                pair_available.sum()
            )


            pair_recall = (
                n_pairs_retained /
                len(truth)
            )


            n_causal_retained_list.append(
                n_causal_retained
            )

            n_pairs_retained_list.append(
                n_pairs_retained
            )

            snp_recalls.append(
                snp_recall
            )

            pair_recalls.append(
                pair_recall
            )


        # ----------------------------------------------------
        # Convert
        # ----------------------------------------------------

        snp_recalls = np.array(
            snp_recalls
        )

        pair_recalls = np.array(
            pair_recalls
        )

        n_causal_retained_list = np.array(
            n_causal_retained_list
        )

        n_pairs_retained_list = np.array(
            n_pairs_retained_list
        )


        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        row = {

            "Pair_count":
                n_pairs,

            "Scenario":
                scenario,

            "N_reps":
                N_REPS,

            "N_total_snps":
                N_TOTAL,

            "N_selected":
                N_SELECT,

            "N_unique_causal_snps":
                len(causal_snps),

            "N_true_pairs":
                len(truth),

            "Expected_random_SNP_recall":
                expected_snp_recall,

            "Expected_random_pair_availability":
                expected_pair_recall,

            "Random_mean_causal_snps":
                float(
                    n_causal_retained_list.mean()
                ),

            "Random_mean_SNP_recall":
                float(
                    snp_recalls.mean()
                ),

            "Random_SD_SNP_recall":
                float(
                    snp_recalls.std(
                        ddof=1
                    )
                ),

            "Random_SNP_recall_2.5pct":
                float(
                    np.quantile(
                        snp_recalls,
                        0.025
                    )
                ),

            "Random_SNP_recall_97.5pct":
                float(
                    np.quantile(
                        snp_recalls,
                        0.975
                    )
                ),

            "Random_mean_pairs_available":
                float(
                    n_pairs_retained_list.mean()
                ),

            "Random_mean_pair_availability":
                float(
                    pair_recalls.mean()
                ),

            "Random_SD_pair_availability":
                float(
                    pair_recalls.std(
                        ddof=1
                    )
                ),

            "Random_pair_availability_2.5pct":
                float(
                    np.quantile(
                        pair_recalls,
                        0.025
                    )
                ),

            "Random_pair_availability_97.5pct":
                float(
                    np.quantile(
                        pair_recalls,
                        0.975
                    )
                ),
        }


        summary_rows.append(
            row
        )


        print(
            f"Random SNP recall: "
            f"{row['Random_mean_SNP_recall']:.3f} "
            f"[{row['Random_SNP_recall_2.5pct']:.3f}, "
            f"{row['Random_SNP_recall_97.5pct']:.3f}]"
        )

        print(
            f"Random pair availability: "
            f"{row['Random_mean_pair_availability']:.3f} "
            f"[{row['Random_pair_availability_2.5pct']:.3f}, "
            f"{row['Random_pair_availability_97.5pct']:.3f}]"
        )


# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame(
    summary_rows
)

result.to_csv(
    OUT,
    index=False
)


print()
print(
    "=" * 100
)

print(
    "RANDOM 2,000-SNP CONTROL"
)

print(
    "=" * 100
)


display_cols = [

    "Pair_count",
    "Scenario",

    "Random_mean_SNP_recall",
    "Random_SNP_recall_2.5pct",
    "Random_SNP_recall_97.5pct",

    "Random_mean_pair_availability",
    "Random_pair_availability_2.5pct",
    "Random_pair_availability_97.5pct",
]


print(
    result[
        display_cols
    ].to_string(
        index=False
    )
)


print()
print(
    "Saved:"
)

print(
    OUT
)