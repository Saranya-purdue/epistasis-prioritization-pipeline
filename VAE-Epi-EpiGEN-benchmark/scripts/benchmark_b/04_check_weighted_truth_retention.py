#!/usr/bin/env python3

from pathlib import Path
import re
import argparse

import pandas as pd


# ============================================================
# PATHS
# ============================================================

_cli = argparse.ArgumentParser(description="Check truth retention after phenotype-weighted VAE selection.")
_cli.add_argument("--base", required=True, help="Benchmark B root directory.")
_cli.add_argument("--out", default=None, help="Output CSV; defaults under <base>/summary_vae_weighted.")
_cli_args = _cli.parse_args()
BASE = Path(_cli_args.base)
OUT = Path(_cli_args.out) if _cli_args.out else BASE / "summary_vae_weighted/vae_truth_retention.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# BENCHMARK SETTINGS
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


# ============================================================
# VARIANT NORMALIZATION
# ============================================================

def normalize(x):
    """
    Normalize variant identifiers.

    Examples:
        rs6826415:1365623:A:G  -> rs6826415
        rs6826415              -> rs6826415
        1:3504541:C:T          -> 1:3504541:c:t

    For rsID-containing variants, use the rsID.
    For positional variants, retain the complete identifier.
    """

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
# READ PHENOTYPE-SPECIFIC WEIGHTED VAE SNP SET
# ============================================================

def read_selected_snps(selected_file):

    if not selected_file.exists():
        raise FileNotFoundError(
            f"Selected SNP file not found:\n{selected_file}"
        )

    df = pd.read_csv(
        selected_file
    )

    # Preferred column produced by B03e
    if "VARIANT_ID" in df.columns:

        variant_col = "VARIANT_ID"

    # Fallbacks in case file structure differs
    elif "Feature" in df.columns:

        variant_col = "Feature"

    elif "SNP" in df.columns:

        variant_col = "SNP"

    else:

        raise ValueError(
            f"Could not identify SNP column in:\n"
            f"{selected_file}\n"
            f"Columns: {list(df.columns)}"
        )

    selected_snps = {
        normalize(x)
        for x in df[variant_col]
        if pd.notna(x)
    }

    return selected_snps


# ============================================================
# READ TRUTH
# ============================================================

def read_truth(truth_file):

    if not truth_file.exists():
        raise FileNotFoundError(
            f"Truth file not found:\n{truth_file}"
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

        if truth.shape[1] < 2:

            raise ValueError(
                f"Truth file must contain at least "
                f"two SNP columns:\n{truth_file}"
            )

        c1 = truth.columns[0]
        c2 = truth.columns[1]

    truth = truth.copy()

    truth["SNP1_norm"] = (
        truth[c1]
        .map(normalize)
    )

    truth["SNP2_norm"] = (
        truth[c2]
        .map(normalize)
    )

    return truth


# ============================================================
# MAIN
# ============================================================

rows = []


for n_pairs in PAIR_COUNTS:

    for scenario in SCENARIOS:

        print()
        print(
            "=" * 80
        )

        print(
            f"Pair count: {n_pairs}"
        )

        print(
            f"Scenario:   {scenario}"
        )

        print(
            "=" * 80
        )


        # ====================================================
        # Phenotype-specific weighted VAE selection
        # ====================================================

        selected_file = (
            BASE /
            "vae_weighted" /
            f"pairs_{n_pairs}" /
            scenario /
            "FINAL_2000_Weighted_SHAP_LD_filtered_SNPs.csv"
        )

        vae_snps = read_selected_snps(
            selected_file
        )


        print(
            "Weighted VAE selected SNPs:",
            len(vae_snps)
        )


        if len(vae_snps) != 2000:

            print(
                "WARNING: expected 2000 selected SNPs, "
                f"found {len(vae_snps)} unique normalized SNPs."
            )


        # ====================================================
        # Ground-truth causal pairs
        #
        # Use original/full truth denominator, not only
        # variants surviving VAE selection.
        # ====================================================

        truth_file = (
            BASE /
            "inputs" /
            "full_10000" /
            f"pairs_{n_pairs}" /
            scenario /
            "truth_all_pairs.tsv"
        )


        # Fallback if truth was only copied to vae_2000
        if not truth_file.exists():

            truth_file = (
                BASE /
                "inputs" /
                "vae_2000" /
                f"pairs_{n_pairs}" /
                scenario /
                "truth_all_pairs.tsv"
            )


        truth = read_truth(
            truth_file
        )


        # ====================================================
        # Unique causal SNPs
        # ====================================================

        s1 = truth[
            "SNP1_norm"
        ]

        s2 = truth[
            "SNP2_norm"
        ]


        unique_causal = (
            set(s1)
            |
            set(s2)
        )


        retained_causal = (
            unique_causal
            &
            vae_snps
        )


        # ====================================================
        # Complete causal-pair availability
        #
        # Both SNPs of a true pair must survive selection.
        # ====================================================

        pair_available = (
            s1.isin(
                vae_snps
            )
            &
            s2.isin(
                vae_snps
            )
        )


        n_true_pairs = len(
            truth
        )

        n_unique_causal = len(
            unique_causal
        )

        n_retained_causal = len(
            retained_causal
        )

        n_pairs_available = int(
            pair_available.sum()
        )


        causal_snp_recall = (
            n_retained_causal
            /
            n_unique_causal
            if n_unique_causal > 0
            else float("nan")
        )


        pair_availability = (
            n_pairs_available
            /
            n_true_pairs
            if n_true_pairs > 0
            else float("nan")
        )


        # ====================================================
        # Store result
        # ====================================================

        rows.append({

            "Pair_count":
                n_pairs,

            "Scenario":
                scenario,

            "N_VAE_selected_snps":
                len(vae_snps),

            "N_true_pairs":
                n_true_pairs,

            "N_unique_causal_snps":
                n_unique_causal,

            "N_causal_snps_VAE_retained":
                n_retained_causal,

            "Causal_SNP_recall":
                causal_snp_recall,

            "N_true_pairs_VAE_available":
                n_pairs_available,

            "True_pair_availability":
                pair_availability,
        })


        # ====================================================
        # Print individual result
        # ====================================================

        print(
            f"Causal SNPs retained: "
            f"{n_retained_causal}/"
            f"{n_unique_causal} "
            f"({causal_snp_recall:.3f})"
        )

        print(
            f"True pairs available: "
            f"{n_pairs_available}/"
            f"{n_true_pairs} "
            f"({pair_availability:.3f})"
        )


# ============================================================
# COMBINE RESULTS
# ============================================================

result = pd.DataFrame(
    rows
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUT,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print(
    "=" * 100
)

print(
    "PHENOTYPE-WEIGHTED VAE CAUSAL RETENTION"
)

print(
    "=" * 100
)

print(
    result.to_string(
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