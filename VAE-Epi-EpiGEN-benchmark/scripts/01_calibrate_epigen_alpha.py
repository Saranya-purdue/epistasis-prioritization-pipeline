#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd


# ============================================================
# Convert genotype coding to minor-allele dosage
# ============================================================

def minor_allele_dosage(g):

    g = pd.to_numeric(g, errors="coerce")

    if g.isna().any():
        raise ValueError("Missing genotype detected.")

    af = g.mean() / 2.0

    if af > 0.5:
        g = 2 - g

    return g.to_numpy(dtype=float)


# ============================================================
# Calculate EpiGEN genetic component
# ============================================================

def calculate_risk(
    G,
    pairs,
    scenario,
    alpha
):

    # Number of individuals, NOT number of SNPs
    n = len(next(iter(G.values())))

    # One risk value per individual
    risk = np.ones(
        n,
        dtype=float
    )

    for _, row in pairs.iterrows():

        g1 = G[row["SNP1"]]
        g2 = G[row["SNP2"]]

        # ----------------------------------------------------
        # Joint dominant
        # ----------------------------------------------------
        if scenario == "joint-dominant":

            active = (
                (g1 > 0) &
                (g2 > 0)
            )

            contribution = np.where(
                active,
                alpha,
                1.0
            )

        # ----------------------------------------------------
        # Joint recessive
        # ----------------------------------------------------
        elif scenario == "joint-recessive":

            active = (
                (g1 == 2) &
                (g2 == 2)
            )

            contribution = np.where(
                active,
                alpha,
                1.0
            )

        # ----------------------------------------------------
        # EpiGEN multiplicative
        # alpha^(G1 + G2)
        # ----------------------------------------------------
        elif scenario == "multiplicative":

            contribution = (
                alpha ** (g1 + g2)
            )

        # ----------------------------------------------------
        # EpiGEN exponential
        # alpha^(G1 * G2)
        # ----------------------------------------------------
        elif scenario == "exponential":

            contribution = (
                alpha ** (g1 * g2)
            )

        else:
            raise ValueError(
                f"Unknown scenario: {scenario}"
            )

        risk *= contribution

    return risk
# ============================================================
# Expected variance fraction
# ============================================================

def variance_fraction(
    risk,
    noise_sd
):

    vg = np.var(
        risk,
        ddof=0
    )

    ve = noise_sd ** 2

    fraction = vg / (vg + ve)

    return fraction, vg


# ============================================================
# Binary search alpha
# ============================================================

def find_alpha(
    G,
    pairs,
    scenario,
    target,
    noise_sd,
    lower=1.000001,
    upper=50.0,
    tolerance=0.0001,
    max_iter=100
):

    # Make sure upper bound can reach target
    risk = calculate_risk(
        G,
        pairs,
        scenario,
        upper
    )

    frac, _ = variance_fraction(
        risk,
        noise_sd
    )

    if frac < target:

        raise ValueError(
            f"Even alpha={upper} only gives "
            f"variance fraction={frac:.4f}. "
            "Increase upper bound."
        )

    for _ in range(max_iter):

        mid = (
            lower + upper
        ) / 2

        risk = calculate_risk(
            G,
            pairs,
            scenario,
            mid
        )

        frac, vg = variance_fraction(
            risk,
            noise_sd
        )

        if abs(frac - target) < tolerance:

            return (
                mid,
                frac,
                vg
            )

        if frac < target:
            lower = mid

        else:
            upper = mid

    return (
        mid,
        frac,
        vg
    )


# ============================================================
# Pair diagnostics
# ============================================================

def pair_diagnostics(
    G,
    pairs
):

    rows = []

    for _, row in pairs.iterrows():

        snp1 = row["SNP1"]
        snp2 = row["SNP2"]

        g1 = G[snp1]
        g2 = G[snp2]

        maf1 = np.mean(g1) / 2
        maf2 = np.mean(g2) / 2

        r = np.corrcoef(
            g1,
            g2
        )[0, 1]

        r2 = r ** 2

        dominant_frequency = np.mean(
            (g1 > 0) &
            (g2 > 0)
        )

        recessive_frequency = np.mean(
            (g1 == 2) &
            (g2 == 2)
        )

        rows.append({
            "Pair_ID":
                row.get("Pair_ID", ""),
            "SNP1": snp1,
            "SNP2": snp2,
            "MAF1": maf1,
            "MAF2": maf2,
            "Genotype_r2": r2,
            "Joint_dominant_frequency":
                dominant_frequency,
            "Joint_recessive_frequency":
                recessive_frequency
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--genotype",
        required=True
    )

    parser.add_argument(
        "--pairs",
        required=True
    )

    parser.add_argument(
        "--target",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--noise_sd",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--out",
        default="epigen_calibration"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        args.genotype
    )

    pairs = pd.read_csv(
        args.pairs
    )

    causal_snps = list(
        dict.fromkeys(
            pairs["SNP1"].tolist()
            +
            pairs["SNP2"].tolist()
        )
    )

    print(
        f"Individuals: {len(df)}"
    )

    print(
        f"Causal SNPs: {len(causal_snps)}"
    )

    print(
        f"True pairs: {len(pairs)}"
    )

    # --------------------------------------------------------
    # Extract causal genotypes
    # --------------------------------------------------------

    G = {}

    for snp in causal_snps:

        if snp not in df.columns:

            raise ValueError(
                f"{snp} not found "
                "in genotype file."
            )

        G[snp] = (
            minor_allele_dosage(
                df[snp]
            )
        )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    diagnostics = pair_diagnostics(
        G,
        pairs
    )

    diagnostic_file = (
        args.out
        + "_pair_diagnostics.csv"
    )

    diagnostics.to_csv(
        diagnostic_file,
        index=False
    )

    print("\nPair diagnostics:")
    print(
        diagnostics.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Calibrate each scenario
    # --------------------------------------------------------

    scenarios = [
        "joint-dominant",
        "joint-recessive",
        "multiplicative",
        "exponential"
    ]

    results = []

    print(
        "\n=========================================="
    )

    print(
        f"Target epistatic variance fraction: "
        f"{args.target}"
    )

    print(
        "=========================================="
    )

    for scenario in scenarios:

        alpha, frac, vg = find_alpha(
            G=G,
            pairs=pairs,
            scenario=scenario,
            target=args.target,
            noise_sd=args.noise_sd
        )

        print(
            f"\n{scenario}"
        )

        print(
            f"  calibrated alpha = "
            f"{alpha:.8f}"
        )

        print(
            f"  genetic variance = "
            f"{vg:.6f}"
        )

        print(
            f"  expected variance fraction = "
            f"{frac:.6f}"
        )

        results.append({
            "Scenario": scenario,
            "Target_fraction":
                args.target,
            "Noise_SD":
                args.noise_sd,
            "Calibrated_alpha":
                alpha,
            "Genetic_variance":
                vg,
            "Expected_fraction":
                frac
        })

    result_df = pd.DataFrame(
        results
    )

    result_file = (
        args.out
        + "_alphas.csv"
    )

    result_df.to_csv(
        result_file,
        index=False
    )

    print(
        f"\nSaved calibration: "
        f"{result_file}"
    )

    print(
        f"Saved pair diagnostics: "
        f"{diagnostic_file}"
    )


if __name__ == "__main__":
    main()