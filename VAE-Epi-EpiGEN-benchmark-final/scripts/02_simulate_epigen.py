#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ParametrizedModel = None

def configure_epigen(epigen_root=None):
    """Make the external EpiGEN package importable without hard-coded paths."""
    global ParametrizedModel
    import sys
    if epigen_root:
        root = str(Path(epigen_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        from utils.parametrized_model import ParametrizedModel as _PM
    except ImportError as exc:
        raise ImportError(
            "Could not import EpiGEN. Clone EpiGEN and pass --epigen-root PATH "
            "or add the EpiGEN repository to PYTHONPATH."
        ) from exc
    ParametrizedModel = _PM



# ============================================================
# Utility functions
# ============================================================

ID_COLUMNS = {
    "FID", "IID", "SUBJID", "ID", "sample", "sample_id",
    "SampleID", "subject_id", "PAT", "MAT", "SEX", "PHENOTYPE"
}


def identify_snp_columns(df):
    """
    Identify genotype columns.

    SNP columns must contain only 0/1/2, ignoring missing values.
    """

    snp_cols = []

    for col in df.columns:

        if col in ID_COLUMNS:
            continue

        vals = pd.to_numeric(df[col], errors="coerce")

        non_missing = vals.dropna()

        if len(non_missing) == 0:
            continue

        unique_vals = set(non_missing.unique())

        if unique_vals.issubset({0, 1, 2, 0.0, 1.0, 2.0}):
            snp_cols.append(col)

    return snp_cols


def convert_to_minor_allele_dosage(df, snp_cols):
    """
    EpiGEN interprets genotype as number of minor alleles:
        0, 1, 2

    If the coded allele frequency > 0.5,
    flip dosage:

        G_minor = 2 - G
    """

    G = df[snp_cols].copy()

    maf_info = []

    for snp in snp_cols:

        g = pd.to_numeric(G[snp], errors="coerce")

        if g.isna().any():
            raise ValueError(
                f"Missing genotype detected for {snp}. "
                "Please impute genotypes before simulation."
            )

        coded_af = g.mean() / 2.0

        flipped = False

        if coded_af > 0.5:
            g = 2 - g
            minor_af = 1 - coded_af
            flipped = True
        else:
            minor_af = coded_af

        G[snp] = g.astype(int)

        maf_info.append({
            "SNP": snp,
            "MAF": minor_af,
            "Flipped_to_minor_allele_count": flipped
        })

    maf_df = pd.DataFrame(maf_info)

    return G, maf_df


# ============================================================
# Select causal SNP pairs
# ============================================================

def select_causal_pairs(
    maf_df,
    n_pairs,
    maf_min,
    maf_max,
    seed
):

    eligible = maf_df.loc[
        (maf_df["MAF"] >= maf_min) &
        (maf_df["MAF"] <= maf_max),
        "SNP"
    ].tolist()

    required = 2 * n_pairs

    if len(eligible) < required:
        raise ValueError(
            f"Need {required} SNPs for {n_pairs} independent pairs, "
            f"but only {len(eligible)} SNPs satisfy "
            f"MAF {maf_min}-{maf_max}."
        )

    rng = np.random.default_rng(seed)

    chosen = rng.choice(
        eligible,
        size=required,
        replace=False
    )

    pairs = []

    for i in range(n_pairs):

        snp1 = chosen[2 * i]
        snp2 = chosen[2 * i + 1]

        pairs.append({
            "Pair_ID": f"PAIR_{i + 1}",
            "SNP1": snp1,
            "SNP2": snp2
        })

    return pd.DataFrame(pairs)


# ============================================================
# Generate EpiGEN XML
# ============================================================

def make_epigen_xml(
    pairs,
    causal_snps,
    interaction_type,
    alpha,
    stdev,
    outfile
):

    snp_position = {
        snp: i
        for i, snp in enumerate(causal_snps)
    }

    lines = []

    lines.append('<?xml version="1.0"?>')

    lines.append(
        f'<ParametrizedModel '
        f'size="{len(causal_snps)}" '
        f'phenotype="quantitative" '
        f'stdev="{stdev}">'
    )

    lines.append(
        '    <BaselineModel alpha="1.0"/>'
    )

    for _, row in pairs.iterrows():

        pos1 = snp_position[row["SNP1"]]
        pos2 = snp_position[row["SNP2"]]

        lines.append(
            f'    <InteractionModel '
            f'alpha="{alpha}" '
            f'type="{interaction_type}">'
        )

        lines.append(
            f'        <pos>{pos1}</pos>'
        )

        lines.append(
            f'        <pos>{pos2}</pos>'
        )

        lines.append(
            '    </InteractionModel>'
        )

    lines.append(
        '</ParametrizedModel>'
    )

    with open(outfile, "w") as f:
        f.write("\n".join(lines))


# ============================================================
# Calculate genetic component separately
# ============================================================

def calculate_genetic_component(model, genotype_vector):

    risk = model.baseline_alpha

    for func in model.marginal_models.values():
        risk *= func(genotype_vector)

    for func in model.interaction_models.values():
        risk *= func(genotype_vector)

    return risk


# ============================================================
# Simulate one phenotype
# ============================================================

def simulate_scenario(
    df,
    G,
    pairs,
    scenario,
    alpha,
    stdev,
    seed,
    output_dir
):

    os.makedirs(output_dir, exist_ok=True)

    causal_snps = list(
        dict.fromkeys(
            pairs["SNP1"].tolist()
            +
            pairs["SNP2"].tolist()
        )
    )

    xml_file = os.path.join(
        output_dir,
        f"{scenario}_model.xml"
    )

    make_epigen_xml(
        pairs=pairs,
        causal_snps=causal_snps,
        interaction_type=scenario,
        alpha=alpha,
        stdev=stdev,
        outfile=xml_file
    )

    # Load EpiGEN model
    model = ParametrizedModel(
        xml_file,
        seed
    )

    # Causal genotype matrix
    causal_G = G[causal_snps].to_numpy(
        dtype=int
    )

    n = causal_G.shape[0]

    genetic_component = np.zeros(n)

    phenotype = np.zeros(n)

    for i in range(n):

        genotype_vector = causal_G[i, :]

        genetic_component[i] = (
            calculate_genetic_component(
                model,
                genotype_vector
            )
        )

        phenotype[i] = model(
            genotype_vector
        )

    # --------------------------------------------
    # Output phenotype
    # --------------------------------------------

    out = pd.DataFrame()

    if "FID" in df.columns:
        out["FID"] = df["FID"]

    if "IID" in df.columns:
        out["IID"] = df["IID"]

    elif "SUBJID" in df.columns:
        out["IID"] = df["SUBJID"]

    elif "ID" in df.columns:
        out["IID"] = df["ID"]

    else:
        out["IID"] = np.arange(
            1,
            len(df) + 1
        )

    out["Phenotype"] = phenotype

    out["Epistatic_Component"] = (
        genetic_component
    )

    phenotype_file = os.path.join(
        output_dir,
        f"{scenario}_phenotype.csv"
    )

    out.to_csv(
        phenotype_file,
        index=False
    )

    # --------------------------------------------
    # Save truth
    # --------------------------------------------

    truth = pairs.copy()

    truth["Interaction_Type"] = scenario
    truth["Alpha"] = alpha

    truth.to_csv(
        os.path.join(
            output_dir,
            f"{scenario}_true_pairs.csv"
        ),
        index=False
    )

    # --------------------------------------------
    # Summary
    # --------------------------------------------

    summary = {
        "Scenario": scenario,
        "N": len(out),
        "Num_true_pairs": len(pairs),
        "Alpha": alpha,
        "Noise_SD": stdev,
        "Phenotype_mean": phenotype.mean(),
        "Phenotype_SD": phenotype.std(),
        "Epistatic_mean":
            genetic_component.mean(),
        "Epistatic_SD":
            genetic_component.std(),
        "Epistatic_variance":
            np.var(genetic_component),
        "Phenotype_variance":
            np.var(phenotype)
    }

    summary[
        "Approx_epistatic_variance_fraction"
    ] = (
        np.var(genetic_component)
        /
        np.var(phenotype)
        if np.var(phenotype) > 0
        else np.nan
    )

    print("\n====================================")
    print(f"Scenario: {scenario}")
    print("====================================")

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--genotype",
        default="genotype_FINAL_2000.csv"
    )

    parser.add_argument(
        "--scenario",
        choices=[
            "joint-dominant",
            "joint-recessive",
            "multiplicative",
            "exponential",
            "all"
        ],
        default="all"
    )

    parser.add_argument(
        "--pairs",
        default=None,
        help=(
            "Existing causal-pair CSV. "
            "If omitted, new causal pairs are selected."
        )
    )

    parser.add_argument(
        "--n_pairs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--maf_min",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--maf_max",
        type=float,
        default=0.50
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.20
    )

    parser.add_argument(
        "--stdev",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026
    )

    parser.add_argument(
        "--outdir",
        default="epigen_simulations"
    )

    parser.add_argument(
        "--epigen-root",
        default=None,
        help="Path to a local clone of the EpiGEN repository."
    )

    args = parser.parse_args()

    configure_epigen(args.epigen_root)

    # ========================================================
    # Load genotype
    # ========================================================

    print(
        f"Reading genotype: {args.genotype}"
    )

    df = pd.read_csv(
        args.genotype
    )

    print(
        f"Individuals: {df.shape[0]}"
    )

    # ========================================================
    # Identify SNP columns
    # ========================================================

    snp_cols = identify_snp_columns(df)

    print(
        f"Detected genotype SNPs: "
        f"{len(snp_cols)}"
    )

    if len(snp_cols) == 0:
        raise ValueError(
            "No 0/1/2 genotype columns detected."
        )

    # ========================================================
    # Convert to minor allele dosage
    # ========================================================

    G, maf_df = (
        convert_to_minor_allele_dosage(
            df,
            snp_cols
        )
    )

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    maf_df.to_csv(
        os.path.join(
            args.outdir,
            "vae2000_snp_maf.csv"
        ),
        index=False
    )

    # ========================================================
    # Causal SNP pairs
    # ========================================================

    if args.pairs is not None:

        pairs = pd.read_csv(
            args.pairs
        )

        print(
            f"Loaded {len(pairs)} "
            "existing causal pairs."
        )

    else:

        pairs = select_causal_pairs(
            maf_df=maf_df,
            n_pairs=args.n_pairs,
            maf_min=args.maf_min,
            maf_max=args.maf_max,
            seed=args.seed
        )

        pair_file = os.path.join(
            args.outdir,
            "causal_pairs.csv"
        )

        pairs.to_csv(
            pair_file,
            index=False
        )

        print(
            f"Selected {len(pairs)} "
            "causal SNP pairs."
        )

        print(
            f"Saved to: {pair_file}"
        )

    # ========================================================
    # Determine scenarios
    # ========================================================

    if args.scenario == "all":

        scenarios = [
            "joint-dominant",
            "joint-recessive",
            "multiplicative",
            "exponential"
        ]

    else:

        scenarios = [
            args.scenario
        ]

    # ========================================================
    # Simulate
    # ========================================================

    summaries = []

    for scenario in scenarios:

        scenario_dir = os.path.join(
            args.outdir,
            scenario.replace("-", "_")
        )

        summary = simulate_scenario(
            df=df,
            G=G,
            pairs=pairs,
            scenario=scenario,
            alpha=args.alpha,
            stdev=args.stdev,
            seed=args.seed,
            output_dir=scenario_dir
        )

        summaries.append(summary)

    # ========================================================
    # Summary
    # ========================================================

    pd.DataFrame(
        summaries
    ).to_csv(
        os.path.join(
            args.outdir,
            "simulation_summary.csv"
        ),
        index=False
    )

    print("\nAll simulations completed.")


if __name__ == "__main__":
    main()