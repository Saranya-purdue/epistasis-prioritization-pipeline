#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# EpiGEN import.
# Run this script from the EpiGEN repository or make sure EpiGEN is on PYTHONPATH.
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
# EpiGEN architectures
# ============================================================

SCENARIOS = {
    "joint_dominant": "joint-dominant",
    "joint_recessive": "joint-recessive",
    "multiplicative": "multiplicative",
    "exponential": "exponential",
}


# ============================================================
# Load genotype
# ============================================================

def load_genotype(genotype_file):

    print("======================================================")
    print("Loading genotype")
    print("======================================================")
    print(genotype_file)

    df = pd.read_csv(genotype_file)

    if df.shape[1] < 2:
        raise ValueError(
            "Genotype file must contain sample ID + SNP columns."
        )

    # First column = sample identifier
    sample_col = df.columns[0]

    sample_ids = (
        df.iloc[:, 0]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    geno = df.iloc[:, 1:].copy()

    print("Sample ID column:", sample_col)
    print("Individuals:", len(sample_ids))
    print("SNP columns:", geno.shape[1])

    # --------------------------------------------------------
    # Check genotype coding and convert to minor allele dosage
    # --------------------------------------------------------

    maf_rows = []

    for snp in geno.columns:

        g = pd.to_numeric(
            geno[snp],
            errors="coerce"
        )

        if g.isna().any():
            raise ValueError(
                f"Missing/non-numeric genotype detected: {snp}"
            )

        vals = set(
            np.unique(g.to_numpy())
        )

        if not vals.issubset({0, 1, 2, 0.0, 1.0, 2.0}):
            raise ValueError(
                f"{snp} contains values outside 0/1/2: {vals}"
            )

        g = g.to_numpy(dtype=np.int8)

        coded_af = g.mean() / 2.0

        flipped = False

        if coded_af > 0.5:

            g = (
                2 - g
            ).astype(np.int8)

            maf = 1.0 - coded_af
            flipped = True

        else:

            maf = coded_af

        geno[snp] = g

        maf_rows.append({
            "SNP": snp,
            "MAF": maf,
            "Flipped_to_minor_allele": flipped,
        })

    maf_df = pd.DataFrame(
        maf_rows
    )

    return sample_ids, geno, maf_df


# ============================================================
# Pair utilities
# ============================================================

def genotype_r2(g1, g2):

    r = np.corrcoef(
        g1,
        g2
    )[0, 1]

    if not np.isfinite(r):
        return np.nan

    return float(r ** 2)


def normalize_existing_pairs(pairs):

    required = {
        "SNP1",
        "SNP2"
    }

    missing = required - set(
        pairs.columns
    )

    if missing:
        raise ValueError(
            f"Existing pair file missing columns: {missing}"
        )

    out = pairs[
        ["SNP1", "SNP2"]
    ].copy()

    out["SNP1"] = (
        out["SNP1"]
        .astype(str)
        .str.strip()
    )

    out["SNP2"] = (
        out["SNP2"]
        .astype(str)
        .str.strip()
    )

    out.insert(
        0,
        "Pair_ID",
        [
            f"PAIR_{i+1}"
            for i in range(len(out))
        ]
    )

    return out


# ============================================================
# Extend 10 pairs -> 100 pairs
# ============================================================

def extend_pair_set(
    existing_pairs,
    geno,
    maf_df,
    target_pairs,
    maf_min,
    maf_max,
    ld_r2_max,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    pairs = existing_pairs.copy()

    used = set(
        pairs["SNP1"]
    ) | set(
        pairs["SNP2"]
    )

    eligible = (
        maf_df.loc[
            (maf_df["MAF"] >= maf_min)
            &
            (maf_df["MAF"] <= maf_max),
            "SNP"
        ]
        .tolist()
    )

    eligible = [
        snp
        for snp in eligible
        if snp not in used
    ]

    print()
    print(
        f"Eligible unused SNPs: {len(eligible)}"
    )

    needed = (
        target_pairs
        -
        len(pairs)
    )

    if needed <= 0:
        return pairs.iloc[
            :target_pairs
        ].copy()

    if len(eligible) < (
        needed * 2
    ):
        raise ValueError(
            "Not enough eligible SNPs to create "
            f"{target_pairs} non-overlapping pairs."
        )

    remaining = list(
        eligible
    )

    rng.shuffle(
        remaining
    )

    new_pairs = []

    while len(new_pairs) < needed:

        if len(remaining) < 2:
            raise RuntimeError(
                "Could not construct enough causal pairs "
                "under the LD criterion."
            )

        # Pick one SNP
        snp1 = remaining.pop()

        # Randomize candidate partners
        candidate_indices = np.arange(
            len(remaining)
        )

        rng.shuffle(
            candidate_indices
        )

        partner_index = None

        for idx in candidate_indices:

            snp2 = remaining[idx]

            r2 = genotype_r2(
                geno[snp1].to_numpy(),
                geno[snp2].to_numpy()
            )

            if (
                np.isfinite(r2)
                and
                r2 <= ld_r2_max
            ):

                partner_index = idx
                break

        if partner_index is None:

            # Put SNP back and reshuffle
            remaining.insert(
                0,
                snp1
            )

            rng.shuffle(
                remaining
            )

            continue

        snp2 = remaining.pop(
            partner_index
        )

        new_pairs.append({
            "SNP1": snp1,
            "SNP2": snp2,
        })

    new_df = pd.DataFrame(
        new_pairs
    )

    pairs = pd.concat(
        [
            pairs[
                ["SNP1", "SNP2"]
            ],
            new_df
        ],
        ignore_index=True
    )

    pairs.insert(
        0,
        "Pair_ID",
        [
            f"PAIR_{i+1}"
            for i in range(len(pairs))
        ]
    )

    return pairs


# ============================================================
# Pair diagnostics
# ============================================================

def pair_diagnostics(
    pairs,
    geno,
    maf_df
):

    maf_map = dict(
        zip(
            maf_df["SNP"],
            maf_df["MAF"]
        )
    )

    rows = []

    for _, row in pairs.iterrows():

        s1 = row["SNP1"]
        s2 = row["SNP2"]

        g1 = geno[
            s1
        ].to_numpy()

        g2 = geno[
            s2
        ].to_numpy()

        r2 = genotype_r2(
            g1,
            g2
        )

        dom_freq = np.mean(
            (g1 > 0)
            &
            (g2 > 0)
        )

        rec_freq = np.mean(
            (g1 == 2)
            &
            (g2 == 2)
        )

        rows.append({
            "Pair_ID":
                row["Pair_ID"],
            "SNP1":
                s1,
            "SNP2":
                s2,
            "MAF1":
                maf_map[s1],
            "MAF2":
                maf_map[s2],
            "Genotype_r2":
                r2,
            "Joint_dominant_frequency":
                dom_freq,
            "Joint_recessive_frequency":
                rec_freq,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Exact EpiGEN risk rules
# ============================================================

def calculate_epigen_risk(
    geno,
    pairs,
    scenario,
    alpha
):

    n = len(
        geno
    )

    risk = np.ones(
        n,
        dtype=np.float64
    )

    for _, row in pairs.iterrows():

        g1 = geno[
            row["SNP1"]
        ].to_numpy(
            dtype=np.float64
        )

        g2 = geno[
            row["SNP2"]
        ].to_numpy(
            dtype=np.float64
        )

        # ----------------------------------------------------
        # Joint dominant
        # ----------------------------------------------------

        if scenario == "joint-dominant":

            contribution = np.where(
                (g1 > 0)
                &
                (g2 > 0),
                alpha,
                1.0
            )

        # ----------------------------------------------------
        # Joint recessive
        # ----------------------------------------------------

        elif scenario == "joint-recessive":

            contribution = np.where(
                (g1 == 2)
                &
                (g2 == 2),
                alpha,
                1.0
            )

        # ----------------------------------------------------
        # EpiGEN multiplicative
        #
        # alpha^(G1 + G2)
        # ----------------------------------------------------

        elif scenario == "multiplicative":

            contribution = (
                alpha **
                (g1 + g2)
            )

        # ----------------------------------------------------
        # EpiGEN exponential
        #
        # alpha^(G1 * G2)
        # ----------------------------------------------------

        elif scenario == "exponential":

            contribution = (
                alpha **
                (g1 * g2)
            )

        else:

            raise ValueError(
                f"Unknown EpiGEN scenario: {scenario}"
            )

        # EpiGEN combines interaction terms multiplicatively
        risk *= contribution

    return risk


# ============================================================
# Calibrate alpha
# ============================================================

def expected_variance_fraction(
    risk,
    noise_sd
):

    vg = np.var(
        risk,
        ddof=0
    )

    ve = (
        noise_sd ** 2
    )

    fraction = (
        vg /
        (vg + ve)
    )

    return fraction, vg


def find_calibrated_alpha(
    geno,
    pairs,
    scenario,
    target_fraction,
    noise_sd,
    tolerance=1e-6,
    max_iterations=100
):

    lower = 1.00000001
    upper = 1.01

    # --------------------------------------------------------
    # Expand upper bound until target is reached
    # --------------------------------------------------------

    while True:

        risk = calculate_epigen_risk(
            geno,
            pairs,
            scenario,
            upper
        )

        frac, _ = (
            expected_variance_fraction(
                risk,
                noise_sd
            )
        )

        if frac >= target_fraction:
            break

        upper = (
            1.0
            +
            2.0 * (upper - 1.0)
        )

        if upper > 10:

            raise RuntimeError(
                f"Could not reach target variance for "
                f"{scenario}. Last alpha={upper}"
            )

    # --------------------------------------------------------
    # Binary search
    # --------------------------------------------------------

    for _ in range(
        max_iterations
    ):

        mid = (
            lower + upper
        ) / 2.0

        risk = calculate_epigen_risk(
            geno,
            pairs,
            scenario,
            mid
        )

        frac, vg = (
            expected_variance_fraction(
                risk,
                noise_sd
            )
        )

        if abs(
            frac - target_fraction
        ) <= tolerance:

            return (
                mid,
                frac,
                vg
            )

        if frac < target_fraction:

            lower = mid

        else:

            upper = mid

    return (
        mid,
        frac,
        vg
    )


# ============================================================
# Generate EpiGEN XML
# ============================================================

def write_epigen_xml(
    pairs,
    scenario,
    alpha,
    noise_sd,
    outfile
):

    causal_snps = list(
        dict.fromkeys(
            pairs["SNP1"].tolist()
            +
            pairs["SNP2"].tolist()
        )
    )

    snp_pos = {
        snp: i
        for i, snp
        in enumerate(
            causal_snps
        )
    }

    lines = []

    lines.append(
        '<?xml version="1.0"?>'
    )

    lines.append(
        f'<ParametrizedModel '
        f'size="{len(causal_snps)}" '
        f'phenotype="quantitative" '
        f'stdev="{noise_sd}">'
    )

    lines.append(
        '    <BaselineModel alpha="1.0"/>'
    )

    for _, row in pairs.iterrows():

        lines.append(
            f'    <InteractionModel '
            f'alpha="{alpha}" '
            f'type="{scenario}">'
        )

        lines.append(
            f'        <pos>'
            f'{snp_pos[row["SNP1"]]}'
            f'</pos>'
        )

        lines.append(
            f'        <pos>'
            f'{snp_pos[row["SNP2"]]}'
            f'</pos>'
        )

        lines.append(
            '    </InteractionModel>'
        )

    lines.append(
        '</ParametrizedModel>'
    )

    with open(
        outfile,
        "w"
    ) as f:

        f.write(
            "\n".join(lines)
        )

    return causal_snps


# ============================================================
# Validate our vectorized risk against EpiGEN
# ============================================================

def validate_against_epigen(
    geno,
    pairs,
    scenario,
    alpha,
    xml_file,
    seed,
    n_validate=25
):

    causal_snps = list(
        dict.fromkeys(
            pairs["SNP1"].tolist()
            +
            pairs["SNP2"].tolist()
        )
    )

    G = geno[
        causal_snps
    ].to_numpy(
        dtype=int
    )

    model = ParametrizedModel(
        str(xml_file),
        seed
    )

    vector_risk = calculate_epigen_risk(
        geno,
        pairs,
        scenario,
        alpha
    )

    n_validate = min(
        n_validate,
        len(G)
    )

    for i in range(
        n_validate
    ):

        x = G[i, :]

        model_risk = (
            model.baseline_alpha
        )

        for func in (
            model.marginal_models.values()
        ):

            model_risk *= func(
                x
            )

        for func in (
            model.interaction_models.values()
        ):

            model_risk *= func(
                x
            )

        if not np.isclose(
            model_risk,
            vector_risk[i],
            rtol=1e-10,
            atol=1e-10
        ):

            raise RuntimeError(
                f"EpiGEN validation failed at individual {i}: "
                f"{model_risk} != {vector_risk[i]}"
            )

    return True


# ============================================================
# Simulate one scenario
# ============================================================

def simulate_scenario(
    sample_ids,
    geno,
    pairs,
    scenario_folder,
    scenario,
    alpha,
    noise,
    noise_sd,
    seed
):

    scenario_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    xml_file = (
        scenario_folder
        /
        f"{scenario}_model.xml"
    )

    write_epigen_xml(
        pairs=pairs,
        scenario=scenario,
        alpha=alpha,
        noise_sd=noise_sd,
        outfile=xml_file
    )

    # Confirm vectorized implementation matches EpiGEN
    validate_against_epigen(
        geno=geno,
        pairs=pairs,
        scenario=scenario,
        alpha=alpha,
        xml_file=xml_file,
        seed=seed
    )

    # --------------------------------------------------------
    # Genetic component
    # --------------------------------------------------------

    genetic = calculate_epigen_risk(
        geno,
        pairs,
        scenario,
        alpha
    )

    # --------------------------------------------------------
    # Quantitative phenotype:
    #
    # EpiGEN: Y ~ Normal(risk, stdev)
    #
    # equivalent to:
    #
    # Y = risk + Normal(0, stdev)
    # --------------------------------------------------------

    phenotype = (
        genetic
        +
        noise
    )

    out = pd.DataFrame({
        "IID":
            sample_ids.values,
        "Phenotype":
            phenotype,
        "Epistatic_Component":
            genetic,
    })

    pheno_file = (
        scenario_folder
        /
        f"{scenario}_phenotype.csv"
    )

    out.to_csv(
        pheno_file,
        index=False
    )

    truth = pairs.copy()

    truth[
        "Interaction_Type"
    ] = scenario

    truth[
        "Alpha"
    ] = alpha

    truth.to_csv(
        scenario_folder
        /
        f"{scenario}_true_pairs.csv",
        index=False
    )

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    vg = np.var(
        genetic,
        ddof=0
    )

    vp = np.var(
        phenotype,
        ddof=0
    )

    expected_fraction = (
        vg /
        (
            vg
            +
            noise_sd ** 2
        )
    )

    observed_fraction = (
        vg / vp
    )

    summary = {
        "Scenario":
            scenario,
        "Num_true_pairs":
            len(pairs),
        "Num_causal_snps":
            len(
                set(
                    pairs["SNP1"]
                )
                |
                set(
                    pairs["SNP2"]
                )
            ),
        "Alpha":
            alpha,
        "Noise_SD":
            noise_sd,
        "Phenotype_mean":
            phenotype.mean(),
        "Phenotype_SD":
            phenotype.std(),
        "Epistatic_mean":
            genetic.mean(),
        "Epistatic_SD":
            genetic.std(),
        "Epistatic_variance":
            vg,
        "Phenotype_variance":
            vp,
        "Expected_epistatic_fraction":
            expected_fraction,
        "Observed_epistatic_fraction":
            observed_fraction,
        "Noise_seed":
            seed,
    }

    return summary


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
        "--base_pairs_10",
        required=True
    )

    parser.add_argument(
        "--outdir",
        required=True
    )

    parser.add_argument(
        "--pair_counts",
        nargs="+",
        type=int,
        default=[
            50,
            100
        ]
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
        "--ld_r2_max",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--pair_seed",
        type=int,
        default=4242
    )

    parser.add_argument(
        "--noise_seed",
        type=int,
        default=2026
    )

    parser.add_argument(
        "--epigen-root",
        default=None,
        help="Path to a local clone of the EpiGEN repository."
    )

    args = parser.parse_args()

    configure_epigen(args.epigen_root)

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Genotype
    # ========================================================

    sample_ids, geno, maf_df = (
        load_genotype(
            args.genotype
        )
    )

    maf_df.to_csv(
        outdir
        /
        "vae2000_snp_maf.csv",
        index=False
    )

    # ========================================================
    # Existing 10-pair truth
    # ========================================================

    base_pairs = pd.read_csv(
        args.base_pairs_10
    )

    base_pairs = (
        normalize_existing_pairs(
            base_pairs
        )
    )

    print()
    print(
        "Existing causal pairs:",
        len(base_pairs)
    )

    if len(base_pairs) != 10:

        print(
            "WARNING: base pair file contains "
            f"{len(base_pairs)} pairs rather than 10."
        )

    # Check all base causal SNPs exist
    all_columns = set(
        geno.columns
    )

    base_snps = set(
        base_pairs["SNP1"]
    ) | set(
        base_pairs["SNP2"]
    )

    missing = (
        base_snps -
        all_columns
    )

    if missing:

        raise ValueError(
            "Existing causal SNPs missing from genotype: "
            + ", ".join(
                sorted(missing)
            )
        )

    # ========================================================
    # Build largest nested pair set once
    # ========================================================

    max_pairs = max(
        args.pair_counts
    )

    full_pairs = extend_pair_set(
        existing_pairs=
            base_pairs,
        geno=
            geno,
        maf_df=
            maf_df,
        target_pairs=
            max_pairs,
        maf_min=
            args.maf_min,
        maf_max=
            args.maf_max,
        ld_r2_max=
            args.ld_r2_max,
        seed=
            args.pair_seed
    )

    # ========================================================
    # Same residual-noise realization across all architectures
    # and pair counts
    # ========================================================

    noise_rng = (
        np.random.default_rng(
            args.noise_seed
        )
    )

    shared_noise = (
        noise_rng.normal(
            loc=0.0,
            scale=args.noise_sd,
            size=len(sample_ids)
        )
    )

    # ========================================================
    # 50 and 100 pairs
    # ========================================================

    for n_pairs in sorted(
        args.pair_counts
    ):

        print()
        print(
            "======================================================"
        )
        print(
            f"PAIR COUNT: {n_pairs}"
        )
        print(
            "======================================================"
        )

        pair_dir = (
            outdir
            /
            f"pairs_{n_pairs}"
        )

        pair_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        pairs = (
            full_pairs
            .iloc[
                :n_pairs
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        pairs["Pair_ID"] = [
            f"PAIR_{i+1}"
            for i in range(
                len(pairs)
            )
        ]

        # ----------------------------------------------------
        # Save ground truth
        # ----------------------------------------------------

        pair_file = (
            pair_dir
            /
            f"causal_pairs_{n_pairs}.csv"
        )

        pairs.to_csv(
            pair_file,
            index=False
        )

        # ----------------------------------------------------
        # Pair diagnostics
        # ----------------------------------------------------

        diagnostics = (
            pair_diagnostics(
                pairs,
                geno,
                maf_df
            )
        )

        diagnostics.to_csv(
            pair_dir
            /
            f"pair_diagnostics_{n_pairs}.csv",
            index=False
        )

        print(
            "Unique causal SNPs:",
            len(
                set(
                    pairs["SNP1"]
                )
                |
                set(
                    pairs["SNP2"]
                )
            )
        )

        print(
            "Maximum within-pair r2:",
            diagnostics[
                "Genotype_r2"
            ].max()
        )

        # ====================================================
        # Calibrate all four architectures
        # ====================================================

        calibration_rows = []
        simulation_rows = []

        for folder_name, scenario in (
            SCENARIOS.items()
        ):

            print()
            print(
                "----------------------------------------------"
            )
            print(
                f"{n_pairs} pairs | {scenario}"
            )
            print(
                "----------------------------------------------"
            )

            alpha, expected_frac, vg = (
                find_calibrated_alpha(
                    geno=geno,
                    pairs=pairs,
                    scenario=scenario,
                    target_fraction=
                        args.target,
                    noise_sd=
                        args.noise_sd
                )
            )

            print(
                f"Calibrated alpha: "
                f"{alpha:.10f}"
            )

            print(
                f"Expected epistatic fraction: "
                f"{expected_frac:.6f}"
            )

            calibration_rows.append({
                "Num_pairs":
                    n_pairs,
                "Scenario":
                    scenario,
                "Target_fraction":
                    args.target,
                "Calibrated_alpha":
                    alpha,
                "Epistatic_variance":
                    vg,
                "Expected_fraction":
                    expected_frac,
            })

            scenario_dir = (
                pair_dir
                /
                folder_name
            )

            summary = (
                simulate_scenario(
                    sample_ids=
                        sample_ids,
                    geno=
                        geno,
                    pairs=
                        pairs,
                    scenario_folder=
                        scenario_dir,
                    scenario=
                        scenario,
                    alpha=
                        alpha,
                    noise=
                        shared_noise,
                    noise_sd=
                        args.noise_sd,
                    seed=
                        args.noise_seed
                )
            )

            simulation_rows.append(
                summary
            )

            print(
                f"Observed epistatic fraction: "
                f"{summary['Observed_epistatic_fraction']:.6f}"
            )

        # ----------------------------------------------------
        # Save calibration
        # ----------------------------------------------------

        pd.DataFrame(
            calibration_rows
        ).to_csv(
            pair_dir
            /
            "calibration_summary.csv",
            index=False
        )

        # ----------------------------------------------------
        # Save phenotype QC
        # ----------------------------------------------------

        pd.DataFrame(
            simulation_rows
        ).to_csv(
            pair_dir
            /
            "simulation_summary.csv",
            index=False
        )

    print()
    print(
        "======================================================"
    )
    print(
        "ALL EpiGEN DENSITY SIMULATIONS COMPLETE"
    )
    print(
        "======================================================"
    )


if __name__ == "__main__":
    main()