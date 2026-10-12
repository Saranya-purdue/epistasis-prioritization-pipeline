#!/usr/bin/env python3

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# ============================================================
# Utilities
# ============================================================

def clean_ids(values):

    return pd.Index(
        pd.Series(values, dtype="object")
        .astype(str)
        .str.strip()
        .str.replace(r"^\d+_", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
    )


def load_module(script_path):

    spec = importlib.util.spec_from_file_location(
        "vae_pipeline",
        script_path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_phenotype(path):

    # automatically detect comma/tab delimiter
    df = pd.read_csv(
        path,
        sep=None,
        engine="python"
    )

    print(
        "Phenotype columns:",
        list(df.columns)
    )

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    # ID column
    id_candidates = [
        "id",
        "sample_id",
        "iid",
        "snp"
    ]

    id_col = None

    for x in id_candidates:

        if x in lower:
            id_col = lower[x]
            break

    if id_col is None:
        id_col = df.columns[0]

    # phenotype column
    pheno_candidates = [
        "phenotype",
        "pheno"
    ]

    pheno_col = None

    for x in pheno_candidates:

        if x in lower:
            pheno_col = lower[x]
            break

    if pheno_col is None:

        if len(df.columns) == 2:
            pheno_col = df.columns[1]
        else:
            raise ValueError(
                "Could not automatically identify phenotype column."
            )

    print("ID column:", id_col)
    print("Phenotype column:", pheno_col)

    ids = clean_ids(
        df[id_col]
    )

    result = pd.DataFrame({
        "ID": ids,
        "phenotype": pd.to_numeric(
            df[pheno_col],
            errors="coerce"
        )
    })

    if result["ID"].duplicated().any():
        raise ValueError(
            "Duplicate phenotype IDs after cleaning."
        )

    return result.set_index("ID")


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--vae-dir",
        required=True
    )

    ap.add_argument(
        "--vae-script",
        required=True
    )

    ap.add_argument(
        "--phenotype-file",
        required=True
    )

    ap.add_argument(
        "--outdir",
        required=True
    )

    ap.add_argument(
        "--final-n",
        type=int,
        default=2000
    )

    ap.add_argument(
        "--maf-min",
        type=float,
        default=0.01
    )

    ap.add_argument(
        "--ld-r2",
        type=float,
        default=0.20
    )

    ap.add_argument(
        "--ld-window",
        type=int,
        default=500000
    )

    ap.add_argument(
        "--discovery-frac",
        type=float,
        default=0.80
    )

    ap.add_argument(
        "--rf-n-estimators",
        type=int,
        default=500
    )

    ap.add_argument(
        "--rf-max-depth",
        type=int,
        default=6
    )

    ap.add_argument(
        "--max-shap-samples",
        type=int,
        default=5000
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = ap.parse_args()


    vae_dir = Path(args.vae_dir)
    outdir = Path(args.outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # Load VAE helper functions
    # ========================================================

    mod = load_module(
        args.vae_script
    )


    # ========================================================
    # Load cached genotype
    # ========================================================

    print("\nLoading genotype cache...")

    G = np.load(
        vae_dir /
        "cache/genotype_int8.npy",
        mmap_mode="r"
    )

    sample_ids = np.loadtxt(
        vae_dir /
        "cache/sample_ids.txt",
        dtype=str,
        ndmin=1
    )

    sample_ids_clean = clean_ids(
        sample_ids
    )

    if sample_ids_clean.duplicated().any():

        raise ValueError(
            "Duplicate genotype IDs after cleaning."
        )

    n_samples, n_snps = G.shape

    print(
        f"Genotype: {n_samples:,} x {n_snps:,}"
    )


    # ========================================================
    # Latent representation
    # ========================================================

    print("\nLoading latent space...")

    Z = np.load(
        vae_dir /
        "z_mean_full.npy",
        mmap_mode="r"
    )

    print(
        "Latent:",
        Z.shape
    )

    if Z.shape[0] != n_samples:

        raise ValueError(
            "Latent/genotype sample count mismatch."
        )


    # ========================================================
    # Phenotype
    # ========================================================

    print("\nLoading phenotype...")

    pheno = load_phenotype(
        args.phenotype_file
    )


    # Preserve genotype row order
    keep = sample_ids_clean.isin(
        pheno.index
    )

    genotype_rows = np.flatnonzero(
        keep
    )

    aligned_ids = sample_ids_clean[
        keep
    ]

    y = (
        pheno.loc[
            aligned_ids,
            "phenotype"
        ]
        .to_numpy(dtype=float)
    )

    good = np.isfinite(y)

    genotype_rows = genotype_rows[
        good
    ]

    aligned_ids = aligned_ids[
        good
    ]

    y = y[
        good
    ]

    Z_aligned = np.asarray(
        Z[genotype_rows, :],
        dtype=np.float32
    )


    print(
        "Aligned samples:",
        len(y)
    )

    print(
        "Aligned latent:",
        Z_aligned.shape
    )


    # ========================================================
    # Discovery / holdout
    # ========================================================

    all_pos = np.arange(
        len(y)
    )

    discovery_pos, holdout_pos = train_test_split(
        all_pos,
        train_size=args.discovery_frac,
        random_state=args.seed,
        shuffle=True
    )

    pd.Series(
        aligned_ids[discovery_pos],
        name="sample_id"
    ).to_csv(
        outdir /
        "discovery_ids.csv",
        index=False
    )

    pd.Series(
        aligned_ids[holdout_pos],
        name="sample_id"
    ).to_csv(
        outdir /
        "holdout_ids.csv",
        index=False
    )


    X_disc = Z_aligned[
        discovery_pos,
        :
    ]

    y_disc = y[
        discovery_pos
    ]


    print(
        "Discovery:",
        len(discovery_pos)
    )

    print(
        "Holdout:",
        len(holdout_pos)
    )


    # ========================================================
    # RF: latent -> phenotype
    # ========================================================

    print(
        "\nFitting latent -> phenotype RF..."
    )

    rf = RandomForestRegressor(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        n_jobs=-1,
        random_state=args.seed
    )

    rf.fit(
        X_disc,
        y_disc
    )


    # ========================================================
    # SHAP samples for latent phenotype relevance
    # ========================================================

    rng = np.random.default_rng(
        args.seed
    )

    n_shap = min(
        args.max_shap_samples,
        len(X_disc)
    )

    shap_pos = rng.choice(
        len(X_disc),
        size=n_shap,
        replace=False
    )

    X_shap = X_disc[
        shap_pos,
        :
    ]


    print(
        "Latent SHAP samples:",
        n_shap
    )


    # ========================================================
    # Latent phenotype weights
    # ========================================================

    print(
        "\nCalculating phenotype relevance weights..."
    )

    explainer = shap.TreeExplainer(
        rf
    )

    shap_values = explainer.shap_values(
        X_shap,
        check_additivity=False
    )

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim != 2:

        raise RuntimeError(
            f"Unexpected latent SHAP shape: "
            f"{shap_values.shape}"
        )


    latent_mean_abs = (
        np.abs(
            shap_values
        )
        .mean(axis=0)
    )

    total = latent_mean_abs.sum()

    if total <= 0:

        raise RuntimeError(
            "All latent phenotype SHAP values are zero."
        )


    latent_weights = (
        latent_mean_abs /
        total
    )


    weights_df = pd.DataFrame({
        "latent_index":
            np.arange(
                Z.shape[1]
            ),

        "Latent_Mean_SHAP":
            latent_mean_abs,

        "Latent_Weight":
            latent_weights
    })

    weights_df = weights_df.sort_values(
        "Latent_Weight",
        ascending=False
    )


    weights_df.to_csv(
        outdir /
        "latent_phenotype_weights.csv",
        index=False
    )


    print("\nTop phenotype-relevant latents:")

    print(
        weights_df.head(10)
        .to_string(index=False)
    )


    # ========================================================
    # Phenotype-weighted SNP SHAP using ALL 10,000 SNPs
    # ========================================================

    print(
        "\nCalculating phenotype-weighted SHAP "
        "for all input SNPs..."
    )

    # --------------------------------------------------------
    # Load per-latent SNP SHAP matrix
    #
    # shape:
    #   n_SHAP_latents x n_SNPs
    # --------------------------------------------------------

    per_latent_shap = np.load(
        vae_dir /
        "per_latent_mean_abs_shap.npy",
        mmap_mode="r"
    )

    shap_latent_indices = np.load(
        vae_dir /
        "per_latent_shap_latent_indices.npy"
    ).astype(int)

    print(
        "Per-latent SHAP matrix:",
        per_latent_shap.shape
    )

    print(
        "SHAP latent indices:",
        len(shap_latent_indices)
    )

    if per_latent_shap.shape[1] != n_snps:

        raise ValueError(
            "Per-latent SHAP SNP dimension does not "
            "match genotype SNP count."
        )

    if per_latent_shap.shape[0] != len(
        shap_latent_indices
    ):

        raise ValueError(
            "Number of SHAP latent indices does not "
            "match per-latent SHAP matrix."
        )


    # --------------------------------------------------------
    # Select phenotype weights for the latent dimensions
    # represented in the SNP-SHAP matrix
    # --------------------------------------------------------

    used_weights = latent_weights[
        shap_latent_indices
    ].astype(np.float64)

    weight_sum = used_weights.sum()

    if weight_sum <= 0:

        raise RuntimeError(
            "Phenotype latent weights sum to zero."
        )

    used_weights = (
        used_weights /
        weight_sum
    )


    # --------------------------------------------------------
    # Weighted SHAP for every SNP
    #
    # per_latent_shap:
    #       latent x SNP
    #
    # used_weights:
    #       latent
    #
    # result:
    #       SNP
    # --------------------------------------------------------

    weighted_scores = np.asarray(
        used_weights @ per_latent_shap,
        dtype=np.float64
    )

    print(
        "Weighted SNP score vector:",
        weighted_scores.shape
    )


    # ========================================================
    # Variant metadata
    # ========================================================

    meta = pd.read_csv(
        vae_dir /
        "variant_metadata_with_MAF.csv"
    )

    if len(meta) != n_snps:

        raise ValueError(
            "Variant metadata count does not match "
            "genotype SNP count."
        )


    ranking = meta.copy()

    ranking[
        "Weighted_SHAP"
    ] = weighted_scores

    # shap_guided_ld_filter() expects a column called SHAP_score
    ranking[
        "SHAP_score"
    ] = weighted_scores


    print(
        "Candidates before MAF filter:",
        len(ranking)
    )


    # ========================================================
    # MAF filter
    # ========================================================

    ranking = ranking[
        ranking["MAF"] >
        args.maf_min
    ].copy()


    print(
        f"Candidates after MAF > "
        f"{args.maf_min}:",
        len(ranking)
    )


    ranking = ranking.sort_values(
        [
            "Weighted_SHAP",
            "MAF"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(
        drop=True
    )


    ranking[
        "Weighted_SHAP_rank"
    ] = np.arange(
        1,
        len(ranking) + 1
    )


    ranking.to_csv(
        outdir /
        "all_SNP_Weighted_SHAP_ranking.csv",
        index=False
    )


    # ========================================================
    # LD filtering to exactly 2,000
    # ========================================================

    print(
        "\nRunning phenotype-weighted LD filtering..."
    )


    selected, rejected = (
        mod.shap_guided_ld_filter(
            ranking_df=ranking,
            G=G,
            target_n=args.final_n,
            r2_threshold=args.ld_r2,
            window_bp=args.ld_window
        )
    )


    selected.to_csv(
        outdir /
        f"FINAL_{args.final_n}_"
        f"Weighted_SHAP_LD_filtered_SNPs.csv",
        index=False
    )


    rejected.to_csv(
        outdir /
        "SNPs_removed_due_to_high_LD.csv",
        index=False
    )


    # ========================================================
    # Variant lists
    # ========================================================

    np.savetxt(
        outdir /
        f"FINAL_{args.final_n}_VARIANT_IDs.txt",
        selected[
            "VARIANT_ID"
        ].astype(str),
        fmt="%s"
    )


    # ========================================================
    # Extract phenotype-specific genotype
    # ========================================================

    final_indices = (
        selected[
            "feature_index"
        ]
        .astype(int)
        .to_numpy()
    )


    final_G = np.asarray(
        G[
            :,
            final_indices
        ],
        dtype=np.int8
    )


    np.save(
        outdir /
        f"genotype_FINAL_{args.final_n}.npy",
        final_G
    )


    final_df = pd.DataFrame(
        final_G,
        index=sample_ids,
        columns=selected[
            "VARIANT_ID"
        ].astype(str)
    )

    final_df.index.name = (
        "sample_id"
    )


    final_df.to_csv(
        outdir /
        f"genotype_FINAL_{args.final_n}.csv"
    )


    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "n_samples":
            int(n_samples),

        "n_input_snps":
            int(n_snps),

        "latent_dimensions":
            int(Z.shape[1]),

        "n_discovery":
            int(len(discovery_pos)),

        "n_holdout":
            int(len(holdout_pos)),

        "latent_shap_samples":
            int(n_shap),

        "candidate_universe_before_MAF":
            int(n_snps),

        "candidate_universe_after_MAF":
            int(len(ranking)),

        "final_selected_snps":
            int(len(selected)),

        "maf_min":
            float(args.maf_min),

        "ld_r2":
            float(args.ld_r2),

        "ld_window":
            int(args.ld_window)
    }


    with open(
        outdir /
        "weighted_selection_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )


    print("\n======================================")
    print("PHENOTYPE-WEIGHTED SELECTION COMPLETE")
    print("======================================")

    print(
        "Selected SNPs:",
        len(selected)
    )

    print(
        "Genotype:",
        final_G.shape
    )

    print(
        "Output:",
        outdir
    )


if __name__ == "__main__":
    main()