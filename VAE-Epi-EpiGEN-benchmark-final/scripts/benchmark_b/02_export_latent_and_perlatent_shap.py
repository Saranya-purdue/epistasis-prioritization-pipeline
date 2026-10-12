#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_vae_module(script_path):
    spec = importlib.util.spec_from_file_location(
        "vae_pipeline",
        script_path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--vae-dir", required=True)
    ap.add_argument("--vae-script", required=True)
    ap.add_argument("--top-per-latent", type=int, default=100)

    args = ap.parse_args()

    vae_dir = Path(args.vae_dir)

    # =========================================================
    # Load run configuration
    # =========================================================

    with open(
        vae_dir / "run_arguments.json"
    ) as f:

        cfg = json.load(f)

    print("Run configuration:")
    print(json.dumps(cfg, indent=2))


    # =========================================================
    # Import exact VAE architecture used originally
    # =========================================================

    mod = load_vae_module(
        args.vae_script
    )

    import tensorflow as tf
    import shap


    # =========================================================
    # Existing genotype cache
    # =========================================================

    geno_file = (
        vae_dir /
        "cache/genotype_int8.npy"
    )

    sample_file = (
        vae_dir /
        "cache/sample_ids.txt"
    )

    meta_file = (
        vae_dir /
        "variant_metadata_with_MAF.csv"
    )

    G = np.load(
        geno_file,
        mmap_mode="r"
    )

    sample_ids = np.loadtxt(
        sample_file,
        dtype=str,
        ndmin=1
    )

    meta = pd.read_csv(
        meta_file
    )

    n_samples, n_snps = G.shape

    print(
        f"Genotype cache: {n_samples:,} x {n_snps:,}"
    )


    # =========================================================
    # Reconstruct trained VAE
    # =========================================================

    latent_dim = int(
        cfg.get("latent_dim", 128)
    )

    vae = mod.VAE(
        original_dim=n_snps,
        latent_dim=latent_dim,
        hidden1=int(cfg.get("hidden1", 512)),
        hidden2=int(cfg.get("hidden2", 256)),
        dropout=float(cfg.get("dropout", 0.10)),
        l2_reg=float(cfg.get("l2_reg", 1e-5)),
        beta=float(cfg.get("beta", 0.01)),
    )

    # Build model first
    dummy = tf.zeros(
        (1, n_snps),
        dtype=tf.float32
    )

    _ = vae(
        dummy,
        training=False
    )

    weight_file = (
        vae_dir /
        "model" /
        f"vae_latent_{latent_dim}.weights.h5"
    )

    print(
        "Loading weights:",
        weight_file
    )

    vae.load_weights(
        weight_file
    )


    # =========================================================
    # Export full z_mean latent matrix
    # =========================================================

    latent_out = (
        vae_dir /
        "z_mean_full.npy"
    )

    if latent_out.exists():

        print(
            "Reusing existing:",
            latent_out
        )

        Z = np.load(
            latent_out,
            mmap_mode="r"
        )

    else:

        print(
            "Exporting full latent matrix..."
        )

        Z = mod.encode_z_mean(
            vae,
            G,
            np.arange(n_samples),
            batch_size=256
        )

        Z = Z.astype(
            np.float32
        )

        np.save(
            latent_out,
            Z
        )

        print(
            "Saved:",
            latent_out,
            Z.shape
        )


    # =========================================================
    # Latent dimensions previously used for SHAP
    # =========================================================

    latent_indices = np.load(
        vae_dir /
        "SHAP_latent_indices.npy"
    ).astype(int)

    print(
        "SHAP latent dimensions:",
        len(latent_indices)
    )


    # =========================================================
    # Reuse EXACT background/explain samples from original run
    # =========================================================

    bg_idx = np.load(
        vae_dir /
        "shap_background_global_indices.npy"
    )

    ex_idx = np.load(
        vae_dir /
        "shap_explain_global_indices.npy"
    )

    background = (
        np.asarray(
            G[bg_idx, :],
            dtype=np.float32
        )
        / 2.0
    )

    explain_data = (
        np.asarray(
            G[ex_idx, :],
            dtype=np.float32
        )
        / 2.0
    )

    print(
        "Background:",
        background.shape
    )

    print(
        "Explained samples:",
        explain_data.shape
    )


    # =========================================================
    # Per-latent SNP SHAP
    #
    # Output matrix:
    # rows    = latent dimensions
    # columns = 10,000 SNPs
    # =========================================================

    out_dir = (
        vae_dir /
        "per_latent_shap"
    )

    out_dir.mkdir(
        exist_ok=True
    )

    output_chunk = int(
        cfg.get(
            "shap_output_chunk",
            8
        )
    )

    nsamples = int(
        cfg.get(
            "shap_nsamples",
            32
        )
    )

    full_matrix = np.zeros(
        (
            len(latent_indices),
            n_snps
        ),
        dtype=np.float32
    )


    for chunk_start in range(
        0,
        len(latent_indices),
        output_chunk
    ):

        chunk_latents = latent_indices[
            chunk_start:
            chunk_start + output_chunk
        ]

        tag = "_".join(
            map(
                str,
                chunk_latents.tolist()
            )
        )

        outfile = (
            out_dir /
            f"per_latent_abs_shap_{tag}.npy"
        )

        if outfile.exists():

            print(
                "Reusing:",
                outfile.name
            )

            chunk_matrix = np.load(
                outfile
            )

        else:

            print(
                "\nComputing latent SHAP:",
                chunk_latents.tolist()
            )

            chunk_model = (
                mod.build_selected_mean_encoder(
                    vae,
                    n_snps,
                    chunk_latents.tolist()
                )
            )

            explainer = shap.GradientExplainer(
                chunk_model,
                background,
                batch_size=min(
                    32,
                    len(background)
                ),
            )

            shap_values = (
                explainer.shap_values(
                    explain_data,
                    nsamples=nsamples,
                    rseed=42 + chunk_start,
                )
            )

            values = mod.normalize_shap_array(
                shap_values,
                n_samples=len(explain_data),
                n_features=n_snps,
                n_outputs=len(chunk_latents),
            )

            # values:
            # samples x SNPs x latent outputs
            #
            # retain latent dimension!
            #
            # mean over samples only:
            # SNPs x latent
            mean_abs = np.abs(
                values
            ).mean(
                axis=0
            )

            # latent x SNP
            chunk_matrix = (
                mean_abs.T
                .astype(np.float32)
            )

            np.save(
                outfile,
                chunk_matrix
            )

            del (
                values,
                shap_values,
                explainer,
                chunk_model
            )


        stop = (
            chunk_start
            +
            len(chunk_latents)
        )

        full_matrix[
            chunk_start:stop,
            :
        ] = chunk_matrix


    # =========================================================
    # Save complete matrix
    # =========================================================

    matrix_file = (
        vae_dir /
        "per_latent_mean_abs_shap.npy"
    )

    np.save(
        matrix_file,
        full_matrix
    )

    np.save(
        vae_dir /
        "per_latent_shap_latent_indices.npy",
        latent_indices
    )

    print(
        "\nSaved complete per-latent matrix:"
    )

    print(
        matrix_file,
        full_matrix.shape
    )


    # =========================================================
    # Also generate your original-style top-100 files
    #
    # This makes your existing phenotype-weighting script usable.
    # =========================================================

    top_dir = (
        vae_dir /
        "latent_top_features"
    )

    top_dir.mkdir(
        exist_ok=True
    )

    features = (
        meta["VARIANT_ID"]
        .astype(str)
        .to_numpy()
    )

    top_n = min(
        args.top_per_latent,
        n_snps
    )


    for row_idx, latent_idx in enumerate(
        latent_indices
    ):

        scores = full_matrix[
            row_idx,
            :
        ]

        top_idx = np.argpartition(
            scores,
            -top_n
        )[-top_n:]

        top_idx = top_idx[
            np.argsort(
                scores[top_idx]
            )[::-1]
        ]

        out = pd.DataFrame({
            "Feature":
                features[top_idx],

            "Mean_SHAP":
                scores[top_idx],
        })

        out.to_csv(
            top_dir /
            f"latent_{latent_idx}_top_features.csv",
            index=False
        )


    print(
        "\nTop-feature files:",
        top_dir
    )

    print(
        "Number generated:",
        len(latent_indices)
    )

    print(
        "\nDONE"
    )


if __name__ == "__main__":
    main()