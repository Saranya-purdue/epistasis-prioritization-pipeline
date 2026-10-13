from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _standardize_selected_genotypes(geno_mat: np.ndarray) -> np.ndarray:
    """
    Mean-impute NaNs and standardize each SNP column to mean 0, variance 1.
    Zero-variance columns are set to 0 after standardization.
    """
    geno = np.asarray(geno_mat, dtype=float).copy()

    col_means = np.nanmean(geno, axis=0)
    inds = np.where(np.isnan(geno))
    if len(inds[0]) > 0:
        geno[inds] = np.take(col_means, inds[1])

    col_means = np.mean(geno, axis=0)
    col_stds = np.std(geno, axis=0, ddof=0)
    col_stds[col_stds == 0] = 1.0

    geno = (geno - col_means) / col_stds
    geno = np.nan_to_num(geno, nan=0.0, posinf=0.0, neginf=0.0)
    return geno


def _rescale_effects(score: np.ndarray, target_variance: float) -> float:
    """
    Return multiplicative factor so that Var(score * factor) = target_variance.
    """
    current_variance = float(np.var(score))
    if current_variance <= 0:
        raise ValueError("Cannot rescale component with zero variance.")
    return np.sqrt(target_variance / current_variance)


def simulate_phenotype_sim1(
    geno_df: pd.DataFrame,
    num_causal_snp: int,
    tot_h2: float = 0.2,
    rho: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulation 1:
    - select N causal SNPs
    - split them into s1, s2, s3
    - build epistatic interactions only between s1 and s2
    - build marginal effects from all selected SNPs

    Parameters
    ----------
    geno_df : pd.DataFrame
        Genotype matrix with rows = samples and columns = SNPs.
    num_causal_snp : int
        Number of causal SNPs to select.
    tot_h2 : float, default=0.2
        Broad-sense heritability target.
    rho : float, default=0.05
        Fraction of genetic variance assigned to marginal effects.
        The remaining (1-rho) is assigned to epistatic effects.
    seed : int, default=42
        Random seed.

    Returns
    -------
    dict
        Simulation outputs and metadata.
    """
    if not (0 <= tot_h2 <= 1):
        raise ValueError("tot_h2 must be between 0 and 1.")
    if not (0 <= rho <= 1):
        raise ValueError("rho must be between 0 and 1.")
    if num_causal_snp < 3:
        raise ValueError("num_causal_snp must be at least 3.")
    if num_causal_snp > geno_df.shape[1]:
        raise ValueError("num_causal_snp exceeds number of available SNP columns.")

    rng = np.random.default_rng(seed)

    # Step 1: Select causal SNPs
    causal_snp_list = rng.choice(geno_df.columns.to_numpy(), size=num_causal_snp, replace=False)
    geno_mat = geno_df.loc[:, causal_snp_list].to_numpy(dtype=float)

    # Step 2: Mean-impute + standardize selected genotypes
    geno_std = _standardize_selected_genotypes(geno_mat)

    # Step 3: Determine interaction structure
    max_pairs = min(100, int(num_causal_snp * 0.8))
    s1_len = int(np.floor(np.sqrt(max_pairs)))
    s2_len = s1_len
    used_snps = s1_len + s2_len
    s3_len = num_causal_snp - used_snps

    if s1_len < 1 or s2_len < 1:
        raise ValueError("Too few SNPs to construct epistatic blocks.")

    snp_idx = np.arange(num_causal_snp)

    s1 = rng.choice(snp_idx, size=s1_len, replace=False)
    remaining = np.setdiff1d(snp_idx, s1)
    s2 = rng.choice(remaining, size=s2_len, replace=False)
    s3 = np.setdiff1d(remaining, s2)

    Xcausal1 = geno_std[:, s1]
    Xcausal2 = geno_std[:, s2]
    Xcausal3 = geno_std[:, s3] if s3_len > 0 else np.empty((geno_std.shape[0], 0))

    # Step 4: Epistatic component from s1 x s2 cross-block interactions
    Xepi = np.column_stack([
        Xcausal1[:, i] * Xcausal2[:, j]
        for i in range(s1_len)
        for j in range(s2_len)
    ])

    # Standardize interaction terms
    Xepi = Xepi - Xepi.mean(axis=0)
    epi_std = Xepi.std(axis=0, ddof=0)
    epi_std[epi_std == 0] = 1.0
    Xepi = Xepi / epi_std

    beta_epi = rng.normal(0.0, 1.0, size=Xepi.shape[1])
    y_epi_raw = Xepi @ beta_epi
    epi_scale = _rescale_effects(y_epi_raw, tot_h2 * (1.0 - rho))
    beta_epi = beta_epi * epi_scale
    y_epi = Xepi @ beta_epi

    # Step 5: Marginal component from all selected SNPs
    Xmarginal = np.column_stack([Xcausal1, Xcausal2, Xcausal3])
    beta_marg = rng.normal(0.0, 1.0, size=Xmarginal.shape[1])
    y_marg_raw = Xmarginal @ beta_marg
    marg_scale = _rescale_effects(y_marg_raw, tot_h2 * rho)
    beta_marg = beta_marg * marg_scale
    y_marginal = Xmarginal @ beta_marg

    # Step 6: Final phenotype
    n_ind = geno_std.shape[0]
    y_true = y_marginal + y_epi
    err = rng.normal(0.0, np.sqrt(max(0.0, 1.0 - tot_h2)), size=n_ind)
    y_obs = y_true + err

    sample_ids = geno_df.index.astype(str)

    phenotype_df = pd.DataFrame({
        "SampleID": sample_ids,
        "Phenotype": y_obs,
        "Marginal_Component": y_marginal,
        "Epistatic_Component": y_epi,
        "Genetic_Component": y_true,
    })

    s1_snps = causal_snp_list[s1].tolist()
    s2_snps = causal_snp_list[s2].tolist()
    s3_snps = causal_snp_list[s3].tolist()

    interaction_pairs = [
        (str(causal_snp_list[i]), str(causal_snp_list[j]))
        for i in s1 for j in s2
    ]

    beta_epistatic_df = pd.DataFrame(interaction_pairs, columns=["SNP1", "SNP2"])
    beta_epistatic_df["SNP_Pair"] = beta_epistatic_df["SNP1"] + "x" + beta_epistatic_df["SNP2"]
    beta_epistatic_df["Beta"] = beta_epi

    marginal_ids = causal_snp_list[np.concatenate([s1, s2, s3])]
    beta_marginal_df = pd.DataFrame({
        "SNP": marginal_ids,
        "Beta": beta_marg,
    })

    return {
        "phenotype_df": phenotype_df,
        "beta_epistatic_df": beta_epistatic_df,
        "beta_marginal_df": beta_marginal_df,
        "causal_snps": [str(x) for x in causal_snp_list.tolist()],
        "s1_snps": [str(x) for x in s1_snps],
        "s2_snps": [str(x) for x in s2_snps],
        "s3_snps": [str(x) for x in s3_snps],
        "num_epistatic_pairs": int(Xepi.shape[1]),
        "max_pairs_target": int(max_pairs),
        "s1_len": int(s1_len),
        "s2_len": int(s2_len),
        "s3_len": int(s3_len),
        "realized_var_marginal": float(np.var(y_marginal)),
        "realized_var_epistatic": float(np.var(y_epi)),
        "realized_var_genetic": float(np.var(y_true)),
        "realized_var_noise": float(np.var(err)),
    }
