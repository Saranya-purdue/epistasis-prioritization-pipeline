from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def standardize_genotype_df(geno_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize each SNP column to mean 0 and variance 1.
    Any zero-variance columns are set to 0 after standardization.
    """
    geno_std = (geno_df - geno_df.mean(axis=0)) / geno_df.std(axis=0, ddof=0)
    geno_std = geno_std.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return geno_std


def _rescale_to_target_variance(
    score: np.ndarray,
    target_variance: float,
) -> float:
    """
    Return the multiplicative scaling factor needed so that
    Var(score * scale_factor) = target_variance.
    """
    current_variance = float(np.var(score))
    if current_variance <= 0:
        raise ValueError("Cannot rescale a component with zero variance.")
    return np.sqrt(target_variance / current_variance)


def simulate_phenotype_with_overlap(
    geno_df: pd.DataFrame,
    num_additive: int,
    num_epistatic: int,
    overlap_proportion: float,
    h2_total: float,
    rho: float,
    seed: int,
    standardize_genotypes: bool = True,
) -> Dict[str, Any]:
    """
    Simulate phenotype with overlapping additive and epistatic SNP sets.

    Parameters
    ----------
    geno_df : pd.DataFrame
        Genotype matrix of shape (n_samples, n_snps).
    num_additive : int
        Number of SNPs in the additive set.
    num_epistatic : int
        Number of SNPs in the epistatic set.
    overlap_proportion : float
        Proportion of overlap between additive and epistatic SNP sets.
    h2_total : float
        Target broad-sense heritability.
    rho : float
        Fraction of genetic variance assigned to additive effects.
        The remaining (1 - rho) is assigned to epistatic effects.
    seed : int
        Random seed for reproducibility.
    standardize_genotypes : bool, default=True
        Whether to standardize genotype columns inside the function.

    Returns
    -------
    dict
        Simulation outputs and metadata.
    """
    if not (0 <= overlap_proportion <= 1):
        raise ValueError("overlap_proportion must be between 0 and 1.")
    if not (0 <= h2_total <= 1):
        raise ValueError("h2_total must be between 0 and 1.")
    if not (0 <= rho <= 1):
        raise ValueError("rho must be between 0 and 1.")
    if num_additive < 1 or num_epistatic < 2:
        raise ValueError("num_additive must be >= 1 and num_epistatic must be >= 2.")

    rng = np.random.default_rng(seed)

    geno_use = standardize_genotype_df(geno_df) if standardize_genotypes else geno_df.copy()
    n_samples, n_snps_total = geno_use.shape

    # Determine overlap and unique SNP counts
    num_overlap = int(min(num_additive, num_epistatic) * overlap_proportion)
    num_unique_add = num_additive - num_overlap
    num_unique_epi = num_epistatic - num_overlap
    total_required_snps = num_overlap + num_unique_add + num_unique_epi

    if total_required_snps > n_snps_total:
        raise ValueError(
            "Not enough SNPs in geno_df to satisfy the requested overlap configuration."
        )

    # Select SNP indices
    selected_indices = rng.choice(n_snps_total, size=total_required_snps, replace=False)

    shared_indices = selected_indices[:num_overlap].tolist()
    unique_add_indices = selected_indices[num_overlap:num_overlap + num_unique_add].tolist()
    unique_epi_indices = selected_indices[num_overlap + num_unique_add:].tolist()

    additive_indices = shared_indices + unique_add_indices
    epistatic_indices = shared_indices + unique_epi_indices

    snp_names = list(geno_use.columns)
    shared_snps = [str(snp_names[i]) for i in shared_indices]
    additive_snps = [str(snp_names[i]) for i in additive_indices]
    epistatic_snps = [str(snp_names[i]) for i in epistatic_indices]

    geno_values = geno_use.to_numpy(dtype=float)

    # Additive component
    X_add = geno_values[:, additive_indices]
    beta_add = rng.normal(loc=0.0, scale=1.0, size=X_add.shape[1])

    y_add_raw = X_add @ beta_add
    add_target_variance = h2_total * rho
    add_scale = _rescale_to_target_variance(y_add_raw, add_target_variance)
    beta_add = beta_add * add_scale
    y_add = X_add @ beta_add

    # Epistatic component: ALL pairwise interactions
    epistatic_pairs_idx: List[Tuple[int, int]] = list(combinations(epistatic_indices, 2))
    if len(epistatic_pairs_idx) == 0:
        raise ValueError("No epistatic pairs were generated. Increase num_epistatic.")

    epistatic_pairs = [(str(snp_names[i]), str(snp_names[j])) for i, j in epistatic_pairs_idx]

    X_epi = np.column_stack([
        geno_values[:, i] * geno_values[:, j]
        for i, j in epistatic_pairs_idx
    ])

    # Standardize interaction terms
    X_epi = X_epi - X_epi.mean(axis=0)
    std_epi = X_epi.std(axis=0, ddof=0)
    std_epi[std_epi == 0] = 1.0
    X_epi = X_epi / std_epi

    beta_epi = rng.normal(loc=0.0, scale=1.0, size=X_epi.shape[1])

    y_epi_raw = X_epi @ beta_epi
    epi_target_variance = h2_total * (1.0 - rho)
    epi_scale = _rescale_to_target_variance(y_epi_raw, epi_target_variance)
    beta_epi = beta_epi * epi_scale
    y_epi = X_epi @ beta_epi

    # Final phenotype
    y_true = y_add + y_epi
    noise_sd = np.sqrt(max(0.0, 1.0 - h2_total))
    noise = rng.normal(loc=0.0, scale=noise_sd, size=n_samples)
    y_obs = y_true + noise

    phenotype_df = pd.DataFrame({
        "Phenotype": y_obs,
        "Additive_Component": y_add,
        "Epistatic_Component": y_epi,
        "Genetic_Component": y_true,
    }, index=geno_use.index)

    return {
        "phenotype_df": phenotype_df,
        "additive_snps": additive_snps,
        "epistatic_snps": epistatic_snps,
        "shared_snps": shared_snps,
        "epistatic_pairs": epistatic_pairs,
        "beta_add": beta_add,
        "beta_epi": beta_epi,
        "num_overlap": num_overlap,
        "num_unique_add": num_unique_add,
        "num_unique_epi": num_unique_epi,
        "num_epistatic_pairs": len(epistatic_pairs),
        "realized_var_add": float(np.var(y_add)),
        "realized_var_epi": float(np.var(y_epi)),
        "realized_var_genetic": float(np.var(y_true)),
        "realized_var_noise": float(np.var(noise)),
    }