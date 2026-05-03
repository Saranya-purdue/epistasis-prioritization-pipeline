
"""
Stage 1: Phenotype-relevance weighting of VAE-SHAP SNPs.

This script implements the first stage of the VAE-RF epistasis prioritization
framework:

1. Load chromosome-specific genotype data and VAE latent representations.
2. Align genotype, latent, phenotype, and covariate data by sample ID.
3. Fit a random forest model from latent dimensions to phenotype.
4. Estimate phenotype relevance of each latent dimension using SHAP values.
5. Combine latent phenotype relevance weights with VAE-SHAP SNP attributions.
6. Rank SNPs by phenotype-weighted SHAP score.
7. Apply MAF filtering and greedy LD pruning.
8. Save chromosome-level and combined ranked SNP outputs.

The code is designed to be phenotype-agnostic and can be used for continuous
traits such as BMI, height, or education years, and binary traits such as
diabetes.

Author: saranya arirangan
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOGGER = logging.getLogger("stage1_phenotype_relevance")


def setup_logging(log_file: Optional[Path] = None) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    LOGGER.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def clean_ids(values: Iterable[Any]) -> pd.Series:
    """Clean sample IDs consistently across genotype, phenotype, and latent files."""
    return (
        pd.Series(values, dtype="object")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"^\d+_", "", regex=True)
    )


def format_path(pattern: str, **kwargs: Any) -> Path:
    """Format a file path pattern using chromosome or latent index values."""
    return Path(pattern.format(**kwargs))


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column from candidates that exists in df."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def extract_rsid(x: Any) -> str:
    """Extract rsID from strings such as rs123, chr1:123:rs123, or rs123:123:A:G."""
    if pd.isna(x):
        return ""
    match = re.search(r"rs\d+", str(x))
    return match.group(0) if match else str(x).strip()


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_phenotype_table(config: Dict[str, Any]) -> pd.DataFrame:
    phenotype_file = Path(config["phenotype_file"])
    id_col = config["id_col"]
    phenotype_col = config["phenotype_col"]
    covariate_cols = config.get("covariate_cols", [])

    LOGGER.info("Loading phenotype file: %s", phenotype_file)
    pheno = pd.read_csv(phenotype_file, low_memory=False)
    pheno.columns = pheno.columns.str.strip()

    required = [id_col, phenotype_col] + covariate_cols
    missing = [c for c in required if c not in pheno.columns]
    if missing:
        raise ValueError(f"Phenotype file is missing required columns: {missing}")

    pheno = pheno[required].copy()
    pheno[id_col] = clean_ids(pheno[id_col]).values

    for col in [phenotype_col] + covariate_cols:
        pheno[col] = safe_numeric(pheno[col])

    pheno = pheno.dropna(subset=[id_col, phenotype_col] + covariate_cols).copy()

    LOGGER.info("Phenotype complete-case sample size: %d", len(pheno))
    return pheno


def load_genotype_header(genotype_file: Path) -> Tuple[str, List[str]]:
    header = pd.read_csv(genotype_file, nrows=0)
    columns = header.columns.tolist()
    if len(columns) < 2:
        raise ValueError(f"Genotype file has too few columns: {genotype_file}")
    id_col = columns[0]
    snp_cols = columns[1:]
    return id_col, snp_cols


def load_genotype_for_snps(
    genotype_file: Path,
    snps: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load genotype file. If snps is provided, load only those SNP columns."""
    genotype_id_col, all_snp_cols = load_genotype_header(genotype_file)

    if snps is None:
        usecols = [genotype_id_col] + all_snp_cols
    else:
        snp_set = set(snps)
        present_snps = [s for s in all_snp_cols if s in snp_set]
        usecols = [genotype_id_col] + present_snps

    geno = pd.read_csv(genotype_file, usecols=usecols, low_memory=False)
    geno = geno.rename(columns={genotype_id_col: "SUBJID"})
    geno["SUBJID"] = clean_ids(geno["SUBJID"]).values

    for col in geno.columns:
        if col != "SUBJID":
            geno[col] = safe_numeric(geno[col])

    return geno


def load_genotype_ids(genotype_file: Path) -> pd.Series:
    genotype_id_col, _ = load_genotype_header(genotype_file)
    ids = pd.read_csv(genotype_file, usecols=[genotype_id_col], low_memory=False)
    return clean_ids(ids[genotype_id_col])


def load_latent_matrix(
    latent_file: Path,
    genotype_ids: pd.Series,
    latent_ids_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Load latent matrix and attach sample IDs."""
    LOGGER.info("Loading latent matrix: %s", latent_file)
    Z = np.load(latent_file)

    if Z.ndim != 2:
        raise ValueError(f"Latent matrix must be 2D. Got shape {Z.shape} from {latent_file}")

    latent_cols = [f"z{i}" for i in range(Z.shape[1])]

    if latent_ids_file is not None and latent_ids_file.exists():
        ids_df = pd.read_csv(latent_ids_file, low_memory=False)
        first_col = ids_df.columns[0]
        latent_ids = clean_ids(ids_df[first_col])
        if len(latent_ids) != Z.shape[0]:
            raise ValueError(
                f"latent_ids length ({len(latent_ids)}) does not match latent rows ({Z.shape[0]})"
            )
    else:
        latent_ids = genotype_ids
        if len(latent_ids) != Z.shape[0]:
            raise ValueError(
                "Latent row count does not match genotype sample count and no latent_ids_file "
                f"was provided. latent rows={Z.shape[0]}, genotype ids={len(latent_ids)}"
            )

    latent_df = pd.DataFrame(Z, columns=latent_cols)
    latent_df.insert(0, "SUBJID", latent_ids.values)

    return latent_df


# -----------------------------------------------------------------------------
# Phenotype preparation
# -----------------------------------------------------------------------------

def residualize_continuous_trait(
    df: pd.DataFrame,
    phenotype_col: str,
    covariate_cols: List[str],
) -> pd.Series:
    """
    Residualize a continuous phenotype against covariates.

    The returned phenotype is residual + original mean, preserving the trait scale.
    """
    if not covariate_cols:
        return df[phenotype_col].astype(float)

    X_cov = df[covariate_cols].to_numpy(dtype=float)
    y = df[phenotype_col].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(X_cov, y)

    predicted = model.predict(X_cov)
    adjusted = y - predicted + np.nanmean(y)

    return pd.Series(adjusted, index=df.index, name=f"{phenotype_col}_adjusted")


def prepare_model_xy(
    latent_df: pd.DataFrame,
    pheno: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Align latent features with phenotype table and return X, y, merged dataframe."""
    id_col = config["id_col"]
    phenotype_col = config["phenotype_col"]
    covariate_cols = config.get("covariate_cols", [])
    trait_type = config["trait_type"].lower()

    latent = latent_df.copy()
    latent["SUBJID"] = clean_ids(latent["SUBJID"]).values

    pheno_use = pheno.copy()
    pheno_use[id_col] = clean_ids(pheno_use[id_col]).values
    pheno_use = pheno_use.rename(columns={id_col: "SUBJID"})

    merged = latent.merge(pheno_use, on="SUBJID", how="inner")
    LOGGER.info("Aligned latent + phenotype sample size: %d", len(merged))

    latent_cols = [c for c in latent.columns if c.startswith("z")]
    X = merged[latent_cols].copy()

    if trait_type == "continuous":
        y = residualize_continuous_trait(
            merged,
            phenotype_col=phenotype_col,
            covariate_cols=covariate_cols,
        )
    elif trait_type == "binary":
        y = merged[phenotype_col].astype(float).round().astype(int)
    else:
        raise ValueError("trait_type must be either 'continuous' or 'binary'.")

    return X, y, merged


# -----------------------------------------------------------------------------
# Random forest + latent SHAP weights
# -----------------------------------------------------------------------------

def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
):
    trait_type = config["trait_type"].lower()

    common_params = dict(
        n_estimators=int(config.get("rf_n_estimators", 500)),
        max_depth=config.get("rf_max_depth", None),
        min_samples_leaf=int(config.get("rf_min_samples_leaf", 2)),
        max_features=config.get("rf_max_features", "sqrt"),
        random_state=int(config.get("random_state", 42)),
        n_jobs=int(config.get("n_jobs", -1)),
    )

    if trait_type == "continuous":
        model = RandomForestRegressor(**common_params)
    else:
        model = RandomForestClassifier(
            **common_params,
            class_weight=config.get("class_weight", "balanced"),
        )

    model.fit(X_train, y_train)
    return model


def compute_latent_importance_with_shap(
    model,
    X_reference: pd.DataFrame,
    trait_type: str,
    use_shap: bool = True,
) -> pd.DataFrame:
    """
    Compute latent dimension importance.

    Prefer SHAP TreeExplainer when available. If SHAP is not installed or fails,
    fall back to random forest feature_importances_.
    """
    latent_cols = X_reference.columns.tolist()

    if use_shap:
        try:
            import shap  # type: ignore

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_reference)

            if isinstance(shap_values, list):
                # Binary classifier: use positive class when available.
                values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                values = shap_values

            values = np.asarray(values)

            if values.ndim == 3:
                # Some SHAP versions return samples x features x classes.
                values = values[:, :, -1]

            importance = np.abs(values).mean(axis=0)

            method = "shap_mean_abs"

        except Exception as exc:
            LOGGER.warning("SHAP failed; falling back to RF feature_importances_. Error: %s", exc)
            importance = np.asarray(model.feature_importances_)
            method = "rf_feature_importance"
    else:
        importance = np.asarray(model.feature_importances_)
        method = "rf_feature_importance"

    total = np.nansum(importance)
    if total <= 0 or np.isnan(total):
        weights = np.repeat(1.0 / len(importance), len(importance))
    else:
        weights = importance / total

    out = pd.DataFrame(
        {
            "latent_dim": latent_cols,
            "latent_index": [int(c.replace("z", "")) for c in latent_cols],
            "latent_importance": importance,
            "latent_weight": weights,
            "importance_method": method,
        }
    ).sort_values("latent_weight", ascending=False)

    return out


def evaluate_rf_model(model, X_test: pd.DataFrame, y_test: pd.Series, trait_type: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if len(X_test) == 0:
        return metrics

    if trait_type == "continuous":
        pred = model.predict(X_test)
        metrics["test_r2"] = float(r2_score(y_test, pred))
    else:
        pred_class = model.predict(X_test)
        metrics["test_accuracy"] = float(accuracy_score(y_test, pred_class))

        try:
            pred_prob = model.predict_proba(X_test)[:, 1]
            metrics["test_auc"] = float(roc_auc_score(y_test, pred_prob))
        except Exception:
            metrics["test_auc"] = math.nan

    return metrics


def compute_latent_phenotype_weights(
    latent_df: pd.DataFrame,
    pheno: pd.DataFrame,
    config: Dict[str, Any],
    out_dir: Path,
) -> pd.DataFrame:
    """Fit RF latent-to-phenotype model and compute latent phenotype weights."""
    X, y, merged = prepare_model_xy(latent_df, pheno, config)

    test_size = float(config.get("test_size", 0.2))
    random_state = int(config.get("random_state", 42))
    trait_type = config["trait_type"].lower()

    stratify = y if trait_type == "binary" and y.nunique() == 2 else None

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        merged["SUBJID"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    LOGGER.info("RF training samples: %d", len(X_train))
    LOGGER.info("RF holdout samples: %d", len(X_test))

    model = fit_random_forest(X_train, y_train, config)
    metrics = evaluate_rf_model(model, X_test, y_test, trait_type)

    for k, v in metrics.items():
        LOGGER.info("%s: %.5f", k, v)

    reference = X_test if len(X_test) > 0 else X_train

    latent_weights = compute_latent_importance_with_shap(
        model=model,
        X_reference=reference,
        trait_type=trait_type,
        use_shap=bool(config.get("use_shap_for_latent_weights", True)),
    )

    for key, value in metrics.items():
        latent_weights[key] = value

    pd.DataFrame({"SUBJID": ids_train}).to_csv(out_dir / "discovery_ids.csv", index=False)
    pd.DataFrame({"SUBJID": ids_test}).to_csv(out_dir / "holdout_ids.csv", index=False)

    latent_weights.to_csv(out_dir / "latent_phenotype_weights.csv", index=False)
    return latent_weights


# -----------------------------------------------------------------------------
# VAE-SHAP SNP score aggregation
# -----------------------------------------------------------------------------

def read_latent_snp_shap_file(
    file_path: Path,
    latent_index: int,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Read one latent-specific VAE-SHAP file."""
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()

    snp_col = find_first_existing_column(
        df,
        config.get(
            "snp_col_candidates",
            ["SNP", "snp", "feature", "feature_name", "Feature", "variant", "ID"],
        ),
    )

    shap_col = find_first_existing_column(
        df,
        config.get(
            "shap_col_candidates",
            [
                "Mean_ABS_SHAP",
                "mean_abs_shap",
                "mean_abs_SHAP",
                "Mean_SHAP",
                "mean_shap",
                "Mean(|SHAP|)",
                "importance",
                "score",
            ],
        ),
    )

    if snp_col is None or shap_col is None:
        raise ValueError(
            f"Could not identify SNP/SHAP columns in {file_path}. Columns: {df.columns.tolist()}"
        )

    out = df[[snp_col, shap_col]].copy()
    out = out.rename(columns={snp_col: "snp", shap_col: "latent_snp_shap"})
    out["snp"] = out["snp"].astype(str).str.strip()
    out["snp_rsid"] = out["snp"].map(extract_rsid)
    out["latent_snp_shap"] = safe_numeric(out["latent_snp_shap"]).abs()
    out["latent_index"] = int(latent_index)

    out = out.dropna(subset=["snp", "latent_snp_shap"]).copy()
    out = out[out["latent_snp_shap"] > 0].copy()

    top_k = config.get("top_k_per_latent", None)
    if top_k is not None:
        out = out.sort_values("latent_snp_shap", ascending=False).head(int(top_k)).copy()

    cumulative_threshold = config.get("cumulative_within_latent", None)
    if cumulative_threshold is not None:
        out = out.sort_values("latent_snp_shap", ascending=False).copy()
        total = out["latent_snp_shap"].sum()
        if total > 0:
            out["cum_importance"] = out["latent_snp_shap"].cumsum() / total
            out = out[out["cum_importance"] <= float(cumulative_threshold)].copy()

    total_score = out["latent_snp_shap"].sum()
    if total_score > 0:
        out["within_latent_snp_weight"] = out["latent_snp_shap"] / total_score
    else:
        out["within_latent_snp_weight"] = np.nan

    return out


def aggregate_weighted_snp_scores(
    chrom: int,
    latent_weights: pd.DataFrame,
    config: Dict[str, Any],
    out_dir: Path,
) -> pd.DataFrame:
    """Combine latent phenotype weights with latent-specific VAE-SHAP SNP scores."""
    latent_start = int(config.get("latent_dim_start", 0))
    latent_end = int(config.get("latent_dim_end", latent_weights["latent_index"].max() + 1))

    shap_dir_pattern = config["vae_shap_dir_pattern"]
    shap_file_pattern = config.get("vae_shap_file_pattern", "latent_{latent_index}_top_features_1.csv")

    shap_dir = format_path(shap_dir_pattern, chrom=chrom)

    latent_weight_map = dict(
        zip(latent_weights["latent_index"].astype(int), latent_weights["latent_weight"].astype(float))
    )

    all_rows: List[pd.DataFrame] = []

    for latent_index in range(latent_start, latent_end):
        file_path = shap_dir / shap_file_pattern.format(
            chrom=chrom,
            latent_index=latent_index,
        )

        if not file_path.exists():
            LOGGER.warning("Missing VAE-SHAP file for chr%s latent %s: %s", chrom, latent_index, file_path)
            continue

        latent_df = read_latent_snp_shap_file(file_path, latent_index, config)

        latent_weight = latent_weight_map.get(latent_index, 0.0)
        latent_df["latent_weight"] = latent_weight
        latent_df["weighted_shap"] = latent_df["within_latent_snp_weight"] * latent_weight

        all_rows.append(latent_df)

    if not all_rows:
        raise FileNotFoundError(f"No VAE-SHAP files found for chromosome {chrom}")

    all_latent_snps = pd.concat(all_rows, ignore_index=True)
    all_latent_snps.to_csv(out_dir / "latent_snp_scores_long.csv", index=False)

    agg = (
        all_latent_snps
        .groupby(["snp", "snp_rsid"], dropna=False)
        .agg(
            weighted_shap=("weighted_shap", "sum"),
            max_weighted_shap=("weighted_shap", "max"),
            mean_weighted_shap=("weighted_shap", "mean"),
            n_latents=("latent_index", "nunique"),
            raw_latent_snp_shap_sum=("latent_snp_shap", "sum"),
        )
        .reset_index()
        .sort_values("weighted_shap", ascending=False)
    )

    agg["rank_weighted_shap"] = np.arange(1, len(agg) + 1)
    agg.to_csv(out_dir / "weighted_snp_scores_unfiltered.csv", index=False)

    return agg


# -----------------------------------------------------------------------------
# MAF filtering and LD pruning
# -----------------------------------------------------------------------------

def compute_maf_from_genotypes(geno: pd.DataFrame, snps: List[str]) -> pd.DataFrame:
    """Compute allele frequency and MAF assuming genotype dosage is coded 0/1/2."""
    rows = []

    for snp in snps:
        if snp not in geno.columns:
            rows.append(
                {
                    "snp": snp,
                    "AF": np.nan,
                    "MAF": np.nan,
                    "N_nonmissing": 0,
                }
            )
            continue

        values = safe_numeric(geno[snp]).dropna()
        values = values[values.isin([0, 1, 2])]

        if len(values) == 0:
            af = np.nan
            maf = np.nan
        else:
            af = values.mean() / 2.0
            maf = min(af, 1.0 - af)

        rows.append(
            {
                "snp": snp,
                "AF": af,
                "MAF": maf,
                "N_nonmissing": int(len(values)),
            }
        )

    return pd.DataFrame(rows)


def apply_maf_filter(
    ranked_snps: pd.DataFrame,
    genotype_file: Path,
    config: Dict[str, Any],
    out_dir: Path,
) -> pd.DataFrame:
    maf_threshold = float(config.get("maf_threshold", 0.01))

    snps = ranked_snps["snp"].astype(str).tolist()
    geno = load_genotype_for_snps(genotype_file, snps)

    maf_df = compute_maf_from_genotypes(geno, snps)
    maf_df.to_csv(out_dir / "snp_maf_values.csv", index=False)

    merged = ranked_snps.merge(maf_df, on="snp", how="left")
    filtered = merged[merged["MAF"].notna() & (merged["MAF"] >= maf_threshold)].copy()

    filtered = filtered.sort_values("weighted_shap", ascending=False).reset_index(drop=True)
    filtered["rank_weighted_shap_maf"] = np.arange(1, len(filtered) + 1)

    filtered.to_csv(out_dir / "final_ranked_snps_by_weighted_shap_maf.csv", index=False)

    LOGGER.info("SNPs before MAF filter: %d", len(ranked_snps))
    LOGGER.info("SNPs after MAF filter: %d", len(filtered))

    return filtered


def greedy_ld_prune(
    ranked_snps: pd.DataFrame,
    genotype_file: Path,
    config: Dict[str, Any],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Greedy LD pruning.

    SNPs are processed by descending weighted SHAP score. At each step, the
    highest-ranked remaining SNP is retained and lower-ranked SNPs with
    pairwise r2 above the threshold are removed.
    """
    ld_r2_threshold = float(config.get("ld_r2_threshold", 0.7))
    max_snps_before_ld = config.get("max_snps_before_ld", None)

    work = ranked_snps.sort_values("weighted_shap", ascending=False).copy()

    if max_snps_before_ld is not None and len(work) > int(max_snps_before_ld):
        LOGGER.info(
            "Limiting SNPs before LD pruning from %d to top %d by weighted SHAP.",
            len(work),
            int(max_snps_before_ld),
        )
        work = work.head(int(max_snps_before_ld)).copy()

    snps = work["snp"].astype(str).tolist()

    if len(snps) <= 1:
        work["rank_weighted_shap_maf_ld"] = np.arange(1, len(work) + 1)
        work.to_csv(out_dir / "final_ranked_snps_by_weighted_shap_maf_ld.csv", index=False)
        return work

    geno = load_genotype_for_snps(genotype_file, snps)
    X = geno[snps].copy()

    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X)

    std = np.nanstd(X_imp, axis=0)
    valid = std > 0

    snps_valid = [s for s, ok in zip(snps, valid) if ok]
    X_valid = X_imp[:, valid]

    if len(snps_valid) == 0:
        LOGGER.warning("No SNPs with non-zero variance available for LD pruning.")
        return work.iloc[0:0].copy()

    corr = np.corrcoef(X_valid, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    r2 = corr ** 2

    keep = np.ones(len(snps_valid), dtype=bool)
    retained_indices: List[int] = []
    removed_records: List[Dict[str, Any]] = []

    for i in range(len(snps_valid)):
        if not keep[i]:
            continue

        retained_indices.append(i)

        high_ld = np.where((r2[i, :] >= ld_r2_threshold) & keep)[0]
        high_ld = high_ld[high_ld != i]

        for j in high_ld:
            keep[j] = False
            removed_records.append(
                {
                    "retained_snp": snps_valid[i],
                    "removed_snp": snps_valid[j],
                    "r2": float(r2[i, j]),
                }
            )

    retained_snps = [snps_valid[i] for i in retained_indices]

    pruned = work[work["snp"].isin(retained_snps)].copy()
    pruned = pruned.sort_values("weighted_shap", ascending=False).reset_index(drop=True)
    pruned["rank_weighted_shap_maf_ld"] = np.arange(1, len(pruned) + 1)

    removed_df = pd.DataFrame(removed_records)
    removed_df.to_csv(out_dir / "ld_pruned_removed_snps.csv", index=False)

    pruned.to_csv(out_dir / "final_ranked_snps_by_weighted_shap_maf_ld.csv", index=False)

    LOGGER.info("SNPs before LD pruning: %d", len(work))
    LOGGER.info("SNPs after LD pruning: %d", len(pruned))

    return pruned


# -----------------------------------------------------------------------------
# Chromosome-level pipeline
# -----------------------------------------------------------------------------

def run_chromosome(chrom: int, pheno: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    output_base = Path(config["output_dir"])
    out_dir = output_base / f"chr{chrom}"
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 80)
    LOGGER.info("Running Stage 1 phenotype relevance for chromosome %s", chrom)

    genotype_file = format_path(config["genotype_pattern"], chrom=chrom)
    latent_file = format_path(config["latent_pattern"], chrom=chrom)

    latent_ids_pattern = config.get("latent_ids_pattern", None)
    latent_ids_file = (
        format_path(latent_ids_pattern, chrom=chrom)
        if latent_ids_pattern not in [None, ""]
        else None
    )

    if not genotype_file.exists():
        raise FileNotFoundError(f"Missing genotype file: {genotype_file}")

    if not latent_file.exists():
        raise FileNotFoundError(f"Missing latent file: {latent_file}")

    genotype_ids = load_genotype_ids(genotype_file)

    latent_df = load_latent_matrix(
        latent_file=latent_file,
        genotype_ids=genotype_ids,
        latent_ids_file=latent_ids_file,
    )

    latent_weights = compute_latent_phenotype_weights(
        latent_df=latent_df,
        pheno=pheno,
        config=config,
        out_dir=out_dir,
    )

    ranked_snps = aggregate_weighted_snp_scores(
        chrom=chrom,
        latent_weights=latent_weights,
        config=config,
        out_dir=out_dir,
    )

    maf_filtered = apply_maf_filter(
        ranked_snps=ranked_snps,
        genotype_file=genotype_file,
        config=config,
        out_dir=out_dir,
    )

    ld_pruned = greedy_ld_prune(
        ranked_snps=maf_filtered,
        genotype_file=genotype_file,
        config=config,
        out_dir=out_dir,
    )

    ld_pruned["chrom"] = chrom
    ld_pruned.to_csv(out_dir / "final_ranked_snps_by_weighted_shap_maf_ld.csv", index=False)

    LOGGER.info("Completed chr%s. Final SNP count: %d", chrom, len(ld_pruned))

    return ld_pruned


def run_pipeline(config: Dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "stage1_phenotype_relevance.log")

    pheno = load_phenotype_table(config)

    chromosomes = config.get("chromosomes", list(range(1, 23)))

    all_outputs: List[pd.DataFrame] = []
    failed: List[Dict[str, Any]] = []

    for chrom in chromosomes:
        try:
            result = run_chromosome(int(chrom), pheno, config)
            all_outputs.append(result)
        except Exception as exc:
            LOGGER.exception("Chromosome %s failed: %s", chrom, exc)
            failed.append({"chrom": chrom, "error": str(exc)})

    if all_outputs:
        combined = pd.concat(all_outputs, ignore_index=True)
        combined = combined.sort_values(["chrom", "rank_weighted_shap_maf_ld"]).reset_index(drop=True)
        combined.to_csv(output_dir / "all_chromosomes_final_ranked_snps_by_weighted_shap_maf_ld.csv", index=False)

        LOGGER.info("Saved combined final SNP file.")
        LOGGER.info("Total final SNPs across chromosomes: %d", len(combined))

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_df.to_csv(output_dir / "failed_chromosomes.csv", index=False)
        LOGGER.warning("Some chromosomes failed. See failed_chromosomes.csv.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 phenotype-relevance weighting for VAE-SHAP SNP prioritization."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to JSON configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
