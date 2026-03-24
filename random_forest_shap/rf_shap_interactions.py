from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, train_test_split


DEFAULT_COVARIATES = [
    "age_at_bmi",
    "sex_M",
    "PC1", "PC2", "PC3", "PC4", "PC5",
    "PC6", "PC7", "PC8", "PC9", "PC10",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_table_with_id(path: str | Path, id_col: str) -> pd.DataFrame:
    """
    Read a CSV and ensure id_col exists as a regular column.
    Works whether the ID was stored as a column or as the index.
    """
    df = pd.read_csv(path)

    if id_col in df.columns:
        return df

    # If the first unnamed column is actually the saved index, rename it.
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.rename(columns={unnamed[0]: id_col})
        return df

    # If the first column looks like an ID column, use it as id_col
    if len(df.columns) > 0 and df.columns[0] != id_col:
        # Do not force rename unless caller knows the file structure;
        # safer to raise a clear error.
        raise ValueError(
            f"Could not find ID column '{id_col}' in {path}. "
            f"Columns found: {list(df.columns)[:10]}"
        )

    return df


def load_snp_list(aggregated_snps_csv: str | Path, feature_col: str) -> list[str]:
    df = pd.read_csv(aggregated_snps_csv)
    if feature_col not in df.columns:
        raise ValueError(
            f"Feature column '{feature_col}' not found in aggregated SNP file. "
            f"Columns found: {list(df.columns)}"
        )
    return df[feature_col].astype(str).tolist()


def prepare_merged_data(
    aggregated_snps_csv: str | Path,
    genotype_csv: str | Path,
    phenotype_csv: str | Path,
    covariates_csv: str | Path,
    id_col: str,
    phenotype_col: str,
    covariate_cols: list[str],
    aggregated_feature_col: str = "Feature",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Load and merge genotype, phenotype, and covariates.
    Returns:
        merged_df, X_snp, y, selected_snps_found, selected_snps_missing
    """
    shap_selected_snps = load_snp_list(aggregated_snps_csv, aggregated_feature_col)

    genotype_df = read_table_with_id(genotype_csv, id_col)
    phenotype_df = read_table_with_id(phenotype_csv, id_col)
    covariates_df = read_table_with_id(covariates_csv, id_col)

    genotype_cols = set(genotype_df.columns)
    selected_found = [s for s in shap_selected_snps if s in genotype_cols]
    selected_missing = [s for s in shap_selected_snps if s not in genotype_cols]

    if len(selected_found) == 0:
        raise ValueError("None of the selected SNPs were found in the genotype matrix.")

    if phenotype_col not in phenotype_df.columns:
        raise ValueError(
            f"Phenotype column '{phenotype_col}' not found. "
            f"Columns found: {list(phenotype_df.columns)}"
        )

    missing_covars = [c for c in covariate_cols if c not in covariates_df.columns]
    if missing_covars:
        raise ValueError(
            f"Missing covariate columns: {missing_covars}. "
            f"Available columns include: {list(covariates_df.columns)[:20]}"
        )

    snp_matrix_ready = genotype_df[[id_col] + selected_found].copy()

    merged = snp_matrix_ready.merge(
        phenotype_df[[id_col, phenotype_col]],
        on=id_col,
        how="inner",
    ).merge(
        covariates_df[[id_col] + covariate_cols],
        on=id_col,
        how="inner",
    )

    merged = merged.dropna(subset=[phenotype_col] + covariate_cols).reset_index(drop=True)

    X_snp = merged[selected_found].copy()
    y = merged[phenotype_col].copy()

    return merged, X_snp, y, selected_found, selected_missing


def ld_prune_r2(X: pd.DataFrame, r2_threshold: float = 0.7) -> tuple[pd.DataFrame, list[str]]:
    """
    Greedy LD pruning using pairwise correlation squared (r^2).
    Suitable for a moderate number of SNPs, not very large matrices.
    """
    X_mat = X.to_numpy(dtype=np.float32)
    corr = np.corrcoef(X_mat, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    r2 = corr ** 2

    n_snps = r2.shape[0]
    keep = np.ones(n_snps, dtype=bool)

    for i in range(n_snps):
        if not keep[i]:
            continue
        high_ld = r2[i, (i + 1):] > r2_threshold
        if np.any(high_ld):
            drop_indices = np.where(high_ld)[0] + (i + 1)
            keep[drop_indices] = False

    kept_snps = X.columns[keep].tolist()
    X_pruned = X.loc[:, kept_snps].copy()

    print(f"LD pruning with r^2 <= {r2_threshold}: kept {len(kept_snps)} of {n_snps} SNPs")
    return X_pruned, kept_snps


def residualize_outcome(
    y: pd.Series,
    covariates_df: pd.DataFrame,
) -> tuple[pd.Series, object]:
    C_with_const = sm.add_constant(covariates_df, has_constant="add")
    model_cov = sm.OLS(y, C_with_const).fit()
    y_resid = y - model_cov.fittedvalues
    return y_resid, model_cov


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int,
    search_iter: int,
    cv_folds: int,
    n_jobs: int,
) -> RandomForestRegressor:
    param_dist = {
        "n_estimators": [100, 200, 500],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    }

    rf = RandomForestRegressor(random_state=random_seed)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=search_iter,
        cv=cv_folds,
        scoring="r2",
        random_state=random_seed,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    return search.best_estimator_


def compute_main_effect_importance(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    permutation_repeats: int,
    random_seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_importance = np.abs(shap_values).mean(axis=0)

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=permutation_repeats,
        random_state=random_seed,
        n_jobs=n_jobs,
    )

    importance_df = pd.DataFrame({
        "snp": X_test.columns.astype(str),
        "shap_importance": shap_importance,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

    return importance_df


def plot_top_snp_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    top_df = importance_df.head(top_n).copy()

    plt.figure(figsize=(8, 5), dpi=300)
    plt.barh(top_df["snp"].iloc[::-1], top_df["shap_importance"].iloc[::-1])
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("SNP")
    plt.title(f"Top {top_n} SNPs by SHAP Importance")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def fit_interaction_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int,
    n_jobs: int,
    n_estimators: int = 200,
    max_depth: int | None = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 4,
    max_features: str | float = "log2",
    max_samples: float | None = 0.8,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        random_state=random_seed,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def compute_shap_interaction_table(
    model: RandomForestRegressor,
    X_explain: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(X_explain)

    interaction_matrix = np.abs(shap_interaction_values).mean(axis=0)
    np.fill_diagonal(interaction_matrix, 0.0)

    snp_names = X_explain.columns.astype(str).tolist()
    n_snps = len(snp_names)

    interaction_list = []
    for i in range(n_snps):
        for j in range(i + 1, n_snps):
            main1 = float(np.abs(shap_interaction_values[:, i, i]).mean())
            main2 = float(np.abs(shap_interaction_values[:, j, j]).mean())
            denom = main1 + main2

            interaction_ratio = np.nan
            if denom > 0:
                interaction_ratio = float(interaction_matrix[i, j] / denom)

            interaction_list.append({
                "snp1_idx": i,
                "snp2_idx": j,
                "snp1": snp_names[i],
                "snp2": snp_names[j],
                "interaction_strength": float(interaction_matrix[i, j]),
                "main_effect_snp1": main1,
                "main_effect_snp2": main2,
                "interaction_ratio": interaction_ratio,
            })

    df_interactions = pd.DataFrame(interaction_list).sort_values(
        "interaction_strength",
        ascending=False,
    ).reset_index(drop=True)

    return df_interactions, interaction_matrix


def plot_top_interactions(
    df_interactions: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    top_data = df_interactions.head(top_n).copy()
    top_data["pair"] = top_data["snp1"] + " × " + top_data["snp2"]

    axes[0].barh(range(len(top_data)), top_data["interaction_strength"])
    axes[0].set_yticks(range(len(top_data)))
    axes[0].set_yticklabels(top_data["pair"], fontsize=8)
    axes[0].set_xlabel("SHAP Interaction Strength")
    axes[0].set_title(f"Top {top_n} SNP-SNP Interactions")
    axes[0].invert_yaxis()

    x_main = df_interactions["main_effect_snp1"] + df_interactions["main_effect_snp2"]
    y_int = df_interactions["interaction_strength"]

    axes[1].scatter(x_main, y_int, alpha=0.5, s=10)
    axes[1].set_xlabel("Sum of Main Effects (SHAP)")
    axes[1].set_ylabel("Interaction Strength (SHAP)")
    axes[1].set_title("Interaction vs Main Effects")

    if len(df_interactions) > 0:
        threshold_95 = df_interactions["interaction_strength"].quantile(0.95)
        axes[1].axhline(y=threshold_95, linestyle="--", label="95th percentile")

    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def parse_chr_pos(name: str) -> tuple[str | None, int | None]:
    """
    Try to parse SNP names like:
      16:12345
      chr16:12345
      chr16:12345:A:G
    Returns (chromosome, position) or (None, None).
    """
    m = re.match(r"^(?:chr)?([^:]+):(\d+)", str(name))
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def enforce_min_distance(df: pd.DataFrame, min_bp: int = 0) -> tuple[pd.DataFrame, int]:
    if min_bp <= 0 or df.empty:
        return df.copy(), 0

    chr1, pos1 = zip(*df["snp1"].map(parse_chr_pos))
    chr2, pos2 = zip(*df["snp2"].map(parse_chr_pos))

    chr1 = np.array(chr1, dtype=object)
    chr2 = np.array(chr2, dtype=object)
    pos1 = np.array(pos1, dtype=float)
    pos2 = np.array(pos2, dtype=float)

    # Keep pairs when chromosomes differ, or when positions cannot be parsed,
    # or when they are far enough apart.
    mask = (
        (chr1 != chr2)
        | np.isnan(pos1)
        | np.isnan(pos2)
        | (np.abs(pos1 - pos2) >= min_bp)
    )

    out = df.loc[mask].reset_index(drop=True)
    return out, min_bp


def ld_prune_pairs_by_corr(
    X: pd.DataFrame,
    pairs_df: pd.DataFrame,
    thresh: float = 0.70,
) -> tuple[pd.DataFrame, dict]:
    if pairs_df.empty:
        return pairs_df.copy(), {"kept": 0, "dropped": 0, "thresh": thresh}

    name_to_idx = {s: i for i, s in enumerate(X.columns.astype(str))}
    X_np = X.to_numpy(dtype=np.float32)

    keep_mask = []
    dropped = 0

    for _, r in pairs_df.iterrows():
        i = name_to_idx.get(str(r["snp1"]))
        j = name_to_idx.get(str(r["snp2"]))

        if i is None or j is None:
            keep_mask.append(False)
            dropped += 1
            continue

        corr = np.corrcoef(X_np[:, i], X_np[:, j])[0, 1]
        if np.isfinite(corr) and abs(corr) >= thresh:
            keep_mask.append(False)
            dropped += 1
        else:
            keep_mask.append(True)

    pruned = pairs_df.loc[keep_mask].reset_index(drop=True)
    return pruned, {"kept": len(pruned), "dropped": dropped, "thresh": thresh}


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random Forest + SHAP main-effect and interaction ranking for one chromosome."
    )

    parser.add_argument("--aggregated_snps_csv", required=True)
    parser.add_argument("--genotype_csv", required=True)
    parser.add_argument("--phenotype_csv", required=True)
    parser.add_argument("--covariates_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--chromosome", required=True, help="Chromosome label used in output filenames.")
    parser.add_argument("--id_col", default="SUBJID")
    parser.add_argument("--phenotype_col", default="BMI")
    parser.add_argument("--aggregated_feature_col", default="Feature")

    parser.add_argument(
        "--covariate_cols",
        nargs="+",
        default=DEFAULT_COVARIATES,
        help="Covariate column names.",
    )

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)

    parser.add_argument("--ld_prune_r2", type=float, default=0.7)
    parser.add_argument("--rf_search_iter", type=int, default=20)
    parser.add_argument("--rf_search_cv", type=int, default=5)
    parser.add_argument("--permutation_repeats", type=int, default=50)

    parser.add_argument("--top_k_snps", type=int, default=200)
    parser.add_argument("--top_n_plot", type=int, default=20)

    parser.add_argument("--interaction_n_estimators", type=int, default=200)
    parser.add_argument("--interaction_max_depth", type=int, default=5)
    parser.add_argument("--interaction_min_samples_split", type=int, default=2)
    parser.add_argument("--interaction_min_samples_leaf", type=int, default=4)
    parser.add_argument("--interaction_max_features", default="log2")
    parser.add_argument("--interaction_max_samples", type=float, default=0.8)

    parser.add_argument(
        "--shap_interaction_max_samples",
        type=int,
        default=500,
        help="Maximum number of test samples used for SHAP interaction values.",
    )

    parser.add_argument("--min_distance_bp", type=int, default=0)
    parser.add_argument("--pair_ld_threshold", type=float, default=0.70)
    parser.add_argument("--top_frac", type=float, default=0.01)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    print("Loading and merging data...")
    merged_df, X_snp, y, selected_found, selected_missing = prepare_merged_data(
        aggregated_snps_csv=args.aggregated_snps_csv,
        genotype_csv=args.genotype_csv,
        phenotype_csv=args.phenotype_csv,
        covariates_csv=args.covariates_csv,
        id_col=args.id_col,
        phenotype_col=args.phenotype_col,
        covariate_cols=args.covariate_cols,
        aggregated_feature_col=args.aggregated_feature_col,
    )

    print(f"Merged samples: {merged_df.shape[0]}")
    print(f"Selected SNPs found in genotype matrix: {len(selected_found)}")
    print(f"Selected SNPs missing from genotype matrix: {len(selected_missing)}")

    if selected_missing:
        pd.Series(selected_missing, name="missing_snp").to_csv(
            outdir / f"chr{args.chromosome}_missing_selected_snps.csv",
            index=False,
        )

    covariates_df = merged_df[args.covariate_cols].copy()

    print("Residualizing phenotype on covariates...")
    y_resid, cov_model = residualize_outcome(y, covariates_df)
    save_json(
        {
            "rsquared_covariate_model": float(cov_model.rsquared),
            "n_samples": int(len(y_resid)),
        },
        outdir / f"chr{args.chromosome}_covariate_model_summary.json",
    )

    print("LD pruning selected SNPs...")
    X_pruned, kept_snps = ld_prune_r2(X_snp, r2_threshold=args.ld_prune_r2)
    X_pruned.to_csv(outdir / f"chr{args.chromosome}_prunedX.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pruned,
        y_resid,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    print("Tuning Random Forest for main-effect ranking...")
    rf_main = tune_random_forest(
        X_train=X_train,
        y_train=y_train,
        random_seed=args.random_seed,
        search_iter=args.rf_search_iter,
        cv_folds=args.rf_search_cv,
        n_jobs=args.n_jobs,
    )

    main_train_r2 = rf_main.score(X_train, y_train)
    main_test_r2 = rf_main.score(X_test, y_test)
    print(f"Main-effect RF Train R^2: {main_train_r2:.4f}")
    print(f"Main-effect RF Test R^2:  {main_test_r2:.4f}")

    importance_df = compute_main_effect_importance(
        model=rf_main,
        X_test=X_test,
        y_test=y_test,
        permutation_repeats=args.permutation_repeats,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
    )
    importance_df.to_csv(outdir / f"chr{args.chromosome}_rf_importance_scores.csv", index=False)

    plot_top_snp_importance(
        importance_df=importance_df,
        output_path=outdir / f"chr{args.chromosome}_top_snps_shap.png",
        top_n=args.top_n_plot,
    )

    top_k = min(args.top_k_snps, len(importance_df))
    top_snps = importance_df["snp"].head(top_k).tolist()
    print(f"Selected top {top_k} SNPs based on SHAP importance.")

    X_top = X_pruned.loc[:, top_snps].copy()

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_top,
        y_resid,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    print("Training Random Forest for interaction detection...")
    rf_interaction = fit_interaction_forest(
        X_train=X_train2,
        y_train=y_train2,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
        n_estimators=args.interaction_n_estimators,
        max_depth=args.interaction_max_depth,
        min_samples_split=args.interaction_min_samples_split,
        min_samples_leaf=args.interaction_min_samples_leaf,
        max_features=args.interaction_max_features,
        max_samples=args.interaction_max_samples,
    )

    int_train_r2 = rf_interaction.score(X_train2, y_train2)
    int_test_r2 = rf_interaction.score(X_test2, y_test2)
    print(f"Interaction RF Train R^2: {int_train_r2:.4f}")
    print(f"Interaction RF Test R^2:  {int_test_r2:.4f}")

    n_explain = min(args.shap_interaction_max_samples, len(X_test2))
    X_explain = X_test2.iloc[:n_explain].copy()
    print(f"Computing SHAP interaction values on {n_explain} test samples...")

    df_interactions, _ = compute_shap_interaction_table(
        model=rf_interaction,
        X_explain=X_explain,
    )

    df_interactions.to_csv(outdir / f"chr{args.chromosome}_allinteractions.csv", index=False)

    plot_top_interactions(
        df_interactions=df_interactions,
        output_path=outdir / f"chr{args.chromosome}_top_interactions.png",
        top_n=args.top_n_plot,
    )

    present = df_interactions[
        df_interactions["snp1"].isin(X_test2.columns.astype(str))
        & df_interactions["snp2"].isin(X_test2.columns.astype(str))
    ].reset_index(drop=True)

    present, used_dist = enforce_min_distance(present, args.min_distance_bp)
    print(f"[chr{args.chromosome}] after distance filter ({used_dist} bp): {len(present)} pairs")

    present, ld_info = ld_prune_pairs_by_corr(
        X=X_test2,
        pairs_df=present,
        thresh=args.pair_ld_threshold,
    )
    print(
        f"[chr{args.chromosome}] pair LD prune |r| >= {ld_info['thresh']}: "
        f"kept {ld_info['kept']} (dropped {ld_info['dropped']})"
    )

    present.to_csv(outdir / f"chr{args.chromosome}_allinteractions_filtered.csv", index=False)

    if present.empty:
        empty = present.copy()
        empty["chr"] = str(args.chromosome)
        empty.to_csv(outdir / f"chr{args.chromosome}_candidates_top{args.top_frac:.3f}.csv", index=False)
        print("No pairs left after filtering.")
        return

    present = present.sort_values("interaction_strength", ascending=False).reset_index(drop=True)
    k = max(1, int(np.ceil(args.top_frac * len(present))))
    cands = present.head(k).copy()
    cands["chr"] = str(args.chromosome)

    if "interaction_ratio" in cands.columns:
        ratio = cands["interaction_ratio"].to_numpy(dtype=float)
        ratio = np.where(np.isfinite(ratio), ratio, 1.0)
        cands["screen_score"] = cands["interaction_strength"] * ratio

    cands.to_csv(outdir / f"chr{args.chromosome}_candidates_top{args.top_frac:.3f}.csv", index=False)
    print(f"[chr{args.chromosome}] Saved {len(cands)} candidate pairs.")

    save_json(
        {
            "chromosome": str(args.chromosome),
            "n_samples_merged": int(len(merged_df)),
            "n_selected_snps_found": int(len(selected_found)),
            "n_selected_snps_missing": int(len(selected_missing)),
            "n_ld_pruned_snps": int(len(kept_snps)),
            "main_effect_train_r2": float(main_train_r2),
            "main_effect_test_r2": float(main_test_r2),
            "interaction_train_r2": float(int_train_r2),
            "interaction_test_r2": float(int_test_r2),
            "n_interaction_pairs_before_filters": int(len(df_interactions)),
            "n_interaction_pairs_after_filters": int(len(present)),
            "n_candidates_saved": int(len(cands)),
        },
        outdir / f"chr{args.chromosome}_run_summary.json",
    )

    print(f"Done. Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
