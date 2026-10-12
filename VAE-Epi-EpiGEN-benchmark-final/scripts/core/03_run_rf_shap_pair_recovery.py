#!/usr/bin/env python3

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)

import shap


def clean_snp(x):
    return str(x).strip()


def pair_key(a, b):
    a = clean_snp(a)
    b = clean_snp(b)
    return "||".join(sorted([a, b]))


def load_benchmark_input(input_dir):
    input_dir = Path(input_dir)

    geno_tsv = input_dir / "matrixepistasis_genotype.tsv"
    geno_gz = input_dir / "matrixepistasis_genotype.tsv.gz"

    if geno_tsv.exists():
        geno = pd.read_csv(geno_tsv, sep="\t")
    elif geno_gz.exists():
        geno = pd.read_csv(geno_gz, sep="\t")
    else:
        raise FileNotFoundError("Missing matrixepistasis_genotype.tsv or .tsv.gz")

    pheno = pd.read_csv(input_dir / "phenotype_for_tools.tsv", sep="\t")
    true_pairs = pd.read_csv(input_dir / "true_pairs_canonical.tsv", sep="\t")

    geno["sample_id"] = geno["sample_id"].astype(str)
    pheno["sample_id"] = pheno["sample_id"].astype(str)

    df = geno.merge(pheno, on="sample_id", how="inner")

    snp_cols = [c for c in geno.columns if c != "sample_id"]

    X = df[snp_cols].astype(np.float32)
    y = pd.to_numeric(df["phenotype"], errors="coerce").values

    valid = np.isfinite(y)
    X = X.loc[valid].reset_index(drop=True)
    y = y[valid]

    X = X.fillna(0)

    if "pair_key" not in true_pairs.columns:
        true_pairs["pair_key"] = [
            pair_key(a, b)
            for a, b in zip(true_pairs["snp1"], true_pairs["snp2"])
        ]

    true_pairs["snp1"] = true_pairs["snp1"].astype(str)
    true_pairs["snp2"] = true_pairs["snp2"].astype(str)

    true_pair_keys = set(true_pairs["pair_key"].astype(str))
    true_epi_snps = set(true_pairs["snp1"]) | set(true_pairs["snp2"])

    return X, y, snp_cols, true_pair_keys, true_epi_snps


def make_model(model_name, n_estimators, max_depth, min_samples_leaf, n_jobs, seed):
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            random_state=seed,
            n_jobs=n_jobs,
        )

    if model_name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            random_state=seed,
            n_jobs=n_jobs,
        )

    raise ValueError(f"Unknown model: {model_name}")


def get_shap_array(shap_values):
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    return shap_values


def get_shap_interaction_array(interaction_values):
    if isinstance(interaction_values, list):
        interaction_values = interaction_values[0]

    interaction_values = np.asarray(interaction_values)

    if interaction_values.ndim == 4:
        interaction_values = interaction_values[:, :, :, 0]

    return interaction_values


def compute_top_snp_shap_scores(model, X_train, snp_cols, max_shap_samples, seed):
    rng = np.random.default_rng(seed)

    n = X_train.shape[0]
    eval_n = min(max_shap_samples, n)

    idx = rng.choice(n, size=eval_n, replace=False)
    X_eval = X_train.iloc[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)

    shap_values = get_shap_array(shap_values)

    shap_main_score = np.mean(np.abs(shap_values), axis=0)

    snp_score_df = pd.DataFrame({
        "ID": snp_cols,
        "shap_main_score": shap_main_score,
    }).sort_values("shap_main_score", ascending=False)

    return snp_score_df


def compute_pair_shap_interactions(model, X_eval, selected_snps):
    explainer = shap.TreeExplainer(model)

    interaction_values = explainer.shap_interaction_values(X_eval)
    interaction_values = get_shap_interaction_array(interaction_values)

    pair_score_mat = np.mean(np.abs(interaction_values), axis=0)

    rows = []

    n_features = len(selected_snps)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            snp1 = selected_snps[i]
            snp2 = selected_snps[j]

            rows.append({
                "snp1": snp1,
                "snp2": snp2,
                "pair_key": pair_key(snp1, snp2),
                "rf_shap_interaction_score": pair_score_mat[i, j],
            })

    pair_df = pd.DataFrame(rows)
    pair_df = pair_df.sort_values(
        "rf_shap_interaction_score",
        ascending=False
    ).reset_index(drop=True)

    return pair_df


def evaluate_pair_recovery(pair_df, true_pair_keys, all_true_pair_keys, k_values):
    rows = []

    tested_pair_keys = set(pair_df["pair_key"])
    testable_true_pairs = true_pair_keys & tested_pair_keys

    n_true_total = len(all_true_pair_keys)
    n_true_testable = len(testable_true_pairs)
    n_ranked = len(pair_df)

    for k in k_values:
        top = pair_df.head(k)
        denom = len(top)

        recovered = int(top["pair_key"].isin(all_true_pair_keys).sum())

        precision = recovered / denom if denom > 0 else np.nan
        recall_all = recovered / n_true_total if n_true_total > 0 else np.nan
        recall_testable = recovered / n_true_testable if n_true_testable > 0 else np.nan

        if precision + recall_all > 0:
            f1_all = 2 * precision * recall_all / (precision + recall_all)
        else:
            f1_all = 0.0

        rows.append({
            "K": k,
            "n_pairs_ranked": n_ranked,
            "n_true_pairs_total": n_true_total,
            "n_true_pairs_testable_after_top_snp_selection": n_true_testable,
            "recovered_true_pairs": recovered,
            "precision_at_k": precision,
            "recall_at_k_all_truth": recall_all,
            "recall_at_k_testable_truth": recall_testable,
            "f1_at_k_all_truth": f1_all,
        })

    return pd.DataFrame(rows)


def compute_pair_auc(pair_df, true_pair_keys):
    labels = pair_df["pair_key"].isin(true_pair_keys).astype(int).values
    scores = pair_df["rf_shap_interaction_score"].values

    out = {
        "pair_auc": np.nan,
        "pair_average_precision": np.nan,
        "n_positive_pairs_in_tested_space": int(labels.sum()),
        "n_pairs_in_tested_space": len(labels),
    }

    if len(np.unique(labels)) == 2:
        out["pair_auc"] = roc_auc_score(labels, scores)
        out["pair_average_precision"] = average_precision_score(labels, scores)

    return out


def compute_snp_auc(snp_score_df, true_epi_snps):
    labels = snp_score_df["ID"].isin(true_epi_snps).astype(int).values
    scores = snp_score_df["shap_main_score"].values

    out = {
        "snp_auc": np.nan,
        "snp_average_precision": np.nan,
        "n_true_epistatic_snps": int(labels.sum()),
        "n_total_snps": len(labels),
    }

    if len(np.unique(labels)) == 2:
        out["snp_auc"] = roc_auc_score(labels, scores)
        out["snp_average_precision"] = average_precision_score(labels, scores)

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--model", choices=["rf", "extratrees"], default="rf")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=8)

    parser.add_argument("--top_m_snps", type=int, default=500)
    parser.add_argument("--max_shap_main_samples", type=int, default=1000)
    parser.add_argument("--max_shap_interaction_samples", type=int, default=500)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark input")
    X, y, snp_cols, true_pair_keys, true_epi_snps = load_benchmark_input(args.input_dir)

    print("X shape:", X.shape)
    print("True epistatic SNPs:", len(true_epi_snps))
    print("True epistatic pairs:", len(true_pair_keys))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
    )

    # ------------------------------------------------------------
    # Stage 1: RF on all 2,000 SNPs
    # ------------------------------------------------------------
    print("Training stage-1 model on all SNPs:", args.model)

    model_all = make_model(
        model_name=args.model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

    model_all.fit(X_train, y_train)

    pred_train_all = model_all.predict(X_train)
    pred_test_all = model_all.predict(X_test)

    train_r2_all = r2_score(y_train, pred_train_all)
    test_r2_all = r2_score(y_test, pred_test_all)

    train_rmse_all = np.sqrt(mean_squared_error(y_train, pred_train_all))
    test_rmse_all = np.sqrt(mean_squared_error(y_test, pred_test_all))

    print("Stage-1 train R2:", train_r2_all)
    print("Stage-1 test R2:", test_r2_all)

    # ------------------------------------------------------------
    # Training-only SHAP main-effect SNP selection
    # ------------------------------------------------------------
    print("Computing SHAP main effects for SNP selection")

    snp_score_df = compute_top_snp_shap_scores(
        model=model_all,
        X_train=X_train,
        snp_cols=snp_cols,
        max_shap_samples=args.max_shap_main_samples,
        seed=args.seed,
    )

    snp_score_df["is_true_epistatic_snp"] = snp_score_df["ID"].isin(true_epi_snps)

    selected_snps = snp_score_df.head(args.top_m_snps)["ID"].tolist()

    selected_true_epi_snps = set(selected_snps) & true_epi_snps

    print("Selected SNPs:", len(selected_snps))
    print("True epistatic SNPs selected:", len(selected_true_epi_snps), "/", len(true_epi_snps))

    snp_score_df.to_csv(
        outdir / f"{args.model}_shap_main_snp_scores.tsv",
        sep="\t",
        index=False,
    )

    snp_auc_metrics = compute_snp_auc(snp_score_df, true_epi_snps)

    # ------------------------------------------------------------
    # Stage 2: Refit RF on selected SNPs only
    # ------------------------------------------------------------
    print("Training stage-2 model on selected SNPs")

    X_train_sel = X_train[selected_snps]
    X_test_sel = X_test[selected_snps]

    model_sel = make_model(
        model_name=args.model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

    model_sel.fit(X_train_sel, y_train)

    pred_train_sel = model_sel.predict(X_train_sel)
    pred_test_sel = model_sel.predict(X_test_sel)

    train_r2_sel = r2_score(y_train, pred_train_sel)
    test_r2_sel = r2_score(y_test, pred_test_sel)

    train_rmse_sel = np.sqrt(mean_squared_error(y_train, pred_train_sel))
    test_rmse_sel = np.sqrt(mean_squared_error(y_test, pred_test_sel))

    print("Stage-2 train R2:", train_r2_sel)
    print("Stage-2 test R2:", test_r2_sel)

    # ------------------------------------------------------------
    # SHAP interaction values on held-out test samples
    # ------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    eval_n = min(args.max_shap_interaction_samples, X_test_sel.shape[0])
    eval_idx = rng.choice(X_test_sel.shape[0], size=eval_n, replace=False)

    X_eval_interaction = X_test_sel.iloc[eval_idx]

    print("Computing SHAP interaction values")
    print("Interaction samples:", X_eval_interaction.shape[0])
    print("Interaction SNPs:", X_eval_interaction.shape[1])

    pair_df = compute_pair_shap_interactions(
        model=model_sel,
        X_eval=X_eval_interaction,
        selected_snps=selected_snps,
    )

    pair_df["is_true_pair"] = pair_df["pair_key"].isin(true_pair_keys)

    pair_df.to_csv(
        outdir / f"{args.model}_rfshap_pair_scores.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    # ------------------------------------------------------------
    # Pair recovery metrics
    # ------------------------------------------------------------
    k_values = [10, 50, 100, 500, 1000, 5000, 10000]

    recovery_df = evaluate_pair_recovery(
        pair_df=pair_df,
        true_pair_keys=true_pair_keys,
        all_true_pair_keys=true_pair_keys,
        k_values=k_values,
    )

    recovery_df["method"] = f"{args.model}_RF_SHAP_interaction"

    recovery_df.to_csv(
        outdir / f"{args.model}_rfshap_pair_recovery_metrics.csv",
        index=False,
    )

    pair_auc_metrics = compute_pair_auc(pair_df, true_pair_keys)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    tested_pair_keys = set(pair_df["pair_key"])
    n_true_testable_after_selection = len(true_pair_keys & tested_pair_keys)

    summary = {
        "method": f"{args.model}_RF_SHAP_interaction",
        "n_samples": X.shape[0],
        "n_total_snps": X.shape[1],
        "top_m_snps_for_interaction": args.top_m_snps,
        "n_true_epistatic_snps": len(true_epi_snps),
        "n_true_epistatic_snps_selected": len(selected_true_epi_snps),
        "true_epistatic_snp_selection_rate": (
            len(selected_true_epi_snps) / len(true_epi_snps)
            if len(true_epi_snps) > 0 else np.nan
        ),
        "n_true_pairs_total": len(true_pair_keys),
        "n_true_pairs_testable_after_top_snp_selection": n_true_testable_after_selection,
        "true_pair_testability_rate_after_top_snp_selection": (
            n_true_testable_after_selection / len(true_pair_keys)
            if len(true_pair_keys) > 0 else np.nan
        ),
        "stage1_train_r2_all_snps": train_r2_all,
        "stage1_test_r2_all_snps": test_r2_all,
        "stage1_r2_gap": train_r2_all - test_r2_all,
        "stage1_train_rmse_all_snps": train_rmse_all,
        "stage1_test_rmse_all_snps": test_rmse_all,
        "stage2_train_r2_selected_snps": train_r2_sel,
        "stage2_test_r2_selected_snps": test_r2_sel,
        "stage2_r2_gap": train_r2_sel - test_r2_sel,
        "stage2_train_rmse_selected_snps": train_rmse_sel,
        "stage2_test_rmse_selected_snps": test_rmse_sel,
        **snp_auc_metrics,
        **pair_auc_metrics,
    }

    summary_df = pd.DataFrame([summary])

    summary_df.to_csv(
        outdir / f"{args.model}_rfshap_summary_metrics.csv",
        index=False,
    )

    print("\nSummary:")
    print(summary_df.T)

    print("\nPair recovery:")
    print(recovery_df)

    print("\nSaved outputs to:", outdir)


if __name__ == "__main__":
    main()