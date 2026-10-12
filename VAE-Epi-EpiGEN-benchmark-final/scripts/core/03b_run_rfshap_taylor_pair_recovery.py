#!/usr/bin/env python3

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)


def clean_snp(x):
    return str(x).strip()


def pair_key(a, b):
    a = clean_snp(a)
    b = clean_snp(b)
    return "||".join(sorted([a, b]))


def load_input(input_dir):
    input_dir = Path(input_dir)

    geno_tsv = input_dir / "matrixepistasis_genotype.tsv"
    geno_gz = input_dir / "matrixepistasis_genotype.tsv.gz"

    if geno_tsv.exists():
        geno = pd.read_csv(geno_tsv, sep="\t")
    elif geno_gz.exists():
        geno = pd.read_csv(geno_gz, sep="\t")
    else:
        raise FileNotFoundError("Missing matrixepistasis_genotype.tsv or matrixepistasis_genotype.tsv.gz")

    pheno = pd.read_csv(input_dir / "phenotype_for_tools.tsv", sep="\t")
    truth = pd.read_csv(input_dir / "true_pairs_canonical.tsv", sep="\t")

    geno["sample_id"] = geno["sample_id"].astype(str)
    pheno["sample_id"] = pheno["sample_id"].astype(str)

    df = geno.merge(pheno, on="sample_id", how="inner")

    snp_cols = [c for c in geno.columns if c != "sample_id"]

    X = df[snp_cols].astype(np.float32).fillna(0)
    y = pd.to_numeric(df["phenotype"], errors="coerce").values

    valid = np.isfinite(y)
    X = X.loc[valid].reset_index(drop=True)
    y = y[valid]

    if "pair_key" not in truth.columns:
        truth["pair_key"] = [
            pair_key(a, b)
            for a, b in zip(truth["snp1"], truth["snp2"])
        ]

    truth["snp1"] = truth["snp1"].astype(str)
    truth["snp2"] = truth["snp2"].astype(str)

    true_pairs = set(truth["pair_key"].astype(str))
    true_snps = set(truth["snp1"]) | set(truth["snp2"])

    return X, y, snp_cols, true_pairs, true_snps


def get_shap_array(shap_values):
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    return shap_values


def fit_rf(n_estimators, max_depth, min_samples_leaf, n_jobs, seed):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=seed,
        n_jobs=n_jobs,
    )


def shap_main_snp_selection(model, X_train, snp_cols, max_samples, top_m, seed):
    rng = np.random.default_rng(seed)

    n = X_train.shape[0]
    eval_n = min(max_samples, n)
    idx = rng.choice(n, size=eval_n, replace=False)

    X_eval = X_train.iloc[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)
    shap_values = get_shap_array(shap_values)

    scores = np.mean(np.abs(shap_values), axis=0)

    score_df = pd.DataFrame({
        "ID": snp_cols,
        "shap_main_score": scores,
    }).sort_values("shap_main_score", ascending=False).reset_index(drop=True)

    score_df["shap_rank"] = np.arange(1, len(score_df) + 1)

    selected_snps = score_df.head(top_m)["ID"].tolist()

    return selected_snps, score_df


def make_local_step_matrix(X, feature_index, genotype_min, genotype_max):
    X_new = X.copy()

    midpoint = (genotype_min + genotype_max) / 2.0

    original = X[:, feature_index].copy()

    step = np.where(original <= midpoint, 1.0, -1.0)
    perturbed = np.clip(original + step, genotype_min, genotype_max)

    actual_step = perturbed - original

    X_new[:, feature_index] = perturbed

    return X_new, actual_step


def taylor_pair_scores(model, X_eval, selected_snps, genotype_min, genotype_max, print_every=1000):
    X = X_eval.astype(np.float32)
    n_features = X.shape[1]

    f0 = model.predict(X)

    # Precompute one-SNP perturbation predictions.
    single_pred = {}
    single_step = {}

    print("Precomputing single-SNP perturbations", flush=True)

    for i in range(n_features):
        Xi, hi = make_local_step_matrix(X, i, genotype_min, genotype_max)
        single_pred[i] = model.predict(Xi)
        single_step[i] = hi

        if (i + 1) % 100 == 0:
            print(f"  single perturbations: {i + 1}/{n_features}", flush=True)

    rows = []
    total_pairs = n_features * (n_features - 1) // 2

    print("Scoring Taylor pair interactions", flush=True)
    print("Total selected-SNP pairs:", total_pairs, flush=True)

    counter = 0

    for i, j in combinations(range(n_features), 2):
        Xij = X.copy()

        midpoint = (genotype_min + genotype_max) / 2.0

        xi = X[:, i].copy()
        xj = X[:, j].copy()

        step_i = np.where(xi <= midpoint, 1.0, -1.0)
        step_j = np.where(xj <= midpoint, 1.0, -1.0)

        xi_new = np.clip(xi + step_i, genotype_min, genotype_max)
        xj_new = np.clip(xj + step_j, genotype_min, genotype_max)

        hi = xi_new - xi
        hj = xj_new - xj

        Xij[:, i] = xi_new
        Xij[:, j] = xj_new

        fij = model.predict(Xij)

        second_order = fij - single_pred[i] - single_pred[j] + f0

        denom = np.abs(hi * hj)
        denom[denom == 0] = 1.0

        local_interaction = second_order / denom

        score = float(np.mean(np.abs(local_interaction)))
        signed_score = float(np.mean(local_interaction))

        snp1 = selected_snps[i]
        snp2 = selected_snps[j]

        rows.append({
            "snp1": snp1,
            "snp2": snp2,
            "pair_key": pair_key(snp1, snp2),
            "taylor_interaction_score": score,
            "taylor_signed_mean": signed_score,
        })

        counter += 1

        if counter % print_every == 0:
            print(f"  scored {counter}/{total_pairs} pairs", flush=True)

    pair_df = pd.DataFrame(rows)
    pair_df = pair_df.sort_values("taylor_interaction_score", ascending=False).reset_index(drop=True)

    return pair_df


def pair_auc(pair_df, true_pairs):
    labels = pair_df["pair_key"].isin(true_pairs).astype(int).values
    scores = pair_df["taylor_interaction_score"].values

    out = {
        "pair_auc": np.nan,
        "pair_average_precision": np.nan,
        "n_positive_pairs_in_taylor_space": int(labels.sum()),
        "n_pairs_in_taylor_space": len(labels),
    }

    if len(np.unique(labels)) == 2:
        out["pair_auc"] = roc_auc_score(labels, scores)
        out["pair_average_precision"] = average_precision_score(labels, scores)

    return out


def topk_recovery(pair_df, true_pairs, k_values):
    method_pairs = set(pair_df["pair_key"])
    n_true_total = len(true_pairs)
    n_true_testable = len(true_pairs & method_pairs)

    rows = []

    for k in k_values:
        top = pair_df.head(k)
        recovered = int(top["pair_key"].isin(true_pairs).sum())
        denom = len(top)

        precision = recovered / denom if denom > 0 else np.nan
        recall_all = recovered / n_true_total if n_true_total > 0 else np.nan
        recall_testable = recovered / n_true_testable if n_true_testable > 0 else np.nan

        if precision + recall_all > 0:
            f1 = 2 * precision * recall_all / (precision + recall_all)
        else:
            f1 = 0.0

        rows.append({
            "method": "RF_SHAP_guided_RF_Taylor",
            "K": k,
            "n_true_pairs_total": n_true_total,
            "n_true_pairs_present_in_method_ranking": n_true_testable,
            "n_pairs_ranked_output": len(pair_df),
            "recovered_true_pairs_at_k": recovered,
            "precision_at_k": precision,
            "recall_at_k_all_truth": recall_all,
            "recall_at_k_testable_pairs": recall_testable,
            "f1_at_k_all_truth": f1,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=8)

    parser.add_argument("--top_m_snps", type=int, default=500)
    parser.add_argument("--max_shap_main_samples", type=int, default=1000)
    parser.add_argument("--max_taylor_samples", type=int, default=500)

    parser.add_argument("--genotype_min", type=float, default=0.0)
    parser.add_argument("--genotype_max", type=float, default=2.0)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading input", flush=True)
    X, y, snp_cols, true_pairs, true_snps = load_input(args.input_dir)

    print("X shape:", X.shape, flush=True)
    print("True epistatic SNPs:", len(true_snps), flush=True)
    print("True epistatic pairs:", len(true_pairs), flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
    )

    print("Training stage-1 RF on all SNPs", flush=True)

    rf_all = fit_rf(
        args.n_estimators,
        args.max_depth,
        args.min_samples_leaf,
        args.n_jobs,
        args.seed,
    )

    rf_all.fit(X_train, y_train)

    pred_train_all = rf_all.predict(X_train)
    pred_test_all = rf_all.predict(X_test)

    stage1_train_r2 = r2_score(y_train, pred_train_all)
    stage1_test_r2 = r2_score(y_test, pred_test_all)

    stage1_train_rmse = np.sqrt(mean_squared_error(y_train, pred_train_all))
    stage1_test_rmse = np.sqrt(mean_squared_error(y_test, pred_test_all))

    print("Stage-1 train R2:", stage1_train_r2, flush=True)
    print("Stage-1 test R2:", stage1_test_r2, flush=True)

    print("Running RF-SHAP SNP selection", flush=True)

    selected_snps, snp_score_df = shap_main_snp_selection(
        model=rf_all,
        X_train=X_train,
        snp_cols=snp_cols,
        max_samples=args.max_shap_main_samples,
        top_m=args.top_m_snps,
        seed=args.seed,
    )

    snp_score_df["is_true_epistatic_snp"] = snp_score_df["ID"].isin(true_snps)

    selected_true_snps = set(selected_snps) & true_snps

    snp_score_df.to_csv(
        outdir / "rf_shap_main_snp_scores.tsv",
        sep="\t",
        index=False,
    )

    print("Selected SNPs:", len(selected_snps), flush=True)
    print("True epistatic SNPs selected:", len(selected_true_snps), "/", len(true_snps), flush=True)

    print("Training stage-2 RF on selected SNPs", flush=True)

    X_train_sel = X_train[selected_snps]
    X_test_sel = X_test[selected_snps]

    rf_sel = fit_rf(
        args.n_estimators,
        args.max_depth,
        args.min_samples_leaf,
        args.n_jobs,
        args.seed,
    )

    rf_sel.fit(X_train_sel, y_train)

    pred_train_sel = rf_sel.predict(X_train_sel)
    pred_test_sel = rf_sel.predict(X_test_sel)

    stage2_train_r2 = r2_score(y_train, pred_train_sel)
    stage2_test_r2 = r2_score(y_test, pred_test_sel)

    stage2_train_rmse = np.sqrt(mean_squared_error(y_train, pred_train_sel))
    stage2_test_rmse = np.sqrt(mean_squared_error(y_test, pred_test_sel))

    print("Stage-2 train R2:", stage2_train_r2, flush=True)
    print("Stage-2 test R2:", stage2_test_r2, flush=True)

    rng = np.random.default_rng(args.seed)
    eval_n = min(args.max_taylor_samples, X_test_sel.shape[0])
    eval_idx = rng.choice(X_test_sel.shape[0], size=eval_n, replace=False)

    X_eval = X_test_sel.iloc[eval_idx].to_numpy(dtype=np.float32)

    print("Running Taylor interaction scoring", flush=True)
    print("Taylor samples:", X_eval.shape[0], flush=True)
    print("Taylor SNPs:", X_eval.shape[1], flush=True)

    pair_df = taylor_pair_scores(
        model=rf_sel,
        X_eval=X_eval,
        selected_snps=selected_snps,
        genotype_min=args.genotype_min,
        genotype_max=args.genotype_max,
    )

    pair_df["is_true_pair"] = pair_df["pair_key"].isin(true_pairs)

    pair_df.to_csv(
        outdir / "rf_shap_guided_taylor_pair_scores.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    k_values = [10, 50, 100, 500, 1000, 5000, 10000]

    recovery_df = topk_recovery(pair_df, true_pairs, k_values)

    recovery_df.to_csv(
        outdir / "rf_shap_guided_taylor_pair_recovery_metrics.csv",
        index=False,
    )

    auc_metrics = pair_auc(pair_df, true_pairs)

    method_pairs = set(pair_df["pair_key"])
    true_pairs_testable = true_pairs & method_pairs

    summary = {
        "method": "RF_SHAP_guided_RF_Taylor",
        "n_samples": X.shape[0],
        "n_total_snps": X.shape[1],
        "top_m_snps_for_taylor": args.top_m_snps,
        "n_true_epistatic_snps": len(true_snps),
        "n_true_epistatic_snps_selected": len(selected_true_snps),
        "true_epistatic_snp_selection_rate": (
            len(selected_true_snps) / len(true_snps)
            if len(true_snps) > 0 else np.nan
        ),
        "n_true_pairs_total": len(true_pairs),
        "n_true_pairs_testable_after_rf_shap_selection": len(true_pairs_testable),
        "true_pair_testability_rate_after_rf_shap_selection": (
            len(true_pairs_testable) / len(true_pairs)
            if len(true_pairs) > 0 else np.nan
        ),
        "stage1_train_r2_all_snps": stage1_train_r2,
        "stage1_test_r2_all_snps": stage1_test_r2,
        "stage1_r2_gap": stage1_train_r2 - stage1_test_r2,
        "stage1_train_rmse_all_snps": stage1_train_rmse,
        "stage1_test_rmse_all_snps": stage1_test_rmse,
        "stage2_train_r2_selected_snps": stage2_train_r2,
        "stage2_test_r2_selected_snps": stage2_test_r2,
        "stage2_r2_gap": stage2_train_r2 - stage2_test_r2,
        "stage2_train_rmse_selected_snps": stage2_train_rmse,
        "stage2_test_rmse_selected_snps": stage2_test_rmse,
        **auc_metrics,
    }

    summary_df = pd.DataFrame([summary])

    summary_df.to_csv(
        outdir / "rf_shap_guided_taylor_summary_metrics.csv",
        index=False,
    )

    print("\nSummary:", flush=True)
    print(summary_df.T, flush=True)

    print("\nRecovery:", flush=True)
    print(recovery_df, flush=True)

    print("\nSaved outputs to:", outdir, flush=True)


if __name__ == "__main__":
    main()
