#!/usr/bin/env python3
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split


def clean_snp(x):
    return str(x).strip()


def pair_key(a, b):
    a = clean_snp(a)
    b = clean_snp(b)
    return "||".join(sorted([a, b]))


def load_data(input_dir):
    input_dir = Path(input_dir)

    geno_file_gz = input_dir / "matrixepistasis_genotype.tsv.gz"
    geno_file = input_dir / "matrixepistasis_genotype.tsv"

    if geno_file.exists():
        geno = pd.read_csv(geno_file, sep="\t")
    else:
        geno = pd.read_csv(geno_file_gz, sep="\t")

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

    true_pairs["snp1"] = true_pairs["snp1"].astype(str)
    true_pairs["snp2"] = true_pairs["snp2"].astype(str)

    true_pair_keys = set(
        pair_key(a, b)
        for a, b in zip(true_pairs["snp1"], true_pairs["snp2"])
    )

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


def collect_tree_path_pair_scores(model, n_features):
    """
    SNPInterForest-style path co-occurrence score.

    If two SNPs repeatedly occur on the same decision paths across trees,
    they receive a higher pair score.
    """
    scores = defaultdict(float)

    for tree_idx, est in enumerate(model.estimators_):
        tree = est.tree_

        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        weighted_samples = tree.weighted_n_node_samples

        stack = [(0, [])]

        while stack:
            node, path = stack.pop()

            left = children_left[node]
            right = children_right[node]

            if left == right:
                feats = sorted(set([f for f in path if f >= 0]))

                if len(feats) >= 2:
                    w = float(weighted_samples[node])

                    for i in range(len(feats)):
                        a = feats[i]
                        for j in range(i + 1, len(feats)):
                            b = feats[j]
                            key = a * n_features + b
                            scores[key] += w

            else:
                f = feature[node]
                new_path = path + [f] if f >= 0 else path

                stack.append((left, new_path))
                stack.append((right, new_path))

        if (tree_idx + 1) % 25 == 0:
            print(f"Processed {tree_idx + 1} trees")

    return scores


def score_to_pairs(pair_scores, snp_cols, top_n=None):
    n_features = len(snp_cols)

    rows = []

    items = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None:
        items = items[:top_n]

    for key, score in items:
        i = key // n_features
        j = key % n_features

        if i >= j:
            continue

        snp1 = snp_cols[i]
        snp2 = snp_cols[j]

        rows.append({
            "snp1": snp1,
            "snp2": snp2,
            "pair_key": pair_key(snp1, snp2),
            "score": score,
        })

    return pd.DataFrame(rows)


def topk_metrics(pair_df, true_pair_keys, k_values):
    rows = []

    n_true = len(true_pair_keys)
    n_ranked = len(pair_df)

    for k in k_values:
        top = pair_df.head(k)
        recovered = int(top["pair_key"].isin(true_pair_keys).sum())

        precision = recovered / k if k > 0 else np.nan
        recall = recovered / n_true if n_true > 0 else np.nan

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        rows.append({
            "K": k,
            "n_true_testable_pairs": n_true,
            "n_pairs_ranked": n_ranked,
            "recovered_true_pairs": recovered,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "f1_at_k": f1,
        })

    return pd.DataFrame(rows)


def snp_level_auc(feature_importance, snp_cols, true_epi_snps):
    labels = np.array([1 if s in true_epi_snps else 0 for s in snp_cols])
    scores = np.asarray(feature_importance)

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


def pair_level_auc(pair_scores, snp_cols, true_pair_keys):
    """
    Computes pair-level ROC-AUC and PR-AUC over all possible SNP pairs.
    For 2,000 SNPs this is ~2 million pairs, which is okay.
    """
    n_features = len(snp_cols)

    labels = []
    scores = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            key_numeric = i * n_features + j
            score = pair_scores.get(key_numeric, 0.0)

            pk = pair_key(snp_cols[i], snp_cols[j])
            label = 1 if pk in true_pair_keys else 0

            labels.append(label)
            scores.append(score)

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    out = {
        "pair_auc": np.nan,
        "pair_average_precision": np.nan,
        "n_all_possible_pairs": len(labels),
        "n_true_pairs": int(labels.sum()),
    }

    if len(np.unique(labels)) == 2:
        out["pair_auc"] = roc_auc_score(labels, scores)
        out["pair_average_precision"] = average_precision_score(labels, scores)

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model", choices=["rf", "extratrees"], required=True)

    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--top_pairs_to_save", type=int, default=100000)
    parser.add_argument("--compute_pair_auc", action="store_true")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data")
    X, y, snp_cols, true_pair_keys, true_epi_snps = load_data(args.input_dir)

    print("X shape:", X.shape)
    print("True epistatic SNPs:", len(true_epi_snps))
    print("True epistatic pairs:", len(true_pair_keys))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
    )

    print("Training model:", args.model)

    model = make_model(
        model_name=args.model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, pred_train)
    test_r2 = r2_score(y_test, pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))

    print("Train R2:", train_r2)
    print("Test R2:", test_r2)
    print("Train RMSE:", train_rmse)
    print("Test RMSE:", test_rmse)

    # SNP-level importance
    feature_importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "ID": snp_cols,
        "Importance": feature_importance,
        "is_true_epistatic_snp": [s in true_epi_snps for s in snp_cols],
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(
        outdir / f"{args.model}_snp_importance.tsv",
        sep="\t",
        index=False,
    )

    snp_auc_metrics = snp_level_auc(
        feature_importance=feature_importance,
        snp_cols=snp_cols,
        true_epi_snps=true_epi_snps,
    )

    # Pair-level path co-occurrence score
    print("Collecting tree-path pair scores")
    pair_scores = collect_tree_path_pair_scores(model, n_features=len(snp_cols))

    print("Number of nonzero pair scores:", len(pair_scores))

    pair_df = score_to_pairs(
        pair_scores=pair_scores,
        snp_cols=snp_cols,
        top_n=args.top_pairs_to_save,
    )

    pair_df.to_csv(
        outdir / f"{args.model}_path_pair_scores_top{args.top_pairs_to_save}.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    k_values = [10, 50, 100, 500, 1000, 5000, 10000]

    recovery_df = topk_metrics(
        pair_df=pair_df,
        true_pair_keys=true_pair_keys,
        k_values=k_values,
    )

    recovery_df["method"] = args.model + "_path_cooccurrence"

    pair_auc_metrics = {
        "pair_auc": np.nan,
        "pair_average_precision": np.nan,
        "n_all_possible_pairs": len(snp_cols) * (len(snp_cols) - 1) // 2,
        "n_true_pairs": len(true_pair_keys),
    }

    if args.compute_pair_auc:
        print("Computing pair-level AUC over all possible SNP pairs")
        pair_auc_metrics = pair_level_auc(
            pair_scores=pair_scores,
            snp_cols=snp_cols,
            true_pair_keys=true_pair_keys,
        )

    summary = {
        "model": args.model,
        "n_samples": X.shape[0],
        "n_snps": X.shape[1],
        "train_r2": train_r2,
        "test_r2": test_r2,
        "r2_gap": train_r2 - test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        **snp_auc_metrics,
        **pair_auc_metrics,
    }

    summary_df = pd.DataFrame([summary])

    summary_df.to_csv(
        outdir / f"{args.model}_summary_metrics.csv",
        index=False,
    )

    recovery_df.to_csv(
        outdir / f"{args.model}_pair_recovery_metrics.csv",
        index=False,
    )

    print("Summary metrics:")
    print(summary_df.T)

    print("Pair recovery:")
    print(recovery_df)

    print("Saved outputs to:", outdir)


if __name__ == "__main__":
    main()