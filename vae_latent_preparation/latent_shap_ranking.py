from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_n_jobs(default_cap: int = 16) -> int:
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    return max(1, min(n, default_cap))


def _compute_shap_for_latent(
    latent_index: int,
    X: np.ndarray,
    latent: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    top_k: int,
    rf_n_estimators: int,
    rf_max_depth: int | None,
    seed: int,
) -> Dict:
    print(f"[LATENT {latent_index}] fitting RF & SHAP...")

    rf = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        n_jobs=1,
        random_state=seed,
    )
    rf.fit(X, latent[:, latent_index])

    explainer = shap.Explainer(rf, X, feature_names=feature_names)
    shap_values = explainer(X)
    mean_abs = np.abs(shap_values.values).mean(axis=0)

    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    top_features = [(feature_names[j], float(mean_abs[j])) for j in top_idx]

    df = pd.DataFrame(top_features, columns=["Feature", "Mean_SHAP"])
    df.to_csv(output_dir / f"latent_{latent_index}_top_features.csv", index=False)

    plt.figure(figsize=(14, 8))
    plt.barh(df["Feature"].iloc[::-1], df["Mean_SHAP"].iloc[::-1])
    plt.xlabel("Mean |SHAP|")
    plt.title(f"Top {top_k} Features - Latent Dimension {latent_index}")
    plt.tight_layout()
    plt.savefig(output_dir / f"latent_{latent_index}_shap_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {"latent_dim": latent_index, "top_features": top_features}


def shap_feature_importance_all_latents(
    X: np.ndarray,
    latent: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    n_jobs: int | None,
    top_k: int,
    rf_n_estimators: int,
    rf_max_depth: int | None,
    seed: int,
    prefer_threads: bool,
) -> List[Dict]:
    ensure_dir(output_dir)

    if n_jobs is None:
        n_jobs = detect_n_jobs()

    prefer = "threads" if prefer_threads else "processes"

    results = Parallel(n_jobs=n_jobs, prefer=prefer, verbose=10)(
        delayed(_compute_shap_for_latent)(
            latent_index=i,
            X=X,
            latent=latent,
            feature_names=feature_names,
            output_dir=output_dir,
            top_k=top_k,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            seed=seed,
        )
        for i in range(latent.shape[1])
    )
    return results


def aggregate_shap_snp_csvs(shap_csv_dir: Path, output_csv: str = "aggregated_top_snps.csv") -> pd.DataFrame:
    all_snps: Dict[str, float] = {}

    for file in shap_csv_dir.iterdir():
        if file.name.startswith("latent_") and file.name.endswith("_top_features.csv"):
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                snp = str(row["Feature"])
                shap_val = float(row["Mean_SHAP"])
                if snp not in all_snps or shap_val > all_snps[snp]:
                    all_snps[snp] = shap_val

    agg_df = pd.DataFrame(list(all_snps.items()), columns=["Feature", "Max_Mean_SHAP"])
    agg_df = agg_df.sort_values(by="Max_Mean_SHAP", ascending=False)
    agg_df.to_csv(shap_csv_dir / output_csv, index=False)
    print(f"Aggregated {len(agg_df)} unique SNPs saved to: {shap_csv_dir / output_csv}")
    return agg_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank original features for each latent dimension using RF + SHAP.")
    parser.add_argument("--input_csv", required=True, help="Original input CSV used for VAE training.")
    parser.add_argument("--latent_csv", required=True, help="Latent CSV produced by vae_latent_preparation.py.")
    parser.add_argument("--output_dir", required=True, help="Directory for SHAP outputs.")
    parser.add_argument("--input_index_col", type=int, default=0, help="Index column for original input CSV.")
    parser.add_argument("--latent_index_col", type=int, default=0, help="Index column for latent CSV.")
    parser.add_argument("--normalize_input", action="store_true", help="Apply MinMax scaling to input features before RF.")
    parser.add_argument("--top_k", type=int, default=100, help="Top features saved per latent dimension.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Parallel jobs across latent dimensions.")
    parser.add_argument("--rf_n_estimators", type=int, default=100)
    parser.add_argument("--rf_max_depth", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--prefer_threads", action="store_true", help="Use joblib threads instead of processes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    latent_csv = Path(args.latent_csv)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print(f"Loading original input data from: {input_csv}")
    X_df = pd.read_csv(input_csv, index_col=args.input_index_col)

    print(f"Loading latent data from: {latent_csv}")
    latent_df = pd.read_csv(latent_csv, index_col=args.latent_index_col)

    common_index = X_df.index.intersection(latent_df.index)
    X_df = X_df.loc[common_index]
    latent_df = latent_df.loc[common_index]

    X = X_df.to_numpy(dtype=float)
    if args.normalize_input:
        X = MinMaxScaler().fit_transform(X)

    latent = latent_df.to_numpy(dtype=float)
    feature_names = X_df.columns.astype(str).tolist()

    shap_feature_importance_all_latents(
        X=X,
        latent=latent,
        feature_names=feature_names,
        output_dir=output_dir,
        n_jobs=args.n_jobs,
        top_k=args.top_k,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        seed=args.random_seed,
        prefer_threads=args.prefer_threads,
    )

    aggregate_shap_snp_csvs(output_dir)
    print(f"Done. SHAP outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
