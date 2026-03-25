from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm


DEFAULT_COVARIATES = [
    "age_at_bmi",
    "sex_M",
    "PC1", "PC2", "PC3", "PC4", "PC5",
    "PC6", "PC7", "PC8", "PC9", "PC10",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_table_with_id(path: str | Path, id_col: str) -> pd.DataFrame:
    """
    Read CSV and ensure id_col exists as a regular column.
    """
    df = pd.read_csv(path)

    if id_col in df.columns:
        return df

    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.rename(columns={unnamed[0]: id_col})
        return df

    raise ValueError(
        f"Could not find ID column '{id_col}' in {path}. "
        f"Columns found: {list(df.columns)[:15]}"
    )


def infer_chrom_from_snp(snp_name: str) -> Optional[str]:
    """
    Parse chromosome from SNP names like:
      chr16:12345:rs123:A:G
      16:12345:rs123
    Returns chromosome string or None.
    """
    m = re.match(r"^(?:chr)?([^:]+):", str(snp_name))
    if not m:
        return None
    return m.group(1)


def prepare_pairs_df(pairs_csv: str | Path) -> pd.DataFrame:
    pairs_df = pd.read_csv(pairs_csv)

    required = {"snp1", "snp2"}
    missing = required - set(pairs_df.columns)
    if missing:
        raise ValueError(f"Pairs file is missing required columns: {missing}")

    if "chrom" not in pairs_df.columns:
        chrom1 = pairs_df["snp1"].astype(str).map(infer_chrom_from_snp)
        chrom2 = pairs_df["snp2"].astype(str).map(infer_chrom_from_snp)

        if chrom1.isna().any() or chrom2.isna().any():
            raise ValueError(
                "Could not infer chromosome from SNP names for some pairs. "
                "Add a 'chrom' column to the pairs file or use parseable SNP names."
            )

        mismatch = chrom1 != chrom2
        if mismatch.any():
            raise ValueError(
                "Found cross-chromosome pairs while inferring chromosome from SNP names. "
                "This script expects same-chromosome pairs."
            )

        pairs_df["chrom"] = chrom1

    return pairs_df


def build_snps_by_chr(pairs_df: pd.DataFrame) -> dict[str, list[str]]:
    snps_by_chr: dict[str, set[str]] = {}

    for row in pairs_df.itertuples(index=False):
        chrom = str(row.chrom)
        snps_by_chr.setdefault(chrom, set()).update([str(row.snp1), str(row.snp2)])

    return {chrom: sorted(list(snps)) for chrom, snps in snps_by_chr.items()}


def load_genotype_subset(genotype_file: str | Path, id_col: str, snps_needed: list[str]) -> pd.DataFrame:
    """
    Load only the requested SNP columns plus ID column from one chromosome genotype file.
    """
    header = pd.read_csv(genotype_file, nrows=0)
    cols = header.columns.tolist()

    if id_col in cols:
        usecols = [id_col] + [s for s in snps_needed if s in cols]
        g = pd.read_csv(genotype_file, usecols=usecols)
        return g

    # Handle case where ID was saved as first unnamed column
    unnamed = [c for c in cols if str(c).startswith("Unnamed:")]
    if unnamed:
        first_col = unnamed[0]
        usecols = [first_col] + [s for s in snps_needed if s in cols]
        g = pd.read_csv(genotype_file, usecols=usecols)
        g = g.rename(columns={first_col: id_col})
        return g

    raise ValueError(
        f"Could not find ID column '{id_col}' in genotype file: {genotype_file}"
    )


def load_all_required_genotypes(
    pairs_df: pd.DataFrame,
    genotype_dir: str | Path,
    genotype_pattern: str,
    id_col: str,
) -> pd.DataFrame:
    """
    Load all required SNPs across chromosomes and merge on ID.
    """
    snps_by_chr = build_snps_by_chr(pairs_df)

    geno_list = []
    for chrom, snp_list in snps_by_chr.items():
        genotype_file = Path(genotype_dir) / genotype_pattern.format(chrom=chrom)
        print(f"Loading chromosome {chrom}: {len(snp_list)} SNPs from {genotype_file}")

        g_sub = load_genotype_subset(genotype_file, id_col=id_col, snps_needed=snp_list)
        geno_list.append(g_sub)

    if not geno_list:
        raise ValueError("No genotype files were loaded.")

    geno_all = geno_list[0].copy()
    for g in geno_list[1:]:
        geno_all = geno_all.merge(g, on=id_col, how="inner")

    return geno_all


def merge_analysis_data(
    geno_all: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    id_col: str,
    phenotype_col: str,
    covariate_cols: list[str],
) -> pd.DataFrame:
    merged = (
        geno_all
        .merge(phenotype_df[[id_col, phenotype_col]], on=id_col, how="inner")
        .merge(covariates_df[[id_col] + covariate_cols], on=id_col, how="inner")
    )

    merged = merged.dropna(subset=[phenotype_col] + covariate_cols).reset_index(drop=True)
    return merged


def compute_pair_delta_r2(
    X_df: pd.DataFrame,
    y: pd.Series,
    cov_df: pd.DataFrame,
    snp1: str,
    snp2: str,
    k_folds: int,
    random_state: int,
) -> dict:
    """
    Compute cross-validated incremental R^2:
      delta_r2 = R2(full) - R2(base)

    base: covariates + SNP1 + SNP2
    full: covariates + SNP1 + SNP2 + SNP1×SNP2
    """
    if snp1 not in X_df.columns or snp2 not in X_df.columns:
        return {
            "snp1": snp1,
            "snp2": snp2,
            "base_r2_mean": np.nan,
            "full_r2_mean": np.nan,
            "delta_r2_mean": np.nan,
            "delta_r2_sd": np.nan,
            "n_valid_folds": 0,
        }

    x1 = X_df[snp1].to_numpy(dtype=float)
    x2 = X_df[snp2].to_numpy(dtype=float)
    x_int = x1 * x2

    cov = cov_df.to_numpy(dtype=float)
    y_vec = y.to_numpy(dtype=float)

    base = np.column_stack([cov, x1, x2])
    full = np.column_stack([cov, x1, x2, x_int])

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    base_r2_list = []
    full_r2_list = []
    delta_list = []

    for train_idx, test_idx in kf.split(base):
        base_train, base_test = base[train_idx], base[test_idx]
        full_train, full_test = full[train_idx], full[test_idx]
        y_train, y_test = y_vec[train_idx], y_vec[test_idx]

        model_base = LinearRegression()
        model_full = LinearRegression()

        model_base.fit(base_train, y_train)
        model_full.fit(full_train, y_train)

        pred_base = model_base.predict(base_test)
        pred_full = model_full.predict(full_test)

        r2_base = r2_score(y_test, pred_base)
        r2_full = r2_score(y_test, pred_full)
        delta = r2_full - r2_base

        base_r2_list.append(r2_base)
        full_r2_list.append(r2_full)
        delta_list.append(delta)

    return {
        "snp1": snp1,
        "snp2": snp2,
        "base_r2_mean": float(np.mean(base_r2_list)),
        "full_r2_mean": float(np.mean(full_r2_list)),
        "delta_r2_mean": float(np.mean(delta_list)),
        "delta_r2_sd": float(np.std(delta_list, ddof=1)) if len(delta_list) > 1 else 0.0,
        "n_valid_folds": len(delta_list),
    }


def cv_incremental_r2_pairs(
    X_df: pd.DataFrame,
    y: pd.Series,
    cov_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    k_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    results = []

    for row in tqdm(pairs_df.itertuples(index=False), total=len(pairs_df), desc="Computing ΔR²"):
        res = compute_pair_delta_r2(
            X_df=X_df,
            y=y,
            cov_df=cov_df,
            snp1=str(row.snp1),
            snp2=str(row.snp2),
            k_folds=k_folds,
            random_state=random_state,
        )

        if "class" in pairs_df.columns:
            res["class"] = getattr(row, "class")
        if "chrom" in pairs_df.columns:
            res["chrom"] = getattr(row, "chrom")

        results.append(res)

    return pd.DataFrame(results)


def summarize_by_class(df_delta: pd.DataFrame) -> pd.DataFrame:
    if "class" not in df_delta.columns:
        return pd.DataFrame()

    summary = (
        df_delta
        .dropna(subset=["delta_r2_mean"])
        .groupby("class", dropna=False)
        .agg(
            n_pairs=("delta_r2_mean", "size"),
            mean_delta_r2=("delta_r2_mean", "mean"),
            median_delta_r2=("delta_r2_mean", "median"),
            sd_delta_r2=("delta_r2_mean", "std"),
            mean_base_r2=("base_r2_mean", "mean"),
            mean_full_r2=("full_r2_mean", "mean"),
        )
        .reset_index()
    )
    return summary


def plot_delta_r2_pairs(df_delta: pd.DataFrame, out_png: str | Path | None = None, dpi: int = 600) -> None:
    dfp = df_delta.sort_values("delta_r2_mean", ascending=True).reset_index(drop=True)

    y = np.arange(len(dfp))
    x = dfp["delta_r2_mean"].to_numpy() * 100.0
    xerr = dfp["delta_r2_sd"].to_numpy() * 100.0

    labels = (dfp["snp1"] + " × " + dfp["snp2"]).tolist()

    plt.figure(figsize=(7, max(3, 0.28 * len(dfp))), dpi=dpi)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.errorbar(x, y, xerr=xerr, fmt="o", capsize=3, markersize=4)
    plt.yticks(y, labels, fontsize=7)
    plt.xlabel("Incremental variance explained by interaction (ΔR², %, 5-fold CV)")
    plt.title("Interaction contribution beyond main effects (per SNP pair)")
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_delta_r2_distribution(df_delta: pd.DataFrame, out_png: str | Path | None = None, dpi: int = 600) -> None:
    vals = df_delta["delta_r2_mean"].dropna().to_numpy() * 100.0

    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.03, size=len(vals))

    plt.figure(figsize=(5, 3), dpi=dpi)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.scatter(vals, jitter, s=30, alpha=0.7)

    if len(vals) > 0:
        med = np.median(vals)
        plt.axvline(med, linewidth=2)

    plt.yticks([])
    plt.xlabel("Incremental variance explained by interaction (ΔR², %, 5-fold CV)")
    plt.title("Distribution of interaction contributions (per-pair mean ΔR²)")
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_delta_r2_by_class(df_delta: pd.DataFrame, out_png: str | Path | None = None, dpi: int = 600) -> None:
    if "class" not in df_delta.columns:
        return

    dfp = df_delta.dropna(subset=["delta_r2_mean"]).copy()
    if dfp.empty:
        return

    class_order = sorted(dfp["class"].dropna().unique().tolist())
    x_positions = {cls: i for i, cls in enumerate(class_order)}

    rng = np.random.default_rng(42)

    plt.figure(figsize=(7, 4), dpi=dpi)
    plt.axhline(0, linestyle="--", linewidth=1)

    for cls in class_order:
        vals = dfp.loc[dfp["class"] == cls, "delta_r2_mean"].to_numpy() * 100.0
        x = np.full(len(vals), x_positions[cls], dtype=float)
        x = x + rng.normal(0, 0.05, size=len(vals))
        plt.scatter(x, vals, s=30, alpha=0.7, label=cls)

        if len(vals) > 0:
            plt.scatter(
                x_positions[cls],
                np.median(vals),
                s=80,
                marker="_",
                linewidths=2,
            )

    plt.xticks(range(len(class_order)), class_order, rotation=20, ha="right")
    plt.ylabel("Incremental variance explained by interaction (ΔR², %)")
    plt.title("Interaction contribution by 3P class")
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validated variance contribution of SNP-SNP interactions."
    )

    parser.add_argument("--pairs_csv", required=True, help="3P-filtered or candidate pair CSV.")
    parser.add_argument("--genotype_dir", required=True, help="Directory containing per-chromosome genotype CSVs.")
    parser.add_argument("--genotype_pattern", default="chr{chrom}_ready.csv")
    parser.add_argument("--phenotype_csv", required=True)
    parser.add_argument("--covariates_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--id_col", default="SUBJID")
    parser.add_argument("--phenotype_col", default="BMI")
    parser.add_argument("--covariate_cols", nargs="+", default=DEFAULT_COVARIATES)

    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    print("Loading pair file...")
    pairs_df = prepare_pairs_df(args.pairs_csv)

    print("Loading phenotype and covariates...")
    phenotype_df = read_table_with_id(args.phenotype_csv, args.id_col)
    covariates_df = read_table_with_id(args.covariates_csv, args.id_col)

    print("Loading required genotype subsets across chromosomes...")
    geno_all = load_all_required_genotypes(
        pairs_df=pairs_df,
        genotype_dir=args.genotype_dir,
        genotype_pattern=args.genotype_pattern,
        id_col=args.id_col,
    )

    print("Merging genotype, phenotype, and covariates...")
    df_merged = merge_analysis_data(
        geno_all=geno_all,
        phenotype_df=phenotype_df,
        covariates_df=covariates_df,
        id_col=args.id_col,
        phenotype_col=args.phenotype_col,
        covariate_cols=args.covariate_cols,
    )

    all_snps = sorted(set(pairs_df["snp1"].astype(str)).union(set(pairs_df["snp2"].astype(str))))
    snp_cols = [c for c in df_merged.columns if c in all_snps]

    X_df = df_merged[snp_cols].copy()
    y = df_merged[args.phenotype_col].copy()
    cov_df = df_merged[args.covariate_cols].copy()

    print(f"Merged data shape: {df_merged.shape}")
    print(f"SNP matrix shape: {X_df.shape}")
    print(f"Number of candidate pairs: {len(pairs_df)}")

    print("Computing cross-validated ΔR² per pair...")
    df_delta = cv_incremental_r2_pairs(
        X_df=X_df,
        y=y,
        cov_df=cov_df,
        pairs_df=pairs_df,
        k_folds=args.k_folds,
        random_state=args.random_seed,
    )

    df_delta.to_csv(outdir / "pairwise_delta_r2.csv", index=False)

    class_summary = summarize_by_class(df_delta)
    if not class_summary.empty:
        class_summary.to_csv(outdir / "delta_r2_by_class_summary.csv", index=False)

    plot_delta_r2_pairs(df_delta, out_png=outdir / "delta_r2_pairs.png")
    plot_delta_r2_distribution(df_delta, out_png=outdir / "delta_r2_distribution.png")
    plot_delta_r2_by_class(df_delta, out_png=outdir / "delta_r2_by_class.png")

    summary = {
        "n_pairs": int(len(df_delta)),
        "n_pairs_non_missing": int(df_delta["delta_r2_mean"].notna().sum()),
        "n_samples_merged": int(len(df_merged)),
        "n_snps_loaded": int(len(snp_cols)),
        "k_folds": int(args.k_folds),
        "mean_delta_r2": float(df_delta["delta_r2_mean"].dropna().mean()) if df_delta["delta_r2_mean"].notna().any() else None,
        "median_delta_r2": float(df_delta["delta_r2_mean"].dropna().median()) if df_delta["delta_r2_mean"].notna().any() else None,
    }
    save_json(summary, outdir / "variance_run_summary.json")

    print("Done.")
    print(f"Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
