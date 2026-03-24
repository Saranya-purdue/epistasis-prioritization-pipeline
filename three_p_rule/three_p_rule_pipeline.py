from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
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


def extract_rsid(snp_name: str) -> Optional[str]:
    if pd.isna(snp_name):
        return None

    parts = str(snp_name).split(":")
    for part in parts:
        if part.startswith("rs"):
            return part

    return None


def parse_plink_gwas(plink_gwas_file: str | Path) -> pd.DataFrame:
    gwas = pd.read_csv(plink_gwas_file, delim_whitespace=True)

    if "TEST" not in gwas.columns:
        raise ValueError("GWAS file must contain a TEST column.")
    if "ID" not in gwas.columns:
        raise ValueError("GWAS file must contain an ID column.")
    if "BETA" not in gwas.columns or "P" not in gwas.columns:
        raise ValueError("GWAS file must contain BETA and P columns.")

    gwas_add = gwas[gwas["TEST"] == "ADD"].copy()
    gwas_add = gwas_add[["ID", "BETA", "P"]].rename(
        columns={"ID": "rsid", "BETA": "beta_gwas", "P": "p_gwas"}
    )

    pvals = gwas_add["p_gwas"].to_numpy()
    _, qvals, _, reject = multipletests(pvals, alpha=0.05, method="fdr_bh")
    gwas_add["q_gwas"] = qvals
    gwas_add["bh_signif_gwas"] = reject

    return gwas_add


def map_gwas_to_pairs(pairs_df: pd.DataFrame, gwas_df: pd.DataFrame) -> pd.DataFrame:
    out = pairs_df.copy()

    out["rs1"] = out["snp1"].astype(str).map(extract_rsid)
    out["rs2"] = out["snp2"].astype(str).map(extract_rsid)

    gwas1 = gwas_df.rename(columns={
        "rsid": "rs1",
        "beta_gwas": "snp1_beta_gwas",
        "p_gwas": "snp1_p_gwas",
        "q_gwas": "snp1_q_gwas",
        "bh_signif_gwas": "snp1_bh_signif_gwas",
    })
    gwas2 = gwas_df.rename(columns={
        "rsid": "rs2",
        "beta_gwas": "snp2_beta_gwas",
        "p_gwas": "snp2_p_gwas",
        "q_gwas": "snp2_q_gwas",
        "bh_signif_gwas": "snp2_bh_signif_gwas",
    })

    out = out.merge(gwas1, on="rs1", how="left")
    out = out.merge(gwas2, on="rs2", how="left")

    out["min_p_gwas"] = np.nanmin(
        np.vstack([out["snp1_p_gwas"].to_numpy(), out["snp2_p_gwas"].to_numpy()]),
        axis=0,
    )
    out["max_mlog10_p"] = -np.log10(out["min_p_gwas"])

    return out


def fit_interaction_model(
    y_vec: np.ndarray,
    geno_chr_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    snp1: str,
    snp2: str,
) -> tuple[float, float]:
    g1 = geno_chr_df[snp1].to_numpy(dtype=float)
    g2 = geno_chr_df[snp2].to_numpy(dtype=float)
    g_int = g1 * g2

    X = cov_df.copy()
    X["SNP1"] = g1
    X["SNP2"] = g2
    X["SNP1xSNP2"] = g_int
    X = sm.add_constant(X, has_constant="add")

    res = sm.OLS(y_vec, X).fit()
    return float(res.params["SNP1xSNP2"]), float(res.pvalues["SNP1xSNP2"])


def genotype_df_merge(
    geno_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    id_col: str,
    phenotype_col: str,
    covariate_cols: list[str],
) -> pd.DataFrame:
    if id_col not in geno_df.columns:
        geno_df = geno_df.reset_index().rename(columns={"index": id_col})

    merged = geno_df.merge(
        phenotype_df[[id_col, phenotype_col]],
        on=id_col,
        how="inner",
    ).merge(
        covariates_df[[id_col] + covariate_cols],
        on=id_col,
        how="inner",
    )

    merged = merged.dropna(subset=[phenotype_col] + covariate_cols).reset_index(drop=True)
    return merged


def compute_linear_interactions_for_chromosome(
    pairs_df: pd.DataFrame,
    geno_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    id_col: str,
    phenotype_col: str,
    covariate_cols: list[str],
) -> pd.DataFrame:
    merged = genotype_df_merge(geno_df, phenotype_df, covariates_df, id_col, phenotype_col, covariate_cols)

    geno_chr = merged.drop(columns=[phenotype_col] + covariate_cols)
    if id_col in geno_chr.columns:
        geno_chr = geno_chr.set_index(id_col)

    y = merged[phenotype_col]
    cov_df = merged[covariate_cols].copy()

    beta_list = []
    p_list = []
    missing_snps = 0

    for row in tqdm(pairs_df.itertuples(index=False), total=len(pairs_df), desc="Linear interaction fits"):
        snp1 = str(row.snp1)
        snp2 = str(row.snp2)

        if snp1 not in geno_chr.columns or snp2 not in geno_chr.columns:
            beta_list.append(np.nan)
            p_list.append(np.nan)
            missing_snps += 1
            continue

        try:
            beta_int, p_int = fit_interaction_model(
                y_vec=y.to_numpy(dtype=float),
                geno_chr_df=geno_chr,
                cov_df=cov_df,
                snp1=snp1,
                snp2=snp2,
            )
        except Exception:
            beta_int, p_int = np.nan, np.nan

        beta_list.append(beta_int)
        p_list.append(p_int)

    out = pairs_df.copy()
    out["beta_interact_lin"] = beta_list
    out["p_interact_lin"] = p_list
    out["missing_snp_flag"] = out["beta_interact_lin"].isna() & out["p_interact_lin"].isna()

    print(f"Missing SNP pairs for this chromosome: {missing_snps}")
    return out


def apply_bh_to_interaction_pvalues(df: pd.DataFrame, p_col: str = "p_interact_lin") -> pd.DataFrame:
    out = df.copy()

    valid = out[p_col].notna()
    qvals = np.full(len(out), np.nan)
    signif = np.full(len(out), False)

    if valid.sum() > 0:
        _, q_sub, _, rej_sub = multipletests(out.loc[valid, p_col].to_numpy(), alpha=0.05, method="fdr_bh")
        qvals[valid.to_numpy()] = q_sub
        signif[valid.to_numpy()] = rej_sub

    out["FDR_q"] = qvals
    out["Significant"] = signif
    return out


def apply_3p_rule(
    df: pd.DataFrame,
    ml_quantile: float = 0.95,
    gwas_mode: str = "min_p",
    gwas_threshold: float = 0.05,
    interaction_mode: str = "nominal_p",
    interaction_threshold: float = 0.05,
    require_p1: bool = True,
    min_total_support: int = 2,
) -> tuple[pd.DataFrame, float]:
    out = df.copy()

    p1_thr = out["interaction_strength"].quantile(ml_quantile)
    out["P1_high_ML"] = out["interaction_strength"] >= p1_thr

    if gwas_mode == "min_p":
        out["P2_gwas_support"] = out["min_p_gwas"] <= gwas_threshold
    elif gwas_mode == "mlog10":
        out["P2_gwas_support"] = out["max_mlog10_p"] >= gwas_threshold
    else:
        raise ValueError("gwas_mode must be 'min_p' or 'mlog10'.")

    if interaction_mode == "nominal_p":
        out["P3_interact_support"] = out["p_interact_lin"] <= interaction_threshold
    elif interaction_mode == "fdr_q":
        out["P3_interact_support"] = out["FDR_q"] <= interaction_threshold
    else:
        raise ValueError("interaction_mode must be 'nominal_p' or 'fdr_q'.")

    out["n_P_support"] = (
        out["P1_high_ML"].astype(int)
        + out["P2_gwas_support"].astype(int)
        + out["P3_interact_support"].astype(int)
    )

    if require_p1:
        keep_mask = out["P1_high_ML"] & (out["n_P_support"] >= min_total_support)
    else:
        keep_mask = out["n_P_support"] >= min_total_support

    out = out[keep_mask].copy()
    return out, p1_thr


def assign_3p_class(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["class"] = np.select(
        [
            out["P1_high_ML"] & out["P2_gwas_support"] & out["P3_interact_support"],
            out["P1_high_ML"] & out["P2_gwas_support"],
            out["P1_high_ML"] & out["P3_interact_support"],
        ],
        [
            "ML + GWAS + INT",
            "ML + GWAS",
            "ML + INT",
        ],
        default="ML only",
    )
    return out


def plot_3p_class_counts(df: pd.DataFrame, output_path: Path) -> None:
    class_counts = df["class"].value_counts()

    plt.figure(figsize=(5, 4), dpi=300)
    class_counts.plot(kind="bar")
    plt.ylabel("Number of SNP pairs")
    plt.title("3P Rule filtered SNP pairs by support class")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_linear_vs_rf(df: pd.DataFrame, output_path: Path) -> float | None:
    valid = df.dropna(subset=["beta_interact_lin", "interaction_strength"]).copy()
    if valid.empty:
        return None

    valid["abs_beta_interact"] = valid["beta_interact_lin"].abs()
    corr = np.corrcoef(valid["abs_beta_interact"], valid["interaction_strength"])[0, 1]

    plt.figure(figsize=(6, 4), dpi=300)
    plt.scatter(valid["abs_beta_interact"], valid["interaction_strength"], s=4, alpha=0.3)
    plt.xlabel("|beta_interact_lin|")
    plt.ylabel("interaction_strength")
    plt.title(f"Linear vs RF interaction strength (r = {corr:.3f})")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return float(corr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3P-rule pipeline for SNP-pair prioritization.")

    parser.add_argument("--pairs_dir", required=True, help="Directory containing per-chromosome pair CSVs.")
    parser.add_argument("--pairs_pattern", default="chr{chrom}_allinteractions_NOLD.csv")

    parser.add_argument("--genotype_dir", required=True, help="Directory containing per-chromosome genotype CSVs.")
    parser.add_argument("--genotype_pattern", default="chr{chrom}_ready.csv")

    parser.add_argument("--phenotype_csv", required=True)
    parser.add_argument("--covariates_csv", required=True)
    parser.add_argument("--gwas_glm_file", required=True)

    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--chromosomes", nargs="+", type=int, default=list(range(1, 23)))
    parser.add_argument("--id_col", default="SUBJID")
    parser.add_argument("--phenotype_col", default="BMI")
    parser.add_argument("--covariate_cols", nargs="+", default=DEFAULT_COVARIATES)

    parser.add_argument("--gwas_mode", choices=["min_p", "mlog10"], default="min_p")
    parser.add_argument("--gwas_threshold", type=float, default=0.05)

    parser.add_argument("--interaction_mode", choices=["nominal_p", "fdr_q"], default="nominal_p")
    parser.add_argument("--interaction_threshold", type=float, default=0.05)

    parser.add_argument("--ml_quantile", type=float, default=0.95)
    parser.add_argument("--require_p1", action="store_true", default=True)
    parser.add_argument("--min_total_support", type=int, default=2)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    phenotype_df = read_table_with_id(args.phenotype_csv, args.id_col)
    covariates_df = read_table_with_id(args.covariates_csv, args.id_col)

    gwas_df = parse_plink_gwas(args.gwas_glm_file)
    gwas_df.to_csv(outdir / "parsed_gwas_additive.csv", index=False)

    all_pairs = []

    for chrom in args.chromosomes:
        print(f"\nProcessing chromosome {chrom}")

        pairs_file = Path(args.pairs_dir) / args.pairs_pattern.format(chrom=chrom)
        geno_file = Path(args.genotype_dir) / args.genotype_pattern.format(chrom=chrom)

        pairs_df = pd.read_csv(pairs_file)
        geno_df = read_table_with_id(geno_file, args.id_col)

        pairs_lin = compute_linear_interactions_for_chromosome(
            pairs_df=pairs_df,
            geno_df=geno_df,
            phenotype_df=phenotype_df,
            covariates_df=covariates_df,
            id_col=args.id_col,
            phenotype_col=args.phenotype_col,
            covariate_cols=args.covariate_cols,
        )
        pairs_lin["chrom"] = chrom

        pairs_gwas = map_gwas_to_pairs(pairs_lin, gwas_df)
        pairs_gwas.to_csv(outdir / f"chr{chrom}_pairs_with_gwas_and_linear.csv", index=False)

        all_pairs.append(pairs_gwas)

    combined = pd.concat(all_pairs, ignore_index=True)
    combined = apply_bh_to_interaction_pvalues(combined)
    combined.to_csv(outdir / "allchr_pairs_with_gwas_and_linear.csv", index=False)

    corr = plot_linear_vs_rf(
        combined,
        outdir / "linear_vs_rf_interaction_strength.png",
    )

    filtered_before_3p = len(combined)

    df_3p, p1_thr = apply_3p_rule(
        combined,
        ml_quantile=args.ml_quantile,
        gwas_mode=args.gwas_mode,
        gwas_threshold=args.gwas_threshold,
        interaction_mode=args.interaction_mode,
        interaction_threshold=args.interaction_threshold,
        require_p1=args.require_p1,
        min_total_support=args.min_total_support,
    )

    df_3p = assign_3p_class(df_3p)
    df_3p.to_csv(outdir / "allchr_pairs_3P_filtered.csv", index=False)

    plot_3p_class_counts(df_3p, outdir / "three_p_class_counts.png")

    class_counts = df_3p["class"].value_counts(dropna=False).to_dict()

    summary = {
        "n_pairs_total": int(len(combined)),
        "n_pairs_before_3P": int(filtered_before_3p),
        "n_pairs_after_3P": int(len(df_3p)),
        "ml_quantile_threshold": float(args.ml_quantile),
        "p1_interaction_strength_cutoff": float(p1_thr),
        "gwas_mode": args.gwas_mode,
        "gwas_threshold": float(args.gwas_threshold),
        "interaction_mode": args.interaction_mode,
        "interaction_threshold": float(args.interaction_threshold),
        "require_p1": bool(args.require_p1),
        "min_total_support": int(args.min_total_support),
        "linear_vs_rf_corr": corr,
        "class_counts": class_counts,
    }
    save_json(summary, outdir / "three_p_run_summary.json")

    print("\nDone.")
    print(f"Pairs before 3P: {filtered_before_3p}")
    print(f"Pairs after 3P:  {len(df_3p)}")
    print(f"Class counts:    {class_counts}")


if __name__ == "__main__":
    main()
