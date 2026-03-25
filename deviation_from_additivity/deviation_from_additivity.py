from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec


GENOS = [0, 1, 2]


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


def extract_rsid(snp_str: str) -> str:
    if pd.isna(snp_str):
        return str(snp_str)

    parts = str(snp_str).split(":")
    for p in parts:
        if p.startswith("rs"):
            return p
    return str(snp_str)


def extract_alleles(snp_str: str) -> tuple[Optional[str], Optional[str]]:
    if pd.isna(snp_str):
        return None, None

    parts = str(snp_str).split(":")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, None


def make_geno_labels(ref: Optional[str], alt: Optional[str]) -> list[str]:
    """
    Assumes genotype coding is alt-allele dosage:
      0 = ref/ref
      1 = ref/alt
      2 = alt/alt
    """
    if ref is not None and alt is not None:
        return [f"{ref}/{ref}", f"{ref}/{alt}", f"{alt}/{alt}"]
    return ["0", "1", "2"]


def merge_genotype_and_phenotype(
    genotype_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    id_col: str,
    phenotype_col: str,
) -> pd.DataFrame:
    merged = genotype_df.merge(
        phenotype_df[[id_col, phenotype_col]],
        on=id_col,
        how="inner",
    )
    merged = merged.dropna(subset=[phenotype_col]).reset_index(drop=True)
    return merged


def epistasis_tables_with_residuals(
    X: pd.DataFrame,
    y: pd.Series,
    snp1: str,
    snp2: str,
) -> dict:
    """
    Build 3x3 observed means and additive expectation from:
        y ~ g1 + g2

    Returns:
      counts
      means
      sem
      additive_expected
      deviation
      residual_df
      additive_model
    """
    df = pd.DataFrame({
        "g1": X[snp1],
        "g2": X[snp2],
        "y": y,
    }).dropna()

    df = df[df["g1"].isin(GENOS) & df["g2"].isin(GENOS)].copy()

    if df.empty:
        raise ValueError(f"No valid 0/1/2 hard-call rows for {snp1} x {snp2}")

    counts = (
        pd.crosstab(df["g1"], df["g2"])
        .reindex(index=GENOS, columns=GENOS, fill_value=0)
    )

    means = (
        df.groupby(["g1", "g2"])["y"]
        .mean()
        .unstack()
        .reindex(index=GENOS, columns=GENOS)
    )

    sem = (
        df.groupby(["g1", "g2"])["y"]
        .sem()
        .unstack()
        .reindex(index=GENOS, columns=GENOS)
    )

    X_add = sm.add_constant(df[["g1", "g2"]].astype(float), has_constant="add")
    model = sm.OLS(df["y"].astype(float), X_add).fit()

    df["y_add"] = model.predict(X_add)
    df["residual_from_additivity"] = df["y"] - df["y_add"]

    grid = pd.DataFrame([(g1, g2) for g1 in GENOS for g2 in GENOS], columns=["g1", "g2"])
    X_grid = sm.add_constant(grid[["g1", "g2"]].astype(float), has_constant="add")
    grid["y_add"] = model.predict(X_grid)

    additive_expected = (
        grid.pivot(index="g1", columns="g2", values="y_add")
        .reindex(index=GENOS, columns=GENOS)
    )

    deviation = means - additive_expected

    return {
        "snp1": snp1,
        "snp2": snp2,
        "n_samples": len(df),
        "counts": counts,
        "means": means,
        "sem": sem,
        "additive_expected": additive_expected,
        "deviation": deviation,
        "residual_df": df,
        "additive_model": model,
    }


def deviation_residual_mats(res: dict) -> dict:
    """
    Convert additive residuals into 3x3 mean residual and SEM.
    """
    df = res["residual_df"].copy()

    resid_mean = (
        df.groupby(["g1", "g2"])["residual_from_additivity"]
        .mean()
        .unstack()
        .reindex(index=GENOS, columns=GENOS)
    )

    resid_sem = (
        df.groupby(["g1", "g2"])["residual_from_additivity"]
        .sem()
        .unstack()
        .reindex(index=GENOS, columns=GENOS)
    )

    out = dict(res)
    out["deviation"] = resid_mean
    out["residual_sem"] = resid_sem
    return out


def max_deviation_in_powered_cells(
    X: pd.DataFrame,
    y: pd.Series,
    snp1: str,
    snp2: str,
    min_n: int = 50,
) -> tuple[float, int]:
    """
    Max absolute mean residual from additive model, using only genotype cells
    with n >= min_n.
    """
    try:
        res = epistasis_tables_with_residuals(X, y, snp1, snp2)
        res_dev = deviation_residual_mats(res)

        counts = res["counts"]
        deviation = res_dev["deviation"]

        powered_mask = counts >= min_n
        deviation_masked = deviation.copy()
        deviation_masked[~powered_mask] = np.nan

        valid_vals = deviation_masked.to_numpy().flatten()
        valid_vals = valid_vals[~np.isnan(valid_vals)]

        if len(valid_vals) == 0:
            return np.nan, 0

        max_dev = float(np.max(np.abs(valid_vals)))
        n_powered_cells = int(powered_mask.to_numpy().sum())
        return max_dev, n_powered_cells

    except Exception as e:
        print(f"Error for {snp1} x {snp2}: {e}")
        return np.nan, 0


def rank_pairs_for_plot(
    df_pairs: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    pair_class: str = "ML + GWAS + INT",
    min_n: int = 50,
    min_powered_cells: int = 9,
    top_k: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selection criteria:
      - class must match pair_class
      - compute max absolute mean residual in powered cells
      - require at least min_powered_cells
      - sort by max_dev_powered descending, then p_interact_lin ascending
    """
    if "class" not in df_pairs.columns:
        raise ValueError("Pair file must contain a 'class' column.")

    work = df_pairs[df_pairs["class"] == pair_class].copy()

    print(f"\nComputing powered deviations for {pair_class} pairs...")
    work["max_dev_powered"] = np.nan
    work["n_powered_cells"] = 0

    for idx, row in work.iterrows():
        max_dev, n_cells = max_deviation_in_powered_cells(
            X=X,
            y=y,
            snp1=str(row["snp1"]),
            snp2=str(row["snp2"]),
            min_n=min_n,
        )
        work.at[idx, "max_dev_powered"] = max_dev
        work.at[idx, "n_powered_cells"] = n_cells

    sort_cols = ["max_dev_powered"]
    asc = [False]
    if "p_interact_lin" in work.columns:
        sort_cols.append("p_interact_lin")
        asc.append(True)

    selected = (
        work.dropna(subset=["max_dev_powered"])
        .query("n_powered_cells >= @min_powered_cells")
        .sort_values(by=sort_cols, ascending=asc)
        .head(top_k)
        .reset_index(drop=True)
    )

    return work, selected


def plot_interaction_lineplot(
    res: dict,
    *,
    x_on: str = "g1",
    y_mat_key: str = "means",
    err_mat_key: str = "sem",
    y_label: str = "Phenotype",
    title: Optional[str] = None,
    geno_labels_x: Optional[list[str]] = None,
    geno_labels_line: Optional[list[str]] = None,
    annotate_n: bool = True,
    ax=None,
    add_zero_line: bool = False,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    mat = res[y_mat_key]
    errs = res[err_mat_key] if err_mat_key in res else None
    counts = res["counts"]

    x = np.array(GENOS)

    if x_on == "g1":
        y_by_line = {g2: mat[g2] for g2 in GENOS}
        e_by_line = {g2: (errs[g2] if errs is not None else None) for g2 in GENOS}
        n_by_line = {g2: counts[g2] for g2 in GENOS}
        xlab = res["snp1"]
        line_name = res["snp2"]
    elif x_on == "g2":
        y_by_line = {g1: mat.loc[g1] for g1 in GENOS}
        e_by_line = {g1: (errs.loc[g1] if errs is not None else None) for g1 in GENOS}
        n_by_line = {g1: counts.loc[g1] for g1 in GENOS}
        xlab = res["snp2"]
        line_name = res["snp1"]
    else:
        raise ValueError("x_on must be 'g1' or 'g2'.")

    if geno_labels_x is None:
        geno_labels_x = [str(g) for g in GENOS]
    if geno_labels_line is None:
        geno_labels_line = [str(g) for g in GENOS]

    line_offsets = {
        0: (-10, 8),
        1: (4, 8),
        2: (4, -10),
    }

    for k in GENOS:
        yv = y_by_line[k].values.astype(float)
        ev = e_by_line[k].values.astype(float) if e_by_line[k] is not None else None
        mask = ~np.isnan(yv)

        if ev is not None:
            ax.errorbar(
                x[mask], yv[mask], yerr=ev[mask],
                marker="o", linestyle="--", linewidth=1.5,
                capsize=3, label=f"{line_name} {geno_labels_line[k]}"
            )
        else:
            ax.plot(
                x[mask], yv[mask],
                marker="o", linestyle="--", linewidth=1.5,
                label=f"{line_name} {geno_labels_line[k]}"
            )

        if annotate_n:
            nv = n_by_line[k].values.astype(int)

            for j, (xi, yi, ni, ok) in enumerate(zip(x, yv, nv, mask)):
                if not ok:
                    continue

                dx, dy = line_offsets.get(k, (4, 4))
                if j == 0:
                    dx += -2
                elif j == 2:
                    dx += 2

                ax.annotate(
                    f"{ni}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    ha="left",
                    va="bottom",
                    fontsize=11,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        fc="white",
                        ec="none",
                        alpha=0.7,
                    ),
                )

    if add_zero_line:
        ax.axhline(0, linestyle="--", linewidth=1, color="gray", alpha=0.8)

    ax.set_xticks(GENOS)
    ax.set_xticklabels(geno_labels_x)
    ax.set_xlabel(xlab)
    ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title)

    ax.legend(frameon=False, fontsize=12)
    ax.grid(alpha=0.2)
    ax.margins(x=0.05, y=0.15)
    return ax


def plot_one_pair(
    fig,
    gs,
    row_idx: int,
    pair_row: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
) -> None:
    snp1 = str(pair_row["snp1"])
    snp2 = str(pair_row["snp2"])

    p_int = float(pair_row.get("p_interact_lin", np.nan))
    beta = float(pair_row.get("beta_interact_lin", np.nan))

    rs1 = extract_rsid(snp1)
    rs2 = extract_rsid(snp2)

    ref1, alt1 = extract_alleles(snp1)
    ref2, alt2 = extract_alleles(snp2)

    geno_labels_x = make_geno_labels(ref1, alt1)
    geno_labels_line = make_geno_labels(ref2, alt2)

    title_obs = f"{rs1} × {rs2}  |  p = {p_int:.2e}  |  beta = {beta:.3g}"

    res = epistasis_tables_with_residuals(X, y, snp1, snp2)
    res_dev = deviation_residual_mats(res)

    ax_obs = fig.add_subplot(gs[row_idx, 0])
    ax_res = fig.add_subplot(gs[row_idx, 1])

    plot_interaction_lineplot(
        res,
        x_on="g1",
        y_mat_key="means",
        err_mat_key="sem",
        y_label="BMI",
        title=title_obs,
        geno_labels_x=geno_labels_x,
        geno_labels_line=geno_labels_line,
        annotate_n=True,
        ax=ax_obs,
        add_zero_line=False,
    )

    plot_interaction_lineplot(
        res_dev,
        x_on="g1",
        y_mat_key="deviation",
        err_mat_key="residual_sem",
        y_label="BMI residual",
        title="Mean residual from additive model",
        geno_labels_x=geno_labels_x,
        geno_labels_line=geno_labels_line,
        annotate_n=True,
        ax=ax_res,
        add_zero_line=True,
    )

    for ax in [ax_obs, ax_res]:
        handles, _ = ax.get_legend_handles_labels()
        new_labels = [f"{rs2} {gl}" for gl in geno_labels_line]
        ax.legend(handles, new_labels, frameon=False, fontsize=11)
        ax.set_xlabel(rs1, fontsize=11)

    ax_obs.text(
        -0.12, 1.05, label,
        transform=ax_obs.transAxes,
        fontsize=11,
        va="top",
        ha="left",
    )


def make_figure(
    intra_pairs: pd.DataFrame,
    inter_pairs: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    output_prefix: Path,
) -> None:
    n_intra = len(intra_pairs)
    n_inter = len(inter_pairs)
    total_rows = n_intra + n_inter

    if total_rows == 0:
        print("No pairs selected for plotting.")
        return

    fig = plt.figure(figsize=(15, max(5, total_rows * 2.5)), dpi=300)
    gs = GridSpec(total_rows, 2, figure=fig, hspace=0.39, wspace=0.20)
    plt.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.04)

    panel_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    for i, (_, row) in enumerate(intra_pairs.iterrows()):
        plot_one_pair(fig, gs, i, row, X, y, label=panel_labels[i])

    for i, (_, row) in enumerate(inter_pairs.iterrows()):
        plot_one_pair(fig, gs, i + n_intra, row, X, y, label=panel_labels[i + n_intra])

    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")

    plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()

    print("\nSaved:")
    print(pdf_path)
    print(png_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deviation-from-additivity plots for selected SNP-SNP interaction pairs."
    )

    parser.add_argument("--genotype_csv", required=True, help="Genotype matrix CSV with SNP columns.")
    parser.add_argument("--phenotype_csv", required=True, help="Phenotype CSV.")
    parser.add_argument("--intra_pairs_csv", required=True, help="Intra-chromosomal pair table.")
    parser.add_argument("--inter_pairs_csv", required=True, help="Inter-chromosomal pair table.")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs.")

    parser.add_argument("--id_col", default="SUBJID")
    parser.add_argument("--phenotype_col", default="BMI")
    parser.add_argument("--pair_class", default="ML + GWAS + INT")
    parser.add_argument("--min_cell_n", type=int, default=50)
    parser.add_argument("--min_powered_cells", type=int, default=9)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output_prefix", default="deviation_from_additivity_figure")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    genotype_df = read_table_with_id(args.genotype_csv, args.id_col)
    phenotype_df = read_table_with_id(args.phenotype_csv, args.id_col)

    df_intra = pd.read_csv(args.intra_pairs_csv)
    df_inter = pd.read_csv(args.inter_pairs_csv)

    merged = merge_genotype_and_phenotype(
        genotype_df=genotype_df,
        phenotype_df=phenotype_df,
        id_col=args.id_col,
        phenotype_col=args.phenotype_col,
    )

    X = merged.drop(columns=[args.phenotype_col]).copy()
    if args.id_col in X.columns:
        X = X.drop(columns=[args.id_col])
    y = merged[args.phenotype_col].copy()

    intra_all, intra_pairs = rank_pairs_for_plot(
        df_pairs=df_intra,
        X=X,
        y=y,
        pair_class=args.pair_class,
        min_n=args.min_cell_n,
        min_powered_cells=args.min_powered_cells,
        top_k=args.top_k,
    )

    inter_all, inter_pairs = rank_pairs_for_plot(
        df_pairs=df_inter,
        X=X,
        y=y,
        pair_class=args.pair_class,
        min_n=args.min_cell_n,
        min_powered_cells=args.min_powered_cells,
        top_k=args.top_k,
    )

    intra_all.to_csv(outdir / "intra_pairs_with_deviation_metrics.csv", index=False)
    inter_all.to_csv(outdir / "inter_pairs_with_deviation_metrics.csv", index=False)
    intra_pairs.to_csv(outdir / "selected_intra_pairs.csv", index=False)
    inter_pairs.to_csv(outdir / "selected_inter_pairs.csv", index=False)

    make_figure(
        intra_pairs=intra_pairs,
        inter_pairs=inter_pairs,
        X=X,
        y=y,
        output_prefix=outdir / args.output_prefix,
    )

    summary = {
        "pair_class": args.pair_class,
        "min_cell_n": args.min_cell_n,
        "min_powered_cells": args.min_powered_cells,
        "top_k_intra": int(len(intra_pairs)),
        "top_k_inter": int(len(inter_pairs)),
        "n_samples_merged": int(len(merged)),
    }
    save_json(summary, outdir / "deviation_from_additivity_summary.json")

    print("\nSelected intra pairs:")
    if not intra_pairs.empty:
        cols = [c for c in [
            "snp1", "snp2", "interaction_strength", "p_interact_lin",
            "beta_interact_lin", "max_dev_powered", "n_powered_cells"
        ] if c in intra_pairs.columns]
        print(intra_pairs[cols])

    print("\nSelected inter pairs:")
    if not inter_pairs.empty:
        cols = [c for c in [
            "snp1", "snp2", "interaction_strength", "p_interact_lin",
            "beta_interact_lin", "max_dev_powered", "n_powered_cells"
        ] if c in inter_pairs.columns]
        print(inter_pairs[cols])


if __name__ == "__main__":
    main()
