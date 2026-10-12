#!/usr/bin/env python3
"""Standardize one EpiGEN phenotype/truth condition for the common benchmark tools."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def clean_id(values):
    return (
        pd.Series(values, dtype="object")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--genotype", required=True, help="Genotype CSV; first column is sample ID.")
    p.add_argument("--phenotype", required=True, help="Raw EpiGEN phenotype CSV.")
    p.add_argument("--truth", required=True, help="Raw EpiGEN true-pair CSV with SNP1/SNP2.")
    p.add_argument("--phenotype-out", required=True)
    p.add_argument("--truth-out", required=True)
    p.add_argument("--phenotype-column", default="Phenotype")
    args = p.parse_args()

    geno_ids = clean_id(pd.read_csv(args.genotype, usecols=[0]).iloc[:, 0])
    pheno = pd.read_csv(args.phenotype)

    if args.phenotype_column not in pheno.columns:
        raise ValueError(
            f"Phenotype column '{args.phenotype_column}' not found. "
            f"Columns: {pheno.columns.tolist()}"
        )
    if len(pheno) != len(geno_ids):
        raise ValueError(
            f"Phenotype N={len(pheno)} does not match genotype N={len(geno_ids)}."
        )

    # Prefer explicit sample-ID matching when possible. Older EpiGEN runs may
    # contain synthetic IIDs, in which case row order is the documented fallback.
    if "IID" in pheno.columns:
        iid = clean_id(pheno["IID"])
        if iid.nunique() == len(iid) and set(iid) == set(geno_ids):
            tmp = pheno.copy()
            tmp["_IID_clean"] = iid.values
            pheno = tmp.set_index("_IID_clean").loc[geno_ids.values].reset_index(drop=True)

    y = pd.to_numeric(pheno[args.phenotype_column], errors="coerce")
    if y.isna().any():
        raise ValueError("Phenotype contains missing or non-numeric values.")
    sd = y.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("Phenotype SD is zero/non-finite and cannot be standardized.")
    y_z = (y - y.mean()) / sd

    out = pd.DataFrame(
        {
            "SNP": geno_ids.values,
            "Phenotype_raw": y.values,
            "Phenotype": y_z.values,
        }
    )
    if "Epistatic_Component" in pheno.columns:
        out["Epistatic_Component"] = pd.to_numeric(
            pheno["Epistatic_Component"], errors="coerce"
        ).values

    phenotype_out = Path(args.phenotype_out)
    truth_out = Path(args.truth_out)
    phenotype_out.parent.mkdir(parents=True, exist_ok=True)
    truth_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(phenotype_out, index=False)

    truth = pd.read_csv(args.truth)
    if not {"SNP1", "SNP2"}.issubset(truth.columns):
        raise ValueError(f"Truth requires SNP1/SNP2. Columns: {truth.columns.tolist()}")
    truth = truth.copy()
    truth["SNP_Pair"] = (
        truth["SNP1"].astype(str).str.strip()
        + "x"
        + truth["SNP2"].astype(str).str.strip()
    )
    truth = truth[["SNP_Pair"] + [c for c in truth.columns if c != "SNP_Pair"]]
    truth.to_csv(truth_out, index=False)

    print(f"Saved phenotype: {phenotype_out}")
    print(f"Saved truth: {truth_out}")
    print(f"N={len(out):,}; phenotype mean={out['Phenotype'].mean():.6g}; "
          f"SD={out['Phenotype'].std(ddof=0):.6g}; true pairs={len(truth):,}")


if __name__ == "__main__":
    main()
