#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def clean_id(x):
    return str(x).strip().replace(".0", "")


def clean_snp(x):
    x = str(x).strip()
    m = re.search(r"rs\d+", x)
    return m.group(0) if m else x


def canonical_pair(a, b):
    a = clean_snp(a)
    b = clean_snp(b)
    return "||".join(sorted([a, b]))


def read_optional_snp_file(path):
    if path is None:
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    snp_cols = ["snp", "SNP", "rsid", "rsID", "variant", "feature", "feature_name"]

    for col in snp_cols:
        if col in df.columns:
            return df[col].dropna().astype(str).map(clean_snp).unique().tolist()

    if df.shape[1] == 1:
        return df.iloc[:, 0].dropna().astype(str).map(clean_snp).unique().tolist()

    raise ValueError(f"Could not find SNP column in {path}. Columns: {df.columns.tolist()}")


def parse_beta_epistatic_file(beta_file):
    df = pd.read_csv(beta_file)
    df.columns = df.columns.str.strip()

    if "SNP_Pair" not in df.columns:
        raise ValueError("Expected column 'SNP_Pair' in beta_epistatic file.")

    pairs = df["SNP_Pair"].astype(str).str.split("x", n=1, expand=True)

    if pairs.shape[1] != 2:
        raise ValueError("Could not split SNP_Pair column using 'x'.")

    df["snp1"] = pairs[0].map(clean_snp)
    df["snp2"] = pairs[1].map(clean_snp)
    df["pair_key"] = [canonical_pair(a, b) for a, b in zip(df["snp1"], df["snp2"])]

    return df


def write_plink_ped_map(geno, pheno, snps, outdir, sample_col, phenotype_col):
    """
    Create PLINK text files from 0/1/2 genotype dosages.

    Genotype coding:
    0 = A A
    1 = A G
    2 = G G
    missing = 0 0
    """

    ped_file = outdir / "plink_text.ped"
    map_file = outdir / "plink_text.map"

    with open(map_file, "w") as f:
        for i, snp in enumerate(snps, start=1):
            f.write(f"1\t{snp}\t0\t{i}\n")

    geno_values = geno[snps].apply(pd.to_numeric, errors="coerce")

    finite_vals = geno_values.to_numpy()
    finite_vals = finite_vals[np.isfinite(finite_vals)]

    unique_vals = np.unique(np.round(finite_vals, 6))

    allowed = set([0.0, 1.0, 2.0])
    observed_small = set(np.unique(np.round(unique_vals, 0)))

    if not observed_small.issubset(allowed):
        raise ValueError(
            "PLINK PED export requires genotype values coded as 0/1/2. "
            f"Observed rounded values include: {sorted(list(observed_small))[:20]}"
        )

    pheno_map = dict(zip(pheno[sample_col], pheno[phenotype_col]))

    with open(ped_file, "w") as f:
        for sid, row in geno_values.iterrows():
            y = pheno_map.get(sid, -9)

            fields = [
                sid,      # FID
                sid,      # IID
                "0",      # father
                "0",      # mother
                "0",      # sex unknown
                str(y),   # phenotype
            ]

            for g in row.values:
                if not np.isfinite(g):
                    fields.extend(["0", "0"])
                else:
                    dosage = int(round(float(g)))
                    if dosage == 0:
                        fields.extend(["A", "A"])
                    elif dosage == 1:
                        fields.extend(["A", "G"])
                    elif dosage == 2:
                        fields.extend(["G", "G"])
                    else:
                        fields.extend(["0", "0"])

            f.write(" ".join(fields) + "\n")

    return ped_file, map_file


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--geno_csv", required=True)
    parser.add_argument("--pheno_csv", required=True)
    parser.add_argument("--beta_epistatic_csv", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--sample_col", default="SNP")
    parser.add_argument("--phenotype_col", default="non_linear")

    parser.add_argument("--scan_snps_file", default=None)
    parser.add_argument("--n_scan_snps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading genotype:", args.geno_csv)
    geno = pd.read_csv(args.geno_csv, index_col=0)
    geno.index = [clean_id(x) for x in geno.index]
    geno.columns = [clean_snp(c) for c in geno.columns]

    # Drop duplicated SNP columns if any
    geno = geno.loc[:, ~pd.Index(geno.columns).duplicated()].copy()

    print("Genotype shape:", geno.shape)

    print("Loading phenotype:", args.pheno_csv)
    pheno = pd.read_csv(args.pheno_csv)
    pheno.columns = pheno.columns.str.strip()

    if args.sample_col not in pheno.columns:
        raise ValueError(f"Sample column {args.sample_col} not found in phenotype file.")

    if args.phenotype_col not in pheno.columns:
        raise ValueError(f"Phenotype column {args.phenotype_col} not found in phenotype file.")

    pheno[args.sample_col] = pheno[args.sample_col].map(clean_id)
    pheno[args.phenotype_col] = pd.to_numeric(pheno[args.phenotype_col], errors="coerce")
    pheno = pheno.dropna(subset=[args.sample_col, args.phenotype_col]).copy()

    print("Phenotype rows:", len(pheno))

    common_ids = sorted(set(geno.index) & set(pheno[args.sample_col]))

    if len(common_ids) == 0:
        raise ValueError("No matching sample IDs between genotype and phenotype.")

    geno = geno.loc[common_ids].copy()
    pheno = pheno[pheno[args.sample_col].isin(common_ids)].copy()
    pheno = pheno.set_index(args.sample_col).loc[common_ids].reset_index()

    print("Matched samples:", len(common_ids))

    print("Loading true epistatic pairs:", args.beta_epistatic_csv)
    true_pairs = parse_beta_epistatic_file(args.beta_epistatic_csv)

    true_snps = sorted(set(true_pairs["snp1"]) | set(true_pairs["snp2"]))
    available_snps = set(geno.columns)

    true_snps_available = sorted(set(true_snps) & available_snps)
    true_snps_missing = sorted(set(true_snps) - available_snps)

    print("True causal SNPs:", len(true_snps))
    print("True causal SNPs available in genotype:", len(true_snps_available))
    print("True causal SNPs missing:", len(true_snps_missing))

    if len(true_snps_missing) > 0:
        pd.Series(true_snps_missing).to_csv(
            outdir / "missing_true_snps.txt",
            index=False,
            header=False,
        )

    # ------------------------------------------------------------
    # Choose SNPs for PLINK and MatrixEpistasis scan
    # ------------------------------------------------------------
    user_scan_snps = read_optional_snp_file(args.scan_snps_file)

    if user_scan_snps is not None:
        scan_snps = sorted(set(user_scan_snps) & available_snps)

        print("Using user-provided scan SNP file.")
        print("Input scan SNPs:", len(set(user_scan_snps)))
        print("Available scan SNPs:", len(scan_snps))

    else:
        print("No scan SNP file provided.")
        print("Using true causal SNPs + random background SNPs.")

        background_pool = sorted(available_snps - set(true_snps_available))

        n_background = max(0, args.n_scan_snps - len(true_snps_available))
        n_background = min(n_background, len(background_pool))

        background_snps = rng.choice(background_pool, size=n_background, replace=False).tolist()

        scan_snps = sorted(set(true_snps_available) | set(background_snps))

    print("Final SNPs for scan:", len(scan_snps))

    pd.Series(scan_snps).to_csv(
        outdir / "snps_for_scan.txt",
        index=False,
        header=False,
    )

    # ------------------------------------------------------------
    # Save true pairs with testability flag
    # ------------------------------------------------------------
    true_pairs["snp1_in_scan"] = true_pairs["snp1"].isin(scan_snps)
    true_pairs["snp2_in_scan"] = true_pairs["snp2"].isin(scan_snps)
    true_pairs["pair_testable"] = true_pairs["snp1_in_scan"] & true_pairs["snp2_in_scan"]

    true_pairs.to_csv(
        outdir / "true_pairs_canonical.tsv",
        sep="\t",
        index=False,
    )

    print(
        "True pairs testable in scan SNP set:",
        int(true_pairs["pair_testable"].sum()),
        "/",
        len(true_pairs),
    )

    # ------------------------------------------------------------
    # MatrixEpistasis inputs
    # ------------------------------------------------------------
    geno_scan = geno[scan_snps].apply(pd.to_numeric, errors="coerce").copy()
    geno_scan.insert(0, "sample_id", geno_scan.index)

    geno_scan.to_csv(
        outdir / "matrixepistasis_genotype.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    pheno_out = pheno[[args.sample_col, args.phenotype_col]].copy()
    pheno_out = pheno_out.rename(
        columns={
            args.sample_col: "sample_id",
            args.phenotype_col: "phenotype",
        }
    )

    pheno_out.to_csv(
        outdir / "phenotype_for_tools.tsv",
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------
    # PLINK text inputs
    # ------------------------------------------------------------
    pheno_for_plink = pd.DataFrame(
        {
            "FID": pheno[args.sample_col],
            "IID": pheno[args.sample_col],
            "PHENO": pheno[args.phenotype_col],
        }
    )

    pheno_for_plink.to_csv(
        outdir / "pheno_plink.tsv",
        sep="\t",
        index=False,
    )

    write_plink_ped_map(
        geno=geno[scan_snps],
        pheno=pheno,
        snps=scan_snps,
        outdir=outdir,
        sample_col=args.sample_col,
        phenotype_col=args.phenotype_col,
    )

    print("Done preparing inputs.")
    print("Output directory:", outdir)


if __name__ == "__main__":
    main()