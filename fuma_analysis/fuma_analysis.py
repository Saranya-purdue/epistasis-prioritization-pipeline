from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def parse_variant_string(variant: str) -> dict:
    """
    Parse variant strings like:
      chr16:12345:rs123:A:G
      16:12345:rs123:A:G

    Returns a dict with keys:
      chr, pos, rsid, ref, alt
    Missing fields are returned as None.
    """
    parts = str(variant).split(":")
    out = {
        "chr": None,
        "pos": None,
        "rsid": None,
        "ref": None,
        "alt": None,
    }

    if len(parts) >= 1:
        out["chr"] = parts[0].replace("chr", "")
    if len(parts) >= 2:
        try:
            out["pos"] = int(parts[1])
        except Exception:
            out["pos"] = None
    if len(parts) >= 3:
        out["rsid"] = parts[2]
    if len(parts) >= 4:
        out["ref"] = parts[3]
    if len(parts) >= 5:
        out["alt"] = parts[4]

    return out


def extract_rsid(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda x: parse_variant_string(x)["rsid"])


def collect_unique_snps(pairs_df: pd.DataFrame, snp_cols: list[str]) -> pd.Series:
    all_snps = pd.unique(pairs_df[snp_cols].values.ravel())
    return pd.Series(all_snps, name="raw").astype(str)


def prepare_lead_snps_from_pairs(
    pairs_df: pd.DataFrame,
    snp_cols: list[str] = ["snp1", "snp2"],
    keep_only_valid_rsid: bool = True,
) -> pd.DataFrame:
    """
    Build a FUMA lead SNP file from all unique SNPs in a pair table.
    """
    s = collect_unique_snps(pairs_df, snp_cols)

    parsed = s.map(parse_variant_string)
    lead = pd.DataFrame({
        "rsID": [x["rsid"] for x in parsed],
        "chr": [x["chr"] for x in parsed],
        "pos": [x["pos"] for x in parsed],
    })

    lead = lead.dropna(subset=["rsID", "chr", "pos"]).copy()

    if keep_only_valid_rsid:
        lead = lead[lead["rsID"].astype(str).str.match(r"^rs\d+$", na=False)].copy()

    lead["chr_sort"] = pd.to_numeric(lead["chr"], errors="coerce")
    lead = (
        lead.drop_duplicates(subset=["rsID", "chr", "pos"])
        .sort_values(["chr_sort", "chr", "pos"])
        .drop(columns=["chr_sort"])
        .reset_index(drop=True)
    )

    return lead


def load_fuma_snps_file(fuma_snps_file: str | Path) -> pd.DataFrame:
    """
    Load FUMA SNP2GENE snps.txt output.
    """
    fuma = pd.read_csv(fuma_snps_file, sep="\t", dtype=str)

    required = {"rsID", "nearestGene", "gwasP"}
    missing = required - set(fuma.columns)
    if missing:
        raise ValueError(
            f"FUMA file is missing required columns: {missing}. "
            f"Found columns: {list(fuma.columns)}"
        )

    fuma["gwasP_num"] = pd.to_numeric(fuma["gwasP"], errors="coerce")
    return fuma


def build_gene_map(fuma_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rsID -> nearestGene map.
    If duplicate rsIDs exist, keep the row with smallest gwasP.
    """
    gene_map = (
        fuma_df[["rsID", "nearestGene", "gwasP_num"]]
        .dropna(subset=["rsID"])
        .sort_values("gwasP_num", na_position="last")
        .drop_duplicates(subset=["rsID"], keep="first")
        .drop(columns=["gwasP_num"])
        .reset_index(drop=True)
    )
    return gene_map


def annotate_pairs_with_fuma(
    pairs_df: pd.DataFrame,
    fuma_df: pd.DataFrame,
    snp1_col: str = "snp1",
    snp2_col: str = "snp2",
) -> pd.DataFrame:
    out = pairs_df.copy()

    out["snp1_rsid"] = extract_rsid(out[snp1_col])
    out["snp2_rsid"] = extract_rsid(out[snp2_col])

    gene_map = build_gene_map(fuma_df)

    out = out.merge(
        gene_map.rename(columns={"rsID": "snp1_rsid", "nearestGene": "snp1_gene"}),
        on="snp1_rsid",
        how="left",
    )

    out = out.merge(
        gene_map.rename(columns={"rsID": "snp2_rsid", "nearestGene": "snp2_gene"}),
        on="snp2_rsid",
        how="left",
    )

    return out


def prepare_leads_command(args: argparse.Namespace) -> None:
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    pairs_df = pd.read_csv(args.pairs_csv)
    lead = prepare_lead_snps_from_pairs(
        pairs_df=pairs_df,
        snp_cols=[args.snp1_col, args.snp2_col],
        keep_only_valid_rsid=not args.keep_invalid_rsid,
    )

    out_path = outdir / args.output_name
    lead.to_csv(out_path, sep="\t", index=False)

    summary = {
        "pairs_csv": str(args.pairs_csv),
        "n_input_pairs": int(len(pairs_df)),
        "n_lead_snps": int(len(lead)),
        "output_file": str(out_path),
    }
    save_json(summary, outdir / "prepare_leads_summary.json")

    print(f"Saved lead SNP file: {out_path}")
    print(f"Number of lead SNPs: {len(lead)}")
    print(lead.head())


def annotate_pairs_command(args: argparse.Namespace) -> None:
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    pairs_df = pd.read_csv(args.pairs_csv)
    fuma_df = load_fuma_snps_file(args.fuma_snps_file)

    annotated = annotate_pairs_with_fuma(
        pairs_df=pairs_df,
        fuma_df=fuma_df,
        snp1_col=args.snp1_col,
        snp2_col=args.snp2_col,
    )

    out_path = outdir / args.output_name
    annotated.to_csv(out_path, index=False)

    mapped_snp1 = annotated["snp1_gene"].notna().sum() if "snp1_gene" in annotated.columns else 0
    mapped_snp2 = annotated["snp2_gene"].notna().sum() if "snp2_gene" in annotated.columns else 0

    summary = {
        "pairs_csv": str(args.pairs_csv),
        "fuma_snps_file": str(args.fuma_snps_file),
        "n_pairs": int(len(annotated)),
        "n_snp1_gene_mapped": int(mapped_snp1),
        "n_snp2_gene_mapped": int(mapped_snp2),
        "output_file": str(out_path),
    }
    save_json(summary, outdir / "annotate_pairs_summary.json")

    print(f"Saved annotated pairs: {out_path}")
    print(annotated[[args.snp1_col, "snp1_gene", args.snp2_col, "snp2_gene"]].head())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare FUMA lead SNP inputs and annotate SNP pairs with FUMA nearest-gene results."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare-leads
    p1 = subparsers.add_parser(
        "prepare-leads",
        help="Prepare a lead SNP file from a SNP-pair table for FUMA SNP2GENE upload.",
    )
    p1.add_argument("--pairs_csv", required=True)
    p1.add_argument("--output_dir", required=True)
    p1.add_argument("--output_name", default="fuma_predefined_lead_snps.txt")
    p1.add_argument("--snp1_col", default="snp1")
    p1.add_argument("--snp2_col", default="snp2")
    p1.add_argument(
        "--keep_invalid_rsid",
        action="store_true",
        help="Keep rows even if rsID does not match '^rs\\d+$'.",
    )
    p1.set_defaults(func=prepare_leads_command)

    # annotate-pairs
    p2 = subparsers.add_parser(
        "annotate-pairs",
        help="Annotate a SNP-pair table using FUMA SNP2GENE snps.txt output.",
    )
    p2.add_argument("--pairs_csv", required=True)
    p2.add_argument("--fuma_snps_file", required=True, help="FUMA SNP2GENE snps.txt output.")
    p2.add_argument("--output_dir", required=True)
    p2.add_argument("--output_name", default="pairs_with_fuma_genes.csv")
    p2.add_argument("--snp1_col", default="snp1")
    p2.add_argument("--snp2_col", default="snp2")
    p2.set_defaults(func=annotate_pairs_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
