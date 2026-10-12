#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def rsid(x):
    x = str(x).strip()
    m = re.search(r"rs\d+", x, flags=re.I)
    return m.group(0) if m else x


def load_truth(files):
    causal = set()

    for f in files:
        df = pd.read_csv(f)

        if not {"SNP1", "SNP2"}.issubset(df.columns):
            raise ValueError(f"{f}: SNP1/SNP2 missing")

        for c in ["SNP1", "SNP2"]:
            causal.update(df[c].astype(str).map(rsid))

    return causal


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--genotype", required=True)
    ap.add_argument("--snp-map", required=True)
    ap.add_argument("--truth", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[5000, 10000, 20000, 30000],
    )

    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--chunksize", type=int, default=250)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sizes = sorted(args.sizes)

    # --------------------------------------------------------
    # Read only header
    # --------------------------------------------------------

    header = pd.read_csv(
        args.genotype,
        nrows=0
    )

    sample_col = header.columns[0]
    variants = list(header.columns[1:])

    print("Sample column:", sample_col)
    print("Full SNP count:", len(variants))

    variant_to_rsid = {
        v: rsid(v)
        for v in variants
    }

    rsid_to_variant = {}

    for v, r in variant_to_rsid.items():
        if r in rsid_to_variant:
            raise ValueError(
                f"Duplicate rsID in genotype: {r}"
            )
        rsid_to_variant[r] = v

    # --------------------------------------------------------
    # Causal union
    # --------------------------------------------------------

    causal_rsids = load_truth(
        args.truth
    )

    missing = sorted(
        causal_rsids -
        set(rsid_to_variant)
    )

    if missing:
        raise ValueError(
            f"{len(missing)} causal SNPs absent from full genotype.\n"
            f"Examples: {missing[:20]}"
        )

    causal_variants = {
        rsid_to_variant[r]
        for r in causal_rsids
    }

    print("Unique causal SNPs:", len(causal_variants))

    if len(causal_variants) > min(sizes):
        raise ValueError(
            "Smallest benchmark size is smaller than causal SNP count."
        )

    # --------------------------------------------------------
    # Random non-causal priority
    #
    # One permutation -> exact nesting across all target sizes.
    # --------------------------------------------------------

    noncausal = [
        v for v in variants
        if v not in causal_variants
    ]

    rng = np.random.default_rng(
        args.seed
    )

    rng.shuffle(
        noncausal
    )

    selected_by_size = {}

    for size in sizes:

        n_needed = size - len(causal_variants)

        chosen = (
            causal_variants |
            set(noncausal[:n_needed])
        )

        # Preserve original genotype column order
        selected = [
            v for v in variants
            if v in chosen
        ]

        if len(selected) != size:
            raise RuntimeError(
                f"{size}: selected {len(selected)} instead."
            )

        selected_by_size[size] = selected

    # --------------------------------------------------------
    # Verify nesting
    # --------------------------------------------------------

    for small, large in zip(
        sizes[:-1],
        sizes[1:]
    ):
        assert set(
            selected_by_size[small]
        ).issubset(
            selected_by_size[large]
        )

    # --------------------------------------------------------
    # SNP map
    # --------------------------------------------------------

    mapping = pd.read_csv(
        args.snp_map
    )

    if "VARIANT_ID" in mapping.columns:

        mapping["VARIANT_ID"] = (
            mapping["VARIANT_ID"]
            .astype(str)
        )

        map_key = "VARIANT_ID"

    elif "SNP" in mapping.columns:

        mapping["SNP"] = (
            mapping["SNP"]
            .astype(str)
            .map(rsid)
        )

        map_key = "SNP"

    else:
        raise ValueError(
            "SNP map needs VARIANT_ID or SNP."
        )

    if "CHR" not in mapping.columns:
        raise ValueError(
            "SNP map needs CHR."
        )

    # --------------------------------------------------------
    # Write variant manifests
    # --------------------------------------------------------

    summary_rows = []

    for size in sizes:

        selected = selected_by_size[size]

        pd.DataFrame({
            "VARIANT_ID": selected,
            "SNP": [variant_to_rsid[x] for x in selected],
            "Is_causal": [
                x in causal_variants
                for x in selected
            ],
        }).to_csv(
            outdir /
            f"benchmark_{size}_variants.csv",
            index=False
        )

        if map_key == "VARIANT_ID":

            submap = mapping[
                mapping["VARIANT_ID"].isin(selected)
            ].copy()

        else:

            selected_rsids = {
                variant_to_rsid[x]
                for x in selected
            }

            submap = mapping[
                mapping["SNP"].isin(selected_rsids)
            ].copy()

        submap.to_csv(
            outdir /
            f"benchmark_{size}_snp_map.csv",
            index=False
        )

        summary_rows.append({
            "Input_SNPs": size,
            "Causal_SNPs": len(causal_variants),
            "Noncausal_SNPs":
                size - len(causal_variants),
            "Possible_pairs":
                size * (size - 1) // 2,
        })

    # --------------------------------------------------------
    # Stream full CSV only once
    # --------------------------------------------------------

    output_files = {}

    for size in sizes:

        outfile = (
            outdir /
            f"genotype_benchmark_{size}.csv"
        )

        if outfile.exists():
            outfile.unlink()

        output_files[size] = outfile

    first = True

    print("Streaming genotype...")

    for chunk_i, chunk in enumerate(
        pd.read_csv(
            args.genotype,
            chunksize=args.chunksize
        )
    ):

        for size in sizes:

            cols = (
                [sample_col] +
                selected_by_size[size]
            )

            chunk[cols].to_csv(
                output_files[size],
                mode="w" if first else "a",
                header=first,
                index=False
            )

        first = False

        if chunk_i % 20 == 0:
            print(
                "Processed chunk",
                chunk_i,
                flush=True
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        outdir /
        "benchmark_genotype_summary.csv",
        index=False
    )

    with open(
        outdir /
        "benchmark_build_metadata.json",
        "w"
    ) as f:

        json.dump({
            "seed": args.seed,
            "sizes": sizes,
            "n_causal_snps":
                len(causal_variants),
            "truth_files":
                args.truth,
        }, f, indent=2)

    print(summary.to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()