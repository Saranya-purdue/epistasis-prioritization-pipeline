#!/usr/bin/env python3
"""Generate a tiny synthetic input set for smoke-testing repository utilities."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="demo_data")
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--n-snps", type=int, default=40)
    p.add_argument("--n-pairs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if 2 * args.n_pairs > args.n_snps:
        raise ValueError("Need at least 2*n_pairs SNPs.")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sample_ids = [f"sample_{i+1}" for i in range(args.n_samples)]
    snps = [f"rs{100000+i}" for i in range(args.n_snps)]
    maf = rng.uniform(0.20, 0.45, size=args.n_snps)
    G = np.column_stack([rng.binomial(2, p, size=args.n_samples) for p in maf])

    geno = pd.DataFrame(G, index=sample_ids, columns=snps)
    geno.index.name = "sample_id"
    geno.to_csv(out / "genotype.csv")

    pairs = pd.DataFrame(
        {
            "Pair_ID": [f"PAIR_{i+1}" for i in range(args.n_pairs)],
            "SNP1": snps[: 2 * args.n_pairs : 2],
            "SNP2": snps[1 : 2 * args.n_pairs : 2],
        }
    )
    pairs.to_csv(out / "causal_pairs.csv", index=False)

    # A simple phenotype for input-format smoke tests.
    y = rng.normal(size=args.n_samples)
    pd.DataFrame({"SNP": sample_ids, "Phenotype": y}).to_csv(
        out / "phenotype.csv", index=False
    )

    print(f"Demo files written to {out.resolve()}")


if __name__ == "__main__":
    main()
