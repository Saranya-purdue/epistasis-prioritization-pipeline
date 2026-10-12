#!/usr/bin/env python3
"""Portable PLINK 1.9 quantitative epistasis runner for prepared benchmark inputs."""

import argparse
from pathlib import Path
import shutil
import subprocess


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--plink", default="plink", help="PLINK 1.9 executable.")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--epi1", type=float, default=1.0)
    args = p.parse_args()

    exe = shutil.which(args.plink) or (args.plink if Path(args.plink).exists() else None)
    if exe is None:
        raise FileNotFoundError(
            f"Could not find PLINK executable '{args.plink}'. Put PLINK 1.9 on PATH "
            "or pass --plink /path/to/plink."
        )

    inp = Path(args.input_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ped = inp / "plink_text.ped"
    mapf = inp / "plink_text.map"
    pheno = inp / "pheno_plink.tsv"
    for f in (ped, mapf, pheno):
        if not f.exists():
            raise FileNotFoundError(f)

    bed_prefix = inp / "benchmark_epigen"
    if not (bed_prefix.with_suffix(".bed")).exists():
        run(
            [
                exe,
                "--file", inp / "plink_text",
                "--make-bed",
                "--allow-no-sex",
                "--out", bed_prefix,
            ]
        )

    out_prefix = outdir / "plink_epistasis"
    run(
        [
            exe,
            "--bfile", bed_prefix,
            "--pheno", pheno,
            "--pheno-name", "PHENO",
            "--allow-no-sex",
            "--epistasis",
            "--epi1", args.epi1,
            "--threads", args.threads,
            "--out", out_prefix,
        ]
    )
    print(f"PLINK outputs: {out_prefix}.*")


if __name__ == "__main__":
    main()
