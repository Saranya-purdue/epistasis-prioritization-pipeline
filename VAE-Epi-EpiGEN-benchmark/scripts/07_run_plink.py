#!/usr/bin/env python3
"""Run PLINK 1.9 pairwise epistasis on a prepared benchmark input directory."""
import argparse, subprocess
from pathlib import Path

def run(cmd):
    print('+', ' '.join(map(str,cmd)), flush=True)
    subprocess.run(list(map(str,cmd)), check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--plink', default='plink')
    ap.add_argument('--threads', type=int, default=8)
    args=ap.parse_args()
    inp=Path(args.input_dir); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    prefix=inp/'benchmark_epigen'
    if not prefix.with_suffix('.bed').exists():
        run([args.plink,'--file',inp/'plink_text','--make-bed','--allow-no-sex','--out',prefix])
    run([args.plink,'--bfile',prefix,'--pheno',inp/'pheno_plink.tsv','--pheno-name','PHENO','--allow-no-sex','--epistasis','--epi1','1','--threads',args.threads,'--out',out/'plink_epistasis'])

if __name__=='__main__': main()
