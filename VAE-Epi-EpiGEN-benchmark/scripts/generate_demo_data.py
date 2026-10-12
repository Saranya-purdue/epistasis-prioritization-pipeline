#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir', default='demo_data')
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--snps', type=int, default=100)
    ap.add_argument('--pairs', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2026)
    args=ap.parse_args()
    if 2*args.pairs>args.snps: raise ValueError('Need at least 2 SNPs per causal pair')
    rng=np.random.default_rng(args.seed)
    mafs=rng.uniform(.20,.45,args.snps)
    G=np.column_stack([rng.binomial(2,p,args.n) for p in mafs])
    cols=[f'rs{100000+i}' for i in range(args.snps)]
    df=pd.DataFrame(G,columns=cols)
    df.insert(0,'IID',[f'S{i+1}' for i in range(args.n)])
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'genotype.csv',index=False)
    pairs=pd.DataFrame({'Pair_ID':[f'PAIR_{i+1}' for i in range(args.pairs)],'SNP1':cols[0:2*args.pairs:2],'SNP2':cols[1:2*args.pairs:2]})
    pairs.to_csv(out/'causal_pairs.csv',index=False)
    pd.DataFrame({'SNP':cols,'CHR':1,'BP':np.arange(1,args.snps+1)*1000,'VARIANT_ID':cols}).to_csv(out/'snp_map.csv',index=False)
    print(out.resolve())

if __name__=='__main__': main()
