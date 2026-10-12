#!/usr/bin/env python3
"""Standardize an EpiGEN phenotype and truth file for downstream benchmark tools.

This consolidates the repeated preparation logic used in the analysis notebook.
The phenotype is aligned to genotype rows by explicit IID when possible; otherwise
row-order alignment is used, matching the original analysis logic.
"""
import argparse
from pathlib import Path
import pandas as pd


def clean_ids(x):
    return (x.astype(str).str.strip().str.replace(r"\.0$", "", regex=True))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--genotype', required=True)
    ap.add_argument('--phenotype', required=True)
    ap.add_argument('--truth', required=True)
    ap.add_argument('--phenotype-out', required=True)
    ap.add_argument('--truth-out', required=True)
    args=ap.parse_args()

    geno_ids=clean_ids(pd.read_csv(args.genotype, usecols=[0]).iloc[:,0])
    pheno=pd.read_csv(args.phenotype)
    if len(pheno)!=len(geno_ids):
        raise ValueError(f'Phenotype N={len(pheno)} != genotype N={len(geno_ids)}')
    if 'Phenotype' not in pheno.columns:
        raise ValueError("Phenotype file must contain a 'Phenotype' column")

    use_row_order=True
    if 'IID' in pheno.columns:
        iid=clean_ids(pheno['IID'])
        if iid.nunique()==len(iid) and set(iid)==set(geno_ids):
            pheno=pheno.assign(_iid=iid).set_index('_iid').loc[geno_ids].reset_index(drop=True)
            use_row_order=False
    if use_row_order:
        print('IID does not match genotype IDs; using row-order alignment.')

    y=pd.to_numeric(pheno['Phenotype'], errors='coerce')
    if y.isna().any():
        raise ValueError('Missing/non-numeric phenotype values detected.')
    y_z=(y-y.mean())/y.std(ddof=0)
    out=pd.DataFrame({'SNP':geno_ids.values,'Phenotype_raw':y.values,'Phenotype':y_z.values})
    if 'Epistatic_Component' in pheno.columns:
        out['Epistatic_Component']=pd.to_numeric(pheno['Epistatic_Component'], errors='coerce').values
    Path(args.phenotype_out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.phenotype_out,index=False)

    pairs=pd.read_csv(args.truth)
    if not {'SNP1','SNP2'}.issubset(pairs.columns):
        raise ValueError('Truth file must contain SNP1 and SNP2')
    pairs=pairs.copy()
    pairs['SNP_Pair']=pairs['SNP1'].astype(str).str.strip()+'x'+pairs['SNP2'].astype(str).str.strip()
    pairs=pairs[['SNP_Pair']+[c for c in pairs.columns if c!='SNP_Pair']]
    Path(args.truth_out).parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.truth_out,index=False)
    print(f'Saved phenotype: {args.phenotype_out}')
    print(f'Saved truth: {args.truth_out}')

if __name__=='__main__':
    main()
