#!/usr/bin/env python3
import argparse, shutil
from pathlib import Path

REQUIRED_CORE = [
    '00_prepare_sim1_inputs.py',
    '02_run_tree_ml_baselines.py',
    '03_run_rf_shap_pair_recovery.py',
    '03b_run_rfshap_taylor_pair_recovery.py',
    'vae_shap_ld_cpu.py',
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--core-dir', default=None, help='Directory containing the VAE-Epi core scripts referenced by the benchmark notebook.')
    ap.add_argument('--epigen-root', default=None)
    args=ap.parse_args()
    print('Executables:')
    for x in ['python','Rscript','plink']:
        print(f'  {x}:', shutil.which(x) or 'NOT FOUND')
    if args.core_dir:
        d=Path(args.core_dir)
        print('\nVAE-Epi core scripts:')
        for f in REQUIRED_CORE:
            hits=list(d.rglob(f))
            print(f'  {f}:', hits[0] if hits else 'NOT FOUND')
    if args.epigen_root:
        p=Path(args.epigen_root)/'utils'/'parametrized_model.py'
        print('\nEpiGEN ParametrizedModel:', p if p.exists() else 'NOT FOUND')

if __name__=='__main__': main()
