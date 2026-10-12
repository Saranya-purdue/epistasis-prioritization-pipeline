#!/usr/bin/env python3
from pathlib import Path
import argparse
import json, math, re, warnings
import numpy as np
import pandas as pd

_cli = argparse.ArgumentParser(description="Summarize Benchmark B outputs into manuscript-ready metrics.")
_cli.add_argument("--base", required=True, help="Benchmark B root directory.")
_cli_args = _cli.parse_args()
BENCHB = Path(_cli_args.base)
PAIR_COUNTS = [10, 50, 100]
SCENARIOS = ['joint_dominant','joint_recessive','multiplicative','exponential']
ARCH_LABELS = {
    'joint_dominant':'Joint dominant',
    'joint_recessive':'Joint recessive',
    'multiplicative':'Multiplicative',
    'exponential':'Exponential',
}
N_ORIGINAL_SNPS = 10_000
N_ORIGINAL_UNIVERSE = math.comb(N_ORIGINAL_SNPS, 2)
CHUNK_SIZE = 1_000_000
OUT = BENCHB / 'benchmark_B_summary'
CACHE = OUT / 'cache'
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

def canonical_pair(a,b):
    a,b=str(a),str(b)
    return (a,b) if a<=b else (b,a)

def as_bool(x):
    if pd.isna(x): return False
    if isinstance(x,(bool,np.bool_)): return bool(x)
    return str(x).strip().lower() in {'true','1','t','yes','y'}

def elapsed_to_seconds(text):
    parts=str(text).strip().split(':')
    try:
        if len(parts)==3: return 3600*float(parts[0])+60*float(parts[1])+float(parts[2])
        if len(parts)==2: return 60*float(parts[0])+float(parts[1])
        return float(parts[0])
    except Exception:
        return np.nan

def parse_runtime(method_file):
    parent=Path(method_file).parent
    candidates=[parent/'runtime.txt',parent/'runtime_full.txt',parent/'time.txt']+sorted(parent.glob('*runtime*.txt'))
    seen=set(); candidates=[p for p in candidates if not (str(p) in seen or seen.add(str(p)))]
    for p in candidates:
        if not p.exists(): continue
        txt=p.read_text(errors='ignore')
        m1=re.search(r'Elapsed \(wall clock\) time .*?:\s*([0-9:.]+)',txt)
        m2=re.search(r'Maximum resident set size \(kbytes\):\s*([0-9.]+)',txt)
        runtime=elapsed_to_seconds(m1.group(1)) if m1 else np.nan
        peak=float(m2.group(1))/1024.0 if m2 else np.nan
        if np.isfinite(runtime) or np.isfinite(peak): return runtime,peak,str(p)
    return np.nan,np.nan,None

def truth_path(n_pairs,scenario,branch='full'):
    if branch=='weighted':
        p=BENCHB/'inputs'/'vae_weighted'/f'pairs_{n_pairs}'/scenario/'true_pairs_canonical.tsv'
        if p.exists(): return p
    return BENCHB/'inputs'/'full_10000'/f'pairs_{n_pairs}'/scenario/'true_pairs_canonical.tsv'

def load_truth(n_pairs,scenario):
    f=truth_path(n_pairs,scenario,'full')
    df=pd.read_csv(f,sep='\t')
    truth_pairs={canonical_pair(a,b) for a,b in zip(df['snp1'],df['snp2'])}
    causal=set(df['snp1'].astype(str))|set(df['snp2'].astype(str))
    raw={}
    if {'SNP1','SNP2'}.issubset(df.columns):
        for rscol,rawcol in [('snp1','SNP1'),('snp2','SNP2')]:
            for rs,rv in zip(df[rscol],df[rawcol]): raw[str(rs)]=str(rv)
    return df,truth_pairs,causal,raw,f

def weighted_selection_metrics(n_pairs,scenario,truth_pairs,causal_snps):
    f=truth_path(n_pairs,scenario,'weighted')
    if not f.exists() or 'vae_weighted' not in str(f):
        return {'PreRF_Causal_SNP_coverage':np.nan,'N_true_pairs_after_preRF_selection':np.nan,'PreRF_Pair_availability':np.nan,'Weighted_truth_file':None}
    df=pd.read_csv(f,sep='\t')
    selected=set()
    if 'snp1_in_scan' in df.columns:
        selected|={str(s) for s,flag in zip(df['snp1'],df['snp1_in_scan']) if as_bool(flag)}
    if 'snp2_in_scan' in df.columns:
        selected|={str(s) for s,flag in zip(df['snp2'],df['snp2_in_scan']) if as_bool(flag)}
    snp_cov=len(selected & causal_snps)/len(causal_snps) if selected else np.nan
    if 'pair_testable' in df.columns:
        navail=int(sum(as_bool(x) for x in df['pair_testable']))
    else:
        navail=sum((a in selected and b in selected) for a,b in truth_pairs)
    return {'PreRF_Causal_SNP_coverage':snp_cov,'N_true_pairs_after_preRF_selection':navail,'PreRF_Pair_availability':navail/len(truth_pairs),'Weighted_truth_file':str(f)}

def metrics_from_true_ranks(rank_lookup,truth_pairs,causal_snps,n_ranked_pairs,n_scored_snps,causal_cov,n_available,universe=N_ORIGINAL_UNIVERSE):
    T=len(truth_pairs)
    ranks=np.array([rank_lookup.get(p,universe+1) for p in truth_pairs],dtype=float)
    found=sorted(r for r in ranks if r<=n_ranked_pairs)
    ap=sum(i/r for i,r in enumerate(found,start=1))/T if found else 0.0
    prevalence=T/universe
    out={
        'N_true_pairs':T,'N_causal_snps':len(causal_snps),'N_ranked_pairs':int(n_ranked_pairs),'N_scored_snps':int(n_scored_snps),
        'Causal_SNP_coverage':causal_cov,'N_true_pairs_available':int(n_available),'Pair_availability':n_available/T,
        'AP':ap,'Random_AP':prevalence,'Normalized_AP':(ap-prevalence)/(1-prevalence),
        'Median_true_pair_rank':float(np.median(ranks)),'Mean_true_pair_rank':float(np.mean(ranks)),
        'Median_rank_percentile':100*float(np.median(ranks))/universe,
    }
    for mult in [1,5,10]:
        K=T*mult; tp=int(np.sum(ranks<=K)); recall=tp/T; precision=tp/K
        f1=2*precision*recall/(precision+recall) if precision+recall>0 else 0.0
        cond=tp/n_available if n_available>0 else np.nan
        label='T' if mult==1 else f'{mult}T'
        out.update({f'K_{label}':K,f'TP_at_{label}':tp,f'Recall_at_{label}':recall,f'Precision_at_{label}':precision,f'F1_at_{label}':f1,f'Conditional_Recall_at_{label}':cond})
    return out

def summarize_rf(f,method,truth_pairs,causal_snps):
    score='rf_shap_interaction_score' if method=='RF-SHAP' else 'taylor_interaction_score'
    df=pd.read_csv(f,sep='\t',compression='infer')
    df[score]=pd.to_numeric(df[score],errors='coerce'); df=df.dropna(subset=[score]).copy()
    df['pair']=[canonical_pair(a,b) for a,b in zip(df['snp1'],df['snp2'])]
    df=df.sort_values(score,ascending=False,kind='mergesort').drop_duplicates('pair').reset_index(drop=True)
    ranks={p:i+1 for i,p in enumerate(df['pair']) if p in truth_pairs}
    scored=set(df['snp1'].astype(str))|set(df['snp2'].astype(str))
    navail=sum((a in scored and b in scored) for a,b in truth_pairs)
    cov=len(causal_snps & scored)/len(causal_snps)
    return metrics_from_true_ranks(ranks,truth_pairs,causal_snps,len(df),len(scored),cov,navail)

def summarize_matrix(f,n_pairs,scenario,truth_pairs,causal_snps):
    cdir=CACHE/f'pairs_{n_pairs}'/scenario; cdir.mkdir(parents=True,exist_ok=True)
    rc=cdir/'matrix_true_ranks.tsv'; mc=cdir/'matrix_meta.json'
    if rc.exists() and mc.exists():
        rdf=pd.read_csv(rc,sep='\t'); ranks={canonical_pair(a,b):int(r) for a,b,r in zip(rdf['snp1'],rdf['snp2'],rdf['rank'])}; nrows=int(json.loads(mc.read_text())['n_rows'])
    else:
        ranks={}; nrows=0
        for ch in pd.read_csv(f,sep='\t',compression='gzip',usecols=['snp1','snp2','rank'],chunksize=CHUNK_SIZE):
            nrows+=len(ch)
            sub=ch[ch['snp1'].astype(str).isin(causal_snps) & ch['snp2'].astype(str).isin(causal_snps)]
            for a,b,r in zip(sub['snp1'],sub['snp2'],sub['rank']):
                p=canonical_pair(a,b)
                if p in truth_pairs:
                    r=int(r)
                    if p not in ranks or r<ranks[p]: ranks[p]=r
        pd.DataFrame([{'snp1':a,'snp2':b,'rank':r} for (a,b),r in sorted(ranks.items())]).to_csv(rc,sep='\t',index=False)
        mc.write_text(json.dumps({'n_rows':nrows},indent=2))
    if len(ranks)!=len(truth_pairs): warnings.warn(f'Matrix {n_pairs}/{scenario}: {len(ranks)}/{len(truth_pairs)} truth ranks found')
    return metrics_from_true_ranks(ranks,truth_pairs,causal_snps,nrows,N_ORIGINAL_SNPS,1.0,len(truth_pairs))

def parse_truth_token(raw):
    p=str(raw).split(':')
    if len(p)<4: raise ValueError(raw)
    return p[0],int(p[-3]),p[-2].upper(),p[-1].upper()

def build_plink_id_map(map_file, genotype_file, raw_token, causal_snps):
    """
    Crosswalk EpiGEN causal rsIDs to the SNP IDs used by PLINK.

    Strategy
    --------
    The genotype CSV and PLINK MAP were created from the same ordered
    set of 10,000 genotype variants.

    Therefore:

        genotype variant column i
                <-->
        PLINK MAP row i

    We do NOT assume that every genotype column follows an rsID-based
    naming convention.

    For causal variants, raw_token provides the exact original genotype
    column name, e.g.:

        rs167785 -> rs167785:9026854:C:T

    That full token is then mapped by position to the SNP ID PLINK uses.
    """

    # --------------------------------------------------------
    # Read genotype header only
    # --------------------------------------------------------

    header = (
        pd.read_csv(
            genotype_file,
            nrows=0
        )
        .columns
        .astype(str)
        .tolist()
    )

    if len(header) < 2:
        raise RuntimeError(
            f"Genotype file has too few columns: {genotype_file}"
        )

    # First column = sample ID.
    # EVERYTHING after it is a genotype variant.
    variant_cols = header[1:]


    # --------------------------------------------------------
    # Read PLINK MAP
    # --------------------------------------------------------

    plink_map = pd.read_csv(
        map_file,
        sep=r"\s+",
        header=None,
        names=[
            "CHR",
            "SNP",
            "CM",
            "BP",
        ],
        dtype={
            "SNP": str
        },
    )


    # --------------------------------------------------------
    # Fundamental positional check
    # --------------------------------------------------------

    if len(variant_cols) != len(plink_map):

        raise RuntimeError(
            "Cannot build order-based PLINK crosswalk: "
            f"genotype has {len(variant_cols)} variant columns "
            f"but MAP has {len(plink_map)} rows. "
            f"Genotype={genotype_file}; "
            f"MAP={map_file}"
        )


    print(
        f"  Genotype variants: {len(variant_cols):,}"
    )

    print(
        f"  PLINK MAP variants: {len(plink_map):,}"
    )


    # --------------------------------------------------------
    # Exact genotype-token -> PLINK-ID crosswalk
    # --------------------------------------------------------

    token_to_plink = dict(
        zip(
            variant_cols,
            plink_map["SNP"].astype(str)
        )
    )


    # --------------------------------------------------------
    # Map causal rsIDs using exact truth token
    # --------------------------------------------------------

    mapping = {}

    missing_raw_token = []
    missing_genotype_token = []

    for rs in sorted(causal_snps):

        rs = str(rs)

        # We need the original full genotype token for this rsID.
        if rs not in raw_token:

            missing_raw_token.append(
                rs
            )

            continue


        truth_token = str(
            raw_token[rs]
        )


        # Exact truth token should correspond to an original
        # genotype column.
        if truth_token not in token_to_plink:

            missing_genotype_token.append(
                (
                    rs,
                    truth_token
                )
            )

            continue


        mapping[rs] = (
            token_to_plink[
                truth_token
            ]
        )


    # --------------------------------------------------------
    # Fail loudly if any causal variants could not be mapped
    # --------------------------------------------------------

    if missing_raw_token:

        raise RuntimeError(
            f"{len(missing_raw_token)} causal rsIDs do not have "
            "an original truth token in raw_token; "
            f"examples={missing_raw_token[:10]}"
        )


    if missing_genotype_token:

        raise RuntimeError(
            f"{len(missing_genotype_token)} causal truth variants "
            "were not found among the 10,000 genotype columns; "
            f"examples={missing_genotype_token[:10]}"
        )


    if len(mapping) != len(causal_snps):

        raise RuntimeError(
            "Incomplete PLINK causal-SNP crosswalk: "
            f"mapped {len(mapping)} of "
            f"{len(causal_snps)} causal SNPs."
        )


    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print(
        "  PLINK crosswalk: mapped",
        len(mapping),
        "causal rsIDs by genotype/MAP order"
    )


    for rs in sorted(mapping)[:5]:

        print(
            f"    {rs} "
            f"[{raw_token[rs]}] "
            f"-> {mapping[rs]}"
        )


    return mapping

def summarize_plink(f,map_file,genotype_file,n_pairs,scenario,truth_pairs,causal_snps,raw_token):
    cdir=CACHE/f'pairs_{n_pairs}'/scenario; cdir.mkdir(parents=True,exist_ok=True)
    rc=cdir/'plink_true_ranks.tsv'; mc=cdir/'plink_meta.json'
    if rc.exists() and mc.exists():
        rdf=pd.read_csv(rc,sep='\t'); ranks={canonical_pair(a,b):int(r) for a,b,r in zip(rdf['truth_snp1'],rdf['truth_snp2'],rdf['rank'])}; nrows=int(json.loads(mc.read_text())['n_rows'])
    else:
        rs2p=build_plink_id_map(map_file,genotype_file,raw_token,causal_snps)
        truth_plink={canonical_pair(rs2p[a],rs2p[b]):(a,b) for a,b in truth_pairs}
        causal_plink=set(rs2p.values()); all_stats=[]; true_stat={}; nrows=0
        for ch in pd.read_csv(f,sep=r'\s+',usecols=['SNP1','SNP2','STAT'],chunksize=CHUNK_SIZE):
            nrows+=len(ch)
            stat=pd.to_numeric(ch['STAT'],errors='coerce'); good=stat.notna()
            if good.any(): all_stats.append(stat.loc[good].to_numpy(dtype=np.float64,copy=True))
            sub=ch[ch['SNP1'].astype(str).isin(causal_plink) & ch['SNP2'].astype(str).isin(causal_plink)].copy()
            if len(sub):
                sub['STAT']=pd.to_numeric(sub['STAT'],errors='coerce'); sub=sub.dropna(subset=['STAT'])
                for a,b,s in zip(sub['SNP1'],sub['SNP2'],sub['STAT']):
                    pp=canonical_pair(a,b)
                    if pp in truth_plink:
                        tp=truth_plink[pp]; s=float(s)
                        if tp not in true_stat or s>true_stat[tp]: true_stat[tp]=s
        stats=np.concatenate(all_stats); stats.sort(); ranks={}
        for p,s in true_stat.items():
            right=np.searchsorted(stats,s,side='right'); ranks[p]=int(1+(len(stats)-right))
        del stats,all_stats
        pd.DataFrame([{'truth_snp1':a,'truth_snp2':b,'rank':r} for (a,b),r in sorted(ranks.items())]).to_csv(rc,sep='\t',index=False)
        mc.write_text(json.dumps({'n_rows':nrows},indent=2))
    if len(ranks)!=len(truth_pairs): warnings.warn(f'PLINK {n_pairs}/{scenario}: {len(ranks)}/{len(truth_pairs)} truth ranks found')
    return metrics_from_true_ranks(ranks,truth_pairs,causal_snps,nrows,N_ORIGINAL_SNPS,1.0,len(truth_pairs))

def full_files(n_pairs,scenario):
    root=BENCHB/'results'/'full_10000'/f'pairs_{n_pairs}'/scenario
    return {
        'RF-SHAP':root/'rfshap'/'rf_rfshap_pair_scores.tsv.gz',
        'RF-SHAP-Taylor':root/'rftaylor'/'rf_shap_guided_taylor_pair_scores.tsv.gz',
        'MatrixEpistasis':root/'matrixepistasis'/'matrixepistasis_all_pairs.tsv.gz',
        'PLINK':root/'plink'/'plink_epistasis.epi.qt',
    }

def weighted_files(n_pairs,scenario):
    root=BENCHB/'results'/'vae_weighted'/f'pairs_{n_pairs}'/scenario
    exp={
        'RF-SHAP':root/'rfshap'/'rf_rfshap_pair_scores.tsv.gz',
        'RF-SHAP-Taylor':root/'rftaylor'/'rf_shap_guided_taylor_pair_scores.tsv.gz',
    }
    out={}
    for method,p in exp.items():
        if p.exists(): out[method]=p; continue
        hits=[x for x in (BENCHB/'results').rglob(p.name) if 'vae_weighted' in str(x) and f'pairs_{n_pairs}' in str(x) and scenario in str(x)]
        out[method]=hits[0] if len(hits)==1 else p
    return out

rows=[]
for n_pairs in PAIR_COUNTS:
    for scenario in SCENARIOS:
        print('\n'+'='*72); print(f'{n_pairs} true pairs | {scenario}'); print('='*72)
        _,truth,causal,raw,truth_file=load_truth(n_pairs,scenario)
        for method,f in full_files(n_pairs,scenario).items():
            if not f.exists(): warnings.warn(f'MISSING {f}'); continue
            print(f'Full 10k -> {method}')
            if method in {'RF-SHAP','RF-SHAP-Taylor'}: m=summarize_rf(f,method,truth,causal)
            elif method=='MatrixEpistasis': m=summarize_matrix(f,n_pairs,scenario,truth,causal)
            else:
                in_dir=BENCHB/'inputs'/'full_10000'/f'pairs_{n_pairs}'/scenario
                map_file=in_dir/'plink_text.map'
                genotype_file=BENCHB/'data'/'genotype_benchmark_10000.csv'
                if not map_file.exists(): raise FileNotFoundError(map_file)
                if not genotype_file.exists(): raise FileNotFoundError(genotype_file)
                m=summarize_plink(f,map_file,genotype_file,n_pairs,scenario,truth,causal,raw)
            rt,mem,rtfile=parse_runtime(f)
            m.update({'Pair_count':n_pairs,'Scenario':scenario,'Architecture':ARCH_LABELS[scenario],'Branch':'Full 10k','Method':method,'Pipeline':f'Full 10k -> {method}','N_input_snps':10000,'Candidate_pair_space':math.comb(10000,2),'Pair_space_reduction_vs_10k':0.0,'PreRF_Causal_SNP_coverage':1.0,'N_true_pairs_after_preRF_selection':len(truth),'PreRF_Pair_availability':1.0,'Runtime_seconds':rt,'Peak_memory_MB':mem,'Runtime_file':rtfile,'Result_file':str(f),'Truth_file':str(truth_file)})
            rows.append(m)
        sel=weighted_selection_metrics(n_pairs,scenario,truth,causal)
        for method,f in weighted_files(n_pairs,scenario).items():
            if not f.exists(): warnings.warn(f'MISSING weighted {f}'); continue
            print(f'Weighted VAE 2k -> {method}')
            m=summarize_rf(f,method,truth,causal); rt,mem,rtfile=parse_runtime(f)
            m.update({'Pair_count':n_pairs,'Scenario':scenario,'Architecture':ARCH_LABELS[scenario],'Branch':'Weighted VAE 2k','Method':method,'Pipeline':f'Weighted VAE 2k -> {method}','N_input_snps':2000,'Candidate_pair_space':math.comb(2000,2),'Pair_space_reduction_vs_10k':1-math.comb(2000,2)/math.comb(10000,2),**sel,'Runtime_seconds':rt,'Peak_memory_MB':mem,'Runtime_file':rtfile,'Result_file':str(f),'Truth_file':str(truth_file)})
            rows.append(m)

master=pd.DataFrame(rows)
front=['Pair_count','Scenario','Architecture','Branch','Method','Pipeline','N_input_snps','Candidate_pair_space','Pair_space_reduction_vs_10k','N_true_pairs','N_causal_snps','PreRF_Causal_SNP_coverage','N_true_pairs_after_preRF_selection','PreRF_Pair_availability','N_scored_snps','Causal_SNP_coverage','N_true_pairs_available','Pair_availability','TP_at_T','Recall_at_T','TP_at_5T','Recall_at_5T','TP_at_10T','Recall_at_10T','Conditional_Recall_at_T','Conditional_Recall_at_5T','Conditional_Recall_at_10T','AP','Random_AP','Normalized_AP','Median_true_pair_rank','Mean_true_pair_rank','Median_rank_percentile','Runtime_seconds','Peak_memory_MB']
rest=[c for c in master.columns if c not in front]; master=master[[c for c in front if c in master.columns]+rest]
master.to_csv(OUT/'benchmarkB_master_metrics.csv',index=False)
primary_cols=['Pair_count','Architecture','Pipeline','PreRF_Pair_availability','Pair_availability','Recall_at_T','Recall_at_5T','Recall_at_10T','Conditional_Recall_at_5T','Normalized_AP','Median_rank_percentile','Runtime_seconds','Peak_memory_MB','N_input_snps','Candidate_pair_space','Pair_space_reduction_vs_10k']
primary=master[primary_cols].copy()
primary.to_csv(OUT/'benchmarkB_primary_comparison.csv',index=False)
funnel=master[master['Method'].isin(['RF-SHAP','RF-SHAP-Taylor'])][['Pair_count','Architecture','Pipeline','N_true_pairs','N_true_pairs_after_preRF_selection','PreRF_Pair_availability','N_true_pairs_available','Pair_availability','TP_at_T','Recall_at_T','TP_at_5T','Recall_at_5T','TP_at_10T','Recall_at_10T','Conditional_Recall_at_5T']].copy()
funnel.to_csv(OUT/'benchmarkB_funnel.csv',index=False)
runtime=master[['Pair_count','Architecture','Pipeline','N_input_snps','Candidate_pair_space','N_ranked_pairs','Runtime_seconds','Peak_memory_MB','Result_file','Runtime_file']].copy()
runtime.to_csv(OUT/'benchmarkB_runtime_memory.csv',index=False)
desc=(master.groupby(['Pair_count','Pipeline'],as_index=False).agg(Mean_PreRF_Pair_availability=('PreRF_Pair_availability','mean'),Mean_Pair_availability=('Pair_availability','mean'),Mean_Recall_T=('Recall_at_T','mean'),Mean_Recall_5T=('Recall_at_5T','mean'),Mean_Recall_10T=('Recall_at_10T','mean'),Mean_Normalized_AP=('Normalized_AP','mean'),Mean_Runtime_seconds=('Runtime_seconds','mean'),Mean_Peak_memory_MB=('Peak_memory_MB','mean')))
desc.to_csv(OUT/'benchmarkB_descriptive_mean_across_architectures.csv',index=False)
print('\n'+'='*72); print('Benchmark B summary complete'); print('='*72); print(OUT); print(f'Rows: {len(master)}'); print(primary.to_string(index=False))
