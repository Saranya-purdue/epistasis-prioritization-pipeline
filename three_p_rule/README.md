# 3P Rule Classification for SNP-SNP Interaction Prioritization

This module contains the pipeline for integrating three sources of evidence to prioritize candidate epistatic SNP-SNP interactions:

- machine learning-derived interaction strength
- marginal GWAS support
- linear interaction model significance

The goal is to classify candidate SNP pairs into evidence-based categories using a Predictive-Marginal-Parametric (3P) rule.

## Module structure

```text
three_p_rule/
├── README.md
└── three_p_rule_pipeline.py
```
**Overview**

The 3P rule combines three components:

- P1: Predictive support
   - Interaction strength from the Random Forest + SHAP interaction pipeline
- P2: Marginal GWAS support
   - Evidence that at least one SNP in the pair shows marginal association in GWAS
- P3: Parametric interaction support
   - Evidence from a linear interaction model of the form:
       - phenotype ~ covariates + SNP1 + SNP2 + SNP1×SNP2

Candidate pairs are filtered and classified using these three sources of evidence.

**Inputs**

The pipeline expects:

- per-chromosome pair files from the Random Forest + SHAP interaction analysis
- per-chromosome genotype CSV files
- one phenotype CSV file
- one covariate CSV file
- one PLINK2 marginal GWAS .glm.linear file

**Requirements**

The code uses Python with:

- numpy
- pandas
- matplotlib
- statsmodels
- tqdm

**Install with:**

```text
pip install numpy pandas matplotlib statsmodels tqdm
```
**Main processing steps**

1. Load pair files
  - For each chromosome, the script loads the candidate SNP-pair file generated from the Random Forest + SHAP interaction analysis.

2. Fit linear interaction models and extracts:
  - beta_interact_lin
  - p_interact_lin
3. Map marginal GWAS evidence
  - Marginal GWAS effect sizes and p-values are mapped to both SNPs in each pair.
  - The script computes:
    - snp1_beta_gwas
    - snp1_p_gwas
    - snp2_beta_gwas
    - snp2_p_gwas
    - min_p_gwas
    - max_mlog10_p
4. Multiple-testing correction
  - The linear interaction p-values are corrected across all pairs using Benjamini-Hochberg FDR.
  - The script adds:
    - FDR_q
    - Significant
5. Apply the 3P rule
  - Pairs are classified using:
    - P1: high machine learning interaction strength
    - P2: marginal GWAS support
    - P3: linear interaction support

  - By default, the script keeps pairs that:
    - satisfy P1 and satisfy at least one additional support criterion
6. Assign support classes
  - The filtered pairs are classified into:
    - ML + GWAS + INT
    - ML + GWAS
    - ML + INT
    - ML only
  
**How to run**

```text
python three_p_rule_pipeline.py \
  --pairs_dir path/to/chr_results/clean_setup \
  --genotype_dir path/to/genotype_ready \
  --phenotype_csv path/to/BMI_Whites_all.csv \
  --covariates_csv path/to/covariates_all.csv \
  --gwas_glm_file path/to/gwas_bmi_full.BMI.glm.linear \
  --output_dir outputs/three_p_rule \
  --id_col SUBJID \
  --phenotype_col BMI \
  --gwas_mode min_p \
  --gwas_threshold 0.05 \
  --interaction_mode nominal_p \
  --interaction_threshold 0.05
```
**Important arguments**

--pairs_dir
  - Directory containing per-chromosome pair files
    
--pairs_pattern
  - Filename pattern for pair files
  - Default: chr{chrom}_allinteractions_NOLD.csv
    
--genotype_dir
  - Directory containing per-chromosome genotype files
    
--genotype_pattern
  - Filename pattern for genotype files
  - Default: chr{chrom}_ready.csv
    
--gwas_glm_file
  - PLINK2 .glm.linear output file
    
--gwas_mode
  - How GWAS support is defined:
  - min_p
  - mlog10

--gwas_threshold
  - GWAS evidence threshold

--interaction_mode
  - How linear interaction support is defined:
  - nominal_p
  - fdr_q

--interaction_threshold
  - Linear interaction support threshold

--ml_quantile
  - Quantile cutoff used to define high machine-learning interaction strength

--require_p1
  - Require predictive support (P1) for a pair to be retained

--min_total_support
  - Minimum number of support criteria required

**Output files**

```text
outputs/three_p_rule/
├── parsed_gwas_additive.csv
├── chr1_pairs_with_gwas_and_linear.csv
├── chr2_pairs_with_gwas_and_linear.csv
├── ...
├── allchr_pairs_with_gwas_and_linear.csv
├── allchr_pairs_3P_filtered.csv
├── linear_vs_rf_interaction_strength.png
├── three_p_class_counts.png
└── three_p_run_summary.json
```
**Output descriptions**

- parsed_gwas_additive.csv
- Additive GWAS results parsed from the PLINK2 .glm.linear file
- chr*_pairs_with_gwas_and_linear.csv
- Per-chromosome SNP pair tables with GWAS-mapped values and linear interaction results
- allchr_pairs_with_gwas_and_linear.csv
- Combined SNP pair table across all chromosomes
- allchr_pairs_3P_filtered.csv
- Final SNP pairs retained after applying the 3P rule
- linear_vs_rf_interaction_strength.png
- Scatter plot comparing absolute linear interaction effect size and RF+SHAP interaction strength
- three_p_class_counts.png
- Bar plot of final 3P evidence classes
- three_p_run_summary.json
- Summary of total pair counts, thresholds, and class counts

**Notes**

- This module assumes genotype QC were already completed before GWAS and interaction analysis.
- GWAS support is treated as marginal evidence for at least one SNP in a pair.
- The linear interaction model is used as a parametric validation step for candidate SNP-SNP interactions.
- The 3P rule is designed to reduce false positives by requiring agreement across multiple evidence sources.

**Citation**

If you use this code, please cite our manuscript.

