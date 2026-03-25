# GWAS with PLINK2

This module contains the PLINK2 command-line workflow used to run marginal single-SNP genome-wide association analysis (GWAS) for BMI after genotype quality control.

The output from this step is used downstream in the 3P rule pipeline, where marginal GWAS evidence is mapped onto SNPs participating in candidate SNP-SNP interaction pairs.

## Module structure

```text
gwas/
├── README.md
└── run_gwas_plink.sh
```
**Overview**

This module runs marginal GWAS using PLINK2 with:

- genotype data in PLINK binary format
- a phenotype file containing BMI
- a covariate file containing age, sex, and principal components

The PLINK2 --glm output is then used in downstream analyses to provide marginal GWAS support for candidate SNP pairs.

**Input files**

The shell script expects:

- a PLINK binary genotype prefix:
  - .bed
  - .bim
  - .fam
- a phenotype file containing:
  - sample IDs
  - phenotype column (default: BMI)

- a covariate file containing:
  - sample IDs
  - covariates such as:
  - age_at_bmi
  - sex_M
  - PC1 to PC10

**Requirements**

 - plink2

**How to run**

Edit the paths inside run_gwas_plink.sh, then run:

```text
bash run_gwas_plink.sh
```

The script uses a command of the following form:
```text
plink2 \
  --bfile path/to/genotype_prefix \
  --pheno path/to/bmi_pheno_for_plink.txt \
  --pheno-name BMI \
  --covar path/to/bmi_covariates_for_plink.txt \
  --covar-name age_at_bmi,sex_M,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10 \
  --covar-variance-standardize \
  --glm hide-covar \
  --out path/to/gwas_bmi_full
```
**Main output**

PLINK2 writes output files with the chosen output prefix.

The most important downstream file is typically:
    - gwas_bmi_full.BMI.glm.linear

This file contains:

- SNP ID
- effect estimate
- standard error
- p-value
- test type

The downstream 3P rule pipeline uses the additive test results from this file.

**Notes**
- MAF filtering and other genotype QC steps should be completed before this GWAS step.
- This module performs only marginal GWAS, not interaction testing.
- The resulting GWAS p-values are mapped later onto SNPs within candidate SNP-SNP pairs.

**Citation**
If you use this code, please cite our manuscript.
