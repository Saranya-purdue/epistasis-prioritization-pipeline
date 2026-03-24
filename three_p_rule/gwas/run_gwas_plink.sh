#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User inputs
# -----------------------------
PLINK2_BIN="/path/bin/plink2"
BFILE_PREFIX="/path/10_final_qc"

PHENO_FILE="/path/to/bmi_pheno_for_plink.txt"
PHENO_NAME="BMI"

COVAR_FILE="/path/to/bmi_covariates_for_plink.txt"
COVAR_NAMES="age_at_bmi,sex_M,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10"

OUT_PREFIX="/path/to/gwas_bmi_full"

# -----------------------------
# Run GWAS
# -----------------------------
"${PLINK2_BIN}" \
  --bfile "${BFILE_PREFIX}" \
  --pheno "${PHENO_FILE}" \
  --pheno-name "${PHENO_NAME}" \
  --covar "${COVAR_FILE}" \
  --covar-name "${COVAR_NAMES}" \
  --covar-variance-standardize \
  --glm hide-covar \
  --out "${OUT_PREFIX}"

echo "GWAS finished. Output prefix: ${OUT_PREFIX}"
