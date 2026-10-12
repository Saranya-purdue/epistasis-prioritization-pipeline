#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(MatrixEpistasis)
    library(data.table)
})

# ============================================================
# Arguments
# ============================================================

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
    stop(
        "Usage: Rscript 01_run_matrixepistasis_epigen.R ",
        "<input_dir> <out_dir>"
    )
}

input_dir <- args[1]
out_dir <- args[2]

dir.create(
    out_dir,
    recursive = TRUE,
    showWarnings = FALSE
)

# ============================================================
# Input files
# ============================================================

geno_file_gz <- file.path(
    input_dir,
    "matrixepistasis_genotype.tsv.gz"
)

geno_file_tsv <- file.path(
    input_dir,
    "matrixepistasis_genotype.tsv"
)

if (file.exists(geno_file_gz)) {

    geno_file <- geno_file_gz

} else if (file.exists(geno_file_tsv)) {

    geno_file <- geno_file_tsv

} else {

    stop(
        "Could not find MatrixEpistasis genotype file in: ",
        input_dir
    )
}

pheno_file <- file.path(
    input_dir,
    "phenotype_for_tools.tsv"
)

if (!file.exists(pheno_file)) {
    stop(
        "Phenotype file missing: ",
        pheno_file
    )
}

# ============================================================
# Load
# ============================================================

cat("\nReading genotype:\n")
cat(geno_file, "\n")

geno <- fread(
    geno_file,
    data.table = FALSE
)

cat("\nReading phenotype:\n")
cat(pheno_file, "\n")

pheno <- fread(
    pheno_file,
    data.table = FALSE
)

cat("\nRaw genotype dimensions:\n")
cat(
    nrow(geno),
    "samples x",
    ncol(geno) - 1,
    "SNPs\n"
)

cat(
    "Phenotype samples:",
    nrow(pheno),
    "\n"
)

# ============================================================
# Validate columns
# ============================================================

if (!"sample_id" %in% colnames(geno)) {
    stop(
        "sample_id missing from genotype file."
    )
}

if (!"sample_id" %in% colnames(pheno)) {
    stop(
        "sample_id missing from phenotype file."
    )
}

if (!"phenotype" %in% colnames(pheno)) {
    stop(
        "phenotype missing from phenotype file."
    )
}

if (anyDuplicated(geno$sample_id)) {
    stop(
        "Duplicated sample IDs in genotype."
    )
}

if (anyDuplicated(pheno$sample_id)) {
    stop(
        "Duplicated sample IDs in phenotype."
    )
}

# ============================================================
# Match samples explicitly
# ============================================================

common_ids <- intersect(
    pheno$sample_id,
    geno$sample_id
)

cat(
    "Matched samples:",
    length(common_ids),
    "\n"
)

if (length(common_ids) == 0) {
    stop(
        "No matching genotype/phenotype sample IDs."
    )
}

# Preserve phenotype order
pheno <- pheno[
    pheno$sample_id %in% common_ids,
    ,
    drop = FALSE
]

geno_index <- match(
    pheno$sample_id,
    geno$sample_id
)

geno <- geno[
    geno_index,
    ,
    drop = FALSE
]

# Safety check
if (!all(
    pheno$sample_id == geno$sample_id
)) {
    stop(
        "Sample alignment failed."
    )
}

# ============================================================
# Phenotype
# ============================================================

trait <- as.numeric(
    pheno$phenotype
)

keep_sample <- is.finite(
    trait
)

geno <- geno[
    keep_sample,
    ,
    drop = FALSE
]

trait <- trait[
    keep_sample
]

# ============================================================
# Genotype matrix
# ============================================================

snp_cols <- setdiff(
    colnames(geno),
    "sample_id"
)

X <- as.matrix(
    geno[
        ,
        snp_cols,
        drop = FALSE
    ]
)

storage.mode(X) <- "numeric"

cat(
    "Samples after phenotype filtering:",
    nrow(X),
    "\n"
)

cat(
    "SNPs before filtering:",
    ncol(X),
    "\n"
)

# ============================================================
# Mean-impute missing genotypes
# ============================================================

for (j in seq_len(ncol(X))) {

    miss <- !is.finite(
        X[, j]
    )

    if (any(miss)) {

        observed_mean <- mean(
            X[, j],
            na.rm = TRUE
        )

        X[miss, j] <- observed_mean
    }
}

# ============================================================
# Remove zero-variance SNPs
# ============================================================

sds <- apply(
    X,
    2,
    sd
)

keep_snp <- (
    is.finite(sds) &
    sds > 0
)

removed <- sum(
    !keep_snp
)

X <- X[
    ,
    keep_snp,
    drop = FALSE
]

cat(
    "Zero-variance SNPs removed:",
    removed,
    "\n"
)

cat(
    "Final samples:",
    nrow(X),
    "\n"
)

cat(
    "Final SNPs:",
    ncol(X),
    "\n"
)

# ============================================================
# MatrixEpistasis
# ============================================================

cat("\nRunning MatrixEpistasis...\n")

res <- matrixEpistasis(
    snpA = X,
    snpB = X,
    trait = trait,
    covariate = NULL
)

r <- res$r
df <- res$df

cat(
    "Computing interaction p-values...\n"
)

p <- matrixPval(
    r,
    df
)

# ============================================================
# Keep unique pairs only
# ============================================================

upper <- which(
    upper.tri(p),
    arr.ind = TRUE
)

out <- data.frame(
    snp1 = colnames(X)[upper[, 1]],
    snp2 = colnames(X)[upper[, 2]],
    r = r[upper],
    pvalue = p[upper],
    stringsAsFactors = FALSE
)

# ============================================================
# Remove invalid results
# ============================================================

out <- out[
    is.finite(out$pvalue),
    ,
    drop = FALSE
]

# ============================================================
# Sort
# ============================================================

out <- out[
    order(out$pvalue),
    ,
    drop = FALSE
]

out$rank <- seq_len(
    nrow(out)
)

cat(
    "Number SNP-pair tests:",
    nrow(out),
    "\n"
)

# ============================================================
# Save all pairs
# ============================================================

fwrite(
    out,
    file.path(
        out_dir,
        "matrixepistasis_all_pairs.tsv.gz"
    ),
    sep = "\t"
)

# ============================================================
# Save top 100k
# ============================================================

fwrite(
    head(
        out,
        100000
    ),
    file.path(
        out_dir,
        "matrixepistasis_top100k.tsv"
    ),
    sep = "\t"
)

cat("\nMatrixEpistasis complete.\n")