# Variance Computation for Candidate SNP-SNP Interaction Pairs

This module computes the incremental variance explained by SNP-SNP interaction terms using cross-validated linear regression models.

The goal is to quantify how much predictive variance is added by the interaction term beyond the corresponding main effects and covariates.

This same variance-computation framework is applied to all SNP pairs retained by the 3P rule, and results can be summarized across the three 3P support classes:

- `ML + GWAS + INT`
- `ML + GWAS`
- `ML + INT`

## Module structure

```text
variance_computation/
├── README.md
└── variance_computation.py
```
**Overview**

For each candidate SNP pair, the script compares two models:

- Base model
  - phenotype ~ covariates + SNP1 + SNP2
- Full model
  - phenotype ~ covariates + SNP1 + SNP2 + SNP1×SNP2

For each fold in 5-fold cross-validation, the script computes:

  - ΔR² = R²(full) - R²(base)

The final per-pair result is summarized by:

- mean ΔR² across folds
- standard deviation of ΔR² across folds
- mean base-model R²
- mean full-model R²

**Inputs**

The script expects:

- a candidate pair CSV file
- per-chromosome genotype CSV files
- a phenotype CSV file
- a covariate CSV file

**Requirements**

The code uses Python with:

- numpy
- pandas
- matplotlib
- scikit-learn
- tqdm

Install with:

```text
pip install numpy pandas matplotlib scikit-learn tqdm
```

**How to run**

```text
python variance_computation.py \
  --pairs_csv path/to/allchr_pairs_3P_filtered.csv \
  --genotype_dir path/to/genotype_ready \
  --phenotype_csv path/to/BMI_Whites_all.csv \
  --covariates_csv path/to/covariates_all.csv \
  --output_dir outputs/variance_computation
```

**Main processing steps**

1. Load candidate pairs

    - The script reads the candidate pair table and collects all unique SNPs required for analysis.

2. Load genotype subsets across chromosomes

    - Only the SNPs required by the candidate pair file are loaded from each chromosome genotype file.

3. Merge genotype, phenotype, and covariates

    - All datasets are merged by subject ID.

4. Compute per-pair cross-validated ΔR²

    For each pair:
    
    - fit base and full linear models
    - compute test-fold R² values
****calculate ΔR² for each fold
****summarize across folds
5. Summarize by 3P class

    - If the pair file contains a class column, the script produces class-level summaries so the same variance-computation approach is applied consistently across:

    - ML + GWAS + INT
    - ML + GWAS
    - ML + INT

**Output files**

```text

outputs/variance_computation/
├── pairwise_delta_r2.csv
├── delta_r2_by_class_summary.csv
├── delta_r2_pairs.png
├── delta_r2_distribution.png
├── delta_r2_by_class.png
└── variance_run_summary.json

```

**Output descriptions**

- pairwise_delta_r2.csv
- Per-pair incremental variance results, including:
- base_r2_mean
- full_r2_mean
- delta_r2_mean
- delta_r2_sd
- delta_r2_by_class_summary.csv
- Summary of ΔR² results grouped by 3P class
- delta_r2_pairs.png
- Per-pair error-bar plot of mean ΔR² and fold-to-fold variability
- delta_r2_distribution.png
- Distribution of mean per-pair ΔR² values
- delta_r2_by_class.png
- Class-wise plot of ΔR² values for the 3P support groups
- variance_run_summary.json
- Summary of sample counts, SNP counts, pair counts, and overall ΔR² statistics

**Notes**

- This module measures the contribution of an interaction term beyond main effects and covariates.
- The same computation is used for all retained 3P-rule classes so their interaction contributions can be compared on a common scale.
- The pair file can be the final output of the 3P rule pipeline.
- The script is chromosome-agnostic and loads the necessary SNPs automatically from the per-chromosome genotype files.

**Citation**

If you use this code, please cite our manuscript.

