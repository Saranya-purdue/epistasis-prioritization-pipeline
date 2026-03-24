# Random Forest with SHAP Interaction Ranking

This module contains code for SNP prioritization using Random Forest regression, SHAP main-effect importance, and SHAP interaction values.

The workflow is designed to identify candidate SNP-SNP interactions for a given chromosome after an upstream candidate SNP selection step. It uses aggregated SNP rankings as input, merges them with phenotype and covariate data, residualizes the phenotype, ranks SNPs by SHAP importance, and then computes pairwise interaction strength using SHAP interaction values.

## Module structure

```text
random_forest_shap/
├── README.md
└── rf_shap_interactions.py
```
**Overview**

The pipeline performs the following steps:

1. Load aggregated candidate SNPs from an upstream feature-selection step.
2. Load genotype, phenotype, and covariate data for a chromosome.
3. Merge all data on the subject ID.
4. Residualize the phenotype using covariates.
5. LD-prune the selected SNP set.
6. Train and tune a Random Forest model for SNP main-effect ranking.
7. Compute SHAP and permutation importance for SNP main effects.
8. Select the top-ranked SNPs.
9. Train a second Random Forest model optimized for interaction detection.
10. Compute SHAP interaction values for SNP-SNP pairs.
11. Apply distance and pairwise LD filters.
12. Save top candidate interactions for downstream analysis.

**Input files**

The script expects the following inputs:

- aggregated_snps_csv
  - CSV file containing SNPs selected from the upstream latent-space or feature-ranking step.
- Must contain a SNP column.
- genotype_csv
  - Genotype matrix for one chromosome.
  - Rows = subjects, columns = SNPs.
- phenotype_csv
  - Phenotype table containing subject ID and phenotype value.
- covariates_csv
  - Covariate table containing subject ID and covariates such as age, sex, and principal components.

**Requirements**

The code uses Python with:

- numpy
- pandas
- matplotlib
- scikit-learn
- shap
- statsmodels

Install with:
```text
pip install numpy pandas matplotlib scikit-learn shap statsmodels
```
**Main processing steps**

 1. Data preparation    
    - loads the selected SNP list
    - keeps only SNPs present in the genotype matrix
    - merges genotype, phenotype, and covariates on the subject ID
    - removes rows with missing phenotype or covariate values
  
2. Phenotype residualization
    - The phenotype is residualized using an ordinary least squares model with the supplied covariates.

3. LD pruning
    - The SNP matrix is greedily pruned using pairwise correlation squared:
    - r^2 <= threshold, the default threshold is 0.7.

5. Main-effect Random Forest
    - A Random Forest regressor is tuned using randomized hyperparameter search and fitted to the residualized phenotype.

    The script then computes:
    - SHAP importance
    - permutation importance
    for all LD-pruned SNPs.

5. Interaction-focused Random Forest
    - The top-ranked SNPs from the SHAP main-effect step are selected, and a second Random Forest model is trained for interaction detection.
    - SHAP interaction values are then used to compute pairwise SNP-SNP interaction strength.

6. Candidate filtering
    - The interaction table can be filtered by:
    - minimum physical distance between SNPs
    - pairwise LD threshold between SNPs in a candidate pair
    - The top fraction of filtered pairs is saved as the final candidate set.

How to run

```text
python rf_shap_interactions.py \
  --aggregated_snps_csv path/to/aggregated_top_snps_chr16.csv \
  --genotype_csv path/to/chr16_ready.csv \
  --phenotype_csv path/to/BMI_chr16.csv \
  --covariates_csv path/to/covariates_chr16.csv \
  --output_dir outputs/chr16_rf_shap \
  --chromosome 16
```

Important arguments

--chromosome - Chromosome label used in output filenames.
  
--id_col - Subject ID column name. Default: SUBJID
  
--phenotype_col - Phenotype column name. Default: BMI
  
--aggregated_feature_col - Feature column in the aggregated SNP file. Default: Feature
  
--covariate_cols - Covariate columns used for residualization.
  
--ld_prune_r2 - SNP LD-pruning threshold. Default: 0.7
  
--top_k_snps - Number of top SNPs taken forward for interaction analysis. Default: 200
  
--shap_interaction_max_samples - Maximum number of samples used for SHAP interaction values. Default: 500
  
--min_distance_bp - Minimum physical distance between SNPs in a candidate pair. Default: 0
  
--pair_ld_threshold - Correlation threshold used to filter SNP pairs. Default: 0.70
  
--top_frac - Top fraction of filtered interaction pairs to save. Default: 0.01

**Output files**

Example output structure:
```text
outputs/chr16_rf_shap/
├── chr16_prunedX.csv
├── chr16_rf_importance_scores.csv
├── chr16_top_snps_shap.png
├── chr16_allinteractions.csv
├── chr16_top_interactions.png
├── chr16_allinteractions_filtered.csv
├── chr16_candidates_top0.010.csv
├── chr16_covariate_model_summary.json
├── chr16_run_summary.json
└── chr16_missing_selected_snps.csv
```

**Output descriptions**

- chr<CHR>_prunedX.csv 
    - LD-pruned SNP matrix used for main-effect Random Forest analysis.
  
- chr<CHR>_rf_importance_scores.csv
    - SNP importance table including SHAP importance and permutation importance.
  
- chr<CHR>_top_snps_shap.png
    - Bar plot of top SNPs ranked by SHAP importance.
  
- chr<CHR>_allinteractions.csv
    - Full SNP-SNP interaction table from SHAP interaction values.
  
- chr<CHR>_top_interactions.png
    - Plot showing top interaction pairs and interaction-vs-main-effect comparison.
  
- chr<CHR>_allinteractions_filtered.csv
    - Interaction table after distance and pairwise LD filtering.
  
- chr<CHR>_candidates_top<FRAC>.csv
    - Final top-ranked candidate interaction pairs.
  
- chr<CHR>_covariate_model_summary.json
    - Summary of the phenotype residualization model.
  
- chr<CHR>_run_summary.json
    - Summary of the full run, including sample counts, SNP counts, model R² values, and candidate counts.
  
- chr<CHR>_missing_selected_snps.csv
    - Selected SNPs from the aggregated feature file that were not found in the genotype matrix.

**Notes**

- This module is chromosome-agnostic and can be used for any chromosome by changing the input files and the --chromosome argument.
- SHAP interaction values can be computationally expensive for large SNP sets.
- If memory becomes limiting, reduce:
    - --top_k_snps
    - --shap_interaction_max_samples
- The distance filter assumes SNP names follow a chr:position or chromosome:position pattern if physical position filtering is used.
- The script uses SHAP importance for ranking SNPs before interaction analysis.

**Citation**

If you use this code, please cite our manuscript.
