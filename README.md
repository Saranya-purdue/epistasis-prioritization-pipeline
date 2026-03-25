# Exploring epistasis in human genome-wide data using a two-stage representation-learning framework

This repository contains code for a two-stage framework to prioritize epistatic SNP-SNP interactions without exhaustive enumeration.

In the first stage, chromosome-wise variational autoencoders (VAEs) learn low-dimensional genotype representations independent of BMI. SHAP attributions from a supervised latent-prediction model are then used to map latent factors back to SNPs, producing compact chromosome-specific candidate SNP sets.

In the second stage, Random Forest models are used to predict BMI and rank SNP-SNP interactions using SHAP interaction values. Candidate interactions are then evaluated using a Predictive-Marginal-Parametric (3P) rule that integrates:

- machine learning-derived interaction strength,
- marginal GWAS support for at least one SNP,
- significance of a linear SNP×SNP interaction term.

The repository also includes downstream analyses for variance contribution, deviation from additivity, FUMA-based gene mapping, and pathway enrichment.

## Current status

This repository is being organized for manuscript submission and reproducible release.

The codebase is structured into separate modules so that each stage of the pipeline can be inspected, reused, and reviewed independently.

## Repository structure

```text
├── README.md
├── simulation/
│   ├── README.md
│   ├── simulate1.py
│   ├── run_simulation1.py
│   ├── simulate2.py
│   └── run_simulation2.py
├── vae_latent_preparation/
│   ├── README.md
│   ├── vae_latent_preparation.py
│   └── latent_shap_ranking.py
├── random_forest_shap/
│   ├── README.md
│   └── rf_shap_interactions.py
├── gwas/
│   ├── README.md
│   └── run_gwas_plink.sh
├── three_p_rule/
│   ├── README.md
│   └── three_p_rule_pipeline.py
├── variance_computation/
│   ├── README.md
│   └── variance_computation.py
├── deviation_from_additivity/
│   ├── README.md
│   └── deviation_from_additivity.py
├── fuma_analysis/
│   ├── README.md
│   └── fuma_analysis.py
└── pathway_enrichment/
    ├── README.md
    └── pathway_enrichment_analysis.py
```
**Pipeline overview**
1. Simulation

- The simulation/ module contains phenotype simulation settings used to evaluate methods under controlled additive and epistatic architectures.

- It includes:

    - Simulation 1: overlapping additive and epistatic SNP sets with all pairwise interactions
    - Simulation 2: Sparse Epistatic Architecture with Controlled Interaction Complexity
2. VAE latent space preparation

- The vae_latent_preparation/ module trains a variational autoencoder on genotype or feature matrices and exports latent representations.

- It also includes a second script for ranking original SNPs/features per latent dimension using Random Forest and SHAP.

3. Random Forest with SHAP interaction ranking

- The random_forest_shap/ module performs:

    - SNP selection from upstream candidate lists
    - LD pruning
    - phenotype residualization using covariates
    - Random Forest tuning and SNP ranking by SHAP importance
    - SHAP interaction value computation for SNP-SNP pairs
    - candidate interaction filtering
4. GWAS

- The gwas/ module contains the PLINK2 shell workflow for marginal single-SNP association testing.

- These GWAS results provide the marginal evidence used later in the 3P rule.

5. 3P rule classification

- The three_p_rule/ module integrates:

    - P1: predictive support from RF + SHAP interaction strength
    - P2: marginal GWAS support
    - P3: parametric interaction support from a linear SNP×SNP interaction model

- Pairs are filtered and classified into evidence-based categories such as:

    - ML + GWAS + INT
    - ML + GWAS
    - ML + INT
6. Variance computation

- The variance_computation/ module quantifies the incremental variance explained by SNP-SNP interaction terms beyond main effects and covariates using cross-validated ΔR².

- The same analysis is applied across the retained 3P classes.

7. Deviation from additivity

- The deviation_from_additivity/ module visualizes how selected SNP pairs deviate from an additive expectation by comparing observed genotype-cell means to predictions from an additive-only model.

8. FUMA analysis

- The fuma_analysis/ module prepares lead SNP files for FUMA SNP2GENE and annotates prioritized SNP pairs with nearby-gene information from FUMA output.

9. Pathway enrichment analysis

- The pathway_enrichment/ module performs gene-level downstream analysis using:

    - Enrichr pathway enrichment
    - STRING protein-protein interaction analysis
    - optional GWAS gene overlap

- The same workflow is applied across the three 3P classes and across intra- and inter-chromosomal pair sets.

**Requirements**

Different modules use different dependencies. Most Python modules rely on combinations of:

- numpy
- pandas
- matplotlib
- scikit-learn
- statsmodels
- tensorflow
- umap-learn
- shap
- joblib
- requests
- tqdm

The GWAS step requires:

- plink2

Each module has its own README.md with more specific installation and usage details.

**General workflow**

A typical analysis flow is:

- run simulation studies if needed
- train VAE and extract latent representations
- rank SNPs and SNP-SNP interactions using Random Forest + SHAP
- run marginal GWAS with PLINK2
- apply the 3P rule to combine predictive, marginal, and parametric evidence
- quantify variance explained by interaction terms
- visualize deviation from additivity for selected SNP pairs
- annotate prioritized SNPs with FUMA
- run pathway enrichment and interaction-network analysis

**Reproducibility notes**

- The repository is organized so each module can be run independently with explicit file inputs and outputs.
- Each module has its own README.md describing:
    - required inputs
    - command-line usage
    - outputs
    - assumptions and notes

**Notes**

This repository is intended for research and manuscript reproducibility.
Some steps depend on external resources or tools, such as:
- PLINK2
- FUMA SNP2GENE
- Enrichr
- STRING

Upstream genotype QC and filtering are assumed to have been completed before GWAS and downstream prioritization steps.

**Citation**

If you use this code, please cite our manuscript.
