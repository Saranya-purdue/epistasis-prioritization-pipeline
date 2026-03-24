# epistasis-prioritization-pipeline
Two-stage framework for prioritizing epistatic SNP-SNP interactions using VAE, RF-SHAP, and a 3P evidence rule.

# Two-Stage Framework for Epistasis SNP-SNP Prioritization

This repository contains code for a two-stage framework to prioritize epistatic SNP-SNP interactions without exhaustive enumeration.

In the first stage, chromosome-wise variational autoencoders (VAEs) are used to learn low-dimensional genotype representations independent of BMI. SHAP attributions from a supervised latent-prediction model are then used to map latent factors back to SNPs, generating compact chromosome-specific candidate SNP sets.

In the second stage, Random Forest models are used to predict BMI under cross-validation, and SHAP interaction values are used to rank candidate SNP pairs. To reduce false positives, candidate interactions are further evaluated using a Predictive-Marginal-Parametric (3P) rule that integrates machine learning-derived interaction strength, marginal GWAS support for at least one SNP, and significance of a linear SNP×SNP interaction term.


The full repository will include the following modules:

- `simulation/`
- `vae_latent_preparation/`
- `random_forest_shap/`
- `three_p_rule/`
- `variance_computation/`
- `deviation_from_additivity/`
- `fuma_analysis/`
- `pathway_enrichment/`

