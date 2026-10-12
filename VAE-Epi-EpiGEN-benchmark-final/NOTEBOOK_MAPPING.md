# Notebook-to-repository mapping

The public repository was refactored from `Epigen_simulations-for_github.ipynb`. Environment-specific submission cells and hard-coded storage paths were intentionally excluded. The analysis logic was retained in command-line scripts.

| Notebook cell(s) | Repository file | Purpose |
|---|---|---|
| 3 | `scripts/01_calibrate_epigen_alpha.py` | Calibrate alpha to a target epistatic variance fraction |
| 1 | `scripts/02_simulate_epigen.py` | EpiGEN simulation for the four interaction architectures |
| 31 | `scripts/03_build_epigen_density_benchmarks.py` | Build the 50/100-pair nested EpiGEN density benchmarks |
| 23 | `scripts/04_run_matrixepistasis.R` | MatrixEpistasis runner |
| 49 | `scripts/05_compare_benchmark_a.py` | Benchmark A recovery metrics and method comparison |
| 8, 15, 17, 18, 35 | `scripts/06_standardize_phenotype_and_truth.py` | Consolidated phenotype/truth standardization |
| 21, 38 | `scripts/07_run_plink.py` | Portable PLINK epistasis wrapper |
| 60 | `scripts/benchmark_b/01_build_nested_benchmark_genotypes.py` | Nested 5k/10k/20k/30k genotype sets |
| 97 | `scripts/benchmark_b/02_export_latent_and_perlatent_shap.py` | Export latent representation and per-latent SNP attribution |
| 104 | `scripts/benchmark_b/03_weighted_vae_select_2000.py` | Phenotype-weighted VAE SNP selection |
| 111 | `scripts/benchmark_b/04_check_weighted_truth_retention.py` | Weighted-VAE causal-SNP / true-pair retention |
| 114 | `scripts/benchmark_b/05_random_2000_retention.py` | Random 2k retention control |
| 156–157 | `scripts/benchmark_b/06_summarize_benchmark_b.py` | End-to-end Benchmark B metrics |
| 157–160 | `scripts/benchmark_b/07_plot_benchmark_b.py` | Main and supplementary Benchmark B plots |
| 161 | `scripts/benchmark_b/08_plot_architecture_metrics.py` | Architecture-stratified Benchmark B plots |

## Core scripts supplied separately from the notebook

The following exact analysis scripts were supplied separately and are now included under `scripts/core/`:

- `00_prepare_sim1_inputs.py`
- `02_run_tree_ml_baselines.py`
- `03_run_rf_shap_pair_recovery.py`
- `03b_run_rfshap_taylor_pair_recovery.py`
- `vae_shap_ld_cpu.py`

Only portability edits were made where needed; in particular, `vae_shap_ld_cpu.py` uses the optional `VAE_EPI_NUM_THREADS` environment variable rather than a scheduler-specific CPU-count variable.
