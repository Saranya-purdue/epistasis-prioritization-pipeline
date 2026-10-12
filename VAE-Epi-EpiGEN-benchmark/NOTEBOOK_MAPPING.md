# Mapping from analysis notebook to repository scripts

The public repository is a refactor of `Epigen_simulations-for_github.ipynb`. Cluster submission cells and institution-specific path cells were deliberately omitted. Analysis logic was retained in command-line scripts.

| Notebook cell(s) | Repository file | Purpose |
|---|---|---|
| 1 | `scripts/02_simulate_epigen.py` | EpiGEN simulation for four architectures |
| 3 | `scripts/01_calibrate_epigen_alpha.py` | Calibrate alpha to target epistatic variance fraction |
| 8, 15, 17, 18, 35 | `scripts/06_standardize_phenotype_and_truth.py` | Consolidated phenotype/truth standardization logic |
| 23 | `scripts/04_run_matrixepistasis.R` | MatrixEpistasis runner |
| 31 | `scripts/03_build_epigen_density_benchmarks.py` | Build 50/100-pair EpiGEN conditions |
| 49 | `scripts/05_compare_benchmark_a.py` | Benchmark A recovery metrics and comparison |
| 60 | `scripts/benchmark_b/01_build_nested_benchmark_genotypes.py` | Nested 5k/10k/20k/30k genotype sets |
| 97 | `scripts/benchmark_b/02_export_latent_and_perlatent_shap.py` | Export latent representation and per-latent SNP attribution |
| 104 | `scripts/benchmark_b/03_weighted_vae_select_2000.py` | Phenotype-weighted VAE SNP selection |
| 111 | `scripts/benchmark_b/04_check_weighted_truth_retention.py` | Weighted-VAE causal SNP / true-pair retention |
| 114 | `scripts/benchmark_b/05_random_2000_retention.py` | Random 2k retention control |
| 156 | `scripts/benchmark_b/06_summarize_benchmark_b.py` | End-to-end Benchmark B metrics |
| 157 | `scripts/benchmark_b/07_plot_benchmark_b.py` | Main/supplementary Benchmark B plots |
| 161 | `scripts/benchmark_b/08_plot_architecture_metrics.py` | Architecture-stratified Benchmark B plots |

## Notebook-referenced code not embedded in the notebook

The notebook invokes the following exact scripts but does not contain their source. They must be copied from the main VAE-Epi codebase before the repository can reproduce the complete manuscript workflow end to end:

- `00_prepare_sim1_inputs.py`
- `02_run_tree_ml_baselines.py`
- `03_run_rf_shap_pair_recovery.py`
- `03b_run_rfshap_taylor_pair_recovery.py`
- `vae_shap_ld_cpu.py`

No substitute implementation was fabricated during this refactor.
