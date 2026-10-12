# VAE-Epi: EpiGEN benchmarking code

This repository is a portable, publication-oriented version of the EpiGEN benchmarking workflow used to evaluate VAE-Epi. It was refactored from the analysis notebook into command-line scripts with **no institution-specific paths, cluster names, SLURM directives, or module commands**.

## What is included

The repository covers the two EpiGEN analyses used in the manuscript:

- **Benchmark A — detector comparison in a common 2,000-SNP search space.** Four EpiGEN interaction architectures (joint dominant, joint recessive, multiplicative, exponential) are evaluated with 10, 50, and 100 true interaction pairs. The supplied scripts calibrate the interaction parameter, generate the denser simulations, run MatrixEpistasis/PLINK adapters, and summarize method-level recovery metrics.
- **Benchmark B — end-to-end search-space compression.** A nested 5k/10k/20k/30k benchmark genotype set is generated, the 10k set is passed through the VAE/phenotype-weighted selection stage, and the resulting 2k SNP set is compared with the full 10k input. Scripts are included for nested-genotype construction, per-latent SNP attribution export, phenotype-weighted VAE selection, truth-retention checks, end-to-end metric aggregation, and manuscript figure generation.

## Important reproducibility note

The uploaded EpiGEN notebook **calls five VAE-Epi core scripts whose source code was not embedded in the notebook**. They are therefore not recreated here from guesswork:

1. `00_prepare_sim1_inputs.py`
2. `02_run_tree_ml_baselines.py`
3. `03_run_rf_shap_pair_recovery.py`
4. `03b_run_rfshap_taylor_pair_recovery.py`
5. `vae_shap_ld_cpu.py`

For a fully self-contained public repository, copy the exact versions of these five scripts from the main VAE-Epi analysis code into `external/vaee_epi_core/` (or pass their paths explicitly when following the full-run commands). The current repository intentionally preserves this distinction rather than silently replacing the manuscript code with newly invented implementations.

## Installation

### Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

For the VAE/SHAP stages, install the additional dependencies:

```bash
pip install -r requirements-vae.txt
```

### EpiGEN

Clone the EpiGEN repository separately:

```bash
git clone https://github.com/daisybio/epigen.git external/epigen
```

Scripts that use EpiGEN accept `--epigen-root external/epigen`.

### Optional external benchmark tools

For the complete detector comparison install:

- PLINK 1.9 (`plink` available on `PATH`)
- R and the `MatrixEpistasis` and `data.table` packages

```r
install.packages("data.table")
# Install MatrixEpistasis according to its package instructions.
```

Check the local environment with:

```bash
python scripts/check_dependencies.py \
  --epigen-root external/epigen \
  --core-dir external/vaee_epi_core
```

## Quick smoke test

The smoke test creates a small 0/1/2 genotype matrix and causal-pair file, then checks the calibration code. It does not reproduce manuscript-scale results.

```bash
python scripts/generate_demo_data.py --outdir demo_data
python scripts/01_calibrate_epigen_alpha.py \
  --genotype demo_data/genotype.csv \
  --pairs demo_data/causal_pairs.csv \
  --target 0.10 \
  --noise_sd 1.0 \
  --out demo_data/calibration
```

## Benchmark A

### 1. Generate/calibrate EpiGEN phenotypes

For an existing 2,000-SNP genotype matrix:

```bash
python scripts/02_simulate_epigen.py \
  --genotype data/genotype_2000.csv \
  --pairs data/causal_pairs_10.csv \
  --scenario all \
  --stdev 1.0 \
  --seed 2026 \
  --epigen-root external/epigen \
  --outdir results/benchmark_a/pairs_10
```

To construct the 50- and 100-pair density benchmarks with the same target epistatic variance fraction:

```bash
python scripts/03_build_epigen_density_benchmarks.py \
  --genotype data/genotype_2000.csv \
  --base_pairs_10 data/causal_pairs_10.csv \
  --outdir results/benchmark_a \
  --pair_counts 50 100 \
  --target 0.10 \
  --noise_sd 1.0 \
  --maf_min 0.20 \
  --maf_max 0.50 \
  --ld_r2_max 0.20 \
  --pair_seed 4242 \
  --noise_seed 2026 \
  --epigen-root external/epigen
```

### 2. Standardize phenotype/truth inputs

```bash
python scripts/06_standardize_phenotype_and_truth.py \
  --genotype data/genotype_2000.csv \
  --phenotype results/benchmark_a/pairs_50/multiplicative/multiplicative_phenotype.csv \
  --truth results/benchmark_a/pairs_50/multiplicative/multiplicative_true_pairs.csv \
  --phenotype-out work/pairs_50_multiplicative_phenotype.csv \
  --truth-out work/pairs_50_multiplicative_true_pairs.csv
```

The VAE-Epi core preparation script then creates the PLINK, MatrixEpistasis, and RF input formats. The exact core script used in the notebook should be supplied as `external/vaee_epi_core/00_prepare_sim1_inputs.py`.

### 3. Run detector methods

Portable adapters are supplied for PLINK and MatrixEpistasis:

```bash
python scripts/07_run_plink.py \
  --input-dir work/prepared_condition \
  --outdir results/benchmark_a/pairs_50/multiplicative/plink

Rscript scripts/04_run_matrixepistasis.R \
  work/prepared_condition \
  results/benchmark_a/pairs_50/multiplicative/matrixepistasis
```

Run RF-path, RF-SHAP and RF-SHAP-Taylor using the exact VAE-Epi core scripts listed above.

### 4. Summarize Benchmark A

The comparison script recognizes PLINK, MatrixEpistasis, RF-SHAP, RF-SHAP-Taylor, RF-path and ExtraTrees-path result filenames used in the notebook:

```bash
python scripts/05_compare_benchmark_a.py \
  --base results/benchmark_a
```

## Benchmark B

### 1. Build nested genotype sets

```bash
python scripts/benchmark_b/01_build_nested_benchmark_genotypes.py \
  --genotype data/full_genotype.csv \
  --snp-map data/snp_map.csv \
  --truth data/causal_pairs_10.csv data/causal_pairs_50.csv data/causal_pairs_100.csv \
  --outdir results/benchmark_b/data \
  --sizes 5000 10000 20000 30000 \
  --seed 2026
```

The nested sets deliberately contain the union of all causal SNPs plus one fixed random ordering of non-causal SNPs, so the smaller sets are exact subsets of the larger sets.

### 2. Run the VAE on the 10k set

Use the exact `vae_shap_ld_cpu.py` from the VAE-Epi core analysis. The benchmark scripts expect its standard cache/output structure (`cache/genotype_int8.npy`, `cache/sample_ids.txt`, `variant_metadata_with_MAF.csv`, trained weights, and `run_arguments.json`).

### 3. Export latent representations and SNP-to-latent attribution

```bash
python scripts/benchmark_b/02_export_latent_and_perlatent_shap.py \
  --vae-dir results/benchmark_b/vae/input_10000 \
  --vae-script external/vaee_epi_core/vae_shap_ld_cpu.py \
  --top-per-latent 100
```

### 4. Phenotype-weighted selection

For each EpiGEN phenotype:

```bash
python scripts/benchmark_b/03_weighted_vae_select_2000.py \
  --vae-dir results/benchmark_b/vae/input_10000 \
  --vae-script external/vaee_epi_core/vae_shap_ld_cpu.py \
  --phenotype-file data/phenotype.csv \
  --outdir results/benchmark_b/vae_weighted/pairs_50/multiplicative \
  --final-n 2000 \
  --seed 42
```

This reproduces the analysis logic used in the notebook: latent-to-phenotype RF importance is combined with SNP-to-latent attribution, followed by MAF/LD filtering.

### 5. Truth-retention controls

```bash
python scripts/benchmark_b/04_check_weighted_truth_retention.py \
  --base results/benchmark_b

python scripts/benchmark_b/05_random_2000_retention.py \
  --base results/benchmark_b
```

### 6. Summarize and plot Benchmark B

After the RF-SHAP/RF-SHAP-Taylor/PLINK/MatrixEpistasis result directories are populated:

```bash
python scripts/benchmark_b/06_summarize_benchmark_b.py \
  --base results/benchmark_b

python scripts/benchmark_b/07_plot_benchmark_b.py \
  --base results/benchmark_b
```

The master table written by the summarizer is `benchmark_B_summary/benchmarkB_master_metrics.csv`.

## Reproducibility conventions retained from the notebook

- Random seeds are exposed explicitly; manuscript defaults are retained in the CLI defaults where present.
- True-pair metrics use the full causal-pair denominator, so upstream VAE losses are counted as end-to-end false negatives.
- Benchmark A uses a common 2,000-SNP search space to separate detector performance from upstream SNP selection.
- Benchmark B reports both pre-RF true-pair availability and downstream ranking recovery, distinguishing selection, screening, and ranking losses.
- The 10k-to-2k search-space comparison corresponds to 49,995,000 versus 1,999,000 possible unordered SNP pairs.

## Repository map

```text
scripts/
  01_calibrate_epigen_alpha.py
  02_simulate_epigen.py
  03_build_epigen_density_benchmarks.py
  04_run_matrixepistasis.R
  05_compare_benchmark_a.py
  06_standardize_phenotype_and_truth.py
  07_run_plink.py
  check_dependencies.py
  generate_demo_data.py
  benchmark_b/
    01_build_nested_benchmark_genotypes.py
    02_export_latent_and_perlatent_shap.py
    03_weighted_vae_select_2000.py
    04_check_weighted_truth_retention.py
    05_random_2000_retention.py
    06_summarize_benchmark_b.py
    07_plot_benchmark_b.py
    08_plot_architecture_metrics.py
```

## Data availability

Large genotype matrices and manuscript-scale method outputs are intentionally not stored in Git. See `data/README.md` for the expected schemas. Do not commit participant-level genotype data to a public repository unless the data-use agreement explicitly permits redistribution.

## Provenance

`NOTEBOOK_MAPPING.md` records which sections of the supplied analysis notebook were converted into each repository script. This allows the public code to remain traceable to the analysis used for the manuscript.
