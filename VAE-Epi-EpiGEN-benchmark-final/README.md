# VAE-Epi: EpiGEN benchmarking code

This repository contains the analysis code used for the EpiGEN benchmarking of **VAE-Epi**. 
The repository covers two complementary benchmarks:

- **Benchmark A — detector comparison in a common 2,000-SNP search space.** Four EpiGEN interaction architectures (joint dominant, joint recessive, multiplicative, exponential) are evaluated with 10, 50, and 100 true interaction pairs.
- **Benchmark B — end-to-end prioritization benchmark.** A 10,000-SNP input is reduced to a phenotype-weighted VAE-prioritized 2,000-SNP set, followed by RF-SHAP / RF-SHAP-Taylor interaction screening. The full 10,000-SNP branch is retained as the comparison.

## Repository structure

```text
.
├── README.md
├── REPRODUCIBILITY_NOTES.md
├── requirements-core.txt
├── requirements-vae.txt
├── environment.yml
├── Makefile
├── data/
│   ├── README.md
│   └── example/
├── external/
│   └── README.md
├── scripts/
│   ├── 01_calibrate_epigen_alpha.py
│   ├── 02_simulate_epigen.py
│   ├── 03_build_epigen_density_benchmarks.py
│   ├── 04_run_matrixepistasis.R
│   ├── 05_compare_benchmark_a.py
│   ├── 06_standardize_phenotype_and_truth.py
│   ├── 07_run_plink.py
│   ├── check_dependencies.py
│   ├── generate_demo_data.py
│   ├── core/
│   │   ├── 00_prepare_sim1_inputs.py
│   │   ├── 02_run_tree_ml_baselines.py
│   │   ├── 03_run_rf_shap_pair_recovery.py
│   │   ├── 03b_run_rfshap_taylor_pair_recovery.py
│   │   └── vae_shap_ld_cpu.py
│   └── benchmark_b/
│       ├── 01_build_nested_benchmark_genotypes.py
│       ├── 02_export_latent_and_perlatent_shap.py
│       ├── 03_weighted_vae_select_2000.py
│       ├── 04_check_weighted_truth_retention.py
│       ├── 05_random_2000_retention.py
│       ├── 06_summarize_benchmark_b.py
│       ├── 07_plot_benchmark_b.py
│       └── 08_plot_architecture_metrics.py
└── tests/
```

## Installation

### Core Python environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

For the VAE stage:

```bash
pip install -r requirements-vae.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate vae-epi-epigen
```

### EpiGEN

The simulation scripts use EpiGEN's `ParametrizedModel`. Clone EpiGEN separately:

```bash
git clone https://github.com/daisybio/epigen.git external/epigen
```

Then pass:

```bash
--epigen-root external/epigen
```

to scripts that require EpiGEN.

### Optional detector software

The complete Benchmark A comparison also uses:

- PLINK 1.9
- R
- MatrixEpistasis
- data.table

Check the local environment with:

```bash
python scripts/check_dependencies.py --epigen-root external/epigen
```


# Benchmark A: common 2,000-SNP detector comparison

## A1. Calibrate interaction strength

```bash
python scripts/01_calibrate_epigen_alpha.py \
  --genotype data/genotype_2000.csv \
  --pairs data/causal_pairs_10.csv \
  --target 0.10 \
  --noise_sd 1.0 \
  --out results/benchmark_a/calibration_10
```

## A2. Generate the four EpiGEN architectures

```bash
python scripts/02_simulate_epigen.py \
  --genotype data/genotype_2000.csv \
  --pairs data/causal_pairs_10.csv \
  --scenario all \
  --alpha 1.2 \
  --stdev 1.0 \
  --seed 2026 \
  --outdir results/benchmark_a/pairs_10 \
  --epigen-root external/epigen
```

For manuscript reproduction, use the calibrated architecture-specific alpha values rather than a common illustrative alpha.

## A3. Build 50- and 100-pair density benchmarks

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

The generated pair sets are nested: the 50-pair truth is the first 50 pairs of the 100-pair truth.

## A4. Standardize one phenotype/truth condition

```bash
python scripts/06_standardize_phenotype_and_truth.py \
  --genotype data/genotype_2000.csv \
  --phenotype results/benchmark_a/pairs_50/multiplicative/multiplicative_phenotype.csv \
  --truth results/benchmark_a/pairs_50/multiplicative/multiplicative_true_pairs.csv \
  --phenotype-out work/pairs_50_multiplicative_phenotype.csv \
  --truth-out work/pairs_50_multiplicative_true_pairs.csv
```

## A5. Prepare a common detector input

```bash
python scripts/core/00_prepare_sim1_inputs.py \
  --geno_csv data/genotype_2000.csv \
  --pheno_csv work/pairs_50_multiplicative_phenotype.csv \
  --beta_epistatic_csv work/pairs_50_multiplicative_true_pairs.csv \
  --outdir work/pairs_50_multiplicative \
  --sample_col SNP \
  --phenotype_col Phenotype \
  --n_scan_snps 2000 \
  --seed 42
```

This creates the common genotype/phenotype/truth representation used by the detector scripts and PLINK/MatrixEpistasis adapters.

## A6. Run tree and SHAP-based detectors

RF-path:

```bash
python scripts/core/02_run_tree_ml_baselines.py \
  --input_dir work/pairs_50_multiplicative \
  --outdir results/benchmark_a/pairs_50/multiplicative/tree_ml_baselines/rf \
  --model rf \
  --n_estimators 200 \
  --max_depth 12 \
  --min_samples_leaf 20 \
  --n_jobs 8
```

ExtraTrees-path:

```bash
python scripts/core/02_run_tree_ml_baselines.py \
  --input_dir work/pairs_50_multiplicative \
  --outdir results/benchmark_a/pairs_50/multiplicative/tree_ml_baselines/extratrees \
  --model extratrees \
  --n_estimators 200 \
  --max_depth 12 \
  --min_samples_leaf 20 \
  --n_jobs 8
```

RF-SHAP:

```bash
python scripts/core/03_run_rf_shap_pair_recovery.py \
  --input_dir work/pairs_50_multiplicative \
  --outdir results/benchmark_a/pairs_50/multiplicative/rf_shap \
  --model rf \
  --n_estimators 200 \
  --max_depth 12 \
  --min_samples_leaf 20 \
  --n_jobs 8 \
  --top_m_snps 500 \
  --max_shap_main_samples 1000 \
  --max_shap_interaction_samples 500 \
  --seed 42
```

RF-SHAP-Taylor:

```bash
python scripts/core/03b_run_rfshap_taylor_pair_recovery.py \
  --input_dir work/pairs_50_multiplicative \
  --outdir results/benchmark_a/pairs_50/multiplicative/rf_shap_taylor \
  --n_estimators 200 \
  --max_depth 12 \
  --min_samples_leaf 20 \
  --n_jobs 8 \
  --top_m_snps 500 \
  --max_shap_main_samples 1000 \
  --max_taylor_samples 500 \
  --seed 42
```

## A7. Run PLINK and MatrixEpistasis

PLINK:

```bash
python scripts/07_run_plink.py \
  --input-dir work/pairs_50_multiplicative \
  --outdir results/benchmark_a/pairs_50/multiplicative/plink \
  --threads 8
```

MatrixEpistasis:

```bash
Rscript scripts/04_run_matrixepistasis.R \
  work/pairs_50_multiplicative \
  results/benchmark_a/pairs_50/multiplicative/matrixepistasis
```

## A8. Summarize Benchmark A

After the result folders for the 12 EpiGEN conditions are populated:

```bash
python scripts/05_compare_benchmark_a.py \
  --base results/benchmark_a
```

# Benchmark B: end-to-end 10k → weighted-VAE 2k benchmark

## B1. Build nested genotype sets

```bash
python scripts/benchmark_b/01_build_nested_benchmark_genotypes.py \
  --genotype data/full_genotype.csv \
  --snp-map data/snp_map.csv \
  --truth data/causal_pairs_10.csv data/causal_pairs_50.csv data/causal_pairs_100.csv \
  --outdir results/benchmark_b/data \
  --sizes 5000 10000 20000 30000 \
  --seed 2026
```

## B2. Run the VAE on the 10,000-SNP set

```bash
python scripts/core/vae_shap_ld_cpu.py \
  --csv results/benchmark_b/data/genotype_benchmark_10000.csv \
  --snp-map results/benchmark_b/data/benchmark_10000_snp_map.csv \
  --outdir results/benchmark_b/vae/input_10000 \
  --final-n 2000 \
  --seed 42
```

The VAE script can use the optional environment variable `VAE_EPI_NUM_THREADS` to control CPU thread count.

## B3. Export latent representations and per-latent SNP attribution

```bash
python scripts/benchmark_b/02_export_latent_and_perlatent_shap.py \
  --vae-dir results/benchmark_b/vae/input_10000 \
  --vae-script scripts/core/vae_shap_ld_cpu.py \
  --top-per-latent 100
```

## B4. Phenotype-weighted VAE selection

```bash
python scripts/benchmark_b/03_weighted_vae_select_2000.py \
  --vae-dir results/benchmark_b/vae/input_10000 \
  --vae-script scripts/core/vae_shap_ld_cpu.py \
  --phenotype-file data/phenotype.csv \
  --outdir results/benchmark_b/vae_weighted/pairs_50/multiplicative \
  --final-n 2000 \
  --seed 42
```

This stage combines SNP-to-latent attribution with phenotype-specific latent importance and then applies the same MAF/LD filtering logic used by the VAE pipeline.

## B5. Quantify upstream truth retention

```bash
python scripts/benchmark_b/04_check_weighted_truth_retention.py \
  --base results/benchmark_b
```

A random 2,000-SNP retention control is available with:

```bash
python scripts/benchmark_b/05_random_2000_retention.py \
  --base results/benchmark_b \
  --seed 42
```

## B6. Run downstream interaction methods

The weighted 2k and full 10k branches use the same RF-SHAP / RF-SHAP-Taylor scripts from `scripts/core/`. The full 10k exhaustive PLINK and MatrixEpistasis branches can be run with the same wrappers used in Benchmark A.

For manuscript-scale runs, these steps can be computationally intensive. The scripts are ordinary command-line programs and can be launched directly, through a workflow manager, or through any compute scheduler.

## B7. Summarize and plot

```bash
python scripts/benchmark_b/06_summarize_benchmark_b.py \
  --base results/benchmark_b

python scripts/benchmark_b/07_plot_benchmark_b.py \
  --base results/benchmark_b

python scripts/benchmark_b/08_plot_architecture_metrics.py \
  --base results/benchmark_b
```

The main summary table is:

```text
results/benchmark_b/benchmark_B_summary/benchmarkB_master_metrics.csv
```

## Reproducibility conventions

- Benchmark A compares detectors within a common 2,000-SNP candidate space.
- Benchmark B evaluates the full end-to-end pipeline and therefore counts causal pairs lost during upstream VAE prioritization as false negatives in end-to-end recall.
- Conditional recall is retained separately to distinguish downstream ranking performance from upstream selection/screening loss.
- For 10,000 SNPs there are 49,995,000 unordered pairs; for 2,000 SNPs there are 1,999,000, corresponding to a nominal pair-space reduction of about 96%.
- Large genotype matrices and manuscript-scale outputs are not committed to the public repository. See `data/README.md`.

