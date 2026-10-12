# Data layout

Large manuscript-scale genotype and method-output files are not included in this repository.

## Input conventions

### Genotype CSV

- First column: sample identifier.
- Remaining columns: hard-call genotypes coded `0`, `1`, or `2`.
- EpiGEN Benchmark A utilities can use rsID-like SNP names.
- The VAE/LD script expects variant columns in `RSID:BP:REF:ALT` form, or an accompanying SNP map as described by `scripts/core/vae_shap_ld_cpu.py`.

### Causal-pair CSV

Required columns:

```text
SNP1,SNP2
```

An optional `Pair_ID` column is retained when present.

### EpiGEN phenotype CSV

The simulation scripts write:

```text
IID,Phenotype,Epistatic_Component
```

`06_standardize_phenotype_and_truth.py` aligns the phenotype with genotype sample IDs, z-standardizes the phenotype, and creates the `SNP_Pair` truth field expected by the common input-preparation script.

## Example

`data/example/` contains small generated files used for smoke-testing the code. They are synthetic and are not participant data.
