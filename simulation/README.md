# Genetic Architecture Simulation Framework

This module contains code for two phenotype simulation settings designed to evaluate methods under additive and epistatic genetic architectures.

The simulations generate quantitative phenotypes from genotype matrices under controlled heritability, varying causal SNP complexity, and different nonlinear interaction structures.

## Module structure

```text
simulation/
├── README.md
├── simulate1.py
├── run_simulation1.py
├── simulate2.py
└── run_simulation2.py

```
**Requirements**

The code uses Python with:

```text
numpy
pandas
```

Install with:

```text
pip install numpy pandas
```
## Input data
Both simulations expect a genotype matrix in CSV format:

rows = individuals

columns = SNPs

values = genotype dosages or encoded genotype values

Update the genotype path in the run scripts before executing:
```text
run_simulation1.py
run_simulation2.py
```
## Simulation 1: Sparse Epistatic Architecture with Controlled Interaction Complexity

**Overview:**

Simulation 1 generates quantitative phenotypes using a structured nonlinear architecture in which epistatic effects arise only from interactions between two disjoint SNP subsets.

**For each simulation setting:**

N causal SNPs are randomly selected from the genotype matrix,
the selected SNPs are partitioned into three subsets:

s1, s2, s3

- epistatic effects are generated only from cross-block interactions between SNPs in s1 and SNPs in s2,
- marginal effects are generated from all selected causal SNPs (s1, s2, and s3),
- the final phenotype is formed by summing the marginal component, the epistatic component, and Gaussian residual noise.

This simulation creates a nonlinear phenotype architecture while keeping interaction complexity computationally manageable.

**Simulation design**

Simulation 1 evaluates the following causal SNP counts:

15, 50, 100, 250

For each setting, the total broad-sense heritability is fixed at:

H2 = 0.20

The fraction of genetic variance assigned to marginal effects is:

rho = 0.05

The remaining variance is assigned to epistatic effects:

1 - rho = 0.95

**Files:**
- simulate2.py
- Contains the core Simulation 1 function:
- causal SNP selection
- mean imputation and genotype standardization
- construction of s1, s2, and s3
- generation of marginal and epistatic phenotype components
- variance rescaling to match the target heritability settings
- run_simulation1.py
- Runs all Simulation 1 scenarios and saves outputs for each scenario.
- How to run Simulation 1

Update the genotype path in run_simulation1.py:
```text
GENOTYPE_PATH = "/path/to/your/genotype_matrix.csv"
```
Then run:
```text
python run_simulation1.py
```
Simulation 1 outputs

Example output structure:
```text
simulation1_outputs/
├── sim1_n15/
│   ├── simulated_phenotype.csv
│   ├── causal_snps.txt
│   ├── s1_snps.txt
│   ├── s2_snps.txt
│   ├── s3_snps.txt
│   ├── beta_epistatic.csv
│   ├── beta_marginal.csv
│   └── metadata.json
├── sim1_n50/
├── sim1_n100/
├── sim1_n250/
└── simulation1_summary.csv
```

**Output files:**

- simulated_phenotype.csv — simulated phenotype and component values for each individual
- causal_snps.txt — full list of selected causal SNPs
- s1_snps.txt, s2_snps.txt, s3_snps.txt — SNP subsets defining the simulation structure
- beta_epistatic.csv — epistatic SNP-pair effect sizes
- beta_marginal.csv — marginal SNP effect sizes
- metadata.json — scenario-specific metadata, including subset sizes, number of interaction pairs, and realized variance components
- simulation1_summary.csv — summary across all Simulation 1 scenarios
  
## Simulation 2: Overlapping Additive and Epistatic Genetic Architecture

**Overview:**

**For each simulation setting:**

- a set of additive causal SNPs and a set of epistatic causal SNPs are selected from the genotype matrix,
- the two sets are allowed to overlap by a user-defined proportion,
- additive effects are generated as a linear combination of additive SNP genotypes,
- epistatic effects are generated from all pairwise multiplicative interactions among epistatic SNPs,
- the final phenotype is formed by summing additive and epistatic components and adding Gaussian noise.

The current implementation uses:

- broad-sense heritability: H2 = 0.20
- additive variance proportion: rho = 0.05
- epistatic variance proportion: 1 - rho = 0.95

This creates a strongly non-additive phenotype architecture.

**Simulation design**

Simulation 1 evaluates the Cartesian product of:

causal SNP counts: 15, 50, 100, 250

overlap proportions: 0.3, 0.5, 0.7, 0.9

This gives 16 simulation scenarios in total.

For each scenario:

- num_additive = N
- num_epistatic = N
  
the number of overlapping SNPs is defined as:

n_overlap = floor(min(n_add, n_epi) × p_overlap)

The epistatic component includes all pairwise interactions among the epistatic SNPs:

K = n_epi × (n_epi - 1) / 2

So the number of interaction terms is:

- 15 SNPs → 105 pairs
- 50 SNPs → 1,225 pairs
- 100 SNPs → 4,950 pairs
- 250 SNPs → 31,125 pairs

**Files:**
- simulate1.py
- Contains the core Simulation 2 function:
- genotype standardization
- selection of overlapping additive and epistatic SNP sets
- phenotype generation
- variance rescaling for additive and epistatic components
- run_simulation1.py
- Runs all Simulation 2 scenarios, saves outputs for each scenario, and creates a summary table.
- How to run Simulation 2

Update the genotype path in run_simulation2.py:
```text
GENOTYPE_PATH = "/path/to/your/genotype_matrix.csv"
```
Then run:
```text
python run_simulation2.py
```
Simulation 2 outputs

Example output structure:
```text
simulation2_outputs/
├── sim2_n15_overlap0p3/
│   ├── phenotype.csv
│   ├── additive_snps.txt
│   ├── epistatic_snps.txt
│   ├── shared_snps.txt
│   ├── beta_add.csv
│   ├── beta_epi.csv
│   └── metadata.json
├── sim2_n15_overlap0p5/
├── ...
└── simulation2_summary.csv
```
**Output files:**

- phenotype.csv — simulated phenotype and component values for each individual
- additive_snps.txt — SNPs included in the additive set
- epistatic_snps.txt — SNPs included in the epistatic set
- shared_snps.txt — SNPs shared between additive and epistatic sets
- beta_add.csv — additive SNP effect sizes
- beta_epi.csv — epistatic SNP-pair effect sizes
- metadata.json — scenario-specific simulation metadata
- simulation2_summary.csv — summary across all Simulation 2 scenarios

**Notes**

- Genotypes are standardized before phenotype generation.
- Missing values are mean-imputed in Simulation 1 before standardization.
- Additive and epistatic components are rescaled to target the requested variance contributions.
- Different seeds can be used across scenarios for reproducibility.
- Simulation 1 uses structured cross-block interactions only.
- Simulation 2 uses all pairwise interactions among epistatic SNPs.

**Citation**

If you use this code, please cite the our manuscript.


