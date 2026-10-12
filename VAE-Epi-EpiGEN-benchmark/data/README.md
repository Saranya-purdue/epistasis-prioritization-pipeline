# Data layout and schemas

This repository does not redistribute the manuscript genotype matrices.

## Genotype CSV

Expected format:

```text
IID,variant_1,variant_2,...
S1,0,1,...
S2,2,0,...
```

- First column: sample identifier.
- Remaining columns: genotype dosage coded 0/1/2.
- Missing genotypes must be handled before EpiGEN simulation.

## Causal-pair CSV

```text
Pair_ID,SNP1,SNP2
PAIR_1,rs123,rs456
```

At minimum, `SNP1` and `SNP2` are required.

## SNP map

Benchmark B expects a CSV with at least:

```text
VARIANT_ID,SNP,CHR,BP
```

The nested benchmark builder accepts either `VARIANT_ID` or `SNP` as the matching key and requires `CHR`.

## Phenotype

Phenotype files used by the weighted selector may be comma- or tab-delimited. An ID column (`ID`, `sample_id`, `IID`, or `SNP`) and a phenotype column (`Phenotype` or `pheno`) are detected automatically.

## Sensitive data

Do not place HRS or other controlled-access individual-level genotype/phenotype data in a public GitHub repository. Public code should point users to the authorized data-access procedure and accept local input paths through command-line arguments.
