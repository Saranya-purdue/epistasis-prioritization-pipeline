# Deviation from Additivity

This module visualizes how selected SNP-SNP interaction pairs deviate from an additive genetic expectation.

For each pair, the script:

1. computes observed phenotype means across the 3×3 genotype grid,
2. fits an additive-only linear model,
3. derives the expected additive phenotype surface,
4. computes residual deviation from additivity,
5. ranks pairs by the maximum absolute deviation in sufficiently powered genotype cells,
6. plots top intra-chromosomal and inter-chromosomal examples.

## Module structure

```text
deviation_from_additivity/
├── README.md
└── deviation_from_additivity.py
```
**Overview**

The additive reference model is:

y ~ g1 + g2

where:

g1 = genotype dosage for SNP1,
g2 = genotype dosage for SNP2,
y = phenotype

For each genotype combination (g1, g2), the script compares:

- the observed mean phenotype
- the additive-only expected phenotype
- the mean residual from additivity

Pairs are then ranked using the maximum absolute residual among genotype cells that have enough samples.

**Inputs**

The script expects:

- a genotype matrix CSV
- a phenotype CSV
- an intra-chromosomal pair CSV
- an inter-chromosomal pair CSV

**Requirements**

The code uses Python with:

- numpy
- pandas
- matplotlib
- statsmodels

Install with:
```text
pip install numpy pandas matplotlib statsmodels
```
Pair selection logic

Pairs are filtered within a chosen class, by default:

ML + GWAS + INT

For each pair, the script computes the maximum absolute residual-from-additivity among genotype cells with:

cell count >= min_cell_n

It then keeps pairs with at least:

min_powered_cells

powered cells and ranks them by:

- max_dev_powered descending
- p_interact_lin ascending

**How to run**
```text
python deviation_from_additivity.py \
  --genotype_csv path/to/X_pruned.csv \
  --phenotype_csv path/to/BMI_Whites_all.csv \
  --intra_pairs_csv path/to/intra_pairs.csv \
  --inter_pairs_csv path/to/inter_pairs.csv \
  --output_dir outputs/deviation_from_additivity \
  --pair_class "ML + GWAS + INT" \
  --min_cell_n 50 \
  --min_powered_cells 9 \
  --top_k 3
```
**Main outputs**
```text
outputs/deviation_from_additivity/
├── intra_pairs_with_deviation_metrics.csv
├── inter_pairs_with_deviation_metrics.csv
├── selected_intra_pairs.csv
├── selected_inter_pairs.csv
├── deviation_from_additivity_figure.pdf
├── deviation_from_additivity_figure.png
└── deviation_from_additivity_summary.json
```

**Output descriptions**

- intra_pairs_with_deviation_metrics.csv
- Intra-chromosomal pair table with computed deviation metrics
- inter_pairs_with_deviation_metrics.csv
- Inter-chromosomal pair table with computed deviation metrics
- selected_intra_pairs.csv
- Top-ranked intra-chromosomal pairs used for plotting
- selected_inter_pairs.csv
- Top-ranked inter-chromosomal pairs used for plotting
- deviation_from_additivity_figure.pdf / .png
- Multi-panel figure showing observed phenotype means and residual deviation from additivity
- deviation_from_additivity_summary.json
- Summary of thresholds, sample counts, and number of selected pairs

**Notes**
- This module is intended for visualization and interpretation of already prioritized interaction pairs.

**Citation**

If you use this code, please cite our manuscript.
