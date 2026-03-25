# Pathway Enrichment Analysis

This module performs biological follow-up analysis for SNP-pair results that have already been mapped to nearby genes.

It is designed to run the same workflow across:

- the three 3P classes:
  - `ML + GWAS + INT`
  - `ML + GWAS`
  - `ML + INT`
- both pair types:
  - `intra`
  - `inter`

For each subset, the module:

1. extracts unique genes and unique gene pairs from `snp1_gene` and `snp2_gene`
2. runs pathway enrichment using Enrichr
3. analyzes protein-protein interaction support using STRING
4. optionally overlaps the resulting genes with an external GWAS gene TSV

## Module structure

```text
pathway_enrichment/
├── README.md
└── pathway_enrichment_analysis.py
```
**Input requirements**

The input pair tables should already contain mapped gene annotations, typically from the FUMA annotation step.

**Requirements**

Install the required Python packages:
```text
pip install numpy pandas matplotlib requests
```
What the script does

1. Gene extraction

- For each row, the script splits snp1_gene and snp2_gene on semicolons and builds:

  - a unique gene list
  - a unique set of gene pairs
2. Enrichr analysis

- The script submits the gene list to Enrichr and collects results for the requested libraries.

- Default libraries:

  - KEGG_2021_Human
  - GO_Biological_Process_2021
  - WikiPathway_2021_Human
3. STRING interaction analysis

- The script queries STRING for protein-protein interaction support among the genes and summarizes:

  - direct interactions
  - common neighbors
4. GWAS gene overlap

- If a GWAS TSV is provided, the script restricts it to genes overlapping the current subset.

**How to run**
- Run for intra and inter pairs across the three classes
```text
python pathway_enrichment_analysis.py \
  --intra_pairs_csv path/to/intra_pairs_with_fuma_genes.csv \
  --inter_pairs_csv path/to/inter_pairs_with_fuma_genes.csv \
  --output_dir outputs/pathway_enrichment \
  --gwas_tsv_path path/to/bmi_genes_gwas_catalog.csv
```
- Also analyze all rows in each file, not only class-specific subsets
```text
python pathway_enrichment_analysis.py \
  --intra_pairs_csv path/to/intra_pairs_with_fuma_genes.csv \
  --inter_pairs_csv path/to/inter_pairs_with_fuma_genes.csv \
  --output_dir outputs/pathway_enrichment \
  --gwas_tsv_path path/to/bmi_genes_gwas_catalog.csv \
  --run_all_rows
```
Output structure

The script creates one folder per pair type and per class.


```text
outputs/pathway_enrichment/
├── intra/
│   ├── ml_gwas_int/
│   │   ├── pathway_enrichment.csv
│   │   ├── enrichment_plot.png
│   │   ├── string_interactions.csv
│   │   ├── gene_pair_interactions.csv
│   │   ├── gwas_catalog_genes_from_tsv.csv
│   │   ├── analysis_summary.csv
│   │   ├── prepare_leads_summary.json
│   │   └── ...
│   ├── ml_gwas/
│   └── ml_int/
├── inter/
│   ├── ml_gwas_int/
│   ├── ml_gwas/
│   └── ml_int/
└── master_analysis_summary.csv
```
**Core outputs per subset:**

- pathway_enrichment.csv
- Parsed Enrichr results
- enrichment_plot.png
- Bar plot of top enriched terms
- string_interactions.csv
- STRING edges above the chosen score threshold
- gene_pair_interactions.csv
- Direct and shared-neighbor support for gene pairs
- gwas_catalog_genes_from_tsv.csv
- analysis_summary.csv

**Important arguments**

--intra_pairs_csv
    - Annotated intra-chromosomal pair file

--inter_pairs_csv
    - Annotated inter-chromosomal pair file

--class_col
    - Column used for class splitting, default: class

--classes
    - Classes to analyze separately

--gwas_tsv_path
    - Optional external GWAS gene TSV

--gwas_gene_col
    - Gene column in the GWAS TSV, default: gene

--gene_set_libraries
    - Enrichr libraries to query

--string_score_threshold
    - Minimum STRING score, default: 400

--top_n_enrichr
    - Number of terms saved per library, default: 20

--top_n_plot
    - Number of terms shown in the plot, default: 10

**Citation**

If you use this code, please cite our manuscript.

