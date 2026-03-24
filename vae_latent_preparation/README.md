# VAE Latent Space Preparation

This module contains code for preparing low-dimensional latent representations from high-dimensional genotype matrices using a variational autoencoder (VAE), followed by Random Forest and SHAP-based feature ranking for each latent dimension.

The workflow is split into two steps:

1. `vae_latent_preparation.py`  
   Trains a VAE on the input matrix, exports the latent representation, and saves reconstruction and latent-space diagnostics.

2. `latent_shap_ranking.py`  
   Uses Random Forest regression and SHAP values to identify the most important original features for each latent dimension.

## Module structure

```text
vae_latent_preparation/
├── README.md
├── vae_latent_preparation.py
└── latent_shap_ranking.py
```
**Requirements**

The code uses Python with:

- numpy
- pandas
- matplotlib
- tensorflow
- scikit-learn
- umap-learn
- shap
- joblib

Install with:

```text
pip install numpy pandas matplotlib tensorflow scikit-learn umap-learn shap joblib
```

Input data

The module expects an input CSV file with:

rows = samples

columns = SNPs

Example:

first column = sample ID

remaining columns = genotype dosages or feature values

**Step 1: VAE training and latent export**

vae_latent_preparation.py performs:

- input loading
- MinMax normalization
- VAE training
- latent extraction
- training-loss plotting
- reconstruction error plotting
- UMAP visualization of the latent space

How to run

```text
python vae_latent_preparation.py \
  --input_csv path/to/input_features.csv \
  --output_dir outputs/vae_run \
  --latent_dim 250 \
  --epochs 50 \
  --batch_size 128
```
Main outputs

```text
outputs/vae_run/
├── latent_space.csv
├── training_losses.png
├── reconstruction_error_hist.png
├── latent_umap.png
└── run_summary.json
```

**Output files:**

- latent_space.csv — latent representation for each sample
- training_losses.png — training and validation reconstruction loss
- reconstruction_error_hist.png — reconstruction error distribution
- latent_umap.png — UMAP projection of the latent space
- run_summary.json — basic run metadata and reconstruction MSE

**Step 2: RF + SHAP ranking for latent dimensions**

latent_shap_ranking.py uses the original input matrix and the exported latent space to:

- fit one Random Forest model per latent dimension
- compute SHAP values for original features
- save top-ranked features for each latent dimension
- aggregate the top signals across all latent dimensions

How to run:

```text
python latent_shap_ranking.py \
  --input_csv path/to/input_features.csv \
  --latent_csv outputs/vae_run/latent_space.csv \
  --output_dir outputs/vae_shap \
  --top_k 100 \
  --prefer_threads
```

If you want to apply MinMax normalization to the input again before RF:

```text
python latent_shap_ranking.py \
  --input_csv path/to/input_features.csv \
  --latent_csv outputs/vae_run/latent_space.csv \
  --output_dir outputs/vae_shap \
  --top_k 100 \
  --normalize_input \
  --prefer_threads
```

**Main outputs:**

```text
outputs/vae_shap/
├── latent_0_top_features.csv
├── latent_0_shap_barplot.png
├── latent_1_top_features.csv
├── latent_1_shap_barplot.png
├── ...
└── aggregated_top_snps.csv
```

**Output files:**

- latent_i_top_features.csv — top-ranked original features for latent dimension i
- latent_i_shap_barplot.png — barplot of mean absolute SHAP values for latent dimension i
- aggregated_top_snps.csv — aggregated feature ranking across latent dimensions

**Notes**

- vae_latent_preparation.py and latent_shap_ranking.py are intentionally separated so VAE training and SHAP ranking can be run independently.
- SHAP can be computationally expensive for high-dimensional data and many latent dimensions.
- Use --n_jobs and --prefer_threads carefully depending on available memory and CPU resources.

**Citation**

If you use this code, please cite our manuscript.

