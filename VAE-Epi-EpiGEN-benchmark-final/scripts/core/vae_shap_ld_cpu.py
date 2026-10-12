#!/usr/bin/env python3
"""
End-to-end VAE -> SHAP -> LD filtering pipeline for simulated 0/1/2 genotype data.

Input genotype CSV
------------------
- Row index / first column: sample ID
- Remaining columns: variants, e.g. rs7606664:15859491:A:T
- Values: hard-call genotypes 0, 1, 2

Important chromosome note
-------------------------
The example variant ID contains RSID:BP:REF:ALT but NOT chromosome.
Therefore, before LD filtering you must provide EITHER:
  1) --chrom CHR          if the entire genotype CSV is one chromosome, OR
  2) --snp-map FILE.csv   if the CSV contains variants from multiple chromosomes.

The optional SNP map can contain:
  - VARIANT_ID, CHR
or
  - SNP, CHR
and may also contain BP. The BP embedded in the genotype column name is retained.

The script:
1. Converts the large CSV once to a memory-mapped int8 .npy cache.
2. Splits individuals into fixed train/validation/test sets.
3. Trains one VAE (default latent dimension 128).
4. Produces train/validation curves and held-out reconstruction metrics.
5. Selects active latent dimensions by variance.
6. Calculates memory-safe SHAP expected-gradient scores in latent-output chunks.
7. Ranks SNPs by mean absolute SHAP across explained samples/latent outputs.
8. Greedily retains high-SHAP SNPs while removing local high-LD variants.
9. Writes exactly the requested final number of SNPs (default 2000).
10. Optionally checks simulated ground-truth pair retention.

This CPU version intentionally disables CUDA before importing TensorFlow.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import gc
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--csv", required=True, help="Genotype CSV.")
    p.add_argument("--outdir", required=True, help="Output directory.")

    chrom_group = p.add_mutually_exclusive_group(required=True)
    chrom_group.add_argument(
        "--chrom",
        default=None,
        help="Chromosome label if ALL variants in the genotype CSV are from one chromosome."
    )
    chrom_group.add_argument(
        "--snp-map",
        default=None,
        help="CSV mapping variants to chromosomes when genotype CSV contains multiple chromosomes."
    )

    p.add_argument(
        "--ground-truth",
        default=None,
        help="Optional CSV containing true interacting variant pairs."
    )
    p.add_argument("--truth-col1", default="SNP1")
    p.add_argument("--truth-col2", default="SNP2")

    # VAE
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden1", type=int, default=512)
    p.add_argument("--hidden2", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--l2-reg", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # CSV/cache
    p.add_argument(
        "--csv-chunksize",
        type=int,
        default=500,
        help="Rows per chunk when converting CSV to compact int8 NPY cache."
    )

    # SHAP
    p.add_argument("--shap-background", type=int, default=32)
    p.add_argument("--shap-explain", type=int, default=32)
    p.add_argument("--shap-nsamples", type=int, default=32)
    p.add_argument("--shap-output-chunk", type=int, default=8)
    p.add_argument(
        "--latent-var-threshold",
        type=float,
        default=1e-4,
        help="Only latent dimensions above this z_mean variance are eligible for SHAP."
    )
    p.add_argument(
        "--max-shap-latents",
        type=int,
        default=0,
        help="0 = all active latent dimensions; positive integer = highest-variance N active dimensions."
    )

    # LD/final selection
    p.add_argument("--final-n", type=int, default=2000)
    p.add_argument("--ld-r2", type=float, default=0.20)
    p.add_argument("--ld-window", type=int, default=500_000)

    return p.parse_args()


def setup_threads():
    n_threads = int(os.environ.get("VAE_EPI_NUM_THREADS", os.cpu_count() or 1))
    n_threads = max(1, n_threads)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(n_threads)
        tf.config.threading.set_inter_op_parallelism_threads(min(4, n_threads))
    except RuntimeError:
        pass
    return n_threads


def log_versions():
    print("=" * 80, flush=True)
    print("Python:", sys.version.replace("\n", " "), flush=True)
    print("TensorFlow:", tf.__version__, flush=True)
    print("TensorFlow GPUs visible:", tf.config.list_physical_devices("GPU"), flush=True)
    try:
        import shap
        print("SHAP:", shap.__version__, flush=True)
    except Exception as e:
        print("SHAP import FAILED:", repr(e), flush=True)
        raise
    print("=" * 80, flush=True)


def parse_variant_columns(columns):
    records = []
    for i, col in enumerate(columns):
        text = str(col)
        parts = text.split(":", 3)
        if len(parts) != 4:
            raise ValueError(
                f"Variant column '{text}' does not match RSID:BP:REF:ALT."
            )
        rsid, bp, ref, alt = parts
        try:
            bp_int = int(bp)
        except ValueError:
            raise ValueError(
                f"Could not parse base-pair position from column '{text}'."
            )
        records.append(
            {
                "feature_index": i,
                "VARIANT_ID": text,
                "SNP": rsid,
                "BP": bp_int,
                "REF": ref,
                "ALT": alt,
            }
        )
    return pd.DataFrame(records)


def add_chromosome_information(meta, chrom=None, snp_map_file=None):
    meta = meta.copy()

    if chrom is not None:
        meta["CHR"] = str(chrom).replace("chr", "")
        return meta

    mapping = pd.read_csv(snp_map_file)
    mapping.columns = [str(c).strip() for c in mapping.columns]

    if "CHR" not in mapping.columns:
        raise ValueError("--snp-map must contain a CHR column.")

    mapping["CHR"] = (
        mapping["CHR"].astype(str).str.replace("chr", "", regex=False)
    )

    if "VARIANT_ID" in mapping.columns:
        tmp = mapping[["VARIANT_ID", "CHR"]].copy()
        tmp["VARIANT_ID"] = tmp["VARIANT_ID"].astype(str)
        if tmp["VARIANT_ID"].duplicated().any():
            raise ValueError("VARIANT_ID is duplicated in --snp-map.")
        meta = meta.merge(tmp, on="VARIANT_ID", how="left", validate="one_to_one")

    elif "SNP" in mapping.columns:
        tmp = mapping[["SNP", "CHR"]].copy()
        tmp["SNP"] = tmp["SNP"].astype(str)
        if tmp["SNP"].duplicated().any():
            raise ValueError(
                "SNP/rsID is duplicated in --snp-map. "
                "Provide a unique VARIANT_ID column instead."
            )
        meta = meta.merge(tmp, on="SNP", how="left", validate="many_to_one")

    else:
        raise ValueError(
            "--snp-map must contain either VARIANT_ID or SNP in addition to CHR."
        )

    if meta["CHR"].isna().any():
        nmiss = int(meta["CHR"].isna().sum())
        raise ValueError(f"{nmiss} genotype variants are missing chromosome mappings.")

    return meta


def count_csv_rows(csv_file):
    # Binary line count avoids parsing the huge file just to get N.
    with open(csv_file, "rb") as f:
        n_lines = sum(1 for _ in f)
    if n_lines < 2:
        raise ValueError("Genotype CSV has no data rows.")
    return n_lines - 1


def prepare_genotype_cache(csv_file, outdir, chrom=None, snp_map_file=None, chunksize=500):
    """
    Creates:
      cache/genotype_int8.npy
      cache/sample_ids.txt
      cache/variant_metadata.csv

    If all three already exist and dimensions agree, reuses them.
    """
    cache_dir = outdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    geno_path = cache_dir / "genotype_int8.npy"
    sample_path = cache_dir / "sample_ids.txt"
    meta_path = cache_dir / "variant_metadata.csv"

    # Header-only parse is cheap and gives variant order.
    header = pd.read_csv(csv_file, nrows=0, index_col=0)
    variant_columns = header.columns.astype(str).tolist()
    meta_expected = parse_variant_columns(variant_columns)
    meta_expected = add_chromosome_information(
        meta_expected, chrom=chrom, snp_map_file=snp_map_file
    )

    if geno_path.exists() and sample_path.exists() and meta_path.exists():
        G = np.load(geno_path, mmap_mode="r")
        meta = pd.read_csv(meta_path, dtype={"CHR": str})
        sample_ids = np.loadtxt(sample_path, dtype=str, ndmin=1)

        if G.shape[1] == len(meta_expected) and len(meta) == len(meta_expected):
            if np.array_equal(
                meta["VARIANT_ID"].astype(str).to_numpy(),
                meta_expected["VARIANT_ID"].astype(str).to_numpy()
            ) and G.shape[0] == len(sample_ids):
                print(f"Reusing genotype cache: {geno_path}", flush=True)
                return G, sample_ids, meta

        raise RuntimeError(
            "Existing cache does not match the current input CSV. "
            f"Remove {cache_dir} if you intentionally changed the genotype input."
        )

    n_samples = count_csv_rows(csv_file)
    n_snps = len(variant_columns)

    print(f"CSV dimensions inferred: {n_samples:,} samples x {n_snps:,} SNPs", flush=True)
    print("Converting CSV -> int8 NPY cache (one-time step)...", flush=True)

    tmp_geno = cache_dir / "genotype_int8.partial.npy"
    if tmp_geno.exists():
        tmp_geno.unlink()

    mm = np.lib.format.open_memmap(
        tmp_geno,
        mode="w+",
        dtype=np.int8,
        shape=(n_samples, n_snps),
    )

    sample_ids = []
    row0 = 0

    reader = pd.read_csv(
        csv_file,
        index_col=0,
        chunksize=chunksize,
    )

    for chunk_no, chunk in enumerate(reader, start=1):
        if list(chunk.columns.astype(str)) != variant_columns:
            raise ValueError("Variant columns/order changed while reading the CSV.")

        arr = chunk.to_numpy(dtype=np.float32, copy=False)

        if not np.isfinite(arr).all():
            raise ValueError(
                f"Missing/non-finite genotype detected in CSV chunk {chunk_no}. "
                "Impute missing genotypes before running this pipeline."
            )

        valid = (arr == 0.0) | (arr == 1.0) | (arr == 2.0)
        if not valid.all():
            bad = arr[~valid]
            raise ValueError(
                f"Expected hard-call genotypes 0/1/2 only. "
                f"Chunk {chunk_no} contains examples: {bad[:10]}"
            )

        n = arr.shape[0]
        mm[row0:row0+n, :] = arr.astype(np.int8, copy=False)
        sample_ids.extend(chunk.index.astype(str).tolist())
        row0 += n

        if chunk_no % 10 == 0 or row0 == n_samples:
            print(f"  cached {row0:,}/{n_samples:,} samples", flush=True)

    if row0 != n_samples:
        raise RuntimeError(f"Expected {n_samples} rows but read {row0}.")

    mm.flush()
    del mm
    os.replace(tmp_geno, geno_path)

    np.savetxt(sample_path, np.asarray(sample_ids, dtype=str), fmt="%s")
    meta_expected.to_csv(meta_path, index=False)

    G = np.load(geno_path, mmap_mode="r")
    return G, np.asarray(sample_ids, dtype=str), meta_expected


def add_maf(meta, G, batch_rows=1000):
    """
    Streaming MAF computation over the original 0/1/2 genotype matrix.
    """
    allele_sum = np.zeros(G.shape[1], dtype=np.float64)
    n = 0

    for start in range(0, G.shape[0], batch_rows):
        block = np.asarray(G[start:start+batch_rows, :], dtype=np.float32)
        allele_sum += block.sum(axis=0)
        n += block.shape[0]

    af = allele_sum / (2.0 * n)
    maf = np.minimum(af, 1.0 - af)

    out = meta.copy()
    out["MAF"] = maf

    if np.any(maf == 0):
        print(
            f"WARNING: {(maf == 0).sum():,} monomorphic variants detected. "
            "They remain in the VAE input but cannot pass LD correlation meaningfully.",
            flush=True
        )
    return out


class GenotypeSequence(keras.utils.Sequence):
    def __init__(self, G, indices, batch_size=64, shuffle=True, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.G = G
        self.indices = np.asarray(indices, dtype=np.int64).copy()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self):
        return int(math.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, i):
        batch_idx = self.indices[
            i * self.batch_size : (i + 1) * self.batch_size
        ]
        x = np.asarray(self.G[batch_idx, :], dtype=np.float32) / 2.0
        return x

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


class Encoder(layers.Layer):
    def __init__(self, latent_dim, hidden1, hidden2, dropout, l2_reg):
        super().__init__()
        self.d1 = layers.Dense(
            hidden1, activation="relu", kernel_regularizer=l2(l2_reg)
        )
        self.bn1 = layers.BatchNormalization()
        self.drop1 = layers.Dropout(dropout)

        self.d2 = layers.Dense(
            hidden2, activation="relu", kernel_regularizer=l2(l2_reg)
        )
        self.bn2 = layers.BatchNormalization()
        self.drop2 = layers.Dropout(dropout)

        self.z_mean_layer = layers.Dense(latent_dim, name="z_mean")
        self.z_logvar_layer = layers.Dense(latent_dim, name="z_logvar")

    def call(self, x, training=False):
        x = self.d1(x)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)

        x = self.d2(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)

        return self.z_mean_layer(x), self.z_logvar_layer(x)


class Decoder(layers.Layer):
    def __init__(self, original_dim, hidden1, hidden2, dropout, l2_reg):
        super().__init__()
        self.d1 = layers.Dense(
            hidden2, activation="relu", kernel_regularizer=l2(l2_reg)
        )
        self.bn1 = layers.BatchNormalization()
        self.drop1 = layers.Dropout(dropout)

        self.d2 = layers.Dense(
            hidden1, activation="relu", kernel_regularizer=l2(l2_reg)
        )
        self.bn2 = layers.BatchNormalization()
        self.drop2 = layers.Dropout(dropout)

        # Inputs are genotype/2, i.e. 0, 0.5, 1. Sigmoid keeps reconstruction in [0,1].
        self.out = layers.Dense(original_dim, activation="sigmoid")

    def call(self, z, training=False):
        z = self.d1(z)
        z = self.bn1(z, training=training)
        z = self.drop1(z, training=training)

        z = self.d2(z)
        z = self.bn2(z, training=training)
        z = self.drop2(z, training=training)

        return self.out(z)


class VAE(keras.Model):
    def __init__(
        self, original_dim, latent_dim, hidden1, hidden2,
        dropout=0.1, l2_reg=1e-5, beta=0.01
    ):
        super().__init__()
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.beta = float(beta)

        self.encoder = Encoder(latent_dim, hidden1, hidden2, dropout, l2_reg)
        self.decoder = Decoder(original_dim, hidden1, hidden2, dropout, l2_reg)

        self.total_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_tracker = keras.metrics.Mean(name="kl_loss")
        self.reg_tracker = keras.metrics.Mean(name="reg_loss")

    @property
    def metrics(self):
        return [
            self.total_tracker,
            self.recon_tracker,
            self.kl_tracker,
            self.reg_tracker,
        ]

    @staticmethod
    def reparameterize(z_mean, z_logvar):
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps

    def call(self, x, training=False):
        z_mean, z_logvar = self.encoder(x, training=training)
        z = self.reparameterize(z_mean, z_logvar)
        return self.decoder(z, training=training)

    def deterministic_reconstruction(self, x):
        z_mean, _ = self.encoder(x, training=False)
        return self.decoder(z_mean, training=False)

    def _losses_for_batch(self, x, training=False):
        z_mean, z_logvar = self.encoder(x, training=training)
        z = self.reparameterize(z_mean, z_logvar)
        recon = self.decoder(z, training=training)

        recon_loss = tf.reduce_mean(tf.square(x - recon))

        # Average KL over both samples and latent dimensions, so beta is comparable.
        kl_loss = -0.5 * tf.reduce_mean(
            1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
        )

        reg_loss = (
            tf.add_n(self.losses)
            if self.losses
            else tf.constant(0.0, dtype=tf.float32)
        )

        total = recon_loss + self.beta * kl_loss + reg_loss
        return total, recon_loss, kl_loss, reg_loss

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data

        with tf.GradientTape() as tape:
            total, recon, kl, reg = self._losses_for_batch(x, training=True)

        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_tracker.update_state(total)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        self.reg_tracker.update_state(reg)

        return {
            "total_loss": self.total_tracker.result(),
            "recon_loss": self.recon_tracker.result(),
            "kl_loss": self.kl_tracker.result(),
            "reg_loss": self.reg_tracker.result(),
        }

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        total, recon, kl, reg = self._losses_for_batch(x, training=False)

        self.total_tracker.update_state(total)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        self.reg_tracker.update_state(reg)

        return {
            "total_loss": self.total_tracker.result(),
            "recon_loss": self.recon_tracker.result(),
            "kl_loss": self.kl_tracker.result(),
            "reg_loss": self.reg_tracker.result(),
        }


def evaluate_reconstruction(model, G, indices, batch_size=128):
    sse = 0.0
    sae = 0.0
    total_values = 0
    sum_y = 0.0
    sum_y2 = 0.0
    correct = 0

    for start in range(0, len(indices), batch_size):
        idx = indices[start:start+batch_size]
        x = np.asarray(G[idx, :], dtype=np.float32) / 2.0
        pred = model.deterministic_reconstruction(
            tf.convert_to_tensor(x)
        ).numpy()

        diff = x - pred
        sse += np.square(diff).sum(dtype=np.float64)
        sae += np.abs(diff).sum(dtype=np.float64)
        total_values += x.size
        sum_y += x.sum(dtype=np.float64)
        sum_y2 += np.square(x).sum(dtype=np.float64)

        true_gt = np.rint(x * 2.0).astype(np.int8)
        pred_gt = np.clip(np.rint(pred * 2.0), 0, 2).astype(np.int8)
        correct += int((true_gt == pred_gt).sum())

    mse = sse / total_values
    mae = sae / total_values
    mean_y = sum_y / total_values
    sst = sum_y2 - total_values * (mean_y ** 2)
    r2 = 1.0 - (sse / sst) if sst > 0 else np.nan
    concordance = correct / total_values

    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "R2": float(r2),
        "Genotype_concordance": float(concordance),
    }


def plot_training_history(history_df, plot_dir):
    if {"total_loss", "val_total_loss"}.issubset(history_df.columns):
        plt.figure(figsize=(7, 5))
        plt.plot(history_df["epoch"], history_df["total_loss"], label="Train")
        plt.plot(history_df["epoch"], history_df["val_total_loss"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Total loss")
        plt.title("VAE training and validation loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "01_training_total_loss.png", dpi=300)
        plt.close()

    if {"recon_loss", "val_recon_loss"}.issubset(history_df.columns):
        plt.figure(figsize=(7, 5))
        plt.plot(history_df["epoch"], history_df["recon_loss"], label="Train")
        plt.plot(history_df["epoch"], history_df["val_recon_loss"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction MSE")
        plt.title("VAE reconstruction loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "02_training_reconstruction_loss.png", dpi=300)
        plt.close()


def plot_split_metrics(metrics_df, plot_dir):
    ordered = ["train", "validation", "test"]
    m = metrics_df.set_index("split").loc[ordered]

    plt.figure(figsize=(6, 5))
    plt.bar(ordered, m["MSE"].to_numpy())
    plt.ylabel("MSE")
    plt.title("Deterministic reconstruction MSE")
    plt.tight_layout()
    plt.savefig(plot_dir / "03_reconstruction_MSE_by_split.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.bar(ordered, m["R2"].to_numpy())
    plt.ylabel("R²")
    plt.title("Deterministic reconstruction R²")
    plt.tight_layout()
    plt.savefig(plot_dir / "04_reconstruction_R2_by_split.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.bar(ordered, m["Genotype_concordance"].to_numpy())
    plt.ylabel("Genotype concordance")
    plt.title("Rounded-genotype reconstruction concordance")
    plt.tight_layout()
    plt.savefig(plot_dir / "05_genotype_concordance_by_split.png", dpi=300)
    plt.close()


def encode_z_mean(model, G, indices, batch_size=128):
    chunks = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start+batch_size]
        x = np.asarray(G[idx, :], dtype=np.float32) / 2.0
        z_mean, _ = model.encoder(tf.convert_to_tensor(x), training=False)
        chunks.append(z_mean.numpy())
    return np.concatenate(chunks, axis=0)


def plot_latent_variance(latent_var, plot_dir):
    order = np.argsort(latent_var)[::-1]
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(order)+1), latent_var[order], marker=".")
    plt.xlabel("Latent dimensions ranked by variance")
    plt.ylabel("Variance of z_mean")
    plt.title("Latent-space variance")
    plt.tight_layout()
    plt.savefig(plot_dir / "06_latent_variance.png", dpi=300)
    plt.close()


def build_selected_mean_encoder(vae, original_dim, latent_indices):
    inp = keras.Input(shape=(original_dim,), dtype=tf.float32)
    z_mean, _ = vae.encoder(inp, training=False)
    selected = tf.gather(z_mean, latent_indices, axis=1)
    return keras.Model(inp, selected, name="selected_mean_encoder")


def normalize_shap_array(values, n_samples, n_features, n_outputs):
    """
    Current SHAP returns one-input multi-output explanations as
    samples x features x outputs. This also handles common older list forms.
    """
    if isinstance(values, tuple):
        values = values[0]

    if isinstance(values, list):
        # Older style: list of [samples x features], one per output.
        if len(values) == n_outputs:
            values = np.stack([np.asarray(v) for v in values], axis=-1)
        elif len(values) == 1:
            values = np.asarray(values[0])
        else:
            raise RuntimeError(f"Unexpected SHAP list length: {len(values)}")
    else:
        values = np.asarray(values)

    if values.ndim == 2:
        values = values[:, :, None]

    # Rare older orientation: outputs x samples x features
    if (
        values.ndim == 3
        and values.shape[0] == n_outputs
        and values.shape[1] == n_samples
        and values.shape[2] == n_features
    ):
        values = np.transpose(values, (1, 2, 0))

    if values.shape[0] != n_samples or values.shape[1] != n_features:
        raise RuntimeError(
            f"Unexpected SHAP shape {values.shape}; "
            f"expected first axes ({n_samples}, {n_features})."
        )

    return values


def compute_global_shap(
    vae,
    G,
    train_idx,
    latent_indices,
    outdir,
    seed,
    background_n=32,
    explain_n=32,
    nsamples=32,
    output_chunk=8,
):
    import shap

    shap_dir = outdir / "shap_chunks"
    shap_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    background_n = min(background_n, len(train_idx))
    bg_global = rng.choice(train_idx, size=background_n, replace=False)

    remaining = np.setdiff1d(train_idx, bg_global)
    explain_n = min(explain_n, len(remaining))
    ex_global = rng.choice(remaining, size=explain_n, replace=False)

    background = np.asarray(G[bg_global, :], dtype=np.float32) / 2.0
    explain_data = np.asarray(G[ex_global, :], dtype=np.float32) / 2.0

    np.save(outdir / "shap_background_global_indices.npy", bg_global)
    np.save(outdir / "shap_explain_global_indices.npy", ex_global)

    n_features = G.shape[1]
    total_abs_sum = np.zeros(n_features, dtype=np.float64)
    total_terms = 0

    latent_indices = np.asarray(latent_indices, dtype=int)

    for chunk_start in range(0, len(latent_indices), output_chunk):
        chunk_latents = latent_indices[
            chunk_start : chunk_start + output_chunk
        ]
        tag = "_".join(map(str, chunk_latents.tolist()))
        chunk_file = shap_dir / f"global_abs_shap_latents_{tag}.npy"
        count_file = shap_dir / f"global_abs_shap_latents_{tag}.count.txt"

        if chunk_file.exists() and count_file.exists():
            chunk_mean = np.load(chunk_file)
            denom = int(count_file.read_text().strip())
            total_abs_sum += chunk_mean * denom
            total_terms += denom
            print(f"Reused SHAP chunk for latent dimensions {chunk_latents.tolist()}", flush=True)
            continue

        print(
            f"SHAP latent dimensions {chunk_latents.tolist()} "
            f"(background={background_n}, explain={explain_n}, nsamples={nsamples})",
            flush=True,
        )

        chunk_model = build_selected_mean_encoder(
            vae, G.shape[1], chunk_latents.tolist()
        )

        explainer = shap.GradientExplainer(
            chunk_model,
            background,
            batch_size=min(32, background_n),
        )

        shap_values = explainer.shap_values(
            explain_data,
            nsamples=nsamples,
            rseed=seed + int(chunk_start),
        )

        values = normalize_shap_array(
            shap_values,
            n_samples=explain_data.shape[0],
            n_features=n_features,
            n_outputs=len(chunk_latents),
        )

        # Sum |SHAP| over explained individuals and latent outputs.
        abs_sum = np.abs(values).sum(axis=(0, 2), dtype=np.float64)
        denom = values.shape[0] * values.shape[2]
        chunk_mean = abs_sum / denom

        np.save(chunk_file, chunk_mean)
        count_file.write_text(str(denom))

        total_abs_sum += abs_sum
        total_terms += denom

        del values, shap_values, explainer, chunk_model
        gc.collect()

    if total_terms == 0:
        raise RuntimeError("No SHAP terms were calculated.")

    return total_abs_sum / total_terms


def calculate_r2_one_to_many(G, candidate_idx, comparison_indices, row_chunk=5000):
    """
    Unphased hard-call r^2 = squared Pearson correlation of allele counts.
    Streaming over individuals so no large 40k x many-SNP float copy is required.
    """
    comparison_indices = np.asarray(comparison_indices, dtype=int)
    if comparison_indices.size == 0:
        return np.array([], dtype=np.float64)

    n_total = 0
    sx = 0.0
    sx2 = 0.0
    sy = np.zeros(len(comparison_indices), dtype=np.float64)
    sy2 = np.zeros(len(comparison_indices), dtype=np.float64)
    sxy = np.zeros(len(comparison_indices), dtype=np.float64)

    for start in range(0, G.shape[0], row_chunk):
        stop = min(start + row_chunk, G.shape[0])

        x = np.asarray(G[start:stop, candidate_idx], dtype=np.float64)
        Y = np.asarray(G[start:stop, comparison_indices], dtype=np.float64)

        n = len(x)
        n_total += n
        sx += x.sum()
        sx2 += np.square(x).sum()
        sy += Y.sum(axis=0)
        sy2 += np.square(Y).sum(axis=0)
        sxy += x @ Y

    cov_num = sxy - (sx * sy / n_total)
    varx_num = sx2 - (sx * sx / n_total)
    vary_num = sy2 - (sy * sy / n_total)

    denom = np.sqrt(varx_num * vary_num)
    r = np.divide(
        cov_num,
        denom,
        out=np.zeros_like(cov_num),
        where=denom > 0,
    )
    return r ** 2


def shap_guided_ld_filter(
    ranking_df,
    G,
    target_n=2000,
    r2_threshold=0.20,
    window_bp=500_000,
):
    ranked = ranking_df.sort_values(
        ["SHAP_score", "MAF"],
        ascending=[False, False],
    ).reset_index(drop=True)

    selected = []
    rejected = []
    selected_by_chr = {}

    for _, row in ranked.iterrows():
        chrom = str(row["CHR"])
        bp = int(row["BP"])
        feature_idx = int(row["feature_index"])

        local = [
            rec for rec in selected_by_chr.get(chrom, [])
            if abs(rec["BP"] - bp) <= window_bp
        ]

        max_r2 = 0.0
        blocking_variant = None

        if local:
            local_idx = [rec["feature_index"] for rec in local]
            r2_vals = calculate_r2_one_to_many(G, feature_idx, local_idx)
            j = int(np.argmax(r2_vals))
            max_r2 = float(r2_vals[j])
            blocking_variant = local[j]["VARIANT_ID"]

        record = row.to_dict()
        record["max_r2_with_previously_selected"] = max_r2
        record["highest_LD_selected_variant"] = blocking_variant

        if max_r2 < r2_threshold:
            selected.append(record)
            selected_by_chr.setdefault(chrom, []).append(
                {
                    "feature_index": feature_idx,
                    "BP": bp,
                    "VARIANT_ID": row["VARIANT_ID"],
                }
            )
        else:
            record["LD_removal_reason"] = (
                f"r2={max_r2:.6g} >= {r2_threshold} with {blocking_variant}"
            )
            rejected.append(record)

        if len(selected) >= target_n:
            break

        if (len(selected) + len(rejected)) % 250 == 0:
            print(
                f"LD scan: considered {len(selected)+len(rejected):,}; "
                f"selected {len(selected):,}; rejected {len(rejected):,}",
                flush=True
            )

    selected_df = pd.DataFrame(selected)
    rejected_df = pd.DataFrame(rejected)

    if len(selected_df) < target_n:
        raise RuntimeError(
            f"Only {len(selected_df)} SNPs survived the LD rule; target={target_n}. "
            "Consider a less stringent r2 threshold or verify chromosome/position mapping."
        )

    selected_df = selected_df.iloc[:target_n].copy()
    selected_df["Final_rank"] = np.arange(1, len(selected_df) + 1)

    return selected_df, rejected_df


def plot_shap_top(ranking, plot_dir, n=50):
    top = ranking.nlargest(min(n, len(ranking)), "SHAP_score").copy()
    top = top.sort_values("SHAP_score", ascending=True)

    plt.figure(figsize=(9, max(7, 0.22 * len(top))))
    plt.barh(top["SNP"].astype(str), top["SHAP_score"])
    plt.xlabel("Mean absolute SHAP")
    plt.ylabel("SNP")
    plt.title(f"Top {len(top)} SNPs by latent-space SHAP score")
    plt.tight_layout()
    plt.savefig(plot_dir / "07_top_SHAP_scores.png", dpi=300)
    plt.close()


def plot_ld_selected(selected_df, threshold, plot_dir):
    vals = selected_df["max_r2_with_previously_selected"].astype(float)

    plt.figure(figsize=(7, 5))
    plt.hist(vals, bins=30)
    plt.axvline(threshold, linestyle="--", label=f"r² cutoff = {threshold}")
    plt.xlabel("Maximum local r² with previously selected SNP")
    plt.ylabel("Selected SNP count")
    plt.title("LD of final SHAP-prioritized SNP set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "08_final_selected_LD_distribution.png", dpi=300)
    plt.close()


def evaluate_ground_truth(truth_file, col1, col2, selected_df, outdir):
    truth = pd.read_csv(truth_file)
    if col1 not in truth.columns or col2 not in truth.columns:
        raise ValueError(
            f"Ground-truth file must contain '{col1}' and '{col2}'. "
            f"Available: {truth.columns.tolist()}"
        )

    # Accept either rsID-only truth or full RSID:BP:REF:ALT truth.
    selected_ids = set(selected_df["VARIANT_ID"].astype(str))
    selected_ids.update(selected_df["SNP"].astype(str))

    truth[col1] = truth[col1].astype(str)
    truth[col2] = truth[col2].astype(str)

    truth[f"{col1}_selected"] = truth[col1].isin(selected_ids)
    truth[f"{col2}_selected"] = truth[col2].isin(selected_ids)
    truth["Both_selected"] = (
        truth[f"{col1}_selected"] & truth[f"{col2}_selected"]
    )

    truth.to_csv(outdir / "ground_truth_candidate_recovery.csv", index=False)

    pair_recall = float(truth["Both_selected"].mean()) if len(truth) else np.nan

    summary = {
        "n_truth_pairs": int(len(truth)),
        "n_pairs_both_variants_selected": int(truth["Both_selected"].sum()),
        "candidate_pair_recall": pair_recall,
    }

    with open(outdir / "ground_truth_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Ground-truth candidate recovery:", summary, flush=True)


def main():
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir / "plots"
    plot_dir.mkdir(exist_ok=True)

    n_threads = setup_threads()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    log_versions()
    print("CPU threads requested/used:", n_threads, flush=True)

    with open(outdir / "run_arguments.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ----------------------------------------------------------
    # 1. Genotype cache + metadata
    # ----------------------------------------------------------
    G, sample_ids, meta = prepare_genotype_cache(
        csv_file=Path(args.csv).expanduser().resolve(),
        outdir=outdir,
        chrom=args.chrom,
        snp_map_file=(
            str(Path(args.snp_map).expanduser().resolve())
            if args.snp_map
            else None
        ),
        chunksize=args.csv_chunksize,
    )

    n_samples, n_snps = G.shape
    print(f"Genotype cache: {n_samples:,} x {n_snps:,}", flush=True)

    meta = add_maf(meta, G)
    meta.to_csv(outdir / "variant_metadata_with_MAF.csv", index=False)

    # ----------------------------------------------------------
    # 2. Fixed train / validation / test split
    # ----------------------------------------------------------
    split_dir = outdir / "splits"
    split_dir.mkdir(exist_ok=True)

    train_file = split_dir / "train_idx.npy"
    val_file = split_dir / "val_idx.npy"
    test_file = split_dir / "test_idx.npy"

    if train_file.exists() and val_file.exists() and test_file.exists():
        train_idx = np.load(train_file)
        val_idx = np.load(val_file)
        test_idx = np.load(test_file)
        print("Reusing saved train/validation/test split.", flush=True)
    else:
        all_idx = np.arange(n_samples)
        train_idx, tmp_idx = train_test_split(
            all_idx,
            test_size=0.30,
            random_state=args.seed,
            shuffle=True,
        )
        val_idx, test_idx = train_test_split(
            tmp_idx,
            test_size=0.50,
            random_state=args.seed,
            shuffle=True,
        )
        np.save(train_file, train_idx)
        np.save(val_file, val_idx)
        np.save(test_file, test_idx)

    print(
        f"Split sizes: train={len(train_idx):,}, "
        f"validation={len(val_idx):,}, test={len(test_idx):,}",
        flush=True
    )

    train_seq = GenotypeSequence(
        G, train_idx, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_seq = GenotypeSequence(
        G, val_idx, batch_size=args.batch_size, shuffle=False, seed=args.seed
    )

    # ----------------------------------------------------------
    # 3. VAE
    # ----------------------------------------------------------
    vae = VAE(
        original_dim=n_snps,
        latent_dim=args.latent_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
        l2_reg=args.l2_reg,
        beta=args.beta,
    )

    vae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate)
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_total_loss",
        mode="min",
        patience=args.patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_total_loss",
        mode="min",
        factor=0.5,
        patience=max(2, args.patience // 2),
        min_lr=1e-6,
        verbose=1,
    )

    print("\nTraining VAE...", flush=True)
    history = vae.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=[early_stop, reduce_lr],
        verbose=2,
    )

    model_dir = outdir / "model"
    model_dir.mkdir(exist_ok=True)
    vae.save_weights(model_dir / f"vae_latent_{args.latent_dim}.weights.h5")

    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = np.arange(1, len(history_df) + 1)
    history_df.to_csv(outdir / "training_history.csv", index=False)
    plot_training_history(history_df, plot_dir)

    # ----------------------------------------------------------
    # 4. Overfitting/generalization evaluation
    # ----------------------------------------------------------
    print("\nEvaluating deterministic reconstruction...", flush=True)
    metric_rows = []

    for split_name, idx in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:
        metrics = evaluate_reconstruction(
            vae, G, idx, batch_size=max(args.batch_size, 128)
        )
        metric_rows.append({"split": split_name, **metrics})
        print(split_name, metrics, flush=True)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(outdir / "reconstruction_metrics.csv", index=False)

    train_mse = float(
        metrics_df.loc[metrics_df["split"] == "train", "MSE"].iloc[0]
    )
    test_mse = float(
        metrics_df.loc[metrics_df["split"] == "test", "MSE"].iloc[0]
    )
    generalization = {
        "test_minus_train_MSE": test_mse - train_mse,
        "test_over_train_MSE": (
            test_mse / train_mse if train_mse > 0 else None
        ),
    }
    with open(outdir / "generalization_gap.json", "w") as f:
        json.dump(generalization, f, indent=2)

    plot_split_metrics(metrics_df, plot_dir)

    # ----------------------------------------------------------
    # 5. Active latent dimensions
    # ----------------------------------------------------------
    print("\nCalculating latent variance...", flush=True)
    z_train = encode_z_mean(
        vae, G, train_idx, batch_size=max(args.batch_size, 128)
    )
    latent_var = np.var(z_train, axis=0)

    latent_var_df = pd.DataFrame(
        {
            "latent_index": np.arange(args.latent_dim),
            "variance": latent_var,
        }
    ).sort_values("variance", ascending=False)
    latent_var_df.to_csv(outdir / "latent_variance.csv", index=False)
    plot_latent_variance(latent_var, plot_dir)

    active = np.where(latent_var > args.latent_var_threshold)[0]

    if len(active) == 0:
        print(
            "WARNING: no latent dimensions passed the variance threshold; "
            "falling back to all latent dimensions.",
            flush=True,
        )
        active = np.arange(args.latent_dim)

    if args.max_shap_latents > 0 and len(active) > args.max_shap_latents:
        order = active[np.argsort(latent_var[active])[::-1]]
        shap_latents = order[:args.max_shap_latents]
    else:
        shap_latents = active

    np.save(outdir / "SHAP_latent_indices.npy", shap_latents)

    print(
        f"Active latent dimensions: {len(active)}/{args.latent_dim}; "
        f"SHAP will explain {len(shap_latents)}.",
        flush=True,
    )

    del z_train
    gc.collect()

    # ----------------------------------------------------------
    # 6. Global SHAP
    # ----------------------------------------------------------
    print("\nCalculating global SHAP scores...", flush=True)
    shap_scores = compute_global_shap(
        vae=vae,
        G=G,
        train_idx=train_idx,
        latent_indices=shap_latents,
        outdir=outdir,
        seed=args.seed,
        background_n=args.shap_background,
        explain_n=args.shap_explain,
        nsamples=args.shap_nsamples,
        output_chunk=args.shap_output_chunk,
    )

    ranking = meta.copy()
    ranking["SHAP_score"] = shap_scores
    ranking = ranking.sort_values(
        ["SHAP_score", "MAF"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranking["SHAP_rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(outdir / "all_SNP_SHAP_ranking.csv", index=False)
    plot_shap_top(ranking, plot_dir, n=50)

    # ----------------------------------------------------------
    # 7. SHAP-guided local LD filtering
    # ----------------------------------------------------------
    print(
        f"\nLD filtering: r² < {args.ld_r2}, "
        f"window +/-{args.ld_window:,} bp, target={args.final_n:,}",
        flush=True,
    )

    final_snps, ld_rejected = shap_guided_ld_filter(
        ranking_df=ranking,
        G=G,
        target_n=args.final_n,
        r2_threshold=args.ld_r2,
        window_bp=args.ld_window,
    )

    final_snps.to_csv(
        outdir / f"FINAL_{args.final_n}_SHAP_LD_filtered_SNPs.csv",
        index=False,
    )
    ld_rejected.to_csv(outdir / "SNPs_removed_due_to_high_LD.csv", index=False)

    np.savetxt(
        outdir / f"FINAL_{args.final_n}_VARIANT_IDs.txt",
        final_snps["VARIANT_ID"].astype(str).to_numpy(),
        fmt="%s",
    )
    np.savetxt(
        outdir / f"FINAL_{args.final_n}_rsIDs.txt",
        final_snps["SNP"].astype(str).to_numpy(),
        fmt="%s",
    )

    final_indices = final_snps["feature_index"].astype(int).to_numpy()
    np.save(outdir / f"FINAL_{args.final_n}_feature_indices.npy", final_indices)

    # Materialize final 0/1/2 genotype matrix; at 40k x 2k this is ~80 MB as int8.
    final_G = np.asarray(G[:, final_indices], dtype=np.int8)
    np.save(outdir / f"genotype_FINAL_{args.final_n}.npy", final_G)

    final_df = pd.DataFrame(
        final_G,
        index=sample_ids,
        columns=final_snps["VARIANT_ID"].astype(str).to_list(),
    )
    final_df.index.name = "sample_id"
    final_df.to_csv(outdir / f"genotype_FINAL_{args.final_n}.csv")

    plot_ld_selected(final_snps, args.ld_r2, plot_dir)

    # ----------------------------------------------------------
    # 8. Optional simulated ground truth
    # ----------------------------------------------------------
    if args.ground_truth:
        evaluate_ground_truth(
            truth_file=args.ground_truth,
            col1=args.truth_col1,
            col2=args.truth_col2,
            selected_df=final_snps,
            outdir=outdir,
        )

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    summary = {
        "n_samples": int(n_samples),
        "n_input_snps": int(n_snps),
        "latent_dim": int(args.latent_dim),
        "n_active_latents": int(len(active)),
        "n_shap_latents": int(len(shap_latents)),
        "final_n_snps": int(len(final_snps)),
        "ld_r2_threshold": float(args.ld_r2),
        "ld_window_bp": int(args.ld_window),
        "max_selected_local_r2": float(
            final_snps["max_r2_with_previously_selected"].max()
        ),
        "generalization": generalization,
    }

    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print("PIPELINE COMPLETE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("Final SNP table:", outdir / f"FINAL_{args.final_n}_SHAP_LD_filtered_SNPs.csv", flush=True)
    print("Final genotype:", outdir / f"genotype_FINAL_{args.final_n}.npy", flush=True)
    print("Plots:", plot_dir, flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
