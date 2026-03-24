from __future__ import annotations

import os

# Set thread caps early
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import umap
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GaussianNoise,
    LeakyReLU,
)
from tensorflow.keras.optimizers import Adam


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


class Encoder(layers.Layer):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.noise = GaussianNoise(0.1)

        self.dense1 = Dense(1024, kernel_regularizer=regularizers.l2(1e-4))
        self.lrelu1 = LeakyReLU(0.1)
        self.bn1 = BatchNormalization()
        self.drop1 = Dropout(0.2)

        self.dense2 = Dense(512, kernel_regularizer=regularizers.l2(1e-4))
        self.lrelu2 = LeakyReLU(0.1)
        self.bn2 = BatchNormalization()
        self.drop2 = Dropout(0.2)

        self.dense3 = Dense(256, kernel_regularizer=regularizers.l2(1e-4))
        self.lrelu3 = LeakyReLU(0.1)
        self.bn3 = BatchNormalization()
        self.drop3 = Dropout(0.2)

        self.z_mean = Dense(latent_dim)
        self.z_logvar = Dense(latent_dim)

    def call(self, x: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.noise(x, training=training)

        x = self.lrelu1(self.dense1(x))
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)

        x = self.lrelu2(self.dense2(x))
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)

        x = self.lrelu3(self.dense3(x))
        x = self.bn3(x, training=training)
        x = self.drop3(x, training=training)

        return self.z_mean(x), self.z_logvar(x)


def sample(z_mean: tf.Tensor, z_logvar: tf.Tensor) -> tf.Tensor:
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_logvar) * eps


class Decoder(layers.Layer):
    def __init__(self, original_dim: int):
        super().__init__()
        self.dense1 = Dense(256, kernel_regularizer=regularizers.l2(1e-4))
        self.lrelu1 = LeakyReLU(0.1)
        self.bn1 = BatchNormalization()
        self.drop1 = Dropout(0.2)

        self.dense2 = Dense(512, kernel_regularizer=regularizers.l2(1e-4))
        self.lrelu2 = LeakyReLU(0.1)
        self.bn2 = BatchNormalization()
        self.drop2 = Dropout(0.2)

        self.dense3 = Dense(1024, kernel_regularizer=regularizers.l2(1e-4))
        self.lrelu3 = LeakyReLU(0.1)
        self.bn3 = BatchNormalization()
        self.drop3 = Dropout(0.2)

        self.out = Dense(original_dim)

    def call(self, z: tf.Tensor, training: bool = False) -> tf.Tensor:
        z = self.lrelu1(self.dense1(z))
        z = self.bn1(z, training=training)
        z = self.drop1(z, training=training)

        z = self.lrelu2(self.dense2(z))
        z = self.bn2(z, training=training)
        z = self.drop2(z, training=training)

        z = self.lrelu3(self.dense3(z))
        z = self.bn3(z, training=training)
        z = self.drop3(z, training=training)

        return self.out(z)


class VAE(tf.keras.Model):
    def __init__(self, original_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(original_dim)

        self.kl_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self.recon_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.recon_tracker, self.kl_tracker]

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        z_mean, z_logvar = self.encoder(x, training=training)
        z = sample(z_mean, z_logvar)
        x_hat = self.decoder(z, training=training)

        mse_fn = tf.keras.losses.MeanSquaredError()
        recon_loss = mse_fn(x, x_hat)

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)
        )

        self.add_loss(recon_loss)
        self.add_loss(self.kl_weight * kl_loss)

        self.recon_tracker.update_state(recon_loss)
        self.kl_tracker.update_state(kl_loss)

        return x_hat


class KLScheduler(tf.keras.callbacks.Callback):
    def __init__(self, max_weight: float = 2.0, epochs: int = 50):
        super().__init__()
        self.max_weight = max_weight
        self.epochs = epochs

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        new_w = self.max_weight * epoch / self.epochs
        self.model.kl_weight.assign(min(self.max_weight, new_w))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_training_losses(history, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("recon_loss", []), label="Train Recon Loss")
    plt.plot(history.history.get("val_recon_loss", []), label="Validation Recon Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("VAE Training vs Validation Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_reconstruction_error_hist(recon_error: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(recon_error, bins=50, edgecolor="black")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Number of Samples")
    plt.title("Reconstruction Error Distribution (Validation Set)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_umap(latent_2d: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7, s=10)
    plt.title("UMAP Projection of Latent Space")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _compute_shap_for_latent(
    i: int,
    X: np.ndarray,
    latent: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    top_k: int = 100,
    rf_n_estimators: int = 100,
    rf_max_depth: int = 10,
    seed: int = 42,
) -> Dict:
    print(f"[LATENT {i}] fitting RF & SHAP...")

    rf = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        n_jobs=1,
        random_state=seed,
    )
    rf.fit(X, latent[:, i])

    explainer = shap.Explainer(rf, X, feature_names=feature_names)
    sv = explainer(X)
    mean_abs = np.abs(sv.values).mean(axis=0)

    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    top_features = [(feature_names[j], float(mean_abs[j])) for j in top_idx]

    df = pd.DataFrame(top_features, columns=["Feature", "Mean_SHAP"])
    csv_path = output_dir / f"latent_{i}_top_features.csv"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(14, 8))
    plt.barh(df["Feature"].iloc[::-1], df["Mean_SHAP"].iloc[::-1])
    plt.xlabel("Mean |SHAP|")
    plt.title(f"Top {top_k} Features - Latent Dimension {i}")
    plt.tight_layout()
    plt.savefig(output_dir / f"latent_{i}_shap_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {"latent_dim": i, "top_features": top_features}


def detect_n_jobs(default_cap: int = 16) -> int:
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    return max(1, min(n, default_cap))


def shap_feature_importance_all_latents(
    X: np.ndarray,
    latent: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    n_jobs: int | None = None,
    top_k: int = 100,
    prefer_threads: bool = True,
    rf_n_estimators: int = 100,
    rf_max_depth: int = 10,
    seed: int = 42,
) -> List[Dict]:
    ensure_dir(output_dir)
    if n_jobs is None:
        n_jobs = detect_n_jobs()

    prefer = "threads" if prefer_threads else "processes"

    results = Parallel(n_jobs=n_jobs, prefer=prefer, verbose=10)(
        delayed(_compute_shap_for_latent)(
            i=i,
            X=X,
            latent=latent,
            feature_names=feature_names,
            output_dir=output_dir,
            top_k=top_k,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            seed=seed,
        )
        for i in range(latent.shape[1])
    )
    return results


def aggregate_shap_snp_csvs(shap_csv_dir: Path, output_csv: str = "aggregated_top_snps.csv") -> pd.DataFrame:
    all_snps: Dict[str, float] = {}

    for file in shap_csv_dir.iterdir():
        if file.name.startswith("latent_") and file.name.endswith("_top_features.csv"):
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                snp = str(row["Feature"])
                shap_val = float(row["Mean_SHAP"])
                if snp not in all_snps or shap_val > all_snps[snp]:
                    all_snps[snp] = shap_val

    agg_df = pd.DataFrame(list(all_snps.items()), columns=["Feature", "Max_Mean_SHAP"])
    agg_df = agg_df.sort_values(by="Max_Mean_SHAP", ascending=False)
    agg_df.to_csv(shap_csv_dir / output_csv, index=False)

    print(f"Aggregated {len(agg_df)} unique SNPs saved to: {shap_csv_dir / output_csv}")
    return agg_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAE latent space preparation + RF/SHAP feature ranking.")
    parser.add_argument("--input_csv", required=True, help="Path to input genotype/features CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs.")
    parser.add_argument("--index_col", type=int, default=0, help="CSV index column. Use 0 if first column is sample ID.")
    parser.add_argument("--latent_dim", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--kl_max_weight", type=float, default=2.0)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--lr_plateau_patience", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=100, help="Top SHAP features saved per latent dimension.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Parallel jobs for per-latent RF/SHAP.")
    parser.add_argument("--rf_n_estimators", type=int, default=100)
    parser.add_argument("--rf_max_depth", type=int, default=10)
    parser.add_argument("--prefer_threads", action="store_true", help="Use joblib threads instead of processes.")
    parser.add_argument("--skip_shap", action="store_true", help="Skip RF/SHAP feature importance step.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.random_seed)

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print(f"Loading data from: {input_csv}")
    data = pd.read_csv(input_csv, sep=",", index_col=args.index_col)

    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    vae = VAE(original_dim=data_normalized.shape[1], latent_dim=args.latent_dim)
    vae.compile(optimizer=Adam(args.learning_rate))

    train_data, val_data = train_test_split(
        data_normalized,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.early_stop_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=args.lr_plateau_patience, min_lr=1e-6),
        KLScheduler(max_weight=args.kl_max_weight, epochs=args.epochs),
    ]

    history = vae.fit(
        train_data,
        train_data,
        validation_data=(val_data, val_data),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    plot_training_losses(history, output_dir / "training_losses.png")

    z_mean, _ = vae.encoder(data_normalized, training=False)
    latent = z_mean.numpy()
    print("Original data shape:", data_normalized.shape)
    print("Encoded data shape:", latent.shape)

    encoded_df = pd.DataFrame(latent, index=data.index)
    encoded_df.to_csv(output_dir / "latent_space.csv")

    val_recon = vae.predict(val_data, verbose=0)
    val_recon_error = np.mean((val_data - val_recon) ** 2, axis=1)
    plot_reconstruction_error_hist(val_recon_error, output_dir / "reconstruction_error_hist.png")

    x_reconstructed = vae.predict(data_normalized, verbose=0)
    recon_mse = mean_squared_error(data_normalized, x_reconstructed)
    print("Reconstruction MSE:", recon_mse)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=args.random_seed)
    latent_2d = reducer.fit_transform(latent)
    plot_umap(latent_2d, output_dir / "latent_umap.png")

    summary = {
        "input_csv": str(input_csv),
        "n_samples": int(data.shape[0]),
        "n_features": int(data.shape[1]),
        "latent_dim": int(args.latent_dim),
        "epochs_requested": int(args.epochs),
        "epochs_completed": int(len(history.history.get("loss", []))),
        "reconstruction_mse_full_data": float(recon_mse),
    }

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.skip_shap:
        feature_names = data.columns.astype(str).tolist()
        shap_feature_importance_all_latents(
            X=data_normalized,
            latent=latent,
            feature_names=feature_names,
            output_dir=output_dir,
            n_jobs=args.n_jobs,
            top_k=args.top_k,
            prefer_threads=args.prefer_threads,
            rf_n_estimators=args.rf_n_estimators,
            rf_max_depth=args.rf_max_depth,
            seed=args.random_seed,
        )
        aggregate_shap_snp_csvs(output_dir)

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
