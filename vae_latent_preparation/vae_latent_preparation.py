from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import umap
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def sample_latent(z_mean: tf.Tensor, z_logvar: tf.Tensor) -> tf.Tensor:
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
        z = sample_latent(z_mean, z_logvar)
        x_hat = self.decoder(z, training=training)

        recon_loss = tf.keras.losses.MeanSquaredError()(x, x_hat)
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


def plot_training_losses(history, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("recon_loss", []), label="Train Recon Loss")
    plt.plot(history.history.get("val_recon_loss", []), label="Validation Recon Loss")
    plt.xlabel("Epoch")
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
    plt.title("Reconstruction Error Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_umap_projection(latent: np.ndarray, output_path: Path, seed: int) -> None:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed)
    latent_2d = reducer.fit_transform(latent)

    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7, s=10)
    plt.title("UMAP Projection of Latent Space")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE and export latent space.")
    parser.add_argument("--input_csv", required=True, help="Input CSV with rows=samples and columns=features.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--index_col", type=int, default=0, help="Index column for input CSV.")
    parser.add_argument("--latent_dim", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--kl_max_weight", type=float, default=2.0)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--lr_plateau_patience", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.random_seed)

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print(f"Loading data from: {input_csv}")
    data = pd.read_csv(input_csv, index_col=args.index_col)

    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    train_data, val_data = train_test_split(
        data_normalized,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    vae = VAE(original_dim=data_normalized.shape[1], latent_dim=args.latent_dim)
    vae.compile(optimizer=Adam(args.learning_rate))

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

    latent_df = pd.DataFrame(
        latent,
        index=data.index,
        columns=[f"latent_{i}" for i in range(latent.shape[1])]
    )
    latent_df.to_csv(output_dir / "latent_space.csv")

    val_recon = vae.predict(val_data, verbose=0)
    val_recon_error = np.mean((val_data - val_recon) ** 2, axis=1)
    plot_reconstruction_error_hist(val_recon_error, output_dir / "reconstruction_error_hist.png")

    full_recon = vae.predict(data_normalized, verbose=0)
    recon_mse = mean_squared_error(data_normalized, full_recon)

    plot_umap_projection(latent, output_dir / "latent_umap.png", args.random_seed)

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

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
