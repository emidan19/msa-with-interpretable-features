"""Probe how much classical MIR information is recoverable from SSL embeddings.

Trains a shallow MLP regressor (PyTorch) to predict a classical MIR descriptor
(CENS, MFCC or STM) from a frame-level SSL embedding (MusicFM, MuQ or MATPAC),
evaluated with 5-fold cross-validation and reported via R².

Reuses the feature-bundle discovery helpers from `distribution_validation.py`
(same directory) — no data-loading logic is duplicated here.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from distribution_validation import (  # noqa: E402
    discover_feature_bundles,
    load_feature_bundle,
    resolve_project_dir,
)

RANDOM_SEED = 42
TARGET_DESCRIPTORS = ("chroma", "cens", "mfcc", "stm")
EMBEDDING_CHOICES = ("musicfm", "muq", "matpac", "repcl")
DEFAULT_EMBEDDING = "musicfm"

N_FOLDS = 5
VAL_FRACTION = 0.1  # carved out of each fold's training split, for early stopping

HIDDEN_DIM = 256
DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 1000
PATIENCE = 100


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings_and_targets(
    npz_paths: list[Path],
    embedding_name: str,
    target_names: tuple[str, ...] = TARGET_DESCRIPTORS,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Pool beat-synced embeddings and MIR targets across all available songs.

    Returns embeddings of shape (N, D), one target matrix of shape (N, F) per
    descriptor, and a song_ids array of shape (N,) identifying which song each
    frame came from (used to group frames by song during cross-validation, so
    that no song's frames leak across train/test splits).
    """
    embedding_key = f"beat_{embedding_name}"
    embedding_chunks: list[np.ndarray] = []
    target_chunks: dict[str, list[np.ndarray]] = {name: [] for name in target_names}
    song_id_chunks: list[np.ndarray] = []

    for song_idx, npz_path in enumerate(npz_paths):
        bundle = load_feature_bundle(npz_path)
        required_keys = [embedding_key] + [f"beat_{name}" for name in target_names]
        if not all(key in bundle for key in required_keys):
            continue
        embedding_chunk = np.asarray(bundle[embedding_key], dtype=np.float32).T
        embedding_chunks.append(embedding_chunk)
        for name in target_names:
            target_chunks[name].append(np.asarray(bundle[f"beat_{name}"], dtype=np.float32).T)
        song_id_chunks.append(np.full(embedding_chunk.shape[0], song_idx, dtype=np.int64))

    if not embedding_chunks:
        raise FileNotFoundError(
            f"Ningún bundle expone '{embedding_key}' junto con {target_names}."
        )

    embeddings = np.concatenate(embedding_chunks, axis=0)
    targets = {name: np.concatenate(chunks, axis=0) for name, chunks in target_chunks.items()}
    song_ids = np.concatenate(song_id_chunks, axis=0)
    return embeddings, targets, song_ids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> nn.Module:
    """Train with Adam/MSE and early stopping on validation loss, restoring the best weights."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True,
    )
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    best_val_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": None,
        "best_val_loss": None,
    }

    for epoch in range(max_epochs):

        # ==========================
        # Training
        # ==========================
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ==========================
        # Validation
        # ==========================
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # ==========================
        # Early stopping
        # ==========================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }

            history["best_epoch"] = epoch
            history["best_val_loss"] = val_loss

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, history

def save_model_history(history: dict, exp_dir: Path, descriptor_name: str, fold_name: str) -> None:
    plt.figure(figsize=(7, 4))

    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")

    plt.axvline(
        history["best_epoch"],
        color="red",
        linestyle="--",
        label=f"Best epoch ({history['best_epoch']})"
    )

    plt.scatter(
        history["best_epoch"],
        history["best_val_loss"],
        color="red",
        zorder=5
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training history")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_dir= exp_dir / f"{descriptor_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"training_history_{fold_name}.png")

def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    target_scaler: StandardScaler,
    device: torch.device,
) -> float:
    """R² on the original (unscaled) target scale."""
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_true = target_scaler.inverse_transform(y_test_scaled)
    return float(r2_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_descriptor(
    embeddings: np.ndarray,
    targets: np.ndarray,
    song_ids: np.ndarray,
    device: torch.device,
    n_splits: int = N_FOLDS,
    random_state: int = RANDOM_SEED,
) -> tuple[list[float], list[nn.Module]]:
    """Song-grouped CV: all frames of a song stay entirely in train or test,
    since adjacent beat frames from the same song are highly correlated and
    would otherwise leak information between splits."""
    kfold = GroupKFold(n_splits=n_splits)
    fold_scores: list[float] = []
    fold_models: list[nn.Module] = []
    fold_histories: list[dict] = []

    for train_idx, test_idx in kfold.split(embeddings, groups=song_ids):
        X_train_raw, X_test_raw = embeddings[train_idx], embeddings[test_idx]
        y_train_raw, y_test_raw = targets[train_idx], targets[test_idx]
        train_song_ids = song_ids[train_idx]

        embedding_scaler = StandardScaler().fit(X_train_raw)
        target_scaler = StandardScaler().fit(y_train_raw)

        X_train_scaled = embedding_scaler.transform(X_train_raw).astype(np.float32)
        X_test_scaled = embedding_scaler.transform(X_test_raw).astype(np.float32)
        y_train_scaled = target_scaler.transform(y_train_raw).astype(np.float32)
        y_test_scaled = target_scaler.transform(y_test_raw).astype(np.float32)

        splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=random_state)
        train_sub_idx, val_sub_idx = next(splitter.split(X_train_scaled, groups=train_song_ids))
        X_train_sub, X_val_sub = X_train_scaled[train_sub_idx], X_train_scaled[val_sub_idx]
        y_train_sub, y_val_sub = y_train_scaled[train_sub_idx], y_train_scaled[val_sub_idx]

        model = MLPRegressor(input_dim=embeddings.shape[1], output_dim=targets.shape[1])
        model, history = train_mlp(model, X_train_sub, y_train_sub, X_val_sub, y_val_sub, device=device)

        fold_scores.append(evaluate_model(model, X_test_scaled, y_test_scaled, target_scaler, device))
        fold_models.append(model)
        fold_histories.append(history)

    return fold_scores, fold_models, fold_histories


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def build_results_table(all_fold_scores: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for descriptor_name, scores in all_fold_scores.items():
        row = {"Descriptor": descriptor_name.upper()}
        row.update({f"Fold {i + 1}": score for i, score in enumerate(scores)})
        row["Mean R2"] = float(np.mean(scores))
        row["Std R2"] = float(np.std(scores))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_experiment(
    embedding_name: str = DEFAULT_EMBEDDING,
    target_names: tuple[str, ...] = TARGET_DESCRIPTORS,
    output_dir: Path | None = None,
    features_dir: Path | None = None,
) -> pd.DataFrame:
    set_global_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_dir = resolve_project_dir()
    npz_paths = discover_feature_bundles(features_dir)
    if not npz_paths:
        search_location = features_dir if features_dir is not None else f"{project_dir}/feature_outputs"
        raise FileNotFoundError(f"No se encontraron *_features.npz bajo {search_location}.")

    embeddings, targets, song_ids = load_embeddings_and_targets(npz_paths, embedding_name, target_names)
    print(f"Embedding: {embedding_name} -> shape {embeddings.shape} (canciones disponibles: {len(npz_paths)})")

    output_dir = output_dir or (SCRIPT_DIR / "outputs" / embedding_name)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_fold_scores: dict[str, list[float]] = {}
    for descriptor_name in target_names:
        target_matrix = targets[descriptor_name]
        print(f"\n=== {descriptor_name.upper()} (dim={target_matrix.shape[1]}) ===")

        fold_scores, fold_models, fold_histories = cross_validate_descriptor(
            embeddings, target_matrix, song_ids, device=device
        )
        for fold_idx, score in enumerate(fold_scores, start=1):
            print(f"  Fold {fold_idx}: R2 = {score:.4f}")
        print(f"  Mean R2 = {np.mean(fold_scores):.4f} | Std R2 = {np.std(fold_scores):.4f}")

        all_fold_scores[descriptor_name] = fold_scores
        for fold_idx, model in enumerate(fold_models, start=1):
            torch.save(model.state_dict(), models_dir / f"{descriptor_name}_fold{fold_idx}.pt")

        for fold_idx, history in enumerate(fold_histories, start=1):
            save_model_history(history, output_dir,descriptor_name, f"fold_{fold_idx}")

    results_table = build_results_table(all_fold_scores)
    csv_path = output_dir / f"r2_results_{embedding_name}.csv"
    results_table.to_csv(csv_path, index=False)
    print(f"\nTabla de resultados guardada en {csv_path}")
    print(results_table.to_string(index=False))

    return results_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING, choices=EMBEDDING_CHOICES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--features-dir", type=Path, default=None, help="Directorio donde buscar los *_features.npz (por defecto se autodetecta).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(embedding_name=args.embedding, output_dir=args.output_dir, features_dir=args.features_dir)


if __name__ == "__main__":
    main()
