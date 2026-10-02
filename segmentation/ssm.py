"""SSM loading, normalization, and fusion utilities for the structure baseline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MSA_ROOT = REPO_ROOT.parent

SSM_FEATURE_TO_KEY = {
    "stm": "ssm_stm",
    "mfcc": "ssm_mfcc",
    "chroma": "ssm_chroma",
    "cens": "ssm_cens",
    "arrangement": "ssm_arrangement",
    "vocal": "ssm_vocal",
    "bass": "ssm_bass",
    "tonnetz": "ssm_tonnetz",
    "density": "ssm_density",
    "musicfm": "ssm_musicfm",
    "muq": "ssm_muq",
    "matpac": "ssm_matpac",
    "repcl": "ssm_repcl",
    "mule": "ssm_mule",
    "fused": "ssm_fused",
}


def add_repo_to_path(path: Path) -> None:
    path = path.resolve()
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def normalize_ssm(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if matrix.size == 0:
        return matrix
    matrix = 0.5 * (matrix + matrix.T)
    lo, hi = np.percentile(matrix, [5, 95])
    if hi > lo:
        matrix = np.clip((matrix - lo) / (hi - lo), 0.0, 1.0)
    else:
        matrix = np.zeros_like(matrix)
    np.fill_diagonal(matrix, 1.0)
    return matrix.astype(np.float32)


def load_ssm(data: np.lib.npyio.NpzFile, feature: str) -> np.ndarray:
    key = SSM_FEATURE_TO_KEY.get(feature.lower(), f"ssm_{feature.lower()}")
    if key not in data:
        available = sorted(name[4:] for name in data.files if name.startswith("ssm_"))
        raise ValueError(f"SSM feature {feature!r} not found. Available SSMs: {', '.join(available)}")
    return normalize_ssm(np.asarray(data[key]))


def fuse_ssms_with_graphditty(matrices: list[np.ndarray], fusion_cfg: dict[str, Any]) -> np.ndarray:
    add_repo_to_path(MSA_ROOT / "GraphDitty")
    from SimilarityFusion import doSimilarityFusionWs

    if len(matrices) == 1:
        return normalize_ssm(matrices[0])
    shapes = {matrix.shape for matrix in matrices}
    if len(shapes) != 1:
        raise ValueError(f"Cannot fuse SSMs with different shapes: {sorted(shapes)}")
    k_nearest = int(fusion_cfg.get("k", fusion_cfg.get("K", 5)))
    if k_nearest < 0:
        k_nearest = max(1, int(np.sqrt(matrices[0].shape[0])))
    fused = doSimilarityFusionWs(
        [normalize_ssm(matrix) for matrix in matrices],
        K=k_nearest,
        niters=int(fusion_cfg.get("niters", 20)),
        reg_diag=float(fusion_cfg.get("reg_diag", 1.0)),
        reg_neighbs=float(fusion_cfg.get("reg_neighbs", 0.5)),
        do_animation=False,
        verboseTimes=bool(fusion_cfg.get("verbose", False)),
    )
    return normalize_ssm(np.asarray(fused))


def resolve_segmentation_ssm(
    data: np.lib.npyio.NpzFile,
    ssm_features: Iterable[str],
    fusion_cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    features = [str(feature).strip().lower() for feature in ssm_features if str(feature).strip()]
    if not features:
        raise ValueError("At least one `segmentation.ssm_features` value is required.")
    matrices = [load_ssm(data, feature) for feature in features]
    matrix = matrices[0] if len(matrices) == 1 else fuse_ssms_with_graphditty(matrices, fusion_cfg)
    return normalize_ssm(matrix), features


def build_ssm_runs(
    ssm_features: Iterable[str],
    fusion_combos: Iterable[Iterable[str]],
) -> list[tuple[str, list[str]]]:
    """Expand config into one independent SSM run per entry.

    Each feature in `ssm_features` becomes its own single-feature run (they
    are no longer fused together implicitly). Each combo listed under
    `graphditty_fusion.features` becomes an additional fused run over just
    those features. Returns a list of `(run_label, feature_list)` pairs.
    """
    runs: list[tuple[str, list[str]]] = []
    seen_labels: set[str] = set()

    def add_run(label: str, features: list[str]) -> None:
        if label in seen_labels:
            raise ValueError(
                f"Duplicate SSM run label {label!r}; check for repeated entries in "
                "`ssm_features` / `graphditty_fusion.features`."
            )
        seen_labels.add(label)
        runs.append((label, features))

    for feature in ssm_features:
        feature = str(feature).strip().lower()
        if feature:
            add_run(feature, [feature])

    for combo in fusion_combos:
        features = [str(feature).strip().lower() for feature in combo if str(feature).strip()]
        if len(features) < 2:
            raise ValueError(f"`graphditty_fusion.features` entries must list 2+ features, got {combo!r}.")
        add_run("fused_" + "-".join(features), features)

    if not runs:
        raise ValueError(
            "No SSM runs configured: set `segmentation.ssm_features` and/or "
            "`graphditty_fusion.features` in the YAML config."
        )
    return runs
