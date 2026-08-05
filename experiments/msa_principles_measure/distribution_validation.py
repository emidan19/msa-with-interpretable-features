"""Compare MSA principle metrics across similarity functions over precomputed feature bundles."""

from __future__ import annotations

import argparse
from html import parser
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_loading import load_runtime_config
from feature_types import build_section_segment_ids, build_ssm
from msa_principles import (
    compute_homogeneity_scores,
    compute_novelty_scores,
    compute_repetition_scores,
    extract_ssm_regions,
)

SIMILARITY_TYPES = ("cosine", "autocorrelation", "rbf")
METRIC_NAMES = ("homogeneity", "repetition", "novelty")
REGION_SCORE_NAMES = ("section_mean", "repetition_mean", "background_mean")

DEFAULT_RANDOM_SEED = 123
DEFAULT_NUM_GT_EXAMPLES = 3
DEFAULT_N_SIM = 4
DEFAULT_OUTPUT_FILENAME = "ssm_simulation_examples.png"
MAX_N_SIM_SHOW = 10


def resolve_project_dir(start_dir: Path | None = None) -> Path:
    candidates = []
    if start_dir is not None:
        candidates.append(start_dir.resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(PROJECT_DIR)
    for candidate in candidates:
        if (candidate / "pipeline.py").exists():
            return candidate
    return PROJECT_DIR


def discover_feature_bundles(search_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if search_root is not None:
        candidates.append(search_root.expanduser().resolve())
    else:
        candidates.extend(
            [
                PROJECT_DIR / "feature_outputs",
                PROJECT_DIR / "outputs",
                PROJECT_DIR / "out",
                PROJECT_DIR,
            ]
        )

    for root in candidates:
        if root.is_file():
            return [root]
        if not root.exists():
            continue

        matches = sorted(root.glob("*_features.npz"))
        if matches:
            return matches

        for child in sorted(root.iterdir()):
            if child.is_dir():
                matches = sorted(child.glob("*_features.npz"))
                if matches:
                    return matches

    return []


def load_feature_bundle(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def list_feature_names(bundle: dict[str, Any]) -> list[str]:
    feature_names: list[str] = []
    for key in sorted(bundle):
        if not key.startswith("beat_"):
            continue
        value = np.asarray(bundle[key])
        if value.ndim == 2:
            feature_names.append(key[5:])
    return feature_names


def _build_gt_matrix(bundle: dict[str, Any]) -> np.ndarray:
    if "ssm_sections" not in bundle:
        raise KeyError("The feature bundle does not contain an `ssm_sections` entry.")
    return (np.asarray(bundle["ssm_sections"]) > 0).astype(np.uint8)


def _build_segment_ids(bundle: dict[str, Any]) -> np.ndarray | None:
    """Recover per-frame segment instance ids from the raw annotation, if saved.

    `ssm_sections` only encodes label equality, so two temporally adjacent
    segments sharing the same normalized label (very common: repeated verse
    or chorus) are pixel-indistinguishable from a single larger section. The
    raw `sections` list (distinct per annotated segment) plus `beat_boundaries`
    lets us recover that distinction; see `extract_ssm_regions`'s `segment_ids`.
    """
    if "sections" not in bundle or "beat_boundaries" not in bundle:
        return None
    sections = [(float(start), float(end), str(label)) for start, end, label in bundle["sections"]]
    return build_section_segment_ids(sections, np.asarray(bundle["beat_boundaries"]))


def _build_region_masks(ssm_gt: np.ndarray, segment_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sections, repetitions = extract_ssm_regions(ssm_gt, segment_ids=segment_ids)
    section_mask = np.zeros_like(ssm_gt, dtype=bool)
    repetition_mask = np.zeros_like(ssm_gt, dtype=bool)
    for region in sections:
        section_mask[tuple(region.pixels.T)] = True
    for region in repetitions:
        repetition_mask[tuple(region.pixels.T)] = True
    background_mask = ~ssm_gt.astype(bool)
    return section_mask, repetition_mask, background_mask


def compute_section_length_stats(npz_paths: list[Path]) -> dict[str, Any]:
    """Measure the section length mean and standard deviation across all SSMs.
    
    Args:
        npz_paths: List of paths to feature bundle NPZ files.
        
    Returns:
        Dictionary containing:
            - 'mean': Mean section length (in pixels)
            - 'std': Standard deviation of section lengths
            - 'lengths': Array of all individual section lengths
            - 'section_count': Total number of sections analyzed
            - 'min': Minimum section length
            - 'max': Maximum section length
            - 'median': Median section length
    """
    all_section_lengths: list[int] = []
    all_repetition_section_ratios: list[float] = []
    
    for npz_path in npz_paths:
        bundle = load_feature_bundle(npz_path)
        try:
            ssm_gt = _build_gt_matrix(bundle)
        except KeyError:
            continue

        segment_ids = _build_segment_ids(bundle)
        sections, repetitions = extract_ssm_regions(ssm_gt, segment_ids=segment_ids)
        # N_rep / (2 * N_sec) because each rep appears twice at SSM (above and below diagonal)
        all_repetition_section_ratios.append(len(repetitions) / (2 * len(sections)) if sections else float('nan'))

        for region in sections:
            section_length = len(region.pixels)
            all_section_lengths.append(section_length)
        
    if not all_section_lengths:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'lengths': np.array([], dtype=int),
            'section_count': 0,
            'min': float('nan'),
            'max': float('nan'),
            'median': float('nan'),
            'mean_repetition_section_ratio': float('nan'),
            'std_repetition_section_ratio': float('nan'),
            'repetition_section_ratios': np.array(all_repetition_section_ratios, dtype=float),
        }
    
    lengths_array = np.array(all_section_lengths, dtype=int)
    all_repetition_section_ratios = np.array(all_repetition_section_ratios, dtype=float)
    
    return {
        'mean': float(np.mean(lengths_array)),
        'std': float(np.std(lengths_array)),
        'lengths': lengths_array,
        'section_count': len(all_section_lengths),
        'min': int(np.min(lengths_array)),
        'max': int(np.max(lengths_array)),
        'median': float(np.median(lengths_array)),
        'mean_repetition_section_ratio': float(np.mean(all_repetition_section_ratios)),
        'std_repetition_section_ratio': float(np.std(all_repetition_section_ratios)),
        'repetition_section_ratios': all_repetition_section_ratios
}


def _sample_section_widths(
    size: int,
    mean_length: float,
    std_length: float,
    rng: np.random.Generator,
) -> list[int]:
    if not np.isfinite(mean_length) or mean_length <= 0:
        mean_width = 1.0
    else:
        mean_width = max(1.0, np.sqrt(mean_length))

    if not np.isfinite(std_length) or std_length <= 0:
        std_width = 0.0
    else:
        std_width = max(0.5, np.sqrt(std_length) * 0.5)

    n_sections = max(2, int(round(size / max(1.0, mean_width))))
    raw = rng.normal(mean_width, std_width, size=n_sections) if std_width > 0.0 else np.full(n_sections, mean_width)
    widths = np.clip(np.round(raw).astype(int), 1, size)

    current_sum = int(widths.sum())
    if current_sum == 0:
        widths = np.ones(n_sections, dtype=int)
        current_sum = n_sections

    diff = size - current_sum
    if diff > 0:
        for _ in range(diff):
            idx = int(rng.integers(0, n_sections))
            widths[idx] += 1
    elif diff < 0:
        for _ in range(-diff):
            candidates = np.where(widths > 1)[0]
            if candidates.size == 0:
                break
            idx = int(rng.choice(candidates))
            widths[idx] -= 1

    # Final adjustment if the sum is still wrong due to rounding.
    while widths.sum() < size:
        idx = int(rng.integers(0, n_sections))
        widths[idx] += 1
    while widths.sum() > size:
        candidates = np.where(widths > 1)[0]
        if candidates.size == 0:
            break
        idx = int(rng.choice(candidates))
        widths[idx] -= 1

    widths = widths.tolist()
    return widths


def _sample_repetition_pairs(
    widths: list[int],
    mean_ratio: float,
    std_ratio: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:

    n_sections = len(widths)

    if n_sections < 2:
        return []

    mean_ratio = float(mean_ratio) if np.isfinite(mean_ratio) else 0.0
    std_ratio = float(std_ratio) if np.isfinite(std_ratio) else 0.0

    ratio = rng.normal(mean_ratio, std_ratio) if std_ratio > 0 else mean_ratio
    ratio = max(0.0, ratio)

    max_pairs = n_sections // 2
    n_pairs = min(int(round(ratio * n_sections)), max_pairs)

    candidates = [
        (i, j)
        for i in range(n_sections)
        for j in range(i + 2, n_sections) # Filter to not be consecutive repetitions
        if max(widths[i], widths[j]) <= 4 * min(widths[i], widths[j]) # Similar shapes (repetitions should be similar in size)
    ]

    rng.shuffle(candidates)

    pairs = []
    used = set()

    for i, j in candidates:
        if i in used or j in used:
            continue
        pairs.append((i, j))
        used.add(i)
        used.add(j)
        if len(pairs) == n_pairs:
            break

    return pairs


def simulate_ssms_from_section_length_stats(
    length_stats: dict[str, Any],
    ssm_gt: np.ndarray,
    n_sim: int,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    if ssm_gt.ndim != 2 or ssm_gt.shape[0] != ssm_gt.shape[1]:
        raise ValueError("ssm_gt must be a square matrix")
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")

    rng = np.random.default_rng(random_state)
    size = ssm_gt.shape[0]
    mean_length = float(length_stats.get('mean', np.nan))
    std_length = float(length_stats.get('std', 0.0))
    mean_ratio = float(length_stats.get('mean_repetition_section_ratio', 0.0))
    std_ratio = float(length_stats.get('std_repetition_section_ratio', 0.0))

    simulated_ssms = np.zeros((n_sim, size, size), dtype=np.uint8)

    for sim_idx in range(n_sim):
        widths = _sample_section_widths(size, mean_length, std_length, rng)
        section_starts = np.concatenate(([0], np.cumsum(widths)))
        ssm = np.zeros((size, size), dtype=np.uint8)

        for section_idx in range(len(widths)):
            start = int(section_starts[section_idx])
            end = int(section_starts[section_idx + 1])
            ssm[start:end, start:end] = 1

        pairs = _sample_repetition_pairs(widths, mean_ratio, std_ratio, rng)
        for i, j in pairs:
            start_i = int(section_starts[i])
            end_i = int(section_starts[i + 1])
            start_j = int(section_starts[j])
            end_j = int(section_starts[j + 1])
            # block_size = max(1, min(end_i - start_i, end_j - start_j))
            # ssm[start_i:start_i + block_size, start_j:start_j + block_size] = 1
            # ssm[start_j:start_j + block_size, start_i:start_i + block_size] = 1
            ssm[start_i:end_i, start_j:end_j] = 1
            ssm[start_j:end_j, start_i:end_i] = 1

        simulated_ssms[sim_idx] = ssm

    return simulated_ssms


def plot_gt_and_simulated_ssms(
    gt_ssms: list[np.ndarray],
    length_stats: dict[str, Any],
    args: argparse.Namespace,
    file_names: list[str] | None = None,
) -> None:
    
    n_gt = len(gt_ssms)
    n_sim = args.n_sim
    ncols = n_gt
    figsize = (ncols * 3.0, 2 * 3.0)
    fig, axes = plt.subplots(n_sim + 1, n_gt, figsize=figsize, squeeze=False)

    for col in range(ncols):
        simulated_ssms = simulate_ssms_from_section_length_stats(
        length_stats,
        gt_ssms[col],
        args.n_sim,
        random_state=args.random_seed,
    )
        if col < n_gt:
            axes[0, col].imshow(gt_ssms[col], origin="lower", cmap="gray_r", aspect="equal")
            title_text = file_names[col] if file_names else f"GT {col + 1}"
            axes[0, col].set_title(title_text)
        else:
            axes[0, col].axis("off")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        for row, sim_ssm in enumerate(simulated_ssms):
            if col < n_sim:
                axes[row + 1, col].imshow(sim_ssm, origin="lower", cmap="gray_r", aspect="equal")
                axes[row + 1, col].set_title(f"Sim {col + 1}")
            else:
                axes[row + 1, col].axis("off")
            axes[row + 1, col].set_xticks([])
            axes[row + 1, col].set_yticks([])

    fig.suptitle("Ground-truth and simulated SSM examples")
    fig.tight_layout()
    fig.savefig(args.output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_feature_similarity(
    feature_name: str,
    bundle: dict[str, Any],
    ssm_gt: np.ndarray,
    similarity_type: str = "cosine",
    gamma: float | None = None,
    segment_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    beat_key = f"beat_{feature_name}"
    if beat_key not in bundle:
        raise KeyError(f"Feature {feature_name!r} is missing from the bundle.")

    feature_matrix = np.asarray(bundle[beat_key], dtype=float)
    if feature_matrix.ndim != 2:
        raise ValueError(f"Feature {feature_name!r} must be a two-dimensional array.")

    ssm = np.abs(build_ssm(feature_matrix, similarity_type=similarity_type, gamma=gamma))
    if ssm.shape != ssm_gt.shape:
        raise ValueError(
            f"SSM shape mismatch for feature {feature_name!r}: got {ssm.shape}, expected {ssm_gt.shape}."
        )

    homogeneity = compute_homogeneity_scores(ssm, ssm_gt, segment_ids=segment_ids)
    repetition = compute_repetition_scores(ssm, ssm_gt, segment_ids=segment_ids)
    novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=3, L=16, var=0.5, exclude=True, segment_ids=segment_ids)

    section_mask, repetition_mask, background_mask = _build_region_masks(ssm_gt, segment_ids=segment_ids)
    section_mean = float(np.mean(ssm[section_mask])) if np.any(section_mask) else float("nan")
    repetition_mean = float(np.mean(ssm[repetition_mask])) if np.any(repetition_mask) else float("nan")
    background_mean = float(np.mean(ssm[background_mask])) if np.any(background_mask) else float("nan")
    global_quality = (
        0.5 * (section_mean + repetition_mean) - background_mean
        if np.isfinite(section_mean) and np.isfinite(repetition_mean) and np.isfinite(background_mean)
        else float("nan")
    )

    return {
        "feature": feature_name,
        "similarity_type": similarity_type,
        "metrics": {
            "homogeneity": homogeneity,
            "repetition": repetition,
            "novelty": novelty,
        },
        "region_scores": {
            "section_mean": section_mean,
            "repetition_mean": repetition_mean,
            "background_mean": background_mean,
        },
        "global_quality": global_quality,
    }


def build_comparison_table(
    npz_paths: list[Path],
    similarity_types: tuple[str, ...] = SIMILARITY_TYPES,
    gamma: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for npz_path in npz_paths:
        bundle = load_feature_bundle(npz_path)
        ssm_gt = _build_gt_matrix(bundle)
        segment_ids = _build_segment_ids(bundle)
        song_name = npz_path.stem.replace("_features", "")
        for feature_name in list_feature_names(bundle):
            for similarity_type in similarity_types:
                result = evaluate_feature_similarity(
                    feature_name,
                    bundle,
                    ssm_gt,
                    similarity_type=similarity_type,
                    gamma=gamma,
                    segment_ids=segment_ids,
                )
                rows.append(
                    {
                        "song": song_name,
                        "bundle": npz_path.name,
                        "feature": result["feature"],
                        "similarity_type": result["similarity_type"],
                        "homogeneity": result["metrics"]["homogeneity"]["global"],
                        "repetition": result["metrics"]["repetition"]["global"],
                        "novelty": result["metrics"]["novelty"]["global"],
                        "global_quality": result["global_quality"],
                        **result["region_scores"],
                    }
                )
    return pd.DataFrame(rows)


def build_feature_ssm_validation_sets(
    npz_paths: list[Path],
    similarity_type: str,
    gamma: float | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray | None]]:
    feature_sets: dict[str, dict[str, Any]] = {}
    song_gt_ssms: dict[str, np.ndarray] = {}
    song_segment_ids: dict[str, np.ndarray | None] = {}

    for npz_path in npz_paths:
        bundle = load_feature_bundle(npz_path)
        try:
            ssm_gt = _build_gt_matrix(bundle)
        except KeyError:
            continue

        song_name = npz_path.stem.replace("_features", "")
        if song_name not in song_gt_ssms:
            song_gt_ssms[song_name] = ssm_gt
            song_segment_ids[song_name] = _build_segment_ids(bundle)

        for feature_name in list_feature_names(bundle):
            feature_matrix = np.asarray(bundle[f"beat_{feature_name}"], dtype=float)
            try:
                ssm = np.abs(build_ssm(feature_matrix, similarity_type=similarity_type, gamma=gamma))
            except Exception:
                continue

            if ssm.shape != ssm_gt.shape:
                continue

            feature_data = feature_sets.setdefault(
                feature_name,
                {
                    "ssms": [],
                    "song_names": [],
                },
            )
            feature_data["ssms"].append(ssm)
            feature_data["song_names"].append(song_name)

    return feature_sets, song_gt_ssms, song_segment_ids


def run_feature_validation(
    args: argparse.Namespace,
    npz_paths: list[Path],
    length_stats: dict[str, Any],
) -> None:
    if args.similarity_type is None:
        raise ValueError("Similarity type is required for feature-based SSM validation.")

    output_dir = args.output_file.parent / f"validation_{args.similarity_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_sets, song_gt_ssms, song_segment_ids = build_feature_ssm_validation_sets(
        npz_paths,
        args.similarity_type,
        gamma=getattr(args, "gamma", None),
    )

    if not feature_sets:
        raise ValueError(
            f"No valid feature-based SSMs could be constructed using similarity type '{args.similarity_type}'."
        )

    for feature_name, feature_data in sorted(feature_sets.items()):
        if not feature_data["ssms"]:
            continue

        gt_ssms = [song_gt_ssms[song_name] for song_name in feature_data["song_names"]]
        gt_segment_ids = [song_segment_ids[song_name] for song_name in feature_data["song_names"]]
        results = scores_from_ssms(
            ssms=feature_data["ssms"],
            ssms_gt=gt_ssms,
            length_stats=length_stats,
            args=args,
            segment_ids_list=gt_segment_ids,
        )

        file_base = output_dir / f"ssm_simulation_{feature_name}"
        plot_score_histograms(results, file_base)
        save_scores(results, file_base)
        print(f"Saved validation outputs for feature '{feature_name}' to {output_dir}")


def save_global_comparison_plots(results_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    features = sorted(results_df["feature"].dropna().unique().tolist())
    similarity_types = sorted(results_df["similarity_type"].dropna().unique().tolist())

    for metric_name in METRIC_NAMES:
        summary = results_df.pivot_table(index="feature", columns="similarity_type", values=metric_name, aggfunc="mean")
        stds = results_df.pivot_table(index="feature", columns="similarity_type", values=metric_name, aggfunc="std").fillna(0.0)
        summary = summary.reindex(index=features, columns=similarity_types)
        stds = stds.reindex(index=features, columns=similarity_types)

        fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(features) + 2.0), 4.5))
        x = np.arange(len(features))
        bar_width = 0.22
        offsets = np.linspace(-bar_width * (len(similarity_types) - 1) / 2.0, bar_width * (len(similarity_types) - 1) / 2.0, len(similarity_types))
        for idx, similarity_type in enumerate(similarity_types):
            values = summary[similarity_type].to_numpy(dtype=float)
            errors = stds[similarity_type].to_numpy(dtype=float)
            ax.bar(x + offsets[idx], values, width=bar_width, yerr=errors, label=similarity_type, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"{metric_name.replace('_', ' ').title()} by feature and similarity")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"{metric_name}_global_comparison.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

        fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(similarity_types) + 2.0), 4.5))
        values_by_similarity = []
        for similarity_type in similarity_types:
            values = results_df.loc[results_df["similarity_type"] == similarity_type, metric_name].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                finite_values = np.array([0.0], dtype=float)
            values_by_similarity.append(finite_values)
        parts = ax.violinplot(values_by_similarity, showmeans=True, showmedians=True)
        for body, similarity_type in zip(parts["bodies"], similarity_types):
            body.set_facecolor(plt.cm.viridis(0.2 + 0.7 * similarity_types.index(similarity_type) / max(1, len(similarity_types) - 1)))
            body.set_alpha(0.85)
        ax.set_xticks(range(1, len(similarity_types) + 1))
        ax.set_xticklabels(similarity_types)
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"{metric_name.replace('_', ' ').title()} distribution by similarity")
        fig.tight_layout()
        violin_path = output_dir / f"{metric_name}_violin.png"
        fig.savefig(violin_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(violin_path)

    for region_name in REGION_SCORE_NAMES:
        pivot = results_df.pivot_table(index="feature", columns="similarity_type", values=region_name, aggfunc="mean")
        pivot = pivot.reindex(index=features, columns=similarity_types)
        fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(similarity_types) + 2.0), max(4.0, 0.6 * len(features) + 2.0)))
        image = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(len(similarity_types)))
        ax.set_xticklabels(similarity_types)
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features)
        ax.set_title(f"{region_name.replace('_', ' ').title()} by feature and similarity")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        heatmap_path = output_dir / f"{region_name}_heatmap.png"
        fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(heatmap_path)

    return saved_paths


def _finite_limits(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [1, 99])
    if np.isclose(lo, hi):
        lo, hi = float(values.min()), float(values.max())
    if np.isclose(lo, hi):
        hi = lo + 1.0
    return float(lo), float(hi)


def _plot_region_boxes(ax, regions, color, label, ls="solid"):
    import matplotlib.patches as patches

    used_label = False
    for region in regions:
        rows = region.pixels[:, 0]
        cols = region.pixels[:, 1]
        row0, row1 = int(rows.min()), int(rows.max())
        col0, col1 = int(cols.min()), int(cols.max())
        ax.add_patch(
            patches.Rectangle(
                (col0 - 0.5, row0 - 0.5),
                col1 - col0 + 1,
                row1 - row0 + 1,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
                ls=ls,
                label=label if not used_label else None,
            )
        )
        used_label = True


def save_all_ssm_images(npz_paths: list[Path], output_dir: Path, similarity_types: tuple[str, ...], gamma: float | None = None) -> list[Path]:
    saved: list[Path] = []
    for npz_path in npz_paths:
        bundle = load_feature_bundle(npz_path)
        ssm_gt = _build_gt_matrix(bundle)
        segment_ids = _build_segment_ids(bundle)
        sections, repetitions = extract_ssm_regions(ssm_gt, segment_ids=segment_ids)
        song_name = npz_path.stem.replace("_features", "")
        target_song_dir = output_dir / "SSMs" / song_name
        target_song_dir.mkdir(parents=True, exist_ok=True)

        for feature_name in list_feature_names(bundle):
            beat_key = f"beat_{feature_name}"
            features = np.asarray(bundle[beat_key], dtype=float)
            for similarity_type in similarity_types:
                try:
                    ssm = np.abs(build_ssm(features, similarity_type=similarity_type, gamma=gamma))
                except Exception:
                    continue

                novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=3, L=16, var=0.5, exclude=True, segment_ids=segment_ids)
                novelty_response = novelty.get("novelty_response")

                # Plot SSM (with boxes) and novelty response side-by-side
                fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
                vmin, vmax = _finite_limits(ssm)
                axes[0].imshow(ssm, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
                _plot_region_boxes(axes[0], sections, "cyan", "section")
                _plot_region_boxes(axes[0], repetitions, "lime", "repetition")
                axes[0].set_title(f"{song_name} - {feature_name} - {similarity_type}")
                axes[0].set_xlabel("Frame / beat index")
                axes[0].set_ylabel("Frame / beat index")
                axes[0].legend(loc="upper right")

                if novelty_response is not None:
                    axes[1].plot(novelty_response, color="tab:purple", linewidth=1.5)
                else:
                    axes[1].text(0.5, 0.5, "No novelty response", ha="center", va="center")
                axes[1].set_title("Checkerboard novelty response")
                axes[1].set_xlabel("Frame / beat index")
                axes[1].set_ylabel("Novelty")
                axes[1].grid(alpha=0.25)

                out_path = target_song_dir / f"{feature_name}_{similarity_type}.png"
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                saved.append(out_path)
    return saved


def save_combined_ssm_svgs(npz_paths: list[Path], output_dir: Path, similarity_types: tuple[str, ...], gamma: float | None = None) -> list[Path]:
    """For each song bundle, save a single SVG with all feature SSMs.

    For each feature, one row shows the SSMs with one column per similarity
    type, and the row below shows each corresponding novelty response.
    File is saved as <song>_SSMs.svg inside output_dir/SSMs/.
    """
    if not similarity_types:
        raise ValueError("At least one similarity type is required.")

    saved: list[Path] = []
    for npz_path in npz_paths:
        bundle = load_feature_bundle(npz_path)
        ssm_gt = _build_gt_matrix(bundle)
        segment_ids = _build_segment_ids(bundle)
        sections, repetitions = extract_ssm_regions(ssm_gt, segment_ids=segment_ids)
        song_name = npz_path.stem.replace("_features", "")
        target_song_dir = output_dir / "SSMs"
        target_song_dir.mkdir(parents=True, exist_ok=True)

        feature_names = list_feature_names(bundle)
        if not feature_names:
            continue

        n_cols = len(similarity_types)
        n_rows = 2 * len(feature_names)
        ssm_axis_inches = 4.0
        novelty_axis_inches = 1.5
        height_ratios = [ssm_axis_inches if row_index % 2 == 0 else novelty_axis_inches for row_index in range(n_rows)]
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(ssm_axis_inches * max(1, n_cols), sum(height_ratios) + 1.5),
            constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)

        for feature_index, feature_name in enumerate(feature_names):
            beat_key = f"beat_{feature_name}"
            features = np.asarray(bundle[beat_key], dtype=float)
            ssm_row = 2 * feature_index
            novelty_row = ssm_row + 1

            for col_index, similarity_type in enumerate(similarity_types):
                ssm_ax = axes[ssm_row, col_index]
                novelty_ax = axes[novelty_row, col_index]

                try:
                    ssm = np.abs(build_ssm(features, similarity_type=similarity_type, gamma=gamma))
                except Exception as exc:
                    ssm_ax.text(0.5, 0.5, f"Failed: {similarity_type}\n{exc}", ha="center", va="center")
                    novelty_ax.text(0.5, 0.5, "No novelty", ha="center", va="center")
                    for ax in (ssm_ax, novelty_ax):
                        ax.set_xticks([])
                        ax.set_yticks([])
                    continue

                vmin, vmax = _finite_limits(ssm)
                ssm_ax.imshow(ssm, origin="lower", aspect="equal", cmap="magma", vmin=vmin, vmax=vmax)
                ssm_ax.set_box_aspect(1)
                _plot_region_boxes(ssm_ax, sections, "cyan", "section")
                _plot_region_boxes(ssm_ax, repetitions, "lime", "repetition")
                ssm_ax.set_xticks([])
                ssm_ax.set_yticks([])

                novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=3, L=16, var=0.5, exclude=True, segment_ids=segment_ids)
                novelty_response = novelty.get("novelty_response")
                if novelty_response is not None:
                    novelty_ax.plot(novelty_response, color="tab:purple", linewidth=1.2)
                else:
                    novelty_ax.text(0.5, 0.5, "No novelty response", ha="center", va="center")
                novelty_ax.set_xlabel("Frame / beat index")
                novelty_ax.grid(alpha=0.2)

                if feature_index == 0:
                    ssm_ax.set_title(similarity_type)
                if col_index == 0:
                    ssm_ax.set_ylabel(f"{feature_name}\nSSM")
                    novelty_ax.set_ylabel(f"{feature_name}\nNovelty")

        out_path = target_song_dir / f"{song_name}_SSMs.svg"
        fig.suptitle(f"{song_name} - SSMs by feature and similarity")
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


def run_comparison(
    features_dir: Path | None = None,
    output_dir: Path | None = None,
    similarity_types: tuple[str, ...] = SIMILARITY_TYPES,
    gamma: float | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    project_dir = resolve_project_dir(features_dir)
    search_root = features_dir if features_dir is not None else project_dir
    npz_paths = discover_feature_bundles(search_root)
    if not npz_paths:
        raise FileNotFoundError(
            f"No precomputed feature bundles were found under {search_root}. "
            "Point --features-dir to a directory containing *_features.npz files."
        )

    results_df = build_comparison_table(npz_paths, similarity_types=similarity_types, gamma=gamma)
    if results_df.empty:
        raise ValueError("No comparison rows could be generated from the available bundles.")
    target_dir = output_dir if output_dir is not None else project_dir / "experiments" / "msa_principles_measure" / "similarity_comparison_outputs"
    target_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(target_dir / "similarity_comparison_results.csv", index=False)
    # Save combined per-song SSM SVGs (two rows per feature, one column per similarity type)
    saved_ssm_paths = save_combined_ssm_svgs(npz_paths, target_dir, similarity_types=similarity_types, gamma=gamma)
    print(f"Wrote {len(saved_ssm_paths)} combined SSM SVGs to {target_dir / 'SSMs'}")
    saved_paths = save_global_comparison_plots(results_df, target_dir)
    print(f"Wrote {len(saved_paths)} figures to {target_dir}")
    print(results_df.head().to_string(index=False))
    return results_df, saved_paths

def scores_from_ssms(ssms: np.ndarray,
                     ssms_gt: np.ndarray,
                     length_stats: dict[str, float],
                     args: argparse.Namespace,
                     segment_ids_list: list[np.ndarray | None] | None = None,
    ) -> dict[str, Any]:

    results = {
        "gt": {
            "homogeneity": [],
            "repetition": [],
            "novelty": [],
        },
        "sim": {
            "homogeneity": [],
            "repetition": [],
            "novelty": [],
        },
        "p_values": {
            "homogeneity": [],
            "repetition": [],
            "novelty": [],
            },
        }

    for index, (ssm, ssm_gt) in enumerate(zip(ssms, ssms_gt)):
        if ssm.shape != ssm_gt.shape:
            raise ValueError(f"SSM shape mismatch: got {ssm.shape}, expected {ssm_gt.shape}.")

        segment_ids = segment_ids_list[index] if segment_ids_list is not None else None

        homogeneity_gt_mask = compute_homogeneity_scores(ssm, ssm_gt, segment_ids=segment_ids)['global']
        repetition_gt_mask = compute_repetition_scores(ssm, ssm_gt, segment_ids=segment_ids)['global']
        novelty_gt_mask = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=3, L=16, var=0.5, exclude=True, segment_ids=segment_ids)['global']

        # Ground truth mask
        results["gt"]["homogeneity"].append(homogeneity_gt_mask)
        results["gt"]["repetition"].append(repetition_gt_mask)
        results["gt"]["novelty"].append(novelty_gt_mask)

        # Simulated masks
        simulated = simulate_ssms_from_section_length_stats(
            length_stats,
            ssm_gt,
            args.n_sim,
            random_state=args.random_seed
            )
        
        for sim in simulated:

            results["sim"]["homogeneity"].append(
                compute_homogeneity_scores(ssm, sim)['global']
            )

            results["sim"]["repetition"].append(
                compute_repetition_scores(ssm, sim)['global']
            )

            results["sim"]["novelty"].append(
                compute_novelty_scores(
                    ssm, sim, boundary_band_width=3, L=16, var=0.5, exclude=True)['global']
            )

        results["p_values"]["homogeneity"].append(float(np.mean(np.asarray(results["sim"]["homogeneity"]) >= homogeneity_gt_mask)))
        results["p_values"]["repetition"].append(float(np.mean(np.asarray(results["sim"]["repetition"]) >= repetition_gt_mask)))
        results["p_values"]["novelty"].append(float(np.mean(np.asarray(results["sim"]["novelty"]) >= novelty_gt_mask)))
        
    return results

def save_scores(
    results: dict,
    output_file: Path,
) -> None:
    
    # Save ground truth scores
    gt_scores = {
        "homogeneity": results["gt"]["homogeneity"],
        "repetition": results["gt"]["repetition"],
        "novelty": results["gt"]["novelty"],
    }
    gt_save_path = (
        output_file.parent
        / f"{output_file.stem}_gt_scores.csv"
    )
    gt_df = pd.DataFrame(gt_scores)
    gt_df.to_csv(gt_save_path, index=False)

    # Save simulated scores
    sim_scores = {
        "homogeneity": results["sim"]["homogeneity"],
        "repetition": results["sim"]["repetition"],
        "novelty": results["sim"]["novelty"],
    }
    sim_save_path = (
        output_file.parent
        / f"{output_file.stem}_sim_scores.csv"
    )
    sim_df = pd.DataFrame(sim_scores)
    sim_df.to_csv(sim_save_path, index=False)

    print(f"Saved GT and simulated scores to {gt_save_path} and {sim_save_path}.")

def plot_score_histograms(
    results: dict,
    output_file: Path,
) -> None:
    """
    Plot score distributions of simulated SSMs together with the scores
    obtained from the ground-truth SSMs.

    Parameters
    ----------
    results : dict
        Dictionary of the form

        {
            "gt": {
                "homogeneity": [...],
                "repetition": [...],
                "novelty": [...],
            },
            "sim": {
                "homogeneity": [...],
                "repetition": [...],
                "novelty": [...],
            },
        }

    output_file : Path
        Base output file. The histogram figure will be saved in the same
        directory as

            <stem>_score_histograms.png
    """

    metrics = [
        ("homogeneity", "Homogeneity"),
        ("repetition", "Repetition"),
        ("novelty", "Novelty"),
    ]

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(8, 10),
        constrained_layout=True,
    )

    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, metrics):

        sim = np.asarray(results["sim"][key], dtype=float)
        gt = np.asarray(results["gt"][key], dtype=float)
        p_values = np.asarray(results["p_values"][key], dtype=float)

        # Histogram of simulations
        ax.hist(
            sim,
            bins=30,
            density=True,
            alpha=0.7,
            label="Simulations",
            color="tab:red"
        )

        # Histogram of simulations
        ax.hist(
            gt,
            bins=30,
            density=True,
            alpha=0.7,
            label="Ground Truth",
            color="tab:green"
        )

        # # Ground-truth range
        # gt_min = gt.min()
        # gt_max = gt.max()

        # ax.axvspan(
        #     gt_min,
        #     gt_max,
        #     alpha=0.15,
        #     color="green",
        #     label="GT range",
        # )

        # # One line per GT song
        # for i, value in enumerate(gt):
        #     ax.axvline(
        #         value,
        #         color="black",
        #         linewidth=2,
        #         alpha=0.6,
        #         label="GT scores" if i == 0 else None,
        #     )

        ax.set_title(f"{title}")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3)
    fig.tight_layout() 

    save_path = (
        output_file.parent
        / f"{output_file.stem}_score_histograms.png"
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved score histograms to {save_path}")

    # ------------------------------------------------------------------
    # Save p-values
    # ------------------------------------------------------------------

    csv_path = (
        output_file.parent
        / f"{output_file.stem}_pvalues.csv"
    )

    df = pd.DataFrame(results["p_values"])
    df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # Print summary statistics
    # ------------------------------------------------------------------

    print("\nP-value summary")
    print("-" * 60)

    for metric in ["homogeneity", "repetition", "novelty"]:

        p = np.asarray(results["p_values"][metric], dtype=float)

        print(
            f"{metric:14s}"
            f" mean={p.mean():.4f}"
            f"  median={np.median(p):.4f}"
            f"  p<0.05={100*np.mean(p < 0.05):5.1f}%"
        )

    print("-" * 60)
    print(f"Saved p-values to {csv_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=None, help="Directory containing *_features.npz bundles")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for SSM simulation")
    parser.add_argument("--num-gt-examples", type=int, default=DEFAULT_NUM_GT_EXAMPLES, help="Number of ground-truth SSM examples to show")
    parser.add_argument("--n-sim", type=int, default=DEFAULT_N_SIM, help="Number of simulated SSMs to generate")
    parser.add_argument("--view-sims", action="store_true", help="Show example ground-truth and simulated SSMs.")
    parser.add_argument("--similarity-type", choices=SIMILARITY_TYPES, default=None, help="Similarity type to use when constructing feature SSMs for validation.")
    parser.add_argument("--output-file", type=Path, default=Path(__file__).resolve().parent / DEFAULT_OUTPUT_FILENAME, help="Output image file path")
    args = parser.parse_args()

    project_dir = resolve_project_dir(args.features_dir)
    search_root = args.features_dir if args.features_dir is not None else project_dir
    npz_paths = discover_feature_bundles(search_root)
    if not npz_paths:
        raise FileNotFoundError(
            f"No precomputed feature bundles were found under {search_root}. "
            "Point --features-dir to a directory containing *_features.npz files."
        )

    length_stats = compute_section_length_stats(npz_paths)

    print(f'Stats from labeled section data={length_stats}')

    gt_ssms: list[np.ndarray] = []
    gt_segment_ids: list[np.ndarray | None] = []
    file_names: list[str] = []
    random.seed(args.random_seed)
    random.shuffle(npz_paths)
    for npz_path in npz_paths:
        if len(gt_ssms) >= args.num_gt_examples:
            break
        try:
            bundle = load_feature_bundle(npz_path)
            gt_ssm = _build_gt_matrix(bundle)
            gt_ssms.append(gt_ssm)
            gt_segment_ids.append(_build_segment_ids(bundle))
            file_names.append(npz_path.stem.replace("_features", ""))
        except KeyError:
            continue

    if args.similarity_type is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        run_feature_validation(args, npz_paths, length_stats)
        return

    if not gt_ssms:
        raise ValueError("No valid ground-truth SSMs could be loaded from the discovered bundles.")

    results = scores_from_ssms(
        ssms=gt_ssms,
        ssms_gt=gt_ssms,
        length_stats=length_stats,
        args=args,
        segment_ids_list=gt_segment_ids,
    )
    plot_score_histograms(results, args.output_file)


    if args.view_sims:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        plot_gt_and_simulated_ssms(gt_ssms[:MAX_N_SIM_SHOW], length_stats, args, file_names=file_names[:MAX_N_SIM_SHOW])
        print(f"Saved combined SSM example figure to {args.output_file}")

if __name__ == "__main__":
    main()
