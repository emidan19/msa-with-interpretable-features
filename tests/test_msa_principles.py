import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# SCRIPT_PATH = REPO_ROOT / "experiments" / "msa_principles_measure" / "by_similarity_function.py"
# SPEC = importlib.util.spec_from_file_location("by_similarity_function", SCRIPT_PATH)
# by_similarity_function = importlib.util.module_from_spec(SPEC)
# assert SPEC.loader is not None
# SPEC.loader.exec_module(by_similarity_function)

from msa_principles import (
    compute_homogeneity_scores,
    compute_novelty_scores,
    compute_repetition_scores,
    extract_ssm_regions,
    ssm_quality_metrics,
)


def _build_toy_ssm_gt() -> np.ndarray:
    section_sizes = [5, 5, 5, 5]
    section_bounds = []
    start = 0
    for section_size in section_sizes:
        stop = start + section_size
        section_bounds.append((start, stop))
        start = stop

    ssm_gt = np.zeros((sum(section_sizes), sum(section_sizes)), dtype=int)
    for row_start, row_stop in section_bounds:
        ssm_gt[row_start:row_stop, row_start:row_stop] = 1

    repetition_blocks = [
        (section_bounds[0][0], section_bounds[0][1], section_bounds[-1][0], section_bounds[-1][1]),
        (section_bounds[-1][0], section_bounds[-1][1], section_bounds[0][0], section_bounds[0][1]),
    ]
    for row_start, row_stop, col_start, col_stop in repetition_blocks:
        ssm_gt[row_start:row_stop, col_start:col_stop] = 1

    return ssm_gt


def test_extract_ssm_regions_classifies_sections():
    ssm_gt = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=int,
    )

    sections, repetitions = extract_ssm_regions(ssm_gt)

    assert len(sections) == 2
    assert len(repetitions) == 0


def test_homogeneity_and_novelty_scores_are_finite():
    ssm = np.array(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ],
        dtype=float,
    )
    ssm_gt = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=int,
    )

    homogeneity = compute_homogeneity_scores(ssm, ssm_gt)
    novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=1, L=3, exclude=False)

    assert len(homogeneity["per_section"]) == 2
    assert np.isfinite(homogeneity["global"])
    assert len(novelty["per_boundary"]) == 1
    assert len(novelty["per_transition"]) == 1
    assert np.allclose(novelty["per_boundary"], novelty["per_transition"])
    assert np.isfinite(novelty["global"])


def test_novelty_scores_returns_transition_scores():
    ssm = np.eye(5, dtype=float)
    ssm_gt = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=int,
    )

    novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=1, L=3, exclude=False)

    assert "per_transition" in novelty
    assert len(novelty["per_transition"]) == 2
    assert np.allclose(novelty["per_transition"], novelty["per_boundary"])
    assert np.isfinite(novelty["global"])


def test_ssm_quality_metrics_returns_expected_keys():
    ssm = np.eye(3, dtype=float)
    ssm_gt = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)

    metrics = ssm_quality_metrics(ssm, ssm_gt)

    assert set(metrics) >= {"homogeneity", "repetition", "novelty"}
    assert metrics["section_count"] == 3


def test_evaluate_feature_similarity_returns_expected_keys():
    features = np.array([[1.0, 0.2], [0.9, 0.3], [0.1, 0.8]], dtype=float)
    bundle = {"beat_stm": features, "ssm_sections": np.array([[1, 0], [0, 1]], dtype=int)}

    result = by_similarity_function.evaluate_feature_similarity(
        "stm",
        bundle,
        np.array([[1, 0], [0, 1]], dtype=int),
        similarity_type="cosine",
    )

    assert result["feature"] == "stm"
    assert result["similarity_type"] == "cosine"
    assert set(result["metrics"]) >= {"homogeneity", "repetition", "novelty"}
    assert set(result["region_scores"]) >= {"section_mean", "repetition_mean", "background_mean"}
    assert np.isfinite(result["metrics"]["homogeneity"]["global"])


def test_contrast_scores_for_sections_repetitions_and_novelty():
    rng = np.random.default_rng(0)
    section_sizes = [5, 5, 5, 5]
    section_count = len(section_sizes)
    size = int(np.sum(section_sizes))

    section_bounds = []
    start = 0
    for section_size in section_sizes:
        stop = start + section_size
        section_bounds.append((start, stop))
        start = stop

    ssm_gt = np.zeros((size, size), dtype=int)
    for row_start, row_stop in section_bounds:
        ssm_gt[row_start:row_stop, row_start:row_stop] = 1

    repetition_blocks = [
        (section_bounds[0][0], section_bounds[0][1], section_bounds[-1][0], section_bounds[-1][1]),
        (section_bounds[-1][0], section_bounds[-1][1], section_bounds[0][0], section_bounds[0][1]),
    ]
    for row_start, row_stop, col_start, col_stop in repetition_blocks:
        ssm_gt[row_start:row_stop, col_start:col_stop] = 1

    sections, repetitions = extract_ssm_regions(ssm_gt)
    assert size == 20
    assert len(sections) == section_count
    assert len(repetitions) == 2

    ssm = rng.normal(loc=0.0, scale=0.5, size=(size, size)) # SSM with little noise
    for (row_start, row_stop) in section_bounds:
        ssm[row_start:row_stop, row_start:row_stop] = 0.96 + 0.02 * rng.normal(size=(row_stop - row_start, row_stop - row_start))

    for row_start, row_stop, col_start, col_stop in repetition_blocks:
        ssm[row_start:row_stop, col_start:col_stop] = 0.97 + 0.02 * rng.normal(size=(row_stop - row_start, col_stop - col_start))

    ssm = np.clip(ssm, 0.0, 1.0)

    bad_ssm = rng.normal(loc=0.0, scale=1, size=(size, size))
    bad_ssm = np.clip(bad_ssm, 0.0, 1.0)

    # Compute scores and novelty responses
    homogeneity_gt = compute_homogeneity_scores(ssm_gt, ssm_gt)
    homogeneity_normal = compute_homogeneity_scores(ssm, ssm_gt)
    homogeneity_bad = compute_homogeneity_scores(bad_ssm, ssm_gt)

    repetition_gt = compute_repetition_scores(ssm_gt, ssm_gt)
    repetition_normal = compute_repetition_scores(ssm, ssm_gt)
    repetition_bad = compute_repetition_scores(bad_ssm, ssm_gt)

    novelty_gt = compute_novelty_scores(ssm_gt, ssm_gt, boundary_band_width=1, L=3, exclude=False)
    novelty_normal = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=1, L=3, exclude=False)
    novelty_bad = compute_novelty_scores(bad_ssm, ssm_gt, boundary_band_width=1, L=3, exclude=False)

    # Combine novelty responses
    novelty_responses = {
        "ground_truth": novelty_gt["novelty_response"],
        "normal": novelty_normal["novelty_response"],
        "bad": novelty_bad["novelty_response"],
    }

    # Combine scores
    scores = {
        "homogeneity": [homogeneity_gt["global"], homogeneity_normal["global"], homogeneity_bad["global"]],
        "repetition": [repetition_gt["global"], repetition_normal["global"], repetition_bad["global"]],
        "novelty": [novelty_gt["global"], novelty_normal["global"], novelty_bad["global"]],
    }

    # Call the updated plot function
    _plot_ssm_and_novelty(ssm, ssm_gt, bad_ssm, novelty_responses, scores, title="Contrast Scores")

    # assert len(homogeneity["per_section"]) == section_count
    # assert len(repetition["per_repetition"]) == 2
    # assert len(novelty["per_boundary"]) == section_count - 1
    # assert np.isfinite(homogeneity["global"])
    # assert np.isfinite(repetition["global"])
    # assert np.isfinite(novelty["global"])


def _region_to_mask(shape: tuple[int, int], pixels: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(pixels.T)] = True
    return mask


def _local_background_mask(ssm_gt: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    row_start = int(np.min(pixels[:, 0]))
    row_end = int(np.max(pixels[:, 0]))
    row_selector = np.arange(ssm_gt.shape[0])[:, None]
    return (~ssm_gt.astype(bool)) & (row_selector >= row_start) & (row_selector <= row_end)


def _region_row_span(pixels: np.ndarray) -> tuple[int, int]:
    return int(np.min(pixels[:, 0])), int(np.max(pixels[:, 0]))


def _temporal_indices(pixels: np.ndarray) -> np.ndarray:
    row_start, row_end = _region_row_span(pixels)
    return np.arange(row_start, row_end + 1, dtype=int)


def _plot_mask_overlay(ax, base: np.ndarray, masks: list[tuple[np.ndarray, str, str]], title: str) -> None:
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as patches

    ax.imshow(base, cmap="gray_r", origin="lower", aspect="equal", interpolation="nearest")
    for mask, color, label in masks:
        overlay = np.ma.masked_where(~mask, mask)
        ax.imshow(overlay, cmap=ListedColormap([color]), origin="lower", aspect="equal", interpolation="nearest", alpha=0.45)
        rows, cols = np.nonzero(mask)
        if rows.size:
            ax.add_patch(
                patches.Rectangle(
                    (cols.min() - 0.5, rows.min() - 0.5),
                    cols.max() - cols.min() + 1,
                    rows.max() - rows.min() + 1,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.8,
                    label=label,
                )
            )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=7)


def save_ground_truth_score_region_diagnostics(
    ssm_gt: np.ndarray,
    output_path: Path,
    boundary_band_width: int = 1,
) -> Path:
    """Save a PNG showing the exact GT regions used by each score."""
    if plt is None:
        raise RuntimeError("matplotlib is not available, cannot save score region diagnostics.")

    sections, repetitions = extract_ssm_regions(ssm_gt)
    ordered_sections = sorted(sections, key=lambda region: _region_row_span(region.pixels)[0])
    panel_count = len(sections) + len(repetitions) + max(0, len(ordered_sections) - 1)
    n_cols = 3
    n_rows = int(np.ceil(panel_count / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), squeeze=False)
    axes_flat = axes.ravel()
    panel_index = 0

    for section_index, section in enumerate(ordered_sections, start=1):
        region_mask = _region_to_mask(ssm_gt.shape, section.pixels)
        background_mask = _local_background_mask(ssm_gt, section.pixels)
        _plot_mask_overlay(
            axes_flat[panel_index],
            ssm_gt,
            [(background_mask, "tab:orange", "local contrast region"), (region_mask, "tab:blue", "section")],
            f"Homogeneity section {section_index}",
        )
        panel_index += 1

    for repetition_index, repetition in enumerate(repetitions, start=1):
        region_mask = _region_to_mask(ssm_gt.shape, repetition.pixels)
        background_mask = _local_background_mask(ssm_gt, repetition.pixels)
        _plot_mask_overlay(
            axes_flat[panel_index],
            ssm_gt,
            [(background_mask, "tab:orange", "local contrast region"), (region_mask, "tab:green", "repetition")],
            f"Repetition region {repetition_index}",
        )
        panel_index += 1

    band_width = max(0, int(boundary_band_width))
    for transition_index, (left_section, right_section) in enumerate(zip(ordered_sections, ordered_sections[1:]), start=1):
        ax = axes_flat[panel_index]
        left_start, left_end = _region_row_span(left_section.pixels)
        right_start, right_end = _region_row_span(right_section.pixels)
        boundary_center = int(np.round(0.5 * (left_end + right_start)))
        boundary_start = max(0, boundary_center - band_width)
        boundary_end = min(ssm_gt.shape[0] - 1, boundary_center + band_width)
        boundary_indices = np.arange(boundary_start, boundary_end + 1, dtype=int)
        left_indices = np.setdiff1d(_temporal_indices(left_section.pixels), boundary_indices, assume_unique=True)
        right_indices = np.setdiff1d(_temporal_indices(right_section.pixels), boundary_indices, assume_unique=True)

        ax.imshow(ssm_gt, cmap="gray_r", origin="lower", aspect="equal", interpolation="nearest")
        if left_indices.size:
            ax.axvspan(left_indices.min() - 0.5, left_indices.max() + 0.5, color="tab:blue", alpha=0.25, label="left section samples")
        if right_indices.size:
            ax.axvspan(right_indices.min() - 0.5, right_indices.max() + 0.5, color="tab:green", alpha=0.25, label="right section samples")
        ax.axvspan(boundary_start - 0.5, boundary_end + 0.5, color="yellow", alpha=0.35, label="boundary margin")
        ax.axvline(boundary_center, color="yellow", linewidth=2.5, label="transition center")
        ax.set_title(f"Novelty transition {transition_index}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=7)
        panel_index += 1

    for ax in axes_flat[panel_index:]:
        ax.axis("off")

    fig.suptitle("Ground-truth regions used by MSA principle scores")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_ssm_and_novelty(ssm: np.ndarray, ssm_gt: np.ndarray, bad_ssm: np.ndarray, novelty_responses: dict, scores: dict, title: str) -> None:
    """
    Save a single image showing:
    - The ground truth, normal, and bad SSMs.
    - Their respective novelty responses.
    - Bar plots comparing homogeneity, repetition, and novelty scores.
    """
    if plt is None:
        print("matplotlib is not available, cannot plot SSMs and novelty response.")
        return

    output_dir = REPO_ROOT / "tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a figure with two rows
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # Plot SSMs
    axes[0, 0].imshow(ssm_gt, cmap="gray", origin="lower", aspect="auto")
    axes[0, 0].set_title(f"{title} - Ground Truth SSM")

    axes[0, 1].imshow(ssm, cmap="viridis", origin="lower", aspect="auto")
    axes[0, 1].set_title(f"{title} - Normal SSM")

    axes[0, 2].imshow(bad_ssm, cmap="plasma", origin="lower", aspect="auto")
    axes[0, 2].set_title(f"{title} - Bad SSM")

    # Plot novelty responses
    axes[0, 3].plot(novelty_responses["ground_truth"], marker="o", label="Ground Truth")
    axes[0, 3].plot(novelty_responses["normal"], marker="o", label="Normal")
    axes[0, 3].plot(novelty_responses["bad"], marker="o", label="Bad")
    axes[0, 3].set_title(f"{title} - Novelty Responses")
    axes[0, 3].set_xlabel("Index")
    axes[0, 3].set_ylabel("Novelty")
    axes[0, 3].legend()

    # Add bar plots for scores
    bar_width = 0.25
    x = np.arange(3)
    axes[1, 0].bar(x - bar_width, scores["homogeneity"], width=bar_width, label="Homogeneity")
    axes[1, 1].bar(x, scores["repetition"], width=bar_width, label="Repetition")
    axes[1, 2].bar(x + bar_width, scores["novelty"], width=bar_width, label="Novelty")

    axes[1, 0].set_title("Homogeneity Scores")
    axes[1, 1].set_title("Repetition Scores")
    axes[1, 2].set_title("Novelty Scores")

    for i in range(3):
        axes[1, i].set_xticks(x)
        axes[1, i].set_xticklabels(["GT", "Normal", "Bad"])
        axes[1, i].legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_dir / f"{title.replace(' ', '_')}.png")
    plt.close(fig)


if __name__ == "__main__":
    test_contrast_scores_for_sections_repetitions_and_novelty()
    diagnostic_path = save_ground_truth_score_region_diagnostics(
        _build_toy_ssm_gt(),
        REPO_ROOT / "tests" / "ground_truth_score_regions.png",
        boundary_band_width=1,
    )
    print(f"Saved score-region diagnostic plot to {diagnostic_path}")
