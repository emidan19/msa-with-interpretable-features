import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msa_principles import (
    compute_homogeneity_scores,
    compute_novelty_scores,
    compute_repetition_scores,
    extract_ssm_regions,
    ssm_quality_metrics,
)


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
    novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=2, L=2)

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

    novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=2, L=2)

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


def test_contrast_scores_for_sections_repetitions_and_novelty():
    rng = np.random.default_rng(0)
    section_sizes = [5, 5, 5, 6]
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
    assert size == 21
    assert len(sections) == section_count
    assert len(repetitions) == 2

    ssm = rng.normal(loc=0.0, scale=0.1, size=(size, size))
    for (row_start, row_stop) in section_bounds:
        ssm[row_start:row_stop, row_start:row_stop] = 0.96 + 0.02 * rng.normal(size=(row_stop - row_start, row_stop - row_start))

    for row_start, row_stop, col_start, col_stop in repetition_blocks:
        ssm[row_start:row_stop, col_start:col_stop] = 0.97 + 0.02 * rng.normal(size=(row_stop - row_start, col_stop - col_start))

    ssm = np.clip(ssm, 0.0, 1.0)

    bad_ssm = rng.normal(loc=0.0, scale=1, size=(size, size))
    bad_ssm = np.clip(bad_ssm, 0.0, 1.0)

    print("SSM GT:\n", ssm_gt)
    print("SSM:\n", ssm)
    print("Bad SSM:\n", bad_ssm)

    homogeneity = compute_homogeneity_scores(ssm_gt, ssm_gt)
    repetition = compute_repetition_scores(ssm_gt, ssm_gt)
    novelty = compute_novelty_scores(ssm_gt, ssm_gt, boundary_band_width=2, L=2)

    print('\n### SCORES FOR GROUND-TRUTH SSM ###')
    if plt is not None:
        _plot_ssm_and_novelty(ssm_gt, ssm_gt, novelty["novelty_response"], "GROUND-TRUTH SSM")
    print("Homogeneity per section:", homogeneity["per_section"])
    print("Homogeneity global:", homogeneity["global"])
    print("Repetition per repetition:", repetition["per_repetition"])
    print("Repetition global:", repetition["global"])
    print("Novelty per boundary:", novelty["per_boundary"])
    print("Novelty response:", novelty["novelty_response"])
    print("Novelty global:", novelty["global"])

    homogeneity = compute_homogeneity_scores(ssm, ssm_gt)
    repetition = compute_repetition_scores(ssm, ssm_gt)
    novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=2, L=2)

    print('\n### SCORES FOR SSM ###')
    if plt is not None:
        _plot_ssm_and_novelty(ssm, ssm_gt, novelty["novelty_response"], "SSM")
    print("Homogeneity per section:", homogeneity["per_section"])
    print("Homogeneity global:", homogeneity["global"])
    print("Repetition per repetition:", repetition["per_repetition"])
    print("Repetition global:", repetition["global"])
    print("Novelty per boundary:", novelty["per_boundary"])
    print("Novelty response:", novelty["novelty_response"])
    print("Novelty global:", novelty["global"])

    homogeneity = compute_homogeneity_scores(bad_ssm, ssm_gt)
    repetition = compute_repetition_scores(bad_ssm, ssm_gt)
    novelty = compute_novelty_scores(bad_ssm, ssm_gt, boundary_band_width=2, L=2)

    print('\n### SCORES FOR BAD SSM ###')
    if plt is not None:
        _plot_ssm_and_novelty(bad_ssm, ssm_gt, novelty["novelty_response"], "Bad SSM")
    print("Homogeneity per section:", homogeneity["per_section"])
    print("Homogeneity global:", homogeneity["global"])
    print("Repetition per repetition:", repetition["per_repetition"])
    print("Repetition global:", repetition["global"])
    print("Novelty per boundary:", novelty["per_boundary"])
    print("Novelty global:", novelty["global"])

    assert len(homogeneity["per_section"]) == section_count
    assert len(repetition["per_repetition"]) == 2
    assert len(novelty["per_boundary"]) == section_count - 1
    assert np.isfinite(homogeneity["global"])
    assert np.isfinite(repetition["global"])
    assert np.isfinite(novelty["global"])


def _plot_ssm_and_novelty(ssm: np.ndarray, ssm_gt: np.ndarray, novelty_response: np.ndarray, title: str) -> None:
    if plt is None:
        print("matplotlib is not available, cannot plot SSMs and novelty response.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(ssm_gt, cmap="gray", origin="lower", aspect="auto")
    axes[0].set_title(f"{title} - SSM_GT")
    axes[1].imshow(ssm, cmap="viridis", origin="lower", aspect="auto")
    axes[1].set_title(f"{title} - SSM")
    axes[2].plot(novelty_response, marker="o")
    axes[2].set_title(f"{title} - Novelty response")
    axes[2].set_xlabel("Index")
    axes[2].set_ylabel("Novelty")
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"./tests/{title.replace(' ', '_')}.png")
    plt.show()


if __name__ == "__main__":
    test_contrast_scores_for_sections_repetitions_and_novelty()