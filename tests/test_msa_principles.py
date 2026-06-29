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
    novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=1, L=1)

    assert len(homogeneity["per_section"]) == 2
    assert np.isfinite(homogeneity["global"])
    assert len(novelty["per_boundary"]) == 1
    assert np.isfinite(novelty["global"])


def test_ssm_quality_metrics_returns_expected_keys():
    ssm = np.eye(3, dtype=float)
    ssm_gt = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)

    metrics = ssm_quality_metrics(ssm, ssm_gt)

    assert set(metrics) >= {"homogeneity", "repetition", "novelty"}
    assert metrics["section_count"] == 3


def test_contrast_scores_for_sections_repetitions_and_novelty():
    rng = np.random.default_rng(0)
    section_size = 2
    section_count = 4
    size = section_size * section_count

    ssm_gt = np.zeros((size, size), dtype=int)
    for section_index in range(section_count):
        start = section_index * section_size
        stop = start + section_size
        ssm_gt[start:stop, start:stop] = 1

    repetition_blocks = [
        (0, section_size, 6, 8),
        (6, 8, 0, section_size),
    ]
    for row_start, row_stop, col_start, col_stop in repetition_blocks:
        ssm_gt[row_start:row_stop, col_start:col_stop] = 1

    sections, repetitions = extract_ssm_regions(ssm_gt)
    assert len(sections) == section_count
    assert len(repetitions) == 2

    ssm = rng.normal(loc=0.0, scale=0.1, size=(size, size))
    for section_index in range(section_count):
        start = section_index * section_size
        stop = start + section_size
        ssm[start:stop, start:stop] = 0.96 + 0.02 * rng.normal(size=(section_size, section_size))

    for row_start, row_stop, col_start, col_stop in repetition_blocks:
        ssm[row_start:row_stop, col_start:col_stop] = 0.97 + 0.02 * rng.normal(size=(section_size, section_size))

    ssm = np.clip(ssm, 0.0, 1.0)

    bad_ssm = rng.normal(loc=0.0, scale=1, size=(size, size))
    bad_ssm = np.clip(bad_ssm, 0.0, 1.0)

    print("SSM:\n", ssm)
    print("Bad SSM:\n", bad_ssm)
    print("SSM GT:\n", ssm_gt)

    homogeneity = compute_homogeneity_scores(ssm, ssm_gt)
    repetition = compute_repetition_scores(ssm, ssm_gt)
    novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=1, L=1)

    print('### SCORES FOR SSM ###')
    print("Homogeneity per section:", homogeneity["per_section"])
    print("Homogeneity global:", homogeneity["global"])
    print("Repetition per repetition:", repetition["per_repetition"])
    print("Repetition global:", repetition["global"])
    print("Novelty per boundary:", novelty["per_boundary"])
    print("Novelty global:", novelty["global"])

    homogeneity = compute_homogeneity_scores(bad_ssm, ssm_gt)
    repetition = compute_repetition_scores(bad_ssm, ssm_gt)
    novelty = compute_novelty_scores(bad_ssm, ssm_gt, boundary_band_width=1, L=1)

    print('\n### SCORES FOR BAD SSM ###')
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


if __name__ == "__main__":
    test_contrast_scores_for_sections_repetitions_and_novelty()