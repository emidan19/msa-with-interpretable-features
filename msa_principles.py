"""Quality metrics for self-similarity matrices and structural ground truth.

This module supports region extraction from a binary structural annotation
matrix (SSM_GT) and computes interpretable quality metrics for a candidate
self-similarity matrix (SSM).

The ground-truth matrix SSM_GT contains 1 for pixels that belong to the same
musical section or to a structural repetition, and 0 otherwise.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SsmRegion:
    """Connected component region extracted from a binary SSM_GT."""

    label: int
    region_type: str
    pixels: np.ndarray

    @property
    def is_section(self) -> bool:
        return self.region_type == "section"

    @property
    def is_repetition(self) -> bool:
        return self.region_type == "repetition"

    @property
    def is_boundary(self) -> bool:
        return self.region_type == "boundary"

    @property
    def size(self) -> int:
        return int(self.pixels.shape[0])


def _validate_square_binary_ssm_gt(ssm_gt: np.ndarray) -> np.ndarray:
    matrix = np.asarray(ssm_gt)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square SSM_GT matrix, got shape {matrix.shape}")
    if matrix.size == 0:
        return matrix.astype(np.uint8, copy=False)

    if not np.array_equal(matrix, matrix.astype(bool)):
        raise ValueError("SSM_GT must be binary (0/1 values).")

    return matrix.astype(np.uint8, copy=False)


def _connected_components_4(image: np.ndarray, background: int = 0, connectivity: int = 1) -> tuple[np.ndarray, int]:
    """Simple 4-connected component labeling used as a fallback when scikit-image is unavailable."""

    if connectivity != 1:
        raise ValueError("Only connectivity=1 is supported.")
    if background != 0:
        raise ValueError("Only background=0 is supported.")

    image = np.asarray(image)
    labels = np.zeros(image.shape, dtype=np.int32)
    component_count = 0
    height, width = image.shape

    for row in range(height):
        for col in range(width):
            if image[row, col] == 0 or labels[row, col] != 0:
                continue

            component_count += 1
            stack = [(row, col)]
            labels[row, col] = component_count
            while stack:
                current_row, current_col = stack.pop()
                for next_row, next_col in ((current_row - 1, current_col), (current_row + 1, current_col), (current_row, current_col - 1), (current_row, current_col + 1)):
                    if 0 <= next_row < height and 0 <= next_col < width:
                        if image[next_row, next_col] != 0 and labels[next_row, next_col] == 0:
                            labels[next_row, next_col] = component_count
                            stack.append((next_row, next_col))

    return labels, int(component_count)


def label_ssm_regions(ssm_gt: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected components in a binary structural SSM_GT."""

    matrix = _validate_square_binary_ssm_gt(ssm_gt)

    try:
        from skimage.measure import label as label_connected_components
    except ModuleNotFoundError:
        labels, num = _connected_components_4(matrix, background=0, connectivity=1)
    else:
        labels, num = label_connected_components(matrix, background=0, return_num=True, connectivity=1)

    return labels, int(num)


def _extract_ssm_regions_from_segment_ids(matrix: np.ndarray, segment_ids: np.ndarray) -> tuple[list[SsmRegion], list[SsmRegion]]:
    """Extract section/repetition regions using known segment instance ids.

    Unlike connected-component labeling, this disambiguates two temporally
    adjacent segments that share the same label (e.g. consecutive `verse`
    sections): since each annotated segment has a distinct id, the section
    for id ``s`` is exactly the block where both row and column belong to
    ``s``, and a repetition is any off-diagonal block between two different
    ids that is marked 1 in the SSM_GT.
    """

    ids = np.asarray(segment_ids)
    if ids.shape[0] != matrix.shape[0]:
        raise ValueError(f"segment_ids length {ids.shape[0]} does not match SSM_GT size {matrix.shape[0]}.")

    positive = matrix.astype(bool)
    unique_ids = sorted(int(value) for value in np.unique(ids) if value >= 0)

    sections: list[SsmRegion] = []
    repetitions: list[SsmRegion] = []
    next_label = 1

    for segment_id in unique_ids:
        row_mask = ids == segment_id
        region_mask = row_mask[:, None] & row_mask[None, :] & positive
        if not np.any(region_mask):
            continue
        pixels = np.column_stack(np.nonzero(region_mask))
        sections.append(SsmRegion(label=next_label, region_type="section", pixels=pixels))
        next_label += 1

    for row_id, col_id in itertools.permutations(unique_ids, 2):
        region_mask = (ids == row_id)[:, None] & (ids == col_id)[None, :] & positive
        if not np.any(region_mask):
            continue
        pixels = np.column_stack(np.nonzero(region_mask))
        repetitions.append(SsmRegion(label=next_label, region_type="repetition", pixels=pixels))
        next_label += 1

    return sections, repetitions


def extract_ssm_regions(ssm_gt: np.ndarray, segment_ids: np.ndarray | None = None) -> tuple[list[SsmRegion], list[SsmRegion]]:
    """Extract section and repetition regions from a binary SSM_GT.

    Parameters
    ----------
    segment_ids:
        Optional per-frame annotated segment instance id (distinct even for
        temporally adjacent segments sharing the same label). When provided,
        regions are derived directly from segment identity instead of pixel
        connectivity, which correctly separates a section from an adjacent
        repetition even when they touch with no zero-valued gap between
        them (a case plain connected-component labeling cannot resolve).

    Returns
    -------
    sections:
        Connected components containing at least one main-diagonal pixel
        (or, with `segment_ids`, the exact block for each segment id).
    repetitions:
        Connected components containing no main-diagonal pixels (or, with
        `segment_ids`, the off-diagonal blocks between different segment ids).
    """

    matrix = _validate_square_binary_ssm_gt(ssm_gt)

    if segment_ids is not None:
        return _extract_ssm_regions_from_segment_ids(matrix, segment_ids)

    labels, num_regions = label_ssm_regions(matrix)
    n = matrix.shape[0]
    diagonal_mask = np.eye(n, dtype=bool)

    sections: list[SsmRegion] = []
    repetitions: list[SsmRegion] = []

    for region_label in range(1, num_regions + 1):
        region_mask = labels == region_label
        if not np.any(region_mask):
            continue

        region_type = "section" if np.any(region_mask & diagonal_mask) else "repetition"
        pixels = np.column_stack(np.nonzero(region_mask))
        region = SsmRegion(label=region_label, region_type=region_type, pixels=pixels)

        if region.is_section:
            sections.append(region)
        else:
            repetitions.append(region)

    return sections, repetitions


def _validate_square_ssm(ssm: np.ndarray) -> np.ndarray:
    matrix = np.asarray(ssm, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square SSM, got shape {matrix.shape}")
    return matrix


def _masked_mean(matrix: np.ndarray, mask: np.ndarray) -> float:
    values = matrix[mask]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _pixels_to_mask(shape: tuple[int, int], pixels: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if pixels.size == 0:
        return mask
    mask[tuple(pixels.T)] = True
    return mask


def _region_local_background(background_mask: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    if pixels.size == 0:
        raise ValueError("Cannot compute a local background for an empty region.")

    row_indices = pixels[:, 0]
    row_start = int(np.min(row_indices))
    row_end = int(np.max(row_indices))
    row_selector = np.arange(background_mask.shape[0])[:, None]
    local_mask = background_mask & (row_selector >= row_start) & (row_selector <= row_end)
    if not np.any(local_mask):
        raise ValueError("Local background is empty for the requested region.")
    return local_mask


def _region_contrast_score(matrix: np.ndarray, pixels: np.ndarray, background_mask: np.ndarray, eps: float = 1e-12) -> float:
    if pixels.size == 0:
        raise ValueError("Region is empty.")

    background_pixels = np.column_stack(np.nonzero(background_mask))
    if background_pixels.size == 0:
        raise ValueError("Local background is empty.")

    matrix = np.asarray(matrix)
    if matrix.ndim == 1:
        region_energy = float(np.mean(matrix[pixels[:, 0]]))
        background_energy = float(np.mean(matrix[background_pixels[:, 0]]))
    else:
        region_energy = float(np.mean(matrix[pixels[:, 0], pixels[:, 1]]))
        background_energy = float(np.mean(matrix[background_pixels[:, 0], background_pixels[:, 1]]))

    return (region_energy - background_energy) / (region_energy + background_energy + eps)


def compute_homogeneity_scores(ssm: np.ndarray, ssm_gt: np.ndarray, eps: float = 1e-12, segment_ids: np.ndarray | None = None) -> dict[str, Any]:
    """Compute section-wise contrast scores against a local background.

    The local background is the subset of the complement of the structural
    annotation whose row coordinates are within the vertical extent of the
    target section.
    """

    matrix = _validate_square_ssm(ssm)
    gt = _validate_square_binary_ssm_gt(ssm_gt)
    if matrix.shape != gt.shape:
        raise ValueError("SSM and SSM_GT must have the same shape.")

    sections, _ = extract_ssm_regions(gt, segment_ids=segment_ids)
    background_mask = ~gt.astype(bool)
    scores = []
    for section in sections:
        local_background = _region_local_background(background_mask, section.pixels)
        scores.append(_region_contrast_score(matrix, section.pixels, local_background, eps=eps))

    return {
        "global": float(np.mean(scores)) if scores else float("nan"),
        "per_section": scores,
        "section_labels": [section.label for section in sections],
        "section_pixels": [section.size for section in sections],
    }


def compute_repetition_scores(ssm: np.ndarray, ssm_gt: np.ndarray, eps: float = 1e-12, segment_ids: np.ndarray | None = None) -> dict[str, Any]:
    """Compute repetition-wise contrast scores against a local background."""

    matrix = _validate_square_ssm(ssm)
    gt = _validate_square_binary_ssm_gt(ssm_gt)
    if matrix.shape != gt.shape:
        raise ValueError("SSM and SSM_GT must have the same shape.")

    _, repetitions = extract_ssm_regions(gt, segment_ids=segment_ids)
    background_mask = ~gt.astype(bool)
    scores = []
    for repetition in repetitions:
        local_background = _region_local_background(background_mask, repetition.pixels)
        scores.append(_region_contrast_score(matrix, repetition.pixels, local_background, eps=eps))

    return {
        "global": float(np.mean(scores)) if scores else float("nan"),
        "per_repetition": scores,
        "repetition_labels": [repetition.label for repetition in repetitions],
        "repetition_pixels": [repetition.size for repetition in repetitions],
    }


def compute_kernel_checkerboard_box(L: int) -> np.ndarray:
    """Compute a box-like checkerboard kernel following Meinard Müller.

    Reference: Meinard Müller, Fundamentals of Music Processing, Chapter 4,
    novelty-based segmentation.
    """

    L = int(L)
    if L <= 0:
        raise ValueError("Kernel radius L must be strictly positive.")
    axis = np.arange(-L, L + 1)
    return np.outer(np.sign(axis), np.sign(axis))


def compute_kernel_checkerboard_gaussian(L: int, var: float = 1.0, normalize: bool = True) -> np.ndarray:
    """Compute a Gaussian-tapered checkerboard kernel following Meinard Müller."""

    L = int(L)
    if L <= 0:
        raise ValueError("Kernel radius L must be strictly positive.")
    if var <= 0:
        raise ValueError("Variance var must be strictly positive.")

    taper = np.sqrt(1.0 / 2.0) / (L * var)
    axis = np.arange(-L, L + 1)
    gaussian_1d = np.exp(-(taper**2) * (axis**2))
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian_2d
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel.astype(np.float64)


def compute_novelty_ssm(
    ssm: np.ndarray,
    kernel: np.ndarray | None = None,
    L: int = 10,
    var: float = 0.5,
    normalize: bool = True,
    exclude: bool = False,
) -> np.ndarray:
    """Compute a novelty response from an SSM using a checkerboard kernel."""

    matrix = _validate_square_ssm(ssm)
    if kernel is None:
        kernel = compute_kernel_checkerboard_gaussian(L=L, var=var, normalize=normalize)
    else:
        kernel = np.asarray(kernel, dtype=np.float64)

    n = matrix.shape[0]
    if kernel.shape != (2 * int(L) + 1, 2 * int(L) + 1):
        raise ValueError(f"Expected kernel of shape {(2 * int(L) + 1, 2 * int(L) + 1)}, got {kernel.shape}")

    novelty = np.zeros(n, dtype=np.float64)
    padded = np.pad(matrix, int(L), mode="constant")
    for index in range(n):
        novelty[index] = np.sum(padded[index : index + 2 * L + 1, index : index + 2 * L + 1] * kernel)

    if exclude:
        right = min(L, n)
        left = max(0, n - L)
        novelty[0:right] = 0.0
        novelty[left:n] = 0.0

    return novelty


def _build_boundary_regions(ssm_gt: np.ndarray, band_width: int = 8, segment_ids: np.ndarray | None = None) -> list[SsmRegion]:
    """Build boundary regions between adjacent sections using a diagonal band.

    The band width controls how wide the diagonal neighborhood around each
    section boundary is.
    """

    gt = _validate_square_binary_ssm_gt(ssm_gt)
    sections, _ = extract_ssm_regions(gt, segment_ids=segment_ids)
    if len(sections) < 2:
        return []

    ordered_sections = sorted(sections, key=lambda region: int(np.min(region.pixels[:, 0])))
    boundaries: list[SsmRegion] = []
    n = gt.shape[0]
    band_width = max(0, int(band_width))

    for index, (previous_section, next_section) in enumerate(zip(ordered_sections, ordered_sections[1:])):
        previous_end = int(np.max(previous_section.pixels[:, 0]))
        next_start = int(np.min(next_section.pixels[:, 0]))
        boundary_center = int(np.round(0.5 * (previous_end + next_start)))
        boundary_center = min(max(boundary_center, 0), n - 1)

        row_indices, col_indices = np.indices((n, n))
        mask = (
            (np.abs(row_indices - col_indices) <= band_width)
            & (row_indices >= max(0, boundary_center - band_width))
            & (row_indices <= min(n - 1, boundary_center + band_width))
            & (col_indices >= max(0, boundary_center - band_width))
            & (col_indices <= min(n - 1, boundary_center + band_width))
        )
        pixels = np.column_stack(np.nonzero(mask))
        boundaries.append(SsmRegion(label=index + 1, region_type="boundary", pixels=pixels))

    return boundaries


def _build_boundary_transitions(ssm_gt: np.ndarray, band_width: int = 8, segment_ids: np.ndarray | None = None) -> list[tuple[SsmRegion, SsmRegion, SsmRegion]]:
    """Build transition tuples for adjacent section boundaries.

    Each transition contains a boundary region and the boundary-facing half of
    the left/right sections that it separates.
    """

    gt = _validate_square_binary_ssm_gt(ssm_gt)
    sections, _ = extract_ssm_regions(gt, segment_ids=segment_ids)
    if len(sections) < 2:
        return []

    ordered_sections = sorted(sections, key=lambda region: int(np.min(region.pixels[:, 0])))
    transitions: list[tuple[SsmRegion, SsmRegion, SsmRegion]] = []
    n = gt.shape[0]
    band_width = max(0, int(band_width))

    for index, (previous_section, next_section) in enumerate(zip(ordered_sections, ordered_sections[1:])):
        previous_start = int(np.min(previous_section.pixels[:, 0]))
        previous_end = int(np.max(previous_section.pixels[:, 0]))
        next_start = int(np.min(next_section.pixels[:, 0]))
        next_end = int(np.max(next_section.pixels[:, 0]))
        boundary_center = int(np.round(0.5 * (previous_end + next_start)))
        boundary_center = min(max(boundary_center, 0), n - 1)
        boundary_start = max(0, boundary_center - band_width)
        boundary_end = min(n - 1, boundary_center + band_width)

        row_indices, col_indices = np.indices((n, n))
        mask = (
            (np.abs(row_indices - col_indices) <= band_width)
            & (row_indices >= boundary_start)
            & (row_indices <= boundary_end)
            & (col_indices >= boundary_start)
            & (col_indices <= boundary_end)
        )
        pixels = np.column_stack(np.nonzero(mask))
        boundary = SsmRegion(label=index + 1, region_type="boundary", pixels=pixels)

        previous_middle = int(np.round(0.5 * (previous_start + previous_end)))
        next_middle = int(np.round(0.5 * (next_start + next_end)))
        previous_context = _slice_region_by_rows(previous_section, previous_middle, boundary_start - 1)
        next_context = _slice_region_by_rows(next_section, boundary_end + 1, next_middle)
        transitions.append((boundary, previous_context, next_context))

    return transitions


def _slice_region_by_rows(region: SsmRegion, row_start: int, row_end: int) -> SsmRegion:
    region_start, region_end = _region_row_span(region)
    row_start = max(region_start, int(row_start))
    row_end = min(region_end, int(row_end))
    if row_start > row_end:
        row_start, row_end = region_start, region_end

    row_mask = (region.pixels[:, 0] >= row_start) & (region.pixels[:, 0] <= row_end)
    pixels = region.pixels[row_mask]
    if pixels.size == 0:
        pixels = region.pixels
    return SsmRegion(label=region.label, region_type=region.region_type, pixels=pixels)


def _region_row_span(region: SsmRegion) -> tuple[int, int]:
    row_indices = region.pixels[:, 0]
    return int(np.min(row_indices)), int(np.max(row_indices))


def _region_temporal_indices(region: SsmRegion) -> np.ndarray:
    row_start, row_end = _region_row_span(region)
    return np.arange(row_start, row_end + 1, dtype=int)


def compute_novelty_scores(
    ssm: np.ndarray,
    ssm_gt: np.ndarray,
    eps: float = 1e-12,
    boundary_band_width: int = 3,
    kernel: np.ndarray | None = None,
    L: int = 16,
    var: float = 0.5,
    normalize: bool = True,
    exclude: bool = False,
    segment_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute novelty contrast scores for section transitions.

    Each transition is defined by a boundary region and the two adjacent
    sections that it separates. The score compares the mean novelty response
    on the boundary against the average mean novelty response of the left and
    right sections.

    The checkerboard novelty response is signed, so the contrast denominator
    uses absolute energies. This keeps the score bounded and prevents noisy
    negative section responses from turning weak boundaries into high scores.
    """

    matrix = _validate_square_ssm(ssm)
    gt = _validate_square_binary_ssm_gt(ssm_gt)
    if matrix.shape != gt.shape:
        raise ValueError("SSM and SSM_GT must have the same shape.")

    novelty_response = compute_novelty_ssm(matrix, kernel=kernel, L=L, var=var, normalize=normalize, exclude=exclude)
    transitions = _build_boundary_transitions(gt, band_width=boundary_band_width, segment_ids=segment_ids)
    scores = []
    boundary_means = []
    left_section_means = []
    right_section_means = []
    section_reference_means = []
    for boundary, left_section, right_section in transitions:
        boundary_indices = _region_temporal_indices(boundary)
        left_indices = np.setdiff1d(_region_temporal_indices(left_section), boundary_indices, assume_unique=True)
        right_indices = np.setdiff1d(_region_temporal_indices(right_section), boundary_indices, assume_unique=True)
        if left_indices.size == 0:
            left_indices = _region_temporal_indices(left_section)
        if right_indices.size == 0:
            right_indices = _region_temporal_indices(right_section)

        boundary_mean = float(np.mean(novelty_response[boundary_indices]))
        left_mean = float(np.mean(novelty_response[left_indices]))
        right_mean = float(np.mean(novelty_response[right_indices]))
        section_mean = 0.5 * (left_mean + right_mean)
        scores.append((boundary_mean - section_mean) / (abs(boundary_mean) + abs(section_mean) + eps))
        boundary_means.append(boundary_mean)
        left_section_means.append(left_mean)
        right_section_means.append(right_mean)
        section_reference_means.append(section_mean)

    return {
        "global": float(np.mean(scores)) if scores else float("nan"),
        "per_transition": scores,
        "per_boundary": scores,
        "boundary_means": boundary_means,
        "left_section_means": left_section_means,
        "right_section_means": right_section_means,
        "section_reference_means": section_reference_means,
        "boundary_labels": [boundary.label for boundary, _, _ in transitions],
        "boundary_pixels": [boundary.size for boundary, _, _ in transitions],
        "transition_left_labels": [left.label for _, left, _ in transitions],
        "transition_right_labels": [right.label for _, _, right in transitions],
        "novelty_response": novelty_response,
    }


def ssm_quality_metrics(
    ssm: np.ndarray,
    ssm_gt: np.ndarray,
    *,
    eps: float = 1e-12,
    boundary_band_width: int = 4,
    segment_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute quality metrics for SSM against a binary structural SSM_GT."""

    matrix = _validate_square_ssm(ssm)
    gt = _validate_square_binary_ssm_gt(ssm_gt)
    if matrix.shape != gt.shape:
        raise ValueError("SSM and SSM_GT must have the same shape.")

    sections, repetitions = extract_ssm_regions(gt, segment_ids=segment_ids)
    positive_mask = gt.astype(bool)
    negative_mask = ~positive_mask

    section_mean = float("nan")
    repetition_mean = float("nan")
    if sections:
        section_mask = np.zeros_like(positive_mask)
        for region in sections:
            section_mask[tuple(region.pixels.T)] = True
        section_mean = _masked_mean(matrix, section_mask)

    if repetitions:
        repetition_mask = np.zeros_like(positive_mask)
        for region in repetitions:
            repetition_mask[tuple(region.pixels.T)] = True
        repetition_mean = _masked_mean(matrix, repetition_mask)

    background_mean = _masked_mean(matrix, negative_mask)
    global_quality = float("nan")
    if not np.isnan(section_mean) and not np.isnan(repetition_mean):
        global_quality = 0.5 * (section_mean + repetition_mean) - background_mean

    homogeneity = compute_homogeneity_scores(matrix, gt, eps=eps, segment_ids=segment_ids)
    repetition = compute_repetition_scores(matrix, gt, eps=eps, segment_ids=segment_ids)
    novelty = compute_novelty_scores(matrix, gt, eps=eps, boundary_band_width=boundary_band_width, segment_ids=segment_ids)

    return {
        "section_mean": section_mean,
        "repetition_mean": repetition_mean,
        "background_mean": background_mean,
        "global_quality": global_quality,
        "section_count": len(sections),
        "repetition_count": len(repetitions),
        "section_pixels": int(np.sum([region.size for region in sections])),
        "repetition_pixels": int(np.sum([region.size for region in repetitions])),
        "background_pixels": int(np.sum(negative_mask)),
        "homogeneity": homogeneity,
        "repetition": repetition,
        "novelty": novelty,
    }


__all__ = [
    "SsmRegion",
    "label_ssm_regions",
    "extract_ssm_regions",
    "compute_homogeneity_scores",
    "compute_repetition_scores",
    "compute_kernel_checkerboard_box",
    "compute_kernel_checkerboard_gaussian",
    "compute_novelty_ssm",
    "compute_novelty_scores",
    "ssm_quality_metrics",
]
