"""Music structure analysis metrics inspired by recent MSA literature.

This module currently implements two metric families:

1. Homogeneity / regularity segment scoring inspired by the CBM paper
   (Marmoret et al., WASPAA 2023), operating on self-similarity matrices.
2. Regularity and balance metrics inspired by McFee (ISMIR 2025), supporting
   both musical-time (integer durations) and absolute-time segmentations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import mir_eval


@dataclass(frozen=True)
class Segment:
    """Simple interval container used by the regularity metrics."""

    start: float
    end: float
    label: str | None = None

    @property
    def duration(self) -> float:
        return float(self.end - self.start)


@dataclass(frozen=True)
class RegularityPairStats:
    """Pairwise summary for regularity or balance computations."""

    score: float
    pair_count: int


def _validate_square_ssm(ssm: np.ndarray) -> np.ndarray:
    matrix = np.asarray(ssm, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square SSM, got shape {matrix.shape}")
    return matrix


def _validate_segment_bounds(start: int, end: int, size: int) -> tuple[int, int]:
    start = int(start)
    end = int(end)
    if start < 0 or end < 0 or start >= end or end > size:
        raise ValueError(f"Invalid segment bounds [{start}, {end}) for matrix size {size}")
    return start, end


def make_cbm_kernel(size: int, kind: str = "full", bands: int = 7) -> np.ndarray:
    """Build a CBM convolution kernel for a segment of length `size`.

    Parameters
    ----------
    size:
        Segment size in bars/frames.
    kind:
        `"full"` or `"band"`.
    bands:
        Number of off-diagonal bands used when `kind="band"`.
    """

    size = int(size)
    if size <= 0:
        raise ValueError("Kernel size must be strictly positive.")
    kernel = np.zeros((size, size), dtype=np.float32)
    if kind == "full":
        kernel.fill(1.0)
        np.fill_diagonal(kernel, 0.0)
        return kernel
    if kind != "band":
        raise ValueError(f"Unknown kernel kind: {kind!r}")
    bands = max(1, int(bands))
    for offset in range(1, bands + 1):
        idx = np.arange(size - offset)
        kernel[idx, idx + offset] = 1.0
        kernel[idx + offset, idx] = 1.0
    return kernel


def cbm_normalization(
    ssm: np.ndarray,
    kernel_kind: str = "full",
    bands: int = 7,
    reference_size: int = 8,
) -> float:
    """Compute the CBM normalization constant nu for one SSM.

    The paper defines `nu` as the maximal convolution value obtained by sliding
    a kernel of size 8 across the song SSM.
    """

    matrix = _validate_square_ssm(ssm)
    n = matrix.shape[0]
    reference_size = int(reference_size)
    if n == 0:
        return 1.0
    if reference_size <= 0:
        raise ValueError("reference_size must be strictly positive.")
    if n < reference_size:
        reference_size = n

    kernel = make_cbm_kernel(reference_size, kind=kernel_kind, bands=bands)
    best = 0.0
    for start in range(0, n - reference_size + 1):
        local = matrix[start : start + reference_size, start : start + reference_size]
        conv = float(np.sum(local * kernel))
        if conv > best:
            best = conv
    return best if best > 0.0 else 1.0


def cbm_homogeneity_score(
    ssm: np.ndarray,
    start: int,
    end: int,
    *,
    kernel_kind: str = "full",
    bands: int = 7,
    normalization: float | None = None,
    reference_size: int = 8,
) -> float:
    """Compute the CBM homogeneity term u_K(S_i) for one segment."""

    matrix = _validate_square_ssm(ssm)
    start, end = _validate_segment_bounds(start, end, matrix.shape[0])
    size = end - start
    kernel = make_cbm_kernel(size, kind=kernel_kind, bands=bands)
    local = matrix[start:end, start:end]
    nu = normalization
    if nu is None:
        nu = cbm_normalization(matrix, kernel_kind=kernel_kind, bands=bands, reference_size=reference_size)
    nu = float(max(nu, 1e-8))
    return float(np.sum(local * kernel) / (nu * size))


def cbm_penalty(size: int) -> float:
    """Regularity penalty p(n) from the CBM paper."""

    size = int(size)
    if size <= 0:
        raise ValueError("Segment size must be strictly positive.")
    if size == 8:
        return 0.0
    if size % 4 == 0:
        return 0.25
    if size % 2 == 0:
        return 0.5
    return 1.0


def cbm_segment_score(
    ssm: np.ndarray,
    start: int,
    end: int,
    *,
    kernel_kind: str = "full",
    bands: int = 7,
    normalization: float | None = None,
    reference_size: int = 8,
    lambda_reg: float = 1.0,
    use_penalty: bool = False,
    return_homogeneity_penalty: bool = False,
) -> float:
    """Compute the CBM segment score.

    By default, this returns the homogeneity-only score `u_K(S_i)`. Set
    `use_penalty=True` to obtain the mixed score `u_K(S_i) - lambda * p(n)`.
    """

    start, end = _validate_segment_bounds(start, end, _validate_square_ssm(ssm).shape[0])
    homogeneity_score = cbm_homogeneity_score(
        ssm,
        start,
        end,
        kernel_kind=kernel_kind,
        bands=bands,
        normalization=normalization,
        reference_size=reference_size,
    )
    if not use_penalty and not return_homogeneity_penalty:
        return homogeneity_score
    elif return_homogeneity_penalty:
        penalty = cbm_penalty(end - start)
        return np.array([homogeneity_score - float(lambda_reg) * cbm_penalty(end - start), homogeneity_score, float(lambda_reg) * penalty])
    else:
        return homogeneity_score - float(lambda_reg) * cbm_penalty(end - start)


def coerce_segments(segments: Iterable[Segment | Sequence[float]]) -> list[Segment]:
    """Normalize supported segment representations to `Segment` objects."""

    normalized: list[Segment] = []
    for item in segments:
        if isinstance(item, Segment):
            normalized.append(item)
            continue
        if len(item) == 2:
            start, end = item
            label = None
        elif len(item) >= 3:
            start, end, label = item[:3]
        else:
            raise ValueError(f"Cannot coerce segment from {item!r}")
        normalized.append(Segment(float(start), float(end), None if label is None else str(label)))
    return normalized


def durations_from_boundaries(boundaries: Sequence[float]) -> np.ndarray:
    boundaries = np.asarray(boundaries, dtype=np.float64)
    if boundaries.ndim != 1 or boundaries.size < 2:
        raise ValueError("Need at least two boundaries to define segment durations.")
    diffs = np.diff(boundaries)
    if np.any(diffs <= 0):
        raise ValueError("Boundaries must be strictly increasing.")
    return diffs


def durations_from_segments(segments: Iterable[Segment | Sequence[float]]) -> np.ndarray:
    normalized = coerce_segments(segments)
    if not normalized:
        return np.zeros(0, dtype=np.float64)
    durations = np.asarray([segment.duration for segment in normalized], dtype=np.float64)
    if np.any(durations <= 0):
        raise ValueError("All segment durations must be strictly positive.")
    return durations


def labels_from_segments(segments: Iterable[Segment | Sequence[float]]) -> list[str | None]:
    return [segment.label for segment in coerce_segments(segments)]


def regularity_pair(d1: int, d2: int) -> float:
    """Pairwise regularity rho(d1, d2) from the ISMIR 2025 paper."""

    d1 = int(d1)
    d2 = int(d2)
    if d1 <= 0 or d2 <= 0:
        raise ValueError("Durations must be strictly positive integers.")
    return float(gcd(d1, d2) / min(d1, d2))


def balance_pair(d1: int, d2: int) -> float:
    """Pairwise balance beta(d1, d2) from the ISMIR 2025 paper."""

    d1 = int(d1)
    d2 = int(d2)
    if d1 <= 0 or d2 <= 0:
        raise ValueError("Durations must be strictly positive integers.")
    return float(gcd(d1, d2) / max(d1, d2))


def _all_unordered_pairs(n_items: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]


def _adjacent_pairs(n_items: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(max(0, n_items - 1))]


def _same_label_pairs(labels: Sequence[str | None]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(len(labels)):
        if labels[i] is None:
            continue
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                pairs.append((i, j))
    return pairs


def _aggregate_pair_metric(
    durations: Sequence[int],
    pair_fn,
    pairs: Sequence[tuple[int, int]] | None = None,
) -> RegularityPairStats:
    durations = [int(value) for value in durations]
    if not durations:
        return RegularityPairStats(score=0.0, pair_count=0)
    if len(durations) == 1:
        return RegularityPairStats(score=1.0, pair_count=1)
    if pairs is None:
        pairs = _all_unordered_pairs(len(durations))
    if not pairs:
        return RegularityPairStats(score=0.0, pair_count=0)
    values = [pair_fn(durations[i], durations[j]) for i, j in pairs]
    return RegularityPairStats(score=float(np.mean(values)), pair_count=len(values))


def regularity(durations: Sequence[int]) -> float:
    return _aggregate_pair_metric(durations, regularity_pair).score


def balance(durations: Sequence[int]) -> float:
    return _aggregate_pair_metric(durations, balance_pair).score


def labeled_regularity(
    durations: Sequence[int],
    labels: Sequence[str | None],
) -> float:
    if len(durations) != len(labels):
        raise ValueError("durations and labels must have the same length.")
    return _aggregate_pair_metric(durations, regularity_pair, _same_label_pairs(labels)).score


def labeled_balance(
    durations: Sequence[int],
    labels: Sequence[str | None],
) -> float:
    if len(durations) != len(labels):
        raise ValueError("durations and labels must have the same length.")
    return _aggregate_pair_metric(durations, balance_pair, _same_label_pairs(labels)).score


def adjacent_regularity(durations: Sequence[int]) -> float:
    return _aggregate_pair_metric(durations, regularity_pair, _adjacent_pairs(len(durations))).score


def adjacent_balance(durations: Sequence[int]) -> float:
    return _aggregate_pair_metric(durations, balance_pair, _adjacent_pairs(len(durations))).score


def _quantize_duration(duration: float, frame_rate: float) -> int:
    frames = int(np.floor(float(duration) * float(frame_rate)))
    return max(1, frames)


def _tolerant_pair_metric(duration_a: float, duration_b: float, pair_fn, frame_rate: float, window_s: float) -> float:
    if duration_a <= 0.0 or duration_b <= 0.0:
        raise ValueError("Durations must be strictly positive.")
    frame_rate = float(frame_rate)
    window_s = float(window_s)
    window_frames = int(np.floor(window_s * frame_rate + 1e-9))
    base_a = float(duration_a)
    base_b = float(duration_b)
    best = 0.0
    for da in range(-window_frames, window_frames + 1):
        for db in range(-window_frames, window_frames + 1):
            adj_a = max(1, _quantize_duration(base_a + da / frame_rate, frame_rate))
            adj_b = max(1, _quantize_duration(base_b + db / frame_rate, frame_rate))
            best = max(best, pair_fn(adj_a, adj_b))
    return float(best)


def regularity_pair_absolute(d1_s: float, d2_s: float, frame_rate: float = 10.0, window_s: float = 0.5) -> float:
    return _tolerant_pair_metric(d1_s, d2_s, regularity_pair, frame_rate, window_s)


def balance_pair_absolute(d1_s: float, d2_s: float, frame_rate: float = 10.0, window_s: float = 0.5) -> float:
    return _tolerant_pair_metric(d1_s, d2_s, balance_pair, frame_rate, window_s)


def _aggregate_absolute_pair_metric(
    durations_s: Sequence[float],
    pair_fn,
    *,
    frame_rate: float = 10.0,
    window_s: float = 0.5,
    pairs: Sequence[tuple[int, int]] | None = None,
) -> RegularityPairStats:
    durations_s = [float(value) for value in durations_s]
    if not durations_s:
        return RegularityPairStats(score=0.0, pair_count=0)
    if len(durations_s) == 1:
        return RegularityPairStats(score=1.0, pair_count=1)
    if pairs is None:
        pairs = _all_unordered_pairs(len(durations_s))
    if not pairs:
        return RegularityPairStats(score=0.0, pair_count=0)
    values = [pair_fn(durations_s[i], durations_s[j], frame_rate=frame_rate, window_s=window_s) for i, j in pairs]
    return RegularityPairStats(score=float(np.mean(values)), pair_count=len(values))


def regularity_absolute(durations_s: Sequence[float], frame_rate: float = 10.0, window_s: float = 0.5) -> float:
    return _aggregate_absolute_pair_metric(durations_s, regularity_pair_absolute, frame_rate=frame_rate, window_s=window_s).score


def balance_absolute(durations_s: Sequence[float], frame_rate: float = 10.0, window_s: float = 0.5) -> float:
    return _aggregate_absolute_pair_metric(durations_s, balance_pair_absolute, frame_rate=frame_rate, window_s=window_s).score


def labeled_regularity_absolute(
    durations_s: Sequence[float],
    labels: Sequence[str | None],
    frame_rate: float = 10.0,
    window_s: float = 0.5,
) -> float:
    if len(durations_s) != len(labels):
        raise ValueError("durations and labels must have the same length.")
    return _aggregate_absolute_pair_metric(
        durations_s,
        regularity_pair_absolute,
        frame_rate=frame_rate,
        window_s=window_s,
        pairs=_same_label_pairs(labels),
    ).score


def labeled_balance_absolute(
    durations_s: Sequence[float],
    labels: Sequence[str | None],
    frame_rate: float = 10.0,
    window_s: float = 0.5,
) -> float:
    if len(durations_s) != len(labels):
        raise ValueError("durations and labels must have the same length.")
    return _aggregate_absolute_pair_metric(
        durations_s,
        balance_pair_absolute,
        frame_rate=frame_rate,
        window_s=window_s,
        pairs=_same_label_pairs(labels),
    ).score


def adjacent_regularity_absolute(durations_s: Sequence[float], frame_rate: float = 10.0, window_s: float = 0.5) -> float:
    return _aggregate_absolute_pair_metric(
        durations_s,
        regularity_pair_absolute,
        frame_rate=frame_rate,
        window_s=window_s,
        pairs=_adjacent_pairs(len(durations_s)),
    ).score


def adjacent_balance_absolute(durations_s: Sequence[float], frame_rate: float = 10.0, window_s: float = 0.5) -> float:
    return _aggregate_absolute_pair_metric(
        durations_s,
        balance_pair_absolute,
        frame_rate=frame_rate,
        window_s=window_s,
        pairs=_adjacent_pairs(len(durations_s)),
    ).score


def hierarchical_overlap_pairs(
    upper_level: Iterable[Segment | Sequence[float]],
    lower_level: Iterable[Segment | Sequence[float]],
) -> list[tuple[float, float]]:
    """Build the hierarchical pair set from the ISMIR 2025 paper.

    Each lower-level segment is matched to the maximally overlapping upper-level
    segment.
    """

    upper = coerce_segments(upper_level)
    lower = coerce_segments(lower_level)
    pairs: list[tuple[float, float]] = []
    if not upper or not lower:
        return pairs
    for child in lower:
        overlaps = [max(0.0, min(parent.end, child.end) - max(parent.start, child.start)) for parent in upper]
        best_index = int(np.argmax(overlaps))
        if overlaps[best_index] > 0.0:
            pairs.append((upper[best_index].duration, child.duration))
    return pairs


def hierarchical_regularity(
    upper_level: Iterable[Segment | Sequence[float]],
    lower_level: Iterable[Segment | Sequence[float]],
) -> float:
    pairs = hierarchical_overlap_pairs(upper_level, lower_level)
    if not pairs:
        return 0.0
    values = [regularity_pair(round(a), round(b)) for a, b in pairs]
    return float(np.mean(values))


def hierarchical_balance(
    upper_level: Iterable[Segment | Sequence[float]],
    lower_level: Iterable[Segment | Sequence[float]],
) -> float:
    pairs = hierarchical_overlap_pairs(upper_level, lower_level)
    if not pairs:
        return 0.0
    values = [balance_pair(round(a), round(b)) for a, b in pairs]
    return float(np.mean(values))


def hierarchical_regularity_absolute(
    upper_level: Iterable[Segment | Sequence[float]],
    lower_level: Iterable[Segment | Sequence[float]],
    frame_rate: float = 10.0,
    window_s: float = 0.5,
) -> float:
    pairs = hierarchical_overlap_pairs(upper_level, lower_level)
    if not pairs:
        return 0.0
    values = [regularity_pair_absolute(a, b, frame_rate=frame_rate, window_s=window_s) for a, b in pairs]
    return float(np.mean(values))


def hierarchical_balance_absolute(
    upper_level: Iterable[Segment | Sequence[float]],
    lower_level: Iterable[Segment | Sequence[float]],
    frame_rate: float = 10.0,
    window_s: float = 0.5,
) -> float:
    pairs = hierarchical_overlap_pairs(upper_level, lower_level)
    if not pairs:
        return 0.0
    values = [balance_pair_absolute(a, b, frame_rate=frame_rate, window_s=window_s) for a, b in pairs]
    return float(np.mean(values))


def segmentation_regularity_metrics(
    segments: Iterable[Segment | Sequence[float]],
    *,
    musical_durations: Sequence[int] | None = None,
    frame_rate: float = 10.0,
    window_s: float = 0.5,
) -> dict[str, float]:
    """Convenience wrapper returning the main regularity and balance metrics."""

    normalized = coerce_segments(segments)
    labels = [segment.label for segment in normalized]
    durations_s = [segment.duration for segment in normalized]
    result = {
        "regularity_absolute": regularity_absolute(durations_s, frame_rate=frame_rate, window_s=window_s),
        "balance_absolute": balance_absolute(durations_s, frame_rate=frame_rate, window_s=window_s),
        "labeled_regularity_absolute": labeled_regularity_absolute(
            durations_s,
            labels,
            frame_rate=frame_rate,
            window_s=window_s,
        ),
        "labeled_balance_absolute": labeled_balance_absolute(
            durations_s,
            labels,
            frame_rate=frame_rate,
            window_s=window_s,
        ),
        "adjacent_regularity_absolute": adjacent_regularity_absolute(
            durations_s,
            frame_rate=frame_rate,
            window_s=window_s,
        ),
        "adjacent_balance_absolute": adjacent_balance_absolute(
            durations_s,
            frame_rate=frame_rate,
            window_s=window_s,
        ),
    }
    if musical_durations is not None:
        durations_beats = [int(value) for value in musical_durations]
        if len(durations_beats) != len(normalized):
            raise ValueError("musical_durations must have the same length as segments.")
        result.update(
            {
                "regularity": regularity(durations_beats),
                "balance": balance(durations_beats),
                "labeled_regularity": labeled_regularity(durations_beats, labels),
                "labeled_balance": labeled_balance(durations_beats, labels),
                "adjacent_regularity": adjacent_regularity(durations_beats),
                "adjacent_balance": adjacent_balance(durations_beats),
            }
        )
    return result

def segmentation_boundary_metrics(
    reference_segments: Iterable[Segment | Sequence[float]],
    estimated_segments: Iterable[Segment | Sequence[float]],
    *,
    boundary_windows: Sequence[float] = (0.5, 3.0),
    trim: bool = True,
) -> dict[str, float]:
    """
    Compute boundary-based segmentation metrics from segment intervals.

    With ``trim=True`` (MSAF / MIREX convention) the first and last boundaries,
    i.e. the track start and end, are discarded before matching; both
    segmentations always contain them, so keeping them inflates precision and
    recall.
    """

    ref = coerce_segments(reference_segments)
    est = coerce_segments(estimated_segments)

    if not ref:
        raise ValueError("reference_segments is empty.")

    if not est:
        raise ValueError("estimated_segments is empty.")

    ref_intervals = np.asarray(
        [[seg.start, seg.end] for seg in ref],
        dtype=float,
    )

    est_intervals = np.asarray(
        [[seg.start, seg.end] for seg in est],
        dtype=float,
    )

    result = {}

    # ---------------------------------------------------------
    # Boundary detection
    # ---------------------------------------------------------

    for window in boundary_windows:

        precision, recall, f_measure = mir_eval.segment.detection(
            ref_intervals,
            est_intervals,
            window=float(window),
            trim=trim,
        )

        suffix = str(window).replace(".", "_")

        result.update(
            {
                f"boundary_precision_{suffix}s": float(precision),
                f"boundary_recall_{suffix}s": float(recall),
                f"boundary_f_{suffix}s": float(f_measure),
            }
        )

    # ---------------------------------------------------------
    # Boundary deviation
    # ---------------------------------------------------------

    if len(ref_intervals) > 0 and len(est_intervals) > 0:

        dev_ref_to_est, dev_est_to_ref = mir_eval.segment.deviation(
            ref_intervals,
            est_intervals,
            trim=trim,
        )

        result.update(
            {
                "median_deviation_ref_to_est_s": float(dev_ref_to_est),
                "median_deviation_est_to_ref_s": float(dev_est_to_ref),
            }
        )
    else:

        result.update(
            {
                "median_deviation_ref_to_est_s": np.nan,
                "median_deviation_est_to_ref_s": np.nan,
            }
        )

    return result


def boundary_hit_vector(
    reference_segments: Iterable[Segment | Sequence[float]],
    estimated_segments: Iterable[Segment | Sequence[float]],
    *,
    window: float = 0.5,
    trim: bool = True,
) -> np.ndarray:
    """
    Binary hit/miss vector over the reference boundaries.

    Entry i is 1 if reference boundary i is matched by an estimated boundary
    within ``window`` seconds, else 0. Uses the same one-to-one matching and
    ``trim`` convention as ``mir_eval.segment.detection``, so ``hits.mean()``
    equals the boundary recall at that window.
    """

    ref = coerce_segments(reference_segments)
    est = coerce_segments(estimated_segments)

    if not ref:
        raise ValueError("reference_segments is empty.")

    ref_boundaries = mir_eval.util.intervals_to_boundaries(
        np.asarray([[seg.start, seg.end] for seg in ref], dtype=float)
    )
    if trim:
        ref_boundaries = ref_boundaries[1:-1]
    hits = np.zeros(len(ref_boundaries), dtype=int)

    if est:
        est_boundaries = mir_eval.util.intervals_to_boundaries(
            np.asarray([[seg.start, seg.end] for seg in est], dtype=float)
        )
        if trim:
            est_boundaries = est_boundaries[1:-1]
        for ref_idx, _ in mir_eval.util.match_events(ref_boundaries, est_boundaries, float(window)):
            hits[ref_idx] = 1

    return hits

__all__ = [
    "Path",
    "Segment",
    "RegularityPairStats",
    "make_cbm_kernel",
    "cbm_normalization",
    "cbm_homogeneity_score",
    "cbm_penalty",
    "cbm_segment_score",
    "coerce_segments",
    "durations_from_boundaries",
    "durations_from_segments",
    "labels_from_segments",
    "regularity_pair",
    "balance_pair",
    "regularity",
    "balance",
    "labeled_regularity",
    "labeled_balance",
    "adjacent_regularity",
    "adjacent_balance",
    "regularity_pair_absolute",
    "balance_pair_absolute",
    "regularity_absolute",
    "balance_absolute",
    "labeled_regularity_absolute",
    "labeled_balance_absolute",
    "adjacent_regularity_absolute",
    "adjacent_balance_absolute",
    "hierarchical_overlap_pairs",
    "hierarchical_regularity",
    "hierarchical_balance",
    "hierarchical_regularity_absolute",
    "hierarchical_balance_absolute",
    "segmentation_regularity_metrics",
    "segmentation_boundary_metrics",
    "boundary_hit_vector",
]
