"""Bar grid helpers on top of the beat-synchronous grid.

The SSMs and every segmentation index live on the beat grid (SSM row i is the
interval ``[beat_boundaries[i], beat_boundaries[i + 1])``). Bars are expressed
as beat indices: ``bar_indices[k]`` is the beat where bar ``k`` starts, with
``bar_indices[0] == 0`` and ``bar_indices[-1] == n_beats``. This lets
bar-wise algorithms (CBM) run on a bar-level SSM and map their boundaries back
to the beat grid without touching the rest of the pipeline.
"""

from __future__ import annotations

import warnings

import numpy as np


def snap_times_to_beat_indices(times: np.ndarray, beat_boundaries: np.ndarray) -> np.ndarray:
    """Index of the nearest beat boundary for each time."""
    times = np.asarray(times, dtype=np.float64)
    beats = np.asarray(beat_boundaries, dtype=np.float64)
    if times.size == 0 or beats.size == 0:
        return np.zeros(0, dtype=int)
    right = np.clip(np.searchsorted(beats, times), 1, beats.size - 1)
    left = right - 1
    nearest = np.where(np.abs(times - beats[left]) <= np.abs(beats[right] - times), left, right)
    return nearest.astype(int)


def enforce_min_bar_length(indices: np.ndarray, n_beats: int, min_beats: int = 2) -> np.ndarray:
    """Sorted unique bar starts with 0 and n_beats, dropping bars shorter than `min_beats` beats."""
    idxs = sorted({0, int(n_beats), *(int(i) for i in indices if 0 < int(i) < n_beats)})
    kept = [idxs[0]]
    for idx in idxs[1:-1]:
        if idx - kept[-1] >= min_beats:
            kept.append(idx)
    # Merge a too-short final bar into the previous one.
    if len(kept) > 1 and n_beats - kept[-1] < min_beats:
        kept.pop()
    kept.append(int(n_beats))
    return np.asarray(kept, dtype=int)


def downbeats_on_beat_grid(
    downbeat_times: np.ndarray, beat_boundaries: np.ndarray, min_beats: int = 2
) -> np.ndarray:
    """Snap downbeat times onto the beat grid; returns beat indices of the bar starts."""
    n_beats = int(np.asarray(beat_boundaries).size) - 1
    snapped = snap_times_to_beat_indices(downbeat_times, beat_boundaries)
    return enforce_min_bar_length(snapped, n_beats, min_beats=min_beats)


def bar_beat_indices(
    beat_boundaries: np.ndarray,
    downbeat_boundaries: np.ndarray | None,
    *,
    min_beats: int = 2,
    fallback_beats_per_bar: int = 4,
    label: str = "",
) -> tuple[np.ndarray, str]:
    """Bar starts as beat indices, plus where they came from (`downbeats` or `fixed_<n>`).

    Falls back to a fixed number of beats per bar when there are no downbeats,
    or when they look corrupted (on average fewer than `min_beats` beats per
    bar, e.g. downbeats regularized at beat rate by older extraction code).
    """
    beat_boundaries = np.asarray(beat_boundaries)
    n_beats = int(beat_boundaries.size) - 1
    if n_beats <= 0:
        raise ValueError("A beat grid is required to build the bar grid.")
    downbeats = np.asarray(downbeat_boundaries if downbeat_boundaries is not None else [], dtype=np.float64)
    prefix = f"{label}: " if label else ""
    if downbeats.size >= 2:
        n_bars_raw = downbeats.size - 1
        if n_bars_raw * min_beats <= n_beats:
            return downbeats_on_beat_grid(downbeats, beat_boundaries, min_beats=min_beats), "downbeats"
        warnings.warn(
            f"{prefix}{n_bars_raw} downbeat intervals for {n_beats} beats look corrupted; "
            f"using a fixed {fallback_beats_per_bar}-beat bar grid. Recompute them with recompute_downbeats.py.",
            stacklevel=2,
        )
    else:
        warnings.warn(
            f"{prefix}no downbeats available; using a fixed {fallback_beats_per_bar}-beat bar grid.",
            stacklevel=2,
        )
    fixed = np.arange(0, n_beats, fallback_beats_per_bar)
    return enforce_min_bar_length(fixed, n_beats, min_beats=min_beats), f"fixed_{fallback_beats_per_bar}"


def beat_ssm_to_bar_ssm(matrix: np.ndarray, bar_indices: np.ndarray) -> np.ndarray:
    """Bar-level SSM: entry (i, j) is the mean of the beat-SSM block of bars i and j."""
    matrix = np.asarray(matrix, dtype=np.float64)
    starts = np.asarray(bar_indices[:-1], dtype=int)
    sizes = np.diff(np.asarray(bar_indices, dtype=int)).astype(np.float64)
    if starts.size == 0 or np.any(sizes <= 0) or int(bar_indices[-1]) != matrix.shape[0]:
        raise ValueError(
            f"Bar grid (last index {int(bar_indices[-1])}) does not cover an SSM of {matrix.shape[0]} beats."
        )
    sums = np.add.reduceat(np.add.reduceat(matrix, starts, axis=0), starts, axis=1)
    return (sums / np.outer(sizes, sizes)).astype(np.float32)
