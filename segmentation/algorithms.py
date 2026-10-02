"""Music segmentation algorithms and the dispatch registry.

Two families of algorithm live here, both returning a sorted list of boundary
frame indices that index the beat-synchronous grid (and always include ``0`` and
``n_frames``):

* **SSM-based** (``cbm``, ``foote``, ``sf``) consume a precomputed, normalized
  self-similarity matrix (see :func:`segmentation.ssm.normalize_ssm`). CBM is a
  bar-wise algorithm: by default (``params.grid: bar``) it runs on a bar-level
  SSM pooled from the beat SSM and its boundaries are mapped back to beats.
* **Feature-based** (``scluster`` Laplacian spectral clustering) ignore the SSM
  and instead read beat-synchronous feature arrays from the loaded ``.npz``.

:func:`run_algorithm` receives both the SSM and the loaded feature mapping so it
can dispatch to either family transparently.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import scipy

from .grid import beat_ssm_to_bar_ssm
from .loaders import (
    load_as_seg_cbm_module,
    load_msaf_cbm_module,
    load_msaf_foote_module,
    load_msaf_scluster_module,
    load_msaf_sf_module,
)
from .ssm import normalize_ssm

# Algorithm ids accepted by :func:`run_algorithm`, grouped by family.
SUPPORTED_ALGORITHMS = (
    "cbm",
    "as_seg_cbm",
    "msaf_cbm",
    "foote",
    "msaf_foote",
    "sf",
    "msaf_sf",
    "scluster",
    "msaf_scluster",
)


def normalize_algorithms(items: Iterable[Any] | None) -> list[dict[str, Any]]:
    if not items:
        return [{"id": "cbm", "source": "as_seg"}, {"id": "foote", "source": "msaf"}]
    algorithms = []
    for item in items:
        if isinstance(item, str):
            algorithms.append({"id": item})
        elif isinstance(item, dict):
            if "id" not in item:
                raise ValueError(f"Segmentation algorithm entry lacks an `id`: {item!r}")
            if "run" in item and not bool(item["run"]):
                continue
            algorithms.append(dict(item))
        else:
            raise ValueError(f"Invalid segmentation algorithm entry: {item!r}")
    return algorithms


def cbm_boundaries(
    matrix: np.ndarray, algorithm: dict[str, Any], bar_indices: np.ndarray | None = None
) -> list[int]:
    """Correlation Block-Matching (Marmoret et al., TISMIR 2023).

    CBM is designed for bar-wise SSMs: `min_size`, `max_size`, the `modulo8`
    penalty and the size-8 score normalization are all in bars. With
    ``params.grid: bar`` (default) the beat SSM is pooled into a bar SSM using
    `bar_indices` (bar starts as beat indices, see
    :func:`segmentation.grid.bar_beat_indices`), CBM runs there, and its bar
    boundaries are returned as beat indices. ``grid: beat`` runs CBM directly on
    the beat SSM (sizes are then in beats).
    """
    params = dict(algorithm.get("params", {}))
    grid = str(params.get("grid", "bar")).lower()
    if grid not in {"bar", "beat"}:
        raise ValueError(f"CBM `grid` must be 'bar' or 'beat', got {grid!r}.")
    if grid == "beat":
        return _cbm_on_ssm(matrix, algorithm)
    if bar_indices is None:
        raise ValueError("Bar-wise CBM (`grid: bar`) needs the bar grid (`bar_indices`).")
    bar_indices = np.asarray(bar_indices, dtype=int)
    bar_matrix = normalize_ssm(beat_ssm_to_bar_ssm(matrix, bar_indices))
    bar_bounds = _cbm_on_ssm(bar_matrix, algorithm)
    return [int(bar_indices[idx]) for idx in bar_bounds]


def _cbm_on_ssm(matrix: np.ndarray, algorithm: dict[str, Any]) -> list[int]:
    source = str(algorithm.get("source", "as_seg")).lower()
    if source == "msaf":
        cbm = load_msaf_cbm_module()
    else:
        cbm = load_as_seg_cbm_module()

    params = dict(algorithm.get("params", {}))
    cbm_kwargs = {
        "min_size": int(params.get("min_size", 1)),
        "max_size": int(params.get("max_size", 32)),
        "penalty_weight": float(params.get("penalty_weight", 1.0)),
        "penalty_func": str(params.get("penalty_func", "modulo8")),
        "bands_number": params.get("bands_number", None),
    }
    if source != "msaf":
        cbm_kwargs["compute_score_using_convolve"] = bool(params.get("compute_score_using_convolve", False))
    segments, _score = cbm.compute_cbm(matrix, **cbm_kwargs)
    boundaries = [0]
    boundaries.extend(int(end) for _start, end in segments)
    return boundaries


def foote_boundaries(matrix: np.ndarray, algorithm: dict[str, Any]) -> list[int]:
    foote = load_msaf_foote_module()
    params = dict(algorithm.get("params", {}))
    gaussian_size = int(params.get("M_gaussian", 66))
    if gaussian_size % 2 == 1:
        gaussian_size += 1
    kernel = foote.compute_gaussian_krnl(gaussian_size)
    novelty_curve = foote.compute_nc(matrix, kernel)
    peaks = foote.pick_peaks(novelty_curve, L=int(params.get("L_peaks", 64)))
    return [0, *map(int, peaks), matrix.shape[0]]


def sf_boundaries(matrix: np.ndarray, algorithm: dict[str, Any]) -> list[int]:
    """Serrà Structural Features novelty, run on a precomputed SSM.

    The original MSAF implementation builds a librosa recurrence matrix from
    embedded features; here the normalized SSM is fed in as that recurrence/lag
    matrix, then the structural-feature pipeline (circular shift -> gaussian
    filtering -> novelty -> adaptive peak picking) proceeds unchanged. Defaults
    mirror ``msaf/.../sf/config.py``.
    """
    sf = load_msaf_sf_module()
    params = dict(algorithm.get("params", {}))
    m_gaussian = int(params.get("M_gaussian", 27))
    mp_adaptive = int(params.get("Mp_adaptive", 28))
    offset_thres = float(params.get("offset_thres", 0.05))

    lag = sf.circular_shift(matrix.astype(np.float64, copy=False))
    # MSAF's gaussian_filter works in place, so its two consecutive calls on
    # `L.T` apply both filters; chain them explicitly here.
    structural = sf.gaussian_filter(sf.gaussian_filter(lag.T, M=m_gaussian, axis=1), M=1, axis=0)
    novelty_curve = sf.compute_nc(structural)
    peaks = sf.pick_peaks(novelty_curve, L=mp_adaptive, offset_denom=offset_thres)
    return [0, *map(int, peaks), matrix.shape[0]]

def scluster_boundaries(
    matrix: np.ndarray, algorithm: dict[str, Any], features: Mapping[str, Any], feature: str
) -> list[int]:
    return scluster_segmentation(matrix, algorithm, features, feature)[0]


def scluster_segmentation(
    matrix: np.ndarray, algorithm: dict[str, Any], features: Mapping[str, Any], feature: str
) -> tuple[list[int], list[str]]:
    """Laplacian spectral-clustering segmentation (McFee & Ellis, 2014).

    This algorithm does not use the SSM; it builds its own recurrence and path
    affinities from beat-synchronous features, computes the normalized
    Laplacian, and clusters its spectral embedding. A single ``feature`` (one
    of ``segmentation.ssm_features``) is used both as the repetition
    (harmonic) and the local (timbral) representation. Clustering parameters
    are taken from the config, with defaults mirroring
    ``msaf/.../scluster/config.py``.
    """
    scluster = load_msaf_scluster_module()
    params = dict(algorithm.get("params", {}))

    config = {
        "scluster_k": int(params.get("scluster_k", 4)),
        "num_layers": int(params.get("num_layers", 10)),
        "evec_smooth": int(params.get("evec_smooth", 9)),
        "rec_smooth": int(params.get("rec_smooth", 9)),
        "rec_width": int(params.get("rec_width", 9)),
        "hier": False,
    }
    n_frames = matrix.shape[0]
    feature_array = _select_feature(features, feature, n_frames)

    est_idxs, est_labels, _cnorm = scluster.do_segmentation(feature_array, feature_array, config, None)
    # MSAF closes the list with the last beat index (n_frames - 1) and gives one
    # cluster label per segment start. Replace the closing index with n_frames
    # and drop a 1-beat tail segment starting at n_frames - 1 (merged into the
    # previous one), keeping each surviving segment's label.
    label_by_start = {int(idx): f"C{int(label)}" for idx, label in zip(np.asarray(est_idxs)[:-1], est_labels)}
    starts = sorted(idx for idx in label_by_start if 0 <= idx < n_frames - 1)
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    labels = [label_by_start.get(start, label_by_start.get(0, "C0")) for start in starts]
    return [*starts, n_frames], labels

def _select_feature(features: Mapping[str, Any], requested: str, n_frames: int) -> np.ndarray:
    """Return a ``(n_dims, n_frames)`` feature array from the loaded ``.npz``.

    Prefers the ``beat_`` (beat-synchronous) variant, then the requested key
    as-is -- the beat grid is what the SSMs and boundary indices are aligned
    to (raw keys such as ``stm`` or ``mfcc`` live on their own time grids).
    Orients the array so the frame axis matches ``n_frames``.
    """
    candidates = [f"beat_{requested}", requested] if not requested.startswith("beat_") else [requested]
    key = next((name for name in candidates if name in features), None)
    if key is None:
        available = sorted(name for name in features if not name.startswith("ssm_"))
        raise ValueError(
            f"Feature {requested!r} not found in features file. Available: {', '.join(available)}"
        )
    array = np.asarray(features[key], dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Feature {key!r} must be 2-D, got shape {array.shape}.")
    # Orient as (n_dims, n_frames) to match the beat-synchronous SSM grid.
    if array.shape[1] == n_frames:
        pass
    elif array.shape[0] == n_frames:
        array = array.T
    else:
        # Feature arrays are stored as (n_dims, n_time): resample the time axis to n_frames.
        array = scipy.interpolate.interp1d(
            np.linspace(0, 1, num=array.shape[1]), array, axis=1, kind="nearest", fill_value="extrapolate"
        )(np.linspace(0, 1, num=n_frames))
    return array

def run_algorithm(
    matrix: np.ndarray,
    algorithm: dict[str, Any],
    features: Mapping[str, Any] | None = None,
    feature: str | None = None,
    return_labels: bool = False,
    bar_indices: np.ndarray | None = None,
) -> list[int] | tuple[list[int], list[str] | None]:
    """Run one algorithm and return its boundary indices.

    With ``return_labels=True`` returns ``(boundaries, labels)``, where
    ``labels`` has one entry per segment for algorithms that label segments
    (``scluster``) and is ``None`` for boundary-only algorithms. `bar_indices`
    (bar starts as beat indices) is used by bar-wise CBM.
    """
    labels: list[str] | None = None
    boundaries = _run_algorithm(matrix, algorithm, features, feature, bar_indices)
    if isinstance(boundaries, tuple):
        boundaries, labels = boundaries
    return (boundaries, labels) if return_labels else boundaries


def _run_algorithm(
    matrix: np.ndarray,
    algorithm: dict[str, Any],
    features: Mapping[str, Any] | None,
    feature: str | None,
    bar_indices: np.ndarray | None,
) -> list[int] | tuple[list[int], list[str]]:
    algorithm_id = str(algorithm["id"]).lower()
    if algorithm_id in {"cbm", "as_seg_cbm", "msaf_cbm"}:
        if algorithm_id == "msaf_cbm" and "source" not in algorithm:
            algorithm = {**algorithm, "source": "msaf"}
        return cbm_boundaries(matrix, algorithm, bar_indices)
    if algorithm_id in {"foote", "msaf_foote"}:
        return foote_boundaries(matrix, algorithm)
    if algorithm_id in {"sf", "msaf_sf"}:
        return sf_boundaries(matrix, algorithm)
    if algorithm_id in {"scluster", "msaf_scluster"}:
        if features is None or feature is None:
            raise ValueError(
                f"Algorithm {algorithm['id']!r} requires the loaded feature arrays and a single feature name."
            )
        return scluster_segmentation(matrix, algorithm, features, feature)
    raise ValueError(
        f"Unsupported segmentation algorithm {algorithm['id']!r}. "
        f"Supported ids: {', '.join(SUPPORTED_ALGORITHMS)}."
    )
