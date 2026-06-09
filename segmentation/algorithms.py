"""Music segmentation algorithms and the dispatch registry.

Two families of algorithm live here, both returning a sorted list of boundary
frame indices that index the beat-synchronous grid (and always include ``0`` and
``n_frames``):

* **SSM-based** (``cbm``, ``foote``, ``sf``) consume a precomputed, normalized
  self-similarity matrix (see :func:`segmentation.ssm.normalize_ssm`).
* **Feature-based** (``scluster`` Laplacian spectral clustering) ignore the SSM
  and instead read beat-synchronous feature arrays from the loaded ``.npz``.

:func:`run_algorithm` receives both the SSM and the loaded feature mapping so it
can dispatch to either family transparently.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .loaders import (
    load_as_seg_cbm_module,
    load_msaf_cbm_module,
    load_msaf_foote_module,
    load_msaf_scluster_module,
    load_msaf_sf_module,
)

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


def cbm_boundaries(matrix: np.ndarray, algorithm: dict[str, Any]) -> list[int]:
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
    boundaries = [0, matrix.shape[0]]
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
    structural = sf.gaussian_filter(lag.T, M=m_gaussian, axis=1)
    structural = sf.gaussian_filter(lag.T, M=1, axis=0)
    novelty_curve = sf.compute_nc(structural)
    peaks = sf.pick_peaks(novelty_curve, L=mp_adaptive, offset_denom=offset_thres)
    return [0, *map(int, peaks), matrix.shape[0]]

def scluster_boundaries(
    matrix: np.ndarray, algorithm: dict[str, Any], features: Mapping[str, Any]
) -> list[int]:
    """Laplacian spectral-clustering segmentation (McFee & Ellis, 2014).

    This algorithm does not use the SSM; it builds its own recurrence and path
    affinities from two beat-synchronous feature streams (chroma/PCP + MFCC),
    computes the normalized Laplacian, and clusters its spectral embedding.
    Feature keys and clustering parameters are taken from the config, with
    defaults mirroring ``msaf/.../scluster/config.py``.
    """
    scluster = load_msaf_scluster_module()
    params = dict(algorithm.get("params", {}))

    config = {
    "rep_feature": str(params.get("rep_feature", "cens")),
    "loc_feature": str(params.get("loc_feature", "mfcc")),
    "scluster_k": int(params.get("scluster_k", 4)),
    "num_layers": int(params.get("num_layers", 10)),
    "evec_smooth": int(params.get("evec_smooth", 9)),
    "rec_smooth": int(params.get("rec_smooth", 9)),
    "rec_width": int(params.get("rec_width", 9)),
    "hier": False,
}
    n_frames = matrix.shape[0]
    harmonic = _select_feature(features, config["rep_feature"], n_frames)
    timbral = _select_feature(features, config["loc_feature"], n_frames)


    est_idxs, _est_labels, _cnorm = scluster.do_segmentation(harmonic, timbral, config, None)
    return [0, *map(int, np.asarray(est_idxs)), n_frames]

def _select_feature(features: Mapping[str, Any], requested: str, n_frames: int) -> np.ndarray:
    """Return a ``(n_dims, n_frames)`` feature array from the loaded ``.npz``.

    Prefers the requested key as-is, then its ``beat_`` (beat-synchronous)
    variant -- the beat grid is what the SSMs and boundary indices are aligned
    to. Orients the array so the frame axis matches ``n_frames``.
    """
    candidates = [requested, f"beat_{requested}"] if not requested.startswith("beat_") else [requested]
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
    if array.shape[1] != n_frames and array.shape[0] == n_frames:
        array = array.T
    return array

def run_algorithm(
    matrix: np.ndarray, algorithm: dict[str, Any], features: Mapping[str, Any] | None = None
) -> list[int]:
    algorithm_id = str(algorithm["id"]).lower()
    if algorithm_id in {"cbm", "as_seg_cbm", "msaf_cbm"}:
        if algorithm_id == "msaf_cbm" and "source" not in algorithm:
            algorithm = {**algorithm, "source": "msaf"}
        return cbm_boundaries(matrix, algorithm)
    if algorithm_id in {"foote", "msaf_foote"}:
        return foote_boundaries(matrix, algorithm)
    if algorithm_id in {"sf", "msaf_sf"}:
        return sf_boundaries(matrix, algorithm)
    if algorithm_id in {"scluster", "msaf_scluster"}:
        if features is None:
            raise ValueError(f"Algorithm {algorithm['id']!r} requires the loaded feature arrays.")
        return scluster_boundaries(matrix, algorithm, features)
    raise ValueError(
        f"Unsupported segmentation algorithm {algorithm['id']!r}. "
        f"Supported ids: {', '.join(SUPPORTED_ALGORITHMS)}."
    )
