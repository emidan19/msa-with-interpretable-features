"""Modular SSM-based music segmentation utilities for the structure baseline.

This package separates the three concerns that previously lived in
``run_structure_baseline.py``:

* :mod:`segmentation.ssm` -- loading, normalizing, and fusing self-similarity matrices.
* :mod:`segmentation.loaders` -- importing external algorithm code from the
  vendored ``msaf`` and ``autosimilarity_segmentation`` repositories.
* :mod:`segmentation.algorithms` -- the per-algorithm wrappers and the
  :func:`run_algorithm` dispatch.

All supported algorithms operate on a precomputed SSM and return boundary frame
indices.
"""

from __future__ import annotations

from .algorithms import (
    SUPPORTED_ALGORITHMS,
    cbm_boundaries,
    foote_boundaries,
    normalize_algorithms,
    run_algorithm,
    sf_boundaries,
)
from .ssm import (
    SSM_FEATURE_TO_KEY,
    add_repo_to_path,
    fuse_ssms_with_graphditty,
    load_ssm,
    normalize_ssm,
    resolve_segmentation_ssm,
)

__all__ = [
    "SUPPORTED_ALGORITHMS",
    "SSM_FEATURE_TO_KEY",
    "add_repo_to_path",
    "cbm_boundaries",
    "foote_boundaries",
    "fuse_ssms_with_graphditty",
    "load_ssm",
    "normalize_algorithms",
    "normalize_ssm",
    "resolve_segmentation_ssm",
    "run_algorithm",
    "sf_boundaries",
]
