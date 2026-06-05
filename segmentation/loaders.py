"""Loaders for external segmentation algorithm modules.

These helpers pull the standalone algorithm functions out of the vendored
``msaf`` and ``autosimilarity_segmentation`` repositories without importing the
full packages (which carry heavy, partially-incompatible dependencies). Each
loader slices out the pieces it needs, injects the required third-party modules,
and execs the source into a throwaway module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MSA_ROOT = REPO_ROOT.parent


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_as_seg_cbm_module():
    module_name = "_local_as_seg_cbm_algorithm"
    if module_name in sys.modules:
        return sys.modules[module_name]

    class ToDebugException(Exception):
        pass

    as_seg_pkg = sys.modules.setdefault("as_seg", ModuleType("as_seg"))
    model_pkg = sys.modules.setdefault("as_seg.model", ModuleType("as_seg.model"))
    errors_mod = sys.modules.setdefault("as_seg.model.errors", ModuleType("as_seg.model.errors"))
    errors_mod.ToDebugException = ToDebugException
    as_seg_pkg.model = model_pkg
    model_pkg.errors = errors_mod
    module = ModuleType(module_name)
    module.__file__ = str(MSA_ROOT / "autosimilarity_segmentation" / "as_seg" / "CBM_algorithm.py")
    sys.modules[module_name] = module
    source = Path(module.__file__).read_text(encoding="utf-8")
    source = source.split("# %% Scikit-learn class", 1)[0]
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def load_msaf_foote_module():
    module_name = "_local_msaf_foote_segmenter"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = ModuleType(module_name)
    module.__file__ = str(MSA_ROOT / "msaf" / "msaf" / "algorithms" / "foote" / "segmenter.py")
    sys.modules[module_name] = module
    source = Path(module.__file__).read_text(encoding="utf-8")
    source = source[source.index("def median_filter") : source.index("class Segmenter")]
    import scipy.ndimage as ndimage
    import scipy.signal as signal
    from scipy.spatial import distance

    module.__dict__.update({"np": np, "ndimage": ndimage, "signal": signal, "distance": distance})
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def load_msaf_sf_module():
    """Load Serrà's Structural Features helper functions from MSAF.

    Slices the standalone functions (``circular_shift``, ``gaussian_filter``,
    ``compute_nc``, ``pick_peaks``, ...) out of ``sf/segmenter.py`` before the
    ``Segmenter`` class, mirroring :func:`load_msaf_foote_module`.
    """
    module_name = "_local_msaf_sf_segmenter"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = ModuleType(module_name)
    module.__file__ = str(MSA_ROOT / "msaf" / "msaf" / "algorithms" / "sf" / "segmenter.py")
    sys.modules[module_name] = module
    source = Path(module.__file__).read_text(encoding="utf-8")
    source = source[source.index("def median_filter") : source.index("class Segmenter")]
    import librosa
    import scipy.ndimage as ndimage
    import scipy.signal as signal
    from scipy.spatial import distance

    module.__dict__.update(
        {"np": np, "ndimage": ndimage, "signal": signal, "distance": distance, "librosa": librosa}
    )
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def load_msaf_cbm_module():
    module_name = "_local_msaf_cbm_algorithm"
    if module_name in sys.modules:
        return sys.modules[module_name]
    return load_module_from_path(
        module_name,
        MSA_ROOT / "msaf" / "msaf" / "algorithms" / "cbm" / "CBM_algorithm.py",
    )


def load_msaf_scluster_module():
    """Load Brian McFee's Laplacian spectral-clustering segmentation from MSAF.

    ``scluster/main2.py`` only depends on stdlib, librosa, numpy, scipy, and
    sklearn (no ``msaf`` package import), so it loads directly from its path.
    The public entry point is ``do_segmentation(C, M, config, in_bound_idxs)``.
    """
    module_name = "_local_msaf_scluster_main2"
    if module_name in sys.modules:
        return sys.modules[module_name]
    return load_module_from_path(
        module_name,
        MSA_ROOT / "msaf" / "msaf" / "algorithms" / "scluster" / "main2.py",
    )
