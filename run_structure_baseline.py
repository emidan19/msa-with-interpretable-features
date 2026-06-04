#!/usr/bin/env python3
"""Run SSM-based music segmentation baselines over a folder of songs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

_CACHE_ROOT = os.path.join(os.getcwd(), ".cache")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE_ROOT, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_CACHE_ROOT, "xdg"))
for _cache_dir in (os.environ["MPLCONFIGDIR"], os.environ["XDG_CACHE_HOME"]):
    os.makedirs(_cache_dir, exist_ok=True)

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parent
MSA_ROOT = REPO_ROOT.parent
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif")
SSM_FEATURE_TO_KEY = {
    "stm": "ssm_stm",
    "mfcc": "ssm_mfcc",
    "chroma": "ssm_chroma",
    "cens": "ssm_cens",
    "arrangement": "ssm_arrangement",
    "vocal": "ssm_vocal",
    "bass": "ssm_bass",
    "tonnetz": "ssm_tonnetz",
    "density": "ssm_density",
    "musicfm": "ssm_musicfm",
    "muq": "ssm_muq",
    "matpac": "ssm_matpac",
    "fused": "ssm_fused",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Folder of audio files for online extraction.")
    parser.add_argument("--feature-dir", type=Path, default=None, help="Folder containing *_features.npz files.")
    parser.add_argument("--experiment-dir", type=Path, default=None, help="Folder where JAMS outputs are written.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument(
        "--beat-tracking-method",
        choices=["librosa", "beat_this", "madmom"],
        default=None,
    )
    return parser.parse_args(argv)


def add_repo_to_path(path: Path) -> None:
    path = path.resolve()
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


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


def load_msaf_cbm_module():
    module_name = "_local_msaf_cbm_algorithm"
    if module_name in sys.modules:
        return sys.modules[module_name]
    return load_module_from_path(
        module_name,
        MSA_ROOT / "msaf" / "msaf" / "algorithms" / "cbm" / "CBM_algorithm.py",
    )


def normalize_ssm(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if matrix.size == 0:
        return matrix
    matrix = 0.5 * (matrix + matrix.T)
    lo, hi = np.percentile(matrix, [5, 95])
    if hi > lo:
        matrix = np.clip((matrix - lo) / (hi - lo), 0.0, 1.0)
    else:
        matrix = np.zeros_like(matrix)
    np.fill_diagonal(matrix, 1.0)
    return matrix.astype(np.float32)


def parse_segmentation_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    cfg = raw_config.get("segmentation", {})
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("`segmentation` must be a mapping in the YAML config.")
    return cfg


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
            algorithms.append(dict(item))
        else:
            raise ValueError(f"Invalid segmentation algorithm entry: {item!r}")
    return algorithms


def list_audio_files(audio_dir: Path, limit: int | None) -> list[Path]:
    audio_paths = sorted(path for path in audio_dir.iterdir() if path.suffix.lower() in AUDIO_EXTENSIONS)
    return audio_paths[:limit] if limit is not None else audio_paths


def list_feature_files(feature_dir: Path, limit: int | None) -> list[Path]:
    feature_paths = sorted(feature_dir.glob("*_features.npz"))
    return feature_paths[:limit] if limit is not None else feature_paths


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config at {path} must contain a mapping at the top level.")
    return payload


def build_feature_config(args: argparse.Namespace, raw_config: dict[str, Any]):
    from dataclasses import asdict

    from data_loading import FeatureConfig, config_from_mapping, resolve_value

    yaml_cfg = config_from_mapping(raw_config.get("feature_config", raw_config))
    return FeatureConfig(
        **{
            **asdict(yaml_cfg),
            "sr": resolve_value(args.sr, yaml_cfg.sr, FeatureConfig.sr),
            "hop_length": resolve_value(args.hop_length, yaml_cfg.hop_length, FeatureConfig.hop_length),
            "beat_tracking_method": resolve_value(
                args.beat_tracking_method,
                yaml_cfg.beat_tracking_method,
                FeatureConfig.beat_tracking_method,
            ),
        }
    )


def find_feature_npz(feature_dir: Path, audio_path: Path) -> Path | None:
    candidates = [
        feature_dir / f"{audio_path.stem}_features.npz",
        feature_dir / f"{audio_path.name}_features.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_features(audio_path: Path, feature_dir: Path, cfg: FeatureConfig) -> Path:
    existing = find_feature_npz(feature_dir, audio_path)
    if existing is not None:
        return existing
    from pipeline import extract_all

    summary = extract_all(audio_path, feature_dir, cfg)
    return Path(summary["outputs"]["features_npz"])


def load_ssm(data: np.lib.npyio.NpzFile, feature: str) -> np.ndarray:
    key = SSM_FEATURE_TO_KEY.get(feature.lower(), f"ssm_{feature.lower()}")
    if key not in data:
        available = sorted(name[4:] for name in data.files if name.startswith("ssm_"))
        raise ValueError(f"SSM feature {feature!r} not found. Available SSMs: {', '.join(available)}")
    return normalize_ssm(np.asarray(data[key]))


def fuse_ssms_with_graphditty(matrices: list[np.ndarray], fusion_cfg: dict[str, Any]) -> np.ndarray:
    add_repo_to_path(MSA_ROOT / "GraphDitty")
    from SimilarityFusion import doSimilarityFusionWs

    if len(matrices) == 1:
        return normalize_ssm(matrices[0])
    shapes = {matrix.shape for matrix in matrices}
    if len(shapes) != 1:
        raise ValueError(f"Cannot fuse SSMs with different shapes: {sorted(shapes)}")
    k_nearest = int(fusion_cfg.get("k", fusion_cfg.get("K", 5)))
    if k_nearest < 0:
        k_nearest = max(1, int(np.sqrt(matrices[0].shape[0])))
    fused = doSimilarityFusionWs(
        [normalize_ssm(matrix) for matrix in matrices],
        K=k_nearest,
        niters=int(fusion_cfg.get("niters", 20)),
        reg_diag=float(fusion_cfg.get("reg_diag", 1.0)),
        reg_neighbs=float(fusion_cfg.get("reg_neighbs", 0.5)),
        do_animation=False,
        verboseTimes=bool(fusion_cfg.get("verbose", False)),
    )
    return normalize_ssm(np.asarray(fused))


def resolve_segmentation_ssm(
    data: np.lib.npyio.NpzFile,
    ssm_features: Iterable[str],
    fusion_cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    features = [str(feature).strip().lower() for feature in ssm_features if str(feature).strip()]
    if not features:
        raise ValueError("At least one `segmentation.ssm_features` value is required.")
    matrices = [load_ssm(data, feature) for feature in features]
    matrix = matrices[0] if len(matrices) == 1 else fuse_ssms_with_graphditty(matrices, fusion_cfg)
    return normalize_ssm(matrix), features


def frame_index_to_times(indices: Iterable[int], n_frames: int, beat_boundaries: np.ndarray, duration: float) -> list[float]:
    idxs = sorted({int(idx) for idx in indices if int(idx) >= 0})
    normalized = []
    for idx in idxs:
        normalized.append(n_frames if idx >= n_frames - 1 else idx)
    normalized = sorted({0, n_frames, *normalized})
    if beat_boundaries.size >= n_frames + 1:
        times = [float(beat_boundaries[min(idx, beat_boundaries.size - 1)]) for idx in normalized]
    else:
        times = [float(duration) * idx / max(1, n_frames) for idx in normalized]
    times[0] = 0.0
    times[-1] = float(duration)
    return sorted({round(time_value, 6) for time_value in times})


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


def run_algorithm(matrix: np.ndarray, algorithm: dict[str, Any]) -> list[int]:
    algorithm_id = str(algorithm["id"]).lower()
    if algorithm_id in {"cbm", "as_seg_cbm", "msaf_cbm"}:
        if algorithm_id == "msaf_cbm" and "source" not in algorithm:
            algorithm = {**algorithm, "source": "msaf"}
        return cbm_boundaries(matrix, algorithm)
    if algorithm_id in {"foote", "msaf_foote"}:
        return foote_boundaries(matrix, algorithm)
    raise ValueError(
        f"Unsupported SSM-based segmentation algorithm {algorithm['id']!r}. "
        "Supported ids: cbm, as_seg_cbm, msaf_cbm, foote, msaf_foote."
    )


def annotation_intervals(boundary_times: list[float]) -> list[dict[str, Any]]:
    intervals = []
    for idx, (start, end) in enumerate(zip(boundary_times[:-1], boundary_times[1:])):
        if end <= start:
            continue
        intervals.append(
            {
                "time": float(start),
                "duration": float(end - start),
                "value": {"label": f"S{idx + 1:02d}"},
                "confidence": None,
            }
        )
    return intervals


def write_jams(
    output_path: Path,
    audio_path: str,
    duration: float,
    algorithm: dict[str, Any],
    ssm_features: list[str],
    boundary_times: list[float],
) -> None:
    annotation = {
        "namespace": "segment_open",
        "time": 0,
        "duration": float(duration),
        "annotation_metadata": {
            "data_source": "run_structure_baseline.py",
            "annotation_tools": f"{algorithm.get('source', 'as_seg')}:{algorithm['id']}",
        },
        "data": annotation_intervals(boundary_times),
        "sandbox": {
            "boundary_times": boundary_times,
            "ssm_features": ssm_features,
            "algorithm": algorithm,
        },
    }
    jams_payload = {
        "file_metadata": {
            "title": Path(audio_path).stem,
            "duration": float(duration),
        },
        "annotations": [annotation],
        "sandbox": {
            "audio_path": audio_path,
            "task": "music_structure_segmentation",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jams_payload, indent=2), encoding="utf-8")


def algorithm_slug(algorithm: dict[str, Any]) -> str:
    source = str(algorithm.get("source", "as_seg" if str(algorithm["id"]).lower() == "cbm" else "msaf"))
    algorithm_id = str(algorithm["id"])
    slug = algorithm_id if algorithm_id.startswith(f"{source}_") else f"{source}_{algorithm_id}"
    return slug.replace("/", "_").replace(" ", "_")


def process_feature_file(
    feature_npz: Path,
    experiment_dir: Path,
    algorithms: list[dict[str, Any]],
    ssm_features: list[str],
    fusion_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    with np.load(feature_npz, allow_pickle=True) as data:
        matrix, resolved_features = resolve_segmentation_ssm(data, ssm_features, fusion_cfg)
        n_frames = matrix.shape[0]
        duration = float(np.asarray(data["duration"]).item()) if "duration" in data else float(n_frames)
        beat_boundaries = np.asarray(data["beat_boundaries"]) if "beat_boundaries" in data else np.array([])
        audio_path = str(np.asarray(data["audio_path"]).item()) if "audio_path" in data else feature_npz.stem
        track_id = Path(audio_path).stem
        for algorithm in algorithms:
            start = time.perf_counter()
            slug = algorithm_slug(algorithm)
            boundary_indices = run_algorithm(matrix, algorithm)
            boundary_times = frame_index_to_times(boundary_indices, n_frames, beat_boundaries, duration)
            jams_path = experiment_dir / slug / f"{track_id}.jams"
            write_jams(jams_path, audio_path, duration, algorithm, resolved_features, boundary_times)
            rows.append(
                {
                    "track_id": track_id,
                    "algorithm": slug,
                    "status": "ok",
                    "duration_s": round(duration, 6),
                    "n_ssm_frames": int(n_frames),
                    "n_boundaries": len(boundary_times),
                    "elapsed_s": round(time.perf_counter() - start, 3),
                    "features_npz": str(feature_npz),
                    "jams": str(jams_path),
                }
            )
    return rows


def write_summary(experiment_dir: Path, rows: list[dict[str, Any]]) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "segmentation_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with (experiment_dir / "segmentation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    raw_config = load_config(args.config)
    segmentation_cfg = parse_segmentation_config(raw_config)

    audio_dir = args.audio_dir or (Path(raw_config["audio_dir"]) if raw_config.get("audio_dir") else None)
    feature_dir = args.feature_dir or (
        Path(segmentation_cfg["feature_dir"]) if segmentation_cfg.get("feature_dir") else None
    )
    experiment_dir = args.experiment_dir or Path(
        segmentation_cfg.get("experiment_dir", raw_config.get("experiment_dir", "experiments/structure_baseline"))
    )
    limit = args.limit if args.limit is not None else raw_config.get("limit")
    limit = int(limit) if limit is not None else None
    algorithms = normalize_algorithms(segmentation_cfg.get("algorithms"))
    ssm_features = list(segmentation_cfg.get("ssm_features", ["fused"]))
    fusion_cfg = dict(segmentation_cfg.get("graphditty_fusion", {}))

    if feature_dir is None:
        if audio_dir is None:
            raise SystemExit("Set `segmentation.feature_dir` or `audio_dir` in the YAML config.")
        feature_cfg = build_feature_config(args, raw_config)
        feature_dir = Path(raw_config.get("out_dir", experiment_dir / "features"))
        audio_paths = list_audio_files(audio_dir, limit)
        if not audio_paths:
            raise SystemExit(f"No audio files found in {audio_dir}")
        feature_paths = [ensure_features(audio_path, feature_dir, feature_cfg) for audio_path in audio_paths]
    else:
        feature_paths = list_feature_files(feature_dir, limit)
        if not feature_paths:
            raise SystemExit(f"No *_features.npz files found in {feature_dir}")

    rows: list[dict[str, Any]] = []
    for idx, feature_npz in enumerate(feature_paths, start=1):
        print(f"[{idx:03d}/{len(feature_paths):03d}] {feature_npz.name}", flush=True)
        try:
            rows.extend(process_feature_file(feature_npz, experiment_dir, algorithms, ssm_features, fusion_cfg))
        except Exception as exc:
            rows.append({"features_npz": str(feature_npz), "status": "error", "error": repr(exc)})
            write_summary(experiment_dir, rows)
            raise
        write_summary(experiment_dir, rows)

    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
