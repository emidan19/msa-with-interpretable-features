#!/usr/bin/env python3
"""Run SSM-based music segmentation baselines over a folder of songs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

_CACHE_ROOT = os.path.join(os.getcwd(), ".cache")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE_ROOT, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_CACHE_ROOT, "xdg"))
for _cache_dir in (os.environ["MPLCONFIGDIR"], os.environ["XDG_CACHE_HOME"]):
    os.makedirs(_cache_dir, exist_ok=True)

import numpy as np
import yaml

from msa_metrics import Segment, cbm_segment_score, segmentation_regularity_metrics, segmentation_boundary_metrics
from segmentation import build_ssm_runs, normalize_algorithms, resolve_segmentation_ssm, run_algorithm


REPO_ROOT = Path(__file__).resolve().parent
MSA_ROOT = REPO_ROOT.parent
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default="/home/emartinez/emilio/maestria/tesis/msa/MSA-with-Interpretable-Features/msa_feature_config.template.yaml", help="YAML config path.")
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


def parse_segmentation_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    cfg = raw_config.get("segmentation", {})
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("`segmentation` must be a mapping in the YAML config.")
    return cfg


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


def _npz_missing_expected_sections(npz_path: Path, audio_path: Path, cfg: FeatureConfig) -> bool:
    """True if `cfg` expects section annotations that `npz_path` doesn't have yet."""
    if cfg.sections_dir is None:
        return False
    with np.load(npz_path, allow_pickle=True) as data:
        if "sections" in data:
            return False
        duration = float(np.asarray(data["duration"]).item()) if "duration" in data else None

    from data_loading import load_sections

    expected = load_sections(cfg.sections_dir, audio_path.stem, cfg.label_format, duration)
    return bool(expected)


def ensure_features(audio_path: Path, feature_dir: Path, cfg: FeatureConfig) -> Path:
    existing = find_feature_npz(feature_dir, audio_path)
    if existing is not None and not _npz_missing_expected_sections(existing, audio_path, cfg):
        return existing
    from pipeline import extract_all

    summary = extract_all(audio_path, feature_dir, cfg)
    return Path(summary["outputs"]["features_npz"])


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


def annotation_intervals(boundary_times: list[float], segment_metrics: list[dict[str, float]] | None = None) -> list[dict[str, Any]]:
    intervals = []
    for idx, (start, end) in enumerate(zip(boundary_times[:-1], boundary_times[1:])):
        if end <= start:
            continue
        interval = {
            "time": float(start),
            "duration": float(end - start),
            "value": {"label": f"S{idx + 1:02d}"},
            "confidence": None,
        }
        if segment_metrics is not None and idx < len(segment_metrics):
            interval["metrics"] = {k: float(v) for k, v in segment_metrics[idx].items()}
        intervals.append(interval)
    return intervals


def reference_annotation_intervals(reference_segments: list[Segment]) -> list[dict[str, Any]]:
    return [
        {
            "time": float(segment.start),
            "duration": float(segment.duration),
            "value": {"label": segment.label or ""},
            "confidence": None,
        }
        for segment in reference_segments
        if segment.end > segment.start
    ]


def write_jams(
    output_path: Path,
    audio_path: str,
    duration: float,
    algorithm: dict[str, Any],
    ssm_features: list[str],
    boundary_times: list[float],
    metrics: dict[str, float],
    segment_metrics: list[dict[str, float]] | None = None,
    reference_segments: list[Segment] | None = None,
) -> None:
    annotation = {
        "namespace": "segment_open",
        "time": 0,
        "duration": float(duration),
        "annotation_metadata": {
            "data_source": "run_structure_baseline.py",
            "annotation_tools": f"{algorithm.get('source', 'as_seg')}:{algorithm['id']}",
        },
        "data": annotation_intervals(boundary_times, segment_metrics),
        "sandbox": {
            "kind": "estimate",
            "boundary_times": boundary_times,
            "ssm_features": ssm_features,
            "algorithm": algorithm,
            "metrics": metrics,
            "segment_metrics": segment_metrics or [],
        },
    }
    annotations = [annotation]
    # Also embed the ground-truth segmentation (when available) as a second
    # annotation, so downstream evaluation (see evaluate_structure_baseline.py)
    # can compare against it without needing the original sections files.
    reference_intervals = reference_annotation_intervals(reference_segments or [])
    if reference_intervals:
        annotations.append(
            {
                "namespace": "segment_open",
                "time": 0,
                "duration": float(duration),
                "annotation_metadata": {
                    "data_source": "run_structure_baseline.py",
                    "annotation_tools": "ground_truth",
                },
                "data": reference_intervals,
                "sandbox": {"kind": "reference"},
            }
        )
    jams_payload = {
        "file_metadata": {
            "title": Path(audio_path).stem,
            "duration": float(duration),
        },
        "annotations": annotations,
        "sandbox": {
            "audio_path": audio_path,
            "task": "music_structure_segmentation",
            "metrics": metrics,
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
    ssm_runs: list[tuple[str, list[str]]],
    fusion_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    with np.load(feature_npz, allow_pickle=True) as data:
        base_duration = float(np.asarray(data["duration"]).item()) if "duration" in data else None
        sections = np.asarray(data["sections"]) if "sections" in data else None
        beat_boundaries = np.asarray(data["beat_boundaries"]) if "beat_boundaries" in data else np.array([])
        audio_path = str(np.asarray(data["audio_path"]).item()) if "audio_path" in data else feature_npz.stem
        track_id = Path(audio_path).stem

        # Reference segments (if available); independent of the SSM run.
        ref_segments = [
            Segment(start=float(start), end=float(end), label=f"{idx} : {label}")
            for idx, (start, end, label) in enumerate(sections if sections is not None else [])
        ]

        for ssm_label, ssm_feature_list in ssm_runs:
            matrix, resolved_features = resolve_segmentation_ssm(data, ssm_feature_list, fusion_cfg)
            n_frames = matrix.shape[0]
            duration = base_duration if base_duration is not None else float(n_frames)
            # If this run fused multiple SSMs with GraphDitty, save the fused
            # SSM to the experiment folder so it can be inspected later, under
            # <experiment_dir>/fusion_ssms/<track_id>__<ssm_label>_fused_ssm.npz
            # with key "ssm_fused".
            if len(resolved_features) > 1:
                try:
                    fusion_dir = experiment_dir / "fusion_ssms"
                    fusion_dir.mkdir(parents=True, exist_ok=True)
                    fusion_path = fusion_dir / f"{track_id}__{ssm_label}_fused_ssm.npz"
                    np.savez_compressed(
                        fusion_path,
                        ssm_fused=matrix,
                        ssm_features=np.asarray(resolved_features, dtype=object),
                        audio_path=audio_path,
                    )
                except Exception:
                    # Do not fail the whole run if saving the fused SSM fails.
                    pass
            for algorithm in algorithms:
                # Get algo name, suffixed with the SSM run so each feature /
                # fusion combo writes to its own subdirectory.
                slug = algorithm_slug(algorithm)
                run_slug = f"{slug}__{ssm_label}"
                is_scluster = str(algorithm["id"]).lower() in {"scluster", "msaf_scluster"}
                # Spectral clustering works on a single raw feature, so it is
                # only run on single-feature runs (not on GraphDitty fusions).
                if is_scluster and len(resolved_features) != 1:
                    continue
                start_time = time.perf_counter()
                # Run segmentation algo
                if is_scluster:
                    boundary_indices = run_algorithm(matrix, algorithm, data, feature=resolved_features[0])
                else:
                    boundary_indices = run_algorithm(matrix, algorithm)
                # Convert frame indices to times, using beat boundaries if available.
                boundary_times = frame_index_to_times(boundary_indices, n_frames, beat_boundaries, duration)
                # Set segments
                segments = [
                    Segment(start=float(start), end=float(end), label=f"S{idx + 1:02d}")
                    for idx, (start, end) in enumerate(zip(boundary_times[:-1], boundary_times[1:]))
                ]

                ### Compute metrics

                # Homogeneity with convolitonal block-matching (CBM) and regularity penalty
                cbm_scores = np.array([
                    cbm_segment_score(matrix, start_idx, end_idx, return_homogeneity_penalty=True)
                    for start_idx, end_idx in zip(boundary_indices[:-1], boundary_indices[1:])
                ])
                cbm_mean_scores = np.mean(cbm_scores, axis=0) if cbm_scores.size else np.array([0.0, 0.0, 0.0], dtype=np.float32)

                segment_metrics = [
                    {
                        "cbm_score": float(score[0]),
                        "homogeneity_score": float(score[1]),
                        "penalty_score": float(score[2]),
                    }
                    for score in cbm_scores
                ]

                cbm_metrics = {
                    "cbm_score_mean": float(cbm_mean_scores[0]),
                    "homogeneity_mean_score": float(cbm_mean_scores[1]),
                    "penalty_mean_score": float(cbm_mean_scores[2]),
                }

                # Regularity metrics
                regularity_metrics = segmentation_regularity_metrics(segments)

                segmentation_metrics = segmentation_boundary_metrics(ref_segments, segments)

                metrics = {**cbm_metrics, **regularity_metrics, **segmentation_metrics}
                jams_path = experiment_dir / run_slug / f"{track_id}.jams"
                write_jams(
                    jams_path,
                    audio_path,
                    duration,
                    algorithm,
                    resolved_features,
                    boundary_times,
                    metrics,
                    segment_metrics,
                    ref_segments,
                )
                rows.append(
                    {
                        "track_id": track_id,
                        "algorithm": slug,
                        "ssm_run": ssm_label,
                        "ssm_features": ",".join(resolved_features),
                        "status": "ok",
                        "duration_s": round(duration, 6),
                        "n_ssm_frames": int(n_frames),
                        "n_boundaries": len(boundary_times),
                        "elapsed_s": round(time.perf_counter() - start_time, 3),
                        "features_npz": str(feature_npz),
                        "jams": str(jams_path),
                        **{key: float(value) for key, value in metrics.items()},
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
    fusion_combos = fusion_cfg.get("features") or []
    if not isinstance(fusion_combos, list):
        raise SystemExit("`segmentation.graphditty_fusion.features` must be a list of feature-name lists.")
    # Every entry in `ssm_features` and every combo in `graphditty_fusion.features`
    # is run and reported separately (see `build_ssm_runs`).
    ssm_runs = build_ssm_runs(ssm_features, fusion_combos)

    if feature_dir is None:
        if audio_dir is None:
            raise SystemExit("Set `segmentation.feature_dir` or `audio_dir` in the YAML config.")
        feature_dir = Path(raw_config.get("out_dir", experiment_dir / "features"))
        feature_paths: list[Path] = []
    else:
        feature_paths = list_feature_files(feature_dir, limit)

    if not feature_paths:
        # No precomputed *_features.npz available: fall back to extracting
        # them at runtime from the audio files pointed to by the config.
        if audio_dir is None:
            raise SystemExit(
                f"No *_features.npz files found in {feature_dir} and no `audio_dir` "
                "configured in the YAML to compute them from."
            )
        feature_cfg = build_feature_config(args, raw_config)
        audio_paths = list_audio_files(audio_dir, limit)
        if not audio_paths:
            raise SystemExit(f"No audio files found in {audio_dir}")
        print(
            f"No precomputed features found in {feature_dir}; extracting from audio in {audio_dir}...",
            flush=True,
        )
        feature_paths = [ensure_features(audio_path, feature_dir, feature_cfg) for audio_path in audio_paths]

    rows: list[dict[str, Any]] = []
    for idx, feature_npz in enumerate(feature_paths, start=1):
        print(f"[{idx:03d}/{len(feature_paths):03d}] {feature_npz.name}", flush=True)
        try:
            rows.extend(process_feature_file(feature_npz, experiment_dir, algorithms, ssm_runs, fusion_cfg))
        except Exception as exc:
            rows.append({"features_npz": str(feature_npz), "status": "error", "error": repr(exc)})
            write_summary(experiment_dir, rows)
            raise
        write_summary(experiment_dir, rows)

    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
