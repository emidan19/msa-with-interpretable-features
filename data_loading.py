"""Configuration, audio loading, and annotation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from runtime_env import ensure_local_cache

ensure_local_cache()

import librosa
import numpy as np
import soundfile as sf
import yaml


RWC_POP_SECTION_LABEL_MAP = {
    "bridge": "bridge",
    "chorus": "chorus",
    "ending": "ending",
    "intro": "intro",
    "nothing": "nothing",
    "post-chorus": "post-chorus",
    "pre-chorus": "pre-chorus",
    "verse": "verse",
}

HARMONIX_SECTION_LABEL_MAP = {
    "bridge": "bridge",
    "chorus": "chorus",
    "ending": "ending",
    "end": "ending",
    "outro": "outro",
    "intro": "intro",
    "nothing": "nothing",
    "silence": "nothing",
    "post-chorus": "post-chorus",
    "postchorus": "post-chorus",
    "pre-chorus": "pre-chorus",
    "prechorus": "pre-chorus",
    "verse": "verse",
    "inst": "instrumental",
    "instrumental": "instrumental",
    "solo": "instrumental",
    "gtr": "instrumental",
    "guitar": "instrumental",
    "guitarsolo": "instrumental",
}


@dataclass
class FeatureConfig:
    sr: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 40
    n_mfcc: int = 20
    beat_tracking_method: str = "beat_this"
    beat_tracking_model_path: str | None = None
    sections_dir: Path | None = None
    label_format: str = "rwc_pop"
    computing_features: tuple[str, ...] = ("stm", "cens", "sections", "fused")
    preview_features: tuple[str, ...] = ("stm", "cens", "sections")
    preview_formats: tuple[str, ...] = ("png", "svg")
    ssm_similarity_type: str = "cosine"
    ssm_similarity_gamma: float | None = None
    stm_window_s: float = 8.0
    stm_hop_s: float = 0.5
    stm_min_beats: int = 5
    stm_coeffs: int = 400
    f0_min_note: str = "C2"
    f0_max_note: str = "C7"
    f0_confidence_threshold: float = 0.2
    f0_max_interp_gap_s: float = 0.25
    ssl_models_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "SSL_models"
    )
    matpac_checkpoint_path: Path | None = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "SSL_models"
        / "matpac"
        / "ckpts_folder"
        / "matpac_10_2048.pt"
    )
    musicfm_stat_path: Path | None = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "SSL_models"
        / "musicfm"
        / "data"
        / "msd_stats.json"
    )
    musicfm_model_path: Path | None = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "SSL_models"
        / "musicfm"
        / "data"
        / "pretrained_msd.pt"
    )
    muq_model_name: str = "OpenMuQ/MuQ-large-msd-iter"
    mule_model_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "SSL_models"
        / "music-audio-representations"
        / "supporting_data"
        / "model"
        / "model.keras"
    )
    # Hop between MULE embeddings in mel frames (10 ms each). The original MULE timeline uses 200 (2 s).
    mule_hop_frames: int = 50
    mule_batch_size: int = 64

    triplet_mining_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "SSL_models" / "Triplet_Mining"
    )
    repcl_checkpoint_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "SSL_models"
        / "Triplet_Mining"
        / "results"
        / "models"
        / "model_20k.pt"
    )

    ssl_device: str = "cpu"
    
FEATURE_NAME_ORDER = (
    "stm",
    "mfcc",
    "chroma",
    "cens",
    "arrangement",
    "vocal",
    "bass",
    "tonnetz",
    "density",
    "musicfm",
    "muq",
    "matpac",
    "repcl",
    "mule",
    "sections",
    "fused",
)
PREVIEW_FEATURE_ORDER = FEATURE_NAME_ORDER


def normalize_section_label(
    label: str, section_map: dict[str, str] = RWC_POP_SECTION_LABEL_MAP
) -> str:
    normalized = label.strip().strip('"').lower()
    stripped = normalized.rstrip("0123456789")
    for candidate in (normalized, stripped):
        for prefix, grouped in section_map.items():
            if candidate == prefix or candidate.startswith(f"{prefix} "):
                return grouped
    return normalized


def normalize_label_format(label_format: str | None) -> str:
    if label_format is None:
        return "rwc_pop"
    normalized = str(label_format).strip().lower()
    aliases = {
        "rwc_pop": "rwc_pop",
        "rwc-pop": "rwc_pop",
        "rwc": "rwc_pop",
        "harmonix": "harmonix",
        "harmonixset": "harmonix",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown label format: {label_format!r}. Valid options: rwc_pop, harmonix"
        )
    return aliases[normalized]


def _load_rwc_pop_sections(labels_path: Path) -> list[tuple[float, float, str]]:
    if not labels_path.exists():
        return []
    sections = []
    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            start_cs, end_cs, label = parts
            sections.append(
                (
                    int(start_cs) / 100.0,
                    int(end_cs) / 100.0,
                    normalize_section_label(label, RWC_POP_SECTION_LABEL_MAP),
                )
            )
    return sections


def _load_harmonix_sections(
    labels_path: Path, duration: float | None = None
) -> list[tuple[float, float, str]]:
    # Per the Harmonix README, each line is "the label of the segment that
    # starts on the current boundary" - including the last line (typically
    # "end"), whose segment runs until the true end of the audio. Without a
    # known `duration` we can't close that final segment, so it's dropped.
    if not labels_path.exists():
        return []
    rows: list[tuple[float, str]] = []
    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            rows.append((float(parts[0]), parts[1]))
    sections = []
    for (start, label), (end, _) in zip(rows, rows[1:]):
        sections.append((start, end, normalize_section_label(label, HARMONIX_SECTION_LABEL_MAP)))
    if rows:
        last_start, last_label = rows[-1]
        if duration is not None and duration > last_start:
            sections.append(
                (last_start, float(duration), normalize_section_label(last_label, HARMONIX_SECTION_LABEL_MAP))
            )
    return sections


def load_sections(
    sections_dir: Path | None,
    audio_id: str,
    label_format: str = "rwc_pop",
    duration: float | None = None,
) -> list[tuple[float, float, str]]:
    if sections_dir is None:
        return []
    label_format = normalize_label_format(label_format)
    path = sections_path(sections_dir, audio_id, label_format)
    if label_format == "harmonix":
        return _load_harmonix_sections(path, duration)
    return _load_rwc_pop_sections(path)


def sections_path(sections_dir: Path, audio_id: str, label_format: str = "rwc_pop") -> Path:
    """Path of the section annotation file for `audio_id` (it may not exist)."""
    if normalize_label_format(label_format) == "harmonix":
        return Path(sections_dir) / f"{audio_id}.txt"
    return Path(sections_dir) / f"{audio_id}.CHORUS.TXT"


def _normalize_feature_names(features: Iterable[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not features:
        return default
    normalized = []
    for name in features:
        key = str(name).strip().lower()
        if key not in FEATURE_NAME_ORDER:
            raise ValueError(
                f"Unknown feature name: {name!r}. Valid options: {', '.join(FEATURE_NAME_ORDER)}"
            )
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def normalize_computing_features(features: Iterable[str] | None) -> tuple[str, ...]:
    return _normalize_feature_names(features, FeatureConfig.computing_features)


def normalize_preview_features(features: Iterable[str] | None) -> tuple[str, ...]:
    return _normalize_feature_names(features, FeatureConfig.preview_features)


def normalize_preview_formats(formats: Iterable[str] | None) -> tuple[str, ...]:
    valid_formats = {"png", "svg"}
    if not formats:
        return FeatureConfig.preview_formats
    normalized = []
    for fmt in formats:
        key = str(fmt).strip().lower()
        if key not in valid_formats:
            raise ValueError(f"Unknown preview format: {fmt!r}. Valid options: png, svg")
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def normalize_ssm_similarity_type(similarity_type: str | None) -> str:
    if similarity_type is None:
        return "cosine"
    normalized = str(similarity_type).strip().lower()
    aliases = {
        "autocorr": "autocorrelation",
        "autocorrelation": "autocorrelation",
        "covariance": "autocorrelation",
        "rbf": "rbf",
        "centered_rbf": "centered_rbf",
        "cosine": "cosine",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown SSM similarity type: {similarity_type!r}. Valid options: cosine, autocorrelation, covariance, rbf, centered_rbf"
        )
    return aliases[normalized]


def config_from_mapping(data: dict[str, Any]) -> FeatureConfig:
    payload = dict(data)
    for path_key in (
        "sections_dir",
        "ssl_models_root",
        "matpac_checkpoint_path",
        "musicfm_stat_path",
        "musicfm_model_path",
        "triplet_mining_root",
        "repcl_checkpoint_path",
        "mule_model_path",
    ):
        if path_key in payload and payload[path_key] is not None:
            payload[path_key] = Path(payload[path_key])
    if "computing_features" in payload:
        payload["computing_features"] = normalize_computing_features(payload["computing_features"])
    if "preview_features" in payload:
        payload["preview_features"] = normalize_preview_features(payload["preview_features"])
    if "preview_formats" in payload:
        payload["preview_formats"] = normalize_preview_formats(payload["preview_formats"])
    if "ssm_similarity_type" in payload:
        payload["ssm_similarity_type"] = normalize_ssm_similarity_type(payload["ssm_similarity_type"])
    if "label_format" in payload:
        payload["label_format"] = normalize_label_format(payload["label_format"])
    if "ssm_similarity_gamma" in payload and payload["ssm_similarity_gamma"] is not None:
        payload["ssm_similarity_gamma"] = float(payload["ssm_similarity_gamma"])
    return FeatureConfig(**payload)


def load_yaml_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config at {path} must contain a mapping at the top level.")
    return payload


def load_runtime_config(config_path: Path | None) -> tuple[dict[str, Any], FeatureConfig]:
    raw = load_yaml_config(config_path) if config_path is not None else {}
    feature_cfg = raw.get("feature_config", raw)
    if not isinstance(feature_cfg, dict):
        raise ValueError("`feature_config` must be a mapping when present in the YAML config.")
    return raw, config_from_mapping(feature_cfg)


def resolve_value(cli_value: Any, yaml_value: Any, default_value: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if yaml_value is not None:
        return yaml_value
    return default_value


def load_audio(audio_path: Path, sr: int) -> tuple[np.ndarray, float]:
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    return y, float(duration)


def make_demo_audio(path: Path, sr: int) -> None:
    seconds = 36.0
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    y = np.zeros_like(t)
    sections = [
        (0.0, 12.0, 220.0, 2.0),
        (12.0, 24.0, 330.0, 3.0),
        (24.0, 36.0, 220.0, 2.0),
    ]
    for start, end, freq, beat_hz in sections:
        mask = (t >= start) & (t < end)
        local_t = t[mask] - start
        carrier = 0.35 * np.sin(2 * np.pi * freq * local_t)
        harmonic = 0.15 * np.sin(2 * np.pi * 2 * freq * local_t)
        pulse = 0.5 + 0.5 * np.maximum(0.0, np.sin(2 * np.pi * beat_hz * local_t))
        y[mask] = (carrier + harmonic) * (0.4 + 0.6 * pulse)
    sf.write(path, y.astype(np.float32), sr)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
