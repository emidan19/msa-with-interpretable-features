#!/usr/bin/env python3
"""Recompute only the downbeats (`beat_downbeat_boundaries`) of existing *_features.npz files.

Older extractions ran the downbeats through the beat regularization, which
inserts fake downbeats at beat rate. This script re-runs beat tracking with
the configuration stored in each track's *_summary.json and rewrites only the
downbeat key. The SSMs depend on the beat grid, so a file is only updated when
the recomputed beats match the stored ones; otherwise it is reported and left
untouched (that track needs a full re-extraction).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from beat_tracking import estimate_beats_and_downbeats
from data_loading import FeatureConfig, load_audio


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_dir", type=Path, help="Folder containing *_features.npz files.")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Folder to look the audio up by file name when the path stored in the npz no longer exists.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-3, help="Tolerance (s) when comparing beat grids.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    return parser.parse_args(argv)


def feature_config_for(npz_path: Path) -> FeatureConfig:
    summary_path = npz_path.with_name(npz_path.name.replace("_features.npz", "_summary.json"))
    stored = {}
    if summary_path.exists():
        stored = json.loads(summary_path.read_text(encoding="utf-8")).get("config", {}) or {}
    defaults = FeatureConfig()
    return FeatureConfig(
        sr=int(stored.get("sr", defaults.sr)),
        hop_length=int(stored.get("hop_length", defaults.hop_length)),
        beat_tracking_method=str(stored.get("beat_tracking_method", defaults.beat_tracking_method)),
        beat_tracking_model_path=stored.get("beat_tracking_model_path", defaults.beat_tracking_model_path),
    )


def recompute_file(npz_path: Path, atol: float, dry_run: bool, audio_dir: Path | None = None) -> str:
    with np.load(npz_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    audio_path = Path(str(np.asarray(payload["audio_path"]).item()))
    if not audio_path.exists() and audio_dir is not None:
        audio_path = audio_dir / audio_path.name
    if not audio_path.exists():
        return f"skipped: audio not found ({audio_path})"

    cfg = feature_config_for(npz_path)
    y, _duration = load_audio(audio_path, cfg.sr)
    beats, downbeats = estimate_beats_and_downbeats(y, cfg)
    stored_beats = np.asarray(payload["beat_boundaries"])
    if beats.shape != stored_beats.shape or not np.allclose(beats, stored_beats, atol=atol):
        return f"skipped: beat grid changed ({stored_beats.size} -> {beats.size} boundaries); re-extract this track"
    if downbeats is None:
        return f"skipped: {cfg.beat_tracking_method} gives no downbeats"

    old = np.asarray(payload.get("beat_downbeat_boundaries", np.array([])))
    message = f"{old.size} -> {downbeats.size} downbeat boundaries ({stored_beats.size} beat boundaries)"
    if dry_run:
        return f"would update: {message}"
    payload["beat_downbeat_boundaries"] = downbeats
    tmp_path = npz_path.with_name(npz_path.name + ".tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, npz_path)
    return f"updated: {message}"


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    npz_paths = sorted(args.feature_dir.glob("*_features.npz"))
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        raise SystemExit(f"No *_features.npz files in {args.feature_dir}")
    n_updated = 0
    for idx, npz_path in enumerate(npz_paths, start=1):
        try:
            status = recompute_file(npz_path, args.atol, args.dry_run, args.audio_dir)
        except Exception as exc:
            status = f"error: {exc!r}"
        n_updated += status.startswith(("updated", "would update"))
        print(f"[{idx:03d}/{len(npz_paths):03d}] {npz_path.name}: {status}", flush=True)
    print(f"{n_updated}/{len(npz_paths)} file(s) {'would be ' if args.dry_run else ''}updated.")
    return 0 if n_updated == len(npz_paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
