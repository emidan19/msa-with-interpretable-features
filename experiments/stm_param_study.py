#!/usr/bin/env python3
"""Study the STM extraction parameters, alone and fused with the SSL features.

Sweeps a grid of `stm_window_s` x `stm_hop_s` (`stm_coeffs` and `stm_min_beats` come from
the config). As in `ssl_partner_fusion.ipynb`, every SSL feature and the STM are segmented
alone, every SSL feature is fused with the STM through GraphDitty (`<SSL>-STM`), and every
run is scored with the boundary F-measure at 0.5 s and 3 s.

The study runs track by track so it can cover a whole dataset with little disk:

* The STM parameters do not change the beat grid, so the beat grid and the SSL features of
  a track are extracted once, kept in memory only, and every grid point is segmented on
  them (only the STM is recomputed). The SSL-alone runs are segmented once per track.
* Segmentation outputs (JAMS, fused SSMs) go to a temporary folder and are discarded once
  scored; only the scores are kept, in `evals.jsonl` (a few KB per track and grid point).
* The SSMs shown in the images are kept only for the `--display-tracks`.

Reruns skip every (track, grid point) already in `evals.jsonl` with the same settings, so
adding tracks or grid points only computes what is new. The SSL features are not cached:
a track with new grid points has its SSL features extracted again.

Outputs, under `--out-dir`:

* `evals.jsonl`: boundary F of every run, one line per (track, SSL-alone runs | grid point).
* `display/<combo>/<track>.npz`: STM and fused SSMs (float16, for plotting only) of the
  display tracks.
* `results/stm_<combo>.png`: one image per grid point with the F@0.5s / F@3s tables (per
  track when there are few tracks, else mean ± std), the mean F per algorithm, the fusion
  gain table and the STM / fused SSMs of the display tracks.
* `results/stm_grid_summary.png`: mean F of the STM and of each fusion across the grid.
* `results/boundary_f_detailed.csv`, `results/fusion_gain.csv`, `results/grid_summary.csv`.

Example:
    python experiments/stm_param_study.py --config configs/harmonix_config.template.yaml \\
        --window-s 4 8 12 --hop-s 0.25 0.5 1.0            # every track with ground truth
    python experiments/stm_param_study.py --config configs/harmonix_config.template.yaml \\
        --tracks 0021_better 0217_pokerface 0764_iwish    # a few tracks
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imported first: it points the matplotlib / XDG caches to ./.cache before the heavy imports.
from run_structure_baseline import (
    algorithm_slug,
    build_feature_config,
    list_audio_files,
    load_config,
    parse_args as parse_baseline_args,
    parse_segmentation_config,
    process_feature_data,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from data_loading import load_audio, normalize_computing_features, sections_path
from evaluate_structure_baseline import (
    BOUNDARY_F1_METRICS,
    SECONDARY_INK,
    algorithm_colors,
    draw_grouped_metric,
    evaluate_jams,
)
from feature_types import aggregate_by_intervals, build_ssm, extract_stm
from pipeline import extract_all
from segmentation import build_ssm_runs, load_ssm, normalize_algorithms
DEFAULT_SSL = ["muq", "musicfm", "mule", "matpac", "repcl"]
DEFAULT_WINDOWS_S = [4.0, 6.0, 8.0, 12.0, 16.0]
DEFAULT_HOPS_S = [0.25, 0.5, 1.0]
SSL_ONLY = "ssl_only"  # `combo` of the evals.jsonl lines holding the SSL-alone runs

# Config fields that change the beat grid or the SSL SSMs.
BASE_CFG_FIELDS = (
    "sr", "hop_length", "n_fft", "n_mels", "beat_tracking_method", "ssm_similarity_type",
    "ssm_similarity_gamma", "mule_hop_frames",
)
# STM fields taken from the config (window and hop are swept).
FIXED_STM_FIELDS = ("stm_coeffs", "stm_min_beats")
BASE_KEYS = ("audio_path", "sr", "duration", "beat_boundaries", "beat_downbeat_boundaries", "sections")

METRIC_LABELS = {"boundary_f_0_5s": "F@0.5s", "boundary_f_3_0s": "F@3s"}
DISPLAY_NAMES = {
    "muq": "MuQ", "musicfm": "MusicFM", "mule": "MULE", "matpac": "MATPAC", "repcl": "RepCL",
    "chroma": "Chroma", "cens": "CENS", "tonnetz": "Tonnetz",
}
SSM_CMAP = "Blues"  # single-hue sequential: darker = more similar
GT_COLOR = "#eb6834"
F_STYLE = (plt.get_cmap("Blues"), Normalize(0.0, 1.0))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="YAML config used by run_structure_baseline.py.")
    parser.add_argument(
        "--tracks", nargs="+", default=None,
        help="Track ids (audio file stems inside `audio_dir`). Default: every track with ground truth.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Keep the first N tracks (default: config `limit`).")
    parser.add_argument("--window-s", type=float, nargs="+", default=DEFAULT_WINDOWS_S, help="stm_window_s values.")
    parser.add_argument("--hop-s", type=float, nargs="+", default=DEFAULT_HOPS_S, help="stm_hop_s values.")
    parser.add_argument("--ssl", nargs="+", default=DEFAULT_SSL, help="SSL features fused with the STM.")
    parser.add_argument(
        "--display-tracks", nargs="*", default=None,
        help="Tracks whose SSMs are kept and shown in the images (default: the first 3 tracks).",
    )
    parser.add_argument(
        "--max-track-columns", type=int, default=6,
        help="Up to this many tracks the F tables have one column per track; above it, mean ± std.",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/stm_param_study")
    parser.add_argument(
        "--reuse-cache-dir",
        type=Path,
        default=REPO_ROOT / "feature_outputs/ssl_partner_fusion_notebook",
        help="Feature cache of ssl_partner_fusion.ipynb; its SSL SSMs are reused when the config matches.",
    )
    parser.add_argument("--ssl-device", default=None, help="Overrides `feature_config.ssl_device`.")
    parser.add_argument("--force", action="store_true", help="Recompute everything, ignoring evals.jsonl.")
    return parser.parse_args(argv)


def display_name(feature: str) -> str:
    return DISPLAY_NAMES.get(feature, feature.upper())


def combo_label(window_s: float, hop_s: float) -> str:
    return f"w{window_s:g}_h{hop_s:g}"


def fingerprint(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def base_cfg_values(cfg) -> dict[str, Any]:
    # Round-trip through JSON so the values compare equal to the ones read back from a fingerprint.
    return json.loads(json.dumps({name: getattr(cfg, name) for name in BASE_CFG_FIELDS}, default=str))


def reusable_notebook_cache(path: Path, cfg) -> dict[str, Any]:
    """Arrays of the notebook's feature cache, if it was extracted with the same base config."""
    if not path.exists():
        return {}
    cached = load_npz(path)
    try:
        notebook_cfg = json.loads(str(np.asarray(cached.get("notebook_cfg", "")).item()))
    except ValueError:
        return {}
    if any(notebook_cfg.get(name) != value for name, value in base_cfg_values(cfg).items()):
        return {}
    return cached


def base_features_for(audio_path: Path, cfg, ssl_features: list[str], reuse_cache_dir: Path | None) -> dict[str, Any]:
    """Beat grid, sections and SSL features of one track, in memory."""
    wanted = lambda key: key in BASE_KEYS or any(key in (f"beat_{s}", f"ssm_{s}") for s in ssl_features)
    if reuse_cache_dir is not None:
        cached = reusable_notebook_cache(reuse_cache_dir / f"{audio_path.stem}_features.npz", cfg)
        cached = {key: value for key, value in cached.items() if wanted(key)}
        if "beat_boundaries" in cached and all(f"ssm_{s}" in cached and f"beat_{s}" in cached for s in ssl_features):
            print(f"  reusing {reuse_cache_dir.name}/{audio_path.stem}_features.npz")
            return cached
    print(f"  extracting: {', '.join(map(display_name, ssl_features))}")
    extract_cfg = replace(cfg, computing_features=normalize_computing_features([*ssl_features, "sections"]))
    _, payload = extract_all(audio_path, None, extract_cfg, return_payload=True)
    return {key: value for key, value in payload.items() if wanted(key)}


def compute_stm(y: np.ndarray, beats: np.ndarray, cfg, window_s: float, hop_s: float) -> dict[str, Any]:
    """Beat-synchronous STM and its SSM for one grid point, on the track's beat grid."""
    stm_cfg = replace(cfg, stm_window_s=window_s, stm_hop_s=hop_s)
    stm, stm_times, _onset_env, effective_window_s = extract_stm(y, stm_cfg, beats)
    beat_stm = aggregate_by_intervals(stm, stm_times, beats)
    return {
        "beat_stm": beat_stm,
        "ssm_stm": build_ssm(beat_stm, similarity_type=cfg.ssm_similarity_type, gamma=cfg.ssm_similarity_gamma),
        "stm_effective_window_s": float(effective_window_s),
        "n_stm_frames": int(stm.shape[1]),
    }


def gt_beat_indices(base: Mapping[str, Any]) -> np.ndarray:
    """Internal ground-truth boundaries (excluding the track start and end) as beat indices."""
    beats = np.asarray(base["beat_boundaries"])
    duration = float(np.asarray(base["duration"]).item())
    sections = base.get("sections", np.empty((0, 3)))
    gt_times = sorted({float(t) for start, end, _ in sections for t in (start, end) if 0.0 < float(t) < duration - 1e-3})
    return np.clip(np.searchsorted(beats, gt_times), 0, len(beats) - 1)


# ---------------------------------------------------------------------------
# Segmentation, evaluation and the evals.jsonl store
# ---------------------------------------------------------------------------


def segment_and_score(
    data: Mapping[str, Any],
    algorithms: list[dict[str, Any]],
    ssm_runs: list[tuple[str, list[str]]],
    fusion_cfg: dict[str, Any],
    cfg,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Segment one track in a temporary folder and score it; return the F rows and the fused SSMs."""
    with tempfile.TemporaryDirectory(prefix="stm_study_") as tmp:
        seg_rows = process_feature_data(data, "", Path(tmp), algorithms, ssm_runs, fusion_cfg)
        rows = []
        for row in seg_rows:
            result = evaluate_jams(Path(row["jams"]), cfg.sections_dir, cfg.label_format)
            if result is None:
                continue
            metrics, _hits = result
            rows.append(
                {
                    "algorithm": row["algorithm"],
                    "ssm_run": row["ssm_run"],
                    "n_boundaries": row["n_boundaries"],
                    **{metric: metrics[metric] for metric in BOUNDARY_F1_METRICS},
                }
            )
        fused = {
            path.name.split("__", 1)[1].removesuffix("_fused_ssm.npz"): load_npz(path)["ssm_fused"]
            for path in (Path(tmp) / "fusion_ssms").glob("*_fused_ssm.npz")
        }
    return rows, fused


def load_evals(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Latest evals.jsonl entry per (combo, track)."""
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue  # a line cut short by an interrupted run
            entries[(entry["combo"], entry["track_id"])] = entry
    return entries


def append_eval(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def fusion_gain(combo_df: pd.DataFrame, ssl_features: list[str], algorithms: list[str]) -> pd.DataFrame:
    """Mean F of each <SSL>-STM fusion and its difference with the SSL alone and the STM alone."""
    mean_by = combo_df.groupby(["ssm_run", "algorithm"])[list(BOUNDARY_F1_METRICS)].mean()
    rows = []
    for algorithm in algorithms:
        for ssl in ssl_features:
            fused = f"fused_{ssl}-stm"
            if (fused, algorithm) not in mean_by.index:
                continue
            row = {"algorithm": algorithm, "fusion": f"{display_name(ssl)}-STM"}
            for metric in BOUNDARY_F1_METRICS:
                label = METRIC_LABELS[metric]
                f_fused = mean_by.loc[(fused, algorithm), metric]
                row[f"{label} fused"] = f_fused
                row[f"{label} Δ vs SSL"] = f_fused - mean_by.loc[(ssl, algorithm), metric]
                row[f"{label} Δ vs STM"] = f_fused - mean_by.loc[("stm", algorithm), metric]
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def text_color(rgba) -> str:
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if luminance < 0.55 else "black"


def group_spans(groups: list[str]) -> list[tuple[str, int, int]]:
    """(group, start, stop) for each run of equal consecutive labels."""
    spans: list[tuple[str, int, int]] = []
    for idx, group in enumerate(groups):
        if spans and spans[-1][0] == group:
            spans[-1] = (group, spans[-1][1], idx + 1)
        else:
            spans.append((group, idx, idx + 1))
    return spans


def draw_table(
    ax,
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    styles: list[tuple[Any, Normalize]],
    fmts: list[str],
    *,
    texts: np.ndarray | None = None,
    col_groups: list[str] | None = None,
    row_groups: list[str] | None = None,
    outline: np.ndarray | None = None,
    title: str | None = None,
    fontsize: float = 8.5,
) -> None:
    """Annotated colored table: one (cmap, norm) and format per column, "–" for NaN.

    `texts` overrides the cell text (the color still comes from `values`). Stands in for the
    notebook's pandas Styler tables, which do not render to an image.
    """
    n_rows, n_cols = values.shape
    for i in range(n_rows):
        for j in range(n_cols):
            value = values[i, j]
            cmap, norm = styles[j]
            if np.isnan(value):
                face, text, ink = "white", "–", SECONDARY_INK
            else:
                face = cmap(norm(value))
                text = texts[i, j] if texts is not None else fmts[j].format(value)
                ink = text_color(face)
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", linewidth=0.8))
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=fontsize, color=ink)
            if outline is not None and outline[i, j]:
                ax.add_patch(Rectangle((j + 0.05, i + 0.08), 0.9, 0.84, fill=False, edgecolor="black", linewidth=1.5))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(col_labels, fontsize=fontsize)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(row_labels, fontsize=fontsize)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if col_groups is not None:
        for group, start, stop in group_spans(col_groups):
            ax.annotate(
                group, xy=((start + stop) / 2, 0), xytext=(0, 6 + 1.8 * fontsize), textcoords="offset points",
                ha="center", va="bottom", fontsize=fontsize + 1, fontweight="bold",
            )
            if start > 0:
                ax.axvline(start, color=SECONDARY_INK, linewidth=1.2)
    if row_groups is not None:
        for _group, start, _stop in group_spans(row_groups):
            if start > 0:
                ax.axhline(start, color=SECONDARY_INK, linewidth=1.2)
    if title:
        ax.set_title(title, fontsize=fontsize + 3, pad=14 + (3.6 * fontsize if col_groups is not None else 1.6 * fontsize), loc="left")


def show_ssm(ax, matrix: np.ndarray, title: str, gt_beats: np.ndarray) -> None:
    ax.imshow(matrix, origin="lower", cmap=SSM_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
    for idx in gt_beats:
        ax.axvline(idx, color=GT_COLOR, linestyle="--", linewidth=0.7, alpha=0.8)
        ax.axhline(idx, color=GT_COLOR, linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)


def f_table(combo_df: pd.DataFrame, metric: str, run_order: list[str], algorithms: list[str], tracks: list[str] | None):
    """Values, texts, column labels and column groups of the F table of one metric.

    With `tracks`, one column per track plus the mean, per algorithm (as in the notebook);
    without, one "mean ± std" column per algorithm.
    """
    if tracks is None:
        stats = combo_df.groupby(["run", "algorithm"])[metric].agg(["mean", "std"])
        values = np.full((len(run_order), len(algorithms)), np.nan)
        texts = np.full(values.shape, "", dtype=object)
        for i, run in enumerate(run_order):
            for j, algorithm in enumerate(algorithms):
                if (run, algorithm) in stats.index:
                    mean, std = stats.loc[(run, algorithm)]
                    values[i, j] = mean
                    texts[i, j] = f"{mean:.2f} ± {std:.2f}"
        return values, texts, list(algorithms), None

    per_track = combo_df.pivot_table(index="run", columns=["algorithm", "track_id"], values=metric, aggfunc="mean")
    mean = combo_df.pivot_table(index="run", columns="algorithm", values=metric, aggfunc="mean")
    columns, col_labels, col_groups = [], [], []
    for algorithm in algorithms:
        for track in [*tracks, "mean"]:
            source, key = (mean, algorithm) if track == "mean" else (per_track, (algorithm, track))
            columns.append(source[key].reindex(run_order) if key in source.columns else pd.Series(np.nan, index=run_order))
            col_labels.append(track)
            col_groups.append(algorithm)
    return np.column_stack([column.to_numpy(dtype=float) for column in columns]), None, col_labels, col_groups


def stm_info_text(window_s: float, stm_info: dict[str, dict[str, Any]], per_track: bool) -> str:
    """Effective STM window and STM frames, per track or summarized over the tracks."""
    if per_track:
        return "Effective STM window / STM frames per track: " + ", ".join(
            f"{track}: {info['stm_effective_window_s']:.2f} s / {info['n_stm_frames']} frames"
            for track, info in stm_info.items()
        )
    windows = np.array([info["stm_effective_window_s"] for info in stm_info.values()])
    frames = np.array([info["n_stm_frames"] for info in stm_info.values()])
    enlarged = int(np.count_nonzero(windows > window_s + 1e-6))
    return (
        f"Effective STM window: median {np.median(windows):.2f} s (max {windows.max():.2f} s; "
        f"{enlarged} of {len(windows)} tracks enlarged by stm_min_beats). STM frames: median {np.median(frames):.0f}."
    )


def plot_combo(
    window_s: float,
    hop_s: float,
    combo_df: pd.DataFrame,
    gain_df: pd.DataFrame,
    gain_limit: float,
    stm_info: dict[str, dict[str, Any]],
    display: dict[str, dict[str, Any]],
    ctx: dict[str, Any],
    out_path: Path,
) -> Path:
    tracks, algorithms, ssl_features = ctx["tracks"], ctx["algorithms"], ctx["ssl"]
    run_order, cfg = ctx["run_order"], ctx["cfg"]
    per_track = len(tracks) <= ctx["max_track_columns"]
    n_table_cols = len(algorithms) * (len(tracks) + 1) if per_track else len(algorithms)
    n_ssm_cols = 1 + len(ssl_features)
    width = max(18.0, 0.95 * n_table_cols + 3.0, 3.0 * n_ssm_cols)
    heights = [
        1.1,
        0.32 * len(run_order) + 1.4,
        0.32 * len(run_order) + 1.4,
        4.6,
        0.32 * max(1, len(gain_df)) + 1.2,
    ]
    if display:
        heights.append(3.1 * len(display))
    fig = plt.figure(figsize=(width, sum(heights)), layout="constrained")
    header, *subfigs = fig.subfigures(len(heights), 1, height_ratios=heights)

    # Header: the grid point and what it actually produced on the tracks.
    header.suptitle(f"STM study: window = {window_s:g} s, hop = {hop_s:g} s", fontsize=16, fontweight="bold", x=0.01, ha="left")
    header.text(
        0.01, 0.05,
        f"stm_coeffs = {cfg.stm_coeffs}, stm_min_beats = {cfg.stm_min_beats}, SSM similarity = {cfg.ssm_similarity_type}, "
        f"beat tracking = {cfg.beat_tracking_method}, GraphDitty = {ctx['fusion_cfg_text']}\n"
        f"{stm_info_text(window_s, stm_info, per_track)}\n"
        f"Algorithms: {', '.join(algorithms)}. {len(tracks)} tracks.",
        fontsize=9.5, color=SECONDARY_INK, ha="left", va="bottom",
    )

    # F tables (cell 14 of the notebook): per track and mean, or mean ± std over many tracks.
    for subfig, metric in zip(subfigs[:2], BOUNDARY_F1_METRICS):
        values, texts, col_labels, col_groups = f_table(combo_df, metric, run_order, algorithms, tracks if per_track else None)
        title = (
            f"{METRIC_LABELS[metric]} per track (and mean over the {len(tracks)} tracks)"
            if per_track
            else f"{METRIC_LABELS[metric]}: mean ± std over the {len(tracks)} tracks"
        )
        ax = subfig.subplots()
        draw_table(
            ax, values, run_order, col_labels, [F_STYLE] * values.shape[1], ["{:.2f}"] * values.shape[1],
            texts=texts, col_groups=col_groups, title=title,
        )

    # Mean F per algorithm (cell 16).
    mean_df = combo_df.groupby(["run", "algorithm"])[list(BOUNDARY_F1_METRICS)].mean().reset_index().rename(columns={"run": "ssm_run"})
    mean_df["ssm_run"] = pd.Categorical(mean_df["ssm_run"], categories=run_order, ordered=True)
    bar_axes = subfigs[2].subplots(1, len(BOUNDARY_F1_METRICS))
    for idx, (ax, metric) in enumerate(zip(bar_axes, BOUNDARY_F1_METRICS)):
        draw_grouped_metric(ax, mean_df, metric, algorithms, ctx["colors"])
        if idx < len(BOUNDARY_F1_METRICS) - 1 and ax.get_legend() is not None:
            ax.get_legend().remove()

    # Fusion gain (cell 18), diverging around 0 with the same limit for every grid point.
    ax = subfigs[3].subplots()
    if gain_df.empty:
        ax.axis("off")
        ax.set_title("Fusion gain: no fused runs", loc="left")
    else:
        gain_cols = [col for col in gain_df.columns if col not in ("algorithm", "fusion")]
        delta_style = (plt.get_cmap("RdBu"), Normalize(-gain_limit, gain_limit))
        draw_table(
            ax,
            gain_df[gain_cols].to_numpy(dtype=float),
            [f"{row.algorithm} · {row.fusion}" for row in gain_df.itertuples()],
            [col.split(" ", 1)[1] for col in gain_cols],
            [delta_style if "Δ" in col else F_STYLE for col in gain_cols],
            ["{:+.2f}" if "Δ" in col else "{:.2f}" for col in gain_cols],
            col_groups=[col.split(" ", 1)[0] for col in gain_cols],
            row_groups=list(gain_df["algorithm"]),
            title=f"Fusion gain (mean over tracks): Δ > 0 means the fusion beats that feature alone (color scale ±{gain_limit:.2f}, shared by the grid)",
        )

    # STM and fused SSMs of the display tracks (cell 12): the matrices the algorithms segmented.
    if display:
        ssm_axes = subfigs[4].subplots(len(display), n_ssm_cols, squeeze=False)
        for row, (track, ssms) in enumerate(display.items()):
            gt_beats = ssms["gt_beats"]
            stm_ssm = ssms["ssm_stm"].astype(np.float32)
            show_ssm(ssm_axes[row, 0], stm_ssm, f"{track}: STM ({stm_ssm.shape[0]} beats)", gt_beats)
            for col, ssl in enumerate(ssl_features, start=1):
                show_ssm(ssm_axes[row, col], ssms[f"ssm_fused_{ssl}"].astype(np.float32), f"{display_name(ssl)}-STM", gt_beats)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_grid_summary(
    summary_df: pd.DataFrame,
    combos: list[tuple[float, float]],
    ctx: dict[str, Any],
    out_path: Path,
) -> Path:
    """Mean F of the STM and each <SSL>-STM fusion (rows) over the grid points (columns).

    One panel per metric and algorithm. The best grid point of each row is outlined, and the
    labels of the fusion rows carry the SSL-alone score (which does not depend on the grid).
    """
    algorithms, ssl_features = ctx["algorithms"], ctx["ssl"]
    runs = ["stm", *(f"fused_{ssl}-stm" for ssl in ssl_features)]
    col_labels = [f"{w:g}/{h:g}" for w, h in combos]
    fig, axes = plt.subplots(
        len(BOUNDARY_F1_METRICS), len(algorithms),
        figsize=(len(algorithms) * (0.62 * len(combos) + 2.8), len(BOUNDARY_F1_METRICS) * (0.36 * len(runs) + 1.9)),
        layout="constrained", squeeze=False,
    )
    mean_by = summary_df.set_index(["combo", "ssm_run", "algorithm"])
    for row_idx, metric in enumerate(BOUNDARY_F1_METRICS):
        for col_idx, algorithm in enumerate(algorithms):
            ax = axes[row_idx, col_idx]
            values = np.array(
                [
                    [mean_by[metric].get((combo_label(w, h), run, algorithm), np.nan) for w, h in combos]
                    for run in runs
                ],
                dtype=float,
            )
            if np.all(np.isnan(values)):
                ax.axis("off")
                continue
            row_labels = []
            for run in runs:
                if run == "stm":
                    row_labels.append("STM")
                    continue
                ssl = run.removeprefix("fused_").removesuffix("-stm")
                ssl_mean = mean_by[metric].get((combo_label(*combos[0]), ssl, algorithm), np.nan)
                row_labels.append(f"{display_name(ssl)}-STM ({display_name(ssl)} {ssl_mean:.2f})")
            # Rows with no value (scluster only runs on single features) stay as "–" so every panel has the same rows.
            outline = np.zeros_like(values, dtype=bool)
            for i, row_values in enumerate(values):
                if not np.all(np.isnan(row_values)):
                    outline[i, int(np.nanargmax(row_values))] = True
            draw_table(
                ax, values, row_labels, col_labels,
                [F_STYLE] * len(combos), ["{:.2f}"] * len(combos), outline=outline,
                title=f"{METRIC_LABELS[metric]} · {algorithm}",
            )
            ax.set_xlabel("stm_window_s / stm_hop_s", fontsize=8)
    fig.suptitle(
        f"STM grid: mean F over {len(ctx['tracks'])} tracks (outlined = best grid point of the row; "
        "in parentheses, the SSL feature alone)",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def select_tracks(args: argparse.Namespace, raw_config: dict[str, Any], cfg) -> dict[str, Path]:
    """Audio path of each studied track; tracks without a ground-truth file are left out."""
    audio_dir = Path(raw_config["audio_dir"]).expanduser()
    audio_by_stem = {path.stem: path for path in list_audio_files(audio_dir, None)}
    if args.tracks:
        missing = [track for track in args.tracks if track not in audio_by_stem]
        if missing:
            raise SystemExit(f"No audio for {missing} in {audio_dir}")
        candidates = list(args.tracks)
    else:
        candidates = sorted(audio_by_stem)
    with_gt = [track for track in candidates if sections_path(cfg.sections_dir, track, cfg.label_format).exists()]
    if len(with_gt) < len(candidates):
        print(f"Leaving out {len(candidates) - len(with_gt)} track(s) without ground truth in {cfg.sections_dir}")
    limit = args.limit if args.limit is not None else raw_config.get("limit")
    if limit is not None and not args.tracks:
        with_gt = with_gt[: int(limit)]
    return {track: audio_by_stem[track] for track in with_gt}


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reuse_cache_dir = args.reuse_cache_dir.resolve() if args.reuse_cache_dir is not None else None

    raw_config = load_config(args.config)
    segmentation_cfg = parse_segmentation_config(raw_config)
    cfg = build_feature_config(parse_baseline_args(["--config", str(args.config)]), raw_config)
    cfg = replace(cfg, ssl_device=args.ssl_device or cfg.ssl_device)
    if cfg.sections_dir is None:
        raise SystemExit("Set `feature_config.sections_dir` in the config: the F-measure needs ground truth.")

    ssl_features = [str(feature).lower() for feature in args.ssl]
    algorithms = normalize_algorithms(segmentation_cfg.get("algorithms"))
    algorithm_slugs = [algorithm_slug(algorithm) for algorithm in algorithms]
    fusion_cfg = dict(segmentation_cfg.get("graphditty_fusion", {}))
    fusion_cfg.pop("features", None)  # the fusions are fixed to <SSL>-STM here
    ssl_runs = build_ssm_runs(ssl_features, [])
    stm_runs = build_ssm_runs(["stm"], [[ssl, "stm"] for ssl in ssl_features])
    run_names = {label: "-".join(map(display_name, features)) for label, features in [*ssl_runs, *stm_runs]}
    combos = sorted({(float(w), float(h)) for w in args.window_s for h in args.hop_s})

    audio_paths = select_tracks(args, raw_config, cfg)
    tracks = list(audio_paths)
    display_tracks = [t for t in (args.display_tracks if args.display_tracks is not None else tracks[:3]) if t in audio_paths]

    print(f"Tracks: {len(tracks)} ({', '.join(tracks[:5])}{', ...' if len(tracks) > 5 else ''})")
    print(f"SSL features: {', '.join(map(display_name, ssl_features))}; algorithms: {', '.join(algorithm_slugs)}")
    print(f"Grid ({len(combos)} points): {', '.join(combo_label(w, h) for w, h in combos)}")

    # Settings each stored score depends on; an evals.jsonl entry is reused only if they match.
    shared_fp = {**base_cfg_values(cfg), "ssl": ssl_features, "algorithms": algorithms, "fusion": fusion_cfg}
    ssl_fp = fingerprint(shared_fp)
    combo_fps = {
        combo_label(w, h): fingerprint(
            {**shared_fp, **{name: getattr(cfg, name) for name in FIXED_STM_FIELDS}, "stm_window_s": w, "stm_hop_s": h}
        )
        for w, h in combos
    }
    evals_path = args.out_dir / "evals.jsonl"
    evals = {} if args.force else load_evals(evals_path)

    def is_done(combo: str, track: str, fp: str) -> bool:
        return evals.get((combo, track), {}).get("fp") == fp

    def display_path(combo: str, track: str) -> Path:
        return args.out_dir / "display" / combo / f"{track}.npz"

    # 1. Track by track: SSL features in memory, SSL-alone runs once, then every missing grid point.
    failed: list[str] = []
    for idx, (track, audio_path) in enumerate(audio_paths.items(), start=1):
        show = track in display_tracks
        ssl_todo = not is_done(SSL_ONLY, track, ssl_fp)
        combos_todo = [
            (w, h) for w, h in combos
            if not is_done(combo_label(w, h), track, combo_fps[combo_label(w, h)])
            or (show and not display_path(combo_label(w, h), track).exists())
        ]
        if not ssl_todo and not combos_todo:
            print(f"[{idx:03d}/{len(tracks):03d}] {track} (done)")
            continue
        print(f"[{idx:03d}/{len(tracks):03d}] {track}: {len(combos_todo)} grid point(s) to compute")
        try:
            base = base_features_for(audio_path, cfg, ssl_features, reuse_cache_dir)
            if ssl_todo:
                rows, _ = segment_and_score(base, algorithms, ssl_runs, fusion_cfg, cfg)
                entry = {"combo": SSL_ONLY, "track_id": track, "fp": ssl_fp, "rows": rows}
                append_eval(evals_path, entry)
                evals[(SSL_ONLY, track)] = entry
            if not combos_todo:
                continue
            beats = np.asarray(base["beat_boundaries"])
            y, _ = load_audio(audio_path, cfg.sr)
            for window_s, hop_s in combos_todo:
                label = combo_label(window_s, hop_s)
                stm = compute_stm(y, beats, cfg, window_s, hop_s)
                data = {**base, "beat_stm": stm["beat_stm"], "ssm_stm": stm["ssm_stm"]}
                rows, fused = segment_and_score(data, algorithms, stm_runs, fusion_cfg, cfg)
                if show:
                    path = display_path(label, track)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        path,
                        ssm_stm=load_ssm(stm, "stm").astype(np.float16),
                        gt_beats=gt_beat_indices(base),
                        **{f"ssm_fused_{ssl}": fused[f"fused_{ssl}-stm"].astype(np.float16) for ssl in ssl_features},
                    )
                entry = {
                    "combo": label, "track_id": track, "fp": combo_fps[label], "rows": rows,
                    "stm_effective_window_s": stm["stm_effective_window_s"], "n_stm_frames": stm["n_stm_frames"],
                }
                append_eval(evals_path, entry)
                evals[(label, track)] = entry
        except Exception:
            traceback.print_exc()
            failed.append(track)

    # 2. Scores of the tracks that have the SSL-alone runs and every grid point.
    complete = [
        track for track in tracks
        if is_done(SSL_ONLY, track, ssl_fp) and all(is_done(label, track, fp) for label, fp in combo_fps.items())
    ]
    if failed:
        print(f"{len(failed)} track(s) failed: {', '.join(failed)}")
    if not complete:
        raise SystemExit("No track has every grid point scored; nothing to plot.")
    if len(complete) < len(tracks):
        print(f"Results over the {len(complete)} of {len(tracks)} tracks with every grid point scored.")

    ssl_rows = [
        {"combo": SSL_ONLY, "track_id": track, **row} for track in complete for row in evals[(SSL_ONLY, track)]["rows"]
    ]
    combo_rows = []
    stm_infos: dict[str, dict[str, dict[str, Any]]] = {}
    for window_s, hop_s in combos:
        label = combo_label(window_s, hop_s)
        stm_infos[label] = {}
        for track in complete:
            entry = evals[(label, track)]
            stm_infos[label][track] = {key: entry[key] for key in ("stm_effective_window_s", "n_stm_frames")}
            info = {"combo": label, "stm_window_s": window_s, "stm_hop_s": hop_s, "track_id": track,
                    "stm_effective_window_s": entry["stm_effective_window_s"]}
            combo_rows += [{**info, **row} for row in entry["rows"]]

    results_dir = args.out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    # The SSL-alone rows are written once (combo "ssl_only"); every grid point is compared against them.
    detailed = pd.DataFrame([*ssl_rows, *combo_rows])
    detailed["run"] = detailed["ssm_run"].map(run_names)
    detailed.to_csv(results_dir / "boundary_f_detailed.csv", index=False)

    ssl_df = pd.DataFrame(ssl_rows).drop(columns="combo")
    combo_df = pd.DataFrame(combo_rows)
    eval_dfs = {
        combo_label(w, h): pd.concat(
            [ssl_df, combo_df[combo_df["combo"] == combo_label(w, h)].drop(columns=["combo", "stm_window_s", "stm_hop_s", "stm_effective_window_s"])],
            ignore_index=True,
        ).assign(run=lambda df: df["ssm_run"].map(run_names))
        for w, h in combos
    }

    # 3. Fusion gain for every grid point; the color limit is shared by the whole grid.
    gains = {label: fusion_gain(df, ssl_features, algorithm_slugs) for label, df in eval_dfs.items()}
    pd.concat([gain.assign(combo=label) for label, gain in gains.items()]).to_csv(results_dir / "fusion_gain.csv", index=False)
    deltas = [gain[[c for c in gain.columns if "Δ" in c]].to_numpy(dtype=float) for gain in gains.values() if not gain.empty]
    gain_limit = max([0.05, *(float(np.nanmax(np.abs(d))) for d in deltas if d.size and not np.all(np.isnan(d)))])

    # 4. One image per grid point, then the grid summary.
    ctx = {
        "tracks": complete, "algorithms": algorithm_slugs, "ssl": ssl_features,
        "run_order": list(run_names.values()), "cfg": cfg, "colors": algorithm_colors(algorithm_slugs),
        "max_track_columns": args.max_track_columns,
        "fusion_cfg_text": ", ".join(f"{k}={v}" for k, v in fusion_cfg.items() if k != "verbose"),
    }
    for window_s, hop_s in combos:
        label = combo_label(window_s, hop_s)
        display = {
            track: load_npz(display_path(label, track))
            for track in display_tracks
            if track in complete and display_path(label, track).exists()
        }
        out_path = plot_combo(
            window_s, hop_s, eval_dfs[label], gains[label], gain_limit, stm_infos[label],
            display, ctx, results_dir / f"stm_{label}.png",
        )
        print(f"  wrote {out_path}")

    summary_df = (
        pd.concat([df.assign(combo=label) for label, df in eval_dfs.items()])
        .groupby(["combo", "ssm_run", "run", "algorithm"])[list(BOUNDARY_F1_METRICS)]
        .mean()
        .reset_index()
    )
    summary_df.to_csv(results_dir / "grid_summary.csv", index=False)
    out_path = plot_grid_summary(summary_df, combos, ctx, results_dir / "stm_grid_summary.png")
    print(f"  wrote {out_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
