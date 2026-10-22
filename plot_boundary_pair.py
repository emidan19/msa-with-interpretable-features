#!/usr/bin/env python3
"""Plot which reference boundaries two SSM features recover, side by side.

For one segmentation algorithm, one hit-rate window and a pair of SSM runs
(A, B) from a `run_structure_baseline.py` experiment, every internal
ground-truth boundary of a track falls in one of four cases, using the same
hit/miss rule as `evaluate_structure_baseline.py` (`boundary_hit_vector`):

* both: A and B recover it within the window,
* only A / only B: just one of them does,
* neither: both miss it.

One figure per track: the SSMs of A and B side by side (axes in seconds, on
the beat grid the SSMs live on) with the reference boundaries drawn as lines
colored by case and each feature's estimated boundaries as ticks on the top
edge, and below them a timeline with A's estimated boundaries, the colored
reference boundaries and B's estimated boundaries.

A global `summary.png` pools every track: the 2x2 contingency matrix of
reference boundaries (rows: A miss/hit, columns: B miss/hit) and the false
positives of A and B, i.e. estimated boundaries left unmatched by the same
one-to-one matching, split into only A / shared / only B, where a shared false
positive is a pair of A and B false positives within the window of each other.

SSMs are read from `<feature_dir>/<track_id>_features.npz` with the same
normalization the segmentation used (`resolve_segmentation_ssm`); GraphDitty
fusions are read from `<experiment_dir>/fusion_ssms/`. Output goes to
`<experiment_dir>/results/boundary_pairs/<A>_<B>_<algorithm>/`: `summary.png`,
the same numbers in `summary.csv` (pooled) and `per_track.csv`, and the
per-track figures under `tracks/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from evaluate_structure_baseline import (
    CATEGORICAL_PALETTE,
    SECONDARY_INK,
    load_jams_segment_sets,
    load_reference_segments_from_config,
)
from msa_metrics import Segment, boundary_hit_vector
from run_structure_baseline import load_config, parse_segmentation_config
from segmentation import resolve_segmentation_ssm

# (A hit, B hit) -> case. Colors: both = aqua, only A = blue, only B = orange,
# neither = muted gray, so the two "exclusive" cases stand out.
CASES = ("both", "only_a", "only_b", "neither")
CASE_COLORS = {
    "both": CATEGORICAL_PALETTE[2],
    "only_a": CATEGORICAL_PALETTE[0],
    "only_b": CATEGORICAL_PALETTE[1],
    "neither": "#a3a29c",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="YAML config used by run_structure_baseline.py.")
    parser.add_argument("--experiment-dir", type=Path, default=None, help="Overrides `segmentation.experiment_dir`.")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=None,
        help="Folder with <track_id>_features.npz (overrides `segmentation.feature_dir`).",
    )
    parser.add_argument("--algorithm", required=True, help="Algorithm slug, e.g. as_seg_cbm.")
    parser.add_argument("--pair", nargs=2, required=True, metavar=("A", "B"), help="Two SSM run labels, e.g. mfcc chroma.")
    parser.add_argument("--window", type=float, default=3.0, help="Hit-rate window in seconds (default: 3.0).")
    parser.add_argument("--tracks", nargs="*", default=None, help="Track ids (default: every track with both JAMS).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to write the figures.")
    return parser.parse_args(argv)


def internal_boundaries(segments: list[Segment]) -> np.ndarray:
    """Internal boundary times, with the same `trim` convention as `boundary_hit_vector`."""
    intervals = np.asarray([[seg.start, seg.end] for seg in segments], dtype=float)
    return np.asarray(mir_eval.util.intervals_to_boundaries(intervals))[1:-1]


def boundary_cases(hits_a: np.ndarray, hits_b: np.ndarray) -> list[str]:
    lookup = {(1, 1): "both", (1, 0): "only_a", (0, 1): "only_b", (0, 0): "neither"}
    return [lookup[(int(a), int(b))] for a, b in zip(hits_a, hits_b)]


def load_ssm_on_beats(
    track_id: str, ssm_run: str, feature_dir: Path, experiment_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ssm, beat_boundaries): the SSM the algorithm saw and its beat grid in seconds."""
    npz_path = feature_dir / f"{track_id}_features.npz"
    with np.load(npz_path, allow_pickle=True) as data:
        beat_boundaries = np.asarray(data["beat_boundaries"], dtype=float)
        fused_path = experiment_dir / "fusion_ssms" / f"{track_id}__{ssm_run}_fused_ssm.npz"
        if fused_path.exists():
            with np.load(fused_path, allow_pickle=True) as fused:
                ssm = np.asarray(fused["ssm_fused"], dtype=float)
        else:
            ssm, _ = resolve_segmentation_ssm(data, [ssm_run], {})
    if beat_boundaries.size != ssm.shape[0] + 1:
        raise ValueError(
            f"{track_id}/{ssm_run}: SSM has {ssm.shape[0]} frames but the beat grid has "
            f"{beat_boundaries.size} boundaries."
        )
    return ssm, beat_boundaries


def draw_ssm(
    ax: plt.Axes,
    ssm: np.ndarray,
    beat_boundaries: np.ndarray,
    ref_times: np.ndarray,
    cases: list[str],
    est_times: np.ndarray,
    title: str,
) -> None:
    # pcolormesh on the beat edges keeps the axes in seconds even though beats
    # are not evenly spaced (imshow + extent would stretch them uniformly).
    ax.pcolormesh(beat_boundaries, beat_boundaries, ssm, cmap="Greys", vmin=0.0, vmax=1.0, rasterized=True)
    # Light full-length lines so the blocks stay readable, plus a marker on the
    # diagonal, where a boundary is the corner between two homogeneous blocks.
    for time, case in zip(ref_times, cases):
        color = CASE_COLORS[case]
        ax.axvline(time, color=color, linewidth=0.9, alpha=0.45)
        ax.axhline(time, color=color, linewidth=0.9, alpha=0.45)
    ax.scatter(
        ref_times, ref_times, c=[CASE_COLORS[case] for case in cases],
        s=40, edgecolors="white", linewidths=0.8, zorder=4,
    )
    # Estimated boundaries as ticks hanging from the top edge.
    ax.plot(est_times, np.full_like(est_times, beat_boundaries[0]), "v", color="black", markersize=5, clip_on=False)
    ax.set_xlim(beat_boundaries[0], beat_boundaries[-1])
    ax.set_ylim(beat_boundaries[-1], beat_boundaries[0])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("time (s)")


def draw_timeline(
    ax: plt.Axes,
    ref_times: np.ndarray,
    cases: list[str],
    est_a: np.ndarray,
    est_b: np.ndarray,
    label_a: str,
    label_b: str,
    window: float,
    duration: float,
) -> None:
    rows = {f"{label_a} est.": 2, "ground truth": 1, f"{label_b} est.": 0}
    ax.vlines(est_a, 2 - 0.3, 2 + 0.3, color="black", linewidth=1.5)
    ax.vlines(est_b, 0 - 0.3, 0 + 0.3, color="black", linewidth=1.5)
    for time, case in zip(ref_times, cases):
        color = CASE_COLORS[case]
        # Hit-rate window around each reference boundary, then the boundary itself.
        ax.axvspan(time - window, time + window, color=color, alpha=0.12, linewidth=0)
        ax.vlines(time, 1 - 0.35, 1 + 0.35, color=color, linewidth=2.5)
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels(list(rows.keys()))
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(0.0, duration)
    ax.set_xlabel("time (s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def unmatched_boundaries(ref_times: np.ndarray, est_times: np.ndarray, window: float) -> np.ndarray:
    """Estimated boundaries left out of the one-to-one matching, i.e. the false positives."""
    if est_times.size == 0 or ref_times.size == 0:
        return est_times
    matched = {est_idx for _, est_idx in mir_eval.util.match_events(ref_times, est_times, float(window))}
    return np.asarray([time for idx, time in enumerate(est_times) if idx not in matched], dtype=float)


def false_positive_split(
    ref_times: np.ndarray, est_a: np.ndarray, est_b: np.ndarray, window: float
) -> dict[str, int]:
    """False positives of A and B, and how many of them coincide (one-to-one, within the window)."""
    fp_a = unmatched_boundaries(ref_times, est_a, window)
    fp_b = unmatched_boundaries(ref_times, est_b, window)
    shared = len(mir_eval.util.match_events(fp_a, fp_b, float(window))) if fp_a.size and fp_b.size else 0
    return {"n_est_a": est_a.size, "n_est_b": est_b.size, "fp_a": fp_a.size, "fp_b": fp_b.size, "shared": shared}


def load_track_boundaries(
    track_id: str,
    algorithm: str,
    run_a: str,
    run_b: str,
    window: float,
    experiment_dir: Path,
    sections_dir: Path | None,
    label_format: str,
) -> dict | None:
    """Reference/estimated boundaries of one track, with per-boundary hits and false-positive counts."""
    jams_a = experiment_dir / f"{algorithm}__{run_a}" / f"{track_id}.jams"
    jams_b = experiment_dir / f"{algorithm}__{run_b}" / f"{track_id}.jams"
    est_a_segments, ref_segments, duration = load_jams_segment_sets(jams_a)
    est_b_segments, _, _ = load_jams_segment_sets(jams_b)
    if not ref_segments:
        ref_segments = load_reference_segments_from_config(sections_dir, label_format, track_id, duration)
    if not ref_segments or not est_a_segments or not est_b_segments:
        print(f"Skipping {track_id}: missing reference or estimated segmentation.")
        return None

    hits_a = boundary_hit_vector(ref_segments, est_a_segments, window=window)
    hits_b = boundary_hit_vector(ref_segments, est_b_segments, window=window)
    ref_times = internal_boundaries(ref_segments)
    est_a = internal_boundaries(est_a_segments)
    est_b = internal_boundaries(est_b_segments)
    return {
        "track_id": track_id,
        "ref_segments": ref_segments,
        "duration": duration,
        "ref_times": ref_times,
        "hits_a": hits_a,
        "hits_b": hits_b,
        "cases": boundary_cases(hits_a, hits_b),
        "est_a": est_a,
        "est_b": est_b,
        **false_positive_split(ref_times, est_a, est_b, window),
    }


def plot_track(
    track: dict,
    algorithm: str,
    run_a: str,
    run_b: str,
    window: float,
    experiment_dir: Path,
    feature_dir: Path,
    out_dir: Path,
) -> Path | None:
    track_id = track["track_id"]
    ref_segments, ref_times, cases = track["ref_segments"], track["ref_times"], track["cases"]
    hits_a, hits_b, est_a, est_b = track["hits_a"], track["hits_b"], track["est_a"], track["est_b"]

    try:
        ssm_a, beats_a = load_ssm_on_beats(track_id, run_a, feature_dir, experiment_dir)
        ssm_b, beats_b = load_ssm_on_beats(track_id, run_b, feature_dir, experiment_dir)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Skipping {track_id} figure: cannot load SSM ({exc}).")
        return None

    counts = {case: cases.count(case) for case in CASES}
    duration = track["duration"]
    duration = float(duration) if duration is not None else float(max(beats_a[-1], ref_segments[-1].end))

    fig = plt.figure(figsize=(12, 8.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(4.2, 1.0), hspace=0.18, wspace=0.25)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], sharex=ax_a, sharey=ax_a)
    ax_t = fig.add_subplot(grid[1, :])

    draw_ssm(ax_a, ssm_a, beats_a, ref_times, cases, est_a, f"A: {run_a}  (recall {hits_a.mean():.2f})")
    draw_ssm(ax_b, ssm_b, beats_b, ref_times, cases, est_b, f"B: {run_b}  (recall {hits_b.mean():.2f})")
    draw_timeline(ax_t, ref_times, cases, est_a, est_b, run_a, run_b, window, duration)

    case_labels = {"both": "both", "only_a": f"only {run_a}", "only_b": f"only {run_b}", "neither": "neither"}
    handles = [
        Line2D([], [], color=CASE_COLORS[case], linewidth=2.5, label=f"{case_labels[case]} ({counts[case]})")
        for case in CASES
    ]
    handles.append(Line2D([], [], color="black", marker="v", linestyle="", label="estimated boundary"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"{track_id} · {algorithm} · {window}s window · {len(cases)} reference boundaries",
        color=SECONDARY_INK,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{track_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{track_id}: " + ", ".join(f"{case_labels[case]}={counts[case]}" for case in CASES) + f" -> {out_path}")
    return out_path


def contingency_matrix(hits_a: np.ndarray, hits_b: np.ndarray) -> np.ndarray:
    """2x2 counts: rows = A miss/hit, columns = B miss/hit."""
    return np.array([[int(np.sum((hits_a == a) & (hits_b == b))) for b in (0, 1)] for a in (0, 1)])


def boundary_counts(hits_a: np.ndarray, hits_b: np.ndarray, fp: dict[str, int]) -> dict[str, float]:
    """Contingency cells, recall, false positives and precision of A and B as one flat row."""
    matrix = contingency_matrix(hits_a, hits_b)
    n_ref = int(matrix.sum())

    def ratio(num: float, den: float) -> float:
        return num / den if den else np.nan

    return {
        "n_ref": n_ref,
        "both": int(matrix[1, 1]),
        "only_a": int(matrix[1, 0]),
        "only_b": int(matrix[0, 1]),
        "neither": int(matrix[0, 0]),
        "recall_a": ratio(matrix[1].sum(), n_ref),
        "recall_b": ratio(matrix[:, 1].sum(), n_ref),
        "n_est_a": int(fp["n_est_a"]),
        "n_est_b": int(fp["n_est_b"]),
        "fp_a": int(fp["fp_a"]),
        "fp_b": int(fp["fp_b"]),
        "fp_shared": int(fp["shared"]),
        "fp_only_a": int(fp["fp_a"] - fp["shared"]),
        "fp_only_b": int(fp["fp_b"] - fp["shared"]),
        "precision_a": 1 - ratio(fp["fp_a"], fp["n_est_a"]),
        "precision_b": 1 - ratio(fp["fp_b"], fp["n_est_b"]),
    }


def write_csvs(tracks: list[dict], meta: dict[str, object], out_dir: Path) -> tuple[Path, Path]:
    """`per_track.csv` (one row per track) and `summary.csv` (pooled over every track)."""
    per_track = pd.DataFrame([
        {**meta, "track_id": track["track_id"], **boundary_counts(track["hits_a"], track["hits_b"], track)}
        for track in tracks
    ])
    fp = {key: sum(track[key] for track in tracks) for key in ("n_est_a", "n_est_b", "fp_a", "fp_b", "shared")}
    pooled = boundary_counts(
        np.concatenate([track["hits_a"] for track in tracks]),
        np.concatenate([track["hits_b"] for track in tracks]),
        fp,
    )
    # Pooled recall/precision weight every boundary equally; the per-track means
    # weight every track equally, as the averaged metrics of the evaluation do.
    track_means = {
        f"{metric}_track_mean": float(per_track[metric].mean())
        for metric in ("recall_a", "recall_b", "precision_a", "precision_b")
    }
    summary = pd.DataFrame([{**meta, "n_tracks": len(tracks), **pooled, **track_means}])

    out_dir.mkdir(parents=True, exist_ok=True)
    per_track_path = out_dir / "per_track.csv"
    summary_path = out_dir / "summary.csv"
    per_track.to_csv(per_track_path, index=False)
    summary.to_csv(summary_path, index=False)
    return summary_path, per_track_path


def draw_contingency(ax: plt.Axes, matrix: np.ndarray, run_a: str, run_b: str) -> None:
    n_ref = int(matrix.sum())
    # Same case colors as the per-track figures: rows/cols are (miss, hit).
    cell_cases = (("neither", "only_b"), ("only_a", "both"))
    for row in range(2):
        for col in range(2):
            count = int(matrix[row, col])
            ax.add_patch(Rectangle(
                (col + 0.02, row + 0.02), 0.96, 0.96,
                facecolor=CASE_COLORS[cell_cases[row][col]], alpha=0.3, linewidth=0,
            ))
            ax.text(col + 0.5, row + 0.45, str(count), ha="center", va="center", fontsize=18, fontweight="bold")
            share = count / n_ref if n_ref else 0.0
            ax.text(col + 0.5, row + 0.66, f"{share:.0%}", ha="center", va="center", fontsize=10, color=SECONDARY_INK)
    # Marginal totals: right of each row, below each column.
    for idx in range(2):
        ax.text(2.06, idx + 0.5, f"n={matrix[idx].sum()}", ha="left", va="center", fontsize=9, color=SECONDARY_INK)
        ax.text(idx + 0.5, 2.06, f"n={matrix[:, idx].sum()}", ha="center", va="top", fontsize=9, color=SECONDARY_INK)

    recall_a = matrix[1].sum() / n_ref if n_ref else 0.0
    recall_b = matrix[:, 1].sum() / n_ref if n_ref else 0.0
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_aspect("equal")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks([0.5, 1.5], ["miss", "hit"])
    ax.set_yticks([0.5, 1.5], ["miss", "hit"])
    ax.set_xlabel(f"B: {run_b}  (recall {recall_b:.2f})")
    ax.set_ylabel(f"A: {run_a}  (recall {recall_a:.2f})")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Reference boundaries ({n_ref})", pad=12)


def draw_false_positives(ax: plt.Axes, fp: dict[str, int], run_a: str, run_b: str, window: float) -> None:
    shared = fp["shared"]
    segments = (("only_a", fp["fp_a"] - shared), ("both", shared), ("only_b", fp["fp_b"] - shared))
    total = sum(width for _, width in segments)
    ax.set_title("False positives", pad=12)
    if total == 0:
        ax.text(0.5, 0.5, "no false positives", ha="center", va="center", color=SECONDARY_INK, transform=ax.transAxes)
        ax.axis("off")
        return

    # One bar split like a Venn diagram: only A | shared | only B, with a
    # white gap between segments and the count inside each one.
    left = 0
    for case, width in segments:
        if width:
            ax.barh(0, width, left=left, height=0.5, color=CASE_COLORS[case], edgecolor="white", linewidth=2)
            if width / total > 0.04:
                ax.text(left + width / 2, 0, str(width), ha="center", va="center", color="white", fontweight="bold")
        left += width

    def precision(n_fp: int, n_est: int) -> str:
        return f"{1 - n_fp / n_est:.2f}" if n_est else "n/a"

    def bracket(x0: float, x1: float, y: float, text: str, above: bool) -> None:
        tick = -0.06 if above else 0.06
        ax.plot([x0, x0, x1, x1], [y + tick, y, y, y + tick], color=SECONDARY_INK, linewidth=1)
        ax.text(
            (x0 + x1) / 2, y - tick, text, ha="center", va="bottom" if above else "top",
            fontsize=9, color=SECONDARY_INK,
        )

    only_a = segments[0][1]
    bracket(0, only_a + shared, 0.4, f"FP {run_a} = {fp['fp_a']}  (precision {precision(fp['fp_a'], fp['n_est_a'])})", True)
    bracket(only_a, total, -0.4, f"FP {run_b} = {fp['fp_b']}  (precision {precision(fp['fp_b'], fp['n_est_b'])})", False)

    ax.set_xlim(0, total)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel(f"estimated boundaries with no reference within ±{window}s")
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    labels = {"only_a": f"only {run_a}", "both": f"shared (≤ {window}s apart)", "only_b": f"only {run_b}"}
    handles = [Patch(color=CASE_COLORS[case], label=labels[case]) for case, _ in segments]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)


def plot_summary(tracks: list[dict], algorithm: str, run_a: str, run_b: str, window: float, out_path: Path) -> Path:
    """Global figure: pooled hit/miss contingency matrix and false-positive overlap of A and B."""
    matrix = contingency_matrix(
        np.concatenate([track["hits_a"] for track in tracks]),
        np.concatenate([track["hits_b"] for track in tracks]),
    )
    fp = {key: sum(track[key] for track in tracks) for key in ("n_est_a", "n_est_b", "fp_a", "fp_b", "shared")}

    fig = plt.figure(figsize=(13, 5))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.6), wspace=0.35)
    draw_contingency(fig.add_subplot(grid[0, 0]), matrix, run_a, run_b)
    draw_false_positives(fig.add_subplot(grid[0, 1]), fp, run_a, run_b, window)
    fig.suptitle(
        f"{algorithm} · A: {run_a} vs B: {run_b} · {window}s window · {len(tracks)} tracks",
        color=SECONDARY_INK,
        y=1.04,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(
        f"Contingency (rows {run_a} miss/hit, cols {run_b} miss/hit): "
        f"[[{matrix[0, 0]}, {matrix[0, 1]}], [{matrix[1, 0]}, {matrix[1, 1]}]]"
    )
    print(
        f"False positives: {run_a}={fp['fp_a']}/{fp['n_est_a']}, {run_b}={fp['fp_b']}/{fp['n_est_b']}, "
        f"shared={fp['shared']}"
    )
    return out_path


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    raw_config = load_config(args.config)
    segmentation_cfg = parse_segmentation_config(raw_config)
    feature_cfg_raw = raw_config.get("feature_config", raw_config)
    if not isinstance(feature_cfg_raw, dict):
        feature_cfg_raw = {}

    experiment_dir = args.experiment_dir or (
        Path(segmentation_cfg["experiment_dir"]) if segmentation_cfg.get("experiment_dir") else None
    )
    feature_dir = args.feature_dir or (
        Path(segmentation_cfg["feature_dir"]) if segmentation_cfg.get("feature_dir") else None
    )
    if experiment_dir is None or feature_dir is None:
        raise SystemExit("Provide --experiment-dir and --feature-dir (or set them under `segmentation` in --config).")
    experiment_dir = experiment_dir.resolve()
    sections_dir = Path(feature_cfg_raw["sections_dir"]).expanduser() if feature_cfg_raw.get("sections_dir") else None
    label_format = feature_cfg_raw.get("label_format", "rwc_pop")

    run_a, run_b = args.pair
    dir_a = experiment_dir / f"{args.algorithm}__{run_a}"
    dir_b = experiment_dir / f"{args.algorithm}__{run_b}"
    for run_dir in (dir_a, dir_b):
        if not run_dir.is_dir():
            raise SystemExit(f"Run directory not found: {run_dir}")

    track_ids = args.tracks or sorted(
        {path.stem for path in dir_a.glob("*.jams")} & {path.stem for path in dir_b.glob("*.jams")}
    )
    out_dir = args.out_dir or (experiment_dir / "results" / "boundary_pairs" / f"{run_a}_{run_b}_{args.algorithm}")
    tracks_dir = out_dir / "tracks"

    tracks = []
    written = 0
    for track_id in track_ids:
        if not (dir_a / f"{track_id}.jams").exists() or not (dir_b / f"{track_id}.jams").exists():
            print(f"Skipping {track_id}: no JAMS for both runs.")
            continue
        track = load_track_boundaries(
            track_id, args.algorithm, run_a, run_b, args.window, experiment_dir, sections_dir, label_format
        )
        if track is None:
            continue
        # The summary only needs the boundaries, so a track counts even if its SSM figure fails.
        tracks.append(track)
        path = plot_track(track, args.algorithm, run_a, run_b, args.window, experiment_dir, feature_dir, tracks_dir)
        written += path is not None
    print(f"Wrote {written} track figure(s) to {tracks_dir}")

    if not tracks:
        print("No tracks with reference and estimated boundaries; no summary written.")
        return 0
    summary_path = plot_summary(tracks, args.algorithm, run_a, run_b, args.window, out_dir / "summary.png")
    print(f"Wrote summary to {summary_path}")
    meta = {"algorithm": args.algorithm, "run_a": run_a, "run_b": run_b, "window_s": args.window}
    for csv_path in write_csvs(tracks, meta, out_dir):
        print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
