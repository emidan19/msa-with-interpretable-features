#!/usr/bin/env python3
"""Aggregate and plot segmentation-quality metrics for a structure-baseline experiment.

Walks the `<algorithm>__<ssm_run>/<track_id>.jams` tree produced by
`run_structure_baseline.py` (each JAMS carries the estimated segmentation as its
first annotation and, when available, the ground-truth segmentation as a second
annotation tagged `sandbox.kind == "reference"`). For every track it recomputes,
via `msa_metrics.py`:

* Regularity/balance metrics on both the estimated and the reference
  segmentation, and reports the absolute error between them.
* Boundary detection F1 at 0.5s and 3.0s hit-rate windows (already comparing
  estimated boundaries against the reference ones).
* Normalized mutual information between SSM features/fusions: for each
  reference boundary, a binary hit/miss (1/0) says whether the estimate
  recovered it within the window. For every algorithm and every pair of SSM
  runs (e.g. `musicfm` vs `mfcc`), the NMI between their hit/miss vectors
  (`sklearn.metrics.normalized_mutual_info_score`) measures how much the two
  features agree on *which* reference boundaries they find, independently of
  how many they find.

The experiment directory and the ground-truth label lookup are normally taken
from the same YAML config used by `run_structure_baseline.py` (`--config`):
`segmentation.experiment_dir` and `feature_config.sections_dir` /
`label_format`. Only JAMS files whose track has a reference annotation file
in `sections_dir` are evaluated; the rest are skipped. When a JAMS file doesn't
carry the reference annotation itself (older runs, or `sections_dir` unset at
generation time), the ground truth is loaded from `sections_dir`, and the
row's `reference_source` column records whether it came from the JAMS or from
the config.

Results are written under `<experiment_dir>/results/`: a per-track detailed
CSV, a per-(feature/fusion, algorithm) aggregated CSV, and one grouped
vertical bar chart per metric (one group per SSM feature/fusion, one bar per
segmentation algorithm within the group). The boundary NMI goes to
`boundary_nmi_detailed.csv` (per track and pair), `boundary_nmi_summary.csv`
(per algorithm, window and pair, pooling the hit/miss vectors of all tracks
both runs share) and one feature x feature heatmap per algorithm and window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import normalized_mutual_info_score

from msa_metrics import (
    Segment,
    boundary_hit_vector,
    segmentation_boundary_metrics,
    segmentation_regularity_metrics,
)
from run_structure_baseline import load_config, parse_segmentation_config

REGULARITY_BALANCE_METRICS = (
    "regularity_absolute",
    "balance_absolute",
    "labeled_regularity_absolute",
    "labeled_balance_absolute",
    "adjacent_regularity_absolute",
    "adjacent_balance_absolute",
)
LABELED_METRICS = ("labeled_regularity_absolute", "labeled_balance_absolute")
BOUNDARY_WINDOWS = (0.5, 3.0)
BOUNDARY_F1_METRICS = tuple(f"boundary_f_{str(window).replace('.', '_')}s" for window in BOUNDARY_WINDOWS)

# Validated CVD-safe categorical palette (light-surface steps), assigned to
# algorithms in fixed order so a color always identifies the same algorithm
# across every metric's chart.
CATEGORICAL_PALETTE = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
SECONDARY_INK = "#52514e"
GRIDLINE = "#e1e0d9"

# (title, y-axis label) per metric, so charts read as findings rather than
# raw column names.
METRIC_INFO: dict[str, tuple[str, str]] = {
    "regularity_absolute_error": ("Regularity error vs. ground truth", "|estimated - true| regularity (lower is better)"),
    "balance_absolute_error": ("Balance error vs. ground truth", "|estimated - true| balance (lower is better)"),
    "labeled_regularity_absolute_error": (
        "Labeled regularity error vs. ground truth",
        "|estimated - true| labeled regularity (lower is better)",
    ),
    "labeled_balance_absolute_error": (
        "Labeled balance error vs. ground truth",
        "|estimated - true| labeled balance (lower is better)",
    ),
    "adjacent_regularity_absolute_error": (
        "Adjacent-segment regularity error vs. ground truth",
        "|estimated - true| adjacent regularity (lower is better)",
    ),
    "adjacent_balance_absolute_error": (
        "Adjacent-segment balance error vs. ground truth",
        "|estimated - true| adjacent balance (lower is better)",
    ),
    "boundary_f_0_5s": ("Boundary detection F1 @ 0.5s", "F1 score, 0.5s hit-rate window (higher is better)"),
    "boundary_f_3_0s": ("Boundary detection F1 @ 3.0s", "F1 score, 3.0s hit-rate window (higher is better)"),
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config also used by run_structure_baseline.py. Supplies "
        "`segmentation.experiment_dir` (unless --experiment-dir is given) and the "
        "ground-truth label lookup `feature_config.sections_dir` / `label_format` "
        "(required): only JAMS files with a reference annotation file in "
        "`sections_dir` are evaluated.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Experiment directory produced by run_structure_baseline.py "
        "(contains <algorithm>__<ssm_run>/<track_id>.jams subfolders). "
        "Overrides `segmentation.experiment_dir` from --config.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Where to write the CSV/plots (default: <experiment_dir>/results).",
    )
    return parser.parse_args(argv)


def find_run_dirs(experiment_dir: Path) -> list[tuple[str, str, Path]]:
    """Return (algorithm, ssm_run, dir) for every `<algorithm>__<ssm_run>` subfolder."""
    runs = []
    for path in sorted(experiment_dir.iterdir()):
        if not path.is_dir() or "__" not in path.name:
            continue
        algorithm, ssm_run = path.name.split("__", 1)
        runs.append((algorithm, ssm_run, path))
    return runs


def _segments_from_annotation(annotation: dict[str, Any]) -> list[Segment]:
    segments = []
    for row in annotation.get("data", []):
        value = row.get("value", {})
        label = value.get("label") if isinstance(value, dict) else None
        start = float(row.get("time", 0.0))
        duration = float(row.get("duration", 0.0))
        if duration <= 0.0:
            continue
        segments.append(Segment(start=start, end=start + duration, label=label))
    return segments


def load_jams_segment_sets(jams_path: Path) -> tuple[list[Segment], list[Segment], float | None]:
    """Return (estimated_segments, reference_segments, duration) stored in a JAMS file."""
    payload = json.loads(jams_path.read_text(encoding="utf-8"))
    duration = payload.get("file_metadata", {}).get("duration")
    estimate_segments: list[Segment] = []
    reference_segments: list[Segment] = []
    for annotation in payload.get("annotations", []):
        sandbox = annotation.get("sandbox", {})
        kind = sandbox.get("kind") if isinstance(sandbox, dict) else None
        segments = _segments_from_annotation(annotation)
        if kind == "reference":
            reference_segments = segments
        else:
            estimate_segments = segments
    return estimate_segments, reference_segments, (float(duration) if duration is not None else None)


def load_reference_segments_from_config(
    sections_dir: Path | None,
    label_format: str,
    track_id: str,
    duration: float | None,
) -> list[Segment]:
    """Fallback ground truth lookup for JAMS files with no embedded reference."""
    if sections_dir is None:
        return []
    from data_loading import load_sections

    sections = load_sections(sections_dir, track_id, label_format, duration)
    return [Segment(start=float(start), end=float(end), label=str(label)) for start, end, label in sections]


def _has_repeated_labels(segments: list[Segment]) -> bool:
    labels = [segment.label for segment in segments if segment.label]
    return len(set(labels)) < len(labels)


def evaluate_jams(
    jams_path: Path,
    sections_dir: Path | None,
    label_format: str,
) -> tuple[dict[str, float | str], dict[float, np.ndarray]] | None:
    """Compute regularity/balance error + boundary F1 for one JAMS file.

    Also returns the per-window hit/miss vector over the reference boundaries,
    used for the cross-feature mutual information.
    """
    est_segments, ref_segments, duration = load_jams_segment_sets(jams_path)
    reference_source = "jams"
    if not ref_segments:
        ref_segments = load_reference_segments_from_config(sections_dir, label_format, jams_path.stem, duration)
        reference_source = "config" if ref_segments else "missing"
    if not est_segments or not ref_segments:
        return None

    est_metrics = segmentation_regularity_metrics(est_segments)
    ref_metrics = segmentation_regularity_metrics(ref_segments)
    boundary_metrics = segmentation_boundary_metrics(ref_segments, est_segments, boundary_windows=BOUNDARY_WINDOWS)
    hits = {window: boundary_hit_vector(ref_segments, est_segments, window=window) for window in BOUNDARY_WINDOWS}

    # Labeled metrics only average over same-label segment pairs. A segmentation
    # without repeated labels (boundary-only algorithms label every segment
    # uniquely as S01, S02, ...) has no such pairs and would score a
    # meaningless 0, so those values are reported as NaN instead.
    est_labeled = _has_repeated_labels(est_segments)
    ref_labeled = _has_repeated_labels(ref_segments)
    for metric in LABELED_METRICS:
        if not est_labeled:
            est_metrics[metric] = float("nan")
        if not ref_labeled:
            ref_metrics[metric] = float("nan")

    row: dict[str, float | str] = {"reference_source": reference_source}
    for metric in REGULARITY_BALANCE_METRICS:
        row[f"{metric}_error"] = abs(est_metrics[metric] - ref_metrics[metric])
        row[f"est_{metric}"] = est_metrics[metric]
        row[f"ref_{metric}"] = ref_metrics[metric]
    for metric in BOUNDARY_F1_METRICS:
        row[metric] = boundary_metrics[metric]
    return row, hits


# (algorithm, ssm_run, track_id, window) -> binary hit/miss vector over the reference boundaries.
HitIndex = dict[tuple[str, str, str, float], np.ndarray]


def collect_rows(
    experiment_dir: Path, sections_dir: Path, label_format: str
) -> tuple[pd.DataFrame, HitIndex]:
    from data_loading import sections_path

    rows = []
    hit_index: HitIndex = {}
    skipped = []
    unlabeled = set()
    for algorithm, ssm_run, run_dir in find_run_dirs(experiment_dir):
        for jams_path in sorted(run_dir.glob("*.jams")):
            # Only evaluate tracks with reference labels in sections_dir.
            if not sections_path(sections_dir, jams_path.stem, label_format).exists():
                unlabeled.add(jams_path.stem)
                continue
            result = evaluate_jams(jams_path, sections_dir, label_format)
            if result is None:
                skipped.append(jams_path)
                continue
            metrics, hits = result
            rows.append({"track_id": jams_path.stem, "algorithm": algorithm, "ssm_run": ssm_run, **metrics})
            for window, vector in hits.items():
                hit_index[(algorithm, ssm_run, jams_path.stem, window)] = vector
    if unlabeled:
        print(f"Skipped {len(unlabeled)} track(s) with no reference labels in {sections_dir}.")
    if skipped:
        print(f"Skipped {len(skipped)} JAMS with an empty estimated or reference segmentation.")
    if not rows:
        raise SystemExit(
            "No usable <algorithm>__<ssm_run>/*.jams files with both an estimated and a "
            f"reference segmentation were found under {experiment_dir} for the tracks "
            f"labeled in {sections_dir}."
        )
    df = pd.DataFrame(rows)
    n_from_config = int((df["reference_source"] == "config").sum())
    if n_from_config:
        print(f"Recovered ground truth from --config's sections_dir for {n_from_config} row(s) missing it in the JAMS.")
    return df, hit_index


def boundary_mutual_information(hit_index: HitIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NMI between the hit/miss vectors of every pair of SSM runs, per algorithm and window.

    Returns (detailed, summary). `detailed` has one row per track and pair.
    `summary` pools the vectors of all tracks both runs share into one
    sequence before computing the NMI: a single track has ~10 reference
    boundaries, so per-track NMI is noisy and degenerate whenever a run hits
    all or none of them (sklearn then returns 1.0 or 0.0 regardless of
    agreement). `nmi_track_mean` is kept only as a secondary reference.
    """
    runs_by_algorithm: dict[str, set[str]] = {}
    for algorithm, ssm_run, _, _ in hit_index:
        runs_by_algorithm.setdefault(algorithm, set()).add(ssm_run)

    tracks_by_run: dict[tuple[str, str, float], set[str]] = {}
    for algorithm, ssm_run, track_id, window in hit_index:
        tracks_by_run.setdefault((algorithm, ssm_run, window), set()).add(track_id)

    detailed_rows = []
    summary_rows = []
    for algorithm in sorted(runs_by_algorithm):
        runs = sorted(runs_by_algorithm[algorithm])
        for window in BOUNDARY_WINDOWS:
            for idx, run_a in enumerate(runs):
                for run_b in runs[idx + 1 :]:
                    shared = sorted(
                        tracks_by_run.get((algorithm, run_a, window), set())
                        & tracks_by_run.get((algorithm, run_b, window), set())
                    )
                    pooled_a, pooled_b, track_nmis = [], [], []
                    for track_id in shared:
                        hits_a = hit_index[(algorithm, run_a, track_id, window)]
                        hits_b = hit_index[(algorithm, run_b, track_id, window)]
                        if hits_a.shape != hits_b.shape:
                            # Different reference boundaries (e.g. one JAMS embeds
                            # ground truth and the other fell back to the config).
                            print(
                                f"Skipping NMI for {track_id} ({algorithm}: {run_a} vs {run_b}): "
                                "reference boundary counts differ."
                            )
                            continue
                        if hits_a.size == 0:
                            # No internal reference boundaries (single-segment ground truth).
                            continue
                        nmi = float(normalized_mutual_info_score(hits_a, hits_b))
                        track_nmis.append(nmi)
                        pooled_a.append(hits_a)
                        pooled_b.append(hits_b)
                        detailed_rows.append(
                            {
                                "algorithm": algorithm,
                                "window_s": window,
                                "ssm_run_a": run_a,
                                "ssm_run_b": run_b,
                                "track_id": track_id,
                                "n_boundaries": len(hits_a),
                                "hit_rate_a": float(hits_a.mean()),
                                "hit_rate_b": float(hits_b.mean()),
                                "agreement": float((hits_a == hits_b).mean()),
                                "nmi": nmi,
                            }
                        )
                    if not pooled_a:
                        continue
                    all_a = np.concatenate(pooled_a)
                    all_b = np.concatenate(pooled_b)
                    summary_rows.append(
                        {
                            "algorithm": algorithm,
                            "window_s": window,
                            "ssm_run_a": run_a,
                            "ssm_run_b": run_b,
                            "n_tracks": len(pooled_a),
                            "n_boundaries": len(all_a),
                            "hit_rate_a": float(all_a.mean()),
                            "hit_rate_b": float(all_b.mean()),
                            "agreement": float((all_a == all_b).mean()),
                            "nmi_pooled": float(normalized_mutual_info_score(all_a, all_b)),
                            "nmi_track_mean": float(np.mean(track_nmis)),
                        }
                    )
    return pd.DataFrame(detailed_rows), pd.DataFrame(summary_rows)


def metric_columns(detailed_df: pd.DataFrame) -> list[str]:
    error_cols = [f"{metric}_error" for metric in REGULARITY_BALANCE_METRICS if f"{metric}_error" in detailed_df.columns]
    f1_cols = [metric for metric in BOUNDARY_F1_METRICS if metric in detailed_df.columns]
    return error_cols + f1_cols


def aggregate(detailed_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    return detailed_df.groupby(["ssm_run", "algorithm"])[metrics].mean().reset_index()


def algorithm_colors(algorithms: list[str]) -> dict[str, str]:
    if len(algorithms) > len(CATEGORICAL_PALETTE):
        raise SystemExit(
            f"{len(algorithms)} algorithms exceed the {len(CATEGORICAL_PALETTE)}-color categorical "
            "palette; fold some into a comparison group or facet before plotting."
        )
    return {algorithm: CATEGORICAL_PALETTE[idx] for idx, algorithm in enumerate(algorithms)}


def plot_grouped_metric(
    agg_df: pd.DataFrame,
    metric: str,
    algorithms: list[str],
    colors: dict[str, str],
    results_dir: Path,
) -> Path:
    """One group per SSM feature/fusion, one bar per algorithm within the group.

    Grouped (not stacked) because these scores are independent per-algorithm
    values, not parts of a whole: a stacked total would sum to a meaningless
    number.
    """
    n_groups = agg_df["ssm_run"].nunique()
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * n_groups), 5))
    draw_grouped_metric(ax, agg_df, metric, algorithms, colors)
    fig.tight_layout()

    out_path = results_dir / f"{metric}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_grouped_metric(
    ax: plt.Axes,
    agg_df: pd.DataFrame,
    metric: str,
    algorithms: list[str],
    colors: dict[str, str],
) -> None:
    """Draw the grouped bars of `plot_grouped_metric` on an existing axis."""
    pivot = agg_df.pivot(index="ssm_run", columns="algorithm", values=metric).reindex(columns=algorithms)

    n_groups = len(pivot.index)
    n_series = len(algorithms)
    group_positions = np.arange(n_groups)
    bar_width = 0.8 / max(1, n_series)

    top_value = 0.0
    for idx, algorithm in enumerate(algorithms):
        values = pivot[algorithm].to_numpy(dtype=float)
        offsets = group_positions + (idx - (n_series - 1) / 2) * bar_width
        bars = ax.bar(
            offsets,
            values,
            width=bar_width * 0.9,  # thin surface gap between adjacent bars
            label=algorithm,
            color=colors[algorithm],
            zorder=3,
        )
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            top_value = max(top_value, float(value))
            ax.annotate(
                f"{value:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=SECONDARY_INK,  # labels use text ink, never the series color
            )

    title, ylabel = METRIC_INFO.get(metric, (metric, metric))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("SSM feature / fusion")
    ax.set_xticks(group_positions)
    ax.set_xticklabels(pivot.index.astype(str), rotation=30, ha="right")
    ax.set_ylim(0, max(top_value * 1.2, 0.1))
    ax.grid(axis="y", color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if n_series > 1:
        ax.legend(title="algorithm", bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0, frameon=False)


def plot_nmi_heatmap(nmi_summary: pd.DataFrame, algorithm: str, window: float, results_dir: Path) -> Path | None:
    """Symmetric SSM-run x SSM-run matrix of pooled boundary hit/miss NMI."""
    subset = nmi_summary[(nmi_summary["algorithm"] == algorithm) & (nmi_summary["window_s"] == window)]
    if subset.empty:
        return None
    runs = sorted(set(subset["ssm_run_a"]) | set(subset["ssm_run_b"]))
    matrix = pd.DataFrame(np.nan, index=runs, columns=runs)
    for row in subset.itertuples():
        matrix.loc[row.ssm_run_a, row.ssm_run_b] = row.nmi_pooled
        matrix.loc[row.ssm_run_b, row.ssm_run_a] = row.nmi_pooled

    n = len(runs)
    fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * n + 2.5), max(4, 0.9 * n + 1.5)))
    # Single-hue sequential ramp on a fixed [0, 1] scale so heatmaps are
    # comparable across algorithms and windows.
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(n):
        for j in range(n):
            value = matrix.iat[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > 0.6 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(runs, rotation=30, ha="right")
    ax.set_yticklabels(runs)
    ax.set_title(f"Boundary hit/miss NMI between features\n{algorithm}, {window}s window")
    fig.colorbar(image, ax=ax, label="normalized mutual information", fraction=0.046, pad=0.04)
    fig.tight_layout()

    suffix = str(window).replace(".", "_")
    out_path = results_dir / f"boundary_nmi_{algorithm}_{suffix}s.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
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
    if experiment_dir is None:
        raise SystemExit(
            "Provide --experiment-dir, or set `segmentation.experiment_dir` in the YAML passed via --config."
        )
    experiment_dir = experiment_dir.resolve()
    if not experiment_dir.exists():
        raise SystemExit(f"Experiment directory not found: {experiment_dir}")

    sections_dir = Path(feature_cfg_raw["sections_dir"]).expanduser() if feature_cfg_raw.get("sections_dir") else None
    label_format = feature_cfg_raw.get("label_format", "rwc_pop")
    if sections_dir is None:
        raise SystemExit(
            "Set `feature_config.sections_dir` in the YAML passed via --config: only JAMS "
            "files with reference labels in that folder are evaluated."
        )
    if not sections_dir.is_dir():
        raise SystemExit(f"sections_dir not found: {sections_dir}")

    results_dir = args.results_dir or (experiment_dir / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    detailed_df, hit_index = collect_rows(experiment_dir, sections_dir, label_format)
    metrics = metric_columns(detailed_df)
    if not metrics:
        raise SystemExit(f"None of the expected regularity/balance/boundary-F1 metrics were found under {experiment_dir}.")

    detailed_csv = results_dir / "segmentation_eval_detailed.csv"
    detailed_df.to_csv(detailed_csv, index=False)

    agg_df = aggregate(detailed_df, metrics)
    agg_csv = results_dir / "segmentation_eval_summary.csv"
    agg_df.to_csv(agg_csv, index=False)

    # Fixed algorithm order/colors so a color identifies the same algorithm
    # across every metric's chart.
    algorithms = sorted(detailed_df["algorithm"].unique())
    colors = algorithm_colors(algorithms)
    plot_paths = [plot_grouped_metric(agg_df, metric, algorithms, colors, results_dir) for metric in metrics]

    nmi_detailed, nmi_summary = boundary_mutual_information(hit_index)
    nmi_csvs = []
    if nmi_summary.empty:
        print("Boundary NMI skipped: needs at least two SSM runs sharing tracks under the same algorithm.")
    else:
        nmi_csvs = [results_dir / "boundary_nmi_detailed.csv", results_dir / "boundary_nmi_summary.csv"]
        nmi_detailed.to_csv(nmi_csvs[0], index=False)
        nmi_summary.to_csv(nmi_csvs[1], index=False)
        for algorithm in algorithms:
            for window in BOUNDARY_WINDOWS:
                heatmap = plot_nmi_heatmap(nmi_summary, algorithm, window, results_dir)
                if heatmap is not None:
                    plot_paths.append(heatmap)

    print(f"Rows (track x algorithm x ssm_run): {len(detailed_df)}")
    print(f"Wrote {detailed_csv}")
    print(f"Wrote {agg_csv}")
    for path in nmi_csvs:
        print(f"Wrote {path}")
    for path in plot_paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
