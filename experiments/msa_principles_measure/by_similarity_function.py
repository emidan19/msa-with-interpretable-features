"""Compare MSA principle metrics across similarity functions over precomputed feature bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_loading import load_runtime_config
from feature_types import build_section_segment_ids, build_ssm
from msa_principles import (
    compute_homogeneity_scores,
    compute_novelty_scores,
    compute_repetition_scores,
    extract_ssm_regions,
)

SIMILARITY_TYPES = ("cosine", "autocorrelation", "rbf")
METRIC_NAMES = ("homogeneity", "repetition", "novelty")
REGION_SCORE_NAMES = ("section_mean", "repetition_mean", "background_mean")


def resolve_project_dir(start_dir: Path | None = None) -> Path:
    candidates = []
    if start_dir is not None:
        candidates.append(start_dir.resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(PROJECT_DIR)
    for candidate in candidates:
        if (candidate / "pipeline.py").exists():
            return candidate
    return PROJECT_DIR


def discover_feature_bundles(search_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if search_root is not None:
        candidates.append(search_root.expanduser().resolve())
    else:
        candidates.extend(
            [
                PROJECT_DIR / "feature_outputs",
                PROJECT_DIR / "outputs",
                PROJECT_DIR / "out",
                PROJECT_DIR,
            ]
        )

    for root in candidates:
        if root.is_file():
            return [root]
        if not root.exists():
            continue

        matches = sorted(root.glob("*_features.npz"))
        if matches:
            return matches

        for child in sorted(root.iterdir()):
            if child.is_dir():
                matches = sorted(child.glob("*_features.npz"))
                if matches:
                    return matches

    return []


def load_feature_bundle(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def list_feature_names(bundle: dict[str, Any]) -> list[str]:
    feature_names: list[str] = []
    for key in sorted(bundle):
        if not key.startswith("beat_"):
            continue
        value = np.asarray(bundle[key])
        if value.ndim == 2:
            feature_names.append(key[5:])
    return feature_names


def _build_gt_matrix(bundle: dict[str, Any]) -> np.ndarray:
    if "ssm_sections" not in bundle:
        raise KeyError("The feature bundle does not contain an `ssm_sections` entry.")
    return (np.asarray(bundle["ssm_sections"]) > 0).astype(np.uint8)


def _build_segment_ids(bundle: dict[str, Any]) -> np.ndarray | None:
    """Recover per-frame segment instance ids from the raw annotation, if saved.

    `ssm_sections` only encodes label equality, so two temporally adjacent
    segments sharing the same normalized label (very common: repeated verse
    or chorus) are pixel-indistinguishable from a single larger section. The
    raw `sections` list (distinct per annotated segment) plus `beat_boundaries`
    lets us recover that distinction; see `extract_ssm_regions`'s `segment_ids`.
    """
    if "sections" not in bundle or "beat_boundaries" not in bundle:
        return None
    sections = [(float(start), float(end), str(label)) for start, end, label in bundle["sections"]]
    return build_section_segment_ids(sections, np.asarray(bundle["beat_boundaries"]))


def _build_region_masks(ssm_gt: np.ndarray, segment_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sections, repetitions = extract_ssm_regions(ssm_gt, segment_ids=segment_ids)
    section_mask = np.zeros_like(ssm_gt, dtype=bool)
    repetition_mask = np.zeros_like(ssm_gt, dtype=bool)
    for region in sections:
        section_mask[tuple(region.pixels.T)] = True
    for region in repetitions:
        repetition_mask[tuple(region.pixels.T)] = True
    background_mask = ~ssm_gt.astype(bool)
    return section_mask, repetition_mask, background_mask


def evaluate_feature_similarity(
    feature_name: str,
    bundle: dict[str, Any],
    ssm_gt: np.ndarray,
    similarity_type: str = "cosine",
    gamma: float | None = None,
    segment_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    beat_key = f"beat_{feature_name}"
    if beat_key not in bundle:
        raise KeyError(f"Feature {feature_name!r} is missing from the bundle.")

    feature_matrix = np.asarray(bundle[beat_key], dtype=float)
    if feature_matrix.ndim != 2:
        raise ValueError(f"Feature {feature_name!r} must be a two-dimensional array.")

    ssm = np.abs(build_ssm(feature_matrix, similarity_type=similarity_type, gamma=gamma))
    if ssm.shape != ssm_gt.shape:
        raise ValueError(
            f"SSM shape mismatch for feature {feature_name!r}: got {ssm.shape}, expected {ssm_gt.shape}."
        )

    homogeneity = compute_homogeneity_scores(ssm, ssm_gt, segment_ids=segment_ids)
    repetition = compute_repetition_scores(ssm, ssm_gt, segment_ids=segment_ids)
    novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=3, L=16, var=0.5, exclude=True, segment_ids=segment_ids)

    section_mask, repetition_mask, background_mask = _build_region_masks(ssm_gt, segment_ids=segment_ids)
    section_mean = float(np.mean(ssm[section_mask])) if np.any(section_mask) else float("nan")
    repetition_mean = float(np.mean(ssm[repetition_mask])) if np.any(repetition_mask) else float("nan")
    background_mean = float(np.mean(ssm[background_mask])) if np.any(background_mask) else float("nan")
    global_quality = (
        0.5 * (section_mean + repetition_mean) - background_mean
        if np.isfinite(section_mean) and np.isfinite(repetition_mean) and np.isfinite(background_mean)
        else float("nan")
    )

    return {
        "feature": feature_name,
        "similarity_type": similarity_type,
        "metrics": {
            "homogeneity": homogeneity,
            "repetition": repetition,
            "novelty": novelty,
        },
        "region_scores": {
            "section_mean": section_mean,
            "repetition_mean": repetition_mean,
            "background_mean": background_mean,
        },
        "global_quality": global_quality,
    }


def build_comparison_table(
    npz_paths: list[Path],
    similarity_types: tuple[str, ...] = SIMILARITY_TYPES,
    gamma: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for npz_path in npz_paths:
        bundle = load_feature_bundle(npz_path)
        ssm_gt = _build_gt_matrix(bundle)
        segment_ids = _build_segment_ids(bundle)
        song_name = npz_path.stem.replace("_features", "")
        for feature_name in list_feature_names(bundle):
            for similarity_type in similarity_types:
                result = evaluate_feature_similarity(
                    feature_name,
                    bundle,
                    ssm_gt,
                    similarity_type=similarity_type,
                    gamma=gamma,
                    segment_ids=segment_ids,
                )
                rows.append(
                    {
                        "song": song_name,
                        "bundle": npz_path.name,
                        "feature": result["feature"],
                        "similarity_type": result["similarity_type"],
                        "homogeneity": result["metrics"]["homogeneity"]["global"],
                        "repetition": result["metrics"]["repetition"]["global"],
                        "novelty": result["metrics"]["novelty"]["global"],
                        "global_quality": result["global_quality"],
                        **result["region_scores"],
                    }
                )
    return pd.DataFrame(rows)


def save_global_comparison_plots(results_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    features = sorted(results_df["feature"].dropna().unique().tolist())
    similarity_types = sorted(results_df["similarity_type"].dropna().unique().tolist())

    for metric_name in METRIC_NAMES:
        summary = results_df.pivot_table(index="feature", columns="similarity_type", values=metric_name, aggfunc="mean")
        stds = results_df.pivot_table(index="feature", columns="similarity_type", values=metric_name, aggfunc="std").fillna(0.0)
        summary = summary.reindex(index=features, columns=similarity_types)
        stds = stds.reindex(index=features, columns=similarity_types)

        fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(features) + 2.0), 4.5))
        x = np.arange(len(features))
        bar_width = 0.22
        offsets = np.linspace(-bar_width * (len(similarity_types) - 1) / 2.0, bar_width * (len(similarity_types) - 1) / 2.0, len(similarity_types))
        for idx, similarity_type in enumerate(similarity_types):
            values = summary[similarity_type].to_numpy(dtype=float)
            errors = stds[similarity_type].to_numpy(dtype=float)
            ax.bar(x + offsets[idx], values, width=bar_width, yerr=errors, label=similarity_type, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"{metric_name.replace('_', ' ').title()} by feature and similarity")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"{metric_name}_global_comparison.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    fig, axes = plt.subplots(
        1,
        len(METRIC_NAMES),
        figsize=(max(6.0, 1.2 * len(similarity_types) + 2.0) * len(METRIC_NAMES), 4.5),
        sharey=True,
    )
    if len(METRIC_NAMES) == 1:
        axes = [axes]
    for ax, metric_name in zip(axes, METRIC_NAMES):
        values_by_similarity = []
        for similarity_type in similarity_types:
            values = results_df.loc[results_df["similarity_type"] == similarity_type, metric_name].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                finite_values = np.array([0.0], dtype=float)
            values_by_similarity.append(finite_values)
        parts = ax.violinplot(values_by_similarity, showmeans=True, showmedians=True)
        for body, similarity_type in zip(parts["bodies"], similarity_types):
            body.set_facecolor(plt.cm.viridis(0.2 + 0.7 * similarity_types.index(similarity_type) / max(1, len(similarity_types) - 1)))
            body.set_alpha(0.85)
        ax.set_xticks(range(1, len(similarity_types) + 1))
        ax.set_xticklabels(similarity_types)
        ax.set_title(metric_name.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    fig.suptitle("Metric distributions by similarity function")
    fig.tight_layout()
    violin_path = output_dir / "metrics_violin.png"
    fig.savefig(violin_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(violin_path)

    for region_name in REGION_SCORE_NAMES:
        pivot = results_df.pivot_table(index="feature", columns="similarity_type", values=region_name, aggfunc="mean")
        pivot = pivot.reindex(index=features, columns=similarity_types)
        fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(similarity_types) + 2.0), max(4.0, 0.6 * len(features) + 2.0)))
        image = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(len(similarity_types)))
        ax.set_xticklabels(similarity_types)
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features)
        ax.set_title(f"{region_name.replace('_', ' ').title()} by feature and similarity")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        heatmap_path = output_dir / f"{region_name}_heatmap.png"
        fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(heatmap_path)

    return saved_paths


def _finite_limits(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [1, 99])
    if np.isclose(lo, hi):
        lo, hi = float(values.min()), float(values.max())
    if np.isclose(lo, hi):
        hi = lo + 1.0
    return float(lo), float(hi)


def _plot_region_boxes(ax, regions, color, label):
    import matplotlib.patches as patches

    used_label = False
    for region in regions:
        rows = region.pixels[:, 0]
        cols = region.pixels[:, 1]
        row0, row1 = int(rows.min()), int(rows.max())
        col0, col1 = int(cols.min()), int(cols.max())
        ax.add_patch(
            patches.Rectangle(
                (col0 - 0.5, row0 - 0.5),
                col1 - col0 + 1,
                row1 - row0 + 1,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
                label=label if not used_label else None,
            )
        )
        used_label = True


def save_all_ssm_images(npz_paths: list[Path], output_dir: Path, similarity_types: tuple[str, ...], gamma: float | None = None) -> list[Path]:
    saved: list[Path] = []
    for npz_path in npz_paths:
        bundle = load_feature_bundle(npz_path)
        ssm_gt = _build_gt_matrix(bundle)
        segment_ids = _build_segment_ids(bundle)
        sections, repetitions = extract_ssm_regions(ssm_gt, segment_ids=segment_ids)
        song_name = npz_path.stem.replace("_features", "")
        target_song_dir = output_dir / "SSMs" / song_name
        target_song_dir.mkdir(parents=True, exist_ok=True)

        for feature_name in list_feature_names(bundle):
            beat_key = f"beat_{feature_name}"
            features = np.asarray(bundle[beat_key], dtype=float)
            for similarity_type in similarity_types:
                try:
                    ssm = np.abs(build_ssm(features, similarity_type=similarity_type, gamma=gamma))
                except Exception:
                    continue

                novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=3, L=16, var=0.5, exclude=True, segment_ids=segment_ids)
                novelty_response = novelty.get("novelty_response")

                # Plot SSM (with boxes) and novelty response side-by-side
                fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
                vmin, vmax = _finite_limits(ssm)
                axes[0].imshow(ssm, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
                _plot_region_boxes(axes[0], sections, "cyan", "section")
                _plot_region_boxes(axes[0], repetitions, "lime", "repetition")
                axes[0].set_title(f"{song_name} - {feature_name} - {similarity_type}")
                axes[0].set_xlabel("Frame / beat index")
                axes[0].set_ylabel("Frame / beat index")
                axes[0].legend(loc="upper right")

                if novelty_response is not None:
                    axes[1].plot(novelty_response, color="tab:purple", linewidth=1.5)
                else:
                    axes[1].text(0.5, 0.5, "No novelty response", ha="center", va="center")
                axes[1].set_title("Checkerboard novelty response")
                axes[1].set_xlabel("Frame / beat index")
                axes[1].set_ylabel("Novelty")
                axes[1].grid(alpha=0.25)

                out_path = target_song_dir / f"{feature_name}_{similarity_type}.png"
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                saved.append(out_path)
    return saved


def save_combined_ssm_svgs(npz_paths: list[Path], output_dir: Path, similarity_types: tuple[str, ...], gamma: float | None = None) -> list[Path]:
    """For each song bundle, save a single SVG with all feature SSMs.

    For each feature, one row shows the SSMs with one column per similarity
    type, and the row below shows each corresponding novelty response.
    File is saved as <song>_SSMs.svg inside output_dir/SSMs/.
    """
    if not similarity_types:
        raise ValueError("At least one similarity type is required.")

    saved: list[Path] = []
    for npz_path in npz_paths:
        bundle = load_feature_bundle(npz_path)
        ssm_gt = _build_gt_matrix(bundle)
        segment_ids = _build_segment_ids(bundle)
        sections, repetitions = extract_ssm_regions(ssm_gt, segment_ids=segment_ids)
        song_name = npz_path.stem.replace("_features", "")
        target_song_dir = output_dir / "SSMs"
        target_song_dir.mkdir(parents=True, exist_ok=True)

        feature_names = list_feature_names(bundle)
        if not feature_names:
            continue

        n_cols = len(similarity_types)
        n_rows = 2 * len(feature_names)
        ssm_axis_inches = 4.0
        novelty_axis_inches = 1.5
        height_ratios = [ssm_axis_inches if row_index % 2 == 0 else novelty_axis_inches for row_index in range(n_rows)]
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(ssm_axis_inches * max(1, n_cols), sum(height_ratios) + 1.5),
            constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)

        for feature_index, feature_name in enumerate(feature_names):
            beat_key = f"beat_{feature_name}"
            features = np.asarray(bundle[beat_key], dtype=float)
            ssm_row = 2 * feature_index
            novelty_row = ssm_row + 1

            for col_index, similarity_type in enumerate(similarity_types):
                ssm_ax = axes[ssm_row, col_index]
                novelty_ax = axes[novelty_row, col_index]

                try:
                    ssm = np.abs(build_ssm(features, similarity_type=similarity_type, gamma=gamma))
                except Exception as exc:
                    ssm_ax.text(0.5, 0.5, f"Failed: {similarity_type}\n{exc}", ha="center", va="center")
                    novelty_ax.text(0.5, 0.5, "No novelty", ha="center", va="center")
                    for ax in (ssm_ax, novelty_ax):
                        ax.set_xticks([])
                        ax.set_yticks([])
                    continue

                vmin, vmax = _finite_limits(ssm)
                ssm_ax.imshow(ssm, origin="lower", aspect="equal", cmap="magma", vmin=vmin, vmax=vmax)
                ssm_ax.set_box_aspect(1)
                _plot_region_boxes(ssm_ax, sections, "cyan", "section")
                _plot_region_boxes(ssm_ax, repetitions, "lime", "repetition")
                ssm_ax.set_xticks([])
                ssm_ax.set_yticks([])

                novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=3, L=16, var=0.5, exclude=True, segment_ids=segment_ids)
                novelty_response = novelty.get("novelty_response")
                if novelty_response is not None:
                    novelty_ax.plot(novelty_response, color="tab:purple", linewidth=1.2)
                else:
                    novelty_ax.text(0.5, 0.5, "No novelty response", ha="center", va="center")
                novelty_ax.set_xlabel("Frame / beat index")
                novelty_ax.grid(alpha=0.2)

                if feature_index == 0:
                    ssm_ax.set_title(similarity_type)
                if col_index == 0:
                    ssm_ax.set_ylabel(f"{feature_name}\nSSM")
                    novelty_ax.set_ylabel(f"{feature_name}\nNovelty")

        out_path = target_song_dir / f"{song_name}_SSMs.svg"
        fig.suptitle(f"{song_name} - SSMs by feature and similarity")
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


def run_comparison(
    features_dir: Path | None = None,
    output_dir: Path | None = None,
    similarity_types: tuple[str, ...] = SIMILARITY_TYPES,
    gamma: float | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    project_dir = resolve_project_dir(features_dir)
    search_root = features_dir if features_dir is not None else project_dir
    npz_paths = discover_feature_bundles(search_root)
    if not npz_paths:
        raise FileNotFoundError(
            f"No precomputed feature bundles were found under {search_root}. "
            "Point --features-dir to a directory containing *_features.npz files."
        )

    results_df = build_comparison_table(npz_paths, similarity_types=similarity_types, gamma=gamma)
    if results_df.empty:
        raise ValueError("No comparison rows could be generated from the available bundles.")
    target_dir = output_dir if output_dir is not None else project_dir / "experiments" / "msa_principles_measure" / "similarity_comparison_outputs"
    target_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(target_dir / "similarity_comparison_results.csv", index=False)
    # Save combined per-song SSM SVGs (two rows per feature, one column per similarity type)
    saved_ssm_paths = save_combined_ssm_svgs(npz_paths, target_dir, similarity_types=similarity_types, gamma=gamma)
    print(f"Wrote {len(saved_ssm_paths)} combined SSM SVGs to {target_dir / 'SSMs'}")
    saved_paths = save_global_comparison_plots(results_df, target_dir)
    print(f"Wrote {len(saved_paths)} figures to {target_dir}")
    print(results_df.head().to_string(index=False))
    return results_df, saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=None, help="Directory containing *_features.npz bundles")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write CSV and plots")
    parser.add_argument("--similarity-types", nargs="+", default=list(SIMILARITY_TYPES), help="Similarity functions to compare")
    parser.add_argument("--gamma", type=float, default=None, help="RBF gamma override")
    args = parser.parse_args()

    run_comparison(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        similarity_types=tuple(args.similarity_types),
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
