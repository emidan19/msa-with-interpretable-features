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
from feature_types import build_ssm
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


def _build_region_masks(ssm_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sections, repetitions = extract_ssm_regions(ssm_gt)
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

    homogeneity = compute_homogeneity_scores(ssm, ssm_gt)
    repetition = compute_repetition_scores(ssm, ssm_gt)
    novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=3, L=16, var=0.5, exclude=True)

    section_mask, repetition_mask, background_mask = _build_region_masks(ssm_gt)
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
        song_name = npz_path.stem.replace("_features", "")
        for feature_name in list_feature_names(bundle):
            for similarity_type in similarity_types:
                result = evaluate_feature_similarity(
                    feature_name,
                    bundle,
                    ssm_gt,
                    similarity_type=similarity_type,
                    gamma=gamma,
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

        fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(similarity_types) + 2.0), 4.5))
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
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"{metric_name.replace('_', ' ').title()} distribution by similarity")
        fig.tight_layout()
        violin_path = output_dir / f"{metric_name}_violin.png"
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
        sections, repetitions = extract_ssm_regions(ssm_gt)
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

                novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=3, L=16, var=0.5, exclude=True)
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


def _choose_primary_beat_feature(bundle: dict) -> str | None:
    keys = [k for k in bundle.keys()]
    beat_keys = [k for k in keys if k.startswith("beat_")]
    if not beat_keys:
        return None
    # prefer beat_stm or beat_cens
    for pref in ("beat_stm", "beat_cens", "beat_mfcc"):
        if pref in beat_keys:
            return pref
    return beat_keys[0]


def save_combined_ssm_svgs(npz_paths: list[Path], output_dir: Path, similarity_types: tuple[str, ...], gamma: float | None = None) -> list[Path]:
    """For each song bundle, save a single SVG with one column per similarity type.

    Each column shows the SSM (with section/repetition boxes) and the novelty response below.
    File is saved as <song>_SSMs.svg inside output_dir/SSMs/.
    """
    saved: list[Path] = []
    for npz_path in npz_paths:
        bundle = load_feature_bundle(npz_path)
        ssm_gt = _build_gt_matrix(bundle)
        sections, repetitions = extract_ssm_regions(ssm_gt)
        song_name = npz_path.stem.replace("_features", "")
        target_song_dir = output_dir / "SSMs"
        target_song_dir.mkdir(parents=True, exist_ok=True)

        primary_key = _choose_primary_beat_feature(bundle)
        if primary_key is None:
            continue
        features = np.asarray(bundle[primary_key], dtype=float)

        n = len(similarity_types)
        fig, axes = plt.subplots(2, n, figsize=(4 * max(1, n), 6), constrained_layout=True)
        if n == 1:
            axes = axes.reshape(2, 1)

        for i, similarity_type in enumerate(similarity_types):
            try:
                ssm = np.abs(build_ssm(features, similarity_type=similarity_type, gamma=gamma))
            except Exception:
                axes[0, i].text(0.5, 0.5, f"Failed: {similarity_type}", ha="center", va="center")
                axes[1, i].text(0.5, 0.5, "No novelty", ha="center", va="center")
                continue

            vmin, vmax = _finite_limits(ssm)
            axes[0, i].imshow(ssm, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
            _plot_region_boxes(axes[0, i], sections, "cyan", "section")
            _plot_region_boxes(axes[0, i], repetitions, "lime", "repetition")
            axes[0, i].set_title(f"{similarity_type}")
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

            novelty = compute_novelty_scores(ssm, ssm_gt, boundary_band_width=3, L=16, var=0.5, exclude=True)
            novelty_response = novelty.get("novelty_response")
            if novelty_response is not None:
                axes[1, i].plot(novelty_response, color="tab:purple", linewidth=1.2)
            else:
                axes[1, i].text(0.5, 0.5, "No novelty response", ha="center", va="center")
            axes[1, i].set_xlabel("Frame / beat index")
            axes[1, i].set_ylabel("Novelty")
            axes[1, i].grid(alpha=0.2)

        out_path = target_song_dir / f"{song_name}_SSMs.svg"
        fig.suptitle(f"{song_name} - primary:{primary_key}")
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
    # Save combined per-song SSM SVGs (one column per similarity type)
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
