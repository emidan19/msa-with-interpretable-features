"""Measure how much annotated (GT) structures outscore their random simulated structures.

`distribution_validation.py` writes, per feature, one GT score row per song and
`n_sim` simulated rows per song, in the same song order: GT row i is paired with
simulated rows [i * n_sim, (i + 1) * n_sim). This script uses that pairing to report,
for each song and measure:

    - improvement        GT score minus the mean simulated score
    - rel_improvement    improvement / |mean simulated score|
    - z                  improvement / std of simulated scores
    - win_rate           fraction of simulations the GT beats (ties count 0.5)
    - p_value            empirical one-sided p-value (1 + #{sim >= gt}) / (n_sim + 1)

and aggregates them per feature, adding a paired Wilcoxon signed-rank test of the GT
score against the per-song simulated mean.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MEASURES = ["homogeneity", "repetition", "novelty"]
ALPHA = 0.05


def infer_n_sim(n_gt, n_sim_rows, n_sim=None):
    """Return the number of simulations per GT row, checking it is consistent."""
    if n_gt == 0:
        raise ValueError("Empty GT score file.")
    if n_sim is None:
        if n_sim_rows % n_gt != 0:
            raise ValueError(
                f"{n_sim_rows} simulated rows are not a multiple of {n_gt} GT rows; "
                "pass --n-sim explicitly."
            )
        n_sim = n_sim_rows // n_gt
    if n_gt * n_sim != n_sim_rows:
        raise ValueError(
            f"Expected {n_gt} x {n_sim} = {n_gt * n_sim} simulated rows, found {n_sim_rows}."
        )
    return n_sim


def song_improvement(gt_value, sim_values):
    """Compare one GT score against the scores of its own simulations."""
    sim_values = np.asarray(sim_values, dtype=float)
    sim_values = sim_values[np.isfinite(sim_values)]

    nan_row = {
        "gt": gt_value,
        "sim_mean": np.nan,
        "sim_std": np.nan,
        "improvement": np.nan,
        "rel_improvement": np.nan,
        "z": np.nan,
        "win_rate": np.nan,
        "p_value": np.nan,
        "n_valid_sim": sim_values.size,
    }
    if not np.isfinite(gt_value) or sim_values.size == 0:
        return nan_row

    sim_mean = sim_values.mean()
    sim_std = sim_values.std(ddof=1) if sim_values.size > 1 else np.nan
    improvement = gt_value - sim_mean

    wins = np.sum(gt_value > sim_values) + 0.5 * np.sum(gt_value == sim_values)

    return {
        "gt": gt_value,
        "sim_mean": sim_mean,
        "sim_std": sim_std,
        "improvement": improvement,
        "rel_improvement": improvement / abs(sim_mean) if sim_mean != 0 else np.nan,
        "z": improvement / sim_std if sim_std and np.isfinite(sim_std) and sim_std > 0 else np.nan,
        "win_rate": wins / sim_values.size,
        "p_value": (1 + np.sum(sim_values >= gt_value)) / (sim_values.size + 1),
        "n_valid_sim": sim_values.size,
    }


def per_song_table(df_gt, df_sim, n_sim):
    """Build a long table with one row per (song, measure)."""
    rows = []
    for measure in MEASURES:
        gt_values = df_gt[measure].to_numpy(dtype=float)
        sim_values = df_sim[measure].to_numpy(dtype=float).reshape(len(gt_values), n_sim)

        for song_idx, (gt_value, sims) in enumerate(zip(gt_values, sim_values)):
            rows.append(
                {
                    "song_idx": song_idx,
                    "measure": measure,
                    **song_improvement(gt_value, sims),
                }
            )
    return pd.DataFrame(rows)


def summarize(per_song, feature):
    """Aggregate per-song improvements into one row per measure."""
    rows = []
    for measure in MEASURES:
        df = per_song[per_song["measure"] == measure].dropna(subset=["improvement"])
        n_songs = len(df)

        row = {"feature": feature, "measure": measure, "n_songs": n_songs}
        if n_songs == 0:
            rows.append(row)
            continue

        # Paired test across songs: GT score vs. the mean of its own simulations.
        if n_songs > 1 and np.any(df["improvement"] != 0):
            wilcoxon_p = wilcoxon(df["gt"], df["sim_mean"], alternative="greater").pvalue
        else:
            wilcoxon_p = np.nan

        row.update(
            {
                "gt_mean": df["gt"].mean(),
                "sim_mean": df["sim_mean"].mean(),
                "improvement_mean": df["improvement"].mean(),
                "improvement_median": df["improvement"].median(),
                "rel_improvement_median": df["rel_improvement"].median(),
                "z_median": df["z"].median(),
                # How often the annotated structure scores higher than a random one.
                "win_rate_mean": df["win_rate"].mean(),
                "songs_gt_above_sim_mean": (df["improvement"] > 0).mean(),
                "songs_gt_above_all_sims": (df["win_rate"] == 1.0).mean(),
                f"songs_p<{ALPHA}": (df["p_value"] < ALPHA).mean(),
                "wilcoxon_p": wilcoxon_p,
            }
        )
        rows.append(row)
    return rows


def compute_improvement_for_folder(folder, n_sim=None):

    gt_files = sorted(glob.glob(os.path.join(folder, "ssm_simulation_*_gt_scores.csv")))

    if not gt_files:
        raise SystemExit(f"No ssm_simulation_*_gt_scores.csv files found in {folder}")

    summary_rows = []
    per_song_tables = []

    for gt_file in gt_files:

        sim_file = gt_file.replace("_gt_scores.csv", "_sim_scores.csv")

        if not os.path.exists(sim_file):
            print(f"Not found: {os.path.basename(sim_file)}")
            continue

        feature = os.path.basename(gt_file)
        feature = feature.replace("ssm_simulation_", "")
        feature = feature.replace("_gt_scores.csv", "")

        df_gt = pd.read_csv(gt_file)
        df_sim = pd.read_csv(sim_file)

        feature_n_sim = infer_n_sim(len(df_gt), len(df_sim), n_sim)

        per_song = per_song_table(df_gt, df_sim, feature_n_sim)
        per_song.insert(0, "feature", feature)
        per_song_tables.append(per_song)

        summary_rows.extend(summarize(per_song, feature))

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    summary = pd.DataFrame(summary_rows)
    summary_file = os.path.join(folder, "improvement_summary.csv")
    summary.to_csv(summary_file, index=False)

    per_song_all = pd.concat(per_song_tables, ignore_index=True)
    per_song_file = os.path.join(folder, "improvement_per_song.csv")
    per_song_all.to_csv(per_song_file, index=False)

    # -----------------------------------------------------
    # Show results
    # -----------------------------------------------------

    display_columns = [
        "feature",
        "n_songs",
        "improvement_mean",
        "rel_improvement_median",
        "z_median",
        "win_rate_mean",
        "songs_gt_above_sim_mean",
        f"songs_p<{ALPHA}",
        "wilcoxon_p",
    ]

    for measure in MEASURES:
        print(f"\n{measure.title()} (GT vs. simulated structures):")
        print(
            summary.loc[summary["measure"] == measure, display_columns].to_string(
                index=False,
                float_format=lambda x: f"{x:.3g}",
            )
        )

    print(f"\nFiles saved to:")
    print(summary_file)
    print(per_song_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "validation",
        help="Directory with the ssm_simulation_*_gt_scores.csv / *_sim_scores.csv "
             "files written by distribution_validation.py",
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=None,
        help="Simulations per GT row (default: inferred as #sim rows / #GT rows)",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {args.results_dir}")

    compute_improvement_for_folder(str(args.results_dir), n_sim=args.n_sim)


if __name__ == "__main__":
    main()
