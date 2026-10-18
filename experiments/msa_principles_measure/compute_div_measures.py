"""Compute KL divergences between GT and simulated MSA principle score distributions."""

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, entropy


def kl_divergence_kde(x_gt, x_sim, num_points=1000, eps=1e-12):
    """
    Estimate the KL divergence KL(P||Q) between two continuous
    distributions using KDE.

    Parameters
    ----------
    x_gt : array-like
        Samples from the reference distribution (P).
    x_sim : array-like
        Samples from the distribution to compare (Q).
    num_points : int
        Number of grid points.
    eps : float
        Constant to avoid log(0).

    Returns
    -------
    float
        KL divergence KL(P||Q).
    """

    x_gt = np.asarray(x_gt)
    x_sim = np.asarray(x_sim)

    xmin = min(x_gt.min(), x_sim.min())
    xmax = max(x_gt.max(), x_sim.max())

    # small margin to avoid issues at the edges
    margin = 0.05 * (xmax - xmin + 1e-8)

    grid = np.linspace(xmin - margin, xmax + margin, num_points)

    kde_gt = gaussian_kde(x_gt)
    kde_sim = gaussian_kde(x_sim)

    p = kde_gt(grid)
    q = kde_sim(grid)

    # Normalize to ensure they integrate to 1
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    p /= np.trapezoid(p, grid)
    q /= np.trapezoid(q, grid)

    dx = grid[1] - grid[0]

    return entropy(p * dx, q * dx)


def compute_kl_for_folder(folder):

    gt_files = sorted(glob.glob(os.path.join(folder, "ssm_simulation_*_gt_scores.csv")))

    measures = ["homogeneity", "repetition", "novelty"]

    kl_results = []
    stats_results = []

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

        # -------------------------------------------------
        # KL
        # -------------------------------------------------

        kl_row = {"feature": feature}

        for measure in measures:
            kl_row[measure] = kl_divergence_kde(
                df_gt[measure].dropna(),
                df_sim[measure].dropna()
            )

        kl_results.append(kl_row)

        # -------------------------------------------------
        # GT statistics
        # -------------------------------------------------

        stats_row = {"feature": feature}

        for measure in measures:

            values = df_gt[measure].dropna()

            stats_row[f"{measure}_mean"] = values.mean()
            stats_row[f"{measure}_std"] = values.std()
            stats_row[f"{measure}_median"] = values.median()

        stats_results.append(stats_row)

    # -----------------------------------------------------
    # Save KL
    # -----------------------------------------------------

    kl_results = pd.DataFrame(kl_results)

    kl_file = os.path.join(
        folder,
        "kl_divergences.csv"
    )

    kl_results.to_csv(
        kl_file,
        index=False
    )

    # -----------------------------------------------------
    # Save statistics
    # -----------------------------------------------------

    stats_results = pd.DataFrame(stats_results)

    stats_file = os.path.join(
        folder,
        "gt_statistics.csv"
    )

    stats_results.to_csv(
        stats_file,
        index=False
    )

    # -----------------------------------------------------
    # Show results
    # -----------------------------------------------------

    print("\nKL divergences:")
    print(
        kl_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.2g}"
        )
    )

    print("\nGT statistics:")
    print(
        stats_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print(f"\nFiles saved to:")
    print(kl_file)
    print(stats_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "validation",
        help="Directory with the ssm_simulation_*_gt_scores.csv / *_sim_scores.csv "
             "files written by distribution_validation.py",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {args.results_dir}")

    compute_kl_for_folder(str(args.results_dir))


if __name__ == "__main__":
    main()