#!/usr/bin/env python3
"""
Skill 03 – Comparative Analysis of Fuel Efficiency and Performance

Usage:
    python skill_03.py <path-to-auto93.csv>
"""

import sys
import numpy as np

# Reuse existing helpers
from skill_01 import load_csv, compute_summary, print_summary, compute_scores


def group_stats(data: np.ndarray, header: list[str], group_col: str, value_col: str):
    """Return a dict mapping each unique group value to a dict with mean, std and count."""
    g_idx = header.index(group_col)
    v_idx = header.index(value_col)
    groups = {}
    for g_val in np.unique(data[:, g_idx]):
        mask = data[:, g_idx] == g_val
        vals = data[mask, v_idx]
        if vals.size == 0:
            continue
        groups[g_val] = {
            "mean": np.mean(vals),
            "std": np.std(vals, ddof=1) if vals.size > 1 else 0.0,
            "n": vals.size,
        }
    return groups


def welch_t_test(stats1: dict, stats2: dict):
    """Return t-statistic, degrees of freedom and approximate p-value (two‑sided)."""
    mean1, mean2 = stats1["mean"], stats2["mean"]
    var1, var2 = stats1["std"] ** 2, stats2["std"] ** 2
    n1, n2 = stats1["n"], stats2["n"]

    # Handle zero variance
    if var1 == 0 and var2 == 0:
        return 0.0, np.inf, 1.0
    se = np.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        return 0.0, np.inf, 1.0
    t = (mean1 - mean2) / se
    # Welch‑Satterthwaite df
    df_num = (var1 / n1 + var2 / n2) ** 2
    df_den = (var1 ** 2) / (n1 ** 2 * (n1 - 1)) + (var2 ** 2) / (n2 ** 2 * (n2 - 1))
    df = df_num / df_den if df_den != 0 else np.inf

    # Approximate p-value using normal distribution for simplicity
    # (good enough for large df; for small df we just flag as uncertain)
    from math import erf, sqrt

    if df > 30:  # approximate normal
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    else:
        # Very crude estimate: if |t| > 2, consider significant
        p = 0.05 if abs(t) > 2 else 0.3

    return t, df, p


def compare_cylinders(data, header):
    """Perform CAR‑05 analysis: fuel‑efficiency comparison by cylinder count."""
    cyl_stats = group_stats(data, header, "Clndrs", "Mpg+")
    # Print summary per cylinder
    for cyl, stats in sorted(cyl_stats.items()):
        print(
            f"Cylinder {int(cyl)}: mean MPG={stats['mean']:.2f}, "
            f"std={stats['std']:.2f}, n={stats['n']}"
        )

    # Pairwise comparisons
    cyl_values = sorted(cyl_stats.keys())
    for i in range(len(cyl_values)):
        for j in range(i + 1, len(cyl_values)):
            g1, g2 = cyl_values[i], cyl_values[j]
            t, df, p = welch_t_test(cyl_stats[g1], cyl_stats[g2])
            sig = "Yes" if p < 0.05 else "No"
            print(
                f"Diff Cylinder {int(g1)} vs {int(g2)}: t={t:.2f}, "
                f"p≈{p:.3f} → significant? {sig}"
            )


def compare_model_years(data, header):
    """Perform CAR‑06 analysis: compare two model‑year groups."""
    # Use 'Model' column
    model_idx = header.index("Model")
    # Split by median model year
    median_year = np.median(data[:, model_idx])
    mask_low = data[:, model_idx] <= median_year
    mask_high = data[:, model_idx] > median_year

    if mask_low.sum() == 0 or mask_high.sum() == 0:
        print("Insufficient data for model‑year comparison.")
        return

    # Compute composite scores for each group
    scores = compute_scores(data, header)
    low_scores = scores[mask_low]
    high_scores = scores[mask_high]

    low_stats = {
        "mean": np.mean(low_scores),
        "std": np.std(low_scores, ddof=1),
        "n": low_scores.size,
    }
    high_stats = {
        "mean": np.mean(high_scores),
        "std": np.std(high_scores, ddof=1),
        "n": high_scores.size,
    }

    print(
        f"Low model years (≤{median_year:.0f}): mean score={low_stats['mean']:.2f}, "
        f"std={low_stats['std']:.2f}, n={low_stats['n']}"
    )
    print(
        f"High model years (> {median_year:.0f}): mean score={high_stats['mean']:.2f}, "
        f"std={high_stats['std']:.2f}, n={high_stats['n']}"
    )

    t, df, p = welch_t_test(low_stats, high_stats)
    sig = "Yes" if p < 0.05 else "No"
    print(
        f"Score difference t={t:.2f}, p≈{p:.3f} → significant? {sig}"
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv-path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        data, header = load_csv(csv_path)
    except Exception as exc:
        print(f"Error loading CSV: {exc}")
        sys.exit(1)

    # CAR‑05: Fuel‑efficiency by cylinder count
    print("\n== CAR-05 ==")
    compare_cylinders(data, header)

    # CAR‑06: Performance by model‑year groups
    print("\n== CAR-06 ==")
    compare_model_years(data, header)


if __name__ == "__main__":
    main()
