#!/usr/bin/env python3
"""
Apache performance diagnostics: unexpected performance and outlier configuration patterns.

This script implements two new analysis points:

- APACHE-17: Detect configurations whose recorded performance values are unusually far
  from the predictions made by the linear regression model based on their settings.
- APACHE-18: Flag configurations whose combinations of settings are markedly
  dissimilar to the most similar historical configurations, even when their
  measured performance is not obviously poor.

The analysis reuses functions from the existing skill modules, particularly
skill_01, skill_02, and skill_08.
"""

import sys
import statistics
import skill_01
import skill_02
import skill_08


def load_csv(path: str):
    """Delegate to skill_01's load_csv."""
    return skill_01.load_csv(path)


def find_unexpected_performance(rows):
    """
    Identify rows whose actual performance differs significantly from
    the prediction of the regression model.
    Returns a list of tuples: (row, predicted, residual).
    """
    if not rows:
        return []

    beta, feature_names, predictions = skill_02.train_regression(rows)
    actuals = [row['Performance-'] for row in rows]
    residuals = [act - pred for act, pred in zip(actuals, predictions)]

    mean_residual = statistics.mean(residuals) if residuals else 0.0
    std_residual = statistics.stdev(residuals) if len(residuals) > 1 else 0.0

    # We flag any configuration whose absolute residual exceeds one stddev
    # above the mean (which is close to 0 for regression residuals).
    threshold = mean_residual + std_residual

    flagged = []
    for row, pred, resid in zip(rows, predictions, residuals):
        if abs(resid) > threshold:
            flagged.append((row, pred, resid))
    return flagged


def find_sparsely_represented(rows, distance_threshold=3):
    """
    Find configurations that are far from all other configurations in terms
    of hamming distance, yet whose performance is not obviously poor.
    Returns a list of tuples: (row, nearest_row, min_distance).
    """
    if not rows:
        return []

    features = skill_08.feature_names(rows)
    perf_mean = statistics.mean([r['Performance-'] for r in rows])
    perf_std = statistics.stdev([r['Performance-'] for r in rows]) if len(rows) > 1 else 0.0
    perf_cutoff = perf_mean - perf_std  # performance not obviously poor

    flagged = []
    for i, row in enumerate(rows):
        min_dist = None
        nearest = None
        for j, other in enumerate(rows):
            if i == j:
                continue
            dist = skill_08.hamming_distance(row, other, features)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                nearest = other
        if min_dist is not None and min_dist >= distance_threshold and row['Performance-'] >= perf_cutoff:
            flagged.append((row, nearest, min_dist))
    return flagged


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    # ---------- APACHE-17 ----------
    print("== APACHE-17 ==")
    unexpected = find_unexpected_performance(rows)
    if not unexpected:
        print("No configurations with unexpected performance found.")
    else:
        print(f"Found {len(unexpected)} configuration(s) with significant prediction errors:")
        for row, pred, resid in unexpected[:min(3, len(unexpected))]:
            print(f"- Config: {skill_01.format_config(row)}")
            print(f"  Actual Performance: {row['Performance-']}")
            print(f"  Predicted Performance: {pred:.2f}")
            print(f"  Residual: {resid:+.2f}")
        if len(unexpected) > 3:
            print(f"... and {len(unexpected)-3} more.")

    # ---------- APACHE-18 ----------
    print("\n== APACHE-18 ==")
    sparse = find_sparsely_represented(rows, distance_threshold=3)
    if not sparse:
        print("No configurations with substantially dissimilar setting patterns found.")
    else:
        print(f"Found {len(sparse)} configuration(s) with large hamming distance to all others:")
        for row, nearest, dist in sparse[:min(3, len(sparse))]:
            print(f"- Config: {skill_01.format_config(row)}")
            print(f"  Hamming distance to nearest neighbor: {dist}")
            print(f"  Nearest config: {skill_01.format_config(nearest)}")
            print(f"  Performance: {row['Performance-']}")
        if len(sparse) > 3:
            print(f"... and {len(sparse)-3} more.")


if __name__ == "__main__":
    main()
