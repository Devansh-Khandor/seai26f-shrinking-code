#!/usr/bin/env python3
"""
Skill 07: Efficient discovery of low‑performance configurations and budget‑aware
evaluation ordering.

Usage:
    python skill_07.py <csv_path>
"""

import sys
from pathlib import Path

# Re‑use helper functions from earlier skills
from skill_01 import read_csv, compute_statistics, top_percentile_threshold, format_config
from skill_02 import train_linear_model, predict

# ----------------------------------------------------------------------
# APACHE-13: Find a low‑performance configuration using minimal evaluations
# ----------------------------------------------------------------------
def discover_low_perf(rows, performance):
    """
    Use the linear model to predict performance for all rows,
    then evaluate them in ascending predicted order until a
    configuration with actual performance <= bottom‑10% threshold
    is found.

    Returns a tuple:
        (found_conf, found_perf, eval_count)
    If no such configuration exists, returns (None, None, eval_count).
    """
    if not rows:
        return None, None, 0

    # Train model
    coeffs, feature_names = train_linear_model(rows, performance)

    # Compute predictions
    preds = [predict(r, coeffs, feature_names) for r in rows]

    # Bottom‑10% threshold
    bottom_threshold = top_percentile_threshold(performance, percent=10)

    # Sort indices by predicted ascending (lowest predicted first)
    sorted_indices = sorted(range(len(rows)), key=lambda i: preds[i])

    eval_count = 0
    for idx in sorted_indices:
        eval_count += 1
        actual = performance[idx]
        if actual <= bottom_threshold:
            return rows[idx], actual, eval_count

    return None, None, eval_count


# ----------------------------------------------------------------------
# APACHE-14: Recommend evaluation order for a limited budget
# ----------------------------------------------------------------------
def recommend_evaluation_order(rows, performance, top_n=5):
    """
    Return a list of (config, predicted_perf) tuples sorted by predicted
    performance ascending (i.e., best low values first). The list contains
    at most top_n entries.
    """
    if not rows:
        return []

    # Train model
    coeffs, feature_names = train_linear_model(rows, performance)

    # Predict
    preds = [predict(r, coeffs, feature_names) for r in rows]

    # Sort by predicted performance
    sorted_configs = sorted(zip(rows, preds), key=lambda x: x[1])

    # Take top_n
    return sorted_configs[:top_n]


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main(csv_path_str):
    csv_path = Path(csv_path_str)
    if not csv_path.is_file():
        print(f"Error: file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)

    rows, performance = read_csv(csv_path)

    # ---- APACHE-13 ----
    print("== APACHE-13 ==")
    conf, perf, evals = discover_low_perf(rows, performance)
    if conf is None:
        print(f"No configuration with performance <= bottom‑10% threshold "
              f"found after evaluating {evals} configurations.")
    else:
        print(f"Found low‑performance configuration after evaluating {evals} configurations:")
        print(f"  Config: {format_config(conf)}")
        print(f"  Actual performance: {perf:.2f} ms")
    print()

    # ---- APACHE-14 ----
    print("== APACHE-14 ==")
    recommendations = recommend_evaluation_order(rows, performance, top_n=5)
    if not recommendations:
        print("No configurations available to recommend.")
    else:
        print(f"Recommend evaluating the following configurations first (top 5 predicted low values):")
        for i, (conf, pred) in enumerate(recommendations, 1):
            print(f"  {i}. Predicted perf: {pred:.2f} ms | Config: {format_config(conf)}")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
