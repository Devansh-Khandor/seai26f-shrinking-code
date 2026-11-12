#!/usr/bin/env python3
"""
Skill 06: Recommendation justification and configuration comparison.

Usage:
    python skill_06.py <csv_path>
"""

import sys
from pathlib import Path

# Reuse helpers from earlier skills
from skill_01 import (
    read_csv,
    compute_statistics,
    best_worst_configs,
    format_config,
    find_distinguishing_settings,
)
from skill_02 import standard_deviation, train_linear_model, predict
from skill_03 import top_percentile_threshold

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def get_setting_differences(conf1, conf2):
    """
    Return a list of tuples (setting, value1, value2) for settings that differ.
    """
    diffs = []
    for key in conf1:
        if key == "Performance-":
            continue
        val1 = conf1[key]
        val2 = conf2.get(key, None)
        if val2 is not None and val1 != val2:
            diffs.append((key, val1, val2))
    return diffs


def explain_recommendation(recommended, rows, performance, coeffs, feature_names):
    """
    Build an explanation for the recommended configuration.
    """
    explanations = []

    # Basic stats
    mean_perf, median_perf, min_perf, max_perf = compute_statistics(performance)
    best_conf, worst_conf = best_worst_configs(rows, performance)

    explanations.append(
        f"Recommended configuration has performance {recommended['Performance-']} ms."
    )
    explanations.append(f"Overall mean performance: {mean_perf:.2f} ms.")
    explanations.append(f"Overall median performance: {median_perf:.2f} ms.")
    explanations.append(f"Best performance recorded: {max_perf:.2f} ms.")
    explanations.append(f"Worst performance recorded: {min_perf:.2f} ms.")

    # Compare with best and worst
    best_perf = float(best_conf["Performance-"])
    worst_perf = float(worst_conf["Performance-"])
    diff_best = float(recommended["Performance-"]) - best_perf
    diff_worst = float(recommended["Performance-"]) - worst_perf
    explanations.append(
        f"Performance relative to best: {diff_best:+.2f} ms."
    )
    explanations.append(
        f"Performance relative to worst: {diff_worst:+.2f} ms."
    )

    # Similar configurations (within one setting difference)
    similar = []
    for conf in rows:
        if conf is recommended:
            continue
        if len(set(conf[k] for k in conf if k != "Performance-") \
               .symmetric_difference(set(recommended[k] for k in recommended if k != "Performance-"))) <= 1:
            similar.append(conf)
    if similar:
        explanations.append(
            f"Found {len(similar)} similar configurations (differ in <=1 setting)."
        )
        # Show one example
        ex = similar[0]
        explanations.append(
            f"Example: {format_config(ex)} with performance {ex['Performance-']} ms."
        )
    else:
        explanations.append("No similar configurations found within one setting difference.")

    # Predictive insight
    pred = predict(recommended, coeffs, feature_names)
    explanations.append(
        f"Linear model predicts performance {pred:.2f} ms for this config."
    )
    explanations.append(
        f"Prediction error: {float(recommended['Performance-']) - pred:.2f} ms."
    )

    return explanations


def explain_difference(conf1, conf2, performance):
    """
    Build an explanation comparing two configurations.
    """
    explanations = []

    # Basic differences
    diffs = get_setting_differences(conf1, conf2)
    if not diffs:
        explanations.append("The two configurations have identical settings.")
        return explanations

    explanations.append(f"Configurations differ in {len(diffs)} settings:")
    for setting, val1, val2 in diffs:
        explanations.append(f" - {setting}: {val1} vs {val2}")

    # Performance comparison
    perf1 = float(conf1["Performance-"])
    perf2 = float(conf2["Performance-"])
    diff = perf2 - perf1
    explanations.append(
        f"Performance difference: {perf2:.2f} - {perf1:.2f} = {diff:+.2f} ms."
    )

    # Predictive comparison using linear model
    coeffs, feature_names = train_linear_model([conf1, conf2], [perf1, perf2])
    pred1 = predict(conf1, coeffs, feature_names)
    pred2 = predict(conf2, coeffs, feature_names)
    pred_diff = pred2 - pred1
    explanations.append(
        f"Predicted difference (model): {pred_diff:+.2f} ms."
    )
    explanations.append(
        f"Prediction error vs actual diff: {diff - pred_diff:+.2f} ms."
    )

    return explanations


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main(csv_path_str):
    csv_path = Path(csv_path_str)
    if not csv_path.is_file():
        print(f"Error: file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)

    rows, performance = read_csv(csv_path)

    # Train linear model for use in explanations
    coeffs, feature_names = train_linear_model(rows, performance)

    # ==== APACHE-11 ====
    print("== APACHE-11 ==")

    # Recommended configuration: top 10% threshold
    threshold = top_percentile_threshold(performance, percent=90)
    top_configs = [row for row, perf in zip(rows, performance) if perf >= threshold]
    if not top_configs:
        print("No configurations meet the top 10% threshold.")
    else:
        recommended = top_configs[0]
        explanations = explain_recommendation(recommended, rows, performance, coeffs, feature_names)
        for line in explanations:
            print(line)
    print()

    # ==== APACHE-12 ====
    print("== APACHE-12 ==")

    # Select two configurations with markedly different assessments
    best_conf, worst_conf = best_worst_configs(rows, performance)
    if best_conf is None or worst_conf is None:
        print("Insufficient data for comparison.")
    else:
        explanations = explain_difference(best_conf, worst_conf, performance)
        for line in explanations:
            print(line)
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
