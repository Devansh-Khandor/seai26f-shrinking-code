#!/usr/bin/env python3
"""
Skill 06: Apache performance analysis – recommendation and pairwise explanation
"""

import sys
import numpy as np

# --------------------------------------------------------------------------- #
# Import helpers from earlier skills
# --------------------------------------------------------------------------- #
from skill_01 import (
    read_csv_to_numpy,
    summarize_performance,
    identify_favorable
)
from skill_02 import (
    train_linear_model,
    predict_with_confidence,
    apache_03,
    apache_04,
    find_one_diff_pairs
)

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def get_mean_features(data):
    """
    Return the mean value for each feature (excluding the performance column).
    """
    return np.mean(data[:, :-1], axis=0)

def format_feature_diffs(header, config, mean_vals):
    """
    Return a list of strings describing how the configuration differs from
    the mean configuration.
    """
    diff_lines = []
    for idx, (name, val, mean_val) in enumerate(zip(header[:-1], config, mean_vals)):
        if np.isclose(val, mean_val, atol=0.5):
            continue  # no meaningful difference for binary feature
        diff_lines.append(
            f"  {name}: {val} (mean: {mean_val:.2f})"
        )
    return diff_lines

def explain_feature_contributions(coef, diff_vector):
    """
    For each differing feature, compute its contribution to the performance
    difference based on the model coefficient.
    """
    contributions = []
    for idx, diff in enumerate(diff_vector):
        if diff != 0:
            contrib = coef[idx] * diff
            sign = "+" if contrib >= 0 else "-"
            contributions.append(
                f"  {sign}{abs(contrib):.2f} (feature {idx}, change {diff})"
            )
    return contributions

# --------------------------------------------------------------------------- #
# APACHE-11: Recommendation explanation
# --------------------------------------------------------------------------- #

def apache_11(header, data, model, XtX_inv, resid_var):
    """
    Recommend the configuration with the best performance and explain
    why it is superior by comparing its settings and performance to
    the overall dataset and to the model predictions.
    """
    print("\n== APACHE-11 ==")

    perf = data[:, -1]
    best_idx = np.argmax(perf)
    best_config = data[best_idx, :-1]
    best_perf = perf[best_idx]

    # Predict performance for recommended config
    pred, low, high = predict_with_confidence(
        model, XtX_inv, resid_var, best_config, confidence=0.95
    )

    mean_perf = np.mean(perf)
    max_perf = np.max(perf)
    min_perf = np.min(perf)

    print(f"Recommended configuration index: {best_idx}")
    print(f"Actual performance: {best_perf}")
    print(f"Predicted performance: {pred:.2f} ± {high - pred:.2f} (95% CI: [{low:.2f}, {high:.2f}])")
    print(f"Mean dataset performance: {mean_perf:.2f}")
    print(f"Best performance in data: {max_perf}")
    print(f"Worst performance in data: {min_perf}")

    # Feature comparison to mean configuration
    mean_feats = get_mean_features(data)
    diff_lines = format_feature_diffs(header, best_config, mean_feats)

    print("\nFeature differences relative to the mean configuration:")
    if diff_lines:
        for line in diff_lines:
            print(line)
    else:
        print("  No significant differences from the mean configuration.")

    # Highlight the features that differ from the mean by at least 0.5
    # (useful for binary features)
    print("\nFeatures that are 1 while the mean is below 0.5 (potentially beneficial):")
    beneficial = [
        f"  {header[i]}: {best_config[i]} (mean: {mean_feats[i]:.2f})"
        for i in range(len(best_config))
        if best_config[i] == 1 and mean_feats[i] < 0.5
    ]
    if beneficial:
        for line in beneficial:
            print(line)
    else:
        print("  None")

# --------------------------------------------------------------------------- #
# APACHE-12: Pairwise explanation
# --------------------------------------------------------------------------- #

def apache_12(header, data, model, XtX_inv, resid_var):
    """
    Compare the best and worst configurations, detailing the settings that differ
    and how those differences relate to the recorded and predicted performance
    outcomes.
    """
    print("\n== APACHE-12 ==")

    perf = data[:, -1]
    best_idx = np.argmax(perf)
    worst_idx = np.argmin(perf)

    best_config = data[best_idx, :-1]
    worst_config = data[worst_idx, :-1]

    best_perf = perf[best_idx]
    worst_perf = perf[worst_idx]
    actual_diff = best_perf - worst_perf

    # Model predictions
    best_pred = model.predict(best_config.reshape(1, -1))[0]
    worst_pred = model.predict(worst_config.reshape(1, -1))[0]
    predicted_diff = best_pred - worst_pred

    # Feature differences
    diff_vector = best_config - worst_config  # +1 if best has 1, -1 if worst has 1

    print(f"Best config index: {best_idx} (performance: {best_perf})")
    print(f"Worst config index: {worst_idx} (performance: {worst_perf})")
    print(f"Actual performance difference: {actual_diff:.2f}")
    print(f"Predicted performance difference: {predicted_diff:.2f}")

    # List differing features
    diff_features = []
    for idx, diff in enumerate(diff_vector):
        if diff != 0:
            feature_name = header[idx]
            best_val = best_config[idx]
            worst_val = worst_config[idx]
            diff_features.append(
                f"  {feature_name}: {worst_val} -> {best_val} (diff {diff})"
            )

    print("\nDiffering settings between best and worst configurations:")
    if diff_features:
        for line in diff_features:
            print(line)
    else:
        print("  No differences (identical configurations)")

    # Contribution analysis using model coefficients
    coeffs = model.coef_
    contributions = explain_feature_contributions(coeffs, diff_vector)

    print("\nModel coefficient contributions to the performance difference:")
    if contributions:
        for line in contributions:
            print(line)
    else:
        print("  No contribution (identical coefficients or no differences)")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, data = read_csv_to_numpy(csv_path)

    # Basic summary
    summarize_performance(header, data)
    identify_favorable(header, data)

    # Model training
    model, XtX_inv, resid_var = train_linear_model(data, header)

    # Reference outputs from previous skills
    apache_03(header, data, model, XtX_inv, resid_var)
    apache_04(header, data, model, XtX_inv, resid_var)

    # New requests
    apache_11(header, data, model, XtX_inv, resid_var)
    apache_12(header, data, model, XtX_inv, resid_var)


if __name__ == "__main__":
    main()
