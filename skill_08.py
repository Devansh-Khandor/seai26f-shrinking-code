#!/usr/bin/env python3
"""
Apache performance diagnostics: targeted improvement recommendations.

This script implements two new analysis points:

- APACHE-15: For a poorly performing configuration, recommend a single
  setting change that would reduce its predicted performance, using only
  historically observed valid configurations as alternatives.
- APACHE-16: Find the smallest number of setting changes required to
  bring a poorly performing configuration up to a specified performance
  target, while ensuring the resulting configuration is valid (i.e.,
  present in the dataset).

Usage:
    python skill_08.py <csv_path>
"""

import sys
import statistics
import skill_01
import skill_02
import skill_07

def load_csv(path: str):
    """Delegate to skill_01's load_csv."""
    return skill_01.load_csv(path)

def feature_names(rows):
    """Return feature names (exclude performance) from rows."""
    return [k for k in rows[0] if k != 'Performance-']

def config_tuple(row, features):
    """Return a tuple representation of a configuration (excluding performance)."""
    return tuple(row[f] for f in features)

def hamming_distance(row_a, row_b, features):
    """Count differing settings between two rows."""
    return sum(1 for f in features if row_a[f] != row_b[f])

def predict_performance(row, beta, features):
    """Predict performance using linear regression coefficients."""
    x = [1] + [float(row[f]) for f in features]
    return sum(b * xi for b, xi in zip(beta, x))

def apache_15(poor_row, rows, features):
    """
    Recommend a single setting change for the given poorly performing row.
    Returns a tuple: (feature, old_value, new_value, predicted_perf, diff)
    or None if no single change improves predicted performance.
    """
    # Prepare regression model
    beta, _, _ = skill_02.train_regression(rows)
    pred_orig = predict_performance(poor_row, beta, features)

    # Build set of existing configuration tuples for quick lookup
    existing = {config_tuple(r, features) for r in rows}

    best_change = None
    best_pred = None

    for f in features:
        # Toggle the feature value
        toggled = poor_row.copy()
        toggled[f] = 1 - poor_row[f]

        if config_tuple(toggled, features) not in existing:
            # Only consider historically observed alternatives
            continue

        pred_toggled = predict_performance(toggled, beta, features)
        if pred_toggled < pred_orig:
            if best_change is None or pred_toggled < best_pred:
                best_change = (f, poor_row[f], 1 - poor_row[f], pred_toggled, pred_orig - pred_toggled)
                best_pred = pred_toggled

    return best_change

def apache_16(poor_row, rows, features, target_perf):
    """
    Find the smallest number of setting changes needed to bring a poorly
    performing configuration up to or above the target performance, using
    only existing configurations. Returns a tuple:
    (num_changes, target_row, target_perf, perf_diff)
    or None if no configuration can reach the target.
    """
    # Identify candidate configurations that meet or exceed the target
    candidates = [r for r in rows if r['Performance-'] >= target_perf]

    # If the poor row already meets the target, return immediately
    if poor_row['Performance-'] >= target_perf:
        return 0, poor_row, poor_row['Performance-'], 0.0

    if not candidates:
        # If no candidate meets the target, choose the best available configuration
        candidates = rows  # fallback to any configuration
        target_perf = max(r['Performance-'] for r in candidates)

    best_config = None
    min_changes = None
    best_perf = None

    for cand in candidates:
        changes = hamming_distance(poor_row, cand, features)
        if min_changes is None or changes < min_changes or (
            changes == min_changes and cand['Performance-'] > best_perf
        ):
            min_changes = changes
            best_config = cand
            best_perf = cand['Performance-']

    if best_config is None:
        return None

    perf_diff = best_perf - target_perf
    return min_changes, best_config, best_perf, perf_diff

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    if not rows:
        print("No data to analyze.")
        sys.exit(0)

    features = feature_names(rows)

    # Identify a poorly performing configuration
    poor_candidates = skill_07.find_unusually_poor(rows)
    if not poor_candidates:
        print("== APACHE-15 ==")
        print("No unusually poor configurations found.")
        print("\n== APACHE-16 ==")
        print("No unusually poor configurations found.")
        return

    poor_row, mean_all, std_all = poor_candidates[0]

    # ===========================
    # APACHE-15: Single-change recommendation
    # ===========================
    print("== APACHE-15 ==")
    change = apache_15(poor_row, rows, features)

    if change is None:
        print("No single setting change found that would improve predicted performance.")
    else:
        f, old_val, new_val, pred_new, delta = change
        print(f"Recommended change: set '{f}' from {old_val} to {new_val}.")
        print(f"Predicted performance after change: {pred_new:.2f} "
              f"(improvement of {delta:.2f}).")
        print(f"Original performance (actual): {poor_row['Performance-']}")

    # ===========================
    # APACHE-16: Minimal change set to reach target
    # ===========================
    print("\n== APACHE-16 ==")
    # Target performance: mean of all configurations
    target = mean_all
    result = apache_16(poor_row, rows, features, target)
    if result is None:
        print("Could not find a configuration that reaches the target performance.")
    else:
        num_changes, target_row, target_perf, perf_diff = result
        print(f"Target performance: {target:.2f} (mean of all configs).")
        print(f"Minimum changes required: {num_changes} setting(s) to reach "
              f"performance {target_perf} (diff {perf_diff:+.2f}).")
        print(f"Resulting configuration: {skill_01.format_config(target_row)}")
        print(f"Actual performance of target configuration: {target_perf}")

if __name__ == "__main__":
    main()
