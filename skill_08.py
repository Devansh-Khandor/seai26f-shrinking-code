#!/usr/bin/env python3
"""
Skill 08: Fine‑grained recommendation and minimal‑change search for poor configurations.

Usage:
    python skill_08.py <csv_path>
"""

import sys
from pathlib import Path

# Reuse helpers from earlier skills
from skill_01 import (
    read_csv,
    compute_statistics,
    format_config,
    best_worst_configs,
)
from skill_02 import (
    train_linear_model,
    predict,
)
from skill_05 import find_poor_configs

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def diff_count(conf1, conf2):
    """
    Return the number of settings that differ between two configuration dicts.
    """
    count = 0
    for key in conf1:
        if key == "Performance-":
            continue
        if conf1[key] != conf2.get(key):
            count += 1
    return count


def diff_setting(conf1, conf2):
    """
    Return the single setting that differs between two configurations.
    Assumes they differ in exactly one setting.
    """
    for key in conf1:
        if key == "Performance-":
            continue
        if conf1[key] != conf2.get(key):
            return key, conf1[key], conf2[key]
    return None, None, None


# ----------------------------------------------------------------------
# APACHE-15: Small change recommendation
# ----------------------------------------------------------------------
def recommend_small_change(rows, performance):
    """
    For a poorly performing configuration, recommend a single setting change
    that reduces the predicted performance, using historically valid configurations.
    """
    # Find poorly performing configurations (bottom 10%)
    poor_list = find_poor_configs(rows, performance, top_percentile_threshold=10)
    if not poor_list:
        return None, None, None, None

    # Take the first poorly performing configuration
    idx, poor_conf, poor_perf = poor_list[0]
    _, _, _, best_perf = compute_statistics(performance)

    # Train a model to predict performance
    coeffs, feature_names = train_linear_model(rows, performance)

    # Predict for the poor configuration
    poor_pred = predict(poor_conf, coeffs, feature_names)

    # Search for a better configuration that differs by exactly one setting
    best_candidate = None
    best_candidate_pred = None
    best_diff = None

    for r in rows:
        if r is poor_conf:
            continue
        if diff_count(poor_conf, r) != 1:
            continue
        cand_pred = predict(r, coeffs, feature_names)
        if cand_pred < poor_pred:
            if (best_candidate is None) or (cand_pred < best_candidate_pred):
                best_candidate = r
                best_candidate_pred = cand_pred

    if best_candidate is None:
        return poor_conf, None, poor_pred, None  # no better one‑step neighbor

    # Identify the changed setting
    setting, old_val, new_val = diff_setting(best_candidate, poor_conf)
    return poor_conf, setting, old_val, new_val, best_candidate_pred, poor_pred


# ----------------------------------------------------------------------
# APACHE-16: Minimal‑change search towards target
# ----------------------------------------------------------------------
def minimal_changes_to_target(rows, performance, target_perf):
    """
    Find the smallest number of setting changes that bring a poorly
    performing configuration to a performance <= target_perf.
    Only historically observed configurations are considered.
    Returns (poor_conf, best_match, num_changes, best_perf).
    """
    # Find the first poor configuration
    poor_list = find_poor_configs(rows, performance, top_percentile_threshold=10)
    if not poor_list:
        return None, None, None, None

    poor_idx, poor_conf, poor_perf = poor_list[0]

    # Search over all historical configurations
    best_match = None
    best_changes = None
    best_perf = None

    for r in rows:
        if r is poor_conf:
            continue
        perf_val = float(r["Performance-"])
        if perf_val > target_perf:
            continue
        changes = diff_count(poor_conf, r)
        if best_changes is None or changes < best_changes:
            best_changes = changes
            best_match = r
            best_perf = perf_val
            if changes == 0:  # cannot get better
                break

    return poor_conf, best_match, best_changes, best_perf


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main(csv_path_str):
    csv_path = Path(csv_path_str)
    if not csv_path.is_file():
        print(f"Error: file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)

    rows, performance = read_csv(csv_path)

    # ---- APACHE-15 ----
    print("== APACHE-15 ==")
    # Get recommendation for a poor config
    # We reuse find_poor_configs from skill_05
    poor_list = find_poor_configs(rows, performance, top_percentile_threshold=10)
    if not poor_list:
        print("No poorly performing configurations detected.")
    else:
        # Use the first poor config for demonstration
        poor_conf = poor_list[0][1]
        # Train model
        coeffs, feature_names = train_linear_model(rows, performance)
        poor_pred = predict(poor_conf, coeffs, feature_names)

        # Find a better neighbor
        best_candidate = None
        best_candidate_pred = None
        setting_changed = None
        old_val = None
        new_val = None

        for r in rows:
            if r is poor_conf:
                continue
            if diff_count(poor_conf, r) != 1:
                continue
            cand_pred = predict(r, coeffs, feature_names)
            if cand_pred < poor_pred:
                if best_candidate is None or cand_pred < best_candidate_pred:
                    best_candidate = r
                    best_candidate_pred = cand_pred

        if best_candidate:
            setting, old_val, new_val = diff_setting(best_candidate, poor_conf)
            print(f"Poor configuration (performance={poor_conf['Performance-']} ms):")
            print(f"  Recommended change: setting '{setting}' from '{old_val}' to '{new_val}'")
            print(f"  Predicted improvement: {poor_pred:.2f} ms → {best_candidate_pred:.2f} ms")
        else:
            print("No single setting change leads to a lower predicted performance.")

    print()

    # ---- APACHE-16 ----
    print("== APACHE-16 ==")
    # Determine a target performance (use mean)
    mean_perf, _, _, _ = compute_statistics(performance)
    target_perf = mean_perf

    if not poor_list:
        print("No poorly performing configurations to adjust.")
    else:
        poor_conf = poor_list[0][1]
        best_match, best_changes, best_perf = None, None, None

        # Find minimal changes
        for r in rows:
            if r is poor_conf:
                continue
            perf_val = float(r["Performance-"])
            if perf_val > target_perf:
                continue
            changes = diff_count(poor_conf, r)
            if best_changes is None or changes < best_changes:
                best_changes = changes
                best_match = r
                best_perf = perf_val
                if changes == 0:
                    break

        if best_match:
            print(f"Target performance: {target_perf:.2f} ms")
            print(f"Minimal number of setting changes needed: {best_changes}")
            print(f"Resulting configuration ({format_config(best_match)}): performance={best_perf:.2f} ms")
        else:
            print("No historically observed configuration achieves the target performance.")

    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
