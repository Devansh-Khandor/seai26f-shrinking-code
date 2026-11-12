#!/usr/bin/env python3
"""
skill_08.py – Recommendation of small changes for poor configurations and
                finding minimal change paths to a target performance.

The module implements two new APACHE analyses:

APACHE-15 – For a poorly performing configuration, recommend a single
            setting change that leads to a lower predicted performance
            using historically observed valid configurations as alternatives.

APACHE-16 – Find the smallest number of configuration-setting changes
            that could bring a poorly performing configuration closer to a
            specified performance target (the best performance in the data),
            while staying within the set of valid combinations.

The implementation reuses the utilities from the earlier skills (skill_01,
skill_02, skill_06) for loading data, feature extraction, regression
modelling, prediction, and Hamming distance calculations.
"""

import sys
import numpy as np
import skill_01 as s1
import skill_02 as s2
import skill_06 as s6  # for hamming_distance

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_csv(path):
    """Delegate to skill_01 for consistency."""
    return s1.load_csv(path)

def feature_matrix(rows, headers):
    """Return numpy array of features for each row (excluding the target)."""
    return s2.feature_matrix(rows, headers)[0], s2.feature_matrix(rows, headers)[1]

def target_vector(rows, headers):
    """Return numpy array of target values (Performance-)."""
    return s2.target_vector(rows, headers)

def train_model(X, y):
    """Train and return a linear regression model."""
    return s2.train_regressor(X, y)

def find_one_setting_pairs(rows, feature_cols):
    """Yield tuples (idx1, idx2, differing_col) for pairs differing by one setting."""
    return s2.find_one_setting_pairs(rows, feature_cols)

# ----------------------------------------------------------------------
# APACHE-15: Small change recommendation
# ----------------------------------------------------------------------
def recommend_small_change(rows, headers, feature_cols, model):
    """
    For a poorly performing configuration, recommend a single setting change
    that yields the lowest predicted performance among neighboring configs.
    """
    y = target_vector(rows, headers)
    perf_values = [int(row['Performance-']) for row in rows]

    mean_perf = np.mean(perf_values)
    std_perf = np.std(perf_values, ddof=0)
    poor_thresh = mean_perf + 2.0 * std_perf

    # Identify the first poor configuration
    poor_idx = None
    for idx, val in enumerate(perf_values):
        if val > poor_thresh:
            poor_idx = idx
            break
    if poor_idx is None:
        return None  # no poor config found

    # Build a set of neighbor indices (differ by one setting)
    neighbors = []
    for i, j, diff_col in find_one_setting_pairs(rows, feature_cols):
        if i == poor_idx:
            neighbors.append((j, diff_col))
        elif j == poor_idx:
            neighbors.append((i, diff_col))

    if not neighbors:
        return None  # no neighbors differing by one setting

    # Predict performance for each neighbor
    X, _ = feature_matrix(rows, headers)
    preds = s2.predict_performance(model, X)

    best_neighbor = None
    best_pred = None
    best_diff_col = None

    for neigh_idx, diff_col in neighbors:
        pred_perf = preds[neigh_idx]
        if best_pred is None or pred_perf < best_pred:
            best_pred = pred_perf
            best_neighbor = neigh_idx
            best_diff_col = diff_col

    # Compute improvement
    current_perf = perf_values[poor_idx]
    improvement = current_perf - best_pred

    return {
        'poor_config_num': poor_idx + 1,
        'poor_perf': current_perf,
        'recommended_change': best_diff_col,
        'new_config_num': best_neighbor + 1,
        'predicted_perf': best_pred,
        'improvement': improvement
    }

# ----------------------------------------------------------------------
# APACHE-16: Minimal change path to target performance
# ----------------------------------------------------------------------
def minimal_change_to_target(rows, headers, feature_cols, target_perf):
    """
    Find the smallest number of setting changes that could bring a poor
    configuration closer to the specified target performance.
    """
    y = target_vector(rows, headers)
    perf_values = [int(row['Performance-']) for row in rows]

    mean_perf = np.mean(perf_values)
    std_perf = np.std(perf_values, ddof=0)
    poor_thresh = mean_perf + 2.0 * std_perf

    # Identify the first poor configuration
    poor_idx = None
    for idx, val in enumerate(perf_values):
        if val > poor_thresh:
            poor_idx = idx
            break
    if poor_idx is None:
        return None  # no poor config found

    # Find all configs that meet or beat the target performance
    candidate_idxs = [i for i, val in enumerate(perf_values) if val <= target_perf]
    if not candidate_idxs:
        return None  # no candidate meets target

    # Compute Hamming distances
    min_dist = None
    best_idx = None
    for cand in candidate_idxs:
        dist, _ = s6.hamming_distance(rows[poor_idx], rows[cand], feature_cols)
        if (min_dist is None) or (dist < min_dist):
            min_dist = dist
            best_idx = cand

    return {
        'poor_config_num': poor_idx + 1,
        'poor_perf': perf_values[poor_idx],
        'target_perf': target_perf,
        'changes_needed': min_dist,
        'target_config_num': best_idx + 1,
        'target_perf_value': perf_values[best_idx]
    }

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    rows, headers = load_csv(csv_path)

    # Build feature matrix, target vector, and model
    X, feature_cols = feature_matrix(rows, headers)
    y = target_vector(rows, headers)
    model = train_model(X, y)

    # ------------------------------------------------------------------
    # APACHE-15
    # ------------------------------------------------------------------
    print("== APACHE-15 ==")
    res15 = recommend_small_change(rows, headers, feature_cols, model)
    if res15 is None:
        print("  No poorly performing configuration identified or no neighboring configs.")
    else:
        print(f"  Poor configuration #{res15['poor_config_num']} (Performance={res15['poor_perf']})")
        print(f"  Recommended change: set '{res15['recommended_change']}' to the value in configuration #{res15['new_config_num']}")
        print(f"  Predicted performance after change: {res15['predicted_perf']:.2f}")
        print(f"  Expected improvement: {res15['improvement']:.2f}")

    # ------------------------------------------------------------------
    # APACHE-16
    # ------------------------------------------------------------------
    print("\n== APACHE-16 ==")
    # Use best (minimum) performance in the dataset as the target
    target_perf = int(np.min(y))
    res16 = minimal_change_to_target(rows, headers, feature_cols, target_perf)
    if res16 is None:
        print("  No poorly performing configuration identified or no target configurations available.")
    else:
        print(f"  Poor configuration #{res16['poor_config_num']} (Performance={res16['poor_perf']})")
        print(f"  Target performance to reach: {res16['target_perf']}")
        print(f"  Minimum number of setting changes required: {res16['changes_needed']}")
        print(f"  Closest target configuration #{res16['target_config_num']} "
              f"(Performance={res16['target_perf_value']})")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_08.py <csv_path>")
    main(sys.argv[1])
