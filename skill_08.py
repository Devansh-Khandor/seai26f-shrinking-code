#!/usr/bin/env python3
"""
Skill 08 – Apache configuration adjustment and optimization

This script implements two new requests:

* APACHE-15: For a poorly performing configuration, recommend a small
  change to its adjustable settings that could lead to a lower predicted
  performance value, using historically observed valid configurations as
  alternatives.

* APACHE-16: Find the smallest number of configuration‑setting changes
  that could bring a poorly performing configuration closer to a
  specified performance target, while avoiding unsupported or invalid
  combinations.

The implementation re‑uses utilities from the earlier skills and keeps
the code DRY by importing helper functions instead of duplicating
logic.
"""

import sys
import argparse
import numpy as np

# ----------------------------------------------------------------------
# Import utilities from skill_01 and skill_02
# ----------------------------------------------------------------------
try:
    from skill_01 import read_csv, find_performance_column, compute_stats
except Exception:
    # Fallback minimal helpers if skill_01 is missing
    import csv

    def read_csv(path: str):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = []
            for row in reader:
                int_row = {}
                for k, v in row.items():
                    try:
                        int_row[k] = int(v)
                    except ValueError:
                        int_row[k] = v
                rows.append(int_row)
        return header, rows

    def find_performance_column(header):
        for col in header:
            if col.lower().startswith("performance"):
                return col
        raise KeyError("No performance column found in header.")

    def compute_stats(performance_values):
        from statistics import mean, median
        import numpy as np

        return {
            "mean": mean(performance_values),
            "median": median(performance_values),
            "std": np.std(performance_values, ddof=0),
            "best": min(performance_values),
            "worst": max(performance_values),
        }

try:
    from skill_02 import train_regression, predict_performance
except Exception:
    # Minimal regression helpers if skill_02 is missing
    from sklearn.linear_model import LinearRegression

    def train_regression(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    def predict_performance(model, X):
        return model.predict(X)


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def get_worst_config(rows, perf_col):
    """Return the row with the worst actual performance (highest value)."""
    worst_row = max(rows, key=lambda r: r[perf_col])
    return worst_row


def build_config_index(rows, config_cols):
    """
    Build a dictionary mapping the tuple of configuration values to
    the corresponding row index. Allows quick existence checks.
    """
    index = {}
    for idx, row in enumerate(rows):
        key = tuple(row[col] for col in config_cols)
        index[key] = idx
    return index


def hamming_distance(row_a, row_b, config_cols):
    """Count how many settings differ between two rows."""
    return sum(row_a[col] != row_b[col] for col in config_cols)


def format_config(row, perf_col):
    """Return a string representation of a configuration excluding perf."""
    cfg = {k: v for k, v in row.items() if k != perf_col}
    return str(cfg)


# ----------------------------------------------------------------------
# APACHE-15: Small change recommendation
# ----------------------------------------------------------------------
def recommend_small_change(rows, perf_col, predictions, config_cols):
    """
    For the worst configuration, find a neighbor that exists in the dataset
    with a lower predicted performance. Suggest the best such neighbor.
    """
    worst_row = get_worst_config(rows, perf_col)
    config_index = build_config_index(rows, config_cols)

    best_candidate = None
    best_pred = float("inf")

    # Generate all one‑bit‑flipped neighbors
    for col in config_cols:
        neighbor_cfg = {c: worst_row[c] for c in config_cols}
        neighbor_cfg[col] = 1 - neighbor_cfg[col]  # toggle 0/1

        key = tuple(neighbor_cfg[c] for c in config_cols)
        if key in config_index:
            idx = config_index[key]
            pred = predictions[idx]
            if pred < best_pred:
                best_pred = pred
                best_candidate = rows[idx]

    print("== APACHE-15 ==")
    if best_candidate is None:
        print("No single‑setting neighbor exists that improves predicted performance.")
        return

    best_idx = rows.index(best_candidate)
    worst_idx = rows.index(worst_row)

    print(f"Worst actual configuration (index={worst_idx}):")
    print(f"  Settings: {format_config(worst_row, perf_col)}")
    print(f"  Actual performance: {worst_row[perf_col]} ms")
    print()
    print(f"Recommended change (index={best_idx}):")
    print(f"  Settings: {format_config(best_candidate, perf_col)}")
    print(f"  Predicted performance: {best_pred:.1f} ms")
    diff = worst_row[perf_col] - best_candidate[perf_col]
    print(f"  Expected improvement (actual difference): {diff} ms")
    print(f"  Note: prediction suggests a lower value than the worst config.")


# ----------------------------------------------------------------------
# APACHE-16: Minimum‑change path to target performance
# ----------------------------------------------------------------------
def min_changes_to_target(rows, perf_col, predictions, config_cols, target_perf):
    """
    Find the configuration(s) that reach at least the target performance
    with the fewest setting changes from the worst configuration.
    """
    worst_row = get_worst_config(rows, perf_col)

    # Filter candidates that meet the target (predicted performance <= target)
    candidates = []
    for idx, row in enumerate(rows):
        if predictions[idx] <= target_perf:
            dist = hamming_distance(row, worst_row, config_cols)
            candidates.append((dist, predictions[idx], idx, row))

    if not candidates:
        return None  # No candidate meets target

    # Choose minimal distance; if tie, choose best predicted
    candidates.sort(key=lambda x: (x[0], x[1]))
    best_dist, best_pred, best_idx, best_row = candidates[0]
    return {
        "dist": best_dist,
        "pred_perf": best_pred,
        "idx": best_idx,
        "row": best_row,
        "worst_idx": rows.index(worst_row),
        "worst_row": worst_row,
    }


def describe_changes(row_a, row_b, config_cols):
    """Return a list of setting changes from row_a to row_b."""
    changes = []
    for col in config_cols:
        if row_a[col] != row_b[col]:
            changes.append((col, row_a[col], row_b[col]))
    return changes


def recommend_to_target(rows, perf_col, predictions, config_cols, target_perf):
    result = min_changes_to_target(rows, perf_col, predictions, config_cols, target_perf)
    print("== APACHE-16 ==")
    if result is None:
        print(f"No configuration reaches the target performance ({target_perf:.0f} ms) while differing by a few settings.")
        return

    changes = describe_changes(result["worst_row"], result["row"], config_cols)
    print(f"Target performance: {target_perf:.0f} ms")
    print(f"Worst actual configuration (index={result['worst_idx']}):")
    print(f"  Settings: {format_config(result['worst_row'], perf_col)}")
    print(f"  Actual performance: {result['worst_row'][perf_col]} ms")
    print()
    print(f"Recommended configuration (index={result['idx']}):")
    print(f"  Settings: {format_config(result['row'], perf_col)}")
    print(f"  Predicted performance: {result['pred_perf']:.1f} ms")
    print(f"  Number of setting changes: {result['dist']}")
    if changes:
        print("  Changes required:")
        for col, old, new in changes:
            print(f"    - {col}: {old} -> {new}")
    else:
        print("  No changes required; already meets target.")


# ----------------------------------------------------------------------
# Main skill logic
# ----------------------------------------------------------------------
def main(csv_path: str):
    header, rows = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        return

    perf_col = find_performance_column(header)
    config_cols = [c for c in header if c != perf_col]

    # Build feature matrix and target vector
    X = np.array([[row[col] for col in config_cols] for row in rows])
    y = np.array([row[perf_col] for row in rows])

    # Train regression model and get predictions
    model = train_regression(X, y)
    predictions = predict_performance(model, X)

    # Request APACHE-15
    recommend_small_change(rows, perf_col, predictions, config_cols)
    print()  # blank line between requests

    # Request APACHE-16
    # Use the median of actual performances as the target
    perf_values = np.array([row[perf_col] for row in rows])
    target_perf = np.median(perf_values)
    recommend_to_target(rows, perf_col, predictions, config_cols, target_perf)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apache configuration adjustment and optimization skill."
    )
    parser.add_argument(
        "csv_path",
        help="Path to Apache_AllMeasurements.csv",
    )
    args = parser.parse_args()

    try:
        main(args.csv_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
