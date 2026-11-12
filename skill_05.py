#!/usr/bin/env python3
"""
skill_05.py – Advanced analysis of poor configurations and frequent bad combinations.

Usage:
    python skill_05.py <path-to-Apache_AllMeasurements.csv>
"""

import sys
import csv
import math
import itertools
import skill_01 as s1  # Reuse utilities from the first skill

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_csv(path):
    """Load CSV into a list of dict rows and return column names."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames
    return rows, headers

def mean_and_std(values):
    """Return mean and population standard deviation."""
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    return mean, std

# ----------------------------------------------------------------------
# APACHE-09: Poor‑config setting association
# ----------------------------------------------------------------------
def analyze_poor_associations(rows, headers, z_thresh=2.0, top_n=5):
    """
    For configurations whose performance is > mean + z_thresh * std,
    compute how strongly each setting value is associated with poor performance.
    """
    target_col = 'Performance-'
    perf_values = [int(row[target_col]) for row in rows]
    mean_perf, std_perf = mean_and_std(perf_values)

    # Determine poor configurations
    poor_thresh = mean_perf + z_thresh * std_perf
    poor_rows = [row for row in rows if int(row[target_col]) > poor_thresh]
    if not poor_rows:
        return []

    # Overall counts per setting value
    overall_counts = {h: {'0': 0, '1': 0} for h in headers if h != target_col}
    poor_counts   = {h: {'0': 0, '1': 0} for h in headers if h != target_col}

    for row in rows:
        for h in overall_counts:
            overall_counts[h][row[h]] += 1

    for row in poor_rows:
        for h in poor_counts:
            poor_counts[h][row[h]] += 1

    # Compute association score: difference in proportions
    associations = []
    for h in overall_counts:
        for val in ('0', '1'):
            overall_prop = overall_counts[h][val] / len(rows)
            poor_prop    = poor_counts[h][val] / len(poor_rows)
            diff = poor_prop - overall_prop
            associations.append((abs(diff), h, val, diff, poor_prop, overall_prop))

    # Sort by absolute difference, then by magnitude of diff
    associations.sort(reverse=True, key=lambda x: (x[0], x[3]))

    # Take top N associations
    top = associations[:top_n]
    return top

# ----------------------------------------------------------------------
# APACHE-10: Frequent bad combinations
# ----------------------------------------------------------------------
def frequent_bad_combinations(rows, headers, z_thresh=2.0, top_n=5):
    """
    Identify pairs of setting values that appear frequently in poor configurations
    relative to all configurations.
    """
    target_col = 'Performance-'
    perf_values = [int(row[target_col]) for row in rows]
    mean_perf, std_perf = mean_and_std(perf_values)

    # Poor threshold
    poor_thresh = mean_perf + z_thresh * std_perf
    poor_rows = [row for row in rows if int(row[target_col]) > poor_thresh]
    if not poor_rows:
        return []

    feature_cols = [h for h in headers if h != target_col]

    combo_stats = []

    # Enumerate all unordered pairs of columns
    for col1, col2 in itertools.combinations(feature_cols, 2):
        # For each value combination (0/1 for each)
        for val1 in ('0', '1'):
            for val2 in ('0', '1'):
                # Count overall and poor
                overall_count = 0
                overall_perf_sum = 0
                poor_count = 0
                poor_perf_sum = 0
                for row in rows:
                    if row[col1] == val1 and row[col2] == val2:
                        overall_count += 1
                        overall_perf_sum += int(row[target_col])
                        if int(row[target_col]) > poor_thresh:
                            poor_count += 1
                            poor_perf_sum += int(row[target_col])

                # Avoid zero overall count
                if overall_count == 0:
                    continue

                overall_prop = overall_count / len(rows)
                poor_prop    = poor_count / len(poor_rows) if len(poor_rows) > 0 else 0

                # Ratio of proportions; handle zero poor_prop
                ratio = poor_prop / overall_prop if overall_prop > 0 else 0

                # Mean performances (optional)
                overall_mean_perf = overall_perf_sum / overall_count
                poor_mean_perf = poor_perf_sum / poor_count if poor_count > 0 else None

                combo_stats.append((
                    ratio,
                    col1, val1,
                    col2, val2,
                    overall_prop,
                    poor_prop,
                    overall_mean_perf,
                    poor_mean_perf,
                ))

    # Sort by ratio descending
    combo_stats.sort(reverse=True, key=lambda x: x[0])

    # Return top N combos with ratio > 1
    top = [c for c in combo_stats if c[0] > 1][:top_n]
    return top

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    rows, headers = load_csv(csv_path)

    # ------------------------------------------------------------------
    # APACHE-09: Poor configuration association analysis
    # ------------------------------------------------------------------
    print("== APACHE-09 ==")
    assoc_results = analyze_poor_associations(rows, headers, z_thresh=2.0, top_n=5)
    if not assoc_results:
        print("  No poor configurations identified.")
    else:
        print("  Top setting values most strongly associated with poor performance:")
        for abs_diff, col, val, diff, poor_prop, overall_prop in assoc_results:
            print(f"  - {col}={val}")
            print(f"      Poor proportion: {poor_prop*100:.1f}%")
            print(f"      Overall proportion: {overall_prop*100:.1f}%")
            print(f"      Difference: {diff*100:.1f}%")

    # ------------------------------------------------------------------
    # APACHE-10: Frequent bad combinations
    # ------------------------------------------------------------------
    print("\n== APACHE-10 ==")
    combo_results = frequent_bad_combinations(rows, headers, z_thresh=2.0, top_n=5)
    if not combo_results:
        print("  No frequent bad combinations identified.")
    else:
        print("  Top setting pairs frequently seen in poor configurations:")
        for ratio, col1, val1, col2, val2, overall_prop, poor_prop, overall_mean_perf, poor_mean_perf in combo_results:
            print(f"  - {col1}={val1} & {col2}={val2}")
            print(f"      Overall proportion: {overall_prop*100:.1f}%")
            print(f"      Poor proportion: {poor_prop*100:.1f}%")
            print(f"      Ratio (poor/overall): {ratio:.2f}")
            print(f"      Mean perf overall: {overall_mean_perf:.2f}")
            if poor_mean_perf is not None:
                print(f"      Mean perf poor configs: {poor_mean_perf:.2f}")
            else:
                print(f"      Mean perf poor configs: N/A")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_05.py <csv_path>")
    main(sys.argv[1])
