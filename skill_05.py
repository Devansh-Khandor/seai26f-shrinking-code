#!/usr/bin/env python3
"""
Skill 05 – Apache configuration diagnostics

This script implements two new requests:

* APACHE-09: For poorly performing configurations, identify the
  individual server settings that are most strongly associated
  with unfavorable performance in the historical data set.
* APACHE-10: Identify combinations of server settings that frequently
  appear among poorly performing configurations and quantify how
  they differ from the overall population.

The implementation re‑uses helpers from skill_01 and follows the same
output convention: each request starts with a line
`== <ID> ==` followed by at least three lines of results.
"""

import sys
import argparse
import numpy as np

# ----------------------------------------------------------------------
# Import utilities from skill_01
# ----------------------------------------------------------------------
try:
    from skill_01 import read_csv, find_performance_column, compute_stats
except Exception:
    # Fallback minimal implementations if skill_01 is missing
    import csv
    from statistics import mean, median

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

# ----------------------------------------------------------------------
# Helper functions for APACHE-09
# ----------------------------------------------------------------------
def point_biserial_correlations(rows, perf_col, setting_cols):
    """
    Compute point‑biserial correlation between each binary setting and
    the continuous performance metric. Positive correlation means the
    setting value of 1 tends to be associated with higher (worse)
    performance.
    """
    perf_vals = np.array([row[perf_col] for row in rows])
    correlations = {}
    for col in setting_cols:
        values = np.array([row[col] for row in rows])
        # Avoid division by zero if constant
        if np.std(values) == 0 or np.std(perf_vals) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(values, perf_vals)[0, 1]
        correlations[col] = corr
    return correlations


def top_correlated_settings(correlations, n=3):
    """
    Return the top `n` settings with the highest positive correlation
    (worst‑associated).  If fewer than `n` positive correlations exist,
    the list may be shorter.
    """
    positive = {k: v for k, v in correlations.items() if v > 0}
    sorted_items = sorted(positive.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_items[:n]


# ----------------------------------------------------------------------
# Helper functions for APACHE-10
# ----------------------------------------------------------------------
def find_poor_indices(rows, perf_col, threshold):
    """Return indices of rows whose performance is >= the given threshold."""
    return [i for i, r in enumerate(rows) if r[perf_col] >= threshold]


def frequent_pairs(rows, setting_cols, poor_indices, min_poor_ratio=0.3):
    """
    Identify pairs of settings that are both enabled (==1) in a
    significant fraction of the poorly performing configurations.
    For each such pair, also compute its prevalence in the overall data.

    Returns a list of tuples:
        (col1, col2, poor_ratio, overall_ratio, difference)
    sorted by decreasing difference (poor_ratio - overall_ratio).
    """
    if not poor_indices:
        return []

    total_poor = len(poor_indices)
    total_all = len(rows)

    # Count co‑occurrence of enabled pair in poor configs
    pair_counts_poor = {}
    for i in poor_indices:
        r = rows[i]
        for idx1 in range(len(setting_cols)):
            col1 = setting_cols[idx1]
            if r[col1] != 1:
                continue
            for idx2 in range(idx1 + 1, len(setting_cols)):
                col2 = setting_cols[idx2]
                if r[col2] != 1:
                    continue
                key = (col1, col2)
                pair_counts_poor[key] = pair_counts_poor.get(key, 0) + 1

    # Compute overall counts
    pair_counts_overall = {}
    for i, r in enumerate(rows):
        for idx1 in range(len(setting_cols)):
            col1 = setting_cols[idx1]
            if r[col1] != 1:
                continue
            for idx2 in range(idx1 + 1, len(setting_cols)):
                col2 = setting_cols[idx2]
                if r[col2] != 1:
                    continue
                key = (col1, col2)
                pair_counts_overall[key] = pair_counts_overall.get(key, 0) + 1

    results = []
    for key, count_poor in pair_counts_poor.items():
        poor_ratio = count_poor / total_poor
        if poor_ratio < min_poor_ratio:
            continue
        overall_ratio = pair_counts_overall.get(key, 0) / total_all
        diff = poor_ratio - overall_ratio
        results.append((key[0], key[1], poor_ratio, overall_ratio, diff))

    # Sort by largest positive difference
    results.sort(key=lambda kv: kv[4], reverse=True)
    return results[:5]  # top 5


# ----------------------------------------------------------------------
# Main skill logic
# ----------------------------------------------------------------------
def skill_apache_09(header, rows):
    """
    Implements request APACHE-09: identify individual settings
    most strongly associated with poor performance.
    """
    perf_col = find_performance_column(header)
    setting_cols = [c for c in header if c != perf_col]

    correlations = point_biserial_correlations(rows, perf_col, setting_cols)
    top_settings = top_correlated_settings(correlations, n=3)

    print("== APACHE-09 ==")
    if not top_settings:
        print("No settings exhibit a positive correlation with performance.")
        return

    print(f"Top {len(top_settings)} settings most associated with higher (worse) performance:")
    for col, corr in top_settings:
        print(f"  - {col}: correlation={corr:.3f}")


def skill_apache_10(header, rows):
    """
    Implements request APACHE-10: identify frequent setting combinations
    among poorly performing configurations.
    """
    perf_col = find_performance_column(header)
    setting_cols = [c for c in header if c != perf_col]

    perf_vals = np.array([row[perf_col] for row in rows])
    threshold = np.percentile(perf_vals, 90)  # define "poor" as >= 90th percentile
    poor_indices = find_poor_indices(rows, perf_col, threshold)

    pairs = frequent_pairs(rows, setting_cols, poor_indices, min_poor_ratio=0.25)

    print("== APACHE-10 ==")
    print(f"Poor performance threshold (90th percentile): {threshold:.0f} ms")
    if not pairs:
        print("No pair of enabled settings occurs frequently in poor configurations.")
        return

    print(f"Top {len(pairs)} frequently co‑enabled setting pairs in poor configs:")
    for col1, col2, pr, orp, diff in pairs:
        print(
            f"  - {col1} & {col2}: "
            f"poor_ratio={pr:.2f}, overall_ratio={orp:.2f}, "
            f"diff={diff:.2f}"
        )


def main(csv_path: str):
    header, rows = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        return

    skill_apache_09(header, rows)
    print()  # blank line between skills
    skill_apache_10(header, rows)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apache diagnostic analysis skill."
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
