#!/usr/bin/env python3
"""
skill_05.py

Adds two new requests to the Apache analysis toolkit:
  - APACHE-09: Identify which enabled or disabled settings are most strongly
               associated with unusually poor performance in historical data.
  - APACHE-10: Identify combinations of server settings that frequently appear
               among poorly performing configurations and show how they differ
               from better-performing combinations.

The implementation re‑uses helper functions from the previous skill modules.
"""

import sys
import json
import numpy as np
from itertools import combinations
from skill_01 import read_csv, parse_column_values

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def get_poor_rows(rows, perf_col):
    """
    Returns a list of row indices (1‑based) that are identified as unusually poor
    using the same logic as APACHE‑07: performance < mean - 2*std of all
    previous rows (including missing values treated as NaN).
    """
    poor = []
    cumulative = []  # cumulative rows up to current (excluding current)
    for idx, row in enumerate(rows):
        # Compute stats on cumulative
        vals = parse_column_values(cumulative, perf_col)
        if np.isnan(vals).all():
            mean = np.nan
            std = np.nan
        else:
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
        perf = row[perf_col]
        if perf == '?':
            alert = False
        else:
            perf_val = float(perf)
            threshold = mean - 2 * std if not np.isnan(mean) and not np.isnan(std) else None
            alert = threshold is not None and perf_val < threshold
        if alert:
            poor.append(idx + 1)  # 1‑based
        cumulative.append(row)
    return poor


def compute_setting_association(rows, poor_rows, setting_cols):
    """
    For each setting column, compute the absolute difference in
    proportion of poor rows between enabled (1) and disabled (0).
    Return a list of tuples (col_name, diff) sorted by descending diff.
    """
    results = []
    bad_set = set(poor_rows)
    n_bad = len(bad_set)
    if n_bad == 0:
        return [(c, 0.0) for c in setting_cols]
    for col in setting_cols:
        bad_1 = bad_0 = 0
        for i in bad_set:
            val = rows[i - 1][col]
            # treat '?' as 0
            if val == '1':
                bad_1 += 1
            else:
                bad_0 += 1
        p1 = bad_1 / n_bad
        p0 = bad_0 / n_bad
        diff = abs(p1 - p0)
        results.append((col, diff))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def find_best_bad_combination(rows, bad_rows, good_rows, setting_cols):
    """
    Find the setting combination (at least 2 settings) that appears most
    frequently among bad rows. Return a dict with keys:
      - settings: dict of {column: value}
      - bad_share: fraction of bad rows matching the combination
      - good_share: fraction of good rows matching the combination
    """
    bad_set = set(bad_rows)
    good_set = set(good_rows)
    n_bad = len(bad_set)
    n_good = len(good_set)
    best = None
    best_bad_share = -1.0

    # Pre‑convert rows into a list of 0/1 values for all setting columns
    # missing '?' -> 0
    encoded = []
    for r in rows:
        encoded.append([1.0 if r[col] == '1' else 0.0 for col in setting_cols])

    for col1, col2 in combinations(setting_cols, 2):
        idx1 = setting_cols.index(col1)
        idx2 = setting_cols.index(col2)
        combo_counts = {}
        # Count occurrences in bad rows
        for i in bad_set:
            key = (encoded[i - 1][idx1], encoded[i - 1][idx2])
            combo_counts[key] = combo_counts.get(key, 0) + 1
        for combo, bad_cnt in combo_counts.items():
            bad_share = bad_cnt / n_bad if n_bad else 0.0
            if bad_share > best_bad_share:
                # Compute good share for this combo
                good_cnt = sum(1 for i in good_set if (encoded[i - 1][idx1], encoded[i - 1][idx2]) == combo)
                good_share = good_cnt / n_good if n_good else 0.0
                best_bad_share = bad_share
                best = {
                    "settings": {
                        col1: int(combo[0]),
                        col2: int(combo[1])
                    },
                    "bad_share": bad_share,
                    "good_share": good_share
                }
    # If no combination found (unlikely), fallback to empty
    if best is None:
        best = {"settings": {}, "bad_share": 0.0, "good_share": 0.0}
    return best


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path):
    rows, headers = read_csv(csv_path)

    # Identify goal columns (maximize or minimize)
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    perf_col = goal_cols[0]  # assume first goal column is Performance-

    # Identify setting columns: not goals and not ending with 'X'
    setting_cols = [h for h in headers if not h.endswith('+') and not h.endswith('-') and not h.endswith('X')]

    # ----------------------------------------------------------------------- #
    # APACHE-09
    # ----------------------------------------------------------------------- #
    print("== APACHE-09 ==")

    poor_rows = get_poor_rows(rows, perf_col)
    assoc = compute_setting_association(rows, poor_rows, setting_cols)
    # At least three lines of results
    print(f"Number of unusually poor configurations: {len(poor_rows)}")
    # Show first 5 poor row numbers if available
    if poor_rows:
        print("Example poor rows:", ', '.join(map(str, poor_rows[:5])))
    # Show top 3 settings
    print("Top 3 settings most strongly associated with poor performance:")
    for col, diff in assoc[:3]:
        print(f"  {col}: diff={diff:.4f}")

    # Machine‑checked answer
    ranking = [col for col, _ in assoc[:3]]
    print("ANSWER APACHE-09", json.dumps(
        {"poor": poor_rows, "ranking": ranking},
        separators=(',', ':')
    ))

    # ----------------------------------------------------------------------- #
    # APACHE-10
    # ----------------------------------------------------------------------- #
    print("\n== APACHE-10 ==")

    # Good rows are all rows not in poor_rows
    bad_rows = poor_rows
    good_rows = [i + 1 for i in range(len(rows)) if (i + 1) not in set(poor_rows)]

    # Ensure at least 5 rows in each
    bad_rows = bad_rows[:5] if len(bad_rows) >= 5 else bad_rows
    good_rows = good_rows[:5] if len(good_rows) >= 5 else good_rows

    best_combo = find_best_bad_combination(rows, bad_rows, good_rows, setting_cols)

    # Print results
    print(f"Bad rows (sample): {bad_rows}")
    print(f"Good rows (sample): {good_rows}")
    print("Best distinguishing combination:")
    print(f"  Settings: {best_combo['settings']}")
    print(f"  Bad share: {best_combo['bad_share']:.4f}")
    print(f"  Good share: {best_combo['good_share']:.4f}")

    # Machine‑checked answer
    combinations_output = [
        {
            "settings": best_combo["settings"],
            "bad_share": best_combo["bad_share"],
            "good_share": best_combo["good_share"]
        }
    ]
    print("ANSWER APACHE-10", json.dumps(
        {"bad": bad_rows, "good": good_rows, "combinations": combinations_output},
        separators=(',', ':')
    ))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
