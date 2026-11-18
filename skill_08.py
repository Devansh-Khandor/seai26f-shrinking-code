#!/usr/bin/env python3
"""
skill_08.py

Implements APACHE-15 and APACHE-16:
  - APACHE-15: For a poorly performing configuration, recommend a small change
    that could lead to a lower predicted performance using observed configurations.
  - APACHE-16: Find the smallest number of setting changes that bring a poorly
    performing configuration closer to a target performance while staying within
    valid observed rows.

The logic re‑uses helpers from previous skills to maintain consistency.
"""

import csv
import json
import sys
from collections import defaultdict

# ----------------------------------------------------------------------
# Re‑use helpers from earlier skills
# ----------------------------------------------------------------------
try:
    from skill_01 import load_csv, to_float, compute_stats
except Exception:
    # Fallback minimal implementation
    def load_csv(path):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            data = []
            for row in reader:
                cleaned = {k: (None if v == '?' else v) for k, v in row.items()}
                data.append(cleaned)
        return data, headers

    def to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def compute_stats(values):
        vals = [v for v in values if v is not None]
        if not vals:
            return (None, None, None, None)
        mean = sum(vals) / len(vals)
        import statistics
        median = statistics.median(vals)
        _min = min(vals)
        _max = max(vals)
        return (mean, median, _min, _max)

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def _percentile(values, pct):
    """Return the value at the given percentile (0-100) from a list of floats."""
    if not values:
        return None
    sorted_vals = sorted(values)
    index = int(pct * len(sorted_vals) // 100)
    if index >= len(sorted_vals):
        index = len(sorted_vals) - 1
    return sorted_vals[index]

def _diff_settings(row_a, row_b, setting_cols):
    """Return dict of differing settings and their (from,to) values."""
    diffs = {}
    for col in setting_cols:
        va = row_a.get(col)
        vb = row_b.get(col)
        if va is None or vb is None:
            continue
        if va != vb:
            diffs[col] = [va, vb]
    return diffs

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    data, headers = load_csv(csv_path)

    # Identify goal and setting columns
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    perf_col = goal_cols[0]   # Assuming a single goal column
    setting_cols = [h for h in headers if h not in goal_cols and not h.endswith('X')]

    # Gather performance values
    perf_values = [to_float(row.get(perf_col)) for row in data]
    valid_perf = [v for v in perf_values if v is not None]
    if not valid_perf:
        print("No valid performance values.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Identify a poorly performing row (>= 75th percentile)
    # ------------------------------------------------------------------
    q3_val = _percentile(valid_perf, 75)
    poor_indices = [idx + 1 for idx, v in enumerate(perf_values) if v is not None and v >= q3_val]
    if not poor_indices:
        # fallback: pick worst row
        worst_idx = max(range(len(perf_values)), key=lambda i: perf_values[i] if perf_values[i] is not None else float('-inf'))
        poor_row_num = worst_idx + 1
    else:
        poor_row_num = poor_indices[0]  # first poor row
    poor_row = data[poor_row_num - 1]
    poor_perf = perf_values[poor_row_num - 1]

    # ------------------------------------------------------------------
    # APACHE-15: recommend a small change (≤2 settings) to a better config
    # ------------------------------------------------------------------
    best_alt = None
    best_improvement = 0.0
    best_diffs = {}
    for idx, alt_row in enumerate(data):
        alt_perf = perf_values[idx]
        if alt_perf is None or alt_perf >= poor_perf:
            continue
        diffs = _diff_settings(poor_row, alt_row, setting_cols)
        if not diffs or len(diffs) > 2:
            continue
        improvement = poor_perf - alt_perf
        if improvement > best_improvement or (improvement == best_improvement and (best_alt is None or len(diffs) < len(best_diffs))):
            best_improvement = improvement
            best_alt = idx + 1
            best_diffs = diffs

    if best_alt is None:
        # Fallback: no alternative found; use same row
        best_alt = poor_row_num
        best_diffs = {}
        alt_perf_after = poor_perf
    else:
        alt_perf_after = perf_values[best_alt - 1]

    print("== APACHE-15 ==")
    print(f"Poor row: {poor_row_num}")
    print(f"Alternative row: {best_alt}")
    print(f"Changes: {json.dumps(best_diffs)}")
    print(f"Performance before: {poor_perf:.2f}")
    print(f"Performance after: {alt_perf_after:.2f}")

    apache15_json = {
        "row": poor_row_num,
        "alternative": best_alt,
        "changes": best_diffs,
        "performance": {
            "before": poor_perf,
            "after": alt_perf_after
        }
    }
    print("ANSWER APACHE-15 " + json.dumps(apache15_json, separators=(',', ':')))

    # ------------------------------------------------------------------
    # APACHE-16: bring a poorly performing configuration closer to a target
    # ------------------------------------------------------------------
    # Define target as 25th percentile (good performance)
    q1_val = _percentile(valid_perf, 25)
    target = q1_val
    # Only consider alternatives that are closer to target than the poor row
    delta_poor = abs(poor_perf - target)
    best_alt2 = None
    best_diff_count = None
    best_delta = None
    best_changes_list = []

    for idx, alt_row in enumerate(data):
        alt_perf = perf_values[idx]
        if alt_perf is None:
            continue
        delta_alt = abs(alt_perf - target)
        if delta_alt >= delta_poor:
            continue  # not closer
        diffs = _diff_settings(poor_row, alt_row, setting_cols)
        if not diffs:
            continue
        diff_count = len(diffs)
        if best_alt2 is None or diff_count < best_diff_count or (diff_count == best_diff_count and delta_alt < best_delta):
            best_alt2 = idx + 1
            best_diff_count = diff_count
            best_delta = delta_alt
            best_changes_list = list(diffs.keys())

    if best_alt2 is None:
        # Fallback: pick the row with minimal diff_count regardless of closeness
        for idx, alt_row in enumerate(data):
            alt_perf = perf_values[idx]
            if alt_perf is None:
                continue
            diffs = _diff_settings(poor_row, alt_row, setting_cols)
            if not diffs:
                continue
            diff_count = len(diffs)
            if best_alt2 is None or diff_count < best_diff_count:
                best_alt2 = idx + 1
                best_diff_count = diff_count
                best_changes_list = list(diffs.keys())

    if best_alt2 is None:
        # No alternative found; use same row
        best_alt2 = poor_row_num
        best_changes_list = []

    print("== APACHE-16 ==")
    print(f"Poor row: {poor_row_num}")
    print(f"Target performance: {target:.2f}")
    print(f"Alternative row: {best_alt2}")
    print(f"Changed settings: {best_changes_list}")

    apache16_json = {
        "row": poor_row_num,
        "target": target,
        "alternative": best_alt2,
        "changed": best_changes_list
    }
    print("ANSWER APACHE-16 " + json.dumps(apache16_json, separators=(',', ':')))

if __name__ == "__main__":
    main()
