#!/usr/bin/env python3
"""
skill_04.py

Implements APACHE-07 and APACHE-08 for incremental analysis.
"""

import csv
import json
import sys
from collections import defaultdict

# ----------------------------------------------------------------------
# Reuse helpers from previous skills
# ----------------------------------------------------------------------
def load_csv(path):
    """
    Load the CSV file and return:
      - data: list of dicts (each row)
      - headers: list of column names
    Missing values represented by '?'
    """
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
    """Return mean, median, min, max for a list of floats"""
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
# Helper: goal columns
# ----------------------------------------------------------------------
def goal_columns(headers):
    return [h for h in headers if h.endswith('+') or h.endswith('-')]

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    data, headers = load_csv(csv_path)
    n_rows = len(data)

    # Identify arrivals (last round(0.2 * n_rows) rows)
    arrivals_count = round(0.2 * n_rows)
    history_rows = data[:-arrivals_count] if arrivals_count < n_rows else []
    arrivals_rows = data[-arrivals_count:] if arrivals_count < n_rows else []

    # ------------------------------------------------------------------
    # APACHE-07: Assess arrivals
    # ------------------------------------------------------------------
    perf_col = goal_columns(headers)[0]  # Assume single goal column for this dataset
    assessments = []
    prev_perfs = []

    # Include history before evaluating arrivals
    for row in history_rows:
        val = to_float(row.get(perf_col))
        if val is not None:
            prev_perfs.append(val)

    for i, row in enumerate(arrivals_rows):
        row_num = len(history_rows) + i + 1
        val = to_float(row.get(perf_col))
        alert = False
        if val is not None:
            if prev_perfs:
                # 25th percentile threshold
                sorted_vals = sorted(prev_perfs)
                idx = len(sorted_vals) // 4
                threshold = sorted_vals[idx]
                alert = val <= threshold
            else:
                alert = False
            prev_perfs.append(val)
        assessments.append({"row": row_num, "alert": alert})

    # Output
    print("== APACHE-07 ==")
    for a in assessments:
        print(f"Row {a['row']}: alert={a['alert']}")
    print("ANSWER APACHE-07 " + json.dumps({"assessments": assessments}, separators=(',', ':')))

    # ------------------------------------------------------------------
    # APACHE-08: Checkpoints
    # ------------------------------------------------------------------
    checkpoints_m = [0, 19, 38]
    # Ensure we don't request more arrivals than we have
    checkpoints_m = [min(m, arrivals_count) for m in checkpoints_m]
    checkpoints = []

    # Build mapping of row number to row dict for ease
    row_map = {i + 1: r for i, r in enumerate(data)}

    for m in checkpoints_m:
        rows_to_consider = history_rows + arrivals_rows[:m]
        row_numbers = [len(history_rows) + i + 1 for i in range(len(rows_to_consider) - m)]
        row_numbers += [len(history_rows) + i + 1 for i in range(m)]
        # Summary
        summary = {}
        for goal in goal_columns(headers):
            goal_vals = [to_float(r.get(goal)) for r in rows_to_consider]
            mean, _, _min, _max = compute_stats(goal_vals)
            summary[goal] = {
                "n": sum(v is not None for v in goal_vals),
                "mean": mean,
                "min": _min,
                "max": _max
            }
        # Shortlist best 5 rows by Performance- (lower better)
        perf_vals_with_row = [(to_float(r.get(perf_col)), idx+1) for idx, r in enumerate(rows_to_consider) if to_float(r.get(perf_col)) is not None]
        perf_vals_with_row.sort()
        shortlist = [row for _, row in perf_vals_with_row[:5]]
        checkpoints.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })

    # Output
    print("== APACHE-08 ==")
    for cp in checkpoints:
        print(f"Checkpoint after {cp['after']} arrivals: shortlist={cp['shortlist']}")
    print("ANSWER APACHE-08 " + json.dumps({"checkpoints": checkpoints}, separators=(',', ':')))

if __name__ == "__main__":
    main()
