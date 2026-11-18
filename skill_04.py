#!/usr/bin/env python3
"""
skill_04.py

Entry point for APACHE-07 and APACHE-08 requests.

Usage:
    python skill_04.py <csv_path>
"""

import csv
import sys
import json
import math

# --------------------------------------------------------------------------- #
# Shared utilities

def parse_cell(val: str):
    """Parse a CSV cell value: convert to int or None if '?'."""
    val = val.strip()
    if val == "?":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def load_csv(path: str):
    """Load CSV, return list of (row_num, dict_of_columns)."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader, start=1):  # row numbers start at 1
            row = {}
            for k, v in raw.items():
                row[k] = parse_cell(v)
            rows.append((i, row))
    return rows


def identify_columns(rows):
    """Return goal_cols, input_cols as lists."""
    col_keys = rows[0][1].keys()
    goal_cols = [col for col in col_keys if col.endswith('+') or col.endswith('-')]
    input_cols = [col for col in col_keys if not (col.endswith('+') or col.endswith('-') or col.endswith('X'))]
    return goal_cols, input_cols


# --------------------------------------------------------------------------- #
# Statistical helpers

def mean_std(values):
    """Return (mean, std) of a list of numbers. std is population std."""
    n = len(values)
    if n == 0:
        return (None, None)
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / n
    std = math.sqrt(var)
    return (m, std)


def stats_summary(values):
    """Return n, mean, min, max for list of numbers."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "min": None, "max": None}
    m = sum(values) / n
    mn = min(values)
    mx = max(values)
    return {"n": n, "mean": m, "min": mn, "max": mx}


# --------------------------------------------------------------------------- #
# Main logic

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)
    goal_cols, input_cols = identify_columns(rows)

    # Split history and arrivals
    total_rows = len(rows)
    arrivals_count = round(0.2 * total_rows)
    history_rows = rows[:total_rows - arrivals_count]
    arrivals_rows = rows[total_rows - arrivals_count:]

    # ------------------------------------------------------------------ APACHE-07
    print("== APACHE-07 ==")
    print("Assessments:")

    assessments = []
    # Keep a running list of previous Performance- values (non-None)
    prev_perf_vals = []

    for rnum, row in arrivals_rows:
        perf_val = row.get('Performance-')
        # Compute alert
        alert = False
        if perf_val is not None and prev_perf_vals:
            m, std = mean_std(prev_perf_vals)
            if std is None or std == 0:
                alert = perf_val > m
            else:
                alert = perf_val > m + 2 * std
        # Append assessment
        assessments.append({"row": rnum, "alert": alert})
        # Update history for next arrivals
        if perf_val is not None:
            prev_perf_vals.append(perf_val)

    # Print simple output
    for a in assessments:
        print(f"Row {a['row']}: alert={a['alert']}")
    print()

    print("ANSWER APACHE-07", json.dumps({"assessments": assessments}))

    # ------------------------------------------------------------------ APACHE-08
    print("\n== APACHE-08 ==")
    checkpoints = []
    # Define the checkpoint thresholds
    m_values = [0, 19, 38]
    # For each checkpoint, gather all rows seen so far
    for m in m_values:
        if m > arrivals_count:
            # Skip checkpoints beyond available arrivals
            continue
        # Rows included: history + first m arrivals
        seen_rows = history_rows + arrivals_rows[:m]
        # Build summary per goal column
        summary = {}
        for gcol in goal_cols:
            vals = [row[gcol] for _, row in seen_rows if row[gcol] is not None]
            stats = stats_summary(vals)
            summary[gcol] = stats
        # Determine shortlist: top 5 best rows so far based on Performance-
        perf_rows = [(rnum, row) for rnum, row in seen_rows if row.get('Performance-') is not None]
        perf_rows.sort(key=lambda x: (x[1]['Performance-'], x[0]))
        shortlist = [rnum for rnum, _ in perf_rows[:5]]
        checkpoints.append({"after": m, "summary": summary, "shortlist": shortlist})

    # Print simple output
    for cp in checkpoints:
        print(f"After {cp['after']} arrivals:")
        for gcol, stats in cp["summary"].items():
            print(f"  {gcol}: n={stats['n']}, mean={stats['mean']}, min={stats['min']}, max={stats['max']}")
        print(f"  Shortlist: {cp['shortlist']}")
    print()

    # Build final JSON structure
    final = {
        "checkpoints": [
            {"after": cp["after"], "summary": cp["summary"]} for cp in checkpoints
        ],
        "shortlist": checkpoints[-1]["shortlist"] if checkpoints else []
    }
    print("ANSWER APACHE-08", json.dumps(final))


if __name__ == "__main__":
    main()
