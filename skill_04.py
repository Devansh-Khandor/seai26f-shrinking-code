#!/usr/bin/env python3
"""
Skill 04: Apache performance monitoring and incremental summarization

This script implements the two requested analyses:
- APACHE-07: Assess each arriving configuration immediately and alert when its
  measured performance is unusually poor compared with previously tested
  configurations.
- APACHE-08: Keep performance summaries and a recommended‑configuration
  shortlist updated as additional benchmark records become available,
  without requiring a complete manual restart of the analysis.

Usage:
    python skill_04.py <path-to-CSV>
"""

import json
import sys
from math import sqrt
import ezr

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_csv(path: str):
    """Load CSV with ezr, return header and data rows."""
    rows = list(ezr.csv(path))
    if not rows:
        raise ValueError("CSV file is empty")
    header = rows[0]
    data = rows[1:]
    return header, data


def is_goal(name: str) -> bool:
    """Return True if column name ends with '+' or '-'."""
    return name.endswith('+') or name.endswith('-')


def mean_std(values):
    """Return mean and population std deviation."""
    n = len(values)
    if n == 0:
        return None, None
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / n
    return m, sqrt(var)


def summarize(values):
    """Return n, mean, min, max for a list of numeric values."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "min": None, "max": None}
    m = sum(values) / n
    mn = min(values)
    mx = max(values)
    return {"n": n, "mean": m, "min": mn, "max": mx}


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def apache_07(header, data):
    """Assess each arrival and alert when its Performance- is unusually poor."""
    perf_idx = next(i for i, n in enumerate(header) if n.endswith('-'))

    total_rows = len(data)
    arrivals_count = round(total_rows * 0.2)
    history_rows = data[:total_rows - arrivals_count]
    arrivals_rows = data[total_rows - arrivals_count:]

    # Keep global row numbers for each row
    all_rows = list(enumerate(data, start=1))

    assessments = []
    prev_vals = []

    # History rows are already evaluated
    for _, row in all_rows[:total_rows - arrivals_count]:
        perf = row[perf_idx]
        if perf != '?':
            prev_vals.append(perf)

    # Process arrivals in order
    for idx, row in enumerate(arrivals_rows, start=1):
        global_row_num = total_rows - arrivals_count + idx
        perf = row[perf_idx]
        alert = False
        if perf != '?' and prev_vals:
            m, std = mean_std(prev_vals)
            if std > 0:
                z = (perf - m) / std
                if z > 2:  # unusually poor
                    alert = True
        assessments.append({"row": global_row_num, "alert": alert})
        if perf != '?':
            prev_vals.append(perf)

    # ---- Print results ----
    print("== APACHE-07 ==")
    for a in assessments:
        print(f"Row {a['row']}: alert = {a['alert']}")
    # Machine‑checked answer
    answer = {"assessments": assessments}
    print(f"ANSWER APACHE-07 {json.dumps(answer, separators=(',', ':'))}")


def apache_08(header, data):
    """Maintain summaries and a shortlist as new rows arrive."""
    perf_idx = next(i for i, n in enumerate(header) if n.endswith('-'))

    total_rows = len(data)
    arrivals_count = round(total_rows * 0.2)
    history_rows = data[:total_rows - arrivals_count]
    arrivals_rows = data[total_rows - arrivals_count:]

    # Global row numbers
    all_rows = list(enumerate(data, start=1))
    history_with_num = all_rows[:total_rows - arrivals_count]
    arrivals_with_num = all_rows[total_rows - arrivals_count:]

    # Determine checkpoints: after 0, 19, and all arrivals
    m_values = [0, min(19, arrivals_count), arrivals_count]
    checkpoints = []

    for m in m_values:
        # Considered rows = history + first m arrivals
        considered = history_with_num + arrivals_with_num[:m]
        # Build summary for each goal column
        summary = {}
        for idx, name in enumerate(header):
            if is_goal(name):
                vals = [r[1][idx] for _, r in considered if r[1][idx] != '?']
                summary[name] = summarize(vals)

        # Build shortlist of 5 best rows (lowest performance-)
        best = sorted(
            [(row_num, r[perf_idx]) for row_num, r in considered if r[perf_idx] != '?'],
            key=lambda x: (x[1], x[0])
        )
        shortlist = [row_num for row_num, _ in best[:5]]

        checkpoints.append({"after": m, "summary": summary, "shortlist": shortlist})

    # ---- Print results ----
    print("== APACHE-08 ==")
    for cp in checkpoints:
        print(f"Checkpoint after {cp['after']} arrivals:")
        for g, stats in cp["summary"].items():
            print(f"  {g}: n={stats['n']} mean={stats['mean']:.2f} min={stats['min']} max={stats['max']}")
        print(f"  Shortlist rows: {cp['shortlist']}")
    # Machine‑checked answer
    answer = {"checkpoints": checkpoints}
    print(f"ANSWER APACHE-08 {json.dumps(answer, separators=(',', ':'))}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    header, data = read_csv(path)
    apache_07(header, data)
    apache_08(header, data)


if __name__ == "__main__":
    main()
