#!/usr/bin/env python3
"""
Skill 09: Apache configuration outliers

This script implements the following analyses:
- APACHE-17: Identify configurations whose recorded performance values are
  unusually different from what their settings and comparable historical
  configurations would lead us to expect.
- APACHE-18: Flag configurations whose combinations of settings differ
  substantially from the most similar configurations in our historical
  benchmark records, even when their measured performance is not obviously poor.
"""

import json
import sys
import math
import ezr  # the provided library


# --------------------------------------------------------------------------- #
# Helpers – reuse from earlier skills
# --------------------------------------------------------------------------- #
def read_csv(path: str):
    """Load CSV with ezr and return header and data rows."""
    rows = list(ezr.csv(path))
    if not rows:
        raise ValueError("CSV file is empty")
    header = rows[0]
    data = rows[1:]
    return header, data


def goal_indices(header):
    """Return list of indices of goal columns (+/-)."""
    return [i for i, n in enumerate(header) if n.endswith(('+', '-'))]


def is_goal(name: str) -> bool:
    """Return True if column name ends with '+' or '-'."""
    return name.endswith('+') or name.endswith('-')


# --------------------------------------------------------------------------- #
# Analysis functions
# --------------------------------------------------------------------------- #
def apache_17(header, data):
    """APACHE-17: flag rows with unusually different recorded performance."""
    goal_idx = goal_indices(header)[0]           # assume at least one goal
    goal_name = header[goal_idx]

    # Determine history rows (exclude held‑out rows: divisible by 5)
    history_rows = [row for i, row in enumerate(data, start=1)
                    if i % 5 != 0]
    history_tbl = ezr.Tbl([header] + history_rows)
    ranker_func = ezr.ranker(history_tbl)       # tree‑based ranker

    diffs = []  # list of tuples (row_num, actual, expected)
    for idx, row in enumerate(data, start=1):
        actual = row[goal_idx]
        if actual == "?":
            continue  # skip rows without a measured performance
        expected = ranker_func(row)
        diff = actual - expected
        diffs.append((idx, actual, expected, diff))

    if not diffs:
        flagged = []
    else:
        # Compute mean and std of differences
        mean_diff = sum(d[3] for d in diffs) / len(diffs)
        var = sum((d[3] - mean_diff) ** 2 for d in diffs) / len(diffs)
        std_diff = math.sqrt(var)

        # Threshold: 2 std deviations from mean
        threshold = 2 * std_diff if std_diff else 0

        flagged = [
            {"row": r, "column": goal_name,
             "actual": round(a, 2), "expected": round(e, 2)}
            for r, a, e, d in diffs
            if abs(d - mean_diff) > threshold
        ]
        # Sort by magnitude of deviation, keep at most 20
        flagged.sort(key=lambda x: abs(x["actual"] - x["expected"]), reverse=True)
        flagged = flagged[:20]

    # Print results
    print("== APACHE-17 ==")
    for f in flagged:
        print(f"row {f['row']}: actual={f['actual']} expected={f['expected']}")
    if not flagged:
        print("No flagged rows")

    # Machine‑checked answer
    answer = {"flagged": flagged}
    print(f"ANSWER APACHE-17 {json.dumps(answer, separators=(',', ':'))}")


def apache_18(header, data):
    """APACHE-18: flag rows with unusually rare combinations of settings."""
    # Identify setting columns (non‑goal, non‑X)
    setting_indices = [i for i, n in enumerate(header)
                       if not is_goal(n) and not n.endswith('X')]

    # Pre‑compute distances between all rows
    n_rows = len(data)
    avg_dists = [0.0] * n_rows  # index 0‑based

    for i in range(n_rows):
        row_i = data[i]
        dists = []
        for j in range(n_rows):
            if i == j:
                continue
            d = ezr.xdist(header, row_i, data[j])
            dists.append(d)
        dists.sort()
        k = min(5, len(dists))
        avg = sum(dists[:k]) / k if k else 0
        avg_dists[i] = avg

    # Compute mean and std of average distances
    mean_avg = sum(avg_dists) / n_rows
    var = sum((x - mean_avg) ** 2 for x in avg_dists) / n_rows
    std_avg = math.sqrt(var)

    # Threshold: 2 std deviations above the mean
    threshold = 2 * std_avg if std_avg else 0

    flagged_rows = []
    scores = {}
    for i, avg in enumerate(avg_dists, start=1):
        if avg > mean_avg + threshold:
            flagged_rows.append(i)
            scores[str(i)] = round(avg, 2)

    flagged_rows.sort(key=lambda r: avg_dists[r-1], reverse=True)
    flagged_rows = flagged_rows[:20]
    scores = {str(r): scores[str(r)] for r in flagged_rows}

    # Print results
    print("== APACHE-18 ==")
    for r in flagged_rows:
        print(f"row {r}: unusualness={scores[str(r)]}")
    if not flagged_rows:
        print("No flagged rows")

    # Machine‑checked answer
    answer = {"flagged": flagged_rows, "scores": scores}
    print(f"ANSWER APACHE-18 {json.dumps(answer, separators=(',', ':'))}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    header, data = read_csv(path)
    apache_17(header, data)
    apache_18(header, data)


if __name__ == "__main__":
    main()
