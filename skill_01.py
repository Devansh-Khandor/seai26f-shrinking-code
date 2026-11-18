#!/usr/bin/env python3
"""
Skill 01: Apache configuration analysis

This script implements the two requested analyses:
- APACHE-01: summary statistics for all goal columns.
- APACHE-02: identify unusually favorable configurations and the
  settings that distinguish them from the rest.

Usage:
    python skill_01.py <path-to-CSV>

The script prints a section for each request, followed by a machine‑checked
ANSWER line with a JSON payload containing the requested values.
"""

import sys
import json
import ezr  # the provided library


def read_csv(path: str):
    """
    Load the CSV file using ezr.csv.

    Returns:
        header (list[str]): column names
        rows (list[list[atom]]): all data rows (no header)
    """
    rows = list(ezr.csv(path))
    if not rows:
        raise ValueError("CSV file is empty")
    header = rows[0]
    data = rows[1:]
    return header, data


def stats_for_goal(col_values):
    """
    Compute mean, median, min, max, best, worst for a list of numeric
    values (excluding missing values).

    The best/worst values depend on the column direction:
    - '-' goal: best = min, worst = max
    - '+' goal: best = max, worst = min
    """
    n = len(col_values)
    if n == 0:
        raise ValueError("No data for goal column")
    mean = sum(col_values) / n
    sorted_vals = sorted(col_values)
    mid = n // 2
    if n % 2 == 1:
        median = sorted_vals[mid]
    else:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    mn = min(col_values)
    mx = max(col_values)
    return {
        "mean": mean,
        "median": median,
        "min": mn,
        "max": mx,
        "best": mn,
        "worst": mx,
    }


def apache_01(header, data):
    """
    Produce summary statistics for all goal columns.
    """
    # Identify goal columns (ending in + or -)
    goal_indices = [i for i, name in enumerate(header) if name.endswith(('+', '-'))]
    goal_stats = {}
    for idx in goal_indices:
        # Gather all numeric values (skip missing)
        values = [row[idx] for row in data if row[idx] != '?']
        if all(isinstance(v, (int, float)) for v in values):
            # Determine direction
            name = header[idx]
            col_stats = stats_for_goal(values)
            # Adjust best/worst depending on direction
            if name.endswith('-'):
                col_stats["best"] = min(values)
                col_stats["worst"] = max(values)
            else:
                col_stats["best"] = max(values)
                col_stats["worst"] = min(values)
            goal_stats[name] = col_stats
    # Print results
    print("== APACHE-01 ==")
    for name, s in goal_stats.items():
        print(f"{name}: mean={s['mean']:.2f} median={s['median']:.2f} "
              f"min={s['min']} max={s['max']} best={s['best']} worst={s['worst']}")
    # Machine‑checked answer
    answer = json.dumps(goal_stats, separators=(',', ':'))
    print(f"ANSWER APACHE-01 {answer}")


def apache_02(header, data):
    """
    Identify unusually favorable configurations and distinguishing settings.
    """
    # Find the performance column (only one in this dataset)
    perf_idx = next(i for i, name in enumerate(header) if name.endswith('-'))
    # Sort rows by performance (ascending: lower is better)
    indices_sorted = sorted(range(len(data)), key=lambda i: data[i][perf_idx])
    # Determine number of top rows: at least 5, at most 20, ~10% of data
    top_n = min(20, max(5, len(data) // 10))
    top_indices = indices_sorted[:top_n]
    top_numbers = [idx + 1 for idx in top_indices]  # 1‑based row numbers
    rest_indices = set(range(len(data))) - set(top_indices)

    # Compute shares for each setting column (exclude goal and X columns)
    setting_indices = [
        i for i, name in enumerate(header)
        if i not in (perf_idx,) and name[-1] != 'X'
    ]

    setting_stats = []
    for i in setting_indices:
        name = header[i]
        top_count = sum(1 for idx in top_indices if data[idx][i] == 1)
        rest_count = sum(1 for idx in rest_indices if data[idx][i] == 1)
        top_share = top_count / top_n if top_n else 0.0
        rest_share = rest_count / (len(data) - top_n) if (len(data) - top_n) else 0.0
        setting_stats.append({
            "setting": name,
            "top_share": top_share,
            "rest_share": rest_share,
        })

    # Sort settings by how distinguishing they are (difference in shares)
    setting_stats.sort(
        key=lambda s: abs(s["top_share"] - s["rest_share"]),
        reverse=True,
    )
    # Take at least two, at most all
    selected_settings = setting_stats[:max(2, min(len(setting_stats), 3))]

    # Print results
    print("== APACHE-02 ==")
    print(f"Top rows: {top_numbers}")
    for s in selected_settings:
        print(f"{s['setting']}: top_share={s['top_share']:.3f} rest_share={s['rest_share']:.3f}")

    # Machine‑checked answer
    answer = json.dumps(
        {"top": top_numbers, "settings": selected_settings},
        separators=(',', ':'),
    )
    print(f"ANSWER APACHE-02 {answer}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    header, data = read_csv(path)
    apache_01(header, data)
    apache_02(header, data)


if __name__ == "__main__":
    main()
