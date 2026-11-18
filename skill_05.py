#!/usr/bin/env python3
"""
skill_05.py – CAR-09 and CAR-10

Implements the two analyses requested by the client:

  CAR-09: Identify rows with unusually poor fuel efficiency and rank
          the other characteristics by their association with low Mpg+.

  CAR-10: Identify the common characteristics of vehicles with
          unfavorable overall performance and how they differ from
          better-performing vehicles.
"""

import csv
import json
import math
import sys

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def parse_value(val):
    """Parse a CSV cell value. Return float or None if missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None


def compute_stats(values):
    """Return dict with mean, median, min, max."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    return {
        'mean': mean,
        'median': median,
        'min': sorted_vals[0],
        'max': sorted_vals[-1],
    }


def pearson_corr(x_vals, y_vals):
    """Return Pearson correlation coefficient between two lists of numbers."""
    if not x_vals or not y_vals or len(x_vals) < 2:
        return 0.0
    n = len(x_vals)
    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n
    num = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_vals))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_vals))
    den = den_x * den_y
    if den == 0:
        return 0.0
    return num / den


def mode_value(values):
    """Return the most common value in a list. Tie → first encountered."""
    freq = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    max_count = max(freq.values())
    for v, cnt in freq.items():
        if cnt == max_count:
            return v
    return None  # should never reach


# ------------------------------------------------------------------
# Analysis helpers
# ------------------------------------------------------------------
def get_columns(rows):
    """Return header columns from the first row of raw rows."""
    if not rows:
        return []
    return list(rows[0].keys())


def is_goal(col):
    return col.endswith('+') or col.endswith('-')


def is_input(col):
    return not is_goal(col) and not col.endswith('X')


def compute_performance(row):
    """Return overall performance score if all goal columns present."""
    try:
        return -row['Lbs-'] + row['Acc+'] + row['Mpg+']
    except (TypeError, KeyError):
        return None


# ------------------------------------------------------------------
# CAR-09 implementation
# ------------------------------------------------------------------
def car_09(rows, parsed_rows, all_columns):
    print("== CAR-09 ==")

    # Find rows with lowest Mpg+
    mpg_values = [(i + 1, r['Mpg+']) for i, r in enumerate(parsed_rows) if r['Mpg+'] is not None]
    mpg_values.sort(key=lambda x: x[1])  # ascending
    poor_rows = [row_num for row_num, _ in mpg_values[:5]]  # at least 5

    # Association ranking
    # Use numeric columns only for correlation; categorical via mean range
    association = {}
    # Prepare data lists
    for col in all_columns:
        if not is_input(col):
            continue
        # Collect pairs where both Mpg+ and col present
        vals_x = []
        vals_y = []
        for r in parsed_rows:
            if r['Mpg+'] is None:
                continue
            v = r[col]
            if v is None:
                continue
            vals_y.append(r['Mpg+'])
            vals_x.append(v)
        if not vals_x:
            continue

        # Determine if numeric or categorical
        if all(isinstance(v, (int, float)) for v in vals_x):
            corr = pearson_corr(vals_x, vals_y)
            association[col] = abs(corr)
        else:
            # categorical: compute range of mean Mpg+ across categories
            cat_to_vals = {}
            for x, y in zip(vals_x, vals_y):
                cat_to_vals.setdefault(x, []).append(y)
            means = [sum(vs) / len(vs) for vs in cat_to_vals.values()]
            if len(means) < 2:
                assoc_score = 0.0
            else:
                assoc_score = max(means) - min(means)
            association[col] = assoc_score

    # Sort by association descending, take top 3
    sorted_association = sorted(association.items(), key=lambda kv: kv[1], reverse=True)
    ranking = [col for col, _ in sorted_association[:3]]

    # Print some results
    print(f"Poor rows (lowest Mpg+): {poor_rows}")
    print(f"Ranking of characteristics: {ranking}")

    answer = {
        "poor": poor_rows,
        "ranking": ranking
    }
    print(f"ANSWER CAR-09 {json.dumps(answer, separators=(',', ':'))}")


# ------------------------------------------------------------------
# CAR-10 implementation
# ------------------------------------------------------------------
def car_10(rows, parsed_rows, all_columns):
    print("\n== CAR-10 ==")

    # Compute performance scores for all rows with complete goal data
    scored = [(i + 1, compute_performance(r)) for i, r in enumerate(parsed_rows)
              if compute_performance(r) is not None]
    scored.sort(key=lambda x: x[1])  # ascending

    n_total = len(scored)
    # Define bad and good as bottom and top 20% (or at least 5)
    def get_top_n(sorted_list, n):
        return [idx for idx, _ in sorted_list[:n]]

    def get_bottom_n(sorted_list, n):
        return [idx for idx, _ in sorted_list[-n:]]

    bottom_count = max(5, n_total * 20 // 100)
    top_count = max(5, n_total * 20 // 100)

    bad_rows = get_bottom_n(scored, bottom_count)
    good_rows = get_top_n(scored[::-1], top_count)  # reverse to get top

    # Typical values for bad and good
    def typical(col, indices):
        vals = [parsed_rows[i - 1][col] for i in indices if parsed_rows[i - 1][col] is not None]
        if not vals:
            return None
        if all(isinstance(v, (int, float)) for v in vals):
            mean = sum(vals) / len(vals)
            return round(mean, 3)
        else:
            return mode_value(vals)

    # Compute differences
    diff_list = []
    for col in all_columns:
        if not is_input(col):
            continue
        bad_val = typical(col, bad_rows)
        good_val = typical(col, good_rows)
        if bad_val is None or good_val is None:
            continue
        if isinstance(bad_val, (int, float)) and isinstance(good_val, (int, float)):
            diff = abs(bad_val - good_val)
        else:
            diff = 0.0 if bad_val == good_val else 1.0
        diff_list.append({"feature": col, "bad": bad_val, "good": good_val, "diff": diff})

    diff_list.sort(key=lambda d: d["diff"], reverse=True)
    differences = [{"feature": d["feature"], "bad": d["bad"], "good": d["good"]} for d in diff_list[:3]]

    # Print some results
    print(f"Bad rows (low performance): {bad_rows}")
    print(f"Good rows (high performance): {good_rows}")
    print(f"Differences: {differences}")

    answer = {
        "bad": bad_rows,
        "good": good_rows,
        "differences": differences
    }
    print(f"ANSWER CAR-10 {json.dumps(answer, separators=(',', ':'))}")


# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    if not raw_rows:
        print("No data found.")
        return

    all_columns = get_columns(raw_rows)

    # Parse numeric values
    parsed_rows = []
    for r in raw_rows:
        parsed = {}
        for k, v in r.items():
            parsed[k] = parse_value(v)
        parsed_rows.append(parsed)

    car_09(raw_rows, parsed_rows, all_columns)
    car_10(raw_rows, parsed_rows, all_columns)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
