#!/usr/bin/env python3
"""
skill_05.py

Usage:
    python skill_05.py <csv_path>
"""

import csv
import json
import sys
from collections import defaultdict, Counter
from statistics import mean


def is_goal_col(col_name):
    return col_name.endswith('+') or col_name.endswith('-')


def is_input_col(col_name):
    return not col_name.endswith('X') and not is_goal_col(col_name)


def parse_value(val, col_name):
    """Parse a CSV cell value.
    - '?' -> None
    - Goal columns: float
    - Input columns: int
    """
    if val == '?':
        return None
    if is_goal_col(col_name):
        try:
            return float(val)
        except ValueError:
            return None
    else:
        try:
            return int(val)
        except ValueError:
            return None


def main(csv_path):
    # ------------------------------------------------------------------ #
    # Read CSV
    # ------------------------------------------------------------------ #
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows_raw = list(reader)

    # Parse all rows
    parsed_rows = []
    for row in rows_raw:
        parsed = {}
        for col in fieldnames:
            parsed[col] = parse_value(row[col], col)
        parsed_rows.append(parsed)

    # Identify columns
    goal_cols = [c for c in fieldnames if is_goal_col(c)]
    input_cols = [c for c in fieldnames if is_input_col(c)]

    # Choose the performance- column (minimization target)
    perf_col = None
    for c in goal_cols:
        if c.endswith('-'):
            perf_col = c
            break
    if not perf_col and goal_cols:
        perf_col = goal_cols[0]
    if not perf_col:
        print("No goal column found.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Utility: filter rows with valid performance
    # ------------------------------------------------------------------ #
    valid_indices = [i for i, r in enumerate(parsed_rows, start=1)
                     if r[perf_col] is not None]

    # ------------------------------------------------------------------ #
    # APACHE-09: Poor rows and ranking
    # ------------------------------------------------------------------ #
    print("== APACHE-09 ==")

    # Determine poor rows: top 5 highest performance- values (worst)
    perf_values = [(i, parsed_rows[i-1][perf_col]) for i in valid_indices]
    perf_values.sort(key=lambda x: x[1], reverse=True)  # descending
    poor_rows = [idx for idx, _ in perf_values[:5]]

    # Ranking of input columns by association strength
    ranking = []
    for col in input_cols:
        ones = [parsed_rows[i-1][perf_col] for i in valid_indices
                if parsed_rows[i-1][col] == 1]
        zeros = [parsed_rows[i-1][perf_col] for i in valid_indices
                 if parsed_rows[i-1][col] == 0]
        if not ones or not zeros:
            continue
        mean_ones = mean(ones)
        mean_zeros = mean(zeros)
        diff = abs(mean_ones - mean_zeros)
        ranking.append((diff, col))
    ranking.sort(reverse=True)
    ranking_cols = [c for _, c in ranking[:10]]  # at least 3

    # Textual output
    print(f"Poor rows (worst 5): {poor_rows}")
    print(f"Ranking of settings by mean difference: {ranking_cols}")

    # Machine‑checked answer
    answer_09 = {
        "poor": poor_rows,
        "ranking": ranking_cols
    }
    print("ANSWER APACHE-09", json.dumps(answer_09, separators=(',', ':')))

    # ------------------------------------------------------------------ #
    # APACHE-10: Good/bad combinations
    # ------------------------------------------------------------------ #
    print("\n== APACHE-10 ==")

    # Good rows: best 5 lowest performance- values
    good_rows = [idx for idx, _ in perf_values[::-1][:5]]

    # Ensure we have at least 5 each
    if len(poor_rows) < 5 or len(good_rows) < 5:
        # If data too small, adjust to available
        pass

    # Find the most frequent pair of settings among bad rows
    pair_counts_bad = Counter()
    pair_value_counts_bad = defaultdict(Counter)  # mapping pair -> Counter of value tuples

    for idx in poor_rows:
        r = parsed_rows[idx-1]
        for i, col1 in enumerate(input_cols):
            for col2 in input_cols[i+1:]:
                val1 = r[col1]
                val2 = r[col2]
                pair = (col1, col2)
                pair_counts_bad[pair] += 1
                pair_value_counts_bad[pair][(val1, val2)] += 1

    # Select pair with maximum count among bad rows
    if pair_counts_bad:
        best_pair = max(pair_counts_bad.items(), key=lambda x: x[1])[0]
        # Determine most common value combination for this pair in bad rows
        val_combo = pair_value_counts_bad[best_pair].most_common(1)[0][0]
        settings_dict = {best_pair[0]: val_combo[0], best_pair[1]: val_combo[1]}
    else:
        best_pair = None
        settings_dict = {}

    # Compute shares
    bad_share = (pair_counts_bad[best_pair] / len(poor_rows)) if best_pair else 0.0
    good_share = 0.0
    if best_pair:
        good_pair_counts = 0
        for idx in good_rows:
            r = parsed_rows[idx-1]
            if r[best_pair[0]] == settings_dict[best_pair[0]] and r[best_pair[1]] == settings_dict[best_pair[1]]:
                good_pair_counts += 1
        good_share = (good_pair_counts / len(good_rows))

    combinations = [{
        "settings": settings_dict,
        "bad_share": round(bad_share, 3),
        "good_share": round(good_share, 3)
    }]

    # Textual output
    print(f"Bad rows (worst 5): {poor_rows}")
    print(f"Good rows (best 5): {good_rows}")
    print(f"Most frequent bad pair: {best_pair} with values {settings_dict}")
    print(f"Bad share: {bad_share:.3f}, Good share: {good_share:.3f}")

    # Machine‑checked answer
    answer_10 = {
        "bad": poor_rows,
        "good": good_rows,
        "combinations": combinations
    }
    print("ANSWER APACHE-10", json.dumps(answer_10, separators=(',', ':')))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
