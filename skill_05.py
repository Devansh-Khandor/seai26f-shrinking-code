#!/usr/bin/env python3
"""
Skill 05: Apache configuration analysis (blame)

This script implements the two requested analyses:
- APACHE-09: Identify the most strongly associated settings with poor
  Performance-.
- APACHE-10: Identify combinations of settings that frequently appear
  among poorly performing configurations and show how they differ from
  better-performing combinations.

The analysis reuses helper functions from the `ezr` library.
"""

import json
import sys
from collections import defaultdict, Counter
import ezr  # the provided library


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def read_csv(path: str):
    """Load the CSV with ezr and return header and data rows."""
    rows = list(ezr.csv(path))
    if not rows:
        raise ValueError("CSV file is empty")
    header = rows[0]
    data = rows[1:]
    return header, data


def is_goal(name: str) -> bool:
    """Return True if column name ends with '+' or '-'."""
    return name.endswith('+') or name.endswith('-')


def get_goal_indices(header):
    """Return a list of goal column indices (there may be more than one)."""
    return [i for i, n in enumerate(header) if is_goal(n)]


def get_setting_indices(header):
    """Return a list of setting column indices (non‑goal, non‑X)."""
    return [
        i
        for i, n in enumerate(header)
        if not is_goal(n) and not n.endswith('X')
    ]


# --------------------------------------------------------------------------- #
# Analysis functions
# --------------------------------------------------------------------------- #
def apache_09(header, data):
    """APACHE-09: Identify settings most strongly associated with poor Performance-."""
    # Identify the performance goal column (first goal)
    goal_idx = get_goal_indices(header)[0]
    setting_indices = get_setting_indices(header)

    # Compute mean and std of performance, ignoring missing values
    performances = [row[goal_idx] for row in data if row[goal_idx] != "?"]
    n_perf = len(performances)
    mean_perf = sum(performances) / n_perf if n_perf else 0.0
    std_perf = (
        (sum((x - mean_perf) ** 2 for x in performances) / n_perf) ** 0.5
        if n_perf > 1
        else 0.0
    )

    # Compute z‑score for each row with a known goal
    z_scores = {}
    for rn, row in enumerate(data, start=1):
        val = row[goal_idx]
        if val == "?":
            continue
        z = (val - mean_perf) / std_perf if std_perf else 0.0
        z_scores[rn] = z

    # Poor rows: top 5 by z‑score (most positive)
    poor_rows = sorted(z_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    poor_numbers = [rn for rn, _ in poor_rows]

    # Ranking of settings by difference in proportion of poor rows
    counts = defaultdict(lambda: {"count0": 0, "poor0": 0, "count1": 0, "poor1": 0})
    for rn, row in enumerate(data, start=1):
        goal_val = row[goal_idx]
        if goal_val == "?":
            continue
        is_poor = z_scores[rn] > 2  # same threshold as APACHE‑07
        for idx in setting_indices:
            val = row[idx]
            if val == "?":
                continue
            if val == 1:
                counts[idx]["count1"] += 1
                if is_poor:
                    counts[idx]["poor1"] += 1
            elif val == 0:
                counts[idx]["count0"] += 1
                if is_poor:
                    counts[idx]["poor0"] += 1

    diff_list = []
    for idx in setting_indices:
        cnt0 = counts[idx]["count0"]
        cnt1 = counts[idx]["count1"]
        if cnt0 == 0 or cnt1 == 0:
            continue  # skip if no data for one side
        prop0 = counts[idx]["poor0"] / cnt0
        prop1 = counts[idx]["poor1"] / cnt1
        diff = abs(prop1 - prop0)
        diff_list.append((diff, header[idx]))

    diff_list.sort(reverse=True)
    ranking = [name for _, name in diff_list[:3]]  # at least 3

    # ----------------------------------------------------------------------- #
    # Output
    # ----------------------------------------------------------------------- #
    print("== APACHE-09 ==")
    print("Poor rows (by z‑score):")
    for rn, z in poor_rows:
        print(f"row {rn}: z={z:.2f}")
    print("\nTop associated settings:")
    for name in ranking:
        print(name)

    # Machine‑checked answer
    answer = {"poor": poor_numbers, "ranking": ranking}
    print(f"ANSWER APACHE-09 {json.dumps(answer, separators=(',', ':'))}")


def apache_10(header, data):
    """APACHE-10: Identify common settings among poor and good configurations."""
    goal_idx = get_goal_indices(header)[0]
    setting_indices = get_setting_indices(header)

    # Compute mean and std of performance
    performances = [row[goal_idx] for row in data if row[goal_idx] != "?"]
    n_perf = len(performances)
    mean_perf = sum(performances) / n_perf if n_perf else 0.0
    std_perf = (
        (sum((x - mean_perf) ** 2 for x in performances) / n_perf) ** 0.5
        if n_perf > 1
        else 0.0
    )

    # Compute z‑score for each row with known goal
    z_scores = {}
    for rn, row in enumerate(data, start=1):
        val = row[goal_idx]
        if val == "?":
            continue
        z = (val - mean_perf) / std_perf if std_perf else 0.0
        z_scores[rn] = z

    # Poor rows: top 5 by z‑score > 2
    poor_rows = sorted(
        [(rn, z) for rn, z in z_scores.items() if z > 2], key=lambda kv: kv[1], reverse=True
    )[:5]
    poor_numbers = [rn for rn, _ in poor_rows]

    # Good rows: top 5 by z‑score <= 0 (most negative)
    good_rows = sorted(
        [(rn, z) for rn, z in z_scores.items() if z <= 0], key=lambda kv: kv[1]
    )[:5]
    good_numbers = [rn for rn, _ in good_rows]

    # Build combo statistics
    bad_combo_counts = defaultdict(Counter)
    good_combo_counts = defaultdict(Counter)

    # For each pair of settings
    for i, j in [(a, b) for a in setting_indices for b in setting_indices if a < b]:
        for rn, row in enumerate(data, start=1):
            val_i = row[i]
            val_j = row[j]
            if val_i == "?" or val_j == "?":
                continue
            val_tuple = (val_i, val_j)
            if rn in poor_numbers:
                bad_combo_counts[(i, j)][val_tuple] += 1
            if rn in good_numbers:
                good_combo_counts[(i, j)][val_tuple] += 1

    combos = []
    for key, counter_bad in bad_combo_counts.items():
        i, j = key
        total_bad = len(poor_numbers)
        total_good = len(good_numbers)
        if total_bad == 0 or total_good == 0:
            continue
        for vals, bad_cnt in counter_bad.items():
            bad_share = bad_cnt / total_bad
            good_cnt = good_combo_counts[key][vals]
            good_share = good_cnt / total_good
            if bad_share > good_share and bad_share >= 0.1:  # some threshold
                combos.append(
                    {
                        "settings": {
                            header[i]: vals[0],
                            header[j]: vals[1],
                        },
                        "bad_share": round(bad_share, 3),
                        "good_share": round(good_share, 3),
                    }
                )

    # Sort combos by bad_share descending
    combos.sort(key=lambda c: c["bad_share"], reverse=True)
    combos = combos[:5]  # at least 1, most typical of bad first

    # ----------------------------------------------------------------------- #
    # Output
    # ----------------------------------------------------------------------- #
    print("== APACHE-10 ==")
    print("Bad rows:")
    for rn in poor_numbers:
        print(f"row {rn}")
    print("\nGood rows:")
    for rn in good_numbers:
        print(f"row {rn}")

    print("\nCommon setting combinations:")
    for combo in combos:
        print(combo)

    # Machine‑checked answer
    answer = {"bad": poor_numbers, "good": good_numbers, "combinations": combos}
    print(f"ANSWER APACHE-10 {json.dumps(answer, separators=(',', ':'))}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    header, data = read_csv(path)
    apache_09(header, data)
    apache_10(header, data)


if __name__ == "__main__":
    main()
