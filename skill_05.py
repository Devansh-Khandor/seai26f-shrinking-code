#!/usr/bin/env python3
"""
skill_05.py

Usage:
    python skill_05.py <path to Apache_AllMeasurements.csv>
"""

import csv
import json
import math
import sys
from collections import defaultdict
from itertools import combinations

# Reuse helpers from skill_01
import skill_01 as s01


def compute_mean_std(values):
    """Return (mean, std) for a list of numeric values."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return mean, std


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = s01.parse_csv(csv_path)

    # Find the goal column (should be only one)
    goal_cols = [c for c in header if s01.is_goal(c)]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    goal_col = goal_cols[0]

    # Settings: all columns that are not goals and not ending in 'X'
    settings = [c for c in header if not s01.is_goal(c) and not c.endswith("X")]

    # Compute overall goal statistics
    goal_values = [r[goal_col] for r in rows if r[goal_col] is not None]
    mean, std = compute_mean_std(goal_values)

    # ==================== APACHE-09 ====================
    print("== APACHE-09 ==")

    # Threshold for unusually poor performance
    poor_threshold = mean + 1.96 * std
    poor_rows = [i + 1 for i, r in enumerate(rows)
                 if r[goal_col] is not None and r[goal_col] > poor_threshold]

    # Ensure at least 5 rows
    if len(poor_rows) < 5:
        sorted_by_val = sorted(
            [(i + 1, r[goal_col]) for i, r in enumerate(rows) if r[goal_col] is not None],
            key=lambda x: x[1],
            reverse=True,
        )
        poor_rows = [idx for idx, _ in sorted_by_val[:5]]

    # Ranking of settings by association with poor performance
    ranking = []
    for setting in settings:
        # Count 1s in poor rows
        num_poor_ones = sum(1 for idx in poor_rows if rows[idx - 1][setting] == 1)
        denom_poor = sum(1 for idx in poor_rows if rows[idx - 1][setting] is not None)
        poor_share = num_poor_ones / denom_poor if denom_poor > 0 else 0.0

        # Overall share of 1s
        num_all_ones = sum(1 for r in rows if r[setting] == 1)
        denom_all = sum(1 for r in rows if r[setting] is not None)
        overall_share = num_all_ones / denom_all if denom_all > 0 else 0.0

        diff = abs(poor_share - overall_share)
        ranking.append((setting, diff))

    ranking.sort(key=lambda x: x[1], reverse=True)
    ranking_names = [name for name, _ in ranking[:max(3, len(ranking))]]

    # Human‑readable output
    print(f"Poor rows: {poor_rows}")
    print(f"Ranking (most associated with poor): {ranking_names}")

    # Machine‑checked JSON
    apache09_answer = {"poor": poor_rows, "ranking": ranking_names}
    print("ANSWER APACHE-09", json.dumps(apache09_answer, separators=(",", ":")))

    # ==================== APACHE-10 ====================
    print("\n== APACHE-10 ==")

    # Threshold for unusually good performance (lower than mean)
    good_threshold = mean - 1.96 * std
    good_rows = [i + 1 for i, r in enumerate(rows)
                 if r[goal_col] is not None and r[goal_col] < good_threshold]

    # Ensure at least 5 rows
    if len(good_rows) < 5:
        sorted_by_val = sorted(
            [(i + 1, r[goal_col]) for i, r in enumerate(rows) if r[goal_col] is not None],
            key=lambda x: x[1],
        )
        good_rows = [idx for idx, _ in sorted_by_val[:5]]

    # Find the most frequent 2‑setting combination among poor rows
    best_combo_settings = None
    best_combo_count = 0
    best_bad_share = 0.0
    best_good_share = 0.0

    for s1, s2 in combinations(settings, 2):
        # Count combinations in poor rows
        counts_poor = defaultdict(int)
        for idx in poor_rows:
            r = rows[idx - 1]
            v1 = r[s1]
            v2 = r[s2]
            if v1 is None or v2 is None:
                continue
            counts_poor[(v1, v2)] += 1

        if not counts_poor:
            continue

        # Most common combo for this pair
        combo, count = max(counts_poor.items(), key=lambda kv: kv[1])
        if count > best_combo_count:
            best_combo_count = count
            best_combo_settings = {s1: combo[0], s2: combo[1]}

            # Shares
            best_bad_share = count / len(poor_rows) if len(poor_rows) > 0 else 0.0

            # Good share for the same combo
            good_combo_count = sum(
                1
                for idx in good_rows
                if (rows[idx - 1][s1] == combo[0] and rows[idx - 1][s2] == combo[1])
            )
            best_good_share = (
                good_combo_count / len(good_rows) if len(good_rows) > 0 else 0.0
            )

    # Fallback if no combination found (unlikely)
    if best_combo_settings is None:
        s1, s2 = settings[:2]
        best_combo_settings = {s1: 0, s2: 0}
        best_bad_share = 0.0
        best_good_share = 0.0

    # Human‑readable output
    print(f"Bad rows: {poor_rows}")
    print(f"Good rows: {good_rows}")
    print(
        f"Top bad combo: {best_combo_settings} "
        f"(bad_share={best_bad_share:.3f}, good_share={best_good_share:.3f})"
    )

    # Machine‑checked JSON
    apache10_answer = {
        "bad": poor_rows,
        "good": good_rows,
        "combinations": [
            {
                "settings": best_combo_settings,
                "bad_share": best_bad_share,
                "good_share": best_good_share,
            }
        ],
    }
    print("ANSWER APACHE-10", json.dumps(apache10_answer, separators=(",", ":")))


if __name__ == "__main__":
    main()
