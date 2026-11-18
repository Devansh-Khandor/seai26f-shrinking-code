#!/usr/bin/env python3
"""
skill_05.py

Entry point for APACHE-09 and APACHE-10 skill.

Usage:
    python skill_05.py <csv_path>
"""

import csv
import json
import sys
import math
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

# Reuse the CSV reading routine from skill_01
from skill_01 import read_csv


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def identify_goal_column(headers: List[str]) -> str:
    """Return the first column name that ends with '-' or '+'."""
    for col in headers:
        if col.endswith('-') or col.endswith('+'):
            return col
    raise ValueError("No goal column found.")


def identify_setting_columns(headers: List[str], goal_col: str) -> List[str]:
    """Return setting columns (exclude goal and columns ending with 'X')."""
    return [c for c in headers if c != goal_col and not c.endswith('X')]


def top_k_rows_by_goal(rows: List[Dict[str, Any]], goal_col: str,
                       k: int, reverse: bool = False) -> List[Dict[str, Any]]:
    """Return k rows with best (reverse=False) or worst (reverse=True) goal values."""
    valid = [r for r in rows if r[goal_col] is not None]
    if not valid:
        return []
    sorted_rows = sorted(valid, key=lambda r: r[goal_col], reverse=reverse)
    return sorted_rows[:k]


def proportion_1_in_rows(rows: List[Dict[str, Any]], col: str) -> float:
    """Proportion of rows where the setting col is 1 (ignoring missing)."""
    vals = [r[col] for r in rows if r[col] is not None]
    if not vals:
        return 0.0
    return sum(1 for v in vals if v == 1) / len(vals)


# --------------------------------------------------------------------------- #
# APACHE-09
# --------------------------------------------------------------------------- #

def compute_apache_09(rows: List[Dict[str, Any]],
                     headers: List[str],
                     goal_col: str,
                     setting_cols: List[str]) -> Tuple[List[int], List[str]]:
    """Identify poorly performing rows and rank settings by association with poor performance."""
    total_rows = len(rows)
    k = max(5, min(20, math.ceil(0.2 * total_rows)))
    # Ensure we don't exceed available rows
    k = min(k, len(rows))

    # Poor rows: worst goal values (descending for a min-goal)
    poor_rows = top_k_rows_by_goal(rows, goal_col, k, reverse=True)
    poor_row_nums = sorted(r['_row_num'] for r in poor_rows)

    # Compute overall proportion of 1 for each setting
    overall_props = {col: proportion_1_in_rows(rows, col) for col in setting_cols}

    # Compute proportion of 1 in poor rows
    poor_props = {col: proportion_1_in_rows(poor_rows, col) for col in setting_cols}

    # Association score: absolute difference of proportions
    assoc = {col: abs(poor_props[col] - overall_props[col]) for col in setting_cols}

    # Rank settings by association (descending)
    ranked = sorted(setting_cols, key=lambda c: assoc[c], reverse=True)[:max(3, len(setting_cols))]
    return poor_row_nums, ranked


# --------------------------------------------------------------------------- #
# APACHE-10
# --------------------------------------------------------------------------- #

def compute_apache_10(rows: List[Dict[str, Any]],
                     goal_col: str,
                     setting_cols: List[str]) -> Tuple[List[int], List[int], List[Dict[str, Any]]]:
    """Find combinations of settings that characterize bad vs good rows."""
    total_rows = len(rows)
    k = max(5, min(20, math.ceil(0.2 * total_rows)))
    k = min(k, len(rows))

    # Bad rows: worst performance
    bad_rows = top_k_rows_by_goal(rows, goal_col, k, reverse=True)
    bad_row_nums = sorted(r['_row_num'] for r in bad_rows)

    # Good rows: best performance
    good_rows = top_k_rows_by_goal(rows, goal_col, k, reverse=False)
    good_row_nums = sorted(r['_row_num'] for r in good_rows)

    # Helper to compute shares for a given pair and value pattern
    def best_pattern_for_pair(col1: str, col2: str) -> Optional[Dict[str, Any]]:
        # Count total rows with both settings present
        bad_counts = {}
        good_counts = {}
        bad_total = 0
        good_total = 0

        for r in bad_rows:
            v1 = r[col1]
            v2 = r[col2]
            if v1 is not None and v2 is not None:
                bad_total += 1
                bad_counts[(v1, v2)] = bad_counts.get((v1, v2), 0) + 1

        for r in good_rows:
            v1 = r[col1]
            v2 = r[col2]
            if v1 is not None and v2 is not None:
                good_total += 1
                good_counts[(v1, v2)] = good_counts.get((v1, v2), 0) + 1

        if bad_total == 0:
            return None  # no data for this pair

        # Find pattern with max bad share
        best_pattern = None
        best_bad_share = -1.0
        best_good_share = 1.0  # initialise high
        for pat, bad_cnt in bad_counts.items():
            bad_share = bad_cnt / bad_total
            good_share = good_counts.get(pat, 0) / good_total if good_total > 0 else 0.0
            # Prefer higher bad share, then lower good share
            if (bad_share > best_bad_share or
                (abs(bad_share - best_bad_share) < 1e-12 and good_share < best_good_share)):
                best_pattern = pat
                best_bad_share = bad_share
                best_good_share = good_share

        if best_pattern is None:
            return None
        return {
            "settings": {col1: best_pattern[0], col2: best_pattern[1]},
            "bad_share": round(best_bad_share, 6),
            "good_share": round(best_good_share, 6)
        }

    # Compute combinations for all pairs
    combos = []
    for col1, col2 in itertools.combinations(setting_cols, 2):
        res = best_pattern_for_pair(col1, col2)
        if res:
            combos.append(res)

    # Sort combos: most typical of bad first (higher bad_share), then lower good_share
    combos.sort(key=lambda d: (-d["bad_share"], d["good_share"]))

    # Keep at least one combo
    if not combos:
        combos = []

    return bad_row_nums, good_row_nums, combos


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(csv_path: str) -> None:
    rows, headers = read_csv(csv_path)

    goal_col = identify_goal_column(headers)
    setting_cols = identify_setting_columns(headers, goal_col)

    # APACHE-09
    poor_rows, ranking = compute_apache_09(rows, headers, goal_col, setting_cols)
    print("== APACHE-09 ==")
    print(f"Poor rows: {poor_rows}")
    print(f"Ranking of settings (most associated with poor first): {ranking}")
    answer_09 = {"poor": poor_rows, "ranking": ranking}
    print("ANSWER APACHE-09", json.dumps(answer_09, separators=(',', ':')))

    # APACHE-10
    bad_rows, good_rows, combos = compute_apache_10(rows, goal_col, setting_cols)
    print("\n== APACHE-10 ==")
    print(f"Bad rows: {bad_rows}")
    print(f"Good rows: {good_rows}")
    print(f"Combinations (top 5): {combos[:5]}")
    answer_10 = {"bad": bad_rows, "good": good_rows, "combinations": combos}
    print("ANSWER APACHE-10", json.dumps(answer_10, separators=(',', ':')))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
