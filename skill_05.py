#!/usr/bin/env python3
"""
Skill 5: APACHE-09 and APACHE-10

The program reads a CSV file (path supplied on the command line) and
performs two analyses:

  APACHE-09 – Identify rows with unusually poor performance and rank the
                configuration settings most strongly associated with that
                poor performance.

  APACHE-10 – Identify pairs of setting combinations that appear more
                frequently among poor rows than good rows, and quantify
                how the two groups differ on those combinations.

The output follows the exact format required by the evaluation harness:
  * a header line for each request (== APACHE-09 ==, == APACHE-10 ==)
  * at least three human‑readable lines per request
  * a machine‑checked JSON line with the required structure
"""

import csv
import sys
import json
import math
from typing import List, Dict, Any
import numpy as np

# ----------------------------------------------------------------------
# CSV helper
# ----------------------------------------------------------------------
def read_csv(path: str) -> (List[Dict[str, Any]], List[str]):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for i, row in enumerate(reader, start=1):
            cleaned = {}
            for k, v in row.items():
                if v == "?":
                    cleaned[k] = None
                else:
                    try:
                        cleaned[k] = float(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append({"row": i, "values": cleaned})
    return rows, fieldnames

# ----------------------------------------------------------------------
# Column utilities
# ----------------------------------------------------------------------
def goal_column(fieldnames: List[str]) -> str:
    for c in fieldnames:
        if c.endswith('+') or c.endswith('-'):
            return c
    return None

def setting_columns(fieldnames: List[str], goal_col: str) -> List[str]:
    return [c for c in fieldnames if c != goal_col and not c.endswith('X')]

# ----------------------------------------------------------------------
# Quantile helpers
# ----------------------------------------------------------------------
def quantile(values: List[float], q: float) -> float:
    """Return q‑th percentile (q in 0..1) of a list of floats."""
    if not values:
        return 0.0
    return float(np.percentile(values, q * 100, interpolation="linear"))

# ----------------------------------------------------------------------
# APACHE‑09 helpers
# ----------------------------------------------------------------------
def identify_poor_rows(rows: List[Dict[str, Any]], goal_col: str) -> List[Dict[str, Any]]:
    # Gather all non‑missing goal values
    vals = [r["values"][goal_col] for r in rows if r["values"][goal_col] is not None]
    if not vals:
        return []

    # Determine threshold: 80th percentile (worst 20%)
    threshold = quantile(vals, 0.80)

    # Select rows above threshold
    poor = [r for r in rows if r["values"][goal_col] is not None and r["values"][goal_col] >= threshold]
    # Ensure at least 5
    if len(poor) < 5:
        # Sort by performance descending
        poor = sorted(rows, key=lambda r: r["values"][goal_col] if r["values"][goal_col] is not None else -math.inf, reverse=True)[:5]
    return poor

def rank_settings_by_poor(rows: List[Dict[str, Any]], poor: List[Dict[str, Any]],
                          goal_col: str, settings: List[str]) -> List[str]:
    # Overall proportions
    overall_counts = {}
    for s in settings:
        ones = 0
        total = 0
        for r in rows:
            v = r["values"].get(s)
            if v is None:
                continue
            total += 1
            if v == 1:
                ones += 1
        overall_counts[s] = (ones / total) if total else 0.0

    # Poor proportions
    poor_counts = {}
    for s in settings:
        ones = 0
        total = 0
        for r in poor:
            v = r["values"].get(s)
            if v is None:
                continue
            total += 1
            if v == 1:
                ones += 1
        poor_counts[s] = (ones / total) if total else 0.0

    # Compute absolute difference
    diffs = []
    for s in settings:
        diff = abs(poor_counts[s] - overall_counts[s])
        diffs.append((diff, s))
    diffs.sort(reverse=True)
    # Return at least 3 columns
    return [s for _, s in diffs[:max(3, len(diffs))]]

# ----------------------------------------------------------------------
# APACHE‑10 helpers
# ----------------------------------------------------------------------
def split_good_bad(rows: List[Dict[str, Any]], goal_col: str,
                   good_frac: float = 0.20, bad_frac: float = 0.20) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    vals = [r["values"][goal_col] for r in rows if r["values"][goal_col] is not None]
    if not vals:
        return [], []

    # Good rows: best 20%
    good_th = quantile(vals, good_frac)
    good = [r for r in rows if r["values"][goal_col] is not None and r["values"][goal_col] <= good_th]
    if len(good) < 5:
        good = sorted(rows, key=lambda r: r["values"][goal_col] if r["values"][goal_col] is not None else math.inf)[:5]

    # Bad rows: worst 20%
    bad_th = quantile(vals, 1 - bad_frac)
    bad = [r for r in rows if r["values"][goal_col] is not None and r["values"][goal_col] >= bad_th]
    if len(bad) < 5:
        bad = sorted(rows, key=lambda r: r["values"][goal_col] if r["values"][goal_col] is not None else -math.inf, reverse=True)[:5]

    return good, bad

def best_combinations(good: List[Dict[str, Any]], bad: List[Dict[str, Any]],
                      settings: List[str], min_pair_share: float = 0.0) -> List[Dict[str, Any]]:
    """Return list of combos (pair of settings) with highest bad share."""
    combos = []
    n_good = len(good)
    n_bad = len(bad)

    # Iterate over all unordered pairs of settings
    for i in range(len(settings)):
        for j in range(i + 1, len(settings)):
            s1, s2 = settings[i], settings[j]
            # Evaluate all 4 possible value pairs
            best = None
            for v1 in (0, 1):
                for v2 in (0, 1):
                    bad_cnt = sum(1 for r in bad if r["values"].get(s1) == v1 and r["values"].get(s2) == v2)
                    good_cnt = sum(1 for r in good if r["values"].get(s1) == v1 and r["values"].get(s2) == v2)
                    bad_share = bad_cnt / n_bad if n_bad else 0.0
                    good_share = good_cnt / n_good if n_good else 0.0
                    # Keep the pair with the highest bad_share
                    if best is None or bad_share > best["bad_share"]:
                        best = {
                            "settings": {s1: v1, s2: v2},
                            "bad_share": bad_share,
                            "good_share": good_share
                        }
            # Optionally filter by minimal share
            if best and best["bad_share"] >= min_pair_share:
                combos.append(best)

    # Sort combos by bad_share descending
    combos.sort(key=lambda x: x["bad_share"], reverse=True)
    return combos[:5]  # keep up to 5

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    g_col = goal_column(fieldnames)
    if g_col is None:
        print("No goal column found.")
        sys.exit(1)

    settings = setting_columns(fieldnames, g_col)

    # ------------------ APACHE-09 ------------------
    print("== APACHE-09 ==")
    poor_rows = identify_poor_rows(rows, g_col)
    poor_nums = [r["row"] for r in poor_rows]
    print(f"Poor rows (n={len(poor_rows)}): {poor_nums[:3]}{'...' if len(poor_nums)>3 else ''}")
    # Show one example
    if poor_rows:
        r = poor_rows[0]
        print(f"Example poor row {r['row']}: {g_col}={r['values'][g_col]}")

    ranking = rank_settings_by_poor(rows, poor_rows, g_col, settings)
    print(f"Top associated settings: {ranking[:3]}{'...' if len(ranking)>3 else ''}")

    answer09 = {
        "poor": poor_nums,
        "ranking": ranking
    }
    print(f"ANSWER APACHE-09 {json.dumps(answer09, separators=(',', ':'))}")

    # ------------------ APACHE-10 ------------------
    print("\n== APACHE-10 ==")
    good_rows, bad_rows = split_good_bad(rows, g_col)
    good_nums = [r["row"] for r in good_rows]
    bad_nums = [r["row"] for r in bad_rows]
    print(f"Good rows (n={len(good_rows)}): {good_nums[:3]}{'...' if len(good_nums)>3 else ''}")
    print(f"Bad rows (n={len(bad_rows)}): {bad_nums[:3]}{'...' if len(bad_nums)>3 else ''}")

    combos = best_combinations(good_rows, bad_rows, settings)
    print(f"Top combinations (n={len(combos)}):")
    for c in combos[:2]:
        print(f"  settings={c['settings']}, bad_share={c['bad_share']:.3f}, good_share={c['good_share']:.3f}")

    answer10 = {
        "bad": bad_nums,
        "good": good_nums,
        "combinations": combos
    }
    print(f"ANSWER APACHE-10 {json.dumps(answer10, separators=(',', ':'))}")

if __name__ == "__main__":
    main()
