#!/usr/bin/env python3
"""
skill_08.py

Usage:
    python skill_08.py <path to Apache_AllMeasurements.csv>

This module implements the APACHE‑15 and APACHE‑16 analyses.
"""

import json
import sys
import math

# Reuse helpers from skill_01 for parsing
import skill_01 as s01


def goal_columns(header):
    """Return list of columns that are goals."""
    return [c for c in header if s01.is_goal(c)]


def setting_columns(header):
    """Return list of columns that are settings (not goals, not ending in 'X')."""
    return [c for c in header if not s01.is_goal(c) and not c.endswith('X')]


def compute_mean_std(values):
    """Return (mean, std) for a list of numeric values."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return mean, std


def find_worst_poor_row(rows, goal_col, threshold):
    """Return the 1‑based index of the poorest row exceeding threshold."""
    worst_idx = None
    worst_val = None
    for idx, r in enumerate(rows, start=1):
        val = r.get(goal_col)
        if val is None or val <= threshold:
            continue
        if worst_val is None or val > worst_val:
            worst_val = val
            worst_idx = idx
    return worst_idx, worst_val


def differing_settings(r1, r2, settings):
    """
    Return list of setting names that differ between two rows
    (ignore None values; if either side is None, treat as difference).
    """
    diff = []
    for s in settings:
        v1 = r1.get(s)
        v2 = r2.get(s)
        if v1 != v2:
            diff.append(s)
    return diff


def best_alternative_by_performance(rows, base_idx, goal_col, settings, max_diff=2):
    """Find alternative row differing in <= max_diff settings with lowest goal value."""
    base_row = rows[base_idx - 1]
    base_val = base_row.get(goal_col)
    best = None
    best_val = None
    best_diff = None
    for idx, r in enumerate(rows, start=1):
        if idx == base_idx:
            continue
        # skip rows with missing goal
        if r.get(goal_col) is None:
            continue
        diff = differing_settings(base_row, r, settings)
        if 0 < len(diff) <= max_diff:
            val = r.get(goal_col)
            if val is None:
                continue
            if best_val is None or val < best_val:
                best_val = val
                best = idx
                best_diff = diff
    return best, best_val, best_diff


def best_alternative_closer_to_target(rows, base_idx, goal_col, target, settings, min_diff=1):
    """
    Among rows that are closer to target than the base row,
    find the one with the fewest differing settings (>= min_diff).
    """
    base_row = rows[base_idx - 1]
    base_val = base_row.get(goal_col)
    if base_val is None:
        return None, None, None
    best = None
    best_diff = None
    best_closeness = None
    for idx, r in enumerate(rows, start=1):
        if idx == base_idx:
            continue
        if r.get(goal_col) is None:
            continue
        diff = differing_settings(base_row, r, settings)
        if len(diff) < min_diff:
            continue
        val = r.get(goal_col)
        # Check closeness: absolute difference to target
        base_dist = abs(base_val - target)
        alt_dist = abs(val - target)
        if alt_dist < base_dist:
            if best_diff is None or len(diff) < len(best_diff):
                best_diff = diff
                best = idx
                best_closeness = alt_dist
    return best, best_diff, best_closeness


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = s01.parse_csv(csv_path)

    # Identify goal and settings
    goal_cols = goal_columns(header)
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    goal_col = goal_cols[0]
    settings = setting_columns(header)

    # Gather goal values for stats
    goal_values = [r[goal_col] for r in rows if r[goal_col] is not None]
    mean, std = compute_mean_std(goal_values)

    # Threshold for poor performance
    poor_threshold = mean + 1.96 * std
    poor_idx, poor_val = find_worst_poor_row(rows, goal_col, poor_threshold)
    if poor_idx is None:
        # Fallback: pick row with highest goal value
        poor_idx, poor_val = max(
            ((idx, r[goal_col]) for idx, r in enumerate(rows, start=1) if r[goal_col] is not None),
            key=lambda kv: kv[1]
        )

    # ---------- APACHE-15 ----------
    print("== APACHE-15 ==")
    alt15, alt15_val, diff15 = best_alternative_by_performance(
        rows, poor_idx, goal_col, settings, max_diff=2
    )
    if alt15 is None:
        # No better alternative found; fall back to same row
        alt15, alt15_val, diff15 = poor_idx, poor_val, []

    # Prepare changes dict
    changes = {s: [rows[poor_idx - 1][s], rows[alt15 - 1][s]] for s in diff15}

    print(f"Poor row: {poor_idx} (performance {poor_val})")
    print(f"Alternative row: {alt15} (performance {alt15_val})")
    print(f"Changes: {json.dumps(changes)}")
    print(f"Performance before/after: {poor_val} -> {alt15_val}")

    apache15_answer = {
        "row": poor_idx,
        "alternative": alt15,
        "changes": changes,
        "performance": {"before": poor_val, "after": alt15_val},
    }
    print("ANSWER APACHE-15", json.dumps(apache15_answer, separators=(",", ":")))

    # ---------- APACHE-16 ----------
    print("\n== APACHE-16 ==")
    target_value = mean - 1.96 * std  # better (lower) target
    alt16, diff16, closeness = best_alternative_closer_to_target(
        rows, poor_idx, goal_col, target_value, settings, min_diff=1
    )
    if alt16 is None:
        # Fallback: pick best row overall
        best_overall_idx, best_overall_val = max(
            ((idx, r[goal_col]) for idx, r in enumerate(rows, start=1) if r[goal_col] is not None),
            key=lambda kv: kv[1]
        )
        alt16 = best_overall_idx
        diff16 = differing_settings(rows[poor_idx - 1], rows[alt16 - 1], settings)

    changed = diff16 if diff16 else []

    print(f"Poor row: {poor_idx} (performance {poor_val})")
    print(f"Target performance: {target_value:.2f}")
    print(f"Alternative row: {alt16} (performance {rows[alt16 - 1][goal_col]})")
    print(f"Changed settings: {changed}")
    print(f"Distance to target: {abs(rows[alt16 - 1][goal_col] - target_value):.2f}")

    apache16_answer = {
        "row": poor_idx,
        "target": target_value,
        "alternative": alt16,
        "changed": changed,
    }
    print("ANSWER APACHE-16", json.dumps(apache16_answer, separators=(",", ":")))


if __name__ == "__main__":
    main()
