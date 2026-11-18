#!/usr/bin/env python3
"""
skill_01.py

Runs two analysis requests on an Apache configuration CSV:
  APACHE-01 – summary statistics of goal columns.
  APACHE-02 – identify top-performing rows and the settings that distinguish them.

The CSV must be supplied as the only command‑line argument.
"""

import csv
import json
import math
import sys
from collections import defaultdict

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def to_float(v: str):
    """Convert a CSV value to float if possible."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def median(lst):
    """Return median of a list of numbers."""
    n = len(lst)
    if n == 0:
        return None
    s = sorted(lst)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]

    # Read CSV, keep row numbers (1‑based)
    rows = []  # list of (row_number, dict_of_cells)
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for rn, rec in enumerate(reader, start=1):
            rows.append((rn, rec))

    fieldnames = reader.fieldnames

    # Identify goal columns (+/-)
    goal_cols = [c for c in fieldnames if c.endswith('+') or c.endswith('-')]

    # ------------------------------------------------------------------
    # APACHE-01: summary of goal columns
    # ------------------------------------------------------------------
    stats = {}
    best_row, worst_row = {}, {}
    for col in goal_cols:
        values = []
        row_nums = []
        for rn, rec in rows:
            val = to_float(rec[col])
            if val is not None:
                values.append(val)
                row_nums.append(rn)
        if not values:
            continue

        n = len(values)
        mean = sum(values) / n
        med = median(values)
        mn = min(values)
        mx = max(values)

        # For direction: '-' means minimize, '+' means maximize
        if col.endswith('-'):
            best_val, worst_val = mn, mx
            best_rn = row_nums[values.index(mn)]
            worst_rn = row_nums[values.index(mx)]
        else:
            best_val, worst_val = mx, mn
            best_rn = row_nums[values.index(mx)]
            worst_rn = row_nums[values.index(mn)]

        stats[col] = {
            "mean": round(mean, 6),
            "median": round(med, 6),
            "min": mn,
            "max": mx,
            "best": best_val,
            "worst": worst_val
        }
        best_row[col] = best_rn
        worst_row[col] = worst_rn

    # Output
    print("== APACHE-01 ==")
    # Three lines of results
    for col, s in stats.items():
        print(f"{col}: mean={s['mean']} median={s['median']} min={s['min']} max={s['max']}")
    print(f"Best row (min performance): {best_row.get('Performance-', 'N/A')}")
    print(f"Worst row (max performance): {worst_row.get('Performance-', 'N/A')}")

    # Machine‑checked answer
    print(
        "ANSWER APACHE-01",
        json.dumps(
            {"Performance-": stats.get("Performance-", {})},
            separators=(",", ":")
        ),
    )

    # ------------------------------------------------------------------
    # APACHE-02: top rows and distinguishing settings
    # ------------------------------------------------------------------
    if "Performance-" not in goal_cols:
        print("\n== APACHE-02 ==")
        print("No goal column found.")
        print("ANSWER APACHE-02", json.dumps({"top": [], "settings": []}, separators=(",", ":")))
        return

    # Gather all rows with a numeric Performance- value
    perf_data = [(rn, to_float(rec["Performance-"])) for rn, rec in rows if to_float(rec["Performance-"]) is not None]
    perf_data.sort(key=lambda x: x[1])  # ascending (better is smaller)

    N = len(perf_data)
    top_k = max(1, math.ceil(0.05 * N))  # top 5% (rounded up)
    top_rows = [rn for rn, _ in perf_data[:top_k]]
    rest_rows = [rn for rn, _ in perf_data[top_k:]]

    # Settings: all columns that are not goal columns and not marked X
    setting_cols = [c for c in fieldnames if c not in goal_cols and not c.endswith("X")]

    settings_stats = []
    for col in setting_cols:
        # Count '1' occurrences in top and rest
        top_ones = sum(1 for rn in top_rows if rows[rn - 1][1][col] == "1")
        rest_ones = sum(1 for rn in rest_rows if rows[rn - 1][1][col] == "1")
        top_share = top_ones / len(top_rows) if top_rows else 0.0
        rest_share = rest_ones / len(rest_rows) if rest_rows else 0.0
        settings_stats.append(
            {
                "setting": col,
                "top_share": round(top_share, 3),
                "rest_share": round(rest_share, 3),
            }
        )

    # Sort by distinguishing power (difference in shares)
    settings_stats.sort(key=lambda x: x["top_share"] - x["rest_share"], reverse=True)
    # Keep at least 2 entries
    top_settings = settings_stats[:max(2, len(settings_stats))]

    # Output
    print("\n== APACHE-02 ==")
    print(f"Top {top_k} rows (best performance): {top_rows}")
    print("Most distinguishing settings:")
    for s in top_settings:
        print(
            f"{s['setting']}: top_share={s['top_share']} rest_share={s['rest_share']}"
        )

    # Machine‑checked answer
    print(
        "ANSWER APACHE-02",
        json.dumps(
            {"top": top_rows, "settings": top_settings},
            separators=(",", ":"),
        ),
    )


if __name__ == "__main__":
    main()
