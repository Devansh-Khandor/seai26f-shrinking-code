#!/usr/bin/env python3
"""
Skill 01: Apache configuration analysis

This program reads an Apache_AllMeasurements.csv file (or any file with the
same column layout) and produces two analyses:

APACHE-01  – Summary statistics for every goal column.
APACHE-02  – Identification of unusually favorable configurations and the
             input settings that best distinguish them from the rest.

The program prints a header line for each analysis, a few human‑readable
lines, and ends each section with a machine‑checked answer line that
contains a JSON object.

The JSON objects are produced from the data in the file that is
supplied on the command line; they are never hard‑coded.
"""

import sys
import csv
import json
from collections import defaultdict
from statistics import mean

# ----------------------------------------------------------------------
# Helper functions for parsing and statistics
# ----------------------------------------------------------------------
def read_csv(path: str):
    """Return header and list of rows (as dict).  '?' -> None."""
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = []
        for row in rdr:
            row_dict = {}
            for h, val in zip(header, row):
                if val == "?":
                    row_dict[h] = None
                else:
                    try:
                        # try int first, then float
                        if "." in val or "e" in val.lower():
                            row_dict[h] = float(val)
                        else:
                            row_dict[h] = int(val)
                    except ValueError:
                        row_dict[h] = val
            rows.append(row_dict)
        return header, rows


def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")


def is_input(col: str) -> bool:
    return not col.endswith("X") and not is_goal(col)


def summarize_goal(values):
    """Return dict with mean, median, min, max, best, worst."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    med = (
        (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        if n % 2 == 0
        else sorted_vals[n // 2]
    )
    mn = sorted_vals[0]
    mx = sorted_vals[-1]
    # best/worst depend on direction of goal
    # we will compute outside after knowing direction
    return {
        "mean": mean(values),
        "median": med,
        "min": mn,
        "max": mx,
    }


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def main(csv_path):
    header, rows = read_csv(csv_path)

    # Identify goal columns and input columns
    goal_cols = [c for c in header if is_goal(c)]
    input_cols = [c for c in header if is_input(c)]

    # ------------------------------------------------------------------
    # APACHE-01: Summary of goal columns
    # ------------------------------------------------------------------
    print("== APACHE-01 ==")
    goal_stats = {}
    for col in goal_cols:
        values = [row[col] for row in rows if row[col] is not None]
        if not values:
            continue  # skip if no data
        stats = summarize_goal(values)
        # determine best/worst based on direction
        if col.endswith("-"):  # minimize
            best_val, worst_val = stats["min"], stats["max"]
        else:  # maximize
            best_val, worst_val = stats["max"], stats["min"]
        stats["best"] = best_val
        stats["worst"] = worst_val
        goal_stats[col] = stats
        # human‑readable lines
        print(
            f"{col}: mean={stats['mean']:.2f} med={stats['median']:.2f} "
            f"min={stats['min']:.2f} max={stats['max']:.2f} "
            f"best={stats['best']:.2f} worst={stats['worst']:.2f}"
        )

    # machine‑checked answer
    print(
        f"ANSWER APACHE-01 {json.dumps({col: stats for col, stats in goal_stats.items()})}"
    )

    # ------------------------------------------------------------------
    # APACHE-02: Unusually favorable configurations
    # ------------------------------------------------------------------
    print("\n== APACHE-02 ==")

    # Use the first goal column for performance comparison
    if not goal_cols:
        print("No goal columns found.")
        print("ANSWER APACHE-02 {}")
        return
    perf_col = goal_cols[0]
    # Sort rows by performance ascending (lower is better)
    sorted_rows = sorted(rows, key=lambda r: r[perf_col] if r[perf_col] is not None else float("inf"))
    N = len(sorted_rows)
    top_k = min(20, max(5, int(0.10 * N)))
    top_k = min(top_k, N)
    top_rows = sorted_rows[:top_k]
    rest_rows = sorted_rows[top_k:]

    # Record row numbers (1‑based)
    top_row_numbers = [rows.index(r) + 1 for r in top_rows]

    # Compute distinguishing settings
    settings_info = []
    for col in input_cols:
        # Count ones in top and rest
        top_ones = sum(1 for r in top_rows if r[col] == 1)
        rest_ones = sum(1 for r in rest_rows if r[col] == 1)
        top_share = top_ones / top_k
        rest_share = rest_ones / (N - top_k) if N - top_k > 0 else 0.0
        diff = top_share - rest_share
        settings_info.append(
            {
                "setting": col,
                "top_share": top_share,
                "rest_share": rest_share,
                "diff": diff,
            }
        )
    # Sort by difference descending, then by diff
    settings_info.sort(key=lambda d: (-d["diff"], -d["top_share"]))
    # Keep at least 2
    chosen_settings = settings_info[: max(2, len(settings_info))]
    # Prepare output format
    output_settings = [
        {
            "setting": s["setting"],
            "top_share": s["top_share"],
            "rest_share": s["rest_share"],
        }
        for s in chosen_settings
    ]

    # human‑readable lines
    print(f"Top {top_k} rows (performance best): {top_row_numbers}")
    for s in output_settings:
        print(
            f"{s['setting']}: top={s['top_share']:.2f} rest={s['rest_share']:.2f}"
        )

    # machine‑checked answer
    answer = {"top": top_row_numbers, "settings": output_settings}
    print(f"ANSWER APACHE-02 {json.dumps(answer)}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
