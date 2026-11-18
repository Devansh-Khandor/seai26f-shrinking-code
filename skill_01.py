#!/usr/bin/env python3
"""
Skill 01: Historical project analysis for NASA-01 & NASA-02

Usage:
    python skill_01.py <path-to-nasa93dem.csv>
"""

import csv
import json
import sys
from statistics import mean
from math import isnan

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def read_csv(path):
    """
    Read the CSV file and return a list of rows.
    Each row is a dict with column names as keys.
    Values are converted to float or None if missing ('?').
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v.strip() == '?':
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        # keep original string for non-numeric columns
                        parsed[k] = v
            rows.append(parsed)
    return rows

def compute_stats(values):
    """
    Compute mean, median, min, max for a list of numeric values.
    Values is a list of floats (no None).
    """
    n = len(values)
    if n == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    sorted_vals = sorted(values)
    # mean
    m = sum(values) / n
    # median
    mid = n // 2
    if n % 2 == 1:
        med = sorted_vals[mid]
    else:
        med = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return {"mean": m, "median": med, "min": sorted_vals[0], "max": sorted_vals[-1]}

def format_number(x):
    """Format a number as int if whole, else 6‑dec precision."""
    if x is None:
        return "null"
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return f"{x:.6f}".rstrip('0').rstrip('.')

# ------------------------------------------------------------
# NASA-01
# ------------------------------------------------------------

def nasa_01(rows):
    """
    Compute summary statistics for Kloc+, Effort-, Defects-.
    Returns the JSON object for the ANSWER line.
    """
    goals = {
        "Kloc+": {"direction": "max"},
        "Effort-": {"direction": "min"},
        "Defects-": {"direction": "min"}
    }

    summary = {}
    for col, meta in goals.items():
        # Extract non‑missing values
        vals = [r[col] for r in rows if r[col] is not None]
        stats = compute_stats(vals)
        # Best / worst based on direction
        if meta["direction"] == "max":
            best = stats["max"]
            worst = stats["min"]
        else:  # min
            best = stats["min"]
            worst = stats["max"]
        summary[col] = {
            "mean": stats["mean"],
            "median": stats["median"],
            "min": stats["min"],
            "max": stats["max"],
            "best": best,
            "worst": worst
        }

    # Pretty printing section
    print("== NASA-01 ==")
    for col, s in summary.items():
        print(f"{col}: mean={format_number(s['mean'])}, median={format_number(s['median'])}, "
              f"min={format_number(s['min'])}, max={format_number(s['max'])}, "
              f"best={format_number(s['best'])}, worst={format_number(s['worst'])}")
    # JSON answer line
    json_str = json.dumps(summary, separators=(',', ':'))
    print(f"ANSWER NASA-01 {json_str}")

# ------------------------------------------------------------
# NASA-02
# ------------------------------------------------------------

def nasa_02(rows):
    """
    Identify top 10 projects with favorable combination of Kloc+, Effort-, Defects-.
    The ranking is lexicographic: higher Kloc+, lower Effort-, lower Defects-.
    Returns the JSON object for the ANSWER line.
    """
    # Prepare list of (row_number, values)
    candidates = []
    for idx, r in enumerate(rows, start=1):
        if r["Kloc+"] is None or r["Effort-"] is None or r["Defects-"] is None:
            continue
        candidates.append((idx, r["Kloc+"], r["Effort-"], r["Defects-"]))

    # Sort by Kloc+ descending, Effort- ascending, Defects- ascending
    candidates.sort(key=lambda t: (-t[1], t[2], t[3]))

    top_rows = [t[0] for t in candidates[:10]]

    print("== NASA-02 ==")
    print(f"Top rows: {top_rows}")

    json_obj = {"top": top_rows}
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-02 {json_str}")

# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    rows = read_csv(path)
    nasa_01(rows)
    nasa_02(rows)

if __name__ == "__main__":
    main()
