#!/usr/bin/env python3
"""
skill_01.py

This script implements the NASA-01 and NASA-02 analyses on the NASA software project dataset.

Usage:
    python skill_01.py <csv_path>
"""

import sys
import csv
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------

def parse_csv(csv_path: Path):
    """
    Read the CSV and return a list of rows (dict) and a list of numeric column names.
    """
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        # Identify numeric columns by attempting to convert sample rows
        numeric_cols = []
        for col in reader.fieldnames:
            try:
                # Try first non-empty value
                for r in rows:
                    val = r[col]
                    if val != '':
                        float(val.replace(',', ''))
                        numeric_cols.append(col)
                        break
            except (ValueError, TypeError):
                continue
    return rows, numeric_cols

def to_floats(rows, cols):
    """Return numpy array of shape (n_samples, n_features) for given cols."""
    arr = []
    for r in rows:
        row_vals = []
        for c in cols:
            val = r[c]
            if val == '':
                row_vals.append(np.nan)
            else:
                try:
                    row_vals.append(float(val.replace(',', '')))
                except ValueError:
                    row_vals.append(np.nan)
        arr.append(row_vals)
    return np.array(arr, dtype=float)

def mean_median(arr):
    """Return mean and median ignoring NaNs."""
    return np.nanmean(arr), np.nanmedian(arr)

def best_worst(rows, eff_col, defect_col):
    """Return best and worst rows based on Effort and Defects."""
    # best: lowest Effort + lowest Defects (sum of ranks)
    eff_arr = np.array([float(r[eff_col]) if r[eff_col] != '' else np.nan for r in rows])
    defect_arr = np.array([float(r[defect_col]) if r[defect_col] != '' else np.nan for r in rows])

    # Combine ranks: lower rank is better
    eff_rank = np.argsort(np.argsort(eff_arr))
    defect_rank = np.argsort(np.argsort(defect_arr))
    total_rank = eff_rank + defect_rank

    best_idx = np.nanargmin(total_rank)
    worst_idx = np.nanargmax(total_rank)

    return rows[best_idx], rows[worst_idx]

def favorable_projects(rows, size_col, eff_col, defect_col):
    """Identify projects with high size but low effort & defects."""
    size_arr = np.array([float(r[size_col]) if r[size_col] != '' else np.nan for r in rows])
    eff_arr = np.array([float(r[eff_col]) if r[eff_col] != '' else np.nan for r in rows])
    defect_arr = np.array([float(r[defect_col]) if r[defect_col] != '' else np.nan for r in rows])

    size_median = np.nanmedian(size_arr)
    eff_median = np.nanmedian(eff_arr)
    defect_median = np.nanmedian(defect_arr)

    fav = []
    for r in rows:
        s = float(r[size_col]) if r[size_col] != '' else np.nan
        e = float(r[eff_col]) if r[eff_col] != '' else np.nan
        d = float(r[defect_col]) if r[defect_col] != '' else np.nan
        if math.isnan(s) or math.isnan(e) or math.isnan(d):
            continue
        if s >= size_median and e <= eff_median and d <= defect_median:
            fav.append(r)
    return fav

def format_float(val, precision=2):
    """Format a numeric value to given precision."""
    if isinstance(val, (float, np.floating)):
        return f"{val:.{precision}f}"
    return str(val)

def print_nasa_01(rows, numeric_cols):
    """Print the summary for NASA-01."""
    print("== NASA-01 ==")

    # Use Kloc+ for size, Effort- for effort, Defects- for defects
    size_col = 'Kloc+'
    eff_col = 'Effort-'
    defect_col = 'Defects-'

    # Convert to arrays
    size_arr = to_floats(rows, [size_col]).flatten()
    eff_arr = to_floats(rows, [eff_col]).flatten()
    defect_arr = to_floats(rows, [defect_col]).flatten()

    size_mean, size_median = mean_median(size_arr)
    eff_mean, eff_median = mean_median(eff_arr)
    defect_mean, defect_median = mean_median(defect_arr)

    print(f"Typical project size: mean {format_float(size_mean)} KLOC, median {format_float(size_median)} KLOC")
    print(f"Typical development effort: mean {format_float(eff_mean)} person‑months, median {format_float(eff_median)} person‑months")
    print(f"Typical defect counts: mean {format_float(defect_mean)}, median {format_float(defect_median)}")

    # Best and worst outcomes
    best, worst = best_worst(rows, eff_col, defect_col)
    print("\nBest outcome:")
    print(f"  Project ID {best['idX']} – Effort {best[eff_col]}, Defects {best[defect_col]}")
    print("\nWorst outcome:")
    print(f"  Project ID {worst['idX']} – Effort {worst[eff_col]}, Defects {worst[defect_col]}")
    print()  # blank line for separation

def print_nasa_02(rows):
    """Print the favorable projects for NASA-02."""
    print("== NASA-02 ==")
    fav = favorable_projects(rows, 'Kloc+', 'Effort-', 'Defects-')
    if not fav:
        print("No projects met the favorable criteria.")
        return

    print("Favorable projects (high size, low effort, low defects):")
    for r in fav:
        print(f"  ID {r['idX']} – KLOC {r['Kloc+']}, Effort {r['Effort-']}, Defects {r['Defects-']}")
    print()

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: {csv_path} does not exist or is not a file.")
        sys.exit(1)

    rows, numeric_cols = parse_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    print_nasa_01(rows, numeric_cols)
    print_nasa_02(rows)

if __name__ == "__main__":
    main()
