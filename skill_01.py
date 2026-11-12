#!/usr/bin/env python3
"""
skill_01.py

This script reads a NASA software project CSV file and answers two
requests:

NASA-01: Summarizes typical project size (Kloc+), development effort
(Effort-), and defect counts (Defects-), and reports the best and
worst observed outcomes based on a simple score = effort + defects.

NASA-02: Identifies projects that achieved unusually favorable
combinations of software size, low development effort, and fewer
defects. A project is considered favorable if its size, effort and
defect count are all below the 25th percentile for the corresponding
metric.
"""

import sys
import csv
from typing import List, Dict
import numpy as np


def read_csv(file_path: str) -> List[Dict[str, str]]:
    """
    Reads a CSV file and returns a list of dictionaries where keys are
    column names. All values are strings.
    """
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def extract_numeric_columns(rows: List[Dict[str, str]],
                            columns: List[str]) -> np.ndarray:
    """
    Extracts numeric data from specified columns, converting to float.
    Missing or non-numeric values are treated as np.nan.
    """
    data = []
    for row in rows:
        try:
            data.append(float(row[columns[0]]))
        except (ValueError, KeyError):
            data.append(np.nan)
    return np.array(data, dtype=float)


def compute_stats(values: np.ndarray) -> Dict[str, float]:
    """
    Computes mean and median of a numeric array, ignoring NaNs.
    """
    return {
        'mean': np.nanmean(values),
        'median': np.nanmedian(values)
    }


def main(csv_path: str):
    rows = read_csv(csv_path)

    # Numeric columns we care about
    id_col = 'idX'
    size_col = 'Kloc+'
    effort_col = 'Effort-'
    defect_col = 'Defects-'

    ids = np.array([int(row[id_col]) for row in rows], dtype=int)
    size = extract_numeric_columns(rows, [size_col])
    effort = extract_numeric_columns(rows, [effort_col])
    defects = extract_numeric_columns(rows, [defect_col])

    # Filter out rows with missing critical data
    mask = ~np.isnan(size) & ~np.isnan(effort) & ~np.isnan(defects)
    ids = ids[mask]
    size = size[mask]
    effort = effort[mask]
    defects = defects[mask]

    # ---- NASA-01 ----
    print("== NASA-01 ==")
    size_stats = compute_stats(size)
    effort_stats = compute_stats(effort)
    defect_stats = compute_stats(defects)

    print(f"Typical project size (KLOC): mean={size_stats['mean']:.2f}, median={size_stats['median']:.2f}")
    print(f"Typical development effort (man-months): mean={effort_stats['mean']:.2f}, median={effort_stats['median']:.2f}")
    print(f"Typical defect counts: mean={defect_stats['mean']:.2f}, median={defect_stats['median']:.2f}")

    # Score = effort + defects
    score = effort + defects
    best_idx = np.argmin(score)
    worst_idx = np.argmax(score)

    print("\nBest outcome project:")
    print(f"  ID={ids[best_idx]}, Size={size[best_idx]:.2f} KLOC, Effort={effort[best_idx]:.2f} months, Defects={defects[best_idx]:.2f}")
    print("\nWorst outcome project:")
    print(f"  ID={ids[worst_idx]}, Size={size[worst_idx]:.2f} KLOC, Effort={effort[worst_idx]:.2f} months, Defects={defects[worst_idx]:.2f}")

    # ---- NASA-02 ----
    print("\n== NASA-02 ==")

    # Compute 25th percentiles
    size_q1 = np.percentile(size, 25)
    effort_q1 = np.percentile(effort, 25)
    defects_q1 = np.percentile(defects, 25)

    favorable_mask = (size <= size_q1) & (effort <= effort_q1) & (defects <= defects_q1)
    favorable_ids = ids[favorable_mask]

    if favorable_ids.size == 0:
        print("No projects found that meet the favorable criteria.")
    else:
        print("Projects with unusually favorable size, effort, and defect counts:")
        for fid in favorable_ids:
            print(f"  ID={fid}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
