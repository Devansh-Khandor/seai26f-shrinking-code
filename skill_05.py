#!/usr/bin/env python3
"""
skill_05.py

A self‑contained script that analyses the NASA project dataset
to find the characteristics most strongly associated with
unusually high development effort and defect counts.

- NASA-09: Identify characteristics most strongly associated with projects
           that required unusually high development effort.
- NASA-10: Identify characteristics that distinguish projects with
           unusually high defect counts from those with better quality.

Usage:
    python skill_05.py path/to/nasa93dem.csv
"""

import csv
import sys
import math
import statistics
from pathlib import Path

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def _read_csv(path: Path):
    """Read CSV file and return list of dicts with numeric conversion."""
    data = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                if v is None:
                    cleaned[k] = None
                    continue
                val = v.strip()
                try:
                    cleaned[k] = float(val)
                except ValueError:
                    cleaned[k] = val
            data.append(cleaned)
    return data


def _numeric_columns(data):
    """
    Return a set of column names that have at least two numeric
    values (converted to float).  Columns that are exclusively
    strings are omitted.
    """
    candidates = {}
    for row in data:
        for k, v in row.items():
            if isinstance(v, (int, float)):
                candidates.setdefault(k, []).append(v)
    # Keep only those with at least two samples
    return {k for k, vals in candidates.items() if len(vals) >= 2}


def _correlation(x_vals, y_vals):
    """
    Compute Pearson correlation coefficient between two numeric
    sequences of the same length.  If fewer than two pairs,
    return None.
    """
    n = len(x_vals)
    if n < 2:
        return None
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)) / (n - 1)
    std_x = statistics.stdev(x_vals)
    std_y = statistics.stdev(y_vals)
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def _correlations_for_target(data, target_col, exclude_cols):
    """
    Compute correlation of every numeric column (excluding the ones in
    exclude_cols and the target itself) with the target column.
    Return a dict of {col: correlation}.
    """
    numeric_cols = _numeric_columns(data) - set(exclude_cols) - {target_col}
    correlations = {}
    for col in numeric_cols:
        # Build paired lists, skipping rows with missing values
        pairs = [(row[col], row[target_col]) for row in data
                 if isinstance(row.get(col), (int, float)) and
                 isinstance(row.get(target_col), (int, float))]
        x_vals, y_vals = zip(*pairs) if pairs else ([], [])
        corr = _correlation(x_vals, y_vals)
        if corr is not None:
            correlations[col] = corr
    return correlations


def _top_correlations(correlations, top_n=3):
    """Return a list of (col, corr) tuples sorted by |corr| descending."""
    sorted_cols = sorted(correlations.items(),
                         key=lambda kv: abs(kv[1]),
                         reverse=True)
    return sorted_cols[:top_n]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    data = _read_csv(csv_path)

    # --------------------------------------------------------------------- #
    # NASA-09: High effort analysis
    # --------------------------------------------------------------------- #
    effort_vals = [_row["Effort-"] for _row in data
                   if isinstance(_row.get("Effort-"), (int, float))]
    effort_mean = statistics.mean(effort_vals)
    effort_std = statistics.stdev(effort_vals) if len(effort_vals) > 1 else 0.0
    high_effort_thresh = effort_mean + 2 * effort_std
    high_effort_projects = [r for r in data
                            if isinstance(r.get("Effort-"), (int, float))
                            and r["Effort-"] > high_effort_thresh]

    eff_correlations = _correlations_for_target(data,
                                                target_col="Effort-",
                                                exclude_cols=set())
    top_eff_corr = _top_correlations(eff_correlations)

    print("== NASA-09 ==")
    print(f"Total projects: {len(data)}")
    print(f"Effort-: mean={effort_mean:.2f}, std={effort_std:.2f}")
    print(f"Projects with Effort- > mean+2σ ({high_effort_thresh:.2f}): {len(high_effort_projects)}")
    print("\nTop 3 characteristics most strongly associated with high effort:")
    for col, corr in top_eff_corr:
        print(f"  {col:12} : correlation = {corr:+.3f}")

    print("\n" + "=" * 80 + "\n")

    # --------------------------------------------------------------------- #
    # NASA-10: High defect analysis
    # --------------------------------------------------------------------- #
    defect_vals = [_row["Defects-"] for _row in data
                   if isinstance(_row.get("Defects-"), (int, float))]
    defect_mean = statistics.mean(defect_vals)
    defect_std = statistics.stdev(defect_vals) if len(defect_vals) > 1 else 0.0
    high_defect_thresh = defect_mean + 2 * defect_std
    high_defect_projects = [r for r in data
                            if isinstance(r.get("Defects-"), (int, float))
                            and r["Defects-"] > high_defect_thresh]

    defect_correlations = _correlations_for_target(data,
                                                   target_col="Defects-",
                                                   exclude_cols=set())
    top_def_corr = _top_correlations(defect_correlations)

    print("== NASA-10 ==")
    print(f"Total projects: {len(data)}")
    print(f"Defects-: mean={defect_mean:.2f}, std={defect_std:.2f}")
    print(f"Projects with Defects- > mean+2σ ({high_defect_thresh:.2f}): {len(high_defect_projects)}")
    print("\nTop 3 characteristics distinguishing high‑defect projects:")
    for col, corr in top_def_corr:
        print(f"  {col:12} : correlation = {corr:+.3f}")


if __name__ == "__main__":
    main()
