#!/usr/bin/env python3
"""
Skill 01: NASA-01 and NASA-02 analyses
Author: Your Name
"""

import sys
import csv
import math
import statistics

# ----------------------------
# Utility functions
# ----------------------------
def read_csv(path):
    """
    Reads the CSV at *path* and returns:
        header: list of column names
        rows: list of dict mapping column -> value (numeric where possible)
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = []
        for r in reader:
            # Convert numeric columns to float where possible
            row = {}
            for k, v in r.items():
                # Strip whitespace
                v = v.strip()
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return header, rows

def mean_std(values):
    """
    Return mean and standard deviation of a list of numeric values.
    """
    m = statistics.mean(values)
    if len(values) > 1:
        s = statistics.stdev(values)
    else:
        s = 0.0
    return m, s

def z_scores(values):
    """
    Return a list of z-scores for *values*. Handles division by zero.
    """
    m, s = mean_std(values)
    if s == 0:
        return [0.0] * len(values)
    return [(v - m) / s for v in values]

# ----------------------------
# NASA-01: Summary
# ----------------------------
def nasa_01(header, rows):
    """
    Prints a summary of typical size, effort and defects,
    along with the best and worst observed outcomes.
    """
    # Numeric columns of interest
    size_col = 'Kloc+'
    effort_col = 'Effort-'
    defect_col = 'Defects-'

    # Extract values
    size_vals = [r[size_col] for r in rows if isinstance(r[size_col], (int, float))]
    effort_vals = [r[effort_col] for r in rows if isinstance(r[effort_col], (int, float))]
    defect_vals = [r[defect_col] for r in rows if isinstance(r[defect_col], (int, float))]

    # Means, medians, std
    size_mean, size_std = mean_std(size_vals)
    effort_mean, effort_std = mean_std(effort_vals)
    defect_mean, defect_std = mean_std(defect_vals)

    size_med = statistics.median(size_vals)
    effort_med = statistics.median(effort_vals)
    defect_med = statistics.median(defect_vals)

    # Efficiency: Effort per Kloc
    for r in rows:
        if isinstance(r[size_col], (int, float)) and isinstance(r[effort_col], (int, float)):
            r['efficiency'] = r[effort_col] / r[size_col] if r[size_col] != 0 else math.inf

    efficiencies = [r['efficiency'] for r in rows if 'efficiency' in r]

    # Best and worst
    best_row = min(rows, key=lambda r: r['efficiency'] if 'efficiency' in r else math.inf)
    worst_row = max(rows, key=lambda r: r['efficiency'] if 'efficiency' in r else -math.inf)

    print("== NASA-01 ==")
    print(f"Typical project size (KLOC+): mean={size_mean:.2f}, median={size_med:.2f}, std={size_std:.2f}")
    print(f"Typical development effort (Effort-): mean={effort_mean:.2f}, median={effort_med:.2f}, std={effort_std:.2f}")
    print(f"Typical defect count (Defects-): mean={defect_mean:.2f}, median={defect_med:.2f}, std={defect_std:.2f}")
    print(f"Best outcome: Project ID={int(best_row.get('idX',0))}, efficiency={best_row['efficiency']:.4f}, size={best_row[size_col]:.2f}, effort={best_row[effort_col]:.2f}, defects={best_row[defect_col]:.2f}")
    print(f"Worst outcome: Project ID={int(worst_row.get('idX',0))}, efficiency={worst_row['efficiency']:.4f}, size={worst_row[size_col]:.2f}, effort={worst_row[effort_col]:.2f}, defects={worst_row[defect_col]:.2f}")
    print()  # blank line for readability

# ----------------------------
# NASA-02: Outlier identification
# ----------------------------
def nasa_02(header, rows):
    """
    Identify projects that achieved unusually favorable combinations
    of software size, low development effort, and fewer defects.
    """
    size_col = 'Kloc+'
    effort_col = 'Effort-'
    defect_col = 'Defects-'

    size_vals = [r[size_col] for r in rows if isinstance(r[size_col], (int, float))]
    effort_vals = [r[effort_col] for r in rows if isinstance(r[effort_col], (int, float))]
    defect_vals = [r[defect_col] for r in rows if isinstance(r[defect_col], (int, float))]

    size_z = z_scores(size_vals)
    effort_z = z_scores(effort_vals)
    defect_z = z_scores(defect_vals)

    # Combine back into rows
    for idx, r in enumerate(rows):
        r['_size_z'] = size_z[idx] if isinstance(r[size_col], (int, float)) else None
        r['_effort_z'] = effort_z[idx] if isinstance(r[effort_col], (int, float)) else None
        r['_defect_z'] = defect_z[idx] if isinstance(r[defect_col], (int, float)) else None

    # Define favorable: all three z < -1.0 (i.e., >1 std below mean)
    favorable = [r for r in rows if r.get('_size_z', 0) < -1.0 and r.get('_effort_z', 0) < -1.0 and r.get('_defect_z', 0) < -1.0]

    # If none found, relax to any two metrics
    if not favorable:
        favorable = [r for r in rows if (r.get('_size_z', 0) < -1.0 and r.get('_effort_z', 0) < -1.0) or
                                  (r.get('_size_z', 0) < -1.0 and r.get('_defect_z', 0) < -1.0) or
                                  (r.get('_effort_z', 0) < -1.0 and r.get('_defect_z', 0) < -1.0)]

    print("== NASA-02 ==")
    if not favorable:
        print("No projects met the favorable criteria.")
    else:
        for r in favorable:
            print(f"Project ID={int(r.get('idX',0))}, size={r[size_col]:.2f}, effort={r[effort_col]:.2f}, defects={r[defect_col]:.2f}")
    print()

# ----------------------------
# Main entry
# ----------------------------
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        header, rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    nasa_01(header, rows)
    nasa_02(header, rows)

if __name__ == "__main__":
    main()
