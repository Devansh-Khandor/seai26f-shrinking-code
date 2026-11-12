#!/usr/bin/env python3
"""
Skill 01: Historical project summary and outlier identification.

Usage:
    python skill_01.py <csv_path>
"""

import csv
import math
import sys

# ---------- Helper functions ----------
def parse_float(val: str):
    """Convert a CSV string to float, handling empty or missing values."""
    try:
        return float(val.strip())
    except (AttributeError, ValueError):
        return None

def mean(values):
    return sum(values) / len(values) if values else 0.0

def stddev(values, mean_val):
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((x - mean_val) ** 2 for x in values) / (len(values) - 1))

# ---------- Main analysis ----------
def analyze(csv_path):
    # Columns of interest
    numeric_cols = ['Kloc+', 'Effort-', 'Defects-']
    rows = []

    # Read CSV
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean numeric fields
            cleaned = {}
            for col in numeric_cols:
                cleaned[col] = parse_float(row.get(col))
            cleaned['id'] = row.get('id')
            # Keep raw row for printing
            cleaned['raw'] = row
            rows.append(cleaned)

    # Filter out rows with missing numeric data
    rows = [r for r in rows if all(r[col] is not None for col in numeric_cols)]

    # Compute statistics
    kloc_vals = [r['Kloc+'] for r in rows]
    effort_vals = [r['Effort-'] for r in rows]
    defect_vals = [r['Defects-'] for r in rows]

    n_projects = len(rows)

    stats = {
        'mean_kloc': mean(kloc_vals),
        'mean_effort': mean(effort_vals),
        'mean_defects': mean(defect_vals),
        'std_kloc': stddev(kloc_vals, mean(kloc_vals)),
        'std_effort': stddev(effort_vals, mean(effort_vals)),
        'std_defects': stddev(defect_vals, mean(defect_vals)),
    }

    # Best and worst effort
    best_effort_row = min(rows, key=lambda r: r['Effort-'])
    worst_effort_row = max(rows, key=lambda r: r['Effort-'])

    # Output NASA-01
    print("== NASA-01 ==")
    print(f"Projects analyzed: {n_projects}")
    print(f"Mean KLOC+: {stats['mean_kloc']:.2f}")
    print(f"Mean Effort-: {stats['mean_effort']:.2f}")
    print(f"Mean Defects-: {stats['mean_defects']:.2f}")
    print(f"Best effort (lowest Effort-): Project {best_effort_row['id']} with Effort- {best_effort_row['Effort-']:.2f}")
    print(f"Worst effort (highest Effort-): Project {worst_effort_row['id']} with Effort- {worst_effort_row['Effort-']:.2f}")
    print()

    # Identify projects with large size, low effort, few defects
    favorable_projects = []
    for r in rows:
        kloc_z = (r['Kloc+'] - stats['mean_kloc']) / stats['std_kloc'] if stats['std_kloc'] > 0 else 0
        effort_z = (r['Effort-'] - stats['mean_effort']) / stats['std_effort'] if stats['std_effort'] > 0 else 0
        defect_z = (r['Defects-'] - stats['mean_defects']) / stats['std_defects'] if stats['std_defects'] > 0 else 0
        if kloc_z > 0 and effort_z < -1 and defect_z < -1:
            favorable_projects.append((r['id'], r['Kloc+'], r['Effort-'], r['Defects-']))

    # Output NASA-02
    print("== NASA-02 ==")
    if favorable_projects:
        print("Projects with large size, low effort, and few defects:")
        for proj in favorable_projects:
            print(f"Project {proj[0]}: KLOC+={proj[1]:.2f}, Effort-={proj[2]:.2f}, Defects-={proj[3]:.2f}")
    else:
        print("No projects meet the criteria for unusually favorable outcomes.")
    print()

# ---------- Entry point ----------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    analyze(sys.argv[1])
