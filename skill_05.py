#!/usr/bin/env python3
"""
Skill 05: Identify characteristics associated with unusually high effort and defects.

NASA-09: For projects that required unusually high development effort, identify which recorded
project characteristics are most strongly associated with those outcomes.

NASA-10: Examine projects with unusually high defect counts and identify the characteristics
that distinguish them from projects with better quality outcomes.

Usage:
    python skill_05.py <csv_path>
"""

import csv
import math
import sys
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def parse_float(val: str | None) -> float | None:
    """Safely parse a string to float; return None on failure."""
    if val is None:
        return None
    try:
        return float(val.strip())
    except (AttributeError, ValueError):
        return None

def mean(values):
    return sum(values) / len(values) if values else 0.0

def stddev(values, mean_val):
    n = len(values)
    if n < 2:
        return 0.0
    return math.sqrt(sum((x - mean_val) ** 2 for x in values) / (n - 1))

def cohens_d(group_vals, overall_vals):
    """Cohen's d for difference between group mean and overall mean."""
    if not group_vals or not overall_vals:
        return 0.0
    g_mean = mean(group_vals)
    o_mean = mean(overall_vals)
    g_std = stddev(group_vals, g_mean)
    o_std = stddev(overall_vals, o_mean)
    pooled_std = math.sqrt((g_std ** 2 + o_std ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return (g_mean - o_mean) / pooled_std

def chi_square_for_categorical(cat_counts, total_counts):
    """
    Compute chi-square statistic for a single categorical variable.
    cat_counts: dict of category -> [count_in_group, count_not_in_group]
    total_counts: [total_in_group, total_not_in_group]
    """
    group_total, non_group_total = total_counts
    grand_total = group_total + non_group_total
    chi2 = 0.0
    for cat, (cnt_g, cnt_ng) in cat_counts.items():
        exp_g = (cnt_g + cnt_ng) * group_total / grand_total if grand_total else 0
        exp_ng = (cnt_g + cnt_ng) * non_group_total / grand_total if grand_total else 0
        if exp_g:
            chi2 += (cnt_g - exp_g) ** 2 / exp_g
        if exp_ng:
            chi2 += (cnt_ng - exp_ng) ** 2 / exp_ng
    return chi2

# --------------------------------------------------------------------------- #
# Core analysis functions
# --------------------------------------------------------------------------- #

def load_data(csv_path):
    """Read CSV and return list of row dicts with parsed numeric values."""
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = raw.copy()
            # Parse numeric columns
            for col in numeric_columns:
                row[col] = parse_float(row.get(col))
            rows.append(row)
    return rows

def identify_group(rows, target_col, threshold):
    """
    Return set of row indices that belong to the group where target_col >= threshold.
    Also return overall indices for reference.
    """
    group_idx = set()
    for i, r in enumerate(rows):
        val = r.get(target_col)
        if val is not None and val >= threshold:
            group_idx.add(i)
    return group_idx

def analyze_group(rows, group_idx, other_col):
    """
    For a given group (set of indices) return association metrics for all columns.
    Returns two dicts:
      numeric_metrics: var -> Cohen's d
      categorical_metrics: var -> chi-square
    """
    total_indices = set(range(len(rows)))
    non_group_idx = total_indices - group_idx

    # Collect overall values for numeric columns
    overall_numeric_vals = defaultdict(list)
    group_numeric_vals = defaultdict(list)
    for idx in total_indices:
        r = rows[idx]
        for col in numeric_columns:
            val = r.get(col)
            if val is not None:
                overall_numeric_vals[col].append(val)
                if idx in group_idx:
                    group_numeric_vals[col].append(val)

    numeric_metrics = {}
    for col in numeric_columns:
        if len(overall_numeric_vals[col]) >= 2:
            d = cohens_d(group_numeric_vals[col], overall_numeric_vals[col])
            numeric_metrics[col] = d

    # Categorical analysis
    categorical_metrics = {}
    for col in categorical_columns:
        # Build counts
        cat_counts = defaultdict(lambda: [0, 0])  # [in_group, not_in_group]
        for idx in total_indices:
            r = rows[idx]
            val = r.get(col, "Missing")
            if val is None:
                val = "Missing"
            if idx in group_idx:
                cat_counts[val][0] += 1
            else:
                cat_counts[val][1] += 1
        group_total = len(group_idx)
        non_group_total = len(rows) - group_total
        chi2 = chi_square_for_categorical(cat_counts, (group_total, non_group_total))
        categorical_metrics[col] = chi2

    return numeric_metrics, categorical_metrics

def print_report(title, group_size, group_threshold, target_col, numeric_metrics, categorical_metrics):
    print(f"== {title} ==")
    print(f"Projects with {target_col} >= {group_threshold:.2f} (n={group_size}):")
    print()

    # Sort numeric metrics by absolute effect size
    top_numeric = sorted(numeric_metrics.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    if top_numeric:
        print(f"Top numeric variables (Cohen's d):")
        for var, d in top_numeric:
            print(f"  {var:>10}: d = {d:+.2f}")
        print()
    else:
        print("No numeric associations found.")
        print()

    # Sort categorical metrics by chi-square
    top_categorical = sorted(categorical_metrics.items(), key=lambda kv: kv[1], reverse=True)[:5]
    if top_categorical:
        print(f"Top categorical variables (Chi-square):")
        for var, chi2 in top_categorical:
            print(f"  {var:>10}: chi2 = {chi2:.2f}")
        print()
    else:
        print("No categorical associations found.")
        print()

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Define columns
    numeric_columns = ['Kloc+', 'Effort-', 'Defects-', 'MonthsX', 'YearX', 'centerX']
    categorical_columns = [col for col in
        # We'll detect all columns from the file later
        []]  # placeholder

    # Load data
    raw_rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_fields = reader.fieldnames or []
        # Determine which columns are numeric
        numeric_columns = ['Kloc+', 'Effort-', 'Defects-', 'MonthsX', 'YearX', 'centerX']
        categorical_columns = [col for col in all_fields if col not in numeric_columns and col != 'id']

    # Re-read to parse numeric values
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for raw in reader:
            r = raw.copy()
            for col in numeric_columns:
                r[col] = parse_float(r.get(col))
            rows.append(r)

    # Compute overall stats for Effort- and Defects-
    effort_vals = [r['Effort-'] for r in rows if r['Effort-'] is not None]
    defect_vals = [r['Defects-'] for r in rows if r['Defects-'] is not None]
    mean_effort = mean(effort_vals)
    std_effort = stddev(effort_vals, mean_effort)
    mean_defects = mean(defect_vals)
    std_defects = stddev(defect_vals, mean_defects)

    # NASA-09: high effort
    eff_thresh = mean_effort + 2 * std_effort
    high_effort_idx = identify_group(rows, 'Effort-', eff_thresh)
    numeric_eff, categorical_eff = analyze_group(rows, high_effort_idx, 'Effort-')
    print_report("NASA-09", len(high_effort_idx), eff_thresh, "Effort-", numeric_eff, categorical_eff)

    # NASA-10: high defects
    def_thresh = mean_defects + 2 * std_defects
    high_defect_idx = identify_group(rows, 'Defects-', def_thresh)
    numeric_def, categorical_def = analyze_group(rows, high_defect_idx, 'Defects-')
    print_report("NASA-10", len(high_defect_idx), def_thresh, "Defects-", numeric_def, categorical_def)
