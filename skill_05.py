#!/usr/bin/env python3
"""
Skill 05: NASA-09 and NASA-10 – Identify characteristics associated with
unusually high effort and defects.
Author: Your Name
"""

import sys
import csv
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# Utility: read CSV and convert numeric values where possible
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Reads a CSV file and returns:
        header: list of column names
        rows: list of dicts mapping column -> value (numeric where possible)
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                if v is None:
                    row[k] = v
                    continue
                v = v.strip()
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return header, rows

# ----------------------------------------------------------------------
# Identify numeric and categorical columns
# ----------------------------------------------------------------------
def classify_columns(rows):
    numeric_cols = []
    categorical_cols = []
    for key in rows[0].keys():
        # skip if missing entirely
        if all(r.get(key, None) is None for r in rows):
            continue
        # Determine if all non-None values are numeric
        vals = [r.get(key, None) for r in rows if r.get(key, None) is not None]
        if all(isinstance(v, (int, float)) for v in vals):
            numeric_cols.append(key)
        else:
            categorical_cols.append(key)
    return numeric_cols, categorical_cols

# ----------------------------------------------------------------------
# Compute overall statistics for numeric columns
# ----------------------------------------------------------------------
def numeric_stats(rows, numeric_cols):
    stats = {}
    for col in numeric_cols:
        vals = [r[col] for r in rows if isinstance(r.get(col, None), (int, float))]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1))
        else:
            std = 0.0
        stats[col] = (mean, std)
    return stats

# ----------------------------------------------------------------------
# Compute overall proportions for categorical columns
# ----------------------------------------------------------------------
def categorical_stats(rows, categorical_cols):
    stats = {}
    for col in categorical_cols:
        vals = [r[col] for r in rows if r.get(col, None) is not None]
        if not vals:
            continue
        total = len(vals)
        freq = defaultdict(int)
        for v in vals:
            freq[v] += 1
        probs = {k: v / total for k, v in freq.items()}
        stats[col] = probs
    return stats

# ----------------------------------------------------------------------
# Extract high‑outlier group (e.g., mean + 1 std)
# ----------------------------------------------------------------------
def get_high_group(rows, target_col, threshold_func):
    target_vals = [r[target_col] for r in rows if isinstance(r.get(target_col, None), (int, float))]
    if not target_vals:
        return []
    mean = sum(target_vals) / len(target_vals)
    if len(target_vals) > 1:
        std = math.sqrt(sum((x - mean) ** 2 for x in target_vals) / (len(target_vals) - 1))
    else:
        std = 0.0
    thresh = threshold_func(mean, std)
    high = [r for r in rows if isinstance(r.get(target_col, None), (int, float)) and r[target_col] >= thresh]
    return high, mean, std, thresh

# ----------------------------------------------------------------------
# Compute effect sizes
# ----------------------------------------------------------------------
def compute_effects(high_group, all_rows, numeric_cols, categorical_cols,
                    num_stats, cat_stats):
    effects = []

    # Numeric columns – Cohen's d
    for col in numeric_cols:
        high_vals = [r[col] for r in high_group if isinstance(r.get(col, None), (int, float))]
        if not high_vals:
            continue
        mean_high = sum(high_vals) / len(high_vals)
        mean_all, std_all = num_stats.get(col, (0.0, 0.0))
        if std_all == 0.0:
            d = 0.0
        else:
            d = (mean_high - mean_all) / std_all
        effects.append((abs(d), col, None, d))

    # Categorical columns – proportion difference
    for col in categorical_cols:
        # overall proportions
        overall = cat_stats.get(col, {})
        # high‑group proportions
        high_vals = [r[col] for r in high_group if r.get(col, None) is not None]
        total_high = len(high_vals)
        if total_high == 0:
            continue
        freq_high = defaultdict(int)
        for v in high_vals:
            freq_high[v] += 1
        for val, prob_overall in overall.items():
            prob_high = freq_high.get(val, 0) / total_high
            diff = prob_high - prob_overall
            effects.append((abs(diff), col, val, diff))
    return effects

# ----------------------------------------------------------------------
# Format and print results
# ----------------------------------------------------------------------
def print_top_effects(title, high_group, overall_mean, overall_std, thresh,
                      effects, num_rows=3):
    print(f"== {title} ==")
    print(f"Total projects: {len(high_group)}")
    print(f"Threshold used: {thresh:.2f} (mean={overall_mean:.2f}, std={overall_std:.2f})")
    if not high_group:
        print("No projects exceed threshold.\n")
        return
    # Sort by absolute effect size
    effects.sort(key=lambda x: x[0], reverse=True)
    print("\nTop associated characteristics:")
    for i, (abs_effect, col, val, effect) in enumerate(effects[:num_rows], start=1):
        if val is None:
            # numeric
            print(f"{i}. {col} – Effect size (Cohen's d) = {effect:.2f}")
        else:
            sign = "+" if effect >= 0 else "-"
            print(f"{i}. {col}='{val}' – Proportion change = {sign}{abs(effect)*100:.1f}%")
    print()

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
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

    if not rows:
        print("No data found in CSV.", file=sys.stderr)
        sys.exit(1)

    numeric_cols, categorical_cols = classify_columns(rows)
    num_stats = numeric_stats(rows, numeric_cols)
    cat_stats = categorical_stats(rows, categorical_cols)

    # NASA-09: High effort
    high_effort, eff_mean, eff_std, eff_thresh = get_high_group(
        rows, 'Effort-', lambda m, s: m + s)
    eff_effects = compute_effects(high_effort, rows, numeric_cols,
                                 categorical_cols, num_stats, cat_stats)
    print_top_effects('NASA-09', high_effort, eff_mean, eff_std, eff_thresh,
                      eff_effects)

    # NASA-10: High defects
    high_defects, def_mean, def_std, def_thresh = get_high_group(
        rows, 'Defects-', lambda m, s: m + s)
    def_effects = compute_effects(high_defects, rows, numeric_cols,
                                 categorical_cols, num_stats, cat_stats)
    print_top_effects('NASA-10', high_defects, def_mean, def_std, def_thresh,
                      def_effects)

if __name__ == "__main__":
    main()
