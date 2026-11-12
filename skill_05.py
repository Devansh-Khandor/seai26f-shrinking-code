#!/usr/bin/env python3
"""
Small data‑analysis skill for NASA project data – new NASA‑09 and NASA‑10.

Usage:
    python skill_05.py <csv_path>
"""

import csv
import sys
import statistics
import math
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Re‑use helpers from earlier skills
# --------------------------------------------------------------------------- #

# From skill_01.py
def load_data(csv_path: str) -> List[Dict[str, str]]:
    """Read CSV and return list of records (dicts)."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_float(value: str) -> float:
    """Convert a string to float, stripping spaces. Return NaN for empty strings."""
    try:
        return float(value.strip())
    except ValueError:
        return float('nan')


def extract_numeric_fields(records: List[Dict[str, str]]) -> Dict[str, List[float]]:
    """Return dictionary mapping numeric column name to list of values."""
    numeric = {
        'Kloc+': [],
        'Effort-': [],
        'Defects-': [],
        'MonthsX': []
    }
    for r in records:
        numeric['Kloc+'].append(parse_float(r['Kloc+']))
        numeric['Effort-'].append(parse_float(r['Effort-']))
        numeric['Defects-'].append(parse_float(r['Defects-']))
        numeric['MonthsX'].append(parse_float(r['MonthsX']))
    return numeric


def compute_stats(values: List[float]) -> Tuple[float, float, float]:
    """Return mean, median, standard deviation for a list of numbers."""
    vals = [v for v in values if not v != v]  # filter NaNs
    if not vals:
        return (float('nan'), float('nan'), float('nan'))
    mean = statistics.mean(vals)
    median = statistics.median(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return (mean, median, stdev)

# --------------------------------------------------------------------------- #
# Additional helpers
# --------------------------------------------------------------------------- #

def pearson_corr(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient between two numeric lists."""
    x_clean = [xi for xi, yi in zip(x, y) if not xi != xi and not yi != yi]
    y_clean = [yi for xi, yi in zip(x, y) if not xi != xi and not yi != yi]
    n = len(x_clean)
    if n < 2:
        return 0.0
    mean_x = statistics.mean(x_clean)
    mean_y = statistics.mean(y_clean)
    sxx = sum((xi - mean_x) ** 2 for xi in x_clean)
    syy = sum((yi - mean_y) ** 2 for yi in y_clean)
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x_clean, y_clean))
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom else 0.0


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float('nan')


# --------------------------------------------------------------------------- #
# Main routine: NASA‑09 & NASA‑10
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = load_data(csv_path)
    numeric = extract_numeric_fields(records)

    # Basic statistics
    eff_mean, eff_med, eff_sd = compute_stats(numeric['Effort-'])
    def_mean, def_med, def_sd = compute_stats(numeric['Defects-'])

    # -----------------------------------------------------------------------
    # NASA‑09 – high effort projects
    # -----------------------------------------------------------------------
    print("== NASA-09 ==")

    eff_thresh = eff_mean + 2 * eff_sd
    high_eff_projects = [
        r for r in records if parse_float(r['Effort-']) > eff_thresh
    ]
    print(f"Unusually high effort threshold: {eff_thresh:.1f} "
          f"(mean {eff_mean:.1f} + 2*sd {eff_sd:.1f})")
    print(f"High‑effort projects: {len(high_eff_projects)}")

    # Identify categorical characteristics
    all_fields = list(records[0].keys())
    numeric_fields = {'Kloc+', 'Effort-', 'Defects-', 'MonthsX'}
    categorical_fields = [f for f in all_fields
                          if f not in numeric_fields and f != 'idX']

    cat_means = {}
    for field in categorical_fields:
        sum_high = 0.0
        cnt_high = 0
        for r in high_eff_projects:
            val = parse_float(r['Effort-'])
            if not val != val:
                sum_high += val
                cnt_high += 1
        mean_high = mean([parse_float(r['Effort-']) for r in high_eff_projects]
                         if parse_float(r['Effort-']) == parse_float(r['Effort-'])]
        cat_means[field] = (mean_high, mean_high - eff_mean)

    # Top 3 categorical characteristics
    top_cat = sorted(cat_means.items(),
                     key=lambda kv: kv[1][1],
                     reverse=True)[:3]

    print("\nTop 3 characteristics most associated with high effort:")
    for field, (m, diff) in top_cat:
        print(f"  {field}: mean high‑effort effort={m:.1f}, "
              f"overall mean={eff_mean:.1f}, diff={diff:.1f}")

    # Numeric predictors of effort
    numeric_predictors = {}
    for field in numeric.keys():
        if field in {'Effort-', 'Defects-'}:
            continue
        corr = pearson_corr(numeric[field], numeric['Effort-'])
        numeric_predictors[field] = corr

    top_numeric = sorted(numeric_predictors.items(),
                         key=lambda kv: kv[1],
                         reverse=True)[:3]

    print("\nTop 3 numeric predictors of effort:")
    for field, corr in top_numeric:
        print(f"  {field}: correlation={corr:.3f}")

    # -----------------------------------------------------------------------
    # NASA‑10 – high defect projects
    # -----------------------------------------------------------------------
    print("\n== NASA-10 ==")

    def_thresh = def_mean + 2 * def_sd
    high_def_projects = [
        r for r in records if parse_float(r['Defects-']) > def_thresh
    ]
    print(f"Unusually high defect threshold: {def_thresh:.1f} "
          f"(mean {def_mean:.1f} + 2*sd {def_sd:.1f})")
    print(f"High‑defect projects: {len(high_def_projects)}")

    # Categorical characteristics for high defects
    cat_means_def = {}
    for field in categorical_fields:
        sum_high = 0.0
        cnt_high = 0
        for r in high_def_projects:
            val = parse_float(r['Defects-'])
            if not val != val:
                sum_high += val
                cnt_high += 1
        mean_high = mean([parse_float(r['Defects-']) for r in high_def_projects]
                         if parse_float(r['Defects-']) == parse_float(r['Defects-'])]
        cat_means_def[field] = (mean_high, mean_high - def_mean)

    top_cat_def = sorted(cat_means_def.items(),
                         key=lambda kv: kv[1][1],
                         reverse=True)[:3]

    print("\nTop 3 characteristics most associated with high defect counts:")
    for field, (m, diff) in top_cat_def:
        print(f"  {field}: mean high‑defect count={m:.1f}, "
              f"overall mean={def_mean:.1f}, diff={diff:.1f}")

    # Numeric predictors of defects
    numeric_predictors_def = {}
    for field in numeric.keys():
        if field in {'Effort-', 'Defects-'}:
            continue
        corr = pearson_corr(numeric[field], numeric['Defects-'])
        numeric_predictors_def[field] = corr

    top_numeric_def = sorted(numeric_predictors_def.items(),
                             key=lambda kv: kv[1],
                             reverse=True)[:3]

    print("\nTop 3 numeric predictors of defects:")
    for field, corr in top_numeric_def:
        print(f"  {field}: correlation={corr:.3f}")


if __name__ == "__main__":
    main()
