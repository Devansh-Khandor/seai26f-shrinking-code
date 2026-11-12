#!/usr/bin/env python3
"""
skill_07.py - NASA-13 and NASA-14: Identify promising project characteristics and
               prioritize proposals for evaluation.

Usage:
    python skill_07.py <csv_path>
"""

import csv
import sys
from statistics import mean
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Import utilities from previous skills
# ----------------------------------------------------------------------
try:
    from skill_01 import load_csv, get_numeric_column
except Exception:
    def load_csv(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = [row for row in reader]
        return rows, header

    def get_numeric_column(rows: List[Dict[str, str]], col: str) -> List[float]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except (ValueError, TypeError):
                pass
        return vals

try:
    from skill_02 import identify_numeric_columns
except Exception:
    # Minimal fallback if skill_02 not available
    def identify_numeric_columns(rows: List[Dict[str, str]], header: List[str]) -> Tuple[set, set]:
        numeric = set()
        categorical = set()
        for col in header:
            if col == 'idX':
                continue
            sample = None
            for r in rows:
                v = r.get(col, '')
                if v != '':
                    sample = v
                    break
            if sample is None:
                categorical.add(col)
            else:
                try:
                    float(sample)
                    numeric.add(col)
                except ValueError:
                    categorical.add(col)
        return numeric, categorical

# ----------------------------------------------------------------------
# Helper to compute metric: size / (effort + defects)
# ----------------------------------------------------------------------
def compute_row_metric(row: Dict[str, str]) -> Tuple[float, float, float, float]:
    """
    Return (size, effort, defects, metric) for a row.
    Rows with missing or non‑numeric values are skipped by returning (None, None, None, None).
    """
    try:
        size = float(row.get('Kloc+', ''))
        effort = float(row.get('Effort-', ''))
        defects = float(row.get('Defects-', ''))
    except (ValueError, TypeError):
        return None, None, None, None
    denom = effort + defects
    if denom <= 0:
        return None, None, None, None
    metric = size / denom
    return size, effort, defects, metric

# ----------------------------------------------------------------------
# NASA-13: identify promising combinations of characteristics
# ----------------------------------------------------------------------
def report_nasa_13(rows: List[Dict[str, str]], header: List[str]) -> None:
    print("\n== NASA-13 ==")

    # Identify numeric vs categorical columns
    numeric_cols, categorical_cols = identify_numeric_columns(rows, header)
    # Exclude target columns from categorical set
    categorical_cols.discard('Kloc+')
    categorical_cols.discard('Effort-')
    categorical_cols.discard('Defects-')
    categorical_cols.discard('idX')

    # Accumulate metrics per categorical value
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for col in categorical_cols:
        stats[col] = {}

    row_metrics = []  # list of tuples (id, size, effort, defects, metric)
    for r in rows:
        size, effort, defects, metric = compute_row_metric(r)
        if size is None:
            continue
        row_metrics.append((r.get('idX', ''), size, effort, defects, metric))
        for col in categorical_cols:
            val = r.get(col, '').strip()
            if val == '':
                continue
            if val not in stats[col]:
                stats[col][val] = {
                    'count': 0,
                    'size_sum': 0.0,
                    'effort_sum': 0.0,
                    'defects_sum': 0.0,
                    'metric_sum': 0.0,
                }
            entry = stats[col][val]
            entry['count'] += 1
            entry['size_sum'] += size
            entry['effort_sum'] += effort
            entry['defects_sum'] += defects
            entry['metric_sum'] += metric

    # Build a list of all (col, val) with their average metric
    avg_list: List[Tuple[str, str, float, float, float, float]] = []  # col, val, avg_size, avg_effort, avg_defects, avg_metric
    for col, vals in stats.items():
        for val, data in vals.items():
            cnt = data['count']
            if cnt == 0:
                continue
            avg_size = data['size_sum'] / cnt
            avg_effort = data['effort_sum'] / cnt
            avg_defects = data['defects_sum'] / cnt
            avg_metric = data['metric_sum'] / cnt
            avg_list.append((col, val, avg_size, avg_effort, avg_defects, avg_metric))

    if not avg_list:
        print("No categorical data available to identify promising combinations.")
        return

    # Sort by avg_metric descending and pick top 3
    avg_list.sort(key=lambda x: x[5], reverse=True)
    top_n = 3
    print(f"Top {top_n} promising categories based on average size/(effort+defects):")
    for i, (col, val, a_size, a_effort, a_defects, a_metric) in enumerate(avg_list[:top_n], 1):
        print(f"  {i}. {col} = '{val}': AvgSize={a_size:.2f}, AvgEffort={a_effort:.2f}, "
              f"AvgDefects={a_defects:.2f}, AvgMetric={a_metric:.3f}")

# ----------------------------------------------------------------------
# NASA-14: recommend which proposals to evaluate first
# ----------------------------------------------------------------------
def report_nasa_14(rows: List[Dict[str, str]]) -> None:
    print("\n== NASA-14 ==")

    # Compute metric for all rows
    metrics = []
    for r in rows:
        size, effort, defects, metric = compute_row_metric(r)
        if size is None:
            continue
        metrics.append((r.get('idX', ''), size, effort, defects, metric))

    if not metrics:
        print("No valid projects found to evaluate.")
        return

    # Sort by metric descending
    metrics.sort(key=lambda x: x[4], reverse=True)

    # Recommend top 5 projects (or fewer if dataset is small)
    top_n = min(5, len(metrics))
    print(f"Recommend evaluating first the top {top_n} projects by size/(effort+defects):")
    for i, (proj_id, size, effort, defects, metric) in enumerate(metrics[:top_n], 1):
        print(f"  {i}. id={proj_id} => Size={size:.2f}, Effort={effort:.2f}, "
              f"Defects={defects:.2f}, Metric={metric:.3f}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, header = load_csv(csv_path)

    report_nasa_13(rows, header)
    report_nasa_14(rows)

if __name__ == "__main__":
    main()
