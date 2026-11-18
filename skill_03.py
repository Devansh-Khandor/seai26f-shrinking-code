#!/usr/bin/env python3
"""
skill_03.py <csv_path>

Performs two NASA analysis requests on the provided CSV file:
NASA-05 – Compare projects with different programmer capability ratings and
tell me whether their development‑effort differences are meaningful enough
to inform future planning.
NASA-06 – Compare projects with different complexity ratings and determine
whether their defect outcomes differ reliably or whether the available evidence
is insufficient.

The script prints the requested sections and a machine‑checked JSON answer
for each request.
"""

import csv
import json
import sys
from statistics import median
from itertools import combinations

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def parse_value(v):
    """Parse a cell value; return float or None for missing."""
    v = v.strip()
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None


def compute_stats(vals):
    """Return mean and median of a list of numeric values (or None if empty)."""
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    md = median(vals)
    return m, md


def group_by(rows, group_col, target_col):
    """
    Group rows by `group_col` and compute count, mean, and median of `target_col`.
    Returns a dict mapping group key to statistics.
    """
    groups = {}
    for r in rows:
        key = r[group_col]
        if key not in groups:
            groups[key] = {'count': 0, 'vals': []}
        groups[key]['count'] += 1
        val = r[target_col]
        if val is not None:
            groups[key]['vals'].append(val)

    result = {}
    for key, data in groups.items():
        m, md = compute_stats(data['vals'])
        result[str(key)] = {'n': data['count'], 'median': md, 'mean': m}
    return result


def compare_groups(groups):
    """
    For every unordered pair of group keys, compare median values.
    Verdict is 'different' if the absolute difference between medians
    is more than 1% of the larger median; otherwise 'same'.
    If a median is None, verdict is 'same'.
    """
    comparisons = []
    keys = sorted(groups.keys())
    for a, b in combinations(keys, 2):
        med_a = groups[a]['median']
        med_b = groups[b]['median']
        if med_a is None or med_b is None:
            verdict = 'same'
        else:
            diff = abs(med_a - med_b)
            max_med = max(med_a, med_b)
            verdict = 'different' if diff > 0.01 * max_med else 'same'
        comparisons.append({'a': a, 'b': b, 'verdict': verdict})
    return comparisons


def load_rows(csv_path):
    """Read CSV and return a list of parsed rows with row numbers."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        all_columns = reader.fieldnames
        rows = []
        for idx, raw in enumerate(reader, start=1):
            parsed = {}
            for col in all_columns:
                if col.endswith('X'):      # non‑input columns
                    parsed[col] = None
                else:
                    parsed[col] = parse_value(raw[col])
            parsed['row_num'] = idx
            rows.append(parsed)
    return rows, all_columns


# --------------------------------------------------------------------------- #
# Main analysis functions
# --------------------------------------------------------------------------- #

def nasa_05(rows):
    """Compare projects by programmer capability (pcap)."""
    groups = group_by(rows, 'pcap', 'Effort-')
    comparisons = compare_groups(groups)
    return groups, comparisons


def nasa_06(rows):
    """Compare projects by complexity (cplx)."""
    groups = group_by(rows, 'cplx', 'Defects-')
    comparisons = compare_groups(groups)
    return groups, comparisons


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(csv_path):
    rows, _ = load_rows(csv_path)

    # NASA-05
    groups_05, comps_05 = nasa_05(rows)
    print("== NASA-05 ==")
    for key, stats in groups_05.items():
        print(f"Group {key}: n={stats['n']}, median={stats['median']}, mean={stats['mean']}")
    print("Comparisons:")
    for comp in comps_05:
        print(f"  {comp['a']} vs {comp['b']}: {comp['verdict']}")
    print("ANSWER NASA-05", json.dumps(
        {"groups": groups_05, "comparisons": comps_05},
        ensure_ascii=False))

    # NASA-06
    groups_06, comps_06 = nasa_06(rows)
    print("\n== NASA-06 ==")
    for key, stats in groups_06.items():
        print(f"Group {key}: n={stats['n']}, median={stats['median']}, mean={stats['mean']}")
    print("Comparisons:")
    for comp in comps_06:
        print(f"  {comp['a']} vs {comp['b']}: {comp['verdict']}")
    print("ANSWER NASA-06", json.dumps(
        {"groups": groups_06, "comparisons": comps_06},
        ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
