#!/usr/bin/env python3
"""
Small data‑analysis skill for NASA project data.

Usage:
    python skill_01.py <csv_path>
"""

import csv
import sys
import statistics
from typing import Dict, List, Tuple


def load_data(csv_path: str) -> List[Dict[str, str]]:
    """
    Read CSV and return list of records (dicts).
    """
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_float(value: str) -> float:
    """
    Convert a string to float, stripping spaces.
    Return NaN for empty strings.
    """
    try:
        return float(value.strip())
    except ValueError:
        return float('nan')


def extract_numeric_fields(records: List[Dict[str, str]]) -> Dict[str, List[float]]:
    """
    Return dictionary mapping numeric column name to list of values.
    """
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
    # Filter out NaN
    vals = [v for v in values if not v != v]  # remove NaNs
    if not vals:
        return (float('nan'), float('nan'), float('nan'))
    mean = statistics.mean(vals)
    median = statistics.median(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return (mean, median, stdev)


def find_best_worst(records: List[Dict[str, str]]) -> Tuple[Tuple[int, float], Tuple[int, float]]:
    """
    Return best and worst project id and defect count.
    Best: lowest Defects-.
    Worst: highest Defects-.
    """
    best = None  # (id, defects)
    worst = None
    for r in records:
        pid = int(r['idX'])
        defects = parse_float(r['Defects-'])
        if defects != defects:
            continue
        if best is None or defects < best[1]:
            best = (pid, defects)
        if worst is None or defects > worst[1]:
            worst = (pid, defects)
    return best, worst


def percentile(values: List[float], p: float) -> float:
    """Return the p-th percentile (0-100) of a list of numbers."""
    vals = sorted(v for v in values if v == v)  # drop NaN
    if not vals:
        return float('nan')
    k = (len(vals) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(vals):
        return vals[f] + c * (vals[f + 1] - vals[f])
    else:
        return vals[f]


def find_favorable(records: List[Dict[str, str]], numeric: Dict[str, List[float]]) -> List[Dict[str, str]]:
    """
    Identify projects that are in the bottom 25% for size, effort, and defects.
    Return list of matching record dicts.
    """
    thresholds = {
        'Kloc+': percentile(numeric['Kloc+'], 25),
        'Effort-': percentile(numeric['Effort-'], 25),
        'Defects-': percentile(numeric['Defects-'], 25),
    }
    fav = []
    for r in records:
        k = parse_float(r['Kloc+'])
        e = parse_float(r['Effort-'])
        d = parse_float(r['Defects-'])
        if k <= thresholds['Kloc+'] and e <= thresholds['Effort-'] and d <= thresholds['Defects-']:
            fav.append({
                'id': r['idX'],
                'Kloc+': f"{k:.1f}",
                'Effort-': f"{e:.1f}",
                'Defects-': f"{d:.1f}",
            })
    return fav


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = load_data(csv_path)
    numeric = extract_numeric_fields(records)

    # NASA-01
    print("== NASA-01 ==")
    size_mean, size_med, size_sd = compute_stats(numeric['Kloc+'])
    effort_mean, effort_med, effort_sd = compute_stats(numeric['Effort-'])
    defects_mean, defects_med, defects_sd = compute_stats(numeric['Defects-'])
    best, worst = find_best_worst(records)

    print(f"Typical project size (Kloc+): mean={size_mean:.1f}, median={size_med:.1f}, std={size_sd:.1f}")
    print(f"Typical development effort (Effort-): mean={effort_mean:.1f}, median={effort_med:.1f}, std={effort_sd:.1f}")
    print(f"Typical defect count (Defects-): mean={defects_mean:.1f}, median={defects_med:.1f}, std={defects_sd:.1f}")
    print(f"Best outcome (lowest defects): project {best[0]} with {best[1]:.1f} defects")
    print(f"Worst outcome (highest defects): project {worst[0]} with {worst[1]:.1f} defects")

    # NASA-02
    print("\n== NASA-02 ==")
    favorable = find_favorable(records, numeric)
    if not favorable:
        print("No projects meet the favorable criteria.")
    else:
        print(f"Projects in bottom 25% for size, effort and defects (IDs):")
        for p in favorable:
            print(f"  - Project {p['id']}: Kloc+={p['Kloc+']}, Effort-={p['Effort-']}, Defects-={p['Defects-']}")

if __name__ == "__main__":
    main()
