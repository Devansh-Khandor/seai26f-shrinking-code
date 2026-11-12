#!/usr/bin/env python3
"""
Small data‑analysis skill for NASA project data – new NASA‑07 and NASA‑08.

Usage:
    python skill_04.py <csv_path>
"""

import csv
import sys
import statistics
import math
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Helpers reused from earlier skills
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

# From skill_02.py
def _filter_nan(xs: List[float]) -> List[float]:
    return [x for x in xs if not x != x]


def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """Perform ordinary least squares regression y = a * x + b."""
    x = _filter_nan(x)
    y = _filter_nan(y)
    if len(x) != len(y) or not x:
        return (float('nan'), float('nan'), float('nan'))
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    a = 0.0 if sxx == 0 else sxy / sxx
    b = mean_y - a * mean_x
    residuals = [yi - (a * xi + b) for xi, yi in zip(x, y)]
    res_std = statistics.stdev(residuals) if len(residuals) > 1 else 0.0
    return (a, b, res_std)


def predict(slope: float, intercept: float, x_val: float) -> float:
    """Compute y = slope * x + intercept."""
    return slope * x_val + intercept

# From skill_03.py
def group_numeric(records: List[Dict[str, str]],
                  group_field: str,
                  numeric_field: str) -> Dict[str, Dict[str, float]]:
    """Group numeric values by a categorical field."""
    groups: Dict[str, List[float]] = {}
    for r in records:
        key = r[group_field].strip()
        val = parse_float(r[numeric_field])
        if val != val:   # skip NaN
            continue
        groups.setdefault(key, []).append(val)

    stats: Dict[str, Dict[str, float]] = {}
    for key, values in groups.items():
        n = len(values)
        mean = statistics.mean(values)
        std = statistics.stdev(values) if n > 1 else 0.0
        stats[key] = {'mean': mean, 'std': std, 'n': n}
    return stats


def ci95(mean: float, std: float, n: int) -> Tuple[float, float]:
    """Return the 95 % confidence interval for a mean."""
    if n <= 1:
        return (mean, mean)
    margin = 1.96 * std / math.sqrt(n)
    return (mean - margin, mean + margin)


def pairwise_no_overlap(ci1: Tuple[float, float], ci2: Tuple[float, float]) -> bool:
    """Return True if two CIs do not overlap."""
    return ci1[1] < ci2[0] or ci2[1] < ci1[0]

# --------------------------------------------------------------------------- #
# Main routine: NASA‑07 & NASA‑08
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = load_data(csv_path)
    numeric = extract_numeric_fields(records)

    # -----------------------------------------------------------------------
    # NASA‑07 – flag unusually high predictions for new proposals
    # -----------------------------------------------------------------------
    print("== NASA-07 ==")

    # Historical statistics used as thresholds
    _, _, eff_sd = compute_stats(numeric['Effort-'])
    _, _, def_sd = compute_stats(numeric['Defects-'])
    eff_mean, _, _ = compute_stats(numeric['Effort-'])
    def_mean, _, _ = compute_stats(numeric['Defects-'])
    eff_thresh = eff_mean + 2 * eff_sd
    def_thresh = def_mean + 2 * def_sd

    # Regressions
    slope_ek, intercept_ek, _ = linear_regression(numeric['Kloc+'], numeric['Effort-'])
    slope_de, intercept_de, _ = linear_regression(numeric['Effort-'], numeric['Defects-'])

    flagged = []
    for r in records:
        pid = int(r['idX'])
        size = parse_float(r['Kloc+'])
        # Prediction of effort from size
        pred_effort = predict(slope_ek, intercept_ek, size)
        # Prediction of defects from predicted effort
        pred_defects = predict(slope_de, intercept_de, pred_effort)
        high_eff = pred_effort > eff_thresh
        high_def = pred_defects > def_thresh
        if high_eff or high_def:
            flagged.append({
                'id': pid,
                'pred_effort': pred_effort,
                'pred_defects': pred_defects,
                'high_eff': high_eff,
                'high_def': high_def
            })

    # Results
    print(f"Historical effort mean={eff_mean:.1f}, 95% threshold={eff_thresh:.1f}")
    print(f"Historical defect mean={def_mean:.1f}, 95% threshold={def_thresh:.1f}")
    if not flagged:
        print("No proposals exceed the high‑prediction thresholds.")
    else:
        print(f"{len(flagged)} proposal(s) exceed thresholds:")
        for p in flagged:
            flags = []
            if p['high_eff']:
                flags.append('effort')
            if p['high_def']:
                flags.append('defects')
            print(f"  - Project {p['id']}: predicted effort={p['pred_effort']:.1f}, "
                  f"predicted defects={p['pred_defects']:.1f} ({', '.join(flags)})")

    # -----------------------------------------------------------------------
    # NASA‑08 – update historical summaries and estimates
    # -----------------------------------------------------------------------
    print("\n== NASA-08 ==")

    # Basic stats
    k_mean, k_med, k_sd = compute_stats(numeric['Kloc+'])
    e_mean, e_med, e_sd = compute_stats(numeric['Effort-'])
    d_mean, d_med, d_sd = compute_stats(numeric['Defects-'])

    print(f"Project size (Kloc+): mean={k_mean:.1f}, median={k_med:.1f}, std={k_sd:.1f}")
    print(f"Development effort (Effort-): mean={e_mean:.1f}, median={e_med:.1f}, std={e_sd:.1f}")
    print(f"Defect count (Defects-): mean={d_mean:.1f}, median={d_med:.1f}, std={d_sd:.1f}")

    # Regression estimates
    slope_ek, intercept_ek, resid_std_ek = linear_regression(numeric['Kloc+'], numeric['Effort-'])
    slope_de, intercept_de, resid_std_de = linear_regression(numeric['Effort-'], numeric['Defects-'])
    print("\nRegression of Effort on Size:")
    print(f"  slope={slope_ek:.3f}, intercept={intercept_ek:.3f}, residual SD={resid_std_ek:.1f}")
    print("\nRegression of Defects on Effort:")
    print(f"  slope={slope_de:.3f}, intercept={intercept_de:.3f}, residual SD={resid_std_de:.1f}")

    # Group statistics (acap & plex) similar to NASA‑05 & NASA‑06
    print("\nProgrammer capability (acap) group-wise mean effort:")
    acap_stats = group_numeric(records, 'acap', 'Effort-')
    for key in sorted(acap_stats.keys()):
        m = acap_stats[key]['mean']
        n = acap_stats[key]['n']
        print(f"  acap {key}: mean effort {m:.1f} (n={n})")

    print("\nProject complexity (plex) group-wise mean defect count:")
    plex_stats = group_numeric(records, 'plex', 'Defects-')
    for key in sorted(plex_stats.keys()):
        m = plex_stats[key]['mean']
        n = plex_stats[key]['n']
        print(f"  plex {key}: mean defects {m:.1f} (n={n})")

    # Check statistical significance for acap groups
    acap_cis = {k: ci95(v['mean'], v['std'], v['n']) for k, v in acap_stats.items()}
    acap_meaningful = False
    keys = sorted(acap_stats.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if pairwise_no_overlap(acap_cis[keys[i]], acap_cis[keys[j]]):
                acap_meaningful = True
                break
        if acap_meaningful:
            break
    if acap_meaningful:
        print("\nDevelopment‑effort differences across acap levels are statistically meaningful.")
    else:
        print("\nDevelopment‑effort differences across acap levels are not statistically significant.")

    # Check statistical significance for plex groups
    plex_cis = {k: ci95(v['mean'], v['std'], v['n']) for k, v in plex_stats.items()}
    plex_reliable = False
    keys = sorted(plex_stats.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if pairwise_no_overlap(plex_cis[keys[i]], plex_cis[keys[j]]):
                plex_reliable = True
                break
        if plex_reliable:
            break
    if plex_reliable:
        print("\nDefect outcomes vary reliably across complexity levels.")
    else:
        print("\nDefect outcomes do not differ reliably across complexity levels.")


if __name__ == "__main__":
    main()
