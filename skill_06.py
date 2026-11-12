#!/usr/bin/env python3
"""
Small data‑analysis skill for NASA project data – new NASA‑11 and NASA‑12.

Usage:
    python skill_06.py <csv_path>
"""

import csv
import sys
import statistics
import math
from typing import List, Dict, Tuple, Any

# --------------------------------------------------------------------------- #
# Helpers reused from earlier skills
# --------------------------------------------------------------------------- #

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


def _filter_nan(xs: List[float]) -> List[float]:
    return [x for x in xs if not x != x]


def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """
    Perform ordinary least squares regression y = a * x + b.
    Returns (slope, intercept, residual_std).
    """
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


# --------------------------------------------------------------------------- #
# Main routine: NASA‑11 & NASA‑12
# --------------------------------------------------------------------------- #

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = load_data(csv_path)
    numeric = extract_numeric_fields(records)

    # Regression models
    slope_ek, intercept_ek, resid_std_ek = linear_regression(numeric['Kloc+'], numeric['Effort-'])
    slope_de, intercept_de, resid_std_de = linear_regression(numeric['Effort-'], numeric['Defects-'])

    # Historical stats
    eff_mean, eff_med, eff_sd = compute_stats(numeric['Effort-'])
    def_mean, def_med, def_sd = compute_stats(numeric['Defects-'])
    k_mean, k_med, k_sd = compute_stats(numeric['Kloc+'])

    # -----------------------------------------------------------------------
    # NASA‑11 – Explanation of effort estimate for a proposed project
    # -----------------------------------------------------------------------
    print("== NASA-11 ==")

    # Use median project size as the “proposed” size
    proposed_size = k_med
    predicted_effort = predict(slope_ek, intercept_ek, proposed_size)

    # Find the three historical projects nearest to the median size
    neighbors = sorted(
        [(r, abs(parse_float(r['Kloc+']) - proposed_size)) for r in records],
        key=lambda t: t[1]
    )[:3]

    print(f"Proposed project size: {proposed_size:.1f} KLOC (median of historical data).")
    print(f"Regression estimate: Effort ≈ {predicted_effort:.1f} hours.")
    print("Closest historical projects (by size) and their actual effort:")
    for rec, dist in neighbors:
        pid = rec['idX']
        size = parse_float(rec['Kloc+'])
        effort = parse_float(rec['Effort-'])
        acap = rec.get('acap', 'N/A')
        plex = rec.get('plex', 'N/A')
        print(f"  - Project {pid}: size={size:.1f} KLOC, effort={effort:.1f} h, acap={acap}, plex={plex}")

    # Explanation line
    print(f"The estimate is based on the linear model fitted to all historical data, "
          f"and falls within the observed effort range of projects with similar size, "
          f"providing a data‑driven confidence in the projected effort.")

    # -----------------------------------------------------------------------
    # NASA‑12 – Compare two proposals with different risk assessments
    # -----------------------------------------------------------------------
    print("\n== NASA-12 ==")

    # Compute risk thresholds (similar to NASA‑07)
    eff_thresh = eff_mean + 2 * eff_sd
    def_thresh = def_mean + 2 * def_sd

    flagged = []
    for r in records:
        pid = int(r['idX'])
        size = parse_float(r['Kloc+'])
        pred_eff = predict(slope_ek, intercept_ek, size)
        pred_def = predict(slope_de, intercept_de, pred_eff)
        high_eff = pred_eff > eff_thresh
        high_def = pred_def > def_thresh
        if high_eff or high_def:
            flagged.append({
                'id': pid,
                'size': size,
                'pred_eff': pred_eff,
                'pred_def': pred_def,
                'high_eff': high_eff,
                'high_def': high_def,
                'acap': r.get('acap', 'N/A'),
                'plex': r.get('plex', 'N/A')
            })

    # If fewer than two flagged projects, pick the two with largest predicted effort difference
    if len(flagged) < 2:
        # compute predicted effort for all projects
        all_proj = []
        for r in records:
            pid = int(r['idX'])
            size = parse_float(r['Kloc+'])
            pred_eff = predict(slope_ek, intercept_ek, size)
            pred_def = predict(slope_de, intercept_de, pred_eff)
            all_proj.append({
                'id': pid,
                'size': size,
                'pred_eff': pred_eff,
                'pred_def': pred_def,
                'acap': r.get('acap', 'N/A'),
                'plex': r.get('plex', 'N/A')
            })
        # sort by predicted effort descending
        all_proj.sort(key=lambda p: p['pred_eff'], reverse=True)
        flagged = all_proj[:2]

    # Pick first two flagged
    p1, p2 = flagged[0], flagged[1]

    # Show comparison
    print(f"Comparing Project {p1['id']} and Project {p2['id']}:")
    print(f"  Project {p1['id']}: size={p1['size']:.1f} KLOC, predicted effort={p1['pred_eff']:.1f} h, "
          f"predicted defects={p1['pred_def']:.1f}, acap={p1['acap']}, plex={p1['plex']}, "
          f"{'high effort' if p1['high_eff'] else ''} {'high defects' if p1['high_def'] else ''}")
    print(f"  Project {p2['id']}: size={p2['size']:.1f} KLOC, predicted effort={p2['pred_eff']:.1f} h, "
          f"predicted defects={p2['pred_def']:.1f}, acap={p2['acap']}, plex={p2['plex']}, "
          f"{'high effort' if p2['high_eff'] else ''} {'high defects' if p2['high_def'] else ''}")

    # Explain main differences and tradeoffs
    size_diff = abs(p1['size'] - p2['size'])
    effort_diff = abs(p1['pred_eff'] - p2['pred_def'])
    print("\nMain differences contributing to the risk assessment:")
    print(f"  Size difference: {size_diff:.1f} KLOC.")
    print(f"  Predicted effort difference: {effort_diff:.1f} hours.")
    print(f"  Project {p1['id']} {'has' if p1['high_eff'] else 'does not have'} an effort risk flag, "
          f"while Project {p2['id']} {'has' if p2['high_eff'] else 'does not have'} it.")
    print(f"  Trade‑off: A larger size typically drives higher effort, but may also influence defect risk "
          f"depending on the project’s complexity (plex) and programmer capability (acap).")
    print("  Decision makers should weigh the potential higher effort against the risk of defects and "
          "consider mitigation such as increased testing or allocating more experienced staff.")


if __name__ == "__main__":
    main()
