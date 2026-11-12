#!/usr/bin/env python3
"""
Extended data‑analysis script for the Auto93 dataset.

Adds two new client requests:
    CAR-09 – For vehicles with unusually poor fuel efficiency, identify the
             characteristics most strongly associated with that outcome.
    CAR-10 – Identify the common characteristics of vehicles with
             unfavorable overall performance and contrast them with
             better‑performing vehicles.

Usage:
    python skill_05.py <path_to_auto93.csv>
"""

import sys
from typing import List, Dict, Any, Tuple

# Import helpers from earlier skill modules
from skill_01 import read_csv, format_float


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def _numeric_features(rows: List[Dict[str, Any]]) -> List[str]:
    """Return list of numeric keys present in the dataset (excluding the target)."""
    sample = next((r for r in rows if r), None)
    if not sample:
        return []
    return [k for k, v in sample.items()
            if isinstance(v, (int, float)) and k != 'Mpg+']


def _compute_correlation(rows: List[Dict[str, Any]],
                         feature: str,
                         target: str = 'Mpg+') -> float:
    """Compute Pearson correlation coefficient between feature and target."""
    values_f = []
    values_t = []
    for r in rows:
        f = r.get(feature)
        t = r.get(target)
        if f is not None and t is not None:
            values_f.append(float(f))
            values_t.append(float(t))
    n = len(values_f)
    if n == 0:
        return 0.0
    mean_f = sum(values_f) / n
    mean_t = sum(values_t) / n
    cov = sum((f - mean_f) * (t - mean_t) for f, t in zip(values_f, values_t))
    std_f = (sum((f - mean_f) ** 2 for f in values_f) / n) ** 0.5
    std_t = (sum((t - mean_t) ** 2 for t in values_t) / n) ** 0.5
    if std_f == 0 or std_t == 0:
        return 0.0
    return cov / (std_f * std_t)


def _percentile(values: List[float], p: float) -> float:
    """Return the p-th percentile (0-100) of a list of numbers."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # ----------------------------------------------------------------------- #
    # CAR-09: Poor mpg characteristic analysis
    # ----------------------------------------------------------------------- #
    print("== CAR-09 ==")

    # Determine low mpg threshold (25th percentile)
    mpg_values = [r['Mpg+'] for r in rows if r['Mpg+'] is not None]
    low_threshold = _percentile(mpg_values, 25)

    # Identify low mpg vehicles
    low_mpg_vehicles = [r for r in rows if r['Mpg+'] is not None and r['Mpg+'] <= low_threshold]

    if not low_mpg_vehicles:
        print("No vehicles fall below the 25th‑percentile mpg threshold.")
    else:
        # Compute correlation for each numeric feature with mpg
        numeric_feats = _numeric_features(rows)
        corr_map: Dict[str, float] = {feat: _compute_correlation(rows, feat) for feat in numeric_feats}

        # Sort features by absolute correlation descending
        sorted_corr = sorted(corr_map.items(), key=lambda kv: abs(kv[1]), reverse=True)

        # Show results for up to three low‑mpg vehicles
        for i, v in enumerate(low_mpg_vehicles[:3], start=1):
            # Find the most correlated feature
            best_feat, best_corr = max(corr_map.items(), key=lambda kv: abs(kv[1]))
            # Provide some context about the value
            val = v.get(best_feat, 'N/A')
            print(f"{i}. Vehicle Clndrs={int(v['Clndrs'])}, Model={int(v['Model'])}, "
                  f"Mpg+={int(v['Mpg+'])} "
                  f"| Most correlated characteristic: {best_feat} "
                  f"(corr={format_float(best_corr)}), value={val}")

    # ----------------------------------------------------------------------- #
    # CAR-10: Common characteristics of unfavorable vs favorable performance
    # ----------------------------------------------------------------------- #
    print("\n== CAR-10 ==")

    # Define unfavorable (<= median) and favorable (> median) groups
    median_mpg = _percentile(mpg_values, 50)
    unfav_rows = [r for r in rows if r['Mpg+'] is not None and r['Mpg+'] <= median_mpg]
    fav_rows = [r for r in rows if r['Mpg+'] is not None and r['Mpg+'] > median_mpg]

    if not unfav_rows or not fav_rows:
        print("Insufficient data to separate unfavorable and favorable groups.")
    else:
        # Compute group means for each numeric feature
        numeric_feats = _numeric_features(rows)
        def _group_mean(group: List[Dict[str, Any]], feat: str) -> float:
            vals = [r[feat] for r in group if r[feat] is not None]
            return sum(vals) / len(vals) if vals else 0.0

        diff_stats: List[Tuple[str, float, float, float]] = []
        for feat in numeric_feats:
            unfav_mean = _group_mean(unfav_rows, feat)
            fav_mean = _group_mean(fav_rows, feat)
            diff = unfav_mean - fav_mean
            diff_stats.append((feat, unfav_mean, fav_mean, diff))

        # Sort by absolute difference descending
        diff_stats.sort(key=lambda kv: abs(kv[3]), reverse=True)

        # Show top 3 characteristic differences
        for i, (feat, unfav_mean, fav_mean, diff) in enumerate(diff_stats[:3], start=1):
            direction = "higher" if diff > 0 else "lower"
            print(f"{i}. {feat} – Unfavorable avg={format_float(unfav_mean)}, "
                  f"Favorable avg={format_float(fav_mean)} "
                  f"({abs(format_float(diff))} {direction} in unfavorable vehicles)")

if __name__ == "__main__":
    main()
