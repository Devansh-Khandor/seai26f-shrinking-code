#!/usr/bin/env python3
"""
Extended data‑analysis script for the Auto93 dataset.

Adds two new client requests:
    CAR-07 – Immediate assessment of each arriving vehicle against the most desirable
             inventory options.
    CAR-08 – Up‑to‑date inventory summaries and recommended‑vehicle shortlist
             without requiring a manual restart.

Usage:
    python skill_04.py <path_to_auto93.csv>
"""

import sys
from typing import List, Dict, Any, Tuple

# Import helpers from the earlier skill modules
from skill_01 import (
    read_csv,
    compute_stats,
    compute_score,
    format_float,
)

# --------------------------------------------------------------------------- #
# Helper to build bounds for normalization
# --------------------------------------------------------------------------- #
def _compute_bounds(rows: List[Dict[str, Any]]) -> Tuple[dict, dict, dict]:
    weight_bounds = {
        'min': min(r['Lbs-'] for r in rows if r['Lbs-'] is not None),
        'max': max(r['Lbs-'] for r in rows if r['Lbs-'] is not None),
    }
    mpg_bounds = {
        'min': min(r['Mpg+'] for r in rows if r['Mpg+'] is not None),
        'max': max(r['Mpg+'] for r in rows if r['Mpg+'] is not None),
    }
    accel_bounds = {
        'min': min(r['Acc+'] for r in rows if r['Acc+'] is not None),
        'max': max(r['Acc+'] for r in rows if r['Acc+'] is not None),
    }
    return weight_bounds, mpg_bounds, accel_bounds

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # ----------------------------------------------------------------------- #
    # Compute bounds for normalization
    # ----------------------------------------------------------------------- #
    weight_bounds, mpg_bounds, accel_bounds = _compute_bounds(rows)

    # ----------------------------------------------------------------------- #
    # Compute composite score for each row that has all required fields
    # ----------------------------------------------------------------------- #
    for r in rows:
        if None in (r['Lbs-'], r['Mpg+'], r['Acc+']):
            continue
        r['score'] = compute_score(r, weight_bounds, mpg_bounds, accel_bounds)

    # ----------------------------------------------------------------------- #
    # Rank all vehicles by score (descending)
    # ----------------------------------------------------------------------- #
    scored_rows = [r for r in rows if 'score' in r]
    scored_rows.sort(key=lambda r: r['score'], reverse=True)

    # ----------------------------------------------------------------------- #
    # CAR-07 – Immediate assessment of each arriving vehicle
    # ----------------------------------------------------------------------- #
    print("== CAR-07 ==")
    # Show top 5 vehicles for illustration; at least 3 lines of results
    for idx, v in enumerate(scored_rows[:5], start=1):
        desirable = "YES" if idx <= 3 else "NO"
        print(f"{idx}. Vehicle: Clndrs={int(v['Clndrs'])}, Volume={int(v['Volume'])}, "
              f"HpX={int(v['HpX'])}, Model={int(v['Model'])}, origin={int(v['origin'])}, "
              f"Lbs-={int(v['Lbs-'])}, Acc+={format_float(v['Acc+'])}, "
              f"Mpg+={int(v['Mpg+'])}, score={format_float(v['score'])} "
              f"| Desirable={desirable}")

    # ----------------------------------------------------------------------- #
    # CAR-08 – Up‑to‑date inventory summaries and recommended shortlist
    # ----------------------------------------------------------------------- #
    print("\n== CAR-08 ==")

    # ---- Summary statistics (CAR-01) ----
    weight_stats = compute_stats(rows, 'Lbs-')
    mpg_stats = compute_stats(rows, 'Mpg+')
    accel_stats = compute_stats(rows, 'Acc+')

    print(f"Weight (lbs): mean={format_float(weight_stats['mean'])}, "
          f"min={format_float(weight_stats['min'])}, "
          f"max={format_float(weight_stats['max'])}")

    print(f"Fuel efficiency (mpg): mean={format_float(mpg_stats['mean'])}, "
          f"min={format_float(mpg_stats['min'])}, "
          f"max={format_float(mpg_stats['max'])}")

    print(f"Acceleration (s): mean={format_float(accel_stats['mean'])}, "
          f"min={format_float(accel_stats['min'])}, "
          f"max={format_float(accel_stats['max'])}")

    # ---- Recommended shortlist (top 3) (CAR-02) ----
    print("\nRecommended shortlist (top 3 vehicles by composite score):")
    for idx, v in enumerate(scored_rows[:3], start=1):
        print(f"{idx}. Vehicle: Clndrs={int(v['Clndrs'])}, Volume={int(v['Volume'])}, "
              f"HpX={int(v['HpX'])}, Model={int(v['Model'])}, origin={int(v['origin'])}, "
              f"Lbs-={int(v['Lbs-'])}, Acc+={format_float(v['Acc+'])}, "
              f"Mpg+={int(v['Mpg+'])}, score={format_float(v['score'])}")

if __name__ == "__main__":
    main()
