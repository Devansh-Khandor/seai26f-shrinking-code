#!/usr/bin/env python3
"""
Extended data‑analysis script for the Auto93 dataset.

Adds two new client requests:
    CAR-13 – Identify a vehicle with a strong balance of fuel efficiency,
             weight, and acceleration while evaluating as few candidate vehicles
             as reasonably possible.
    CAR-14 – When we have a limited inspection budget, recommend which vehicles
             should be evaluated first so that we can discover promising
             inventory options early.

Usage:
    python skill_07.py <path_to_auto93.csv>
"""

import sys
from typing import List, Dict, Any

# Import helper functions from skill_01
from skill_01 import (
    read_csv,
    compute_score,
    compute_stats,
    format_float,
)

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Compute bounds for normalization using summary statistics
    # -----------------------------------------------------------------------
    weight_bounds = compute_stats(rows, 'Lbs-')
    mpg_bounds = compute_stats(rows, 'Mpg+')
    accel_bounds = compute_stats(rows, 'Acc+')

    # -----------------------------------------------------------------------
    # Compute composite score for each row that has all required metrics
    # -----------------------------------------------------------------------
    for r in rows:
        if None in (r['Lbs-'], r['Mpg+'], r['Acc+']):
            continue
        r['score'] = compute_score(
            r,
            {'min': weight_bounds['min'], 'max': weight_bounds['max']},
            {'min': mpg_bounds['min'], 'max': mpg_bounds['max']},
            {'min': accel_bounds['min'], 'max': accel_bounds['max']},
        )

    # -----------------------------------------------------------------------
    # Rank all vehicles by composite score (descending)
    # -----------------------------------------------------------------------
    scored_rows = [r for r in rows if 'score' in r]
    scored_rows.sort(key=lambda r: r['score'], reverse=True)

    # -----------------------------------------------------------------------
    # CAR-13 – Strong balance vehicle(s)
    # -----------------------------------------------------------------------
    print("== CAR-13 ==")
    # Show top 3 vehicles (or fewer if not enough data)
    for i, v in enumerate(scored_rows[:3], start=1):
        print(
            f"{i}. Vehicle: Clndrs={int(v['Clndrs'])}, Volume={int(v['Volume'])}, "
            f"HpX={int(v['HpX'])}, Model={int(v['Model'])}, origin={int(v['origin'])}, "
            f"Lbs-={int(v['Lbs-'])}, Acc+={format_float(v['Acc+'])}, "
            f"Mpg+={int(v['Mpg+'])}, score={format_float(v['score'])}"
        )

    # -----------------------------------------------------------------------
    # CAR-14 – Limited inspection budget recommendation
    # -----------------------------------------------------------------------
    print("\n== CAR-14 ==")
    # Recommend inspecting the same top vehicles first; they are the most promising.
    for i, v in enumerate(scored_rows[:3], start=1):
        print(
            f"{i}. Vehicle: Clndrs={int(v['Clndrs'])}, Volume={int(v['Volume'])}, "
            f"HpX={int(v['HpX'])}, Model={int(v['Model'])}, origin={int(v['origin'])}, "
            f"Lbs-={int(v['Lbs-'])}, Acc+={format_float(v['Acc+'])}, "
            f"Mpg+={int(v['Mpg+'])}, score={format_float(v['score'])}"
        )

if __name__ == "__main__":
    main()
