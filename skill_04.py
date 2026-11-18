#!/usr/bin/env python3
"""
skill_04.py

Usage: python skill_04.py <csv path>
"""

import json
import math
import sys
from typing import Any, Dict, List, Tuple

import ezr

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def numeric(v: Any) -> float | None:
    """Return a numeric value, or None if missing."""
    return None if v == "?" else v


def compute_col_stats(rows: List[List[Any]], goal_names: List[str]) -> Dict[str, Tuple[float, float]]:
    """Compute min and max for each goal column among the given rows."""
    stats: Dict[str, Tuple[float, float]] = {}
    for name in goal_names:
        vals = [numeric(r[header.index(name)]) for r in rows if numeric(r[header.index(name)]) is not None]
        if not vals:
            stats[name] = (0.0, 1.0)
        else:
            mn, mx = min(vals), max(vals)
            if mn == mx:
                mx = mn + 1.0
            stats[name] = (mn, mx)
    return stats


def compute_score(row: List[Any], stats: Dict[str, Tuple[float, float]]) -> float:
    """Compute composite score for a row based on provided min/max per goal."""
    score = 0.0
    for name, (mn, mx) in stats.items():
        val = numeric(row[header.index(name)])
        if val is None:
            norm = 0.0
        else:
            if name.endswith("-"):  # minimize
                norm = (mx - val) / (mx - mn)
            else:  # maximize
                norm = (val - mn) / (mx - mn)
        score += norm
    return score


def median(values: List[float]) -> float:
    """Return median of a list of numbers."""
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


# ----------------------------------------------------------------------
# CAR-07: Arrival assessment
# ----------------------------------------------------------------------
def car_07(
    history_rows: List[List[Any]],
    arrivals_rows: List[List[Any]],
    header: List[str],
    goal_names: List[str],
) -> None:
    print("\n== CAR-07 ==")

    # Precompute scores for history rows
    scores_history: List[float] = []

    assessments: List[Dict[str, Any]] = []

    for idx, row in enumerate(arrivals_rows, start=1):
        # Rows seen so far = history + previous arrivals
        seen_rows = history_rows + arrivals_rows[: idx - 1]

        # Compute stats from seen rows
        stats_seen = compute_col_stats(seen_rows, goal_names)

        # Compute scores of seen rows for median
        scores_seen = [compute_score(r, stats_seen) for r in seen_rows]
        med = median(scores_seen)

        # Score of current arrival
        arrival_score = compute_score(row, stats_seen)

        desirable = arrival_score > med
        row_number = len(history_rows) + idx  # 1‑based among all rows

        assessments.append({"row": row_number, "desirable": desirable})

        # Simple console output
        print(f"Arrival {row_number}: score={arrival_score:.3f} "
              f"median={med:.3f} desirable={desirable}")

    # Machine‑checked answer
    print(f"ANSWER CAR-07 {json.dumps({'assessments': assessments})}")


# ----------------------------------------------------------------------
# CAR-08: Inventory summaries and shortlist
# ----------------------------------------------------------------------
def car_08(
    history_rows: List[List[Any]],
    arrivals_rows: List[List[Any]],
    header: List[str],
    goal_names: List[str],
) -> None:
    print("\n== CAR-08 ==")

    checkpoints = [0, 40, 80]
    checkpoint_results: List[Dict[str, Any]] = []

    # Pre‑compute cumulative rows list for convenience
    cumulative_rows = history_rows + arrivals_rows

    for m in checkpoints:
        # Clamp to available arrivals
        if m > len(arrivals_rows):
            m = len(arrivals_rows)

        rows_seen = history_rows + arrivals_rows[:m]
        after = m

        # Compute statistics per goal
        summary: Dict[str, Dict[str, float]] = {}
        for name in goal_names:
            vals = [numeric(r[header.index(name)]) for r in rows_seen
                    if numeric(r[header.index(name)]) is not None]
            n = len(vals)
            if n == 0:
                mean_val = 0.0
                min_val = 0.0
                max_val = 0.0
            else:
                mean_val = sum(vals) / n
                min_val = min(vals)
                max_val = max(vals)
            summary[name] = {
                "n": n,
                "mean": mean_val,
                "min": min_val,
                "max": max_val,
            }

        # Compute shortlist: top 5 rows by composite score
        stats_for_scoring = compute_col_stats(rows_seen, goal_names)
        scored_rows = [
            (compute_score(r, stats_for_scoring), idx + len(history_rows) + 1)
            for idx, r in enumerate(rows_seen)
        ]
        scored_rows.sort(key=lambda x: (-x[0], x[1]))
        shortlist = [row_num for _, row_num in scored_rows[:5]]

        checkpoint_results.append(
            {"after": after, "summary": summary, "shortlist": shortlist}
        )

        # Simple console output
        print(f"After {after} arrivals:")
        for name, stats in summary.items():
            print(f"  {name}: n={stats['n']} mean={stats['mean']:.3f} "
                  f"min={stats['min']} max={stats['max']}")
        print(f"  Shortlist: {shortlist}")

    # Machine‑checked answer
    print(f"ANSWER CAR-08 {json.dumps({'checkpoints': checkpoint_results})}")


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Load all rows
    rows_iter = ezr.csv(csv_path)
    header = next(rows_iter)
    data_rows: List[List[Any]] = list(rows_iter)

    total_rows = len(data_rows)
    arrivals_count = int(round(0.2 * total_rows))

    history_rows = data_rows[: total_rows - arrivals_count]
    arrivals_rows = data_rows[total_rows - arrivals_count :]

    # Identify goal columns
    goal_names = [name for name in header if name[-1] in "+-"]

    # Run CAR-07 and CAR-08
    car_07(history_rows, arrivals_rows, header, goal_names)
    car_08(history_rows, arrivals_rows, header, goal_names)


if __name__ == "__main__":
    main()
