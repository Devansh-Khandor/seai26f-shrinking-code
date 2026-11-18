#!/usr/bin/env python3
"""
skill_04.py

Usage:
    python skill_04.py <path to Apache_AllMeasurements.csv>
"""

import json
import sys
import math

# Reuse helpers from skill_01
import skill_01 as s01

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def running_stats(values):
    """Return (n, mean, min, max, std) for a list of numeric values (ints/floats)."""
    n = len(values)
    if n == 0:
        return 0, 0.0, None, None, 0.0
    mean = sum(values) / n
    _min = min(values)
    _max = max(values)
    # Population std deviation
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return n, mean, _min, _max, std

def best_rows(rows, goal_col, n_best=5):
    """Return list of 1‑based row numbers of the best rows according to the goal."""
    # Filter rows with valid goal values
    valid = [(idx + 1, r[goal_col]) for idx, r in enumerate(rows) if r[goal_col] is not None]
    if not valid:
        return []
    # Sort: for '-' goals lower values are better; for '+' goals higher are better
    reverse = not goal_col.endswith('-')
    # For '+' goal: higher better, so reverse=True; for '-' goal: lower better, reverse=False
    sorted_rows = sorted(valid, key=lambda t: t[1], reverse=reverse)
    return [idx for idx, _ in sorted_rows[:n_best]]

# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = s01.parse_csv(csv_path)

    # Identify goal column (assume only one)
    goal_cols = [c for c in header if s01.is_goal(c)]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    goal_col = goal_cols[0]

    total_rows = len(rows)
    arrivals_cnt = int(round(0.2 * total_rows))
    history_rows = rows[:total_rows - arrivals_cnt]
    arrivals_rows = rows[-arrivals_cnt:]

    # ---------- APACHE-07 ----------
    print("== APACHE-07 ==")
    assessments = []
    # Running list of goal values seen so far (history + arrivals processed)
    seen_values = [r[goal_col] for r in history_rows if r[goal_col] is not None]

    for arr_idx, arr_row in enumerate(arrivals_rows, start=1):
        val = arr_row[goal_col]
        alert = False
        if val is not None and seen_values:
            n, mean, _, _, std = running_stats(seen_values)
            # For '-' goals: higher value is worse
            if goal_col.endswith('-'):
                threshold = mean + 1.96 * std
                if val > threshold:
                    alert = True
            else:
                threshold = mean - 1.96 * std
                if val < threshold:
                    alert = True
        assessments.append({"row": total_rows - arrivals_cnt + arr_idx, "alert": alert})
        print(f"Row {total_rows - arrivals_cnt + arr_idx}: performance {val}, alert={alert}")

    # JSON answer
    apache07_answer = {"assessments": assessments}
    print("ANSWER APACHE-07", json.dumps(apache07_answer, separators=(",", ":")))

    # ---------- APACHE-08 ----------
    print("\n== APACHE-08 ==")
    checkpoints = [0, 19, 38]
    # Ensure we don't exceed arrivals
    checkpoints = [m for m in checkpoints if m <= arrivals_cnt]
    # Prepare summary data structure
    results = []
    for m in checkpoints:
        # rows considered = history + first m arrivals
        current_rows = history_rows + arrivals_rows[:m]
        # Summary per goal column
        summary = {}
        for gcol in goal_cols:
            values = [r[gcol] for r in current_rows if r[gcol] is not None]
            n, mean, _min, _max, _std = running_stats(values)
            summary[gcol] = {"n": n, "mean": mean, "min": _min if _min is not None else 0,
                             "max": _max if _max is not None else 0}
        # Shortlist: best rows seen so far (history + arrivals[:m])
        shortlist = best_rows(current_rows, goal_col, 5)
        # Store checkpoint
        results.append({"after": m, "summary": summary, "shortlist": shortlist})
        # Print human readable
        print(f"Checkpoint after {m} arrivals:")
        for gcol in goal_cols:
            s = summary[gcol]
            print(f"  {gcol}: n={s['n']}, mean={s['mean']:.2f}, min={s['min']}, max={s['max']}")
        print(f"  Shortlist of best rows: {shortlist}")

    apache08_answer = {"checkpoints": results}
    print("ANSWER APACHE-08", json.dumps(apache08_answer, separators=(",", ":")))


if __name__ == "__main__":
    main()
