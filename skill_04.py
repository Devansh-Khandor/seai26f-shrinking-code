#!/usr/bin/env python3
"""
Skill 04: Real‑time assessment and incremental summarisation

APACHE‑07:  Detect unusually poor performances for arriving
           benchmark configurations.

APACHE‑08:  Incrementally update performance summaries and a
           shortlist of best configurations as new data arrives.
"""

import csv
import json
import sys
import math
import statistics

# --------------------------------------------------------------------
# Common CSV reader – identical to skill_01.read_csv
# --------------------------------------------------------------------
def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            cleaned = {}
            for k, v in row.items():
                if v == '?':
                    cleaned[k] = None
                else:
                    try:
                        cleaned[k] = float(v) if '.' in v else int(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append((idx, cleaned))
    return rows, reader.fieldnames


# --------------------------------------------------------------------
# Utility – statistics for goal columns
# --------------------------------------------------------------------
def goal_stats(rows, goal_columns):
    """Return dict mapping goal column to {n, mean, min, max} over rows."""
    stats = {}
    for col in goal_columns:
        vals = [r[col] for _, r in rows if r[col] is not None]
        n = len(vals)
        if n == 0:
            mean = min_val = max_val = None
        else:
            mean = sum(vals) / n
            min_val = min(vals)
            max_val = max(vals)
        stats[col] = {"n": n, "mean": mean, "min": min_val, "max": max_val}
    return stats


# --------------------------------------------------------------------
# APACHE‑07:  Real‑time assessment of new arrivals
# --------------------------------------------------------------------
def apache_07(rows, goal_columns, arrivals):
    """
    Returns list of dicts: {row: int, alert: bool}
    """
    # Determine which goal to monitor – assume only one goal column
    if not goal_columns:
        raise RuntimeError("No goal columns found")
    goal = goal_columns[0]

    # Determine budget (rows we are allowed to evaluate)
    total_rows = len(rows)
    budget = min(30, total_rows // 5)

    # Keep history of evaluated rows (including history and earlier arrivals)
    evaluated_vals = []

    assessments = []

    # Process arrivals in order, stop when budget reached
    for i, (idx, r) in enumerate(arrivals):
        if i >= budget:
            # Too many to evaluate – cannot assess
            assessments.append({"row": idx, "alert": False})
            continue

        val = r.get(goal)
        if val is None:
            # Cannot assess missing value
            assessments.append({"row": idx, "alert": False})
            continue

        # Compute mean and std of previous evaluated rows
        if evaluated_vals:
            mean_prev = sum(evaluated_vals) / len(evaluated_vals)
            if len(evaluated_vals) > 1:
                std_prev = math.sqrt(
                    sum((x - mean_prev) ** 2 for x in evaluated_vals)
                    / (len(evaluated_vals) - 1)
                )
            else:
                std_prev = 0.0
            threshold = mean_prev + std_prev
            alert = val > threshold
        else:
            # No previous data – no alert
            alert = False

        assessments.append({"row": idx, "alert": alert})
        evaluated_vals.append(val)

    return assessments


# --------------------------------------------------------------------
# APACHE‑08:  Incrementally update summaries & shortlist
# --------------------------------------------------------------------
def apache_08(rows, goal_columns, arrivals, checkpoints=[0, 19, 38]):
    """
    Returns list of checkpoint dicts:
      {after: m, summary: {goal: {...}}, shortlist: [row, ...]}
    """
    # Helper to compute shortlist (best 5 rows for the goal)
    def best_rows(up_to_rows):
        # Best rows are those with lowest Performance- (goal)
        goal = goal_columns[0]
        # Filter rows with non-missing goal
        valid = [(idx, r[goal]) for idx, r in up_to_rows if r[goal] is not None]
        # Sort ascending (better is lower)
        valid.sort(key=lambda x: x[1])
        return [idx for idx, _ in valid[:5]]

    # Build checkpoints
    results = []

    # History count = total rows - arrivals count
    total_rows = len(rows)
    arrivals_count = len(arrivals)
    history_count = total_rows - arrivals_count

    # Convert to list for slicing
    all_rows = rows

    for m in checkpoints:
        # Cap to available arrivals
        m = min(m, arrivals_count)
        upto_rows = all_rows[:history_count + m]
        summary = {col: goal_stats(upto_rows, [col])[col] for col in goal_columns}
        shortlist = best_rows(upto_rows)
        results.append({"after": m, "summary": summary, "shortlist": shortlist})

    return results


# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)

    # Identify goal columns (ending with + or -)
    goal_columns = [col for col in fieldnames if col.endswith(('+', '-'))]
    if not goal_columns:
        print("Error: No goal column found.")
        sys.exit(1)

    # Identify arrivals (last round(0.2*N) rows)
    total_rows = len(rows)
    arrivals_count = round(0.2 * total_rows)
    arrivals = rows[-arrivals_count:]

    # --------------------------------------------------------------------
    # APACHE-07
    # --------------------------------------------------------------------
    print("== APACHE-07 ==")
    assessments = apache_07(rows, goal_columns, arrivals)
    for a in assessments:
        print(f"Row {a['row']}: alert={a['alert']}")
    # JSON answer
    ans_07 = {"assessments": assessments}
    print(f"ANSWER APACHE-07 {json.dumps(ans_07)}")

    # --------------------------------------------------------------------
    # APACHE-08
    # --------------------------------------------------------------------
    print("\n== APACHE-08 ==")
    checkpoints = [0, 19, 38]
    results = apache_08(rows, goal_columns, arrivals, checkpoints)
    for r in results:
        print(f"Checkpoint after {r['after']} arrivals:")
        for gcol, stats in r["summary"].items():
            print(f"  {gcol}: n={stats['n']}, mean={stats['mean']}, min={stats['min']}, max={stats['max']}")
        print(f"  Shortlist: {r['shortlist']}")
    # JSON answer
    ans_08 = {"checkpoints": results}
    print(f"ANSWER APACHE-08 {json.dumps(ans_08)}")

if __name__ == "__main__":
    main()
