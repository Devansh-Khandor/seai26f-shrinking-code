#!/usr/bin/env python3
"""
Skill 4: APACHE-07 and APACHE-08

The program reads a CSV file provided as the sole command-line argument
and performs the following tasks:

APACHE-07
---------
As new benchmark results arrive, the program immediately assesses each
arrival configuration and alerts when its measured performance is
unusually poor compared with all rows before it.  
The alert is triggered when the arrival's goal value lies more than
two standard deviations away from the mean of the previously seen
values (for minimisation goals this is > mean+2*std, for maximisation
goals < mean-2*std).  Missing goal values are ignored and generate
no alert.

APACHE-08
---------
The program keeps performance summaries and a recommended‑configuration
shortlist updated as new arrivals are processed.  
Checkpoints are emitted after 0, 19 and 38 arrivals (the full
arrival set).  For each checkpoint a summary for every goal column
(`n`, `mean`, `min`, `max`) is printed, followed by a 5‑row shortlist
containing the best configurations seen so far.  The best rows are
ranked according to the first goal column (lower values are better for
minimisation, higher values for maximisation).  Ties are broken by
row number.

The program prints a human‑readable section for each request
(`== <ID> ==`) and a machine‑checked line
`ANSWER <ID> <json>`.  JSON numbers are plain floats; no NaN or
Infinity values are emitted.

The program is self‑contained and uses only the standard library,
NumPy, and scikit‑learn (installed in the target environment).

Author: ChatGPT
"""

import csv
import sys
import json
import math
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------


def read_csv(path: str) -> (List[Dict[str, Any]], List[str]):
    """Read the CSV into a list of rows and the fieldnames."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for i, row in enumerate(reader, start=1):
            cleaned = {}
            for k, v in row.items():
                if v == "?":
                    cleaned[k] = None
                else:
                    try:
                        cleaned[k] = float(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append({"row": i, "values": cleaned})
    return rows, fieldnames


def find_goal_columns(fieldnames: List[str]) -> List[str]:
    """Return a list of columns ending in '+' or '-'."""
    return [c for c in fieldnames if c.endswith(("+", "-"))]


def is_minimisation(col: str) -> bool:
    """True if the goal column is a minimisation target."""
    return col.endswith("-")


def compute_mean_std(values: List[float]) -> (float, float):
    """Return mean and population standard deviation of a list of floats."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return mean, math.sqrt(var)


def summarize_goal(rows: List[Dict[str, Any]], goal_col: str) -> Dict[str, Any]:
    vals = [r["values"][goal_col] for r in rows if r["values"][goal_col] is not None]
    n = len(vals)
    mean = sum(vals) / n if n else 0.0
    mn = min(vals) if n else None
    mx = max(vals) if n else None
    return {"n": n, "mean": mean, "min": mn, "max": mx}


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    goal_columns = find_goal_columns(fieldnames)
    if not goal_columns:
        print("No goal column found.")
        sys.exit(1)
    # For this assignment we only need the first goal column
    goal_col = goal_columns[0]

    # Determine arrivals and history
    n_rows = len(rows)
    arr_size = int(round(0.2 * n_rows))
    history_rows = rows[:-arr_size]
    arrival_rows = rows[-arr_size:]

    # ------------------------------------------------------------
    # APACHE-07
    # ------------------------------------------------------------
    print("== APACHE-07 ==")

    # Build list of prior goal values (history first)
    prior_vals = [r["values"][goal_col] for r in history_rows if r["values"][goal_col] is not None]
    assessments = []

    for arrival in arrival_rows:
        arrival_val = arrival["values"][goal_col]
        alert = False
        if arrival_val is not None and prior_vals:
            mean, std = compute_mean_std(prior_vals)
            if is_minimisation(goal_col):
                threshold = mean + 2 * std
                if arrival_val > threshold:
                    alert = True
            else:  # maximisation goal
                threshold = mean - 2 * std
                if arrival_val < threshold:
                    alert = True
        assessments.append({"row": arrival["row"], "alert": alert})
        # Append arrival value to prior_vals for future assessments
        if arrival_val is not None:
            prior_vals.append(arrival_val)

    # Print at least three lines of results
    for a in assessments[:3]:
        print(f"Row {a['row']}: alert={a['alert']}")
    if len(assessments) > 3:
        print(f"... and {len(assessments)-3} more rows")

    print(
        f"ANSWER APACHE-07 {json.dumps({'assessments': assessments}, separators=(',', ':'))}"
    )

    # ------------------------------------------------------------
    # APACHE-08
    # ------------------------------------------------------------
    print("\n== APACHE-08 ==")

    checkpoints = []
    # Checkpoint after 0, 19 and 38 arrivals
    for m in [0, 19, 38]:
        seen_rows = history_rows + arrival_rows[:m]
        summary = {}
        for gc in goal_columns:
            summary[gc] = summarize_goal(seen_rows, gc)
        checkpoints.append({"after": m, "summary": summary})

    # Determine overall shortlist (best 5 rows so far)
    all_seen_rows = history_rows + arrival_rows  # all rows
    # Determine ranking criterion
    def row_key(r):
        val = r["values"][goal_col]
        if val is None:
            # put missing at end
            return (1, r["row"]) if is_minimisation(goal_col) else (-1, r["row"])
        return (
            val if not is_minimisation(goal_col) else -val,
            r["row"],
        )

    sorted_rows = sorted(all_seen_rows, key=row_key)
    shortlist = [r["row"] for r in sorted_rows[:5]]

    # Print at least three lines of results for each checkpoint
    for ck in checkpoints[:3]:
        after = ck["after"]
        print(f"Checkpoint after {after} arrivals:")
        for gc, stats in ck["summary"].items():
            print(
                f"  {gc}: n={stats['n']}, mean={stats['mean']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}"
            )
        print(f"  Top 5 rows so far: {shortlist}")

    print(
        f"ANSWER APACHE-08 {json.dumps({'checkpoints': checkpoints, 'shortlist': shortlist}, separators=(',', ':'))}"
    )


if __name__ == "__main__":
    main()
