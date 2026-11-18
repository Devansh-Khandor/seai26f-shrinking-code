#!/usr/bin/env python3
"""
Skill 04: Apache configuration monitoring

This program implements two new analyses for the Apache configuration data:

* **APACHE-07** – As new benchmark results arrive, assess each configuration
  immediately and alert when its measured performance is unusually poor
  compared with previously tested configurations.

* **APACHE-08** – Keep performance summaries and a recommended
  configuration shortlist updated as additional benchmark records become
  available, without requiring a complete restart of the analysis.

The program prints a header line for each analysis, a few human‑readable
lines, and ends each section with a machine‑checked answer line that
contains a single‑line JSON object as required.

"""

import csv
import json
import math
import sys
from typing import Dict, Iterable, List, Tuple, Any

# ----------------------------------------------------------------------
# CSV utilities
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Read the CSV and return header and list of rows as dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows: List[Dict[str, Any]] = []
        for row in rdr:
            d: Dict[str, Any] = {}
            for h, val in zip(header, row):
                if val == "?":
                    d[h] = None
                else:
                    try:
                        if "." in val or "e" in val.lower():
                            d[h] = float(val)
                        else:
                            d[h] = int(val)
                    except ValueError:
                        d[h] = val
            rows.append(d)
    return header, rows

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")

def mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / max(1, n - 1)
    return (m, math.sqrt(var))

# ----------------------------------------------------------------------
# APACHE-07: anomaly detection on arrivals
# ----------------------------------------------------------------------
def assess_arrivals(
    header: List[str],
    rows: List[Dict[str, Any]],
    history_rows: List[Dict[str, Any]],
    arrivals: List[Dict[str, Any]],
    budget: int,
) -> List[Dict[str, Any]]:
    """Return assessment for each arrival processed (up to budget)."""
    # Find goal column to use for alerting
    goal_cols = [c for c in header if is_goal(c)]
    if not goal_cols:
        raise RuntimeError("No goal column found.")
    goal = goal_cols[0]  # use the first goal column

    # History values for the goal
    hist_vals = [r[goal] for r in history_rows if r[goal] is not None]
    assessments: List[Dict[str, Any]] = []

    processed = 0
    for arrival in arrivals:
        if processed >= budget:
            break
        perf = arrival.get(goal)
        alert = False
        if perf is not None and hist_vals:
            m, sd = mean_std(hist_vals)
            threshold = m + 2 * sd
            alert = perf > threshold
        # Record assessment (row number is 1-based index in original file)
        row_no = rows.index(arrival) + 1
        assessments.append({"row": row_no, "alert": alert})
        # After assessment, add to history if value is present
        if perf is not None:
            hist_vals.append(perf)
        processed += 1
    return assessments

# ----------------------------------------------------------------------
# APACHE-08: progressive summary and shortlist
# ----------------------------------------------------------------------
def compute_checkpoint(
    current_rows: List[Dict[str, Any]],
    header: List[str],
    goal_cols: List[str],
    all_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a checkpoint dictionary for the current set of rows."""
    # Summary statistics for each goal column
    summary: Dict[str, Dict[str, Any]] = {}
    for goal in goal_cols:
        vals = [r[goal] for r in current_rows if r[goal] is not None]
        n = len(vals)
        if n > 0:
            mean_val = sum(vals) / n
            min_val = min(vals)
            max_val = max(vals)
        else:
            mean_val = 0.0
            min_val = 0.0
            max_val = 0.0
        summary[goal] = {"n": n, "mean": mean_val, "min": min_val, "max": max_val}

    # Shortlist of 5 best rows seen so far (by first goal column, minimize)
    if goal_cols:
        goal = goal_cols[0]
        rows_with_perf = [r for r in current_rows if r.get(goal) is not None]
        sorted_rows = sorted(rows_with_perf, key=lambda r: r[goal])
        shortlist_rows = sorted_rows[:5]
    else:
        shortlist_rows = []

    # Convert rows to plain dicts (no extra keys)
    shortlist = [dict(r) for r in shortlist_rows]

    return {"summary": summary, "shortlist": shortlist}

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    header, all_rows = read_csv(csv_path)
    goal_cols = [c for c in header if is_goal(c)]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)

    # Determine arrivals and history
    total_rows = len(all_rows)
    arrivals_count = int(round(0.2 * total_rows))
    history_rows = all_rows[: total_rows - arrivals_count]
    arrivals_rows = all_rows[total_rows - arrivals_count :]

    # Budget for evaluating arrivals
    budget = min(30, total_rows // 5)

    # ------------------------------------------------------------------
    # APACHE-07
    # ------------------------------------------------------------------
    print("== APACHE-07 ==")
    assessments = assess_arrivals(header, all_rows, history_rows, arrivals_rows, budget)
    print(f"Budget allows evaluating up to {budget} arrivals.")
    print(f"Alert threshold: mean + 2*sd of previous Performance-.")
    for a in assessments:
        print(f"Row {a['row']}: alert={a['alert']}")
    # Machine‑checked answer
    print(
        "ANSWER APACHE-07 "
        + json.dumps({"assessments": assessments})
    )

    # ------------------------------------------------------------------
    # APACHE-08
    # ------------------------------------------------------------------
    print("\n== APACHE-08 ==")
    checkpoints = []
    for m in (0, 19, 38):
        if m > arrivals_count:
            continue  # skip checkpoints beyond available arrivals
        current_rows = history_rows + arrivals_rows[:m]
        ckpt = compute_checkpoint(current_rows, header, goal_cols, all_rows)
        ckpt["after"] = m
        checkpoints.append(ckpt)

    # Human‑readable summary
    print(f"Checkpoints after 0, 19, and 38 arrivals.")
    for ckpt in checkpoints:
        print(f"-- after {ckpt['after']} arrivals --")
        for g, stats in ckpt["summary"].items():
            print(
                f"{g}: n={stats['n']} mean={stats['mean']:.2f} min={stats['min']:.2f} max={stats['max']:.2f}"
            )
        print(f"Top 5 best rows so far: {len(ckpt['shortlist'])} entries")

    # Machine‑checked answer
    # Convert list of checkpoints to desired JSON format
    json_ckpts = []
    for ckpt in checkpoints:
        json_ckpts.append(
            {
                "after": ckpt["after"],
                "summary": ckpt["summary"],
                "shortlist": ckpt["shortlist"],
            }
        )
    print(
        "ANSWER APACHE-08 "
        + json.dumps({"checkpoints": json_ckpts})
    )

# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
