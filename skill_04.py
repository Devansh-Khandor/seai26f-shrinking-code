#!/usr/bin/env python3
"""
skill_04.py

APACHE-07 – Detect unusually poor performance in arriving benchmark rows.
APACHE-08 – Keep performance summaries and best‑configuration shortlist
            updated incrementally as new rows arrive.
"""

import csv
import json
import math
import sys
from collections import defaultdict

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def read_csv(path: str):
    """Read CSV and return list of (row_no, row_dict)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for rn, rec in enumerate(reader, start=1):
            rows.append((rn, rec))
    return rows, reader.fieldnames


def is_goal(col: str):
    return col.endswith("+") or col.endswith("-")


def is_x(col: str):
    return col.endswith("X")


def to_float(v: str):
    """Convert to float, return None for missing or bad."""
    if v is None or v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values, mean_val):
    """Population standard deviation."""
    if not values:
        return 0.0
    return math.sqrt(sum((x - mean_val) ** 2 for x in values) / len(values))


# ----------------------------------------------------------------------
# APACHE-07 logic
# ----------------------------------------------------------------------
def assess_arrivals(rows, goal_col):
    """
    For each arrival, alert if its Performance- is > mean + 2*sd of all
    previous rows (history + earlier arrivals).  Return list of
    assessments with row numbers in arrival order.
    """
    # split rows
    N = len(rows)
    arrivals_count = int(round(0.2 * N))
    history_rows = rows[: N - arrivals_count]
    arrivals = rows[N - arrivals_count:]

    assessments = []

    # cache numeric performance values for history
    prior_vals = [to_float(r[goal_col]) for _, r in history_rows]
    prior_vals = [v for v in prior_vals if v is not None]

    for idx, (row_no, rec) in enumerate(arrivals):
        perf_val = to_float(rec[goal_col])
        if perf_val is None or not prior_vals:
            alert = False
        else:
            mu = mean(prior_vals)
            sigma = std(prior_vals, mu)
            alert = perf_val > mu + 2 * sigma
        assessments.append({"row": row_no, "alert": alert})
        # add this row's value to prior_vals if numeric
        if perf_val is not None:
            prior_vals.append(perf_val)

    return assessments


# ----------------------------------------------------------------------
# APACHE-08 logic
# ----------------------------------------------------------------------
def incremental_checkpoints(rows, goal_col):
    """
    Produce checkpoints after 0, 19, and 38 arrivals (if available).
    Each checkpoint contains summaries for all goal columns and a shortlist
    of the 5 best (lowest) rows so far.
    """
    N = len(rows)
    arrivals_count = int(round(0.2 * N))
    history_rows = rows[: N - arrivals_count]
    arrivals = rows[N - arrivals_count:]

    target_m = [0, 19, 38]
    checkpoints = []

    # function to compute summary for a column given a subset of rows
    def col_summary(col, subset):
        vals = [to_float(r[col]) for _, r in subset if to_float(r[col]) is not None]
        n = len(vals)
        if n == 0:
            return {"n": 0, "mean": None, "min": None, "max": None}
        m = round(mean(vals), 6)
        mn = round(min(vals), 6)
        mx = round(max(vals), 6)
        return {"n": n, "mean": m, "min": mn, "max": mx}

    # find all goal columns
    goal_cols = [c for c in rows[0][1].keys() if is_goal(c)]

    for m in target_m:
        if m > arrivals_count:
            continue
        subset = history_rows + arrivals[:m]
        summary = {}
        for g in goal_cols:
            summary[g] = col_summary(g, subset)
        # shortlist: 5 best rows (lowest value of primary goal column)
        # choose first goal column ending with '-'
        primary = next((g for g in goal_cols if g.endswith("-")), None)
        if primary is None:
            primary = goal_cols[0]
        # gather numeric rows
        numeric_rows = [(rn, r) for rn, r in subset if to_float(r[primary]) is not None]
        numeric_rows.sort(key=lambda x: (to_float(x[1][primary]), x[0]))
        shortlist = [rn for rn, _ in numeric_rows[:5]]
        checkpoints.append({"after": m, "summary": summary, "shortlist": shortlist})

    return checkpoints


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)

    # identify primary goal column (minimize)
    goal_cols = [c for c in fieldnames if is_goal(c)]
    if not goal_cols:
        print("No goal column found.", file=sys.stderr)
        sys.exit(1)
    primary_goal = next((c for c in goal_cols if c.endswith("-")), goal_cols[0])

    # ---------------- APACHE-07 ----------------
    print("== APACHE-07 ==")
    assessments = assess_arrivals(rows, primary_goal)
    # simple textual report
    alerts = [a for a in assessments if a["alert"]]
    print(f"Total arrivals: {len(assessments)}")
    print(f"Alerts raised: {len(alerts)}")
    for a in assessments:
        print(f"Row {a['row']}: alert={a['alert']}")
    print("ANSWER APACHE-07", json.dumps({"assessments": assessments}, separators=(",", ":")))

    # ---------------- APACHE-08 ----------------
    print("\n== APACHE-08 ==")
    checkpoints = incremental_checkpoints(rows, primary_goal)
    if checkpoints:
        for ck in checkpoints:
            after = ck["after"]
            print(f"Checkpoint after {after} arrivals")
            for col, s in ck["summary"].items():
                print(f"  {col}: n={s['n']}, mean={s['mean']}, min={s['min']}, max={s['max']}")
            print(f"  Shortlist: {ck['shortlist']}")
    else:
        print("No checkpoints (insufficient arrivals).")
    # Output machine‑checked answer
    # Remove per‑checkpoint shortlist from the JSON (it's already part of checkpoint)
    print("ANSWER APACHE-08", json.dumps({"checkpoints": checkpoints}, separators=(",", ":")))


if __name__ == "__main__":
    main()
