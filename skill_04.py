#!/usr/bin/env python3
"""
skill_04.py

Usage: python skill_04.py <csv_path>
"""

import sys
import csv
import json
import math


# ------------------------------------------------------------
# CSV reading helper
# ------------------------------------------------------------
def read_csv(path):
    """Read CSV and return a list of rows (dict) with 1‑based indices."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=1):
            row = {}
            for k, v in raw.items():
                if v == '?':
                    row[k] = None
                else:
                    try:
                        row[k] = int(v)
                    except ValueError:
                        row[k] = v
            row['_rownum'] = idx
            rows.append(row)
    return rows


# ------------------------------------------------------------
# Helper: statistics for a list of numeric values
# ------------------------------------------------------------
def _mean_std(values):
    n = len(values)
    if n == 0:
        return None, None, None
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / n
    std = math.sqrt(var)
    return m, std, n


# ------------------------------------------------------------
# APACHE-07 : Arrival assessment
# ------------------------------------------------------------
def assess_arrivals(rows, perf_col, arrivals):
    """Return list of dicts {'row':rownum,'alert':bool} for each arrival."""
    assessments = []

    # Keep a running list of performance values seen so far
    seen_perf = []

    for row in arrivals:
        rownum = row['_rownum']
        val = row.get(perf_col)
        # Compute stats from previous rows only
        m, std, _ = _mean_std(seen_perf)
        alert = False
        if val is not None:
            if m is None:
                # no previous data – no alert
                alert = False
            else:
                # threshold: mean + 2*std (if std is zero, mean alone)
                threshold = m + 2 * std if std > 0 else m
                alert = val > threshold
        assessments.append({"row": rownum, "alert": alert})
        if val is not None:
            seen_perf.append(val)
    return assessments


# ------------------------------------------------------------
# APACHE-08 : Checkpoints summary & shortlist
# ------------------------------------------------------------
def compute_summary(rows, goal_cols):
    """Return dict goal -> dict(n, mean, min, max)."""
    summary = {}
    for col in goal_cols:
        values = [r[col] for r in rows if r[col] is not None]
        n = len(values)
        if n == 0:
            mean = None
            mn = None
            mx = None
        else:
            mean = round(sum(values) / n, 2)
            mn = min(values)
            mx = max(values)
        summary[col] = {"n": n, "mean": mean, "min": mn, "max": mx}
    return summary


def shortlist_best(rows, perf_col, count=5):
    """Return list of row numbers of best (lowest) performances."""
    best = [(r['_rownum'], r[perf_col]) for r in rows
            if r[perf_col] is not None]
    # sort by performance asc, then by row number asc
    best.sort(key=lambda x: (x[1], x[0]))
    return [rownum for rownum, _ in best[:count]]


def checkpoints(rows, history, perf_col, goal_cols, checkpoints_after):
    """Return list of checkpoint dicts."""
    results = []
    for after in checkpoints_after:
        # consider history plus first 'after' arrivals
        subset = history + rows[:after]
        summary = compute_summary(subset, goal_cols)
        shortlist = shortlist_best(subset, perf_col, count=5)
        results.append({
            "after": after,
            "summary": summary,
            "shortlist": shortlist
        })
    return results


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        return

    # Identify goal columns (ending with + or -)
    goal_cols = [c for c in rows[0].keys() if c.endswith('+') or c.endswith('-')]
    # Find performance column (the one we will use for alerts and shortlist)
    perf_col = next((c for c in goal_cols if c.endswith('-') and 'Performance' in c),
                    None)
    if not perf_col:
        print("No performance column found.")
        return

    total_rows = len(rows)
    arrivals_count = int(round(0.2 * total_rows))
    arrivals = rows[-arrivals_count:] if arrivals_count > 0 else []
    history = rows[:-arrivals_count] if arrivals_count > 0 else rows

    # --------------------------------------------------
    # APACHE-07
    # --------------------------------------------------
    print("== APACHE-07 ==")
    assessments = assess_arrivals(arrivals, perf_col, arrivals)
    for a in assessments[:3]:
        print(f"Row {a['row']}: alert={a['alert']}")
    print(f"Total arrivals assessed: {len(assessments)}")
    print(f"ANSWER APACHE-07 {json.dumps({'assessments': assessments}, separators=(',', ':'))}")

    # --------------------------------------------------
    # APACHE-08
    # --------------------------------------------------
    print("\n== APACHE-08 ==")
    # Checkpoints after m = 0, 19, 38 arrivals (capped by actual arrivals)
    checkpoints_needed = [0, 19, 38]
    checkpoints_needed = [min(m, arrivals_count) for m in checkpoints_needed]
    checkpoints_needed = sorted(set(checkpoints_needed))
    ckpts = checkpoints(arrivals, history, perf_col, goal_cols, checkpoints_needed)
    for ck in ckpts[:3]:
        print(f"Checkpoint after {ck['after']} arrivals:")
        print(f"  Summary keys: {list(ck['summary'].keys())}")
        print(f"  Shortlist: {ck['shortlist']}")
    print(f"ANSWER APACHE-08 {json.dumps({'checkpoints': ckpts}, separators=(',', ':'))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
