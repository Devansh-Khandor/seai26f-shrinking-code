#!/usr/bin/env python3
"""
skill_06.py – NASA-11 & NASA-12

This script analyses a CSV file containing software project data and
produces explanations for effort estimation (NASA‑11) and risk comparison
between two proposals (NASA‑12).  The program follows the conventions
defined in the project description: goal columns end with '+' or '-',
columns ending with 'X' are not inputs, and '?' represents a missing
value.  It handles any CSV file that uses the same column names.

The output format is fixed.  For each request the script prints a header
line of the form `== <ID> ==`, followed by at least three lines of
human‑readable information, and ends with a machine‑checked JSON line
starting with `ANSWER <ID>`.
"""

import csv
import json
import math
import sys

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def read_csv(path):
    """Return a list of rows as dicts."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(val):
    """Convert string to float; return None for missing or non‑numeric."""
    if val is None:
        return None
    val = val.strip()
    if val == '' or val == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None


def normalize_min_max(values):
    """Return (min, max) ignoring None."""
    filtered = [v for v in values if v is not None]
    if not filtered:
        return (0.0, 0.0)
    return (min(filtered), max(filtered))


def linear_regression(x_vals, y_vals):
    """Least‑squares linear regression: slope and intercept."""
    n = len(x_vals)
    if n == 0:
        return (0.0, 0.0)
    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
    var_x = sum((x - mean_x) ** 2 for x in x_vals)
    if var_x == 0.0:
        return (0.0, mean_y)
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return (slope, intercept)


def std_dev(values):
    """Population standard deviation, ignoring None."""
    filtered = [v for v in values if v is not None]
    n = len(filtered)
    if n == 0:
        return 0.0
    mean = sum(filtered) / n
    return math.sqrt(sum((x - mean) ** 2 for x in filtered) / n)


# ----------------------------------------------------------------------
# NASA‑11 (Effort estimation explanation)
# ----------------------------------------------------------------------
def nasa11(rows, goal_cols):
    """Return the JSON answer for NASA‑11."""
    # Identify held‑out rows (divisible by 5)
    held_out = [i + 1 for i, _ in enumerate(rows) if (i + 1) % 5 == 0]
    if not held_out:
        return {"project": None, "estimate": None, "neighbors": []}

    # Pick first held‑out row
    proj_row_no = held_out[0]
    proj_row = rows[proj_row_no - 1]

    # Training set = all rows except held‑out
    training = [r for i, r in enumerate(rows) if (i + 1) not in held_out]

    # Kloc+ mean for imputation
    kloc_vals = [to_float(r["Kloc+"]) for r in training]
    kloc_vals = [v for v in kloc_vals if v is not None]
    mean_kloc = sum(kloc_vals) / len(kloc_vals) if kloc_vals else 0.0

    # Regression Effort- vs Kloc+
    x_eff = []
    y_eff = []
    for r in training:
        k = to_float(r["Kloc+"])
        e = to_float(r["Effort-"])
        if k is not None and e is not None:
            x_eff.append(k)
            y_eff.append(e)
    slope, intercept = linear_regression(x_eff, y_eff)

    # Predict Effort- for the project
    proj_kloc = to_float(proj_row["Kloc+"])
    if proj_kloc is None:
        proj_kloc = mean_kloc
    eff_est = slope * proj_kloc + intercept

    # Find nearest neighbours by absolute Kloc+ difference
    neighbors = []
    for r in training:
        k = to_float(r["Kloc+"])
        if k is None:
            continue
        dist = abs(k - proj_kloc)
        neighbors.append((dist, r))
    neighbors.sort(key=lambda x: x[0])
    # Keep at least 3
    chosen = neighbors[:3] if len(neighbors) >= 3 else neighbors

    # Build JSON values
    neigh_list = []
    for _, r in chosen:
        values = {}
        for col in r:
            raw = r[col]
            stripped = raw.strip()
            # Keep original numeric representation if possible
            f = to_float(raw)
            if f is not None:
                values[col] = f
            else:
                values[col] = stripped
        neigh_list.append({"row": rows.index(r) + 1, "values": values})

    return {
        "project": proj_row_no,
        "estimate": eff_est,
        "neighbors": neigh_list,
    }


# ----------------------------------------------------------------------
# NASA‑12 (Risk comparison)
# ----------------------------------------------------------------------
def nasa12(rows, goal_cols):
    """Return the JSON answer for NASA‑12."""
    # Identify held‑out rows
    held_out = [i + 1 for i, _ in enumerate(rows) if (i + 1) % 5 == 0]
    # Ensure we have at least two
    if len(held_out) < 2:
        return {
            "a": None,
            "b": None,
            "preferred": None,
            "scores": {"a": None, "b": None},
            "differences": [],
            "tradeoffs": [],
        }

    # Pre‑compute min/max for each goal column
    goal_minmax = {}
    for col in goal_cols:
        vals = [to_float(r[col]) for r in rows]
        mn, mx = normalize_min_max(vals)
        goal_minmax[col] = (mn, mx)

    # Function to compute composite score for a row
    def score_row(row):
        total = 0.0
        for col in goal_cols:
            v = to_float(row[col])
            mn, mx = goal_minmax[col]
            if v is None or mx == mn:
                continue
            if col.endswith('+'):
                norm = (v - mn) / (mx - mn)
            else:  # '-'
                norm = (mx - v) / (mx - mn)
            total += norm
        return total

    # Compute scores for held‑out rows
    scores = {}
    for row_no in held_out:
        row = rows[row_no - 1]
        scores[row_no] = score_row(row)

    # Find two rows with different scores
    a_no, b_no = held_out[0], held_out[1]
    if scores[a_no] == scores[b_no]:
        # If equal, pick next pair
        found = False
        for i in range(len(held_out) - 1):
            if scores[held_out[i]] != scores[held_out[i + 1]]:
                a_no, b_no = held_out[i], held_out[i + 1]
                found = True
                break
        if not found:
            # Fall back to first two
            pass

    score_a = scores[a_no]
    score_b = scores[b_no]
    preferred = "a" if score_a >= score_b else "b"

    # Differences on non‑goal columns
    # Consider all columns that are not goals and not ending with 'X'
    non_goal_cols = [c for c in rows[0].keys()
                     if not c.endswith(('+', '-')) and not c.endswith('X')]
    differences = []
    a_row = rows[a_no - 1]
    b_row = rows[b_no - 1]
    for col in non_goal_cols:
        val_a = a_row.get(col, None)
        val_b = b_row.get(col, None)
        # Normalize empty or '?' to None
        if val_a is not None:
            val_a_strip = val_a.strip()
            if val_a_strip == '?' or val_a_strip == '':
                val_a = None
            else:
                val_a = val_a_strip
        if val_b is not None:
            val_b_strip = val_b.strip()
            if val_b_strip == '?' or val_b_strip == '':
                val_b = None
            else:
                val_b = val_b_strip
        if val_a == val_b:
            continue
        differences.append({
            "feature": col,
            "a": val_a,
            "b": val_b,
        })
    if not differences:
        # Ensure at least one
        for col in non_goal_cols:
            differences.append({
                "feature": col,
                "a": a_row.get(col, None),
                "b": b_row.get(col, None),
            })
            break

    # Tradeoffs: goal columns where non‑preferred row has higher normalized value
    tradeoffs = []
    pref_row = a_row if preferred == "a" else b_row
    non_pref_row = b_row if preferred == "a" else a_row
    for col in goal_cols:
        v_pref = to_float(pref_row[col])
        v_non_pref = to_float(non_pref_row[col])
        mn, mx = goal_minmax[col]
        if v_pref is None or v_non_pref is None or mx == mn:
            continue
        if col.endswith('+'):
            norm_pref = (v_pref - mn) / (mx - mn)
            norm_non = (v_non_pref - mn) / (mx - mn)
        else:
            norm_pref = (mx - v_pref) / (mx - mn)
            norm_non = (mx - v_non_pref) / (mx - mn)
        if norm_non > norm_pref:
            tradeoffs.append(col)

    return {
        "a": a_no,
        "b": b_no,
        "preferred": preferred,
        "scores": {"a": score_a, "b": score_b},
        "differences": differences,
        "tradeoffs": tradeoffs,
    }


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        sys.exit(1)

    header = list(rows[0].keys())
    goal_cols = [c for c in header if c.endswith(('+', '-'))]
    non_goal_cols = [c for c in header if not c.endswith(('+', '-')) and not c.endswith('X')]

    # NASA‑11
    print("== NASA-11 ==")
    ans11 = nasa11(rows, goal_cols)
    if ans11["project"] is not None:
        print(f"Project {ans11['project']} Effort- estimate: {ans11['estimate']:.2f}")
        print("Top 3 similar historical projects:")
        for nb in ans11["neighbors"]:
            print(f"  Row {nb['row']} (values: {nb['values']})")
    else:
        print("No held‑out project available for NASA‑11.")
    print("ANSWER NASA-11", json.dumps(ans11, separators=(',', ':')))

    # NASA‑12
    print("\n== NASA-12 ==")
    ans12 = nasa12(rows, goal_cols)
    if ans12["a"] is not None:
        print(f"Comparing Row {ans12['a']} vs Row {ans12['b']}")
        print(f"  Score A: {ans12['scores']['a']:.4f}  Score B: {ans12['scores']['b']:.4f}")
        print(f"  Preferred row: {ans12['preferred'].upper()}")
        print("  Key differences:")
        for d in ans12["differences"]:
            print(f"    {d['feature']}: A={d['a']}, B={d['b']}")
        print("  Tradeoffs (goal columns where non‑preferred is better):")
        print(f"    {ans12['tradeoffs']}")
    else:
        print("Insufficient held‑out data for NASA‑12.")
    print("ANSWER NASA-12", json.dumps(ans12, separators=(',', ':')))


if __name__ == "__main__":
    main()
