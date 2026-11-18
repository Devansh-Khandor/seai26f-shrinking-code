#!/usr/bin/env python3
"""
skill_07.py

Usage: python skill_07.py <csv path>
"""

import csv
import json
import math
import sys
import itertools
from collections import defaultdict

# ---------- utilities ----------------------------------------------------
def parse_value(v: str):
    """Parse a CSV cell into a numeric value, bool or string."""
    if v == "?":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v

def percentile(values, p):
    """Return the p‑th percentile of a list of numbers."""
    if not values:
        return None
    values = sorted(values)
    pos = p * (len(values) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values[int(pos)]
    weight = pos - lower
    return values[lower] * (1 - weight) + values[upper] * weight

def compute_normalizer(values):
    """Return (min, max) for normalization, handling identical values."""
    mn = min(values)
    mx = max(values)
    return mn, mx

def norm_value(v, mn, mx, maximize=True):
    """Normalize a value to [0,1].  If max==min, return 1."""
    if mx == mn:
        return 1.0
    if maximize:
        return (v - mn) / (mx - mn)
    else:
        return (mx - v) / (mx - mn)

# ---------- NASA‑13 ------------------------------------------------------
def nasa13(rows, headers, goal_idx):
    """
    Find a rule (1‑3 conditions on non‑goal columns) that selects rows
    with high Kloc+, low Effort‑, low Defects‑.
    """
    # Identify goal columns
    goal_cols = [headers[i] for i in goal_idx]
    goal_names = {headers[i]: i for i in goal_idx}

    # Non‑goal columns: not X, not goal
    non_goal_idx = [i for i, h in enumerate(headers)
                    if not h.endswith("X") and not h.endswith(("+", "-"))]
    non_goal_names = [headers[i] for i in non_goal_idx]

    # Determine numeric columns among non‑goal
    numeric_cols = []
    symbolic_cols = []
    col_types = {}
    for i in non_goal_idx:
        col_vals = [rows[r][i] for r in range(len(rows))
                    if rows[r][i] is not None]
        if not col_vals:
            # treat as symbolic if no values
            symbolic_cols.append(i)
            col_types[i] = "sym"
        elif all(isinstance(v, (int, float)) for v in col_vals):
            numeric_cols.append(i)
            col_types[i] = "num"
        else:
            symbolic_cols.append(i)
            col_types[i] = "sym"

    # Pre‑compute percentiles and bounds for numeric columns
    num_bounds = {}
    for i in numeric_cols:
        col_vals = [v for v in [rows[r][i] for r in range(len(rows))]
                    if v is not None]
        mn, mx = compute_normalizer(col_vals)
        low20 = percentile(col_vals, 0.20)
        high80 = percentile(col_vals, 0.80)
        num_bounds[i] = {
            "mn": mn, "mx": mx,
            "low20": low20, "high80": high80
        }

    # Build candidate conditions
    candidates = []  # list of tuples (col, cond_type, cond)
    for i in numeric_cols:
        b = num_bounds[i]
        # <= low20
        candidates.append((i, "range", (b["mn"], b["low20"])))
        # >= high80
        candidates.append((i, "range", (b["high80"], b["mx"])))
    for i in symbolic_cols:
        vals = set(v for v in [rows[r][i] for r in range(len(rows))]
                   if v is not None)
        for v in vals:
            candidates.append((i, "eq", v))

    # Helper to apply a rule to rows
    def apply_rule(cond_set):
        matched = []
        for r in range(len(rows)):
            ok = True
            for (ci, typ, cond) in cond_set:
                val = rows[r][ci]
                if val is None:
                    ok = False
                    break
                if typ == "eq":
                    if val != cond:
                        ok = False
                        break
                elif typ == "range":
                    lo, hi = cond
                    if not (lo <= val <= hi):
                        ok = False
                        break
            if ok:
                matched.append(r)
        return matched

    # Pre‑compute goal ranges for normalization
    goal_ranges = {}
    for g in goal_cols:
        idx = goal_idx[g]
        vals = [rows[r][idx] for r in range(len(rows)) if rows[r][idx] is not None]
        mn, mx = compute_normalizer(vals)
        maximize = g.endswith("+")
        goal_ranges[g] = (mn, mx, maximize)

    best_score = -math.inf
    best_rule = None
    best_count = 0

    # Evaluate all 1‑3 condition combinations
    for k in (1, 2, 3):
        for comb in itertools.combinations(candidates, k):
            matched = apply_rule(comb)
            if not matched:
                continue
            # Compute averages of goals
            avg_goals = {}
            skip = False
            for g in goal_cols:
                idx = goal_idx[g]
                vals = [rows[r][idx] for r in matched if rows[r][idx] is not None]
                if not vals:
                    skip = True
                    break
                avg_goals[g] = sum(vals) / len(vals)
            if skip:
                continue
            # Compute composite score
            score = 0.0
            for g in goal_cols:
                val = avg_goals[g]
                mn, mx, maximize = goal_ranges[g]
                norm = norm_value(val, mn, mx, maximize)
                if g.endswith("+"):
                    score += norm
                else:
                    score -= norm
            if score > best_score:
                best_score = score
                best_rule = comb
                best_count = len(matched)

    # Prepare JSON output
    rule_dict = {}
    if best_rule:
        for ci, typ, cond in best_rule:
            name = headers[ci]
            if typ == "eq":
                rule_dict[name] = cond
            else:  # range
                rule_dict[name] = [cond[0], cond[1]]
    answer = {"rule": rule_dict, "matching": best_count}

    # Print results
    print("== NASA-13 ==")
    print(f"Rule conditions (max 3): {rule_dict}")
    print(f"Matching rows count: {best_count}")
    print(f"Composite score: {best_score:.4f}")
    print(f"ANSWER NASA-13 {json.dumps(answer, separators=(',', ':'))}")
    # ------------------------------------------------------------

# ---------- NASA‑14 ------------------------------------------------------
def nasa14(rows, headers, goal_idx):
    """
    Recommend an evaluation order: list of row numbers to evaluate first.
    """
    # Goal columns
    goal_cols = [headers[i] for i in goal_idx]
    goal_names = {headers[i]: i for i in goal_idx}

    # Normalization ranges for all goals
    goal_ranges = {}
    for g in goal_cols:
        idx = goal_idx[g]
        vals = [rows[r][idx] for r in range(len(rows)) if rows[r][idx] is not None]
        mn, mx = compute_normalizer(vals)
        maximize = g.endswith("+")
        goal_ranges[g] = (mn, mx, maximize)

    # Compute score for each row
    scores = {}
    for r in range(len(rows)):
        # skip rows with missing any goal
        if any(rows[r][goal_idx[g]] is None for g in goal_cols):
            continue
        score = 0.0
        for g in goal_cols:
            val = rows[r][goal_idx[g]]
            mn, mx, maximize = goal_ranges[g]
            norm = norm_value(val, mn, mx, maximize)
            if g.endswith("+"):
                score += norm
            else:
                score -= norm
        scores[r] = score

    # Order by score descending
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    total_rows = len(rows)
    budget = min(30, total_rows // 5)

    order = [r + 1 for r, _ in ordered[:budget]]  # 1‑based

    answer = {"order": order}
    # Print results
    print("== NASA-14 ==")
    print(f"Evaluation budget: {budget} proposals")
    print("Top proposals (row numbers):")
    for idx in order:
        print(f"  {idx}")
    print(f"ANSWER NASA-14 {json.dumps(answer, separators=(',', ':'))}")

# ---------- main ---------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    # Load data
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = [[parse_value(v) for v in row] for row in rdr]

    # Identify goal columns
    goal_idx = {}
    for i, h in enumerate(header):
        if h.endswith(("+", "-")):
            goal_idx[h] = i

    nasa13(rows, header, goal_idx)
    nasa14(rows, header, goal_idx)

if __name__ == "__main__":
    main()
