#!/usr/bin/env python3
"""
Skill 07 – NASA project exploration and evaluation prioritisation

This script reads a NASA project CSV file and prints two analyses:
  NASA-13 – Identify promising combinations of project characteristics that
    historically delivered more software with lower development effort and
    fewer defects.  The result is a rule with 1‑3 conditions on non‑goal
    columns, and the number of rows that match that rule.

  NASA-14 – When detailed evaluations are expensive, recommend which project
    proposals should be evaluated first to discover promising alternatives
    with a limited evaluation budget.  The ordering is based on a weighted
    score of the goal columns only.

Both sections include a machine‑checked JSON line.  The program uses only
the data supplied on the command line and never hard‑codes values.

The script is run as:
    python skill_07.py <csv_path>
"""

import json
import sys

import ezr
import skill_01 as s01


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _norm(val, minv, maxv, goal):
    """Return a normalised value in [0,1] for a goal column."""
    if val == "?":
        return None
    if maxv == minv:
        return 0.5
    if goal == "+":
        return (val - minv) / (maxv - minv)
    # goal == "-"
    return (maxv - val) / (maxv - minv)


def _combined_score(row, goal_cols, stats):
    """Compute combined score for a row using the three goal columns."""
    score = 0.0
    for col in goal_cols:
        val = row[col.at]
        if val == "?":
            return None
        minv, maxv = stats[col.txt]["min"], stats[col.txt]["max"]
        score += _norm(val, minv, maxv, col.txt[-1])
    return score


def _numeric_range(values):
    """Return (min, max) of numeric values, ignoring missing."""
    vals = [v for v in values if v != "?"]
    return min(vals) if vals else None, max(vals) if vals else None


# ----------------------------------------------------------------------
# NASA-13
# ----------------------------------------------------------------------
def nasa_13(tbl):
    # Identify the goal columns (Kloc+, Effort-, Defects-)
    goal_cols = [
        col
        for col in tbl.cols.values()
        if col.txt in ("Kloc+", "Effort-", "Defects-")
    ]

    # Statistics for the goal columns
    goal_stats = {col.txt: s01._stats([r[col.at] for r in tbl.rows if r[col.at] != "?"],
                                      (max([r[col.at] for r in tbl.rows if r[col.at] != "?"]),
                                       min([r[col.at] for r in tbl.rows if r[col.at] != "?"])))
                 for col in goal_cols}

    # Compute scores for all rows
    scored = []
    for i, row in enumerate(tbl.rows, start=1):
        sc = _combined_score(row, goal_cols, goal_stats)
        if sc is not None:
            scored.append((sc, i, row))
    if not scored:
        print("== NASA-13 ==")
        print("No rows with complete goal data.")
        print('ANSWER NASA-13 {"rule":{},"matching":0}')
        return

    # Take top 10 rows
    scored.sort(reverse=True, key=lambda x: x[0])
    top_rows = [r[2] for r in scored[:10]]

    # Analyse non‑goal columns
    condition_candidates = {}
    for col in tbl.cols.values():
        if col.txt in ("Kloc+", "Effort-", "Defects-") or col.txt.endswith("X"):
            continue

        values = [r[col.at] for r in top_rows if r[col.at] != "?"]
        if not values:
            continue

        if isinstance(col, ezr.Sym):
            # most common value
            val = s01._most_common(values)
            if val is not None:
                condition_candidates[col.txt] = ("sym", val)
        else:
            low, high = _numeric_range(values)
            if low is not None:
                condition_candidates[col.txt] = ("num", (low, high))

    # Rank conditions: for symbols by support in top rows, for numbers
    # by how tight the range is relative to overall range
    ranked = []
    for name, (ctype, cval) in condition_candidates.items():
        col = tbl.cols[name]
        if ctype == "sym":
            support = sum(1 for r in top_rows if r[col.at] == cval) / len(top_rows)
            ranked.append((support, name, ctype, cval))
        else:
            overall_low, overall_high = _numeric_range([r[col.at] for r in tbl.rows if r[col.at] != "?"])
            range_span = overall_high - overall_low
            if range_span == 0:
                tightness = 1.0
            else:
                tightness = 1.0 - (cval[1] - cval[0]) / range_span
            ranked.append((tightness, name, ctype, cval))

    ranked.sort(reverse=True, key=lambda x: x[0])
    chosen = ranked[:3]

    # Build rule dict
    rule = {}
    for _, name, ctype, cval in chosen:
        if ctype == "sym":
            rule[name] = cval
        else:
            rule[name] = [cval[0], cval[1]]

    # Count matching rows
    matching = 0
    for r in tbl.rows:
        ok = True
        for name, ctype, cval in [(n, t, v) for (_, n, t, v) in chosen]:
            val = r[tbl.cols[name].at]
            if val == "?":
                ok = False
                break
            if t == "sym":
                if val != cval:
                    ok = False
                    break
            else:
                if not (cval[0] <= val <= cval[1]):
                    ok = False
                    break
        if ok:
            matching += 1

    # Output
    print("== NASA-13 ==")
    print("Proposed rule (1‑3 conditions):")
    for k, v in rule.items():
        if isinstance(v, list):
            print(f"  {k} ∈ [{v[0]:.2f}, {v[1]:.2f}]")
        else:
            print(f"  {k} = {v}")
    print(f"Rows matching the rule: {matching}")

    ans = {"rule": rule, "matching": matching}
    print(f"ANSWER NASA-13 {json.dumps(ans, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# NASA-14
# ----------------------------------------------------------------------
def nasa_14(tbl):
    # Goal columns
    goal_cols = [
        col
        for col in tbl.cols.values()
        if col.txt in ("Kloc+", "Effort-", "Defects-")
    ]

    # Stats for normalisation
    goal_stats = {col.txt: s01._stats([r[col.at] for r in tbl.rows if r[col.at] != "?"],
                                      (max([r[col.at] for r in tbl.rows if r[col.at] != "?"]),
                                       min([r[col.at] for r in tbl.rows if r[col.at] != "?"])))
                 for col in goal_cols}

    # Compute score for every row
    scored = []
    for i, row in enumerate(tbl.rows, start=1):
        sc = _combined_score(row, goal_cols, goal_stats)
        if sc is not None:
            scored.append((sc, i))

    # Order by descending score
    scored.sort(reverse=True, key=lambda x: x[0])
    budget = min(30, len(tbl.rows) // 5)
    order = [idx for _, idx in scored[:budget]]

    # Output
    print("== NASA-14 ==")
    print("Evaluation order (row numbers, 1‑based):")
    for idx in order:
        print(f"  {idx}")

    ans = {"order": order}
    print(f"ANSWER NASA-14 {json.dumps(ans, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    tbl = ezr.Tbl(ezr.csv(csv_path))
    nasa_13(tbl)
    nasa_14(tbl)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_07.py <csv_path>")
    main(sys.argv[1])
