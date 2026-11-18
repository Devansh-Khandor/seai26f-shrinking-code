#!/usr/bin/env python3
"""
Skill 07: NASA-13 and NASA-14

NASA-13 – Identify promising combinations of project characteristics that historically
delivered more software (high Kloc+) with lower development effort (Effort-)
and fewer defects (Defects-).

NASA-14 – Recommend an order for evaluating project proposals under a limited
budget so that the most promising alternatives are examined first.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Helpers ---------------------------------------------------------------
def atom(s: str) -> Any:
    """Coerce a CSV cell to a Python value."""
    s = s.strip()
    if s == "?":
        return s
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return {"True": True, "False": False}.get(s, s)

def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")

def numeric_column(values: List[Any]) -> bool:
    """Return True if all non‑missing values are numeric."""
    for v in values:
        if v == "?":
            continue
        if not isinstance(v, (int, float)):
            return False
    return True

# ----------------------------------------------------------------------
# NASA-13 ---------------------------------------------------------------
def nasa13(rows: List[List[Any]], headers: List[str]) -> Tuple[Dict[str, Any], int]:
    """Find a rule with up to 3 conditions that selects many rows and is
    characteristic of low effort and low defects.
    Returns the rule dict and the number of rows matched.
    """
    # Identify goal and non‑goal columns
    goal_cols = [c for c in headers if is_goal(c)]
    non_goal_cols = [c for c in headers if not is_goal(c)]

    # Find thresholds: 20th percentile for Effort- and Defects-
    effort_col = "Effort-"
    defects_col = "Defects-"
    eff_vals = [r[headers.index(effort_col)] for r in rows if isinstance(r[headers.index(effort_col)], (int, float))]
    def_vals = [r[headers.index(defects_col)] for r in rows if isinstance(r[headers.index(defects_col)], (int, float))]
    eff_thresh = sorted(eff_vals)[max(0, int(len(eff_vals)*0.2)-1)]
    def_thresh = sorted(def_vals)[max(0, int(len(def_vals)*0.2)-1)]

    # Subset S: rows with both low Effort- and low Defects-
    S = [r for r in rows
         if isinstance(r[headers.index(effort_col)], (int, float)) and r[headers.index(effort_col)] <= eff_thresh
         and isinstance(r[headers.index(defects_col)], (int, float)) and r[headers.index(defects_col)] <= def_thresh]

    if not S:
        S = rows  # fallback to all rows if nothing selected

    total_rows = len(rows)
    rule = {}
    col_stats = []

    for col in non_goal_cols:
        at = headers.index(col)
        col_vals = [r[at] for r in S]
        if numeric_column(col_vals):
            # numeric: use min and max
            values = [v for v in col_vals if v != "?"]
            if not values:
                continue
            lo, hi = min(values), max(values)
            # Count matches in full dataset
            match = sum(1 for r in rows
                        if isinstance(r[at], (int, float)) and lo <= r[at] <= hi)
            proportion = match / total_rows
            col_stats.append((proportion, col, [lo, hi], match))
        else:
            # symbolic: most common value in S
            counter = Counter(v for v in col_vals if v != "?")
            if not counter:
                continue
            val, freq = counter.most_common(1)[0]
            match = sum(1 for r in rows if r[at] == val)
            proportion = match / total_rows
            col_stats.append((proportion, col, val, match))

    # Pick up to 3 most selective conditions (smallest proportion)
    col_stats.sort(key=lambda x: x[0])
    selected = col_stats[:3]

    for _, col, cond, _ in selected:
        rule[col] = cond

    # Compute matching count for the rule
    def satisfies(row: List[Any]) -> bool:
        for col, cond in rule.items():
            at = headers.index(col)
            val = row[at]
            if isinstance(cond, list):  # numeric range
                if val == "?":
                    return False
                if not (cond[0] <= val <= cond[1]):
                    return False
            else:  # symbolic
                if val != cond:
                    return False
        return True

    matching = sum(1 for r in rows if satisfies(r))

    return rule, matching

# ----------------------------------------------------------------------
# NASA-14 ---------------------------------------------------------------
def nasa14(rows: List[List[Any]], headers: List[str]) -> List[int]:
    """Recommend an order for evaluating proposals under a limited budget.
    Order is based on a simple score combining Kloc+, Effort- and Defects-.
    """
    # Goal columns
    goal_cols = [c for c in headers if is_goal(c)]
    # Find numeric goals (all of them in this dataset)
    scores = []
    for idx, row in enumerate(rows, start=1):
        # All three goals must be present
        try:
            kloc = row[headers.index("Kloc+")]
            eff = row[headers.index("Effort-")]
            defc = row[headers.index("Defects-")]
            if kloc == "?" or eff == "?" or defc == "?":
                scores.append((None, idx))
            else:
                score = float(kloc) - float(eff) - float(defc)
                scores.append((score, idx))
        except Exception:
            scores.append((None, idx))

    # Sort by descending score, None last
    scores.sort(key=lambda x: (-float('inf') if x[0] is None else x[0]), reverse=True)

    # Budget
    N = len(rows)
    budget = min(30, N // 5)

    order = [idx for _, idx in scores[:budget]]
    return order

# ----------------------------------------------------------------------
# Main ---------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [[atom(v) for v in r] for r in reader]

    # NASA-13
    rule, match_count = nasa13(rows, headers)
    print("== NASA-13 ==")
    print(f"Rule: {rule}")
    print(f"Matching rows: {match_count}")
    print(f"Total rows: {len(rows)}")
    print(f"ANSWER NASA-13 {json.dumps({'rule': rule, 'matching': match_count})}")

    # NASA-14
    order = nasa14(rows, headers)
    print("\n== NASA-14 ==")
    print(f"Evaluation order: {order}")
    print(f"Budget: {len(order)}")
    print(f"ANSWER NASA-14 {json.dumps({'order': order})}")

if __name__ == "__main__":
    main()
