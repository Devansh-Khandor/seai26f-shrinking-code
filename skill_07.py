#!/usr/bin/env python3
"""
skill_07.py

NASA-13: Identify a rule on non‑goal columns that selects projects with
more software (higher Kloc+) while keeping development effort and defects low.

NASA-14: Given a limited evaluation budget, rank proposals by a composite
score (maximize Kloc+, minimize Effort- and Defects-) and output the order
of first evaluations.

Usage:
    python skill_07.py <csv_path>
"""

import csv
import json
import sys
import math
from typing import List, Dict, Tuple, Any, Set

# ------------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------------
def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV and return list of row dicts and header list."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = [row for row in reader]
    return rows, header


def to_float(val: str) -> float:
    """Convert string to float; return NaN if missing or not convertible."""
    if val.strip() == "?":
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


def is_numeric(val: str) -> bool:
    """True if val can be parsed as float and is not '?'."""
    if val.strip() == "?":
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------
# NASA-13 helpers
# ------------------------------------------------------------------
def identify_best_rule(rows: List[Dict[str, str]], header: List[str]) -> Tuple[Dict[str, Any], int, float]:
    """Find a rule (1 condition on a non‑goal column) that best aligns with high Kloc+,
    low Effort- and low Defects-.

    Returns:
        rule_obj: mapping column -> value (string or [low, high] list)
        match_count: number of rows matching the rule
        score: composite score of the rule
    """
    # Determine goal and X columns
    goal_cols = {c for c in header if c.endswith("+") or c.endswith("-")}
    x_cols = {c for c in header if c.endswith("X")}
    non_goal_cols = [c for c in header if c not in goal_cols and c not in x_cols]

    best_rule: Dict[str, Any] | None = None
    best_score = -float("inf")
    best_match = 0

    # Pre‑compute goal values per row
    goal_values: List[Dict[str, float]] = []
    for r in rows:
        gv = {}
        for g in ("Kloc+", "Effort-", "Defects-"):
            gv[g] = to_float(r[g])
        goal_values.append(gv)

    for col in non_goal_cols:
        # Determine if column is numeric (all non‑missing values parseable)
        numeric = True
        for r in rows:
            v = r[col]
            if v.strip() == "?":
                continue
            if not is_numeric(v):
                numeric = False
                break

        # Build mapping value -> list of row indices (1‑based)
        groups: Dict[Any, List[int]] = {}
        for idx, r in enumerate(rows, start=1):
            v = r[col]
            if v.strip() == "?":
                continue
            key = float(v) if numeric else v
            groups.setdefault(key, []).append(idx)

        for val, idxs in groups.items():
            # Count matching rows
            match_count = len(idxs)

            # Compute mean Kloc+, Effort-, Defects- for rows in this group
            k_vals = []
            e_vals = []
            d_vals = []
            for i in idxs:
                gv = goal_values[i - 1]
                if not math.isnan(gv["Kloc+"]) and not math.isnan(gv["Effort-"]) and not math.isnan(gv["Defects-"]):
                    k_vals.append(gv["Kloc+"])
                    e_vals.append(gv["Effort-"])
                    d_vals.append(gv["Defects-"])
            if not k_vals:
                continue  # skip groups with no complete goal data

            mean_k = sum(k_vals) / len(k_vals)
            mean_e = sum(e_vals) / len(e_vals)
            mean_d = sum(d_vals) / len(d_vals)
            # Composite: maximize Kloc+, minimize Effort- and Defects-
            score = mean_k - mean_e - mean_d

            if score > best_score or (abs(score - best_score) < 1e-12 and match_count > best_match):
                best_score = score
                best_match = match_count
                if numeric:
                    # Use inclusive numeric range; if single value, low==high
                    best_rule = {col: [round(float(val), 4), round(float(val), 4)]}
                else:
                    best_rule = {col: val}
    if best_rule is None:
        # Fallback: empty rule
        return {}, 0, 0.0
    return best_rule, best_match, best_score


# ------------------------------------------------------------------
# NASA-14 helpers
# ------------------------------------------------------------------
def compute_composite(row: Dict[str, str]) -> float:
    """Return composite score for a row: Kloc+ - Effort- - Defects-.
    Rows with missing goals return -inf.
    """
    k = to_float(row["Kloc+"])
    e = to_float(row["Effort-"])
    d = to_float(row["Defects-"])
    if math.isnan(k) or math.isnan(e) or math.isnan(d):
        return -float("inf")
    return k - e - d


def rank_projects(rows: List[Dict[str, str]]) -> List[int]:
    """Return order of project indices based on composite score (higher better)."""
    scores = [compute_composite(r) for r in rows]
    # Pair indices (1-based) with score
    paired = list(enumerate(scores, start=1))
    # Sort descending by score
    paired.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in paired]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = argv[1]
    rows, header = parse_csv(csv_path)

    # ---------------- NASA-13 ----------------
    rule, match_count, rule_score = identify_best_rule(rows, header)
    # Prepare JSON answer
    answer_13 = {"rule": rule, "matching": match_count}
    print("== NASA-13 ==")
    # Human‑readable output (at least three lines)
    print(f"Rule: {json.dumps(rule)}")
    print(f"Matching rows: {match_count}")
    print(f"Composite score: {round(rule_score, 4)}")
    print("ANSWER NASA-13 " + json.dumps(answer_13))

    # ---------------- NASA-14 ----------------
    # Determine budget: min(30, n//5)
    n_rows = len(rows)
    budget = min(30, n_rows // 5)
    # Get ranking
    ranked = rank_projects(rows)
    order = ranked[:budget]
    # Prepare JSON answer
    answer_14 = {"order": order}
    print("\n== NASA-14 ==")
    # Human‑readable (at least three lines)
    print(f"Evaluation budget: {budget} projects")
    print(f"Top 3 to evaluate first: {order[:3]}")
    print(f"Full order (first {budget}): {order}")
    print("ANSWER NASA-14 " + json.dumps(answer_14))


if __name__ == "__main__":
    main(sys.argv)
