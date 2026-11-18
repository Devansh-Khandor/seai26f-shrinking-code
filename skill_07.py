#!/usr/bin/env python3
"""
Skill 07 – NASA-13 & NASA-14
"""
import csv
import json
import sys
from collections import defaultdict

# Re‑use helpers from earlier skills
from skill_01 import parse_value, normalise_composite

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GOALS = ["Kloc+", "Effort-", "Defects-"]  # order matters for output


def load_rows(csv_path: str):
    """
    Load CSV rows keeping string values for all columns; parse goal columns to float.
    Returns a list of row dictionaries (1‑based indices implied by list order).
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for col, val in row.items():
                if col in GOALS:
                    parsed[col] = parse_value(val)
                else:
                    parsed[col] = val
            rows.append(parsed)
    return rows


# ----------------------------------------------------------------------
# NASA-13: Identify promising combinations of characteristics
# ----------------------------------------------------------------------
def nasa_13(rows):
    """
    Find up to 3 conditions (non‑goal columns) that together identify rows with low
    Effort- and low Defects-.

    The rule is a mapping of column -> value (for categorical columns).
    """
    # Gather statistics per (column, value) pair
    stats = defaultdict(lambda: {"effort_vals": [], "defect_vals": []})

    for idx, row in enumerate(rows, start=1):
        effort = row["Effort-"]
        defect = row["Defects-"]
        if effort is None or defect is None:
            continue  # need both goals present for comparison
        for col, val in row.items():
            if col in GOALS:
                continue
            if val == "" or val == "?":
                continue
            stats[(col, val)]["effort_vals"].append(effort)
            stats[(col, val)]["defect_vals"].append(defect)

    # For each column find the value that yields the lowest composite average
    best_per_col = {}
    for (col, val), d in stats.items():
        eff_vals = d["effort_vals"]
        def_vals = d["defect_vals"]
        if not eff_vals or not def_vals:
            continue
        avg_eff = sum(eff_vals) / len(eff_vals)
        avg_def = sum(def_vals) / len(def_vals)
        composite_avg = avg_eff + avg_def  # lower is better
        if col not in best_per_col or composite_avg < best_per_col[col]["composite"]:
            best_per_col[col] = {
                "value": val,
                "composite": composite_avg,
                "count": len(eff_vals),
            }

    # Sort columns by the composite score (lower better)
    sorted_cols = sorted(best_per_col.items(), key=lambda x: x[1]["composite"])

    # Pick up to three conditions
    rule = {}
    for col, info in sorted_cols[:3]:
        rule[col] = info["value"]

    # Compute number of rows that satisfy all conditions
    matching = 0
    for row in rows:
        ok = True
        for col, val in rule.items():
            if row.get(col, None) != val:
                ok = False
                break
        if ok:
            matching += 1

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print("== NASA-13 ==")
    print(f"Rule conditions: {rule}")
    print(f"Matching rows: {matching}")
    # Machine‑checked answer
    answer = {"rule": rule, "matching": matching}
    print("ANSWER NASA-13 " + json.dumps(answer, separators=(",", ":")))


# ----------------------------------------------------------------------
# NASA-14: Recommend evaluation order with a budget
# ----------------------------------------------------------------------
def nasa_14(rows):
    """
    Order rows by composite score (higher is better).  Return the top N rows,
    where N = min(30, number_of_rows // 5).
    """
    scores = normalise_composite(rows)  # list of (row_num, composite)

    # Sort descending by composite
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)

    total_rows = len(rows)
    budget = min(30, total_rows // 5)

    # Take top budget rows
    order = [row_num for row_num, _ in scores_sorted[:budget]]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print("\n== NASA-14 ==")
    print(f"Evaluation budget: {budget} rows")
    print(f"Suggested order (first 5 shown): {order[:5]}")
    # Machine‑checked answer
    answer = {"order": order}
    print("ANSWER NASA-14 " + json.dumps(answer, separators=(",", ":")))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_rows(csv_path)

    nasa_13(rows)
    nasa_14(rows)


if __name__ == "__main__":
    main()
