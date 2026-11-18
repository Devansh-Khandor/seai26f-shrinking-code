#!/usr/bin/env python3
"""
Skill 06: NASA-11 and NASA-12

This script implements two NASA requests:

  * NASA-11 – For a held‑out project estimate the effort required,
    explain the estimate using the three closest historical projects
    and their recorded characteristics.

  * NASA-12 – For two held‑out projects that receive different risk
    assessments, explain the main differences that contributed to
    those assessments and identify the important trade‑offs.

The script reads a CSV file of software projects, performs the analyses,
and prints human‑readable output followed by a single‑line JSON answer
for each request.  The JSON is machine‑checked.

The implementation follows the same data‑handling conventions as the
previous skills: missing values are left as “?”, rows are 1‑based, and
the last round(0.2 × N) rows are considered arrivals.  The only
difference is that this skill uses a simple k‑nearest‑neighbour model
to predict Effort‑ and Defects‑, and uses the predictions to
compute a score that combines Kloc+, Effort‑ and Defects‑.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
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


def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return list of rows (dict) and header names."""
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = [dict(zip(header, [atom(v) for v in r])) for r in rdr]
    return rows, header


def is_goal(name: str) -> bool:
    return name.endswith("+") or name.endswith("-")


def is_input(name: str) -> bool:
    return not (name.endswith("X") or is_goal(name))


# ----------------------------------------------------------------------
# Distance & K‑NN
# ----------------------------------------------------------------------
def distance(row1: Dict[str, Any], row2: Dict[str, Any],
             input_names: List[str]) -> float:
    """Euclidean distance over input columns, ignoring missing values."""
    s = 0.0
    count = 0
    for n in input_names:
        v1, v2 = row1[n], row2[n]
        if v1 == "?" or v2 == "?" or v1 == v2 and isinstance(v1, str):
            if v1 == v2:
                continue
            if isinstance(v1, str) and isinstance(v2, str):
                # categorical, unequal
                s += 1
                count += 1
                continue
            else:
                # one missing, skip
                continue
        # numeric
        try:
            diff = float(v1) - float(v2)
            s += diff * diff
            count += 1
        except Exception:
            # treat as categorical
            if v1 != v2:
                s += 1
                count += 1
    return math.sqrt(s) if count else float("inf")


def knn_neighbors(target_row: Dict[str, Any],
                  rows: List[Dict[str, Any]],
                  exclude: set,
                  k: int,
                  input_names: List[str]) -> List[Dict[str, Any]]:
    """Return k nearest neighbours (excluding rows in `exclude`)."""
    dists = []
    for r in rows:
        if id(r) in exclude:
            continue
        dists.append((distance(target_row, r, input_names), r))
    dists.sort(key=lambda x: x[0])
    return [r for _, r in dists[:k]]


def mean_value(values: List[Any]) -> float:
    """Mean of numeric list (ignore non‑numeric)."""
    nums = [v for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else 0.0


def predict_effort(row: Dict[str, Any], rows: List[Dict[str, Any]],
                   exclude: set, k: int,
                   input_names: List[str]) -> float:
    neigh = knn_neighbors(row, rows, exclude, k, input_names)
    vals = [r["Effort-"] for r in neigh if r["Effort-"] != "?"]
    return mean_value(vals)


def predict_defects(row: Dict[str, Any], rows: List[Dict[str, Any]],
                    exclude: set, k: int,
                    input_names: List[str]) -> float:
    neigh = knn_neighbors(row, rows, exclude, k, input_names)
    vals = [r["Defects-"] for r in neigh if r["Defects-"] != "?"]
    return mean_value(vals)


def compute_score(row: Dict[str, Any], est_effort: float,
                  est_defects: float) -> float:
    """Kloc+ - Effort- - Defects-  (higher is better)."""
    kloc = row["Kloc+"]
    if kloc == "?":
        kloc = 0.0
    return float(kloc) - est_effort - est_defects


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows, header = read_csv(csv_path)

    # Identify indices
    goal_names = [n for n in header if is_goal(n)]
    input_names = [n for n in header if is_input(n)]
    # Map from row id to row for quick lookup
    row_id_map = {i + 1: r for i, r in enumerate(rows)}

    # Held‑out rows: divisible by 5
    held_out_indices = [i + 1 for i, _ in enumerate(rows) if (i + 1) % 5 == 0]

    # History rows: all except arrivals
    total = len(rows)
    arrivals = int(round(0.2 * total))
    history_rows = rows[:-arrivals]
    history_set = set(id(r) for r in history_rows)

    # ------------------------------------------------------------
    # NASA‑11 – Explain one held‑out project
    # ------------------------------------------------------------
    if not held_out_indices:
        print("== NASA-11 ==")
        print("No held‑out rows available.")
        print("ANSWER NASA-11 {}")
        print("== NASA-12 ==")
        print("No held‑out rows available.")
        print("ANSWER NASA-12 {}")
        return

    project_no = held_out_indices[0]
    project_row = row_id_map[project_no]

    est_eff = predict_effort(project_row, rows, history_set, 3, input_names)
    est_def = predict_defects(project_row, rows, history_set, 3, input_names)

    neighbors = knn_neighbors(project_row, rows, history_set, 3, input_names)
    neigh_info = []
    for n in neighbors:
        # include all columns
        neigh_info.append({
            "row": list(row_id_map.keys())[list(row_id_map.values()).index(n)],
            "values": {col: n[col] for col in header}
        })

    # Prepare output
    print("== NASA-11 ==")
    print(f"Project (row {project_no}):")
    print(f"Estimated Effort-: {est_eff:.4f}")
    print("Three most similar historical projects:")
    for nb in neigh_info:
        print(f"  Row {nb['row']}: {nb['values']}")

    answer_11 = {
        "project": project_no,
        "estimate": est_eff,
        "neighbors": neigh_info
    }
    print(f"\nANSWER NASA-11 {json.dumps(answer_11, separators=(',', ':'))}")

    # ------------------------------------------------------------
    # NASA‑12 – Compare two held‑out projects
    # ------------------------------------------------------------
    if len(held_out_indices) < 2:
        print("\n== NASA-12 ==")
        print("Only one held‑out row available for comparison.")
        print("ANSWER NASA-12 {}")
        return

    a_no, b_no = held_out_indices[0], held_out_indices[1]
    a_row, b_row = row_id_map[a_no], row_id_map[b_no]

    # Predictions
    a_eff = predict_effort(a_row, rows, history_set, 3, input_names)
    a_def = predict_defects(a_row, rows, history_set, 3, input_names)
    b_eff = predict_effort(b_row, rows, history_set, 3, input_names)
    b_def = predict_defects(b_row, rows, history_set, 3, input_names)

    a_score = compute_score(a_row, a_eff, a_def)
    b_score = compute_score(b_row, b_eff, b_def)

    preferred = a_no if a_score > b_score else b_no
    scores = {"a": a_score, "b": b_score}

    # Differences in feature values
    diff_list = []
    for col in header:
        if col in goal_names:
            continue
        va, vb = a_row[col], b_row[col]
        if va != vb:
            diff_list.append({"feature": col, "a": va, "b": vb})
    if not diff_list:
        diff_list.append({"feature": "none", "a": None, "b": None})

    # Trade‑offs: goal columns where non‑preferred has better value
    tradeoffs = []
    for g in goal_names:
        va, vb = a_row[g], b_row[g]
        # use predictions if actual missing
        if va == "?":
            va = a_eff if g == "Effort-" else a_def if g == "Defects-" else va
        if vb == "?":
            vb = b_eff if g == "Effort-" else b_def if g == "Defects-" else vb
        if isinstance(va, str) or isinstance(vb, str):
            continue
        if g.endswith("+"):
            # higher better
            better = (vb > va) if preferred == a_no else (va > vb)
        else:
            # lower better
            better = (vb < va) if preferred == a_no else (va < vb)
        if better:
            tradeoffs.append(g)

    # Output
    print("\n== NASA-12 ==")
    print(f"Projects {a_no} and {b_no}:")
    print(f"Preferred: Row {preferred} (score {scores['a'] if preferred==a_no else scores['b']:.4f})")
    print(f"Scores: a={scores['a']:.4f}, b={scores['b']:.4f}")
    print("Differences:")
    for d in diff_list:
        print(f"  {d['feature']}: a={d['a']} vs b={d['b']}")
    print("Trade‑offs where the non‑preferred project is better:")
    for t in tradeoffs:
        print(f"  {t}")

    answer_12 = {
        "a": a_no,
        "b": b_no,
        "preferred": preferred,
        "scores": scores,
        "differences": diff_list,
        "tradeoffs": tradeoffs
    }
    print(f"\nANSWER NASA-12 {json.dumps(answer_12, separators=(',', ':'))}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
