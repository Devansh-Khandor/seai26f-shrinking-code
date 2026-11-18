#!/usr/bin/env python3
"""
skill_06.py

Usage:
    python skill_06.py <csv_path>
"""

import csv
import json
import sys
import math
from typing import List, Dict, Any, Tuple, Set

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV and return rows as list of dicts and header list."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return rows, header


def is_numeric(val: str) -> bool:
    """True if val can be parsed as float and is not '?'."""
    if val.strip() == "?":
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False


def to_float(val: str) -> float:
    """Convert string to float; return NaN if missing or not convertible."""
    if val.strip() == "?":
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


def get_feature_names(header: List[str]) -> List[str]:
    """All columns that are neither goal nor X."""
    goal_cols = {c for c in header if c.endswith("+") or c.endswith("-")}
    x_cols = {c for c in header if c.endswith("X")}
    return [c for c in header if c not in goal_cols and c not in x_cols]


def compute_distance(row1: Dict[str, str], row2: Dict[str, str],
                     feature_names: List[str]) -> float:
    """Euclidean-like distance: sum of squared numeric diff + Hamming for categorical."""
    d_sq = 0.0
    for col in feature_names:
        v1 = row1.get(col, "?")
        v2 = row2.get(col, "?")
        if v1.strip() == "?" or v2.strip() == "?":
            continue  # skip missing features
        if is_numeric(v1) and is_numeric(v2):
            diff = to_float(v1) - to_float(v2)
            d_sq += diff * diff
        else:
            d_sq += 0.0 if v1 == v2 else 1.0
    return math.sqrt(d_sq)


def find_neighbors(target_row: Dict[str, str],
                   rows: List[Tuple[int, Dict[str, str]]],
                   feature_names: List[str],
                   k: int = 3) -> List[Tuple[int, Dict[str, str]]]:
    """Return k nearest neighbors (row index, row dict) sorted by distance."""
    distances: List[Tuple[float, int, Dict[str, str]]] = []
    for idx, row in rows:
        dist = compute_distance(target_row, row, feature_names)
        distances.append((dist, idx, row))
    distances.sort(key=lambda x: x[0])
    return [(idx, row) for _, idx, row in distances[:k]]


def round_val(v: Any) -> Any:
    """Round float to 4 decimals; leave other types unchanged."""
    if isinstance(v, float):
        return round(v, 4)
    return v


def get_goal_columns(header: List[str]) -> Dict[str, bool]:
    """Map goal column to direction: True if maximize, False if minimize."""
    goals = {}
    for c in header:
        if c.endswith("+"):
            goals[c] = True
        elif c.endswith("-"):
            goals[c] = False
    return goals


def compute_risk(row: Dict[str, str]) -> float:
    """Risk defined as sum of Effort- and Defects- (both minimized)."""
    eff = to_float(row.get("Effort-", "?"))
    defects = to_float(row.get("Defects-", "?"))
    if math.isnan(eff) or math.isnan(defects):
        return float("nan")
    return eff + defects


def find_min_max_risk_rows(rows: List[Dict[str, str]]) -> Tuple[Tuple[int, Dict[str, str]],
                                                                Tuple[int, Dict[str, str]]]:
    """Return (min risk row, max risk row) with their indices (1-based)."""
    min_risk = float("inf")
    max_risk = -float("inf")
    min_row = None
    max_row = None
    for idx, row in enumerate(rows, start=1):
        risk = compute_risk(row)
        if math.isnan(risk):
            continue
        if risk < min_risk:
            min_risk = risk
            min_row = (idx, row)
        if risk > max_risk:
            max_risk = risk
            max_row = (idx, row)
    return min_row, max_row


def differences_between_rows(row_a: Dict[str, str], row_b: Dict[str, str],
                             non_goal_cols: List[str]) -> List[Dict[str, Any]]:
    diffs = []
    for col in non_goal_cols:
        val_a = row_a.get(col, "?")
        val_b = row_b.get(col, "?")
        if val_a.strip() == "?" or val_b.strip() == "?":
            continue
        if val_a != val_b:
            diffs.append({"feature": col, "a": val_a, "b": val_b})
    return diffs


def tradeoffs_between_rows(row_a: Dict[str, str], row_b: Dict[str, str],
                           goal_columns: Dict[str, bool]) -> List[str]:
    """Return goal columns where non-preferred (row_b) is better."""
    tradeoffs = []
    for col, maximize in goal_columns.items():
        val_a = to_float(row_a.get(col, "?"))
        val_b = to_float(row_b.get(col, "?"))
        if math.isnan(val_a) or math.isnan(val_b):
            continue
        if maximize:
            if val_b > val_a:
                tradeoffs.append(col)
        else:
            if val_b < val_a:
                tradeoffs.append(col)
    return tradeoffs


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)
    csv_path = argv[1]
    rows, header = parse_csv(csv_path)

    # Identify feature, goal and X columns
    feature_names = get_feature_names(header)
    goal_columns = get_goal_columns(header)
    x_columns = {c for c in header if c.endswith("X")}

    # ---------------- NASA-11 ----------------
    # Find held-out rows (divisible by 5)
    held_out_indices = [i + 1 for i in range(len(rows)) if (i + 1) % 5 == 0]
    if not held_out_indices:
        print("== NASA-11 ==")
        print("No held‑out rows available.")
        print("ANSWER NASA-11 {}")
    else:
        target_idx = held_out_indices[0]
        target_row = rows[target_idx - 1]

        # Prepare list of non‑held‑out rows with indices
        non_held_rows = [(i + 1, r) for i, r in enumerate(rows) if (i + 1) not in held_out_indices]

        # Find nearest neighbors
        neighbors = find_neighbors(target_row, non_held_rows, feature_names, k=3)

        # Estimate Effort- as mean of neighbors' Effort-
        neighbor_efforts = [to_float(n[1]["Effort-"]) for n in neighbors]
        estimate = sum(neighbor_efforts) / len(neighbor_efforts)
        estimate = round(estimate, 4)

        # Prepare human‑readable output
        print("== NASA-11 ==")
        print(f"Project row {target_idx} estimate Effort- = {estimate}")
        print(f"Nearest historical projects (by feature similarity):")
        for idx, row in neighbors:
            print(f"  Row {idx}: Effort- = {row['Effort-']}")
        print()

        # Prepare JSON answer
        neighbors_output = []
        for idx, row in neighbors:
            neighbors_output.append({
                "row": idx,
                "values": {col: row[col] for col in header}
            })
        answer_11 = {
            "project": target_idx,
            "estimate": estimate,
            "neighbors": neighbors_output
        }
        print("ANSWER NASA-11 " + json.dumps(answer_11))

    # ---------------- NASA-12 ----------------
    # Identify two rows with most different risk scores
    min_row, max_row = find_min_max_risk_rows(rows)
    if not min_row or not max_row:
        print("\n== NASA-12 ==")
        print("Not enough data to compute risk.")
        print("ANSWER NASA-12 {}")
    else:
        idx_a, row_a = min_row  # preferred (lower risk)
        idx_b, row_b = max_row  # non‑preferred (higher risk)

        risk_a = compute_risk(row_a)
        risk_b = compute_risk(row_b)

        # Compute differences in all non‑goal columns
        non_goal_cols = [c for c in header if c not in goal_columns and c not in x_columns]
        diff_list = differences_between_rows(row_a, row_b, non_goal_cols)

        # Tradeoffs: goal columns where the non‑preferred row is better
        tradeoff_list = tradeoffs_between_rows(row_a, row_b, goal_columns)

        # Human‑readable
        print("\n== NASA-12 ==")
        print(f"Preferred row (lower risk): {idx_a} (Risk={risk_a:.2f})")
        print(f"Non‑preferred row (higher risk): {idx_b} (Risk={risk_b:.2f})")
        print(f"Tradeoffs where row {idx_b} is better: {', '.join(tradeoff_list) if tradeoff_list else 'None'}")
        print()

        # JSON answer
        answer_12 = {
            "a": idx_a,
            "b": idx_b,
            "preferred": idx_a,
            "scores": {
                str(idx_a): round(risk_a, 4),
                str(idx_b): round(risk_b, 4)
            },
            "differences": diff_list[:max(1, len(diff_list))],  # at least one
            "tradeoffs": tradeoff_list
        }
        print("ANSWER NASA-12 " + json.dumps(answer_12))


if __name__ == "__main__":
    main(sys.argv)
