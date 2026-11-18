#!/usr/bin/env python3
"""
Skill 08: NASA-15 and NASA-16

This script identifies plausible changes to adjustable planning or process
characteristics that could reduce predicted effort (NASA-15) or
improve predicted quality (NASA-16) for a selected project proposal.
"""

import csv
import json
import sys
from typing import Any, Dict, List, Tuple

# Import helpers from skill_01
try:
    from skill_01 import read_data, GOAL_COLUMNS
except Exception as e:  # fallback if skill_01 is missing
    def read_data(csv_path: str) -> Tuple[List[Dict[str, Any]], List[int]]:
        rows: List[Dict[str, Any]] = []
        row_nums: List[int] = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for rn, rec in enumerate(reader, start=1):
                row_nums.append(rn)
                parsed = {}
                for k, v in rec.items():
                    v = v.strip()
                    if v == "?":
                        parsed[k] = None
                    else:
                        try:
                            if '.' in v or 'e' in v.lower():
                                parsed[k] = float(v)
                            else:
                                parsed[k] = int(v)
                        except ValueError:
                            parsed[k] = v
                rows.append(parsed)
        return rows, row_nums
    GOAL_COLUMNS = {
        "Kloc+": "max",
        "Effort-": "min",
        "Defects-": "min",
    }

# ------------------------------------------------------------------
# Helper utilities

def is_numeric(val: Any) -> bool:
    return isinstance(val, (int, float))

def compute_value_means(rows: List[Dict[str, Any]], col: str, target_col: str) -> Dict[Any, float]:
    """
    Return a mapping from each distinct value of `col` to the mean of `target_col`
    over all rows that have that value.
    """
    sums: Dict[Any, float] = {}
    counts: Dict[Any, int] = {}
    for r in rows:
        v = r.get(col)
        t = r.get(target_col)
        if v is None or not is_numeric(t):
            continue
        sums[v] = sums.get(v, 0.0) + t
        counts[v] = counts.get(v, 0) + 1
    means: Dict[Any, float] = {}
    for k, s in sums.items():
        c = counts.get(k, 0)
        if c:
            means[k] = s / c
    return means

def best_reduction_change(rows: List[Dict[str, Any]],
                          proj_idx: int,
                          target_col: str,
                          candidate_cols: List[str]) -> Tuple[Dict[str, Tuple[Any, Any]], float, float]:
    """
    For a given project, find the single adjustment (column -> new value)
    that maximizes reduction in the target column.
    Returns:
        changes dict (col -> (current_value, new_value)),
        before value,
        after mean value.
    If no reduction possible, returns empty dict and before==after.
    """
    proj = rows[proj_idx]
    before_val = proj.get(target_col)
    if not is_numeric(before_val):
        return {}, None, None
    best_change = {}
    best_delta = 0.0
    best_after = before_val
    for col in candidate_cols:
        # Compute means for each value
        value_means = compute_value_means(rows, col, target_col)
        current_val = proj.get(col)
        if current_val is None or current_val not in value_means:
            continue
        current_mean = value_means.get(current_val, before_val)
        # Find the value with lowest mean
        min_val, min_mean = min(value_means.items(), key=lambda kv: kv[1])
        if min_mean < current_mean:
            delta = current_mean - min_mean
            if delta > best_delta:
                best_delta = delta
                best_change = {col: (current_val, min_val)}
                best_after = min_mean
    return best_change, before_val, best_after

def get_candidate_columns(header: List[str]) -> List[str]:
    """Return list of non‑goal columns that can be adjusted."""
    return [c for c in header
            if c not in GOAL_COLUMNS and not c.endswith('X')]

# ------------------------------------------------------------------
# Main

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    rows, row_nums = read_data(csv_path)

    if not rows:
        print("No data rows found.")
        sys.exit(1)

    header = rows[0].keys()

    # Candidate columns for adjustment
    candidate_cols = get_candidate_columns(header)

    # ---------- NASA-15 ----------
    # Choose project with highest Effort- (actual)
    eff_projects = [(i, r) for i, r in enumerate(rows)
                    if is_numeric(r.get("Effort-"))]
    if not eff_projects:
        print("== NASA-15 ==")
        print("No projects with Effort- data.")
        print("ANSWER NASA-15 {}")
        # Continue to NASA-16
    else:
        proj_idx, proj = max(eff_projects, key=lambda kv: kv[1]["Effort-"])
        proj_row_num = row_nums[proj_idx]

        # Determine best change
        changes_dict, before_eff, after_eff = best_reduction_change(
            rows, proj_idx, "Effort-", candidate_cols)

        # Prepare human‑readable output
        print("== NASA-15 ==")
        print(f"Project row: {proj_row_num}")
        if before_eff is None:
            before_eff = "N/A"
        print(f"Predicted effort before: {before_eff}")
        print(f"Predicted effort after: {after_eff}")
        if changes_dict:
            print(f"Recommended changes: {changes_dict}")
        else:
            print("No beneficial changes found.")

        # Prepare JSON answer
        answer_n15 = {
            "project": proj_row_num,
            "changes": changes_dict,
            "predicted_effort": {
                "before": before_eff,
                "after": after_eff
            }
        }
        print(f"ANSWER NASA-15 {json.dumps(answer_n15)}")

    # ---------- NASA-16 ----------
    # Choose project with highest Defects- (actual)
    defect_projects = [(i, r) for i, r in enumerate(rows)
                       if is_numeric(r.get("Defects-"))]
    if not defect_projects:
        print("\n== NASA-16 ==")
        print("No projects with Defects- data.")
        print("ANSWER NASA-16 {}")
        return

    proj_idx, proj = max(defect_projects, key=lambda kv: kv[1]["Defects-"])
    proj_row_num = row_nums[proj_idx]

    # Find best change for Defects-
    changes_dict, before_def, after_def = best_reduction_change(
        rows, proj_idx, "Defects-", candidate_cols)

    # Also compute effort trade‑off
    # If change exists, compute mean Effort- for new value
    if changes_dict:
        # There will be only one column in changes_dict
        col, (curr, new_val) = next(iter(changes_dict.items()))
        # Mean Effort- for the new value
        effort_means = compute_value_means(rows, col, "Effort-")
        after_eff = effort_means.get(new_val, proj.get("Effort-"))
    else:
        after_eff = proj.get("Effort-")
    before_eff = proj.get("Effort-")

    # Human‑readable output
    print("\n== NASA-16 ==")
    print(f"Project row: {proj_row_num}")
    print(f"Predicted defects before: {before_def}")
    print(f"Predicted defects after: {after_def}")
    print(f"Predicted effort before: {before_eff}")
    print(f"Predicted effort after: {after_eff}")
    if changes_dict:
        print(f"Recommended changes: {changes_dict}")
    else:
        print("No beneficial changes found.")

    # JSON answer
    answer_n16 = {
        "project": proj_row_num,
        "changes": changes_dict,
        "predicted_defects": {
            "before": before_def,
            "after": after_def
        },
        "predicted_effort": {
            "before": before_eff,
            "after": after_eff
        }
    }
    print(f"ANSWER NASA-16 {json.dumps(answer_n16)}")

if __name__ == "__main__":
    main()
