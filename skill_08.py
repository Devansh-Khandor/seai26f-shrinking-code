#!/usr/bin/env python3
"""
skill_08.py: Identify and recommend changes to reduce effort or defects in NASA proposals.
Usage: python skill_08.py <csv-file>
"""

import json
import sys
import itertools
from typing import List, Dict, Tuple

import ezr


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------
def read_table(path: str) -> ezr.TBL:
    """Read CSV into an ezr table."""
    rows = list(ezr.csv(path))
    return ezr.Tbl(rows)


def get_col_by_name(tbl: ezr.TBL, name: str) -> ezr.COL:
    return next(c for c in tbl.cols.values() if c.txt == name)


# ------------------------------------------------------------------
# Neighbor based prediction (effort & defects)
# ------------------------------------------------------------------
def estimate_value_via_neighbors(
    tbl: ezr.TBL,
    target_row: ezr.ROW,
    others: List[Tuple[int, ezr.ROW]],
    goal_col: ezr.COL,
    min_neighbors: int = 3,
) -> float:
    """Estimate the value of a goal column for target_row using nearest neighbors."""
    distances = []
    for idx, row in others:
        d = ezr.xdist(tbl, target_row, row)
        distances.append((d, idx, row))
    distances.sort(key=lambda x: x[0])

    total = 0.0
    count = 0
    for _, _, row in distances:
        val = row[goal_col.at]
        if val == "?":
            continue
        total += float(val)
        count += 1
        if count >= min_neighbors:
            break
    return total / count if count > 0 else 0.0


# ------------------------------------------------------------------
# Main logic for NASA-15
# ------------------------------------------------------------------
def nasa_15(tbl: ezr.TBL) -> None:
    """Find a proposal with excessive effort and suggest a single change to reduce it."""
    effort_col = get_col_by_name(tbl, "Effort-")
    # Precompute predictions for all rows
    effort_preds: List[float] = []
    for i, row in enumerate(tbl.rows, start=1):
        others = [(j, r) for j, r in enumerate(tbl.rows, start=1) if j != i]
        pred = estimate_value_via_neighbors(tbl, row, others, effort_col)
        effort_preds.append(pred)

    # Pick row with largest predicted effort
    max_idx = max(range(len(effort_preds)), key=lambda k: effort_preds[k])
    max_row = tbl.rows[max_idx]
    before_effort = effort_preds[max_idx]

    # Adjustable columns: not goals, not ending with X, not Kloc+
    adjustable_cols = [
        c for c in tbl.cols.values()
        if not c.txt.endswith(("X", "+", "-")) and c.txt != "Kloc+"
    ]

    # Unique values for each column
    unique_vals: Dict[str, List[ATOM]] = {}
    for c in adjustable_cols:
        vals = sorted({r[c.at] for r in tbl.rows if r[c.at] != "?"})
        unique_vals[c.txt] = vals

    # Search for best single change
    best_change = None  # Dict[str, List[ATOM]]
    best_after_effort = before_effort
    for col in adjustable_cols:
        current = max_row[col.at]
        for new_val in unique_vals[col.txt]:
            if new_val == current:
                continue
            new_row = list(max_row)
            new_row[col.at] = new_val
            others = [(j, r) for j, r in enumerate(tbl.rows, start=1) if j != (max_idx + 1)]
            after_effort = estimate_value_via_neighbors(tbl, new_row, others, effort_col)
            if after_effort < best_after_effort:
                best_after_effort = after_effort
                best_change = {col.txt: [current, new_val]}

    # If no improvement found, keep empty changes
    if best_change is None:
        best_change = {}
        best_after_effort = before_effort

    # Output
    print("== NASA-15 ==")
    print(f"Project row: {max_idx + 1}")
    print(f"Predicted effort before: {before_effort:.3f}")
    print(f"Predicted effort after: {best_after_effort:.3f}")
    if best_change:
        for col, vals in best_change.items():
            print(f"Change: {col} from {vals[0]} to {vals[1]}")
    else:
        print("No improvement found with a single change.")
    print(
        "ANSWER NASA-15 "
        + json.dumps(
            {
                "project": max_idx + 1,
                "changes": best_change,
                "predicted_effort": {"before": round(before_effort, 3), "after": round(best_after_effort, 3)},
            },
            separators=(",", ":"),
        )
    )


# ------------------------------------------------------------------
# Main logic for NASA-16
# ------------------------------------------------------------------
def nasa_16(tbl: ezr.TBL) -> None:
    """Find a proposal with high predicted defects and suggest up to 3 changes."""
    effort_col = get_col_by_name(tbl, "Effort-")
    defects_col = get_col_by_name(tbl, "Defects-")

    # Precompute predictions for all rows
    effort_preds: List[float] = []
    defects_preds: List[float] = []
    for i, row in enumerate(tbl.rows, start=1):
        others = [(j, r) for j, r in enumerate(tbl.rows, start=1) if j != i]
        effort_preds.append(estimate_value_via_neighbors(tbl, row, others, effort_col))
        defects_preds.append(estimate_value_via_neighbors(tbl, row, others, defects_col))

    # Pick row with largest predicted defects
    max_idx = max(range(len(defects_preds)), key=lambda k: defects_preds[k])
    target_row = tbl.rows[max_idx]
    before_effort = effort_preds[max_idx]
    before_defects = defects_preds[max_idx]

    # Adjustable columns
    adjustable_cols = [
        c for c in tbl.cols.values()
        if not c.txt.endswith(("X", "+", "-")) and c.txt != "Kloc+"
    ]

    unique_vals: Dict[str, List[ATOM]] = {}
    for c in adjustable_cols:
        vals = sorted({r[c.at] for r in tbl.rows if r[c.at] != "?"})
        unique_vals[c.txt] = vals

    # Helper to evaluate a set of changes
    def evaluate_changes(changes: Dict[str, List[ATOM]]) -> Tuple[float, float]:
        new_row = list(target_row)
        for col_name, (old_val, new_val) in changes.items():
            new_row[tbl.cols_by_name[col_name].at] = new_val
        others = [(j, r) for j, r in enumerate(tbl.rows, start=1) if j != (max_idx + 1)]
        after_defects = estimate_value_via_neighbors(tbl, new_row, others, defects_col)
        after_effort = estimate_value_via_neighbors(tbl, new_row, others, effort_col)
        return after_defects, after_effort

    # Find best combination of up to 3 changes that improves defects
    best_changes: Dict[str, List[ATOM]] = {}
    best_after_defects = before_defects
    best_after_effort = before_effort
    improvement_found = False

    # Depth-first search from 1 to 3 changes
    for depth in range(1, 4):
        for cols_combo in itertools.combinations(adjustable_cols, depth):
            # Build lists of possible new values for each column
            value_lists = [unique_vals[c.txt] for c in cols_combo]
            for values_combo in itertools.product(*value_lists):
                # Skip combos that don't change anything
                changes = {}
                for col, new_val in zip(cols_combo, values_combo):
                    old_val = target_row[col.at]
                    if new_val == old_val:
                        break
                    changes[col.txt] = [old_val, new_val]
                else:
                    after_defects, after_effort = evaluate_changes(changes)
                    if after_defects < best_after_defects:
                        best_after_defects = after_defects
                        best_after_effort = after_effort
                        best_changes = changes
                        improvement_found = True
                        break
            if improvement_found:
                break
        if improvement_found:
            break

    # Output
    print("== NASA-16 ==")
    print(f"Project row: {max_idx + 1}")
    print(f"Predicted defects before: {before_defects:.3f}")
    print(f"Predicted defects after: {best_after_defects:.3f}")
    print(f"Predicted effort before: {before_effort:.3f}")
    print(f"Predicted effort after: {best_after_effort:.3f}")
    if best_changes:
        for col, vals in best_changes.items():
            print(f"Change: {col} from {vals[0]} to {vals[1]}")
    else:
        print("No improvement found with up to 3 changes.")
    print(
        "ANSWER NASA-16 "
        + json.dumps(
            {
                "project": max_idx + 1,
                "changes": best_changes,
                "predicted_defects": {"before": round(before_defects, 3), "after": round(best_after_defects, 3)},
                "predicted_effort": {"before": round(before_effort, 3), "after": round(best_after_effort, 3)},
            },
            separators=(",", ":"),
        )
    )


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv-file>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    tbl = read_table(csv_path)

    # Build helper mapping of name to column for quick access
    tbl.cols_by_name = {c.txt: c for c in tbl.cols.values()}

    nasa_15(tbl)
    print()
    nasa_16(tbl)


if __name__ == "__main__":
    main()
