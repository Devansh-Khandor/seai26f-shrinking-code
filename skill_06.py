#!/usr/bin/env python3
"""
Skill 06 – NASA explanation and comparison

This script reads a NASA project CSV file and prints two analyses:

  NASA-11 – When the system estimates the effort required for a proposed
    project, explain the estimate using relevant historical projects
    and their recorded characteristics.

  NASA-12 – When two project proposals receive different risk assessments,
    explain the main differences that contributed to those assessments
    and identify the important trade‑offs.

The output follows the required format and includes machine‑checked
JSON lines for each section.  All calculations are derived from the
CSV file supplied on the command line; no hard‑coded values are used.
"""

import json
import sys
import math
from collections import defaultdict

import ezr


# ----------------------------------------------------------------------
# Helpers – tree statistics and prediction (same as in skill_04)
# ----------------------------------------------------------------------
def _augment_stats(node, eff_col):
    """Recursively compute and store average Effort- value in each leaf."""
    if node.kids:
        for kid in node.kids:
            _augment_stats(kid, eff_col)
    else:
        effs = [r[eff_col.at] for r in node.rows if r[eff_col.at] != "?"]
        node.avg_effort = sum(effs) / len(effs) if effs else None


def _predict_effort(node, row):
    leaf = ezr.leaf(node, row)
    return leaf.avg_effort if leaf.avg_effort is not None else 0.0


# ----------------------------------------------------------------------
# Utility to fetch a dict of values for a row (excluding X columns)
# ----------------------------------------------------------------------
def _row_values(tbl, row):
    return {tbl.names[i]: row[i]
            for i, v in enumerate(row)
            if not tbl.names[i].endswith("X")}


# ----------------------------------------------------------------------
# NASA-11 – explanation for one held‑out project
# ----------------------------------------------------------------------
def _nasa_11(tbl):
    # Identify held‑out rows (index divisible by 5)
    held_indices = [i for i, _ in enumerate(tbl.rows, start=1) if i % 5 == 0]
    if not held_indices:
        print("== NASA-11 ==")
        print("No held‑out rows found.")
        print(f"ANSWER NASA-11 {{}}")
        return

    # Use the first held‑out row as the target
    target_row = tbl.rows[held_indices[0] - 1]
    target_idx = held_indices[0]

    # Build tree on all non‑held‑out rows
    history_rows = [r for i, r in enumerate(tbl.rows, start=1) if i not in held_indices]
    lab_tbl = ezr.clone(tbl, history_rows)
    tree_root = ezr.tree(lab_tbl, lab_tbl.rows)
    _augment_stats(tree_root, lab_tbl.cols["Effort-"])

    estimate = _predict_effort(tree_root, target_row)

    # Find 3 nearest neighbors in X space
    distances = [(idx, r,
                  ezr.xdist(tbl, target_row, r))
                 for idx, r in enumerate(tbl.rows, start=1)
                 if idx not in held_indices]
    distances.sort(key=lambda x: x[2])
    neighbors = []
    for idx, r, _ in distances[:3]:
        neighbors.append({
            "row": idx,
            "values": _row_values(tbl, r)
        })

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print("== NASA-11 ==")
    print(f"Target row: {target_idx}")
    print(f"Estimated Effort-: {estimate:.4f}")
    print(f"Nearest historical neighbors (3):")
    for n in neighbors:
        print(f"  row {n['row']} values: {n['values']}")

    ans = {
        "project": target_idx,
        "estimate": estimate,
        "neighbors": neighbors
    }
    print(f"ANSWER NASA-11 {json.dumps(ans, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# NASA-12 – compare two held‑out projects
# ----------------------------------------------------------------------
def _nasa_12(tbl):
    # Identify held‑out rows
    held_indices = [i for i, _ in enumerate(tbl.rows, start=1) if i % 5 == 0]
    if len(held_indices) < 2:
        print("== NASA-12 ==")
        print("Not enough held‑out rows.")
        print(f"ANSWER NASA-12 {{}}")
        return

    # Build tree on all non‑held‑out rows (same as in NASA-11)
    history_rows = [r for i, r in enumerate(tbl.rows, start=1) if i not in held_indices]
    lab_tbl = ezr.clone(tbl, history_rows)
    tree_root = ezr.tree(lab_tbl, lab_tbl.rows)
    _augment_stats(tree_root, lab_tbl.cols["Effort-"])

    # Take first two held‑out rows
    idx_a, idx_b = held_indices[0], held_indices[1]
    row_a, row_b = tbl.rows[idx_a - 1], tbl.rows[idx_b - 1]

    # Scores (higher = lower effort)
    score_a = -_predict_effort(tree_root, row_a)
    score_b = -_predict_effort(tree_root, row_b)

    preferred = "a" if score_a >= score_b else "b"

    # Differences (first few differing features)
    differences = []
    for i, col in enumerate(tbl.names):
        if col.endswith("X"):
            continue
        va, vb = row_a[i], row_b[i]
        if va != vb:
            differences.append({"feature": col,
                                "a": va,
                                "b": vb})
            if len(differences) >= 5:  # keep it manageable
                break

    # Trade‑offs: goal columns where the non‑preferred row is better
    tradeoffs = []
    for col_name, col_obj in lab_tbl.cols.items():
        if not (col_name.endswith("+") or col_name.endswith("-")):
            continue
        val_a, val_b = row_a[col_obj.at], row_b[col_obj.at]
        if val_a == "?" or val_b == "?" or val_a == val_b:
            continue
        # Determine which direction is better
        if col_name.endswith("+"):
            better = val_a > val_b
        else:
            better = val_a < val_b
        if (preferred == "a" and not better) or (preferred == "b" and better):
            tradeoffs.append(col_name)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print("== NASA-12 ==")
    print(f"Comparing rows {idx_a} (a) and {idx_b} (b)")
    print(f"Scores: a={score_a:.4f}, b={score_b:.4f}")
    print(f"Preferred: {preferred}")
    print(f"Differences (first 5): {differences}")
    print(f"Trade‑offs where non‑preferred is better: {tradeoffs}")

    ans = {
        "a": idx_a,
        "b": idx_b,
        "preferred": preferred,
        "scores": {"a": score_a, "b": score_b},
        "differences": differences,
        "tradeoffs": tradeoffs
    }
    print(f"ANSWER NASA-12 {json.dumps(ans, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    tbl = ezr.Tbl(ezr.csv(csv_path))
    _nasa_11(tbl)
    _nasa_12(tbl)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_06.py <csv_path>")
    main(sys.argv[1])
