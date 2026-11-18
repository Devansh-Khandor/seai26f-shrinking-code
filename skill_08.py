#!/usr/bin/env python3
"""
skill_08.py - predict effort / defect impact of planned changes

Usage:  python skill_08.py <csv path>
"""

import json
import sys
import math
import ezr

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_csv(path):
    """Return header and list of rows."""
    rows = list(ezr.csv(path))
    header = rows[0]
    data = rows[1:]
    return header, data

def get_adjustable_columns(header, rows):
    """Return list of column names that can be changed.

    Adjustable columns are all non-goal columns that are not Kloc+ and not ending with X.
    """
    goal_cols = [c for c in header if c.endswith(('+', '-'))]
    adjust = []
    for c in header:
        if c in goal_cols or c == 'Kloc+' or c.endswith('X'):
            continue
        adjust.append(c)
    return adjust

def get_unique_values(rows, col):
    """Return set of non-? values that appear in a column."""
    idx = rows[0].index(col)
    return {row[idx] for row in rows if row[idx] != '?'}

def leaf_mean(node, col_idx):
    """Return mean value of column in a leaf node."""
    vals = [row[col_idx] for row in node.rows if row[col_idx] != '?']
    if not vals:
        return 0.0
    return sum(vals) / len(vals)

def predict_leaf(node, row):
    """Return the leaf node for a row."""
    while node.kids:
        node = node.kids[0 if node.go(row) else 1]
    return node

def held_out_indices(n):
    """Return 1‑based indices of rows that are held‑out (divisible by 5)."""
    return [i+1 for i in range(n) if (i+1) % 5 == 0]

def arrival_len(n):
    """Return number of arrivals given total n."""
    return int(round(0.2 * n))

def history_and_arrivals(rows):
    """Return (history, arrivals) split."""
    n = len(rows)
    a_len = arrival_len(n)
    return rows[:n-a_len], rows[n-a_len:]

def build_tree(history_rows, header):
    """Build a decision tree from history rows only."""
    tbl = ezr.Tbl([header] + history_rows)
    return ezr.tree(tbl, history_rows)

def compute_predictions(row, tree, header):
    """Return predicted Effort- and Defects- for a given row."""
    leaf = predict_leaf(tree, row)
    eff_idx = header.index("Effort-")
    def_idx = header.index("Defects-")
    pred_eff = leaf_mean(leaf, eff_idx)
    pred_def = leaf_mean(leaf, def_idx)
    return pred_eff, pred_def

def find_best_effort_change(row, tree, header, adjust_cols, all_rows):
    """Identify changes that reduce predicted effort the most."""
    orig_eff, _ = compute_predictions(row, tree, header)
    best_changes = []
    for col in adjust_cols:
        idx = header.index(col)
        if row[idx] == '?':
            continue
        cur_val = row[idx]
        for new_val in get_unique_values(all_rows, col):
            if new_val == cur_val:
                continue
            new_row = list(row)
            new_row[idx] = new_val
            new_eff, _ = compute_predictions(new_row, tree, header)
            delta = orig_eff - new_eff
            if delta > 0:  # improvement
                best_changes.append((delta, col, cur_val, new_val, new_eff))
    # keep up to 3 most improving changes
    best_changes.sort(key=lambda x: (-x[0], x[1]))
    return best_changes[:3], orig_eff

def find_best_defect_change(row, tree, header, adjust_cols, all_rows):
    """Identify changes that reduce predicted defects with minimal effort increase."""
    orig_eff, orig_def = compute_predictions(row, tree, header)
    best_changes = []
    for col in adjust_cols:
        idx = header.index(col)
        if row[idx] == '?':
            continue
        cur_val = row[idx]
        for new_val in get_unique_values(all_rows, col):
            if new_val == cur_val:
                continue
            new_row = list(row)
            new_row[idx] = new_val
            new_eff, new_def = compute_predictions(new_row, tree, header)
            delta_def = orig_def - new_def
            delta_eff = new_eff - orig_eff
            if delta_def > 0:  # improvement
                # score: prioritize defect reduction, penalize effort increase
                score = delta_def - delta_eff  # simple linear trade‑off
                best_changes.append((score, delta_def, delta_eff,
                                     col, cur_val, new_val, new_eff, new_def))
    # keep up to 3 best trade‑off changes
    best_changes.sort(key=lambda x: (-x[1], x[2], x[0]))
    return best_changes[:3], orig_eff, orig_def

# ----------------------------------------------------------------------
# NASA-15
# ----------------------------------------------------------------------
def nasa_15(header, data):
    """Identify changes to reduce excessive predicted effort."""
    history, arrivals = history_and_arrivals(data)
    tree = build_tree(history, header)

    # Identify held‑out rows
    h_indices = held_out_indices(len(data))
    # Find the held‑out row with the largest predicted effort
    max_eff = -1e30
    target_row = None
    target_num = None
    for idx in h_indices:
        row = data[idx-1]
        eff, _ = compute_predictions(row, tree, header)
        if eff > max_eff:
            max_eff = eff
            target_row = row
            target_num = idx

    adjust_cols = get_adjustable_columns(header, data)
    best_changes, before_eff = find_best_effort_change(target_row, tree,
                                                       header, adjust_cols, data)

    # Apply changes to compute after effort
    new_row = list(target_row)
    changes_dict = {}
    for delta, col, cur, new, new_eff in best_changes:
        idx_col = header.index(col)
        new_row[idx_col] = new
        changes_dict[col] = [cur, new]
    after_eff = leaf_mean(predict_leaf(tree, new_row), header.index("Effort-"))

    # Human‑readable output
    print("== NASA-15 ==")
    print(f"Project row {target_num}: predicted effort before {before_eff:.2f} after {after_eff:.2f}")
    print("Suggested changes:")
    for col, (cur, new) in changes_dict.items():
        print(f"  {col}: {cur} -> {new}")
    print(f"Predicted effort change: {before_eff - after_eff:.2f}")

    # Machine‑checked answer
    answer = {
        "project": target_num,
        "changes": changes_dict,
        "predicted_effort": {"before": round(before_eff, 3), "after": round(after_eff, 3)}
    }
    print(f"ANSWER NASA-15 {json.dumps(answer, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# NASA-16
# ----------------------------------------------------------------------
def nasa_16(header, data):
    """Identify changes to reduce defects while noting effort trade‑offs."""
    history, arrivals = history_and_arrivals(data)
    tree = build_tree(history, header)

    # Identify held‑out rows with high predicted defects
    h_indices = held_out_indices(len(data))
    max_def = -1e30
    target_row = None
    target_num = None
    for idx in h_indices:
        row = data[idx-1]
        _, def_val = compute_predictions(row, tree, header)
        if def_val > max_def:
            max_def = def_val
            target_row = row
            target_num = idx

    adjust_cols = get_adjustable_columns(header, data)
    best_changes, orig_eff, orig_def = find_best_defect_change(target_row, tree,
                                                                header, adjust_cols, data)

    # Apply changes
    new_row = list(target_row)
    changes_dict = {}
    for score, delta_def, delta_eff, col, cur, new, new_eff, new_def in best_changes:
        idx_col = header.index(col)
        new_row[idx_col] = new
        changes_dict[col] = [cur, new]
    after_def = leaf_mean(predict_leaf(tree, new_row), header.index("Defects-"))
    after_eff = leaf_mean(predict_leaf(tree, new_row), header.index("Effort-"))

    # Human‑readable output
    print("== NASA-16 ==")
    print(f"Project row {target_num}: predicted defects before {orig_def:.2f} after {after_def:.2f}")
    print(f"Predicted effort before {orig_eff:.2f} after {after_eff:.2f}")
    print("Suggested changes:")
    for col, (cur, new) in changes_dict.items():
        print(f"  {col}: {cur} -> {new}")
    print(f"Defects reduction: {orig_def - after_def:.2f}")
    print(f"Effort change: {after_eff - orig_eff:.2f}")

    # Machine‑checked answer
    answer = {
        "project": target_num,
        "changes": changes_dict,
        "predicted_defects": {"before": round(orig_def, 3), "after": round(after_def, 3)},
        "predicted_effort": {"before": round(orig_eff, 3), "after": round(after_eff, 3)}
    }
    print(f"ANSWER NASA-16 {json.dumps(answer, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv path>")
        sys.exit(1)
    path = sys.argv[1]
    header, data = read_csv(path)
    nasa_15(header, data)
    nasa_16(header, data)

if __name__ == "__main__":
    main()
