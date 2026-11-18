#!/usr/bin/env python3
"""
skill_07.py: Identify promising project characteristic combinations and recommend evaluation order.
Usage: python skill_07.py <csv-file>
"""

import json
import sys
from typing import List, Dict, Tuple

import ezr


# ------------------------------------------------------------------
# Helper for NASA-13
# ------------------------------------------------------------------
def build_rule_from_leaf(tbl: ezr.TBL, leaf: ezr.NODE) -> Tuple[Dict, int]:
    """
    Build a rule that describes the rows in the best leaf and count
    how many rows in the whole table satisfy it.
    """
    # Path to leaf
    path: List[ezr.NODE] = []

    def find_path(node: ezr.NODE, target: ezr.NODE, cur: List[ezr.NODE]) -> bool:
        if node is target:
            return True
        for kid in node.kids:
            if find_path(kid, target, cur):
                cur.append(node)
                return True
        return False

    find_path(leaf, leaf, path)
    path_nodes = list(reversed(path))  # root → leaf
    selected_nodes = path_nodes[:3]

    # Map column names to objects for quick lookup
    col_map = {c.txt: c for c in tbl.cols.values()}

    rule: Dict[str, object] = {}
    for node in selected_nodes:
        col = col_map[node.at]
        vals = [row[node.at] for row in leaf.rows if row[node.at] != "?"]
        if not vals:
            continue
        if isinstance(col, ezr.Sym):
            # most common symbol
            val = max(set(vals), key=vals.count)
            rule[col.txt] = val
        else:  # numeric
            low = min(vals)
            high = max(vals)
            rule[col.txt] = [low, high]

    # Count how many rows in the full table satisfy the rule
    def row_matches(r: ezr.ROW) -> bool:
        for colname, cond in rule.items():
            col = col_map[colname]
            v = r[col.at]
            if v == "?":
                return False
            if isinstance(cond, list):
                if not (cond[0] <= v <= cond[1]):
                    return False
            else:
                if v != cond:
                    return False
        return True

    matching = sum(1 for r in tbl.rows if row_matches(r))
    return rule, matching


# ------------------------------------------------------------------
# Helper for NASA-14
# ------------------------------------------------------------------
def recommend_order(tbl: ezr.TBL) -> List[int]:
    """
    Return an order of row numbers (1‑based) to evaluate first,
    limited to the available budget.
    """
    n_rows = len(tbl.rows)
    budget = min(30, n_rows // 5)
    remaining = set(range(n_rows))
    evaluated: List[int] = []
    order: List[int] = []

    while len(order) < budget and remaining:
        if not evaluated:
            # First row (lowest index)
            next_idx = min(remaining)
        else:
            # Build a table containing only evaluated rows
            eval_rows = [tbl.rows[i] for i in evaluated]
            sub_tbl = ezr.Tbl([tbl.names] + eval_rows)
            ranker_fn = ezr.ranker(sub_tbl)
            # Rank remaining rows
            ranked = [(i, ranker_fn(tbl.rows[i])) for i in remaining]
            # Choose row with best (lowest) rank value
            next_idx = min(ranked, key=lambda x: x[1])[0]

        order.append(next_idx + 1)  # convert to 1‑based
        evaluated.append(next_idx)
        remaining.remove(next_idx)

    return order


# ------------------------------------------------------------------
# Main skill logic
# ------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv-file>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = list(ezr.csv(csv_path))
    tbl = ezr.Tbl(rows)

    # ------------------------------------------------------------------
    # NASA-13
    # ------------------------------------------------------------------
    tree_root = ezr.tree(tbl, tbl.rows)
    leafs_list = ezr.leafs(tree_root)
    best_leaf = min(leafs_list, key=lambda l: l.mu)
    rule, matching = build_rule_from_leaf(tbl, best_leaf)

    print("== NASA-13 ==")
    print("Rule conditions:")
    for col, cond in rule.items():
        if isinstance(cond, list):
            print(f"  {col}: [{cond[0]}, {cond[1]}]")
        else:
            print(f"  {col}: {cond}")
    print(f"Matching rows: {matching}")

    print("ANSWER NASA-13", json.dumps({"rule": rule, "matching": matching}))

    # ------------------------------------------------------------------
    # NASA-14
    # ------------------------------------------------------------------
    order = recommend_order(tbl)

    print("\n== NASA-14 ==")
    # Show first few rows of the order for illustration
    print("Evaluation order (row numbers):", order[:10])
    print("First 3 rows:", order[:3])

    print("ANSWER NASA-14", json.dumps({"order": order}))


if __name__ == "__main__":
    main()
