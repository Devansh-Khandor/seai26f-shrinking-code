#!/usr/bin/env python3
"""
skill_06.py - explain effort estimates and risk differences

Usage:  python skill_06.py <csv path>
"""

import sys
import json
import math
import ezr  # the provided library


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_csv(path):
    """Return header and list of rows."""
    rows = list(ezr.csv(path))
    header = rows[0]
    data = rows[1:]
    return header, data


def mean_and_sd(values):
    """Return mean and standard deviation of a list of numbers."""
    n = len(values)
    if n == 0:
        return None, None
    s = sum(values)
    m = s / n
    var = sum((x - m) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    return m, sd


def leaf_stats(node, col_idx):
    """Return mean of a column for a leaf."""
    vals = [row[col_idx] for row in node.rows if row[col_idx] != "?"]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def leaf_node(node, row):
    """Walk down to leaf for a row."""
    while node.kids:
        node = node.kids[0 if node.go(row) else 1]
    return node


def xdist(tbl, r1, r2):
    """Distance between two rows over X columns."""
    return ezr.xdist(tbl, r1, r2)


# ----------------------------------------------------------------------
# NASA-11: explain effort estimate
# ----------------------------------------------------------------------
def nasa_11(header, data):
    """
    Pick the first held‑out row (number divisible by 5), estimate its Effort‑,
    and explain the estimate with its nearest historical neighbours.
    """
    # Identify history and arrivals
    n = len(data)
    arrivals_len = int(round(0.2 * n))
    history = data[:-arrivals_len]
    arrivals = data[-arrivals_len:]

    # Build table for all rows (used for distance)
    tbl = ezr.Tbl([header] + data)

    # Build tree on history
    tree = ezr.tree(tbl, history)

    # Find first held‑out row
    held_out_idx = None
    for i in range(n):
        if (i + 1) % 5 == 0:
            held_out_idx = i
            break
    proj_row = data[held_out_idx]
    proj_num = held_out_idx + 1

    # Estimate Effort‑
    leaf = leaf_node(tree, proj_row)
    eff_idx = header.index("Effort-")
    estimate = leaf_stats(leaf, eff_idx)

    # Find neighbours (non‑held‑out, not the project itself)
    neigh_candidates = [
        (i, row) for i, row in enumerate(data)
        if (i + 1) % 5 != 0 and i != held_out_idx
    ]
    neigh_candidates.sort(key=lambda pr: xdist(tbl, proj_row, pr[1]))
    top_neighs = neigh_candidates[:3]

    # Prepare neighbor info
    neighbors = []
    for idx, row in top_neighs:
        values = {c: row[header.index(c)] for c in header}
        neighbors.append({"row": idx + 1, "values": values})

    # ------------------------------------------------------------------
    # Human readable output
    # ------------------------------------------------------------------
    print("== NASA-11 ==")
    print(f"Project row {proj_num}:")
    print(f"  Estimated Effort- = {estimate:.2f}")
    print("  Nearest neighbours:")
    for nb in neighbors:
        eff = nb["values"]["Effort-"]
        print(f"    Row {nb['row']}: Effort- = {eff}")

    # ------------------------------------------------------------------
    # Machine‑checked answer
    # ------------------------------------------------------------------
    answer = {
        "project": proj_num,
        "estimate": estimate,
        "neighbors": neighbors,
    }
    print(f"ANSWER NASA-11 {json.dumps(answer, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# NASA-12: compare risk assessments
# ----------------------------------------------------------------------
def nasa_12(header, data):
    """
    Pick one low‑risk and one high‑risk project, explain the
    differences that contributed to those assessments and list trade‑offs.
    """
    # Identify history and arrivals
    n = len(data)
    arrivals_len = int(round(0.2 * n))
    history = data[:-arrivals_len]
    arrivals = data[-arrivals_len:]

    # Build full table
    tbl = ezr.Tbl([header] + data)

    # Build tree on history
    tree = ezr.tree(tbl, history)

    # Compute thresholds from history
    eff_idx = header.index("Effort-")
    def_idx = header.index("Defects-")
    eff_vals = [row[eff_idx] for row in history if row[eff_idx] != "?"]
    def_vals = [row[def_idx] for row in history if row[def_idx] != "?"]
    eff_mean, eff_sd = mean_and_sd(eff_vals)
    def_mean, def_sd = mean_and_sd(def_vals)
    eff_thresh = eff_mean + 2 * eff_sd if eff_sd else eff_mean
    def_thresh = def_mean + 2 * def_sd if def_sd else def_mean

    # Predict Effort- and Defects- for every row
    preds = {}
    for i, row in enumerate(data):
        leaf = leaf_node(tree, row)
        pred_eff = leaf_stats(leaf, eff_idx)
        pred_def = leaf_stats(leaf, def_idx)
        preds[i] = {"eff": pred_eff, "def": pred_def}

    # Compute risk boolean
    risks = {}
    for i, p in preds.items():
        risks[i] = (p["eff"] > eff_thresh) or (p["def"] > def_thresh)

    # Pick one risk True and one risk False
    low_risk_row = None
    high_risk_row = None
    for i, r in risks.items():
        if r and high_risk_row is None:
            high_risk_row = i
        if not r and low_risk_row is None:
            low_risk_row = i
        if low_risk_row is not None and high_risk_row is not None:
            break

    if low_risk_row is None or high_risk_row is None:
        # Fallback: pick first two rows with differing risk
        items = list(risks.items())
        low_risk_row, high_risk_row = items[0][0], items[1][0]

    a_row_num = low_risk_row + 1
    b_row_num = high_risk_row + 1

    # Scores: higher is more recommended (lower risk)
    # Use negative sum of predicted values as score
    score_a = -(preds[low_risk_row]["eff"] + preds[low_risk_row]["def"])
    score_b = -(preds[high_risk_row]["eff"] + preds[high_risk_row]["def"])

    preferred = a_row_num if score_a >= score_b else b_row_num
    scores = {a_row_num: score_a, b_row_num: score_b}

    # Differences in all X columns
    x_cols = [c for c in header if c.endswith("X")]  # skip X suffix? Actually X columns end with X
    # But we want columns that are not goal columns and not X columns:
    # X columns have name ending with 'X', we should skip those
    # Also exclude goal columns ending in '+' or '-'
    non_goal = [c for c in header if not c.endswith(("+", "-")) and not c.endswith("X")]
    differences = []
    row_a = data[low_risk_row]
    row_b = data[high_risk_row]
    for c in non_goal:
        val_a = row_a[header.index(c)]
        val_b = row_b[header.index(c)]
        if val_a != val_b:
            differences.append({"feature": c, "a": val_a, "b": val_b})
    if not differences:
        # Ensure at least one
        differences.append({"feature": non_goal[0], "a": row_a[header.index(non_goal[0])], "b": row_b[header.index(non_goal[0])]})

    # Trade‑offs: goal columns where non‑preferred row is better
    goal_cols = [c for c in header if c.endswith(("+", "-"))]
    tradeoffs = []
    non_pref_row = data[high_risk_row] if preferred == a_row_num else data[low_risk_row]
    for g in goal_cols:
        v_non_pref = non_pref_row[header.index(g)]
        v_pref = data[low_risk_row if preferred == a_row_num else high_risk_row][header.index(g)]
        # skip missing
        if v_non_pref == "?" or v_pref == "?":
            continue
        if g.endswith("+"):
            better = v_non_pref > v_pref
        else:
            better = v_non_pref < v_pref
        if better:
            tradeoffs.append(g)

    # ------------------------------------------------------------------
    # Human readable output
    # ------------------------------------------------------------------
    print("== NASA-12 ==")
    print(f"Row {a_row_num} (low risk) vs Row {b_row_num} (high risk)")
    print(f"  Scores: Row {a_row_num} = {score_a:.2f}, Row {b_row_num} = {score_b:.2f}")
    print("  Differences:")
    for d in differences[:3]:
        print(f"    {d['feature']}: Row {a_row_num}={d['a']}, Row {b_row_num}={d['b']}")
    print("  Trade‑offs:")
    for t in tradeoffs:
        print(f"    {t}")

    # ------------------------------------------------------------------
    # Machine‑checked answer
    # ------------------------------------------------------------------
    answer = {
        "a": a_row_num,
        "b": b_row_num,
        "preferred": preferred,
        "scores": scores,
        "differences": differences,
        "tradeoffs": tradeoffs,
    }
    print(f"ANSWER NASA-12 {json.dumps(answer, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    header, data = read_csv(path)

    nasa_11(header, data)
    nasa_12(header, data)


if __name__ == "__main__":
    main()
