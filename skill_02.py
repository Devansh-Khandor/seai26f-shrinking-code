#!/usr/bin/env python3
"""
skill_02.py - estimate effort and defects for NASA-93 data

Usage:  python skill_02.py <csv path>
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


def col_index(header, colname):
    """Return column index of colname."""
    return header.index(colname)


def split_rows(rows, n_divisible):
    """Return (history, arrivals, held_out)."""
    N = len(rows)
    arrivals_len = int(round(0.2 * N))
    history = rows[:N - arrivals_len]
    arrivals = rows[N - arrivals_len:]
    # held-out rows are those with 1‑based index divisible by n_divisible
    held_out = [row for idx, row in enumerate(rows, start=1) if idx % n_divisible == 0]
    return history, arrivals, held_out


def train_rows(history, budget):
    """Select at most `budget` rows from history that are not held‑out."""
    candidates = [row for idx, row in enumerate(history, start=1) if idx % 5 != 0]
    return candidates[:budget]


def leaf_stats(node, col_idx):
    """Return mean and standard deviation of a column in a leaf."""
    vals = [row[col_idx] for row in node.rows if row[col_idx] != "?"]
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        sd = math.sqrt(var)
    else:
        sd = 0.0
    return mean, sd


def tree_leaf(node, row):
    """Return the leaf node for a row."""
    while node.kids:
        node = node.kids[0 if node.go(row) else 1]
    return node


# ----------------------------------------------------------------------
# NASA-03: Effort estimation
# ----------------------------------------------------------------------
def nasa_03(header, rows):
    effort_col = "Effort-"
    effort_idx = col_index(header, effort_col)

    history, arrivals, held_out = split_rows(rows, 5)

    # Budget: min(30, number_of_rows // 5)
    budget = min(30, len(rows) // 5)

    training = train_rows(history, budget)

    # Build full table for the tree (header + all rows)
    tbl = ezr.Tbl([header] + rows)

    # Build tree on training rows
    node = ezr.tree(tbl, training)

    # Pre‑compute leaf statistics
    leaf_stat_map = {}
    for leaf in ezr.leafs(node):
        mean, _ = leaf_stats(leaf, effort_idx)
        leaf_stat_map[leaf] = mean

    # Predict for held‑out rows
    predictions = {}
    mae_total = 0.0
    mae_count = 0
    for idx, row in enumerate(rows, start=1):
        if idx % 5 == 0:  # held‑out
            leaf = tree_leaf(node, row)
            pred = leaf_stat_map.get(leaf, 0.0)
            predictions[str(idx)] = pred
            actual = row[effort_idx]
            if actual != "?":
                mae_total += abs(pred - actual)
                mae_count += 1

    mae = mae_total / mae_count if mae_count else 0.0

    # Print section
    print("== NASA-03 ==")
    # Print each prediction
    for r in sorted(predictions, key=int):
        print(f"Row {r}: Effort- ≈ {predictions[r]:.2f}")
    print(f"MAE: {mae:.3f}")

    # Machine‑checked answer
    answer = {"predictions": predictions, "mae": mae}
    print(f"ANSWER NASA-03 {json.dumps(answer, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# NASA-04: Defects estimation with uncertainty
# ----------------------------------------------------------------------
def nasa_04(header, rows):
    defects_col = "Defects-"
    defects_idx = col_index(header, defects_col)

    history, arrivals, held_out = split_rows(rows, 5)

    # Budget: min(30, number_of_rows // 5)
    budget = min(30, len(rows) // 5)

    training = train_rows(history, budget)

    # Build full table for the tree (header + all rows)
    tbl = ezr.Tbl([header] + rows)

    # Build tree on training rows
    node = ezr.tree(tbl, training)

    # Pre‑compute leaf statistics (mean and sd)
    leaf_stat_map = {}
    for leaf in ezr.leafs(node):
        mean, sd = leaf_stats(leaf, defects_idx)
        leaf_stat_map[leaf] = (mean, sd)

    # Predict for held‑out rows
    predictions = {}
    for idx, row in enumerate(rows, start=1):
        if idx % 5 == 0:  # held‑out
            leaf = tree_leaf(node, row)
            mean, sd = leaf_stat_map.get(leaf, (0.0, 0.0))
            predictions[str(idx)] = {
                "estimate": mean,
                "low": mean - sd,
                "high": mean + sd,
            }

    # Print section
    print("== NASA-04 ==")
    for r in sorted(predictions, key=int):
        p = predictions[r]
        print(f"Row {r}: estimate={p['estimate']:.2f} low={p['low']:.2f} high={p['high']:.2f}")
    # No MAE requested for this metric

    # Machine‑checked answer
    answer = {"predictions": predictions}
    print(f"ANSWER NASA-04 {json.dumps(answer, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    header, rows = read_csv(path)

    nasa_03(header, rows)
    nasa_04(header, rows)


if __name__ == "__main__":
    main()
