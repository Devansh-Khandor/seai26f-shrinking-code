#!/usr/bin/env python3
"""
skill_09.py - identify anomalous historical projects and
             flag projects whose effort or defect counts are suspect

Usage:  python skill_09.py <csv path>
"""

import json
import sys
import math
import ezr

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def read_csv(path):
    """Return header and list of rows."""
    rows = list(ezr.csv(path))
    header = rows[0]
    data = rows[1:]
    return header, data

def get_input_columns(header):
    """Return input column names (exclude goal columns and X suffix)."""
    return [c for c in header if not c.endswith(('+', '-')) and not c.endswith('X')]

def get_goal_indices(header):
    return {c: header.index(c) for c in header if c.endswith(('+', '-'))}

def leaf_mean(node, col_idx):
    """Mean of a column in the leaf."""
    vals = [row[col_idx] for row in node.rows if row[col_idx] != "?"]
    return sum(vals) / len(vals) if vals else 0.0

def leaf_node(tree, row):
    """Return leaf node for a row."""
    # ezr.leaf expects node and row
    return ezr.leaf(tree, row)

def arrival_len(n):
    """Return number of arrivals given total n."""
    return int(round(0.2 * n))

def history_and_arrivals(rows):
    """Return (history, arrivals) split."""
    n = len(rows)
    a_len = arrival_len(n)
    return rows[:n-a_len], rows[n-a_len:]

# ----------------------------------------------------------------------
# NASA-17: flag unusual input combinations
# ----------------------------------------------------------------------
def nasa_17(header, data):
    """
    Flag historical projects whose combinations of project characteristics
    are unusually different from the rest of our records.
    """
    tbl = ezr.Tbl([header] + data)

    # Distance from centroid in the space of all X columns
    mids_tbl = ezr.mids(tbl)
    distances = []
    for i, row in enumerate(data, start=1):
        d = ezr.xdist(tbl, row, mids_tbl)
        distances.append((i, d))

    # Take the 10 rows with largest distance
    distances.sort(key=lambda x: x[1], reverse=True)
    flagged_rows = [i for i, _ in distances[:10]]

    # Prepare scores dictionary
    scores = {str(i): round(d, 4) for i, d in distances if i in flagged_rows}

    # Output
    print("== NASA-17 ==")
    print(f"Flagged rows: {flagged_rows}")
    print(f"Distances: {scores}")
    # Machine‑checked answer
    answer = {"flagged": flagged_rows, "scores": scores}
    print(f"ANSWER NASA-17 {json.dumps(answer, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# NASA-18: flag projects with anomalous effort or defects
# ----------------------------------------------------------------------
def nasa_18(header, data):
    """
    Identify project records whose reported effort or defect counts differ
    substantially from what comparable projects would lead us to expect.
    """
    # Split history & arrivals
    history, arrivals = history_and_arrivals(data)
    # Build tree on history only
    tree = ezr.tree(ezr.Tbl([header] + history), history)

    # Indices
    idx_eff = header.index("Effort-")
    idx_def = header.index("Defects-")

    entries = []

    # Compute prediction for each row
    for i, row in enumerate(data, start=1):
        if row[idx_eff] == "?" and row[idx_def] == "?":
            continue

        leaf = leaf_node(tree, row)

        pred_eff = leaf_mean(leaf, idx_eff)
        pred_def = leaf_mean(leaf, idx_def)

        # Effort
        if row[idx_eff] != "?":
            act_eff = float(row[idx_eff])
            diff = abs(act_eff - pred_eff)
            entries.append({"row": i, "column": "Effort-",
                            "actual": act_eff, "expected": round(pred_eff, 4),
                            "diff": diff})

        # Defects
        if row[idx_def] != "?":
            act_def = float(row[idx_def])
            diff = abs(act_def - pred_def)
            entries.append({"row": i, "column": "Defects-",
                            "actual": act_def, "expected": round(pred_def, 4),
                            "diff": diff})

    # Take top 10 entries with largest absolute difference
    entries.sort(key=lambda e: e["diff"], reverse=True)
    flagged = entries[:10]
    # Remove auxiliary diff field before output
    for e in flagged:
        del e["diff"]

    # Output
    print("== NASA-18 ==")
    print(f"Flagged entries: {flagged}")
    # Machine‑checked answer
    answer = {"flagged": flagged}
    print(f"ANSWER NASA-18 {json.dumps(answer, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    header, data = read_csv(path)

    nasa_17(header, data)
    nasa_18(header, data)

if __name__ == "__main__":
    main()
