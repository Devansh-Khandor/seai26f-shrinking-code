#!/usr/bin/env python3
"""
skill_04.py - assess new proposals and keep updated summaries

Usage:  python skill_04.py <csv path>
"""

import sys
import json
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

def compute_stats(values):
    """Return n, mean, min, max for a list of numeric values."""
    n = len(values)
    if n == 0:
        return n, None, None, None
    s = sum(values)
    mean = s / n
    mn = min(values)
    mx = max(values)
    return n, mean, mn, mx

def standard_dev(values, mean):
    """Return sample standard deviation, or 0 if insufficient data."""
    n = len(values)
    if n <= 1:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)

def leaf_stats(node, col_idx):
    """Return mean of a column for this leaf."""
    vals = [row[col_idx] for row in node.rows if row[col_idx] != "?"]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)

def leaf_node(node, row):
    """Walk down to leaf for a row."""
    while node.kids:
        node = node.kids[0 if node.go(row) else 1]
    return node

# ----------------------------------------------------------------------
# NASA-07: assess arrivals with alerts
# ----------------------------------------------------------------------
def nasa_07(header, data):
    nrows = len(data)
    arrivals_len = int(round(0.2 * nrows))
    history = data[: nrows - arrivals_len]
    arrivals = data[nrows - arrivals_len :]

    eff_idx = header.index("Effort-")
    def_idx = header.index("Defects-")

    # Historical stats (only history rows)
    hist_eff_vals = [float(r[eff_idx]) for r in history if r[eff_idx] != "?"]
    hist_def_vals = [float(r[def_idx]) for r in history if r[def_idx] != "?"]

    _, hist_eff_mean, _, _ = compute_stats(hist_eff_vals)
    _, hist_def_mean, _, _ = compute_stats(hist_def_vals)

    hist_eff_sd = standard_dev(hist_eff_vals, hist_eff_mean) if hist_eff_vals else 0.0
    hist_def_sd = standard_dev(hist_def_vals, hist_def_mean) if hist_def_vals else 0.0

    eff_threshold = hist_eff_mean + 2 * hist_eff_sd
    def_threshold = hist_def_mean + 2 * hist_def_sd

    # Map original row numbers to data index
    all_rows_with_idx = list(enumerate(data, start=1))

    assessments = []

    # Process each arrival
    for i, (row_idx, row) in enumerate(all_rows_with_idx[-arrivals_len:]):
        # Training set = history + earlier arrivals
        training_rows = history + [r for _, r in all_rows_with_idx[-arrivals_len:-arrivals_len + i]]
        tbl = ezr.Tbl([header] + training_rows)
        node = ezr.tree(tbl, training_rows)

        # Compute leaf statistics for effort and defects
        leaf_map = {}
        for leaf in ezr.leafs(node):
            e_mean = leaf_stats(leaf, eff_idx)
            d_mean = leaf_stats(leaf, def_idx)
            leaf_map[leaf] = (e_mean, d_mean)

        leaf = leaf_node(node, row)
        pred_eff, pred_def = leaf_map.get(leaf, (0.0, 0.0))

        alert = (pred_eff > eff_threshold) or (pred_def > def_threshold)

        assessments.append({
            "row": row_idx,
            "predicted_effort": pred_eff,
            "predicted_defects": pred_def,
            "alert": alert
        })

    # Output
    print("== NASA-07 ==")
    for a in assessments:
        print(f"Row {a['row']}: Effort- ≈ {a['predicted_effort']:.3f} "
              f"Defects- ≈ {a['predicted_defects']:.3f} "
              f"Alert: {a['alert']}")
    print(f"ANSWER NASA-07 {json.dumps({'assessments': assessments}, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# NASA-08: updated summaries after arrivals
# ----------------------------------------------------------------------
def nasa_08(header, data):
    nrows = len(data)
    arrivals_len = int(round(0.2 * nrows))
    history = data[: nrows - arrivals_len]
    arrivals = data[nrows - arrivals_len :]

    # Map original row numbers
    all_rows_with_idx = list(enumerate(data, start=1))
    history_with_idx = all_rows_with_idx[: nrows - arrivals_len]
    arrivals_with_idx = all_rows_with_idx[nrows - arrivals_len :]

    goal_cols = [c for c in header if c.endswith(('+', '-'))]
    goal_indices = {c: header.index(c) for c in goal_cols}

    # checkpoints to report
    checkpoints_m = [m for m in [0, 9, 19] if m <= arrivals_len]
    checkpoints = []

    # Pre‑compute normalized ranges per checkpoint
    for m in checkpoints_m:
        considered = history_with_idx + arrivals_with_idx[:m]
        idx_to_row = {idx: row for idx, row in considered}

        # Stats per goal column
        summary = {}
        for col in goal_cols:
            col_idx = goal_indices[col]
            vals = [float(r[col_idx]) for _, r in considered if r[col_idx] != "?"]
            n, mean, mn, mx = compute_stats(vals)
            summary[col] = {"n": n, "mean": mean, "min": mn, "max": mx}

        # Shortlist: best 5 rows by composite score
        # Compute min and max for each goal over considered rows
        norm_bounds = {}
        for col in goal_cols:
            col_idx = goal_indices[col]
            vals = [float(r[col_idx]) for _, r in considered if r[col_idx] != "?"]
            if vals:
                mn = min(vals); mx = max(vals)
                norm_bounds[col] = (mn, mx)
            else:
                norm_bounds[col] = (0, 1)  # dummy

        comps = []
        for idx, row in considered:
            # skip if any goal missing
            if any(row[goal_indices[g]] == "?" for g in goal_cols):
                continue
            comps_val = 0.0
            for g in goal_cols:
                v = float(row[goal_indices[g]])
                mn, mx = norm_bounds[g]
                denom = mx - mn
                if denom == 0:
                    norm = 0.0
                else:
                    if g.endswith("+"):
                        norm = (v - mn) / denom
                    else:  # minimize
                        norm = (mx - v) / denom
                comps_val += norm
            comps.append((comps_val, idx))
        comps.sort(reverse=True)
        shortlist = [idx for _, idx in comps[:5]]

        checkpoints.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })

    # Output
    print("== NASA-08 ==")
    for cp in checkpoints:
        print(f"Checkpoint after {cp['after']} arrivals:")
        for col, stats in cp['summary'].items():
            print(f"  {col}: n={stats['n']} mean={stats['mean']:.3f} "
                  f"min={stats['min']:.3f} max={stats['max']:.3f}")
        print(f"  shortlist: {cp['shortlist']}")
    print(f"ANSWER NASA-08 {json.dumps({'checkpoints': checkpoints}, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    header, data = read_csv(path)

    nasa_07(header, data)
    nasa_08(header, data)

if __name__ == "__main__":
    main()
