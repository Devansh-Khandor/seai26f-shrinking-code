#!/usr/bin/env python3
"""
Skill 09 – NASA anomaly detection

This script reads a NASA project CSV file and prints two analyses:
  NASA-17 – Flag historical projects whose combinations of project characteristics
             are unusually different from the rest of our records.
  NASA-18 – Identify project records whose reported effort or defect counts differ
             substantially from what comparable projects would lead us to expect.

Both sections include a machine‑checked JSON line.  All calculations are derived
from the CSV file supplied on the command line; no hard‑coded values are used.

The program uses only the data supplied on the command line and never
hard‑codes row numbers or values.
"""

import json
import sys

import ezr


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _all_distances(tbl: ezr.TBL):
    """
    Compute the pairwise x‑space distance between every row in the table.
    Returns a list of lists: distances[i][j] is the distance between
    row i and row j.  Rows are 0‑based indices.
    """
    n = len(tbl.rows)
    dists = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = ezr.xdist(tbl, tbl.rows[i], tbl.rows[j])
            dists[i][j] = dists[j][i] = d
    return dists


def _mean_distances(dists):
    """
    From a full distance matrix, compute the mean distance of each row
    to all other rows.  Returns a dict mapping 1‑based row number
    to the mean distance.
    """
    n = len(dists)
    means = {}
    for i in range(n):
        # exclude self
        total = sum(dists[i][j] for j in range(n) if j != i)
        mean = total / (n - 1) if n > 1 else 0.0
        means[i + 1] = mean
    return means


def _nearest_neighbors(dists, idx, k=5):
    """
    Return the indices of the k nearest neighbors of row `idx`
    (0‑based).  The list is sorted by increasing distance.
    """
    n = len(dists)
    # gather (dist, j) for all j != idx
    dist_list = [(dists[idx][j], j) for j in range(n) if j != idx]
    dist_list.sort(key=lambda x: x[0])
    return [j for _, j in dist_list[:k]]


def _expected_value(neighbor_indices, tbl, col):
    """
    Compute the mean value of column `col` (an ezr.NUM or ezr.Sym)
    over the neighbor rows.  Missing values ('?') are ignored.
    If no neighbor has a valid value, returns None.
    """
    vals = []
    for j in neighbor_indices:
        val = tbl.rows[j][col.at]
        if val != "?":
            vals.append(val)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _compute_diff(tbl, dists, col):
    """
    For each row, compute the difference between its actual value for
    `col` (Effort- or Defects-) and the expected value derived from
    its nearest neighbors.  Returns a list of tuples:
    (row_number, diff) where diff is actual - expected.
    Rows with missing actual or no valid neighbors are omitted.
    """
    n = len(tbl.rows)
    col_obj = tbl.cols[col]
    diffs = []
    for i in range(n):
        actual = tbl.rows[i][col_obj.at]
        if actual == "?":
            continue
        neighbors = _nearest_neighbors(dists, i, k=5)
        expected = _expected_value(neighbors, tbl, col_obj)
        if expected is None:
            continue
        diff = actual - expected
        diffs.append((i + 1, diff))
    return diffs


# ----------------------------------------------------------------------
# NASA-17: Unusual combinations of input characteristics
# ----------------------------------------------------------------------
def nasa_17(tbl: ezr.TBL):
    # Compute all pairwise distances once
    dists = _all_distances(tbl)
    mean_dists = _mean_distances(dists)

    # Sort rows by mean distance descending
    sorted_rows = sorted(mean_dists.items(), key=lambda x: x[1], reverse=True)
    flagged = [row for row, _ in sorted_rows[:10]]

    # Prepare the score dictionary
    scores = {str(row): round(val, 5) for row, val in sorted_rows[:10]}

    # Output
    print("== NASA-17 ==")
    print(f"Flagged rows: {flagged}")
    print(f"Top unusualness scores (row: score):")
    for row in flagged:
        print(f"  {row}: {scores[str(row)]}")
    ans = {"flagged": flagged, "scores": scores}
    print(f"ANSWER NASA-17 {json.dumps(ans, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# NASA-18: Unexpected Effort or Defect counts
# ----------------------------------------------------------------------
def nasa_18(tbl: ezr.TBL):
    dists = _all_distances(tbl)

    # Compute diffs for Effort- and Defects-
    effort_diffs = _compute_diff(tbl, dists, "Effort-")
    defect_diffs = _compute_diff(tbl, dists, "Defects-")

    # Combine and sort by absolute difference
    all_diffs = []
    for row, diff in effort_diffs:
        all_diffs.append({"row": row, "column": "Effort-", "actual": tbl.rows[row - 1][tbl.cols["Effort-"].at], "expected": round(tbl.rows[row - 1][tbl.cols["Effort-"].at] - diff, 5), "diff": abs(diff)})
    for row, diff in defect_diffs:
        all_diffs.append({"row": row, "column": "Defects-", "actual": tbl.rows[row - 1][tbl.cols["Defects-"].at], "expected": round(tbl.rows[row - 1][tbl.cols["Defects-"].at] - diff, 5), "diff": abs(diff)})

    # Sort and keep top 10
    all_diffs.sort(key=lambda x: x["diff"], reverse=True)
    flagged = all_diffs[:10]

    # Prepare JSON output (omit internal diff field)
    json_flagged = [{"row": f["row"], "column": f["column"], "actual": f["actual"], "expected": f["expected"]} for f in flagged]

    # Output
    print("== NASA-18 ==")
    print("Flagged records with largest unexpected Effort- or Defects-:")
    for f in flagged:
        print(f"  row={f['row']} column={f['column']} actual={f['actual']} expected={f['expected']}")
    ans = {"flagged": json_flagged}
    print(f"ANSWER NASA-18 {json.dumps(ans, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    tbl = ezr.Tbl(ezr.csv(csv_path))
    nasa_17(tbl)
    nasa_18(tbl)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_09.py <csv_path>")
    main(sys.argv[1])
