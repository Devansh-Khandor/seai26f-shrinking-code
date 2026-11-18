#!/usr/bin/env python3
"""
Skill 09: Apache configuration anomaly detection

This program implements two analyses:

* **APACHE-17** – Identify configurations whose recorded performance
  values are unusually different from what similar historical configurations
  would predict.

* **APACHE-18** – Flag configurations whose combinations of settings are
  substantially different from the most similar historical configurations,
  regardless of their measured performance.

The program reads a CSV file supplied on the command line and prints a
human‑readable section for each analysis followed by a machine‑checked
JSON line.  Rows are identified by their 1‑based index in the file
(header is not counted).

The logic uses a simple distance metric over input columns: numeric
values are z‑normalized (using column mean and SD), symbols are compared
with equality.  Missing values contribute a maximal distance of 1.0.

For **APACHE‑17** the expected performance of a row is the mean
performance of its five nearest neighbours (excluding rows with missing
performance).  Rows with the largest absolute deviations between actual
and expected performance are flagged (up to 20).

For **APACHE‑18** the “unusualness score’’ of a row is the distance to its
nearest neighbour.  The 20 rows with the largest such distances are
reported, together with their scores.

"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# CSV parsing
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[Dict[str, object]]]:
    """Read CSV and return header and list of row dictionaries."""
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows: List[Dict[str, object]] = []
        for row in rdr:
            d: Dict[str, object] = {}
            for h, val in zip(header, row):
                if val == "?":
                    d[h] = None
                else:
                    try:
                        if "." in val or "e" in val.lower():
                            d[h] = float(val)
                        else:
                            d[h] = int(val)
                    except ValueError:
                        d[h] = val
            rows.append(d)
    return header, rows

# ----------------------------------------------------------------------
# Helpers: column types, statistics
# ----------------------------------------------------------------------
def classify_cols(header: List[str], rows: List[Dict[str, object]]):
    goal_cols = [c for c in header if c.endswith("+") or c.endswith("-")]
    input_cols = [c for c in header if not c.endswith("X") and c not in goal_cols]
    # Detect numeric input columns
    numeric_cols = set()
    for col in input_cols:
        for r in rows:
            v = r[col]
            if v is None:
                continue
            if isinstance(v, (int, float)):
                numeric_cols.add(col)
                break
    # Compute mean & SD for numeric columns
    stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        vals = [r[col] for r in rows if r[col] is not None]
        if vals:
            mu = sum(vals) / len(vals)
            var = sum((x - mu) ** 2 for x in vals) / max(1, len(vals) - 1)
            sd = math.sqrt(var)
            stats[col] = {"mu": mu, "sd": sd if sd > 0 else 1e-32}
        else:
            stats[col] = {"mu": 0.0, "sd": 1e-32}
    return goal_cols, input_cols, numeric_cols, stats

# ----------------------------------------------------------------------
# Distance between two rows over input columns
# ----------------------------------------------------------------------
def row_distance(
    a: Dict[str, object],
    b: Dict[str, object],
    input_cols: List[str],
    numeric_cols: set,
    stats: Dict[str, Dict[str, float]],
) -> float:
    sumsq = 0.0
    for col in input_cols:
        va = a[col]
        vb = b[col]
        if col in numeric_cols:
            if va is None or vb is None:
                diff = 1.0
            else:
                mu = stats[col]["mu"]
                sd = stats[col]["sd"]
                na = (va - mu) / sd
                nb = (vb - mu) / sd
                diff = na - nb
            sumsq += diff * diff
        else:  # symbol
            if va is None or vb is None:
                diff = 1.0
            else:
                diff = 0.0 if va == vb else 1.0
            sumsq += diff * diff
    return math.sqrt(sumsq)

# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    header, rows = read_csv(csv_path)

    goal_cols, input_cols, numeric_cols, stats = classify_cols(header, rows)

    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)

    goal_col = goal_cols[0]  # use first goal column

    n = len(rows)

    # Precompute distances: for each row, a list of (dist, idx) to all other rows
    all_dist = []
    for i, r_i in enumerate(rows):
        neigh = []
        for j, r_j in enumerate(rows):
            if i == j:
                continue
            d = row_distance(r_i, r_j, input_cols, numeric_cols, stats)
            neigh.append((d, j))
        neigh.sort(key=lambda x: x[0])
        all_dist.append(neigh)

    # ----------------------------------------------------------------------
    # APACHE-17
    # ----------------------------------------------------------------------
    flagged_17 = []
    for i, r in enumerate(rows):
        actual = r.get(goal_col)
        if actual is None:
            continue
        # find up to 5 nearest neighbors with known performance
        preds = []
        for d, j in all_dist[i]:
            val = rows[j].get(goal_col)
            if val is not None:
                preds.append(val)
                if len(preds) >= 5:
                    break
        if not preds:
            continue
        expected = sum(preds) / len(preds)
        diff = actual - expected
        flagged_17.append((abs(diff), i + 1, actual, expected, diff))

    flagged_17.sort(key=lambda x: x[0], reverse=True)
    flagged_17 = flagged_17[:20]

    print("== APACHE-17 ==")
    print("Rows with largest performance deviations from expected:")
    for absdiff, row_no, actual, expected, diff in flagged_17:
        print(
            f"Row {row_no}: actual={actual:.2f} expected={expected:.2f} diff={diff:.2f}"
        )
    # Machine‑checked answer
    flagged_json = [
        {
            "row": row_no,
            "column": goal_col,
            "actual": actual,
            "expected": expected,
        }
        for _, row_no, actual, expected, _ in flagged_17
    ]
    print(f"ANSWER APACHE-17 {json.dumps({'flagged': flagged_json}, separators=(',', ':'))}")

    # ----------------------------------------------------------------------
    # APACHE-18
    # ----------------------------------------------------------------------
    flagged_18 = []
    for i, neigh in enumerate(all_dist):
        if neigh:
            dist, _ = neigh[0]
            flagged_18.append((dist, i + 1))
    flagged_18.sort(key=lambda x: x[0], reverse=True)
    flagged_18 = flagged_18[:20]

    print("\n== APACHE-18 ==")
    print("Rows whose setting combinations are most unusual:")
    for dist, row_no in flagged_18:
        print(f"Row {row_no}: distance={dist:.3f}")
    # Machine‑checked answer
    flagged_rows = [row_no for _, row_no in flagged_18]
    scores = {str(row_no): round(dist, 6) for dist, row_no in flagged_18}
    print(f"ANSWER APACHE-18 {json.dumps({'flagged': flagged_rows, 'scores': scores}, separators=(',', ':'))}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
