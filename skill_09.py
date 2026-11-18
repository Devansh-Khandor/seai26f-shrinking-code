#!/usr/bin/env python3
"""
Skill 09: Anomaly detection for Apache configurations.

APACHE-17 – Identify configurations whose recorded performance values are
unusually different from what their settings and comparable historical
configurations would lead us to expect.

APACHE-18 – Flag configurations whose combinations of settings differ
substantially from the most similar configurations in our historical
benchmark records, even when their measured performance is not obviously poor.
"""

import json
import sys
import math
from typing import List, Tuple, Dict

import ezr
from ezr import xdist

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def goal_col(tbl: ezr.Tbl) -> ezr.COL:
    """Return the goal column (ends with '-' or '+')."""
    for col in tbl.cols.values():
        if col.txt.endswith("-") or col.txt.endswith("+"):
            return col
    raise RuntimeError("No goal column found.")


def perf_val(row: ezr.ROW, gcol: ezr.COL) -> float | None:
    """Return numeric performance value or None if missing."""
    val = row[gcol.at]
    if val == "?":
        return None
    return float(val)


def neighbor_stats(
    row: ezr.ROW, tbl: ezr.Tbl, gcol: ezr.COL, k: int = 5
) -> Tuple[float | None, float | None, int]:
    """
    Return (mean, std, count) of the k nearest neighbours
    (excluding the row itself) based on xdist.
    """
    distances: List[Tuple[float, int]] = []
    for idx, other in enumerate(tbl.rows):
        if other is row:
            continue
        distances.append((xdist(tbl, row, other), idx))

    distances.sort(key=lambda d: d[0])
    neigh = []
    for d, idx in distances[:k]:
        val = perf_val(tbl.rows[idx], gcol)
        if val is not None:
            neigh.append(val)

    n = len(neigh)
    if n == 0:
        return None, None, 0
    mean = sum(neigh) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((v - mean) ** 2 for v in neigh) / (n - 1)
    std = math.sqrt(var)
    return mean, std, n


def nearest_distance(row: ezr.ROW, tbl: ezr.Tbl) -> float:
    """Return the minimum xdist to any other row (excluding itself)."""
    dists = [xdist(tbl, row, other) for other in tbl.rows if other is not row]
    return min(dists) if dists else 0.0


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows: List[ezr.ROW] = list(ezr.csv(csv_path))
    tbl = ezr.Tbl(rows)
    gcol = goal_col(tbl)
    goal_name = gcol.txt

    # --------------------------------------------------------------
    # APACHE-17
    # --------------------------------------------------------------
    print("== APACHE-17 ==")

    flagged_17: List[Dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        act = perf_val(row, gcol)
        if act is None:
            continue

        mean, std, n = neighbor_stats(row, tbl, gcol, k=5)

        # If we have no neighbours with a known goal, skip
        if mean is None:
            continue

        # Determine threshold: if std == 0, any deviation is flagged
        if std == 0:
            flag = abs(act - mean) > 1e-12
        else:
            flag = abs(act - mean) > 2 * std

        if flag:
            flagged_17.append(
                {
                    "row": idx,
                    "column": goal_name,
                    "actual": act,
                    "expected": round(mean, 4),
                }
            )
            if len(flagged_17) >= 20:
                break

    # At least three output lines
    if not flagged_17:
        print("No anomalous performance rows found.")
    else:
        for f in flagged_17:
            print(
                f"Row {f['row']}: actual={f['actual']:.2f} expected={f['expected']:.2f}"
            )

    print(
        "ANSWER APACHE-17",
        json.dumps({"flagged": flagged_17}, separators=(",", ":")),
    )

    # --------------------------------------------------------------
    # APACHE-18
    # --------------------------------------------------------------
    print("\n== APACHE-18 ==")

    scores: Dict[str, float] = {}
    for idx, row in enumerate(rows, start=1):
        # compute distance to nearest neighbour
        dist = nearest_distance(row, tbl)
        scores[str(idx)] = round(dist, 4)

    # sort rows by descending distance
    sorted_rows = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    flagged_18 = [int(r) for r, _ in sorted_rows[:20]]

    # at least three lines of output
    if not flagged_18:
        print("No unusual configurations found.")
    else:
        for row_num, dist in sorted_rows[:10]:
            print(f"Row {row_num}: unusualness score={dist:.4f}")

    print(
        "ANSWER APACHE-18",
        json.dumps({"flagged": flagged_18, "scores": scores}, separators=(",", ":")),
    )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
