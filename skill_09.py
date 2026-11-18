#!/usr/bin/env python3
"""
skill_09.py: Flag unusual historical projects and outlying effort/defects.
Usage: python skill_09.py <csv-file>
"""

import json
import sys
from math import sqrt
from typing import List, Dict, Tuple

import ezr


# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
def read_table(path: str) -> ezr.TBL:
    """Read CSV into an ezr table (header included)."""
    rows = list(ezr.csv(path))
    return ezr.Tbl(rows)


def split_history_arrivals(rows: List[ezr.ROW]) -> Tuple[List[ezr.ROW], List[ezr.ROW]]:
    """Split rows into history and arrivals (last 20% rounded)."""
    n = len(rows)
    arrivals_count = round(n * 0.2)
    arrivals_start = n - arrivals_count
    return rows[:arrivals_start], rows[arrivals_start:]


def input_columns(tbl: ezr.TBL) -> List[ezr.COL]:
    """Columns that are not X, +, or -."""
    return [c for c in tbl.cols.values() if not c.txt.endswith(("X", "+", "-"))]


# ------------------------------------------------------------------
# NASA-17: Unusual historical projects
# ------------------------------------------------------------------
def compute_unusual_scores(tbl: ezr.TBL, historic_rows: List[ezr.ROW]) -> Dict[int, float]:
    """Return a score for each historic row (row number 1‑based)."""
    # Map 1‑based index to row
    indices = [i + 1 for i in range(len(historic_rows))]
    # Pre‑compute distances between all historic rows
    scores: Dict[int, float] = {}
    for i_idx, i_row in enumerate(historic_rows):
        i_num = indices[i_idx]
        min_dist = float("inf")
        for j_idx, j_row in enumerate(historic_rows):
            j_num = indices[j_idx]
            if i_num == j_num:
                continue
            d = ezr.xdist(tbl, i_row, j_row)
            if d < min_dist:
                min_dist = d
        # If no other row (single historic row), set distance 0
        scores[i_num] = min_dist if min_dist != float("inf") else 0.0
    return scores


# ------------------------------------------------------------------
# NASA-18: Outlying effort / defects
# ------------------------------------------------------------------
def estimate_value_via_neighbors(
    tbl: ezr.TBL,
    target_row: ezr.ROW,
    others: List[Tuple[int, ezr.ROW]],
    goal_col: ezr.COL,
    min_neighbors: int = 3,
) -> float:
    """Estimate value of a goal column using nearest neighbors."""
    distances = []
    for idx, row in others:
        d = ezr.xdist(tbl, target_row, row)
        distances.append((d, idx, row))
    distances.sort(key=lambda x: x[0])

    total = 0.0
    count = 0
    for _, _, row in distances:
        val = row[goal_col.at]
        if val == "?":
            continue
        total += float(val)
        count += 1
        if count >= min_neighbors:
            break
    return total / count if count > 0 else 0.0


def compute_outliers(
    tbl: ezr.TBL,
    all_rows: List[ezr.ROW],
    goal_col: ezr.COL,
) -> List[Dict]:
    """Return list of dictionaries with row, column, actual, expected, diff."""
    results: List[Dict] = []
    all_indices = [i + 1 for i in range(len(all_rows))]
    for i, row in enumerate(all_rows):
        idx = all_indices[i]
        actual = row[goal_col.at]
        if actual == "?":
            continue
        others = [(j, r) for j, r in enumerate(all_rows, start=1) if j != idx]
        expected = estimate_value_via_neighbors(tbl, row, others, goal_col)
        diff = abs(float(actual) - expected)
        results.append(
            {"row": idx, "column": goal_col.txt, "actual": float(actual), "expected": expected, "diff": diff}
        )
    return results


# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv-file>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    tbl = read_table(path)
    all_rows = tbl.rows  # all data rows

    # Split history / arrivals
    history_rows, arrivals_rows = split_history_arrivals(all_rows)
    # NASA-17
    scores = compute_unusual_scores(tbl, history_rows)
    # Flag top 10 by highest score (more unusual)
    flagged17 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    flagged_nums17 = [num for num, _ in flagged17]
    score_map17 = {str(num): round(score, 4) for num, score in flagged17}

    print("== NASA-17 ==")
    print("Unusual historic rows (top 10):")
    for num, score in flagged17:
        print(f"  Row {num}: score {score:.4f}")
    print("Answering JSON:")
    print("ANSWER NASA-17", json.dumps({"flagged": flagged_nums17, "scores": score_map17}))

    # NASA-18
    effort_col = next(c for c in tbl.cols.values() if c.txt == "Effort-")
    defects_col = next(c for c in tbl.cols.values() if c.txt == "Defects-")

    effort_outliers = compute_outliers(tbl, all_rows, effort_col)
    defects_outliers = compute_outliers(tbl, all_rows, defects_col)

    # Compute thresholds
    def compute_threshold(outliers: List[Dict]) -> float:
        diffs = [o["diff"] for o in outliers]
        if not diffs:
            return 0.0
        mean = sum(diffs) / len(diffs)
        sd = sqrt(sum((d - mean) ** 2 for d in diffs) / len(diffs))
        return mean + 2 * sd if sd > 0 else mean

    thr_eff = compute_threshold(effort_outliers)
    thr_def = compute_threshold(defects_outliers)

    flagged18_list = []
    for o in effort_outliers:
        if o["diff"] > thr_eff:
            flagged18_list.append(o)
    for o in defects_outliers:
        if o["diff"] > thr_def:
            flagged18_list.append(o)

    # Sort by diff descending
    flagged18_list.sort(key=lambda o: o["diff"], reverse=True)
    flagged18 = flagged18_list[:10]  # up to 10

    print("\n== NASA-18 ==")
    if flagged18:
        print("Rows with outlying effort/defects:")
        for o in flagged18:
            print(f"  Row {o['row']}, column {o['column']}, actual {o['actual']:.4f}, expected {o['expected']:.4f}")
    else:
        print("  No outliers found.")
    print("Answering JSON:")
    # Prepare JSON
    json_flagged = [
        {"row": o["row"], "column": o["column"], "actual": round(o["actual"], 4), "expected": round(o["expected"], 4)}
        for o in flagged18
    ]
    print("ANSWER NASA-18", json.dumps({"flagged": json_flagged}))


if __name__ == "__main__":
    main()
