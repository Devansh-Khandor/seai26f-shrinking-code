#!/usr/bin/env python3
"""
skill_09.py: Unusual vehicle flagging and outlier detection for auto93 dataset.
Usage: python skill_09.py <csv_path>
"""
import json
import sys
import ezr
from typing import Dict, List, Tuple

# Reuse helpers from skill_01
from skill_01 import read_csv, build_table  # type: ignore

# ------------------------------------------------------------------
# Utility: get column by name
def get_col_by_name(tbl: ezr.TBL, name: str):
    for at, col in tbl.cols.items():
        if col.txt == name:
            return col
    raise KeyError(f"Column {name} not found")

# ------------------------------------------------------------------
# CAR-17: Flag unusual input combinations

def compute_input_stats(tbl: ezr.TBL) -> Dict[int, Tuple[float, float]]:
    """Return mean and std for each input column."""
    stats: Dict[int, Tuple[float, float]] = {}
    for at, col in tbl.cols.items():
        if col.goal != 0:
            continue
        vals = [r[at] for r in tbl.rows if r[at] != "?"]
        n = len(vals)
        if n == 0:
            stats[at] = (0.0, 1.0)
            continue
        mu = sum(vals) / n
        if n > 1:
            var = sum((v - mu) ** 2 for v in vals) / (n - 1)
            sd = var ** 0.5
            if sd == 0:
                sd = 1.0
        else:
            sd = 1.0
        stats[at] = (mu, sd)
    return stats

def unusualness_score(row: List, stats: Dict[int, Tuple[float, float]]) -> float:
    """Sum of absolute z‑scores for the input columns."""
    score = 0.0
    for at, (mu, sd) in stats.items():
        v = row[at]
        if v == "?":
            continue
        z = (v - mu) / sd if sd else 0.0
        score += abs(z)
    return score

def analyze_car_17(tbl: ezr.TBL) -> Tuple[List[int], Dict[int, float]]:
    stats = compute_input_stats(tbl)
    scores = {}
    for i, r in enumerate(tbl.rows):
        scores[i + 1] = unusualness_score(r, stats)
    # sort descending
    sorted_rows = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    flagged = [r for r, _ in sorted_rows[:40]]
    flagged_scores = {r: round(scores[r], 4) for r, _ in sorted_rows[:40]}
    return flagged, flagged_scores

# ------------------------------------------------------------------
# CAR-18: Outlier detection on Mpg+ and Lbs-

def find_neighbors(tbl: ezr.TBL, idx: int, k: int = 5) -> List[Tuple[float, int, List]]:
    """Return k nearest neighbors (distance, index, row)."""
    target_row = tbl.rows[idx]
    dlist = []
    for j, r in enumerate(tbl.rows):
        if j == idx:
            continue
        d = ezr.xdist(tbl, target_row, r)
        if d == 0:
            continue  # skip identical rows
        dlist.append((d, j, r))
    dlist.sort(key=lambda x: x[0])
    return dlist[:k]

def expected_value(neighbors: List[Tuple[float, int, List]], col: ezr.COL):
    vals = [r[col.at] for _, _, r in neighbors if r[col.at] != "?"]
    if not vals:
        return None
    return sum(vals) / len(vals)

def analyze_car_18(tbl: ezr.TBL) -> List[Dict]:
    input_cols = [col for col in tbl.cols.values() if col.goal == 0]
    target_cols = [get_col_by_name(tbl, name) for name in ("Mpg+", "Lbs-")]
    flagged: List[Dict] = []

    for i, r in enumerate(tbl.rows):
        neighbors = find_neighbors(tbl, i, k=5)
        for col in target_cols:
            actual = r[col.at]
            if actual == "?":
                continue
            expected = expected_value(neighbors, col)
            if expected is None:
                continue
            diff = abs(actual - expected)
            flagged.append({
                "row": i + 1,
                "column": col.txt,
                "actual": actual,
                "expected": round(expected, 4),
                "diff": diff
            })

    # sort by difference descending
    flagged.sort(key=lambda x: x["diff"], reverse=True)
    # keep up to 40 entries
    flagged = flagged[:40]
    # strip diff field for output
    for f in flagged:
        f.pop("diff", None)
    return flagged

# ------------------------------------------------------------------
# Main driver

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    header, rows = read_csv(path)
    tbl = build_table(header, rows)

    # CAR-17 ----------------------------------------------------
    flagged_17, scores_17 = analyze_car_17(tbl)
    print("== CAR-17 ==")
    print(f"Flagged {len(flagged_17)} rows with unusual input combinations.")
    print(f"Top 5 flagged rows: {flagged_17[:5]}")
    print("Scores of flagged rows (top 5):")
    for r in flagged_17[:5]:
        print(f"  Row {r}: {scores_17[r]}")
    print(f"ANSWER CAR-17 {json.dumps({'flagged': flagged_17, 'scores': scores_17})}")

    # CAR-18 ----------------------------------------------------
    flagged_18 = analyze_car_18(tbl)
    print("\n== CAR-18 ==")
    print(f"Flagged {len(flagged_18)} row/column pairs with large deviations.")
    print("Top 5 flagged pairs:")
    for f in flagged_18[:5]:
        print(f"  Row {f['row']} column {f['column']} actual={f['actual']} expected={f['expected']}")
    # JSON answer
    print(f"ANSWER CAR-18 {json.dumps({'flagged': flagged_18})}")

if __name__ == "__main__":
    main()
