#!/usr/bin/env python3
"""
skill_06.py: Explain effort estimates and risk differences for NASA data.
Usage: python skill_06.py <csv-file>
"""

import json
import sys
from typing import List, Dict, Tuple

import ezr


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def read_table(path: str) -> ezr.TBL:
    """Read CSV into an ezr table (header included)."""
    rows = list(ezr.csv(path))
    return ezr.Tbl(rows)


def data_rows(tbl: ezr.TBL) -> List[ezr.ROW]:
    """Return just the data rows (excluding header)."""
    return tbl.rows[1:]


def held_out_indices(rows: List[ezr.ROW]) -> List[int]:
    """Indices (1‑based) of rows whose number is divisible by 5."""
    return [i for i, _ in enumerate(rows, start=1) if i % 5 == 0]


def get_goal_cols(tbl: ezr.TBL) -> List[ezr.COL]:
    """Goal columns (ending in + or -)."""
    return [c for c in tbl.cols.values() if c.txt.endswith(("+", "-"))]


def get_effort_col(tbl: ezr.TBL) -> ezr.COL:
    return next(c for c in tbl.cols.values() if c.txt == "Effort-")


def get_defects_col(tbl: ezr.TBL) -> ezr.COL:
    return next(c for c in tbl.cols.values() if c.txt == "Defects-")


def estimate_effort_via_neighbors(
    tbl: ezr.TBL,
    target_row: ezr.ROW,
    non_held_rows: List[Tuple[int, ezr.ROW]],
    min_neighbors: int = 3,
) -> Tuple[float, List[Dict]]:
    """Estimate Effort- for target_row using the nearest non‑held rows with known Effort-."""
    effort_col = get_effort_col(tbl)
    distances = []
    for idx, row in non_held_rows:
        d = ezr.xdist(tbl, target_row, row)
        distances.append((d, idx, row))
    distances.sort(key=lambda x: x[0])

    neighbors = []
    total = 0.0
    count = 0
    for _, idx, row in distances:
        val = row[effort_col.at]
        if val == "?":
            continue
        neighbors.append({"row": idx, "values": {c.txt: row[c.at] for c in tbl.cols.values()}})
        total += float(val)
        count += 1
        if count >= min_neighbors:
            break
    estimate = total / count if count > 0 else 0.0
    return estimate, neighbors


def risk_score(row: ezr.ROW, effort_col: ezr.COL, defects_col: ezr.COL) -> float | None:
    """Compute a simple risk score (Effort- + Defects-) ignoring missing values."""
    eff = row[effort_col.at]
    defc = row[defects_col.at]
    if eff == "?" or defc == "?":
        return None
    return float(eff) + float(defc)


def find_two_different_risk_rows(
    rows_with_indices: List[Tuple[int, ezr.ROW]],
    effort_col: ezr.COL,
    defects_col: ezr.COL,
) -> Tuple[Tuple[int, ezr.ROW], Tuple[int, ezr.ROW]]:
    """Return first two rows with different risk scores."""
    scored = []
    for idx, row in rows_with_indices:
        score = risk_score(row, effort_col, defects_col)
        if score is not None:
            scored.append((idx, row, score))
    # Find pair
    for i in range(len(scored)):
        for j in range(i + 1, len(scored)):
            if scored[i][2] != scored[j][2]:
                return (scored[i][0], scored[i][1]), (scored[j][0], scored[j][1])
    # Fallback: return first two rows
    return (scored[0][0], scored[0][1]), (scored[1][0], scored[1][1])


def differences_between(a: ezr.ROW, b: ezr.ROW, tbl: ezr.TBL) -> List[Dict]:
    """List of differing feature columns."""
    diffs = []
    for col in tbl.cols.values():
        if col.txt.endswith("X"):  # skip X columns
            continue
        val_a = a[col.at]
        val_b = b[col.at]
        if val_a != val_b:
            diffs.append({"feature": col.txt, "a": val_a, "b": val_b})
    return diffs


def tradeoffs_between(
    a: ezr.ROW,
    b: ezr.ROW,
    tbl: ezr.TBL,
    preferred_idx: int,
) -> List[str]:
    """Goal columns where non‑preferred row is better."""
    better = []
    for col in tbl.cols.values():
        if not col.txt.endswith(("+", "-")):
            continue
        val_a = a[col.at]
        val_b = b[col.at]
        if val_a == "?" or val_b == "?":
            continue
        better_a = (float(val_a) > float(val_b)) if col.txt.endswith("+") else (float(val_a) < float(val_b))
        # Non‑preferred row
        if preferred_idx == 1:
            # a preferred, so non‑preferred is b
            if not better_a:
                better.append(col.txt)
        else:
            if better_a:
                better.append(col.txt)
    return better


# ------------------------------------------------------------------
# Main skill logic
# ------------------------------------------------------------------
def nasa_11(tbl: ezr.TBL) -> None:
    """Explain Effort- estimate for one held‑out row."""
    rows = data_rows(tbl)
    indices = held_out_indices(rows)
    if not indices:
        print("No held‑out rows found.")
        return
    # Pick the first held‑out row
    proj_idx = indices[0]
    target_row = rows[proj_idx - 1]

    # Build list of non‑held rows with indices
    non_held = [(i, r) for i, r in enumerate(rows, start=1) if i not in indices]

    estimate, neighbors = estimate_effort_via_neighbors(tbl, target_row, non_held)

    print("== NASA-11 ==")
    print(f"Project row {proj_idx} predicted Effort- = {estimate:.2f}")
    print("Neighbors:")
    for n in neighbors[:3]:
        row_num = n["row"]
        vals = n["values"]
        print(f"  Row {row_num}: Effort-={vals['Effort-']}")
    print(
        f"ANSWER NASA-11 "
        + json.dumps(
            {
                "project": proj_idx,
                "estimate": round(estimate, 2),
                "neighbors": [
                    {"row": n["row"], "values": n["values"]} for n in neighbors[:3]
                ],
            },
            separators=(",", ":"),
        )
    )


def nasa_12(tbl: ezr.TBL) -> None:
    """Explain risk differences between two proposals."""
    rows = data_rows(tbl)
    indices = held_out_indices(rows)
    if len(indices) < 2:
        print("Not enough held‑out rows for comparison.")
        return

    data_with_indices = [(i, r) for i, r in enumerate(rows, start=1) if i in indices]
    effort_col = get_effort_col(tbl)
    defects_col = get_defects_col(tbl)

    a_pair, b_pair = find_two_different_risk_rows(
        data_with_indices, effort_col, defects_col
    )
    a_idx, a_row = a_pair
    b_idx, b_row = b_pair

    score_a = risk_score(a_row, effort_col, defects_col)
    score_b = risk_score(b_row, effort_col, defects_col)
    # Higher score => higher risk; we prefer lower risk => higher recommendation score
    pref = a_idx if score_a < score_b else b_idx
    pref_row = a_row if pref == a_idx else b_row

    diffs = differences_between(a_row, b_row, tbl)
    tradeoffs = tradeoffs_between(a_row, b_row, tbl, preferred_idx=1 if pref == a_idx else 2)

    print("== NASA-12 ==")
    print(f"Row {a_idx} risk score: {score_a}")
    print(f"Row {b_idx} risk score: {score_b}")
    print(f"Preferred row: {pref}")
    print("Differences:")
    for d in diffs[:3]:
        print(f"  {d['feature']}: {d['a']} vs {d['b']}")
    print(f"Tradeoffs (goal columns where non‑preferred is better): {tradeoffs}")

    print(
        f"ANSWER NASA-12 "
        + json.dumps(
            {
                "a": a_idx,
                "b": b_idx,
                "preferred": pref,
                "scores": {
                    str(a_idx): round(score_a, 2),
                    str(b_idx): round(score_b, 2),
                },
                "differences": diffs[:3],
                "tradeoffs": tradeoffs,
            },
            separators=(",", ":"),
        )
    )


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv-file>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    tbl = read_table(path)

    nasa_11(tbl)
    nasa_12(tbl)


if __name__ == "__main__":
    main()
