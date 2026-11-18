#!/usr/bin/env python3
"""
skill_07.py: Recommendations for vehicle evaluation and selection.
Usage: python skill_07.py <csv_path>
"""

import json
import sys

import ezr


def read_csv(path: str):
    """Read CSV and return header and rows as atoms."""
    with open(path, encoding="utf-8-sig") as f:
        header = next(f).strip().split(",")
        rows = []
        for line in f:
            if line.strip():
                rows.append([ezr.atom(s) for s in line.split(",")])
    return header, rows


def build_table(header, rows):
    """Create an ezr table from header and rows."""
    return ezr.Tbl([header] + rows)


def top_n_by_ranker(tbl, n):
    """Return the top n row indices (1‑based) sorted by the default tree ranker."""
    rank = ezr.ranker(tbl)
    indices = sorted(range(len(tbl.rows)), key=lambda i: rank(tbl.rows[i]), reverse=True)
    return [i + 1 for i in indices[:n]]


def best_row_by_ydist(tbl, indices):
    """Return the row number (1‑based) with minimal ydist among given indices."""
    best = None
    best_ydist = None
    for idx in indices:
        row = tbl.rows[idx - 1]
        d = ezr.ydist(tbl, row)
        if best is None or d < best_ydist or (d == best_ydist and idx < best):
            best, best_ydist = idx, d
    return best


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    header, rows = read_csv(path)
    tbl = build_table(header, rows)

    # Determine budget: at most 30 or fewer rows
    budget = min(30, len(rows))

    # ---- CAR-13 ----------------------------------------------------
    # Evaluate the first `budget` rows according to the default ranker
    eval_order = top_n_by_ranker(tbl, budget)
    best = best_row_by_ydist(tbl, eval_order)

    print("== CAR-13 ==")
    print(f"Evaluated rows (budget={budget}): {eval_order}")
    print(f"Best evaluated row: {best}")
    # show the goals of the best row for transparency
    best_row = tbl.rows[best - 1]
    goals = {c.txt: best_row[c.at] for c in tbl.cols.values() if c.goal != 0}
    print(f"Best row goals: {goals}")
    print(
        f"ANSWER CAR-13 {json.dumps({'evaluated': eval_order, 'best': best})}"
    )

    # ---- CAR-14 ----------------------------------------------------
    # Recommend full evaluation order: top rows by ranker, up to the budget
    order = top_n_by_ranker(tbl, budget)

    print("\n== CAR-14 ==")
    print(f"Recommended evaluation order (first {budget} rows): {order}")
    # Show first 5 as example
    print(f"First 5 rows in order: {order[:5]}")
    print(
        f"ANSWER CAR-14 {json.dumps({'order': order})}"
    )


if __name__ == "__main__":
    main()
