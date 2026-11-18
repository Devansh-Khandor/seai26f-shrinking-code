#!/usr/bin/env python3
"""
skill_06.py

Usage: python skill_06.py <csv path>
"""

import json
import sys

import ezr

# ----------------------------------------------------------------------
# Helper functions (copied from skill_04 for reuse)
# ----------------------------------------------------------------------
def compute_col_stats(rows, goal_names):
    """Return min/max for each goal column among the given rows."""
    stats = {}
    for name in goal_names:
        vals = [v for r in rows
                for i, col in enumerate(r)
                if col != "?" and header[i] == name
                for v in [v]
                ]
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        if not vals:
            stats[name] = (0.0, 1.0)
        else:
            mn, mx = min(vals), max(vals)
            if mn == mx:
                mx = mn + 1.0
            stats[name] = (mn, mx)
    return stats


def compute_score(row, stats):
    """Compute composite score for a row based on provided min/max per goal."""
    score = 0.0
    for name, (mn, mx) in stats.items():
        idx = header.index(name)
        val = row[idx]
        if val == "?":
            norm = 0.0
        else:
            val = float(val)
            if name.endswith("-"):
                norm = (mx - val) / (mx - mn)
            else:
                norm = (val - mn) / (mx - mn)
        score += norm
    return score

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Load data
    rows_iter = ezr.csv(csv_path)
    header = next(rows_iter)
    data_rows = list(rows_iter)

    # Build table object for distance calculation
    tbl = ezr.Tbl([header] + data_rows)

    # Identify goal columns
    goal_names = [name for name in header if name[-1] in "+-"]

    # Compute goal min/max
    stats = compute_col_stats(data_rows, goal_names)

    # Compute composite scores
    scores = [compute_score(r, stats) for r in data_rows]

    # Find best and worst rows
    best_idx = max(range(len(data_rows)), key=lambda i: scores[i])
    worst_idx = min(range(len(data_rows)), key=lambda i: scores[i])

    # ------------------------------------------------------------------
    # CAR-11: Recommendation explanation
    # ------------------------------------------------------------------
    print("== CAR-11 ==")

    rec_row_num = best_idx + 1  # 1-based
    rec_row = data_rows[best_idx]

    # Find neighbors
    neighbors = []
    for idx, row in enumerate(data_rows):
        if idx == best_idx:
            continue
        d = ezr.xdist(tbl, rec_row, row)
        neighbors.append((d, idx, row))
    neighbors.sort(key=lambda x: x[0])
    top_neighbors = neighbors[:3]

    # Print some results
    print(f"Recommended row: {rec_row_num}")
    print("Neighbors (closest):")
    for _, idx, row in top_neighbors:
        print(f"  row {idx + 1}")

    # Machine-checked answer
    neighbor_json = []
    for _, idx, row in top_neighbors:
        vals = {col: row[header.index(col)] for col in goal_names}
        neighbor_json.append({"row": idx + 1, "values": vals})

    answer_11 = {"recommended": rec_row_num, "neighbors": neighbor_json}
    print(f"ANSWER CAR-11 {json.dumps(answer_11, separators=(',', ':'))}")

    # ------------------------------------------------------------------
    # CAR-12: Trade‑off explanation between two vehicles
    # ------------------------------------------------------------------
    print("\n== CAR-12 ==")

    a_num = rec_row_num
    b_num = worst_idx + 1
    preferred = a_num  # best row is preferred

    # Scores
    score_a = scores[best_idx]
    score_b = scores[worst_idx]

    # Differences: all columns where values differ
    diffs = []
    for i, col in enumerate(header):
        val_a = data_rows[best_idx][i]
        val_b = data_rows[worst_idx][i]
        if val_a != val_b:
            diffs.append({"feature": col, "a": val_a, "b": val_b})

    # Tradeoffs: goal columns where non‑preferred row is better
    tradeoffs = []
    for col in goal_names:
        idx = header.index(col)
        val_a = data_rows[best_idx][idx]
        val_b = data_rows[worst_idx][idx]
        if val_a == "?" or val_b == "?":
            continue
        val_a = float(val_a)
        val_b = float(val_b)
        better_in_b = False
        if col.endswith("+") and val_b > val_a:
            better_in_b = True
        if col.endswith("-") and val_b < val_a:
            better_in_b = True
        if better_in_b:
            tradeoffs.append(col)

    # Print some results
    print(f"Vehicle A (best): {a_num}  score={score_a:.3f}")
    print(f"Vehicle B (worst): {b_num}  score={score_b:.3f}")
    print(f"Preferred: {preferred}")
    print(f"Tradeoffs: {tradeoffs}")

    # Machine-checked answer
    answer_12 = {
        "a": a_num,
        "b": b_num,
        "preferred": preferred,
        "scores": {"a": score_a, "b": score_b},
        "differences": diffs,
        "tradeoffs": tradeoffs
    }
    print(f"ANSWER CAR-12 {json.dumps(answer_12, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
