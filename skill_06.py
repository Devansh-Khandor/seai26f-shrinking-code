#!/usr/bin/env python3
"""
Skill 06 – Apache performance analytics (APACHE-11 & APACHE-12)

Usage:
    python skill_06.py <csv_path>
"""

import json
import sys

import ezr


def goal_columns(tbl: ezr.TBL):
    """Return list of NUM columns that are goals (end with + or -)."""
    return [c for c in tbl.cols.values() if c.txt.endswith(('+', '-'))]


def setting_columns(tbl: ezr.TBL):
    """Return list of columns that are settings (neither goal nor X)."""
    return [c for c in tbl.cols.values()
            if not c.txt.endswith(('+', '-', 'X'))]


def row_values_dict(row: ezr.ROW, tbl: ezr.TBL):
    """Return a dict mapping column names to the row's values."""
    return {tbl.names[i]: row[i] for i in range(len(row))}


def compute_scores(tbl: ezr.TBL, rows):
    """Compute a score for each row (higher = better)."""
    scores = []
    for idx, row in enumerate(rows, start=1):
        # ydist is distance to best goal; lower is better
        ydist_val = ezr.ydist(tbl, row)
        score = -ydist_val  # higher score means better
        scores.append((idx, row, score))
    return scores


def find_neighbors(rec_row, tbl, rows, count=3):
    """Return the `count` closest rows to `rec_row` in x space."""
    distances = []
    for idx, r in enumerate(rows, start=1):
        if r is rec_row:
            continue
        d = ezr.xdist(tbl, rec_row, r)
        distances.append((d, idx, r))
    distances.sort()
    neighbors = []
    for _, idx, r in distances[:count]:
        neighbors.append((idx, r))
    return neighbors


def differences_between(a_row, b_row, tbl):
    """Return list of columns where a and b differ."""
    diffs = []
    for i, col in enumerate(tbl.names):
        a_val = a_row[i]
        b_val = b_row[i]
        if a_val != b_val:
            diffs.append({"feature": col, "a": a_val, "b": b_val})
    return diffs


def tradeoffs(a_row, b_row, goal_cols):
    """Return list of goal columns where non‑preferred (b) is better."""
    trades = []
    for c in goal_cols:
        a_val = a_row[c.at]
        b_val = b_row[c.at]
        if a_val == "?" or b_val == "?" or not isinstance(a_val, (int, float)):
            continue
        if c.txt.endswith("-") and b_val < a_val:
            trades.append(c.txt)
        if c.txt.endswith("+") and b_val > a_val:
            trades.append(c.txt)
    return trades


def main():
    if len(sys.argv) < 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    tbl = ezr.Tbl(ezr.csv(csv_path))
    rows = tbl.rows

    goals = goal_columns(tbl)
    settings = setting_columns(tbl)

    scores = compute_scores(tbl, rows)

    # Find best and worst rows
    best_idx, best_row, best_score = max(scores, key=lambda t: t[2])
    worst_idx, worst_row, worst_score = min(scores, key=lambda t: t[2])

    # ---------- APACHE-11 ----------
    print("== APACHE-11 ==")

    # Recommendation is the best row
    print(f"Recommended row: {best_idx}")
    print(f"Score: {best_score:.3f}")

    # Find 3 nearest neighbors
    neigh = find_neighbors(best_row, tbl, rows, count=3)
    print("Nearest neighbors:")
    for n_idx, n_row in neigh:
        print(f"  Row {n_idx}")

    # ---------- APACHE-12 ----------
    print("\n== APACHE-12 ==")

    print(f"Row a (best): {best_idx}")
    print(f"Row b (worst): {worst_idx}")
    print(f"Scores: a={best_score:.3f}, b={worst_score:.3f}")

    # Determine preferred (higher score)
    preferred = best_idx if best_score >= worst_score else worst_idx
    non_preferred = worst_idx if preferred == best_idx else best_idx

    print(f"Preferred row: {preferred}")

    # Differences
    diff_rows = best_row if preferred == best_idx else worst_row
    other_rows = worst_row if preferred == best_idx else best_row
    diffs = differences_between(diff_rows, other_rows, tbl)
    print("Differences:")
    for d in diffs[:3]:
        print(f"  {d['feature']}: a={d['a']} b={d['b']}")

    # Tradeoffs where non‑preferred is better
    trades = tradeoffs(diff_rows, other_rows, goals)
    print(f"Tradeoffs (non‑preferred better): {trades}")

    # ---------- Machine‑checked answers ----------
    # APACHE-11 answer
    rec_json = {
        "recommended": best_idx,
        "neighbors": [
            {"row": n_idx, "values": row_values_dict(n_row, tbl)}
            for n_idx, n_row in neigh
        ],
    }
    print(f"ANSWER APACHE-11 {json.dumps(rec_json)}")

    # APACHE-12 answer
    a_json = {
        "a": best_idx,
        "b": worst_idx,
        "preferred": preferred,
        "scores": {"a": best_score, "b": worst_score},
        "differences": diffs,
        "tradeoffs": trades,
    }
    print(f"ANSWER APACHE-12 {json.dumps(a_json)}")


if __name__ == "__main__":
    main()
