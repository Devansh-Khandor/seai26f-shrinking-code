#!/usr/bin/env python3
"""
Skill 05 – Apache performance analytics (APACHE-09 & APACHE-10)

Usage:
    python skill_05.py <csv_path>
"""

import sys
import json
import math
import ezr

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def get_goal_col(tbl):
    """Return the first column whose name ends with '+' or '-'."""
    for c in tbl.cols.values():
        if c.txt.endswith(('+', '-')):
            return c
    raise RuntimeError("No goal column found.")

def get_non_goal_cols(tbl):
    """Return list of columns that are neither goal nor X columns."""
    return [c for c in tbl.cols.values()
            if not c.txt.endswith(('+', '-', 'X'))]

def row_number(row):
    """Return the 1‑based index of a row in the table."""
    return tbl.rows.index(row) + 1  # tbl is global for simplicity

def get_poor_rows(tbl, goal_col, k):
    """Return the k rows with the highest (worst) goal values."""
    # filter rows with a numeric goal
    scored = [(float(r[goal_col.at]), r) for r in tbl.rows
              if r[goal_col.at] != "?"]
    scored.sort(reverse=True)  # worst first
    return [row for _, row in scored[:k]]

def get_best_rows(tbl, goal_col, k):
    """Return the k rows with the lowest (best) goal values."""
    scored = [(float(r[goal_col.at]), r) for r in tbl.rows
              if r[goal_col.at] != "?"]
    scored.sort()  # best first
    return [row for _, row in scored[:k]]

def share_one(rows, col):
    """Return fraction of 1s in given rows for a column, skipping '?'."""
    ones = 0
    total = 0
    for r in rows:
        v = r[col.at]
        if v == "?":
            continue
        total += 1
        if v == 1:
            ones += 1
    return ones / total if total else 0.0

def overall_shares(tbl, cols):
    """Return dict of overall 1‑share per column over all rows."""
    return {c.txt: share_one(tbl.rows, c) for c in cols}

def poor_shares(rows, cols):
    """Return dict of 1‑share per column for given rows."""
    return {c.txt: share_one(rows, c) for c in cols}

# --------------------------------------------------------------------
# APACHE‑09
# --------------------------------------------------------------------
def apache_09(tbl):
    goal_col = get_goal_col(tbl)
    non_goals = get_non_goal_cols(tbl)

    n_rows = len(tbl.rows)
    k = max(5, math.ceil(n_rows * 0.05))

    poor_rows = get_poor_rows(tbl, goal_col, k)
    poor_nums = [tbl.rows.index(r)+1 for r in poor_rows]

    # Compute shares
    overall = overall_shares(tbl, non_goals)
    poor = poor_shares(poor_rows, non_goals)

    # Rank by absolute difference
    diffs = []
    for txt in overall:
        diff = abs(poor[txt] - overall[txt])
        diffs.append((diff, txt))
    diffs.sort(reverse=True)
    ranking = [txt for _, txt in diffs[:max(3, len(diffs))]]

    # Print section
    print("== APACHE-09 ==")
    print(f"Poor rows (k={k}): {poor_nums}")
    print("Top distinguishing settings (most associated with poor):")
    for i, txt in enumerate(ranking, start=1):
        print(f"  {i}. {txt}")

    # Machine‑checked answer
    ans = {
        "poor": poor_nums,
        "ranking": ranking
    }
    print(f"ANSWER APACHE-09 {json.dumps(ans)}")

# --------------------------------------------------------------------
# APACHE‑10
# --------------------------------------------------------------------
def apache_10(tbl):
    goal_col = get_goal_col(tbl)
    non_goals = get_non_goal_cols(tbl)

    n_rows = len(tbl.rows)
    k = max(5, math.ceil(n_rows * 0.05))

    # Poor and good rows
    poor_rows = get_poor_rows(tbl, goal_col, k)
    good_rows = get_best_rows(tbl, goal_col, k)
    poor_nums = [tbl.rows.index(r)+1 for r in poor_rows]
    good_nums = [tbl.rows.index(r)+1 for r in good_rows]

    # Find frequent combinations among poor rows
    combos = []

    # Consider all pairs of columns
    from itertools import combinations
    for c1, c2 in combinations(non_goals, 2):
        counts_bad = {}
        counts_good = {}
        for r in poor_rows:
            v1 = r[c1.at]
            v2 = r[c2.at]
            if v1 == "?" or v2 == "?":
                continue
            key = (int(v1), int(v2))
            counts_bad[key] = counts_bad.get(key, 0) + 1
        for r in good_rows:
            v1 = r[c1.at]
            v2 = r[c2.at]
            if v1 == "?" or v2 == "?":
                continue
            key = (int(v1), int(v2))
            counts_good[key] = counts_good.get(key, 0) + 1

        if not counts_bad:
            continue

        # choose the combination with highest count in bad set
        best_key = max(counts_bad, key=lambda k: counts_bad[k])
        bad_share = counts_bad[best_key] / len(poor_rows)
        good_share = counts_good.get(best_key, 0) / len(good_rows)

        combo_dict = {c1.txt: best_key[0], c2.txt: best_key[1]}
        combos.append({
            "settings": combo_dict,
            "bad_share": bad_share,
            "good_share": good_share
        })

    # sort by bad_share desc
    combos.sort(key=lambda x: x["bad_share"], reverse=True)
    # keep at most 5 combos
    combos = combos[:5] if combos else []

    # Print section
    print("== APACHE-10 ==")
    print(f"Bad rows (k={k}): {poor_nums}")
    print(f"Good rows (k={k}): {good_nums}")
    print("Frequent bad combinations:")
    for i, combo in enumerate(combos, start=1):
        print(f"  {i}. settings={combo['settings']}, bad_share={combo['bad_share']:.3f}, good_share={combo['good_share']:.3f}")

    # Machine‑checked answer
    ans = {
        "bad": poor_nums,
        "good": good_nums,
        "combinations": combos
    }
    print(f"ANSWER APACHE-10 {json.dumps(ans)}")

# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    tbl = ezr.Tbl(ezr.csv(csv_path))

    apache_09(tbl)
    apache_10(tbl)

if __name__ == "__main__":
    main()
