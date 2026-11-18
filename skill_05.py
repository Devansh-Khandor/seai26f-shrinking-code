#!/usr/bin/env python3
"""
skill_05.py: Analysis of poor fuel efficiency and overall performance for auto93 dataset.
Usage: python skill_05.py <csv_path>
"""
import sys
import json
from typing import List, Dict, Any
from collections import Counter
from skill_01 import read_csv, build_table  # reusing existing helpers


# ------------------------------------------------------------------
# Utility: compute percentiles
def percentile(values: List[float], pct: float) -> float:
    """Return the percentile value of a list."""
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = int(pct * len(values_sorted) / 100)
    if k == 0:
        k = 1
    return values_sorted[min(k - 1, len(values_sorted) - 1)]


# ------------------------------------------------------------------
# CAR-09: Poor fuel efficiency and associated characteristics
def car_09(tbl) -> Dict[str, Any]:
    """Identify poorly fuel‑efficient cars and most strongly associated features."""
    mpg_at = tbl.cols["Mpg+"].at

    # collect Mpg+ values (ignoring missing)
    mpg_vals = [r[mpg_at] for r in tbl.rows if r[mpg_at] != "?"]
    n = len(mpg_vals)

    # statistical thresholds
    mean_mpg = sum(mpg_vals) / n
    sd_mpg = (sum((v - mean_mpg) ** 2 for v in mpg_vals) / n) ** 0.5
    thresh = mean_mpg - sd_mpg

    # ensure at least 5 rows: fall back to 20th percentile if needed
    poor_rows = [i + 1 for i, r in enumerate(tbl.rows) if r[mpg_at] != "?"
                 and r[mpg_at] <= thresh]
    if len(poor_rows) < 5:
        thresh = percentile(mpg_vals, 20)
        poor_rows = [i + 1 for i, r in enumerate(tbl.rows) if r[mpg_at] != "?"
                     and r[mpg_at] <= thresh]

    # rank non‑goal columns by association
    non_goal_cols = [col for col in tbl.cols.values() if col.goal == 0]
    associations = []

    for col in non_goal_cols:
        at = col.at
        # values for all rows and for poor rows
        vals_all = [r[at] for r in tbl.rows if r[at] != "?"]
        vals_poor = [tbl.rows[i - 1][at] for i in poor_rows]
        if isinstance(col, ezr.Sym):  # categorical
            mode_all = Counter(vals_all).most_common(1)[0][0]
            mode_poor = Counter(vals_poor).most_common(1)[0][0]
            freq_all = Counter(vals_all)[mode_all] / len(vals_all)
            freq_poor = Counter(vals_poor)[mode_poor] / len(vals_poor)
            diff = abs(freq_poor - freq_all)
        else:  # numeric
            mean_all = sum(vals_all) / len(vals_all)
            mean_poor = sum(vals_poor) / len(vals_poor)
            diff = abs(mean_all - mean_poor)
        associations.append((diff, col.txt))

    # sort by strongest association
    associations.sort(reverse=True)
    ranking = [name for _, name in associations[:5]]
    return {"poor": poor_rows, "ranking": ranking}


# ------------------------------------------------------------------
# CAR-10: Common characteristics of bad vs good overall performance
def car_10(tbl) -> Dict[str, Any]:
    """Find typical features of bad and good overall performers."""
    # compute goal columns
    goal_cols = [(col.at, col) for col in tbl.cols.values() if col.goal != 0]

    # helper: compute combined normalized score for a row
    def row_score(row):
        score = 0.0
        for at, col in goal_cols:
            val = row[at]
            if val == "?":
                continue
            vals = [r[at] for r in tbl.rows if r[at] != "?"]
            mn = min(vals)
            mx = max(vals)
            if mx == mn:
                comp = 1.0
            else:
                if col.goal == 1:  # maximize
                    comp = (val - mn) / (mx - mn)
                else:  # minimize
                    comp = (mx - val) / (mx - mn)
            score += comp
        return score

    scores = [row_score(r) for r in tbl.rows]
    n = len(scores)
    mean_score = sum(scores) / n
    sd_score = (sum((s - mean_score) ** 2 for s in scores) / n) ** 0.5
    low_thresh = mean_score - sd_score
    high_thresh = mean_score + sd_score

    # ensure at least 5 each: use percentiles if needed
    bad_rows = [i + 1 for i, s in enumerate(scores) if s <= low_thresh]
    good_rows = [i + 1 for i, s in enumerate(scores) if s >= high_thresh]
    if len(bad_rows) < 5:
        low_thresh = percentile(scores, 20)
        bad_rows = [i + 1 for i, s in enumerate(scores) if s <= low_thresh]
    if len(good_rows) < 5:
        high_thresh = percentile(scores, 80)
        good_rows = [i + 1 for i, s in enumerate(scores) if s >= high_thresh]

    # non‑goal columns
    non_goal_cols = [col for col in tbl.cols.values() if col.goal == 0]

    def typical(col, row_indices):
        vals = [tbl.rows[i - 1][col.at] for i in row_indices if tbl.rows[i - 1][col.at] != "?"]
        if isinstance(col, ezr.Sym):
            return Counter(vals).most_common(1)[0][0]
        else:
            return sum(vals) / len(vals)

    differences = []
    for col in non_goal_cols:
        bad_val = typical(col, bad_rows)
        good_val = typical(col, good_rows)
        if isinstance(col, ezr.Sym):
            diff = 1.0 if bad_val != good_val else 0.0
        else:
            diff = abs(bad_val - good_val)
        differences.append((diff, col.txt, bad_val, good_val))

    differences.sort(reverse=True)
    diff_list = [
        {"feature": name, "bad": bad, "good": good}
        for _, name, bad, good in differences[:5]
    ]

    return {
        "bad": bad_rows,
        "good": good_rows,
        "differences": diff_list,
    }


# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]

    header, rows = read_csv(path)
    tbl = build_table(header, rows)

    # CAR-09
    print("== CAR-09 ==")
    result_09 = car_09(tbl)
    # some printable output
    print("Poor Mpg+ rows:", result_09["poor"][:5], "...")
    print("Top associated columns:", result_09["ranking"])
    print(f"ANSWER CAR-09 {json.dumps(result_09)}")

    # CAR-10
    print("\n== CAR-10 ==")
    result_10 = car_10(tbl)
    print("Bad rows:", result_10["bad"][:5], "...")
    print("Good rows:", result_10["good"][:5], "...")
    print("Differences (top 5):")
    for d in result_10["differences"]:
        print(f"  {d['feature']}: bad={d['bad']}, good={d['good']}")
    print(f"ANSWER CAR-10 {json.dumps(result_10)}")


if __name__ == "__main__":
    main()
