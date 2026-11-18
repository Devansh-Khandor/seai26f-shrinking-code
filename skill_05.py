#!/usr/bin/env python3
"""
Skill 05 – NASA effort and defect blame

This script reads a NASA project CSV file and prints two analyses:

  * NASA-09 – Identify the characteristics most strongly associated
    with projects that required unusually high development effort.
  * NASA-10 – Identify the characteristics that distinguish projects
    with unusually high defect counts from those with better quality.

The output follows the required format, with a machine‑checked JSON line
for each section.  All calculations are derived from the CSV file
supplied on the command line.
"""

import json
import sys
from collections import Counter, defaultdict

import ezr
import skill_01 as s01  # for compute_goal_stats if needed


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def mean_sd(values):
    """Return mean and standard deviation of a list of numeric values,
    ignoring missing values ('?')."""
    nums = [v for v in values if v != "?"]
    n = len(nums)
    if n == 0:
        return 0.0, 0.0
    m = sum(nums) / n
    var = sum((v - m) ** 2 for v in nums) / (n - 1) if n > 1 else 0.0
    return m, var ** 0.5


def group_mean(values):
    """Mean of numeric values, ignoring missing."""
    vals = [v for v in values if v != "?"]
    n = len(vals)
    return sum(vals) / n if n else 0.0


def most_common(values):
    """Most common value in a list, ignoring missing."""
    vals = [v for v in values if v != "?"]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def proportion_of(value, values):
    """Proportion of given value in values list (ignoring missing)."""
    vals = [v for v in values if v != "?"]
    if not vals:
        return 0.0
    return vals.count(value) / len(vals)


def ensure_minimum(rows, threshold, keep, reverse=False, nmin=5):
    """Adjust selection to guarantee at least nmin items."""
    selected = sorted(rows, key=lambda r: r[keep], reverse=reverse)[:threshold]
    if len(selected) < nmin:
        selected = sorted(rows, key=lambda r: r[keep], reverse=reverse)[:nmin]
    return selected


# ----------------------------------------------------------------------
# NASA-09: High effort blame
# ----------------------------------------------------------------------
def nasa_09(tbl):
    eff_col = next(col for col in tbl.cols.values() if col.txt == "Effort-")
    # compute mean & sd for Effort-
    eff_vals = [row[eff_col.at] for row in tbl.rows]
    eff_mean, eff_sd = mean_sd(eff_vals)
    # threshold for unusually high effort
    thr = eff_mean + 1.5 * eff_sd
    # gather poor rows
    poor_rows = [row for row in tbl.rows if row[eff_col.at] > thr]
    # ensure at least 5
    if len(poor_rows) < 5:
        poor_rows = ensure_minimum(tbl.rows, 5, eff_col.at, reverse=True)
    poor_indices = [tbl.rows.index(r) + 1 for r in poor_rows]

    # compute feature ranking
    feature_names = [
        col.txt
        for col in tbl.cols.values()
        if not col.txt.endswith(('+', '-', 'X'))
    ]
    ranking = []

    for name in feature_names:
        col = tbl.cols[name]
        # values for poor and rest groups
        poor_vals = [r[col.at] for r in poor_rows]
        rest_vals = [r[col.at] for r in tbl.rows if r not in poor_rows]
        if isinstance(col, ezr.Num):
            # numeric: mean difference
            mean_poor = group_mean(poor_vals)
            mean_rest = group_mean(rest_vals)
            diff = abs(mean_poor - mean_rest)
        else:
            # categorical: difference in proportion of most common value
            mc_poor = most_common(poor_vals)
            mc_rest = most_common(rest_vals)
            prop_poor = proportion_of(mc_poor, poor_vals)
            prop_rest = proportion_of(mc_rest, rest_vals)
            diff = abs(prop_poor - prop_rest)
        ranking.append((name, diff))
    ranking.sort(key=lambda x: x[1], reverse=True)
    ranking_names = [name for name, _ in ranking[: max(3, len(ranking))]]

    # Output
    print("== NASA-09 ==")
    print(f"Poor effort rows (count={len(poor_indices)}): {poor_indices[:5]}")
    print(f"Feature ranking (most associated first): {ranking_names[:5]}")
    print(f"Answer: {len(poor_indices)} poor rows, {len(ranking_names)} features ranked.")
    # JSON answer
    ans = {"poor": poor_indices, "ranking": ranking_names}
    print(f"ANSWER NASA-09 {json.dumps(ans, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# NASA-10: High defect blame
# ----------------------------------------------------------------------
def nasa_10(tbl):
    def_col = next(col for col in tbl.cols.values() if col.txt == "Defects-")
    def_vals = [row[def_col.at] for row in tbl.rows]
    def_mean, def_sd = mean_sd(def_vals)

    # thresholds
    bad_thr = def_mean + 1.5 * def_sd
    good_thr = def_mean - 1.5 * def_sd

    bad_rows = [r for r in tbl.rows if r[def_col.at] > bad_thr]
    good_rows = [r for r in tbl.rows if r[def_col.at] < good_thr]

    # ensure at least 5 each
    if len(bad_rows) < 5:
        bad_rows = ensure_minimum(tbl.rows, 5, def_col.at, reverse=True)
    if len(good_rows) < 5:
        good_rows = ensure_minimum(tbl.rows, 5, def_col.at, reverse=False)

    bad_indices = [tbl.rows.index(r) + 1 for r in bad_rows]
    good_indices = [tbl.rows.index(r) + 1 for r in good_rows]

    # typical values per feature
    feature_names = [
        col.txt
        for col in tbl.cols.values()
        if not col.txt.endswith(('+', '-', 'X'))
    ]

    diff_list = []

    for name in feature_names:
        col = tbl.cols[name]
        bad_vals = [r[col.at] for r in bad_rows]
        good_vals = [r[col.at] for r in good_rows]
        if isinstance(col, ezr.Num):
            bad_typ = group_mean(bad_vals)
            good_typ = group_mean(good_vals)
            diff = abs(bad_typ - good_typ)
        else:
            bad_typ = most_common(bad_vals)
            good_typ = most_common(good_vals)
            prop_bad = proportion_of(bad_typ, bad_vals)
            prop_good = proportion_of(good_typ, good_vals)
            diff = abs(prop_bad - prop_good)
        diff_list.append({"feature": name, "bad": bad_typ, "good": good_typ, "diff": diff})
    diff_list.sort(key=lambda x: x["diff"], reverse=True)
    differences = [{"feature": d["feature"], "bad": d["bad"], "good": d["good"]} for d in diff_list[:max(3, len(diff_list))]]

    # Output
    print("== NASA-10 ==")
    print(f"Bad defect rows (count={len(bad_indices)}): {bad_indices[:5]}")
    print(f"Good defect rows (count={len(good_indices)}): {good_indices[:5]}")
    print(f"Differences (most distinct first): {differences[:5]}")
    # JSON answer
    ans = {"bad": bad_indices, "good": good_indices, "differences": differences}
    print(f"ANSWER NASA-10 {json.dumps(ans, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    tbl = ezr.Tbl(ezr.csv(csv_path))
    nasa_09(tbl)
    nasa_10(tbl)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_05.py <csv_path>")
    main(sys.argv[1])
