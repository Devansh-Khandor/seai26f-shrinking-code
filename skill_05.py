#!/usr/bin/env python3
"""
Skill 05: NASA-09 and NASA-10

NASA-09 – For projects that required unusually high development effort, identify which
recorded project characteristics are most strongly associated with those outcomes.

NASA-10 – Examine projects with unusually high defect counts and identify the
characteristics that distinguish them from projects with better quality outcomes.

The script reads a CSV file of software projects, performs the analyses, and
prints machine‑checked answers in JSON.
"""

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from typing import Any, List, Tuple

# ----------------------------------------------------------------------
# Utility: atom conversion (same as in previous skills)
# ----------------------------------------------------------------------
def atom(s: str) -> Any:
    """Coerce a CSV cell to a Python value."""
    s = s.strip()
    if s == "?":
        return s
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return {"True": True, "False": False}.get(s, s)

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def mean_sd(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / n
    return m, math.sqrt(var)

def mode(values: List[Any]) -> Any:
    if not values:
        return None
    counter = Counter(values)
    return counter.most_common(1)[0][0]

def is_numeric(values: List[Any]) -> bool:
    for v in values:
        if v == "?":
            continue
        if not isinstance(v, (int, float)):
            return False
    return True

def round_num(v: Any) -> Any:
    if isinstance(v, float):
        # Remove trailing zeros
        r = round(v, 4)
        # If int after rounding, return int
        if isinstance(r, float) and r.is_integer():
            return int(r)
        return r
    return v

# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    # Load data
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows_raw = [row for row in reader]

    # Convert to Python values
    rows: List[List[Any]] = []
    for row in rows_raw:
        rows.append([atom(v) for v in row])

    N = len(rows)

    # Identify goal and feature columns
    goal_cols = [name for name in headers if name.endswith("+") or name.endswith("-")]
    feature_names = [name for name in headers
                     if not name.endswith("X")
                     and not name.endswith("+")
                     and not name.endswith("-")]
    # Map column name -> index
    col_index = {name: i for i, name in enumerate(headers)}

    # ------------------- NASA-09 -------------------
    # Effort- statistics
    effort_vals = [row[col_index["Effort-"]] for row in rows if row[col_index["Effort-"]] != "?"]
    effort_mean, effort_sd = mean_sd(effort_vals)

    # Determine poor threshold
    threshold_poor = effort_mean + 2 * effort_sd
    poor_indices = [i + 1 for i, row in enumerate(rows)
                    if row[col_index["Effort-"]] != "?" and row[col_index["Effort-"]] > threshold_poor]
    if len(poor_indices) < 5:
        # use top 5 by effort
        sorted_by_effort = sorted([(row[col_index["Effort-"]], i + 1) for i, row in enumerate(rows)
                                  if row[col_index["Effort-"]] != "?"],
                                  key=lambda x: x[0], reverse=True)
        poor_indices = [idx for _, idx in sorted_by_effort[:5]]
    poor_indices.sort()

    # Compute overall and poor statistics for each feature
    feature_stats: Dict[str, Dict[str, Any]] = {}
    for name in feature_names:
        vals = [row[col_index[name]] for row in rows if row[col_index[name]] != "?"]
        if not vals:
            continue
        numeric = is_numeric(vals)
        if numeric:
            m, _ = mean_sd([float(v) for v in vals])
            overall_typ = round_num(m)
            # poor group
            poor_vals = [row[col_index[name]] for row in rows
                         if row[col_index[name]] != "?" and row[col_index["Effort-"]] != "?"]
            # Only consider poor rows
            poor_vals = [row[col_index[name]] for row in rows
                         if row[col_index[name]] != "?" and (row[col_index["Effort-"]] != "?"
                         and row[col_index["Effort-"]] > threshold_poor)]
            if poor_vals:
                pm, _ = mean_sd([float(v) for v in poor_vals])
                poor_typ = round_num(pm)
                diff = abs(float(poor_typ) - float(overall_typ))
                feature_stats[name] = {
                    "type": "numeric",
                    "overall": overall_typ,
                    "poor": poor_typ,
                    "diff": diff,
                }
        else:
            # categorical
            m = mode(vals)
            overall_typ = m
            poor_vals = [row[col_index[name]] for row in rows
                         if row[col_index[name]] != "?" and row[col_index["Effort-"]] != "?"
                         and row[col_index["Effort-"]] > threshold_poor]
            if poor_vals:
                pm = mode(poor_vals)
                # frequency difference
                freq_overall = Counter(vals)[overall_typ] / len(vals)
                freq_poor = Counter(poor_vals)[pm] / len(poor_vals)
                diff = abs(freq_poor - freq_overall)
                feature_stats[name] = {
                    "type": "categorical",
                    "overall": overall_typ,
                    "poor": pm,
                    "diff": diff,
                }

    # Rank features by diff
    ranked_features = sorted(feature_stats.items(),
                             key=lambda kv: kv[1]["diff"],
                             reverse=True)
    ranking_names = [name for name, _ in ranked_features[:max(3, len(ranked_features))]]

    # ------------------- NASA-10 -------------------
    # Defects- statistics
    defects_vals = [row[col_index["Defects-"]] for row in rows if row[col_index["Defects-"]] != "?"]
    defects_mean, defects_sd = mean_sd(defects_vals)

    # Bad threshold
    threshold_bad = defects_mean + 2 * defects_sd
    bad_indices = [i + 1 for i, row in enumerate(rows)
                   if row[col_index["Defects-"]] != "?" and row[col_index["Defects-"]] > threshold_bad]
    if len(bad_indices) < 5:
        sorted_by_defects = sorted([(row[col_index["Defects-"]], i + 1) for i, row in enumerate(rows)
                                    if row[col_index["Defects-"]] != "?"],
                                    key=lambda x: x[0], reverse=True)
        bad_indices = [idx for _, idx in sorted_by_defects[:5]]
    bad_indices.sort()

    # Good threshold
    threshold_good = defects_mean - 2 * defects_sd
    good_indices = [i + 1 for i, row in enumerate(rows)
                    if row[col_index["Defects-"]] != "?" and row[col_index["Defects-"]] < threshold_good]
    if len(good_indices) < 5:
        sorted_by_defects = sorted([(row[col_index["Defects-"]], i + 1) for i, row in enumerate(rows)
                                    if row[col_index["Defects-"]] != "?"],
                                    key=lambda x: x[0])
        good_indices = [idx for _, idx in sorted_by_defects[:5]]
    good_indices.sort()

    # Compute typicals for bad and good groups
    diff_list: List[Dict[str, Any]] = []
    for name in feature_names:
        # Gather values in bad and good
        bad_vals = [row[col_index[name]] for row in rows
                    if row[col_index[name]] != "?" and (row[col_index["Defects-"]] != "?"
                    and row[col_index["Defects-"]] > threshold_bad)]
        good_vals = [row[col_index[name]] for row in rows
                     if row[col_index[name]] != "?" and (row[col_index["Defects-"]] != "?"
                     and row[col_index["Defects-"]] < threshold_good)]
        if not bad_vals or not good_vals:
            continue
        numeric = is_numeric(bad_vals + good_vals)
        if numeric:
            bm, _ = mean_sd([float(v) for v in bad_vals])
            gm, _ = mean_sd([float(v) for v in good_vals])
            bad_typ = round_num(bm)
            good_typ = round_num(gm)
            diff = abs(bad_typ - good_typ)
        else:
            bm = mode(bad_vals)
            gm = mode(good_vals)
            bad_typ = bm
            good_typ = gm
            freq_bad = Counter(bad_vals)[bm] / len(bad_vals)
            freq_good = Counter(good_vals)[gm] / len(good_vals)
            diff = abs(freq_bad - freq_good)
        diff_list.append({
            "feature": name,
            "bad": bad_typ,
            "good": good_typ,
            "diff": diff,
        })

    # Sort differences and pick top 3
    diff_list.sort(key=lambda d: d["diff"], reverse=True)
    top_diffs = diff_list[:max(3, len(diff_list))]
    # Remove 'diff' key for output
    differences_output = [{"feature": d["feature"], "bad": d["bad"], "good": d["good"]}
                          for d in top_diffs]

    # ------------------- Output -------------------
    print("== NASA-09 ==")
    print(f"Poor rows: {poor_indices}")
    print(f"Ranking: {ranking_names}")

    print("\n== NASA-10 ==")
    print(f"Bad rows: {bad_indices}")
    print(f"Good rows: {good_indices}")
    print(f"Differences: {differences_output}")

    # Machine‑checked answers
    answer_09 = {"poor": poor_indices, "ranking": ranking_names}
    answer_10 = {"bad": bad_indices, "good": good_indices, "differences": differences_output}

    print(f"\nANSWER NASA-09 {json.dumps(answer_09)}")
    print(f"ANSWER NASA-10 {json.dumps(answer_10)}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
