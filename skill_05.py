#!/usr/bin/env python3
"""
Skill 05 – NASA-09 & NASA-10
"""
import csv
import json
import sys
import statistics
from collections import Counter
from typing import List, Dict, Any, Tuple

# ----------------------------------------------------------------------
# Helper to parse numeric cells (used for goals)
# ----------------------------------------------------------------------
def parse_value(v: str) -> Any:
    """Parse a CSV cell to float, return None for missing."""
    if v == "" or v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return None

# ----------------------------------------------------------------------
# Load and parse the CSV
# ----------------------------------------------------------------------
def load_rows(csv_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load rows and return:
      - list of dicts with raw string values
      - list of column names (fieldnames)
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row.copy())
    return rows, fieldnames

# ----------------------------------------------------------------------
# Helper: compute mean & stdev of list of floats (ignore None)
# ----------------------------------------------------------------------
def mean_stdev(vals: List[float]) -> Tuple[float, float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return (None, None)
    m = sum(vals) / len(vals)
    if len(vals) > 1:
        sd = statistics.stdev(vals)
    else:
        sd = 0.0
    return (m, sd)

# ----------------------------------------------------------------------
# Helper: compute mean of numeric column in a set of rows
# ----------------------------------------------------------------------
def mean_numeric(rows: List[Dict[str, Any]], col: str) -> float:
    vals = [parse_value(r[col]) for r in rows if r[col] not in ("", "?")]
    return sum(vals) / len(vals) if vals else None

# ----------------------------------------------------------------------
# Helper: compute mode of categorical column in a set of rows
# ----------------------------------------------------------------------
def mode_categorical(rows: List[Dict[str, Any]], col: str) -> Any:
    vals = [r[col] for r in rows if r[col] not in ("", "?")]
    if not vals:
        return None
    cnt = Counter(vals)
    return cnt.most_common(1)[0][0]

# ----------------------------------------------------------------------
# Helper: compute frequency of a specific value in a column
# ----------------------------------------------------------------------
def freq_of_value(rows: List[Dict[str, Any]], col: str, val: Any) -> float:
    vals = [r[col] for r in rows if r[col] not in ("", "?")]
    if not vals:
        return 0.0
    return vals.count(val) / len(vals)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    raw_rows, fieldnames = load_rows(csv_path)

    # Convert rows into a format convenient for numeric operations
    # We keep raw strings for all columns; numeric parsing is done on demand
    # For convenience, create a list of row numbers (1-based)
    rows = raw_rows
    row_nums = list(range(1, len(rows) + 1))

    # ------------------------------------------------------------------
    # NASA-09: Unusually high Effort-
    # ------------------------------------------------------------------
    effort_vals = [parse_value(r["Effort-"]) for r in rows]
    effort_mean, effort_sd = mean_stdev(effort_vals)

    threshold_high_effort = effort_mean + 2 * effort_sd if effort_mean is not None else None
    poor_indices = [i for i, v in enumerate(effort_vals) if v is not None and v > threshold_high_effort]
    poor_rows = [row_nums[i] for i in poor_indices]

    # Association ranking of non‑goal columns
    non_goal_cols = [c for c in fieldnames if c not in ("Kloc+", "Effort-", "Defects-")]
    diffs = []

    # For each non-goal column compute association difference
    for col in non_goal_cols:
        # Determine if numeric
        vals_all = [parse_value(r[col]) for r in rows if r[col] not in ("", "?")]
        vals_poor = [parse_value(rows[i][col]) for i in poor_indices if rows[i][col] not in ("", "?")]
        if vals_all:
            # numeric if all parsed values are not None
            if all(v is not None for v in vals_all):
                mean_all = sum(vals_all) / len(vals_all)
                if vals_poor:
                    mean_poor = sum(vals_poor) / len(vals_poor)
                else:
                    mean_poor = mean_all  # no poor values, diff zero
                diff = abs(mean_poor - mean_all)
                diffs.append((col, diff))
            else:
                # categorical
                mode_all = mode_categorical(rows, col)
                freq_all = freq_of_value(rows, col, mode_all) if mode_all is not None else 0.0
                if vals_poor:
                    mode_poor = mode_categorical([rows[i] for i in poor_indices], col)
                    freq_poor = freq_of_value([rows[i] for i in poor_indices], col, mode_poor) if mode_poor is not None else 0.0
                else:
                    mode_poor = mode_all
                    freq_poor = freq_all
                diff = abs(freq_poor - freq_all)
                diffs.append((col, diff))
        else:
            continue  # skip columns with no data

    # Sort diffs descending by magnitude
    diffs.sort(key=lambda x: x[1], reverse=True)
    ranking_cols = [col for col, _ in diffs[:3]]  # at least 3

    # ------------------------------------------------------------------
    # NASA-10: Unusually high Defects-
    # ------------------------------------------------------------------
    defect_vals = [parse_value(r["Defects-"]) for r in rows]
    defect_mean, defect_sd = mean_stdev(defect_vals)

    threshold_high_defect = defect_mean + 2 * defect_sd if defect_mean is not None else None
    threshold_low_defect = defect_mean - 2 * defect_sd if defect_mean is not None else None

    bad_indices = [i for i, v in enumerate(defect_vals) if v is not None and v > threshold_high_defect]
    bad_rows = [row_nums[i] for i in bad_indices]

    # Good rows: choose lowest Defects- values
    good_candidates = [i for i, v in enumerate(defect_vals) if v is not None]
    good_candidates.sort(key=lambda i: defect_vals[i])
    good_rows = [row_nums[i] for i in good_candidates[:5]]  # at least 5

    # For each non-goal column, compute typical in bad and good rows
    diff_list = []
    for col in non_goal_cols:
        # numeric or categorical
        vals_bad = [parse_value(rows[i][col]) for i in bad_indices if rows[i][col] not in ("", "?")]
        vals_good = [parse_value(rows[i][col]) for i in good_candidates[:5] if rows[i][col] not in ("", "?")]

        if vals_bad and vals_good:
            # numeric
            if all(v is not None for v in vals_bad + vals_good):
                mean_bad = sum(vals_bad) / len(vals_bad)
                mean_good = sum(vals_good) / len(vals_good)
                diff = abs(mean_bad - mean_good)
                diff_list.append((col, diff, mean_bad, mean_good))
            else:
                # categorical
                mode_bad = mode_categorical([rows[i] for i in bad_indices], col)
                mode_good = mode_categorical([rows[i] for i in good_candidates[:5]], col)
                diff = 1.0 if mode_bad != mode_good else 0.0
                diff_list.append((col, diff, mode_bad, mode_good))
        else:
            continue

    # Sort by diff descending, keep top 3
    diff_list.sort(key=lambda x: x[1], reverse=True)
    differences = []
    for col, _, bad_t, good_t in diff_list[:3]:
        differences.append({
            "feature": col,
            "bad": bad_t,
            "good": good_t
        })

    # ------------------------------------------------------------------
    # Output NASA‑09
    # ------------------------------------------------------------------
    print("== NASA-09 ==")
    # Provide some simple statistics
    print(f"Unusually high Effort- rows: {len(poor_rows)}")
    print(f"First 3 poor rows: {poor_rows[:3]}")
    print(f"Top 3 associated characteristics: {ranking_cols}")
    # Machine‑checked answer
    answer_09 = {
        "poor": poor_rows,
        "ranking": ranking_cols
    }
    print("ANSWER NASA-09 " + json.dumps(answer_09, separators=(',', ':')))

    # ------------------------------------------------------------------
    # Output NASA‑10
    # ------------------------------------------------------------------
    print("\n== NASA-10 ==")
    print(f"Unfavorable Defects- rows: {len(bad_rows)}")
    print(f"First 3 bad rows: {bad_rows[:3]}")
    print(f"Better quality rows (low defects): {good_rows}")
    print(f"Differences (top 3): {differences}")
    answer_10 = {
        "bad": bad_rows,
        "good": good_rows,
        "differences": differences
    }
    print("ANSWER NASA-10 " + json.dumps(answer_10, separators=(',', ':')))

if __name__ == "__main__":
    main()
