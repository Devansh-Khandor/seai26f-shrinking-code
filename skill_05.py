#!/usr/bin/env python3
"""
skill_05.py – NASA-09 & NASA-10

For a given CSV file (passed as the sole command‑line argument) the program
computes:

* NASA-09 – projects with unusually high development effort (Effort-).
  It prints the row numbers of those projects and lists the non‑goal
  characteristics that are most strongly associated with high effort.
* NASA-10 – projects with unusually high defect counts (Defects-).
  It prints the row numbers of “bad” projects and of “good” projects, and
  lists the characteristics that best distinguish them.

The program follows the conventions defined in the project description:
  • “goal” columns end with '+' (maximize) or '-' (minimize).
  • “input” columns are all others, except columns whose name ends with 'X'.
  • Missing values are represented by the literal string '?'.

The output format is fixed – each NASA ID section begins with the line
`== <ID> ==`, followed by at least three human‑readable lines, and
concludes with a machine‑checked JSON line starting with `ANSWER <ID>`.
"""

import csv
import json
import math
import statistics
import sys

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(val):
    """Convert string to float; return None for missing/invalid."""
    if val is None:
        return None
    val = val.strip()
    if val == "" or val == "?":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def get_non_goal_columns(header):
    """Return list of column names that are not goals and not X."""
    return [c for c in header if not c.endswith(("+", "-")) and not c.endswith("X")]


def compute_mean_std(vals):
    if not vals:
        return (None, None)
    return (statistics.mean(vals), statistics.stdev(vals)) if len(vals) > 1 else (statistics.mean(vals), 0.0)


def compute_correlation(x_vals, y_vals):
    """Pearson correlation; returns absolute value. Handles zero std."""
    n = len(x_vals)
    if n < 2:
        return 0.0
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_vals) / n)
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_vals) / n)
    if std_x == 0 or std_y == 0:
        return 0.0
    return abs(cov / (std_x * std_y))


def mode(values):
    """Return the most common value among non‑None items."""
    counts = {}
    for v in values:
        if v is None:
            continue
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None
    max_count = max(counts.values())
    # Return any value that achieves the max count
    for val, cnt in counts.items():
        if cnt == max_count:
            return val


def typical_value(vals):
    """Return mean for numeric, mode for others."""
    numeric_vals = [v for v in vals if isinstance(v, (int, float))]
    if numeric_vals:
        return statistics.mean(numeric_vals)
    # For categorical or mixed types, use mode
    return mode(vals)


# ----------------------------------------------------------------------
# NASA-09 logic
# ----------------------------------------------------------------------
def nasa09(rows, header, non_goals):
    """Identify projects with unusually high Effort- and rank associated features."""
    effort_vals = []
    for r in rows:
        v = to_float(r["Effort-"])
        if v is not None:
            effort_vals.append(v)

    if not effort_vals:
        return {"poor": [], "ranking": []}

    eff_mean, eff_std = compute_mean_std(effort_vals)
    threshold = eff_mean + 2 * eff_std if eff_std is not None else eff_mean

    # Poor rows: indices (1‑based)
    poor_rows = [idx + 1 for idx, r in enumerate(rows)
                 if (v := to_float(r["Effort-"])) is not None and v > threshold]

    # If fewer than 5, take top 5 by Effort-
    if len(poor_rows) < 5:
        sorted_by_effort = sorted([(idx + 1, to_float(r["Effort-"])) for idx, r in enumerate(rows)
                                  if to_float(r["Effort-"]) is not None],
                                 key=lambda x: x[1], reverse=True)
        poor_rows = [idx for idx, _ in sorted_by_effort[:5]]

    # Association ranking
    scores = []
    for col in non_goals:
        # Gather pairs of (col_value, Effort-) where both present
        pairs = [(to_float(r[col]) if col in r and r[col].strip() != '?' else
                  (r[col] if r[col] is not None else None), to_float(r["Effort-"]))
                 for r in rows]
        col_vals = [p[0] for p in pairs if p[0] is not None and p[1] is not None]
        eff_vals = [p[1] for p in pairs if p[0] is not None and p[1] is not None]
        if not col_vals or not eff_vals:
            continue
        # Determine if numeric
        if all(isinstance(v, (int, float)) for v in col_vals):
            assoc = compute_correlation(col_vals, eff_vals)
        else:
            # Categorical: use difference of mean Effort- for the mode
            m = mode(col_vals)
            if m is None:
                continue
            vals_for_mode = [e for c, e in zip(col_vals, eff_vals) if c == m]
            if not vals_for_mode:
                continue
            assoc = abs(statistics.mean(vals_for_mode) - eff_mean)
        scores.append((assoc, col))

    scores.sort(key=lambda x: x[0], reverse=True)
    ranking = [col for _, col in scores[:max(3, len(scores))]]

    return {"poor": poor_rows, "ranking": ranking}


# ----------------------------------------------------------------------
# NASA-10 logic
# ----------------------------------------------------------------------
def nasa10(rows, header, non_goals):
    """Identify projects with unusually high Defects- and characteristics that distinguish them."""
    defect_vals = []
    for r in rows:
        v = to_float(r["Defects-"])
        if v is not None:
            defect_vals.append(v)

    if not defect_vals:
        return {"bad": [], "good": [], "differences": []}

    def_mean, def_std = compute_mean_std(defect_vals)
    bad_thresh = def_mean + 2 * def_std if def_std is not None else def_mean
    good_thresh = def_mean - 2 * def_std if def_std is not None else def_mean

    bad_rows = [idx + 1 for idx, r in enumerate(rows)
                if (v := to_float(r["Defects-"])) is not None and v > bad_thresh]
    good_rows = [idx + 1 for idx, r in enumerate(rows)
                 if (v := to_float(r["Defects-"])) is not None and v < good_thresh]

    # Ensure at least 5 each; if not, pad with extremes
    if len(bad_rows) < 5:
        sorted_by_defect = sorted([(idx + 1, to_float(r["Defects-"])) for idx, r in enumerate(rows)
                                  if to_float(r["Defects-"]) is not None],
                                 key=lambda x: x[1], reverse=True)
        bad_rows = [idx for idx, _ in sorted_by_defect[:5]]
    if len(good_rows) < 5:
        sorted_by_defect = sorted([(idx + 1, to_float(r["Defects-"])) for idx, r in enumerate(rows)
                                  if to_float(r["Defects-"]) is not None],
                                 key=lambda x: x[1])
        good_rows = [idx for idx, _ in sorted_by_defect[:5]]

    # Compute typical values for bad and good rows
    def typical_for_rows(rows_idx, col):
        vals = []
        for idx in rows_idx:
            r = rows[idx - 1]  # 1‑based
            if col not in r:
                continue
            v = r[col]
            if v.strip() == "?":
                continue
            # try numeric
            num = to_float(v)
            vals.append(num if num is not None else v)
        if not vals:
            return None
        return typical_value(vals)

    differences = []
    for col in non_goals:
        bad_typ = typical_for_rows(bad_rows, col)
        good_typ = typical_for_rows(good_rows, col)
        if bad_typ is None or good_typ is None:
            continue
        # Determine numeric or categorical
        if isinstance(bad_typ, (int, float)) and isinstance(good_typ, (int, float)):
            diff = abs(bad_typ - good_typ)
        else:
            diff = 0 if bad_typ == good_typ else 1
        differences.append({"feature": col, "bad": bad_typ, "good": good_typ, "diff": diff})

    differences.sort(key=lambda x: x["diff"], reverse=True)
    # Drop the helper field before output
    diff_output = [{"feature": d["feature"], "bad": d["bad"], "good": d["good"]} for d in differences]
    return {"bad": bad_rows, "good": good_rows, "differences": diff_output[:max(3, len(diff_output))]}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    if not rows:
        print("No rows in CSV.")
        sys.exit(1)

    header = rows[0].keys()
    non_goals = get_non_goal_columns(header)

    # NASA-09
    print("== NASA-09 ==")
    res09 = nasa09(rows, header, non_goals)
    print(f"Poor effort projects: {res09['poor']}")
    print(f"Top associated features: {res09['ranking']}")
    print("ANSWER NASA-09", json.dumps(res09, separators=(',', ':')))

    # NASA-10
    print("\n== NASA-10 ==")
    res10 = nasa10(rows, header, non_goals)
    print(f"Bad defect projects: {res10['bad']}")
    print(f"Good defect projects: {res10['good']}")
    print(f"Differences: {res10['differences']}")
    print("ANSWER NASA-10", json.dumps(res10, separators=(',', ':')))


if __name__ == "__main__":
    main()
