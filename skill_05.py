#!/usr/bin/env python3
"""
skill_05.py <csv path>

Implements:
    NASA-09: Identify projects with unusually high Effort- and the
             characteristics most strongly associated with those outcomes.

    NASA-10: Identify projects with unusually high Defects- and
             compare their characteristics to projects with good quality.
"""

import csv
import json
import math
import sys
from collections import Counter, defaultdict

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
def to_float(v):
    """Convert string to float; return None if missing."""
    return None if v == '?' else float(v)

def is_numeric_col(col, rows):
    """Return True if all non‑missing values in col can be parsed as float."""
    for r in rows:
        val = r.get(col, '?')
        if val == '?':
            continue
        try:
            float(val)
        except ValueError:
            return False
    return True

def compute_mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)

def compute_std(values, mean_val):
    if len(values) < 2:
        return 0.0
    var = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)

def pearson_corr(xs, ys):
    """Return Pearson correlation coefficient between two equal‑length lists."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sd_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    sd_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if sd_x == 0 or sd_y == 0:
        return 0.0
    return cov / (sd_x * sd_y)

def mode_counter(counter):
    """Return the most common item and its count from a Counter."""
    if not counter:
        return None, 0
    return counter.most_common(1)[0]

# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------
def main(csv_path):
    # Load rows
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]

    # Map row numbers (1‑based)
    row_nums = list(range(1, len(rows) + 1))

    # Identify non‑goal columns (exclude X and +/-)
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    non_goal_cols = [c for c in header if not c.endswith('X') and c not in goal_cols]

    # Determine numeric vs categorical columns
    numeric_cols = [c for c in non_goal_cols if is_numeric_col(c, rows)]
    categorical_cols = [c for c in non_goal_cols if c not in numeric_cols]

    # Gather Effort- and Defects- values
    effort_vals = [to_float(r['Effort-']) for r in rows]
    effort_vals = [v for v in effort_vals if v is not None]
    defect_vals = [to_float(r['Defects-']) for r in rows]
    defect_vals = [v for v in defect_vals if v is not None]

    # Compute overall statistics
    effort_mean = compute_mean(effort_vals)
    effort_std = compute_std(effort_vals, effort_mean)
    defect_mean = compute_mean(defect_vals)
    defect_std = compute_std(defect_vals, defect_mean)

    # ------------------------------------------------------------------
    # NASA-09: Poor Effort
    # ------------------------------------------------------------------
    poor_thresh = effort_mean + 2 * effort_std
    poor_rows = [rn for rn, val in zip(row_nums, effort_vals) if val > poor_thresh]

    # Ensure at least 5 rows
    if len(poor_rows) < 5:
        # Take top 5 worst
        sorted_by_effort = sorted(
            [(rn, val) for rn, val in zip(row_nums, effort_vals) if val is not None],
            key=lambda x: -x[1]
        )
        poor_rows = [rn for rn, _ in sorted_by_effort[:5]]

    # Ranking of features most associated with poor effort
    feature_scores = {}
    for col in non_goal_cols:
        # Gather pairs of (feature_value, effort)
        vals = []
        effs = []
        for row in rows:
            fv = row[col]
            ev = to_float(row['Effort-'])
            if fv == '?' or ev is None:
                continue
            if col in numeric_cols:
                try:
                    fv_f = float(fv)
                except ValueError:
                    continue
                vals.append(fv_f)
                effs.append(ev)
            else:
                vals.append(fv)
                effs.append(ev)
        if not vals:
            continue
        if col in numeric_cols:
            corr = abs(pearson_corr(vals, effs))
            feature_scores[col] = corr
        else:
            # For categorical, compute mean Effort- per category
            cat_means = {}
            for v, e in zip(vals, effs):
                cat_means.setdefault(v, []).append(e)
            mean_diffs = [abs(compute_mean(es) - effort_mean) for es in cat_means.values()]
            if mean_diffs:
                feature_scores[col] = max(mean_diffs)
    # Sort features by score descending
    ranked_features = sorted(feature_scores.items(), key=lambda x: -x[1])
    ranking = [f for f, _ in ranked_features[:3]]  # at least 3

    # ------------------------------------------------------------------
    # NASA-10: Bad vs Good Defects
    # ------------------------------------------------------------------
    # Bad rows: high defects (mean + 2*std)
    bad_thresh = defect_mean + 2 * defect_std
    bad_rows = [rn for rn, val in zip(row_nums, defect_vals) if val > bad_thresh]
    if len(bad_rows) < 5:
        # top 5 worst
        sorted_by_defect = sorted(
            [(rn, val) for rn, val in zip(row_nums, defect_vals) if val is not None],
            key=lambda x: -x[1]
        )
        bad_rows = [rn for rn, _ in sorted_by_defect[:5]]

    # Good rows: low defects (mean - 2*std)
    good_thresh = defect_mean - 2 * defect_std
    good_rows = [rn for rn, val in zip(row_nums, defect_vals) if val < good_thresh]
    if len(good_rows) < 5:
        sorted_by_defect = sorted(
            [(rn, val) for rn, val in zip(row_nums, defect_vals) if val is not None],
            key=lambda x: x[1]
        )
        good_rows = [rn for rn, _ in sorted_by_defect[:5]]

    # Typical values for bad and good rows per feature
    def typical_values(row_indices, is_bad):
        typ = {}
        for col in non_goal_cols:
            values = []
            for rn in row_indices:
                val = rows[rn-1][col]
                if val == '?':
                    continue
                values.append(val)
            if not values:
                continue
            if col in numeric_cols:
                fvals = [float(v) for v in values]
                typ[col] = compute_mean(fvals)
            else:
                freq = Counter(values)
                m, _ = mode_counter(freq)
                typ[col] = m
        return typ

    bad_typ = typical_values(bad_rows, True)
    good_typ = typical_values(good_rows, False)

    # Compute differences
    diff_list = []
    for col in non_goal_cols:
        if col not in bad_typ or col not in good_typ:
            continue
        if col in numeric_cols:
            diff = abs(bad_typ[col] - good_typ[col])
        else:
            # categorical difference: use proportion difference of mode
            # compute proportion of mode in bad and good sets
            def mode_prop(indices, typ_val):
                total = 0
                matches = 0
                for rn in indices:
                    val = rows[rn-1][col]
                    if val == '?':
                        continue
                    total += 1
                    if val == typ_val:
                        matches += 1
                return matches / total if total else 0.0
            p_bad = mode_prop(bad_rows, bad_typ[col])
            p_good = mode_prop(good_rows, good_typ[col])
            diff = abs(p_bad - p_good)
        diff_list.append({"feature": col, "bad": bad_typ[col], "good": good_typ[col], "diff": diff})

    # Sort by diff descending, take top 3
    diff_list.sort(key=lambda x: -x["diff"])
    differences = [{"feature": d["feature"], "bad": d["bad"], "good": d["good"]} for d in diff_list[:3]]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    # NASA-09
    print("== NASA-09 ==")
    print(f"Unusually high Effort- rows (threshold: {poor_thresh:.2f}):")
    print(f"  Rows: {poor_rows}")
    print(f"Top 3 features most associated with high effort:")
    for feat in ranking:
        print(f"  {feat}")
    print(f"ANSWER NASA-09 {json.dumps({'poor': poor_rows, 'ranking': ranking})}")

    # NASA-10
    print("\n== NASA-10 ==")
    print(f"Bad defect rows (threshold: {bad_thresh:.2f}): {bad_rows}")
    print(f"Good defect rows (threshold: {good_thresh:.2f}): {good_rows}")
    print("Top 3 distinguishing feature differences:")
    for d in differences:
        print(f"  {d['feature']}: bad={d['bad']}, good={d['good']}")
    print(f"ANSWER NASA-10 {json.dumps({'bad': bad_rows, 'good': good_rows, 'differences': differences})}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
