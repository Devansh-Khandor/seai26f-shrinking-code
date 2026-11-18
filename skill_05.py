#!/usr/bin/env python3
"""
Skill 05: Analysis of poor fuel efficiency (CAR‑09) and overall performance (CAR‑10).

The script accepts a single CSV path and prints the required output for CAR‑09 and CAR‑10
according to the specification.  The implementation follows the same conventions as the
previous skill modules: missing values are represented by None, row numbers are 1‑based,
and the CSV header is not considered a data row.

Output format:
    == CAR‑09 ==
    ... (at least three lines of human readable results)
    ANSWER CAR‑09 <JSON>
    == CAR‑10 ==
    ... (at least three lines of human readable results)
    ANSWER CAR‑10 <JSON>
"""

import sys
import csv
import json
import math
import statistics

# ----------------------------------------------------------------------
# CSV utilities
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Read CSV file.  Return a list of (row_index, row_dict) tuples.
    row_index is 1‑based and refers to the position among the data rows.
    Missing values '?' are converted to None.
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(reader, start=1):
            parsed = {}
            for k, v in line.items():
                if v == '?' or v is None:
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = None
            rows.append((i, parsed))
    return rows

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_input(col):
    # Input columns are those that are not goals and not ending with 'X'
    return not is_goal(col) and not col.endswith('X')

def pearson_correlation(xs, ys):
    """Return Pearson correlation coefficient for two numeric lists."""
    if len(xs) < 2:
        return 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    return cov / denom if denom != 0 else 0.0

def compute_quartile_threshold(sorted_vals, n, quarter=0.25):
    """Return the value at the given quartile (0.25 for lower, 0.75 for upper)."""
    idx = int(math.ceil(n * quarter)) - 1
    if idx < 0:
        idx = 0
    if idx >= n:
        idx = n - 1
    return sorted_vals[idx]

# ----------------------------------------------------------------------
# CAR-09 implementation
# ----------------------------------------------------------------------
def car09(rows):
    """Return dict with keys 'poor' and 'ranking'."""
    # Extract Mpg+ values
    mpg_col = 'Mpg+'
    mpg_vals = [(i, r[mpg_col]) for i, r in rows if r[mpg_col] is not None]
    mpg_values = [v for _, v in mpg_vals]
    mpg_values.sort()
    n = len(mpg_values)
    # 25th percentile threshold
    k = max(5, int(math.ceil(n * 0.25)))
    threshold = compute_quartile_threshold(mpg_values, n, 0.25)

    # Identify poor rows (<= threshold), ensure at least 5
    poor_rows = [i for i, v in mpg_vals if v <= threshold]
    if len(poor_rows) < 5:
        # Add next higher rows until 5
        sorted_pairs = sorted(mpg_vals, key=lambda x: x[1])
        for i, v in sorted_pairs:
            if i not in poor_rows:
                poor_rows.append(i)
            if len(poor_rows) >= 5:
                break

    # Correlation ranking
    feature_cols = [k for k, v in rows[0][1].items() if is_input(k)]
    correlations = {}
    for col in feature_cols:
        xs = []
        ys = []
        for _, rec in rows:
            x = rec[col]
            y = rec[mpg_col]
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        corr = pearson_correlation(xs, ys)
        correlations[col] = corr

    # Sort by absolute correlation descending
    ranked = sorted(correlations.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    ranking = [col for col, _ in ranked[:3]]  # at least 3

    return {"poor": sorted(poor_rows), "ranking": ranking}

# ----------------------------------------------------------------------
# CAR-10 implementation
# ----------------------------------------------------------------------
def compute_overall_score(rows):
    """Compute composite score for each row; return dict row_index -> score."""
    goal_cols = ['Lbs-', 'Acc+', 'Mpg+']
    # Determine global min and max for each goal
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [r[col] for _, r in rows if r[col] is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0

    scores = {}
    for idx, rec in rows:
        if any(rec[col] is None for col in goal_cols):
            continue
        # Normalization
        norm_weight = ((maxs["Lbs-"] - rec["Lbs-"]) / (maxs["Lbs-"] - mins["Lbs-"])) if maxs["Lbs-"] != mins["Lbs-"] else 0.0
        norm_acc = ((rec["Acc+"] - mins["Acc+"]) / (maxs["Acc+"] - mins["Acc+"])) if maxs["Acc+"] != mins["Acc+"] else 0.0
        norm_mpg = ((rec["Mpg+"] - mins["Mpg+"]) / (maxs["Mpg+"] - mins["Mpg+"])) if maxs["Mpg+"] != mins["Mpg+"] else 0.0
        scores[idx] = norm_weight + norm_acc + norm_mpg
    return scores

def car10(rows):
    """Return dict with keys 'bad', 'good', 'differences'."""
    scores = compute_overall_score(rows)
    score_items = sorted(scores.items(), key=lambda kv: kv[1])  # ascending
    n_scores = len(score_items)
    k = max(5, int(math.ceil(n_scores * 0.25)))

    # Bad rows: lowest k scores
    bad_rows = [idx for idx, _ in score_items[:k]]
    # Good rows: highest k scores
    good_rows = [idx for idx, _ in score_items[-k:]]

    # Typical values (mean) for non‑goal columns
    non_goal_cols = [k for k, v in rows[0][1].items() if is_input(k)]
    def typical(vals):
        if not vals:
            return None
        return sum(vals) / len(vals)

    bad_vals = {col: [] for col in non_goal_cols}
    good_vals = {col: [] for col in non_goal_cols}

    for idx, rec in rows:
        if idx in bad_rows and all(rec[col] is not None for col in non_goal_cols):
            for col in non_goal_cols:
                bad_vals[col].append(rec[col])
        if idx in good_rows and all(rec[col] is not None for col in non_goal_cols):
            for col in non_goal_cols:
                good_vals[col].append(rec[col])

    bad_templ = {col: typical(bad_vals[col]) for col in non_goal_cols}
    good_templ = {col: typical(good_vals[col]) for col in non_goal_cols}

    # Differences: absolute mean difference
    diffs = []
    for col in non_goal_cols:
        bad_mean = bad_templ[col]
        good_mean = good_templ[col]
        if bad_mean is None or good_mean is None:
            diff = None
        else:
            diff = abs(bad_mean - good_mean)
        diffs.append((col, diff, bad_mean, good_mean))

    diffs_sorted = sorted([d for d in diffs if d[1] is not None], key=lambda d: -d[1])
    diff_list = [{"feature": col, "bad": bad_mean, "good": good_mean}
                 for col, _, bad_mean, good_mean in diffs_sorted[:3]]  # at least 3

    return {
        "bad": sorted(bad_rows),
        "good": sorted(good_rows),
        "differences": diff_list
    }

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)

    # CAR-09
    print("== CAR-09 ==")
    result09 = car09(rows)
    # Human‑readable lines
    print(f"Poor fuel‑efficiency rows (Mpg+ <= 25th percentile): {', '.join(map(str, result09['poor']))}")
    print(f"Ranking of characteristics most negatively correlated with Mpg+: {', '.join(result09['ranking'])}")
    # Machine‑checked answer
    print("ANSWER CAR-09 " + json.dumps(result09, separators=(',', ':')))

    # CAR-10
    print("== CAR-10 ==")
    result10 = car10(rows)
    print(f"Unfavorable overall performance rows (bottom quartile): {', '.join(map(str, result10['bad']))}")
    print(f"Better overall performance rows (top quartile): {', '.join(map(str, result10['good']))}")
    print("Differences between bad and good rows:")
    for d in result10['differences']:
        print(f"  Feature {d['feature']}: bad={d['bad']:.4f}, good={d['good']:.4f}")
    print("ANSWER CAR-10 " + json.dumps(result10, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_05.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
