#!/usr/bin/env python3
"""
skill_05.py - identify characteristics of poor effort and high defect projects

Usage:  python skill_05.py <csv path>
"""

import sys
import json
import math
import ezr


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_csv(path):
    """Return header and list of rows."""
    rows = list(ezr.csv(path))
    header = rows[0]
    data = rows[1:]
    return header, data


def mean_and_sd(values):
    """Return mean and standard deviation of a list of floats."""
    n = len(values)
    if n == 0:
        return None, None
    s = sum(values)
    m = s / n
    var = sum((x - m) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    return m, sd


def get_rows_above_threshold(rows, idx, threshold, min_count=5):
    """Return row numbers whose value at idx is > threshold.
       If fewer than min_count, return top min_count rows by value."""
    good = []
    for i, row in enumerate(rows):
        val = row[idx]
        if val == "?":
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        if v > threshold:
            good.append(i + 1)
    if len(good) < min_count:
        # fallback to top min_count rows
        sorted_indices = sorted(
            [i for i, r in enumerate(rows) if r[idx] != "?"],
            key=lambda i: float(rows[i][idx]),
            reverse=True,
        )[:min_count]
        good = [i + 1 for i in sorted_indices]
    return good


def get_rows_below_threshold(rows, idx, threshold, min_count=5):
    """Return row numbers whose value at idx is < threshold.
       If fewer than min_count, return bottom min_count rows by value."""
    bad = []
    for i, row in enumerate(rows):
        val = row[idx]
        if val == "?":
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        if v < threshold:
            bad.append(i + 1)
    if len(bad) < min_count:
        sorted_indices = sorted(
            [i for i, r in enumerate(rows) if r[idx] != "?"],
            key=lambda i: float(rows[i][idx]),
            reverse=False,
        )[:min_count]
        bad = [i + 1 for i in sorted_indices]
    return bad


def most_common(values):
    """Return the most common value in a list (ties arbitrary)."""
    freq = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    return max(freq, key=freq.get)


def mode_proportion(values, mode):
    """Return proportion of values equal to mode."""
    if not values:
        return 0.0
    return sum(1 for v in values if v == mode) / len(values)


# ----------------------------------------------------------------------
# NASA-09: poor effort projects
# ----------------------------------------------------------------------
def nasa_09(header, rows):
    effort_idx = header.index("Effort-")
    # collect numeric effort values
    effort_vals = [float(r[effort_idx]) for r in rows if r[effort_idx] != "?"]
    mean_eff, sd_eff = mean_and_sd(effort_vals)
    threshold = mean_eff + 2 * sd_eff if sd_eff else mean_eff

    poor_rows = get_rows_above_threshold(rows, effort_idx, threshold, min_count=5)

    # Ranking of non‑goal columns
    non_goal_cols = [c for c in header if not c.endswith(("+", "-")) and not c.endswith("X")]
    all_vals = {c: [] for c in non_goal_cols}
    poor_vals = {c: [] for c in non_goal_cols}

    for r in rows:
        for c in non_goal_cols:
            val = r[header.index(c)]
            if val != "?":
                all_vals[c].append(val)

    for r in rows:
        if r[effort_idx] == "?":
            continue
        val_eff = float(r[effort_idx])
        if val_eff > threshold:  # poor
            for c in non_goal_cols:
                val = r[header.index(c)]
                if val != "?":
                    poor_vals[c].append(val)

    diff_scores = []

    for c in non_goal_cols:
        vals_all = all_vals[c]
        vals_poor = poor_vals[c]
        if not vals_all or not vals_poor:
            continue
        col_obj = ezr.Col(txt=c, at=0)  # dummy; type inferred by first val
        if isinstance(col_obj, ezr.Sym):
            # symbol column
            mode_all = most_common(vals_all)
            mode_poor = most_common(vals_poor)
            diff = abs(
                mode_proportion(vals_all, mode_all) - mode_proportion(vals_poor, mode_poor)
            )
        else:
            # numeric
            vals_all_f = [float(v) for v in vals_all]
            vals_poor_f = [float(v) for v in vals_poor]
            m_all, _ = mean_and_sd(vals_all_f)
            m_poor, _ = mean_and_sd(vals_poor_f)
            diff = abs(m_poor - m_all)
        diff_scores.append((diff, c))

    diff_scores.sort(reverse=True)
    ranking = [c for _, c in diff_scores[:3]]

    # Output
    print("== NASA-09 ==")
    print(f"Poor Effort- rows: {poor_rows}")
    print(f"Ranking of associated columns: {ranking}")
    print(
        f"ANSWER NASA-09 "
        f"{json.dumps({'poor': poor_rows, 'ranking': ranking}, separators=(',', ':'))}"
    )


# ----------------------------------------------------------------------
# NASA-10: high defect projects vs low defect projects
# ----------------------------------------------------------------------
def nasa_10(header, rows):
    defect_idx = header.index("Defects-")
    defect_vals = [float(r[defect_idx]) for r in rows if r[defect_idx] != "?"]
    mean_def, sd_def = mean_and_sd(defect_vals)
    high_thr = mean_def + 2 * sd_def if sd_def else mean_def
    low_thr = mean_def - 2 * sd_def if sd_def else mean_def

    bad_rows = get_rows_above_threshold(rows, defect_idx, high_thr, min_count=5)
    good_rows = get_rows_below_threshold(rows, defect_idx, low_thr, min_count=5)

    # Compute differences
    non_goal_cols = [c for c in header if not c.endswith(("+", "-")) and not c.endswith("X")]
    diffs = []

    for c in non_goal_cols:
        vals_bad = []
        vals_good = []
        for r in rows:
            val = r[header.index(c)]
            if val == "?":
                continue
            if r[defect_idx] == "?":
                continue
            defect_val = float(r[defect_idx])
            if defect_val > high_thr:
                vals_bad.append(val)
            elif defect_val < low_thr:
                vals_good.append(val)
        if not vals_bad or not vals_good:
            continue
        if isinstance(ezr.Col(txt=c, at=0), ezr.Sym):
            mode_bad = most_common(vals_bad)
            mode_good = most_common(vals_good)
            diff = abs(
                mode_proportion(vals_bad, mode_bad) - mode_proportion(vals_good, mode_good)
            )
            bad_val = mode_bad
            good_val = mode_good
        else:
            vals_bad_f = [float(v) for v in vals_bad]
            vals_good_f = [float(v) for v in vals_good]
            bad_val = sum(vals_bad_f) / len(vals_bad_f)
            good_val = sum(vals_good_f) / len(vals_good_f)
            diff = abs(bad_val - good_val)
        diffs.append((diff, {"feature": c, "bad": bad_val, "good": good_val}))

    diffs.sort(reverse=True)
    differences = [d for _, d in diffs[:3]]

    # Output
    print("== NASA-10 ==")
    print(f"Bad Defects- rows: {bad_rows}")
    print(f"Good Defects- rows: {good_rows}")
    print(f"Differences: {differences}")
    print(
        f"ANSWER NASA-10 "
        f"{json.dumps({'bad': bad_rows, 'good': good_rows, 'differences': differences}, separators=(',', ':'))}"
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    header, rows = read_csv(path)

    nasa_09(header, rows)
    nasa_10(header, rows)


if __name__ == "__main__":
    main()
