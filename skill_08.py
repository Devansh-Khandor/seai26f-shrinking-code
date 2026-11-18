#!/usr/bin/env python3
"""
skill_08.py

Usage: python skill_08.py <csv_path>
"""

import sys
import csv
import json
import math
from statistics import mean, stdev, StatisticsError

# ------------------------------------------------------------------
# CSV reading helper
# ------------------------------------------------------------------
def read_csv(path):
    """Read CSV and return a list of row dicts with 1‑based row numbers
    added under the key '_rownum'. Cells containing '?' are converted to None."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=1):
            row = {}
            for k, v in raw.items():
                if v == '?':
                    row[k] = None
                else:
                    try:
                        row[k] = int(v)
                    except ValueError:
                        row[k] = v
            row['_rownum'] = idx
            rows.append(row)
    return rows

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def performance_col(rows):
    """Return the column name that contains the performance metric."""
    for name in rows[0].keys():
        if name.endswith('-') and 'Performance' in name:
            return name
    return None

def setting_cols(rows, perf_col):
    """Return a list of setting column names."""
    return [c for c in rows[0].keys()
            if not (c.endswith('+') or c.endswith('-') or c.endswith('X')) and c != perf_col]

# ------------------------------------------------------------------
# Compute poor rows (mean + 1.5*std, at least 5)
# ------------------------------------------------------------------
def poor_rows(rows, perf_col):
    vals = [r[perf_col] for r in rows if r[perf_col] is not None]
    if not vals:
        return []
    try:
        m = mean(vals)
        s = stdev(vals)
    except StatisticsError:
        # All values identical
        m = vals[0]
        s = 0.0
    threshold = m + 1.5 * s
    bad = [r for r in rows if r[perf_col] is not None and r[perf_col] > threshold]
    if len(bad) < 5:
        bad = sorted([r for r in rows if r[perf_col] is not None],
                     key=lambda r: r[perf_col], reverse=True)[:5]
    return bad

# ------------------------------------------------------------------
# APACHE-15
# ------------------------------------------------------------------
def apache15(rows, perf_col, set_cols):
    """Find a poor row and recommend a small change to a better row."""
    bad = poor_rows(rows, perf_col)
    if not bad:
        return None
    # choose worst among poor rows
    poor_row = max(bad, key=lambda r: r[perf_col])
    poor_num = poor_row['_rownum']
    poor_perf = poor_row[perf_col]

    best_alt = None
    best_improvement = -math.inf
    best_diff_count = None
    best_rownum = None

    for r in rows:
        if r['_rownum'] == poor_num:
            continue
        alt_perf = r.get(perf_col)
        if alt_perf is None or alt_perf >= poor_perf:
            continue
        # count differing settings
        diff = []
        for col in set_cols:
            if poor_row[col] != r[col]:
                diff.append(col)
        if len(diff) == 0 or len(diff) > 2:
            continue
        improvement = poor_perf - alt_perf
        if (improvement > best_improvement or
            (improvement == best_improvement and (len(diff) < best_diff_count or
             (len(diff) == best_diff_count and r['_rownum'] < best_rownum)))):
            best_improvement = improvement
            best_alt = r
            best_diff_count = len(diff)
            best_rownum = r['_rownum']

    if best_alt is None:
        # fallback: pick the best overall row with at least 1 diff
        candidates = [r for r in rows if r['_rownum'] != poor_num]
        best_alt = min(candidates, key=lambda r: r[perf_col])
        best_rownum = best_alt['_rownum']
        diff = [col for col in set_cols if best_alt[col] != poor_row[col]]
        best_diff_count = len(diff)
        changes = {col: [poor_row[col], best_alt[col]] for col in diff}
    else:
        diff = [col for col in set_cols if best_alt[col] != poor_row[col]]
        changes = {col: [poor_row[col], best_alt[col]] for col in diff}

    answer = {
        "row": poor_num,
        "alternative": best_rownum,
        "changes": changes,
        "performance": {
            "before": poor_perf,
            "after": best_alt[perf_col]
        }
    }
    return answer

# ------------------------------------------------------------------
# APACHE-16
# ------------------------------------------------------------------
def apache16(rows, perf_col, set_cols):
    """Find minimal changes to move a poor row closer to the best performance."""
    bad = poor_rows(rows, perf_col)
    if not bad:
        return None
    poor_row = max(bad, key=lambda r: r[perf_col])
    poor_num = poor_row['_rownum']
    poor_perf = poor_row[perf_col]

    # best performance (target)
    all_perf = [r[perf_col] for r in rows if r[perf_col] is not None]
    if not all_perf:
        return None
    target_perf = min(all_perf)

    # Find rows that are closer to target than poor row
    # Since target is min, closer means lower perf
    candidates = [r for r in rows if r[perf_col] is not None and r[perf_col] < poor_perf]
    if not candidates:
        # if none, use best row
        candidates = [r for r in rows if r[perf_col] is not None]

    # Choose candidate with fewest differing settings; tie by rownum
    best_candidate = None
    best_diff = None
    best_rownum = None

    for r in candidates:
        diff = [col for col in set_cols if r[col] != poor_row[col]]
        dlen = len(diff)
        if best_candidate is None or dlen < best_diff or (dlen == best_diff and r['_rownum'] < best_rownum):
            best_candidate = r
            best_diff = dlen
            best_rownum = r['_rownum']

    changes = [col for col in set_cols if best_candidate[col] != poor_row[col]]

    answer = {
        "row": poor_num,
        "target": target_perf,
        "alternative": best_rownum,
        "changed": changes
    }
    return answer

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        return

    perf_col = performance_col(rows)
    if not perf_col:
        print("No performance column found.")
        return

    set_cols = setting_cols(rows, perf_col)

    # --- APACHE-15 ---
    print("== APACHE-15 ==")
    ans15 = apache15(rows, perf_col, set_cols)
    if ans15:
        print(f"Poor row: {ans15['row']}")
        print(f"Alternative row: {ans15['alternative']}")
        print(f"Changes: {ans15['changes']}")
        print(f"Performance before/after: {ans15['performance']}")
        print(f"ANSWER APACHE-15 {json.dumps(ans15, separators=(',', ':'))}")
    else:
        print("No suitable poor row found.")
        print("ANSWER APACHE-15 {}")

    # --- APACHE-16 ---
    print("\n== APACHE-16 ==")
    ans16 = apache16(rows, perf_col, set_cols)
    if ans16:
        print(f"Poor row: {ans16['row']}")
        print(f"Target performance: {ans16['target']}")
        print(f"Alternative row: {ans16['alternative']}")
        print(f"Changed settings: {ans16['changed']}")
        print(f"ANSWER APACHE-16 {json.dumps(ans16, separators=(',', ':'))}")
    else:
        print("No suitable poor row found.")
        print("ANSWER APACHE-16 {}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
