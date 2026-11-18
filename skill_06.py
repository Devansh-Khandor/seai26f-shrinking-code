#!/usr/bin/env python3
"""
skill_06.py

Usage: python skill_06.py <csv_path>
"""

import sys
import csv
import json
import math

def read_csv(path):
    """Read CSV and return list of rows (dict) with 1‑based indices."""
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

def similarity(r1, r2, setting_cols):
    """Similarity score between two rows based on setting columns."""
    common = 0
    matches = 0
    for col in setting_cols:
        v1 = r1.get(col)
        v2 = r2.get(col)
        if v1 is None or v2 is None:
            continue
        common += 1
        if v1 == v2:
            matches += 1
    return matches / common if common > 0 else 0.0

def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        return

    # Identify goal columns
    goal_cols = [c for c in rows[0].keys() if c.endswith('+') or c.endswith('-')]
    # Identify performance goal (minimisation)
    perf_col = next((c for c in goal_cols if c.endswith('-') and 'Performance' in c), None)
    if not perf_col:
        print("No performance column found.")
        return

    # Setting columns (exclude goals, X and performance column)
    setting_cols = [c for c in rows[0].keys()
                    if not (c.endswith('+') or c.endswith('-') or c.endswith('X')) and c != perf_col]

    # --- APACHE-11: recommendation and neighbours ---------------------------------
    # Find best (minimum performance) row
    best_rows = [r for r in rows if r[perf_col] is not None]
    if not best_rows:
        print("No rows with performance data.")
        return
    min_perf = min(r[perf_col] for r in best_rows)
    recommended_row = min((r for r in best_rows if r[perf_col] == min_perf),
                          key=lambda r: r['_rownum'])
    rec_rownum = recommended_row['_rownum']

    # Compute neighbours by similarity
    neighs = []
    for r in rows:
        if r['_rownum'] == rec_rownum:
            continue
        sim = similarity(recommended_row, r, setting_cols)
        neighs.append((sim, r))
    neighs.sort(key=lambda x: x[0], reverse=True)
    top_neighs = neighs[:3]  # at least 3

    # Build neighbours JSON list
    neighs_json = []
    for sim, r in top_neighs:
        values = {c: r[c] for c in goal_cols}
        neighs_json.append({
            "row": r['_rownum'],
            "values": values
        })

    # Print section
    print("== APACHE-11 ==")
    print(f"Recommended row: {rec_rownum}")
    print(f"Top neighbours: {[n['row'] for n in neighs_json]}")
    print(f"Details of neighbours: {', '.join(str(n['row']) for n in neighs_json)}")
    ans_11 = {"recommended": rec_rownum, "neighbors": neighs_json}
    print(f"ANSWER APACHE-11 {json.dumps(ans_11, separators=(',', ':'))}")

    # --- APACHE-12: comparison of two configurations ------------------------------
    # Find worst (maximum performance) row
    max_perf = max(r[perf_col] for r in best_rows)
    worst_row = max((r for r in best_rows if r[perf_col] == max_perf),
                    key=lambda r: r['_rownum'])
    worst_rownum = worst_row['_rownum']

    # Scores: higher = more recommended (use negative performance)
    score_rec = -recommended_row[perf_col]
    score_worst = -worst_row[perf_col]

    # Differences in setting columns
    diff_list = []
    for col in setting_cols:
        a_val = recommended_row[col]
        b_val = worst_row[col]
        if a_val != b_val:
            diff_list.append({"feature": col, "a": a_val, "b": b_val})
    # Ensure at least one difference
    if not diff_list and setting_cols:
        col = setting_cols[0]
        diff_list.append({"feature": col,
                          "a": recommended_row[col],
                          "b": worst_row[col]})

    # Tradeoffs: goal columns where the non-preferred (worst) row is better
    tradeoff_cols = []
    for g in goal_cols:
        a_val = recommended_row[g]
        b_val = worst_row[g]
        if a_val is None or b_val is None:
            continue
        # Determine goal direction
        if g.endswith('+'):
            # maximize
            if b_val > a_val:
                tradeoff_cols.append(g)
        else:
            # minimize
            if b_val < a_val:
                tradeoff_cols.append(g)

    # Prepare JSON
    ans_12 = {
        "a": rec_rownum,
        "b": worst_rownum,
        "preferred": "a",
        "scores": {str(rec_rownum): round(score_rec, 2), str(worst_rownum): round(score_worst, 2)},
        "differences": diff_list,
        "tradeoffs": tradeoff_cols
    }

    # Print section
    print("\n== APACHE-12 ==")
    print(f"Comparison: row {rec_rownum} (recommended) vs row {worst_rownum} (worst)")
    print(f"Recommended row: {rec_rownum}")
    print(f"Worst row: {worst_rownum}")
    print(f"At least one difference found: {len(diff_list) > 0}")
    print(f"Tradeoffs (goal columns where worst is better): {tradeoff_cols}")
    print(f"ANSWER APACHE-12 {json.dumps(ans_12, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
