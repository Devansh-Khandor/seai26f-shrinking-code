#!/usr/bin/env python3
"""
skill_01.py

This script reads the provided Apache measurements CSV and answers two requests:

APACHE-01: Summary statistics for each goal column.
APACHE-02: Identification of unusually favorable configurations and the settings that distinguish them.

The script prints a heading for each request, a brief textual summary, and a machine‑checked answer line
containing JSON that conforms to the specification.
"""

import sys
import csv
import json
import numpy as np

def read_csv(path):
    """
    Reads the CSV file and returns a list of row dictionaries.
    Missing values are kept as the string '?'.
    """
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames

def parse_column_values(rows, column):
    """
    Convert column values to floats where possible, keep '?' as None.
    Returns a NumPy array of floats with np.nan for missing values.
    """
    vals = []
    for r in rows:
        v = r[column]
        if v == '?':
            vals.append(np.nan)
        else:
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(np.nan)
    return np.array(vals, dtype=float)

def compute_stats(values):
    """
    Compute mean, median, min, max for a 1‑D NumPy array that may contain NaNs.
    Returns a dict with plain numbers (no NaN).
    """
    mask = ~np.isnan(values)
    if mask.sum() == 0:
        # No valid data; return None values
        return {"mean": None, "median": None, "min": None, "max": None}
    vals = values[mask]
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals))
    }

def main(csv_path):
    rows, headers = read_csv(csv_path)

    # Identify goal columns: names ending with '+' or '-'
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]

    # Compute summary stats for each goal column
    print("== APACHE-01 ==")
    stats_dict = {}
    for col in goal_cols:
        vals = parse_column_values(rows, col)
        s = compute_stats(vals)
        # best / worst depends on goal direction
        if col.endswith('-'):
            best = s["min"]
            worst = s["max"]
        else:
            best = s["max"]
            worst = s["min"]
        s.update({"best": best, "worst": worst})
        stats_dict[col] = s
        # Textual summary
        print(f"{col}:")
        print(f"  mean={s['mean']:.6g}")
        print(f"  median={s['median']:.6g}")
        print(f"  min={s['min']:.6g}")
        print(f"  max={s['max']:.6g}")
        print(f"  best={s['best']:.6g}")
        print(f"  worst={s['worst']:.6g}")

    # Machine‑checked answer line
    print("ANSWER APACHE-01", json.dumps(stats_dict, separators=(',', ':')))

    # ---------- APACHE-02 ----------
    print("\n== APACHE-02 ==")

    # Determine top (unusually favorable) configurations
    perf_col = goal_cols[0]  # only one: Performance-
    perf_vals = parse_column_values(rows, perf_col)
    valid_mask = ~np.isnan(perf_vals)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        top_rows = []
    else:
        sorted_indices = valid_indices[np.argsort(perf_vals[valid_mask])]
        n_rows = len(sorted_indices)
        n_top = min(max(int(round(n_rows * 0.10)), 5), 20)
        n_top = min(n_top, n_rows)
        top_indices = sorted_indices[:n_top]
        top_rows = [i + 1 for i in top_indices]  # 1‑based row numbers

    print("Top configurations:", ', '.join(map(str, top_rows)))

    # Settings to analyze
    setting_cols = [h for h in headers if h not in goal_cols]
    # For each setting, compute share of 1s in top and rest
    setting_stats = []
    for col in setting_cols:
        col_vals = rows
        top_1 = 0
        top_total = 0
        rest_1 = 0
        rest_total = 0
        for i, r in enumerate(col_vals):
            val = r[col]
            if val == '?':
                # treat as 0, count as total
                is_one = False
            else:
                is_one = val == '1'
            if (i + 1) in top_rows:
                top_total += 1
                if is_one:
                    top_1 += 1
            else:
                rest_total += 1
                if is_one:
                    rest_1 += 1
        top_share = top_1 / top_total if top_total else 0.0
        rest_share = rest_1 / rest_total if rest_total else 0.0
        setting_stats.append({
            "setting": col,
            "top_share": top_share,
            "rest_share": rest_share,
            "diff": abs(top_share - rest_share)
        })

    # Sort by difference descending
    setting_stats.sort(key=lambda x: x["diff"], reverse=True)
    top_settings = setting_stats[:5]  # at least 2, output 5 for clarity

    print("Settings distinguishing top configurations:")
    for s in top_settings:
        print(f"  {s['setting']}: top_share={s['top_share']:.4f}, rest_share={s['rest_share']:.4f}")

    # Machine‑checked answer line
    ans_settings = [{"setting": s["setting"], "top_share": s["top_share"], "rest_share": s["rest_share"]} for s in top_settings]
    print("ANSWER APACHE-02", json.dumps({"top": top_rows, "settings": ans_settings}, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
