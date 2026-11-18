#!/usr/bin/env python3
"""
Skill 01 – Apache performance analytics

Usage:
    python skill_01.py <csv_path>
"""

import sys
import json
import math
import ezr

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    tbl = ezr.Tbl(ezr.csv(csv_path))
    rows = tbl.rows
    n_rows = len(rows)

    # Identify the goal column (ends with '+' or '-')
    goal_cols = [c for c in tbl.cols.values() if c.txt.endswith(('+', '-'))]
    if not goal_cols:
        raise RuntimeError("No goal column found.")
    goal_col = goal_cols[0]
    goal_at = goal_col.at

    # Gather performance values and row numbers (1‑based)
    perf_vals = []
    row_nums = []
    for i, row in enumerate(rows, start=1):
        v = row[goal_at]
        if v == "?":
            continue
        perf_vals.append(float(v))
        row_nums.append(i)

    if not perf_vals:
        print("No performance values to analyze.")
        return

    # Basic statistics
    count = len(perf_vals)
    mean = sum(perf_vals) / count
    sorted_vals = sorted(perf_vals)
    m = len(sorted_vals)
    if m % 2 == 1:
        median = sorted_vals[m // 2]
    else:
        median = (sorted_vals[m // 2 - 1] + sorted_vals[m // 2]) / 2.0
    min_val = min(perf_vals)
    max_val = max(perf_vals)
    best = min_val          # minimised goal
    worst = max_val

    # ---------- APACHE-01 ----------
    print("== APACHE-01 ==")
    print(f"Performance-: mean={mean:.2f} median={median:.2f} min={min_val} max={max_val} best={best} worst={worst}")
    print(f"Rows analysed: {count}")

    # ---------- APACHE-02 ----------
    # Define “unusually favourable” as the lowest 5 % (rounded up)
    k = max(1, math.ceil(count * 0.05))
    sorted_by_perf = sorted(zip(perf_vals, row_nums), key=lambda x: x[0])
    top_perf = sorted_by_perf[:k]
    top_rows = [rnum for _, rnum in top_perf]
    top_set = set(top_rows)

    # Settings: all columns that are not goal columns (and not 'X' columns)
    setting_cols = [
        c for c in tbl.cols.values()
        if not c.txt.endswith(('+', '-', 'X'))
    ]

    # Compute share of '1' in top rows vs. the rest
    settings_info = []
    for col in setting_cols:
        at = col.at
        top_ones = top_tot = rest_ones = rest_tot = 0
        for i, row in enumerate(rows, start=1):
            v = row[at]
            if v == "?":
                continue
            if i in top_set:
                if v == 1:
                    top_ones += 1
                top_tot += 1
            else:
                if v == 1:
                    rest_ones += 1
                rest_tot += 1
        top_share = top_ones / top_tot if top_tot else 0.0
        rest_share = rest_ones / rest_tot if rest_tot else 0.0
        settings_info.append(
            {
                "setting": col.txt,
                "top_share": top_share,
                "rest_share": rest_share,
            }
        )

    # Pick the most distinguishing settings (at least two)
    settings_info.sort(key=lambda s: abs(s["top_share"] - s["rest_share"]), reverse=True)
    top_settings = settings_info[: max(2, min(20, len(settings_info)))]

    print("== APACHE-02 ==")
    print(f"Top {k} rows (by performance): {top_rows}")
    print("Top distinguishing settings:")
    for s in top_settings:
        print(f"{s['setting']}: top={s['top_share']:.2f} rest={s['rest_share']:.2f}")

    # ---------- Machine‑checked answers ----------
    print(
        f"ANSWER APACHE-01 {json.dumps({'Performance-': {'mean': mean, 'median': median, 'min': min_val, 'max': max_val, 'best': best, 'worst': worst}})}"
    )
    print(
        f"ANSWER APACHE-02 {json.dumps({'top': top_rows, 'settings': top_settings})}"
    )


if __name__ == "__main__":
    main()
