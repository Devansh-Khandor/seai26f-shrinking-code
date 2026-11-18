#!/usr/bin/env python3
"""
skill_04.py

Usage: python skill_04.py <csv path>
"""

import csv
import json
import math
import sys
from collections import defaultdict
from statistics import mean

# Use ezr's atom function to parse values
try:
    from ezr import atom
except ImportError:
    # minimal implementation if ezr not available
    def atom(s: str):
        for fn in (int, float):
            try:
                return fn(s)
            except ValueError:
                pass
        return {"True": True, "False": False}.get(s.strip(), s.strip())

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def parse_rows(csv_path):
    """Read CSV and return header and parsed rows."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = [[atom(v) for v in row] for row in rdr]
    return header, rows

def is_numeric(v):
    return isinstance(v, (int, float))

def train_linear(rows, target_idx, pred_idx):
    """Return intercept, slope, mean_y, sd_y, count."""
    xs, ys = [], []
    for r in rows:
        x, y = r[pred_idx], r[target_idx]
        if is_numeric(x) and is_numeric(y):
            xs.append(float(x))
            ys.append(float(y))
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    mx = mean(xs)
    my = mean(ys)
    # slope
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(xs, ys))
    den = sum((xi - mx) ** 2 for xi in xs)
    slope = num / den if den != 0 else 0.0
    intercept = my - slope * mx
    # std dev of y
    sd_y = math.sqrt(sum((yi - my) ** 2 for yi in ys) / n)
    return intercept, slope, my, sd_y, n

def predict_value(intercept, slope, x, default):
    if x is None or not is_numeric(x):
        return default
    return intercept + slope * float(x)

def round_num(v):
    if isinstance(v, float):
        return round(v, 3)
    return v

def compute_stats(rows, col_idx):
    vals = [r[col_idx] for r in rows if is_numeric(r[col_idx])]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "min": None, "max": None}
    m = mean(vals)
    return {"n": n, "mean": round(m, 3), "min": round(min(vals), 3), "max": round(max(vals), 3)}

def rank_row(r, goal_cols, mins, maxs, idx_map):
    score = 0.0
    count = 0
    for col, (mn, mx) in zip(goal_cols, zip(mins, maxs)):
        v = r[idx_map[col]]
        if not is_numeric(v):
            continue
        if mx == mn:
            norm = 1.0
        else:
            if col.endswith('+'):
                norm = (v - mn) / (mx - mn)
            else:  # '-'
                norm = (mx - v) / (mx - mn)
        score += norm
        count += 1
    return score / count if count else 0.0

# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def main(csv_path):
    header, rows = parse_rows(csv_path)

    # Column indices
    idx_map = {name: i for i, name in enumerate(header)}
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    # We use Kloc+ as predictor if present
    predictor_col = 'Kloc+'
    if predictor_col not in idx_map:
        print(f"Missing predictor {predictor_col}", file=sys.stderr)
        sys.exit(1)
    pred_idx = idx_map[predictor_col]

    total_rows = len(rows)
    arrivals_count = round(0.2 * total_rows)
    arrivals_indices = list(range(total_rows - arrivals_count, total_rows))
    history_indices = list(range(0, total_rows - arrivals_count))

    history_rows = [rows[i] for i in history_indices]
    arrivals_rows = [rows[i] for i in arrivals_indices]

    # ------------------------------------------------------------------
    # NASA-07
    # ------------------------------------------------------------------
    print("== NASA-07 ==")

    assessments = []
    seen_rows = history_rows.copy()
    for idx, arr_row in zip(arrivals_indices, arrivals_rows):
        # Train models on seen rows
        effort_target = 'Effort-'
        defects_target = 'Defects-'
        eff_idx = idx_map[effort_target]
        def_idx = idx_map[defects_target]

        eff_int, eff_slope, eff_mean, eff_sd, eff_n = train_linear(seen_rows, eff_idx, pred_idx)
        def_int, def_slope, def_mean, def_sd, def_n = train_linear(seen_rows, def_idx, pred_idx)

        # Predict for arrival
        pred_eff = predict_value(eff_int, eff_slope, arr_row[pred_idx], eff_mean)
        pred_def = predict_value(def_int, def_slope, arr_row[pred_idx], def_mean)

        # Alerts
        eff_thresh = eff_mean + 2 * eff_sd if eff_sd > 0 else eff_mean
        def_thresh = def_mean + 2 * def_sd if def_sd > 0 else def_mean
        alert = pred_eff > eff_thresh or pred_def > def_thresh

        # Record
        assessments.append({
            "row": idx + 1,
            "predicted_effort": round_num(pred_eff),
            "predicted_defects": round_num(pred_def),
            "alert": alert
        })

        # Add arrival to seen
        seen_rows.append(arr_row)

        # Print line
        print(f"Row {idx+1}: Effort={round_num(pred_eff)} Defects={round_num(pred_def)} Alert={alert}")

    # JSON answer
    print(f"ANSWER NASA-07 {json.dumps({'assessments': assessments}, ensure_ascii=False)}")

    # ------------------------------------------------------------------
    # NASA-08
    # ------------------------------------------------------------------
    print("\n== NASA-08 ==")

    checkpoints = []
    for m in [0, 9, 19]:
        m = min(m, arrivals_count)
        seen = history_rows + arrivals_rows[:m]
        # Summary per goal column
        summary = {}
        for col in goal_cols:
            summary[col] = compute_stats(seen, idx_map[col])

        # Shortlist: top 5 rows by rank
        # Precompute mins and maxs for normalization
        mins = []
        maxs = []
        for col in goal_cols:
            vals = [r[idx_map[col]] for r in seen if is_numeric(r[idx_map[col]])]
            if vals:
                mins.append(min(vals))
                maxs.append(max(vals))
            else:
                mins.append(0)
                maxs.append(0)

        ranked = []
        for r in seen:
            score = rank_row(r, goal_cols, mins, maxs, idx_map)
            ranked.append((score, r))
        ranked.sort(key=lambda x: (-x[0], seen.index(x[1])))  # descending score, tie by order
        shortlist = [seen.index(r)+1 for _, r in ranked[:5]]

        # Print checkpoint info
        print(f"After {m} arrivals:")
        for col in goal_cols:
            s = summary[col]
            print(f"  {col}: n={s['n']} mean={s['mean']} min={s['min']} max={s['max']}")
        print(f"  Shortlist: {shortlist}")

        checkpoints.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })

    # JSON answer
    print(f"ANSWER NASA-08 {json.dumps({'checkpoints': checkpoints}, ensure_ascii=False)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])
