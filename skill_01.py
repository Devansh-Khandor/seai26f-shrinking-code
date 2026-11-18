#!/usr/bin/env python3
import csv
import json
import sys
import statistics
from collections import defaultdict

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_csv(path):
    """
    Load the CSV file and return:
      - data: list of dicts (each row)
      - headers: list of column names
    Missing values represented by '?'
    """
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        data = []
        for row in reader:
            # Convert '?' to None, other values stay as strings
            cleaned = {k: (None if v == '?' else v) for k, v in row.items()}
            data.append(cleaned)
    return data, headers

def to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def compute_stats(values):
    """Return mean, median, min, max for a list of floats"""
    vals = [v for v in values if v is not None]
    if not vals:
        return (None, None, None, None)
    mean = sum(vals) / len(vals)
    median = statistics.median(vals)
    _min = min(vals)
    _max = max(vals)
    return (mean, median, _min, _max)

def best_worst(values, direction):
    """direction: '+' to maximize, '-' to minimize"""
    vals = [v for v in values if v is not None]
    if not vals:
        return (None, None)
    if direction == '+':
        return (max(vals), min(vals))
    else:
        return (min(vals), max(vals))

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    data, headers = load_csv(csv_path)

    # Identify goal columns (ending with '+' or '-')
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    # Identify setting columns (all except goal columns and columns ending in 'X')
    setting_cols = [h for h in headers if h not in goal_cols and not h.endswith('X')]

    # --------------------------------------------------------------
    # APACHE-01
    # --------------------------------------------------------------
    # For now we only care about Performance- column
    perf_col = 'Performance-'
    perf_values = [to_float(row.get(perf_col)) for row in data]
    mean, median, _min, _max = compute_stats(perf_values)
    best, worst = best_worst(perf_values, '-')  # '-' to minimize

    print("== APACHE-01 ==")
    print(f"Performance- mean: {mean:.2f}")
    print(f"Performance- median: {median:.2f}")
    print(f"Performance- min: {_min:.2f}")
    print(f"Performance- max: {_max:.2f}")
    print(f"Performance- best: {best:.2f}")
    print(f"Performance- worst: {worst:.2f}")

    # Prepare JSON answer
    apache01_json = {
        perf_col: {
            "mean": mean,
            "median": median,
            "min": _min,
            "max": _max,
            "best": best,
            "worst": worst
        }
    }
    print("ANSWER APACHE-01 " + json.dumps(apache01_json, separators=(',', ':')))

    # --------------------------------------------------------------
    # APACHE-02
    # --------------------------------------------------------------
    # Define "unusually favorable" as performance <= 25th percentile
    # Filter out rows with missing performance
    valid_perf = [(idx, to_float(row.get(perf_col))) for idx, row in enumerate(data)
                  if to_float(row.get(perf_col)) is not None]
    if not valid_perf:
        top_indices = []
        threshold = None
    else:
        sorted_vals = sorted([v for _, v in valid_perf])
        q1_index = len(sorted_vals) // 4
        threshold = sorted_vals[q1_index] if q1_index < len(sorted_vals) else sorted_vals[-1]
        top_indices = [idx for idx, v in valid_perf if v <= threshold]
        # Ensure at least 5 top rows if possible
        if len(top_indices) < 5:
            # take lowest values up to 5
            top_indices = [idx for _, idx in sorted(valid_perf, key=lambda x: x[1])[:5]]

    top_rows = [idx + 1 for idx in top_indices]  # 1-based row numbers

    # Compute setting shares
    setting_stats = []
    for col in setting_cols:
        # Count ones in top and rest
        top_ones = top_ones_total = 0
        top_non_missing = 0
        rest_ones = rest_ones_total = 0
        rest_non_missing = 0
        for idx, row in enumerate(data):
            val = row.get(col)
            if val is None:
                continue  # skip missing
            if val not in ('0', '1'):
                # treat other numeric strings as integer?
                try:
                    val_int = int(val)
                except ValueError:
                    continue
                val = str(val_int)
            is_one = val == '1'
            if idx in top_indices:
                top_non_missing += 1
                if is_one:
                    top_ones += 1
            else:
                rest_non_missing += 1
                if is_one:
                    rest_ones += 1
        # Compute shares if denominator > 0
        if top_non_missing > 0:
            top_share = top_ones / top_non_missing
        else:
            top_share = None
        if rest_non_missing > 0:
            rest_share = rest_ones / rest_non_missing
        else:
            rest_share = None
        if top_share is not None and rest_share is not None:
            diff = abs(top_share - rest_share)
            setting_stats.append((diff, col, top_share, rest_share))

    # Sort by difference descending, keep at least 2
    setting_stats.sort(reverse=True)
    top_settings = setting_stats[:max(2, len(setting_stats))] if setting_stats else []

    settings_output = []
    for _, col, top_share, rest_share in top_settings:
        settings_output.append({
            "setting": col,
            "top_share": round(top_share, 4),
            "rest_share": round(rest_share, 4)
        })

    # Output
    print("== APACHE-02 ==")
    print(f"Top configurations (row numbers): {top_rows}")
    print("Distinguishing settings:")
    for s in settings_output:
        print(f"{s['setting']}: top {s['top_share']*100:.1f}%, rest {s['rest_share']*100:.1f}%")

    apache02_json = {
        "top": top_rows,
        "settings": settings_output
    }
    print("ANSWER APACHE-02 " + json.dumps(apache02_json, separators=(',', ':')))

if __name__ == "__main__":
    main()
