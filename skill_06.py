#!/usr/bin/env python3
"""
skill_06.py

Implements APACHE-11 and APACHE-12 for the Apache configuration dataset.

- APACHE-11: Recommends the best configuration (lowest Performance-) and
  lists the three most similar configurations (in terms of Hamming distance
  across the setting columns).  The JSON output includes the recommended row
  number and the neighbor rows with the full row values (including all goal
  columns).

- APACHE-12: Compares the best and worst configurations, identifies the
  settings that differ between them, and explains the trade‑offs in goal
  performance.  A composite score is calculated for each row so that a
  higher score indicates a better overall configuration.  The JSON output
  contains the row numbers, the preferred row, the composite scores,
  the list of differing settings, and the goal columns where the non‑preferred
  row actually performs better.
"""

import csv
import json
import sys
from collections import defaultdict, Counter

# ----------------------------------------------------------------------
# Reuse helpers from skill_01
# ----------------------------------------------------------------------
try:
    # The helper functions are defined in skill_01.py
    from skill_01 import load_csv, to_float, compute_stats
except Exception:
    # If skill_01 is not importable, reimplement minimal versions
    def load_csv(path):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            data = []
            for row in reader:
                cleaned = {k: (None if v == '?' else v) for k, v in row.items()}
                data.append(cleaned)
        return data, headers

    def to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def compute_stats(values):
        vals = [v for v in values if v is not None]
        if not vals:
            return (None, None, None, None)
        mean = sum(vals) / len(vals)
        import statistics
        median = statistics.median(vals)
        _min = min(vals)
        _max = max(vals)
        return (mean, median, _min, _max)

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def parse_setting(val):
    """Convert a setting value to 0/1 or None."""
    if val is None:
        return None
    if val == '1':
        return 1
    if val == '0':
        return 0
    try:
        return 1 if int(val) != 0 else 0
    except ValueError:
        return None

def hamming_distance(row_a, row_b, setting_cols):
    """Return number of differing settings between two rows."""
    dist = 0
    for col in setting_cols:
        va = parse_setting(row_a.get(col))
        vb = parse_setting(row_b.get(col))
        if va is None or vb is None:
            continue
        if va != vb:
            dist += 1
    return dist

def composite_score(row, goal_cols):
    """
    Compute a composite score for a row: sum of goal values, where
    for '+' goals we use the value directly, for '-' goals we use
    negative of the value (so that higher score indicates better).
    """
    score = 0.0
    for col in goal_cols:
        val = to_float(row.get(col))
        if val is None:
            continue
        if col.endswith('+'):
            score += val
        else:
            score -= val
    return score

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    data, headers = load_csv(csv_path)

    # Identify goal and setting columns
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    setting_cols = [h for h in headers if h not in goal_cols and not h.endswith('X')]

    # ------------------------------------------------------------------
    # APACHE-11
    # ------------------------------------------------------------------
    # Find the best (lowest Performance-) configuration
    perf_col = 'Performance-'
    perf_values = [(idx, to_float(row.get(perf_col))) for idx, row in enumerate(data)]
    perf_values = [(idx, v) for idx, v in perf_values if v is not None]
    if not perf_values:
        best_idx = None
    else:
        best_idx = min(perf_values, key=lambda x: x[1])[0]  # 0-based

    recommended_row = best_idx + 1 if best_idx is not None else None

    # Compute neighbors: smallest Hamming distance across settings
    neighbors = []
    if best_idx is not None:
        distances = []
        for idx, row in enumerate(data):
            if idx == best_idx:
                continue
            dist = hamming_distance(data[best_idx], row, setting_cols)
            distances.append((dist, idx))
        # Sort by distance then by performance (lower better)
        distances.sort(key=lambda x: (x[0], to_float(data[x[1]].get(perf_col)) if data[x[1]].get(perf_col) is not None else float('inf')))
        # Take at least 3 neighbors or all if fewer
        neighbor_count = min(3, len(distances))
        for i in range(neighbor_count):
            idx = distances[i][1]
            row_dict = data[idx]
            # Include all columns in values dict
            values = {}
            for h in headers:
                values[h] = row_dict.get(h)
            neighbors.append({"row": idx + 1, "values": values})

    print("== APACHE-11 ==")
    print(f"Recommended row: {recommended_row}")
    print("Neighbors:")
    for n in neighbors:
        print(f"Row {n['row']}")
    # JSON answer
    apache11_json = {
        "recommended": recommended_row,
        "neighbors": neighbors
    }
    print("ANSWER APACHE-11 " + json.dumps(apache11_json, separators=(',', ':')))

    # ------------------------------------------------------------------
    # APACHE-12
    # ------------------------------------------------------------------
    # Find best and worst configurations
    if not perf_values:
        best_idx = None
        worst_idx = None
    else:
        best_idx = min(perf_values, key=lambda x: x[1])[0]
        worst_idx = max(perf_values, key=lambda x: x[1])[0]

    a_row_num = best_idx + 1 if best_idx is not None else None
    b_row_num = worst_idx + 1 if worst_idx is not None else None

    # Determine preferred row (lower performance- is better)
    preferred = "a" if best_idx is not None else None

    # Composite scores
    score_a = composite_score(data[best_idx], goal_cols) if best_idx is not None else None
    score_b = composite_score(data[worst_idx], goal_cols) if worst_idx is not None else None

    # Find differing settings
    differences = []
    for col in setting_cols:
        va = parse_setting(data[best_idx].get(col)) if best_idx is not None else None
        vb = parse_setting(data[worst_idx].get(col)) if worst_idx is not None else None
        if va is None or vb is None:
            continue
        if va != vb:
            differences.append({
                "feature": col,
                "a": va,
                "b": vb
            })

    # Trade‑offs: goal columns where non‑preferred row is better
    tradeoffs = []
    for col in goal_cols:
        va = to_float(data[best_idx].get(col)) if best_idx is not None else None
        vb = to_float(data[worst_idx].get(col)) if worst_idx is not None else None
        if va is None or vb is None:
            continue
        # For '+' goals, higher better; for '-' goals, lower better
        if col.endswith('+'):
            better_for_non_pref = vb > va  # non-pref is worse (since pref is a)
        else:
            better_for_non_pref = vb < va
        if better_for_non_pref:
            tradeoffs.append(col)

    print("== APACHE-12 ==")
    print(f"A (best): {a_row_num}, B (worst): {b_row_num}")
    print(f"Preferred: {preferred}")
    print(f"Scores: a={score_a:.4f}, b={score_b:.4f}")
    print("Differences:")
    for d in differences:
        print(f"{d['feature']}: a={d['a']}, b={d['b']}")
    print("Trade‑offs:")
    for col in tradeoffs:
        print(col)

    apache12_json = {
        "a": a_row_num,
        "b": b_row_num,
        "preferred": preferred,
        "scores": {
            "a": round(score_a, 4) if score_a is not None else None,
            "b": round(score_b, 4) if score_b is not None else None
        },
        "differences": differences,
        "tradeoffs": tradeoffs
    }
    print("ANSWER APACHE-12 " + json.dumps(apache12_json, separators=(',', ':')))

if __name__ == "__main__":
    main()
