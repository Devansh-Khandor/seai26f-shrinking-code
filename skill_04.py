#!/usr/bin/env python3
"""
skill_04.py

Usage:
    python skill_04.py <csv_path>
"""

import csv
import json
import math
import sys

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def is_goal_col(col_name):
    """Goal columns end with '+' or '-'."""
    return col_name.endswith('+') or col_name.endswith('-')

def is_input_col(col_name):
    """Input columns are those that are not goal columns and do not end with 'X'."""
    return not col_name.endswith('X') and not is_goal_col(col_name)

def parse_value(val, col_name):
    """Parse a CSV cell value."""
    if val == '?':
        return None
    if is_goal_col(col_name):
        try:
            return float(val)
        except ValueError:
            return None
    else:
        try:
            return int(val)
        except ValueError:
            return None

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    n_rows = len(rows)
    arrivals_count = round(0.2 * n_rows)
    history_rows = rows[:-arrivals_count] if arrivals_count > 0 else rows
    arrivals = rows[-arrivals_count:] if arrivals_count > 0 else []

    # Identify goal columns (prefer one that ends with '-')
    goal_cols = [c for c in fieldnames if is_goal_col(c)]
    goal_col = None
    for c in goal_cols:
        if c.endswith('-'):
            goal_col = c
            break
    if not goal_col and goal_cols:
        goal_col = goal_cols[0]
    if not goal_col:
        print("No goal column found.", file=sys.stderr)
        sys.exit(1)

    # Pre‑parse all rows for easier use
    parsed_rows = []
    for r in rows:
        parsed = {}
        for c in fieldnames:
            parsed[c] = parse_value(r[c], c)
        parsed_rows.append(parsed)

    # ------------------------------------------------------------------ #
    # APACHE-07
    # ------------------------------------------------------------------ #
    print("== APACHE-07 ==")

    # Compute running statistics on history rows
    history_values = [parsed_rows[i][goal_col] for i in range(len(history_rows)) if parsed_rows[i][goal_col] is not None]
    m = len(history_values)
    if m > 0:
        mean_hist = sum(history_values) / m
        var_hist = sum((x - mean_hist) ** 2 for x in history_values) / m
        std_hist = math.sqrt(var_hist)
    else:
        mean_hist = None
        std_hist = None

    assessments = []
    for idx, row in enumerate(arrivals, start=1):
        abs_row_num = len(history_rows) + idx  # 1‑based index in whole file
        val = row[goal_col]
        alert = False
        if val is not None:
            if std_hist is None:
                # No variance, alert if worse than current mean (if any)
                if mean_hist is not None and val > mean_hist:
                    alert = True
            else:
                z = (val - mean_hist) / std_hist if std_hist > 0 else float('inf')
                if z > 2.0:
                    alert = True
        assessments.append({"row": abs_row_num, "alert": alert})

    # Simple textual summary
    print(f"History rows: {len(history_rows)}")
    print(f"Arrivals: {len(arrivals)}")
    print("Assessments:")
    for a in assessments:
        print(f"  Row {a['row']}: alert={a['alert']}")

    ans_07 = {"assessments": assessments}
    print("ANSWER APACHE-07", json.dumps(ans_07, separators=(',', ':')))

    # ------------------------------------------------------------------ #
    # APACHE-08
    # ------------------------------------------------------------------ #
    print("\n== APACHE-08 ==")

    # checkpoints after m=0,19,38 arrivals (if exist)
    checkpoints_after = [0, 19, 38]
    checkpoints = []

    # Function to compute stats for goal columns given a subset of rows
    def compute_stats(sub_rows):
        stats = {}
        for c in goal_cols:
            values = [r[c] for r in sub_rows if r[c] is not None]
            n = len(values)
            if n > 0:
                mean_val = sum(values) / n
                min_val = min(values)
                max_val = max(values)
            else:
                mean_val = None
                min_val = None
                max_val = None
            stats[c] = {"n": n, "mean": mean_val, "min": min_val, "max": max_val}
        return stats

    # Keep a list of rows seen so far (history + arrivals processed up to m)
    seen_rows = history_rows.copy()

    # Prepare for shortlist: best rows so far
    def get_best_rows(up_to_rows):
        # Determine objective: minimize if goal_col endswith '-', else maximize
        best_rows = []
        for idx, r in enumerate(up_to_rows, start=1):
            val = r[goal_col]
            if val is None:
                continue
            best_rows.append((val, idx))
        # Sort by value (ascending for minimize, descending for maximize)
        reverse = not goal_col.endswith('-')
        best_rows.sort(key=lambda x: x[0], reverse=reverse)
        return [idx for _, idx in best_rows[:5]]

    for m_arr in checkpoints_after:
        if m_arr > len(arrivals):
            continue  # skip checkpoints beyond available arrivals
        # rows included = history + first m_arr arrivals
        subset_rows = history_rows + arrivals[:m_arr]
        # compute stats
        summary = compute_stats(subset_rows)
        # best rows among subset
        shortlist = get_best_rows(subset_rows)
        checkpoints.append({"after": m_arr, "summary": summary, "shortlist": shortlist})

        # Print some summary
        print(f"Checkpoint after {m_arr} arrivals:")
        for c, s in sorted(summary.items()):
            print(f"  {c}: n={s['n']}, mean={s['mean']}, min={s['min']}, max={s['max']}")
        print(f"  Shortlist (row numbers): {shortlist}")

    ans_08 = {"checkpoints": checkpoints}
    print("ANSWER APACHE-08", json.dumps(ans_08, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
