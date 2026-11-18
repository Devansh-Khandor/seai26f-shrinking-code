#!/usr/bin/env python3
"""
skill_07.py

Usage: python skill_07.py <csv_path>
"""

import sys
import csv
import json

# ------------------------------------------------------------------
# CSV reading helper
# ------------------------------------------------------------------
def read_csv(path):
    """
    Read CSV and return a list of row dicts with 1‑based row numbers
    added under the key '_rownum'.
    Cells containing '?' are converted to None.
    """
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

def compute_budget(num_rows):
    """Maximum number of rows that can be evaluated."""
    return min(30, num_rows // 5)

# ------------------------------------------------------------------
# APACHE-13
# ------------------------------------------------------------------
def apache13(rows, perf_col, budget):
    """Select a sequence of rows to evaluate and return best among them."""
    # Evaluate the first `budget` rows (or fewer if not enough rows)
    evaluated = []
    best_row = None
    best_val = None
    for r in rows[:budget]:
        evaluated.append(r['_rownum'])
        val = r.get(perf_col)
        if val is not None:
            if best_val is None or val < best_val:
                best_val = val
                best_row = r['_rownum']
    return evaluated, best_row

# ------------------------------------------------------------------
# APACHE-14
# ------------------------------------------------------------------
def apache14(rows, perf_col, budget):
    """
    Order rows for evaluation. The goal is to bring low‑performance
    configurations early. We sort all rows by their performance (ascending)
    and take the first `budget` rows.
    """
    # Rows with missing performance are considered worst (placed last)
    def sort_key(r):
        val = r.get(perf_col)
        return (val is None, val if val is not None else 0)
    sorted_rows = sorted(rows, key=sort_key)
    order = [r['_rownum'] for r in sorted_rows[:budget]]
    return order

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        return

    perf_col = performance_col(rows)
    if perf_col is None:
        print("No performance column found.")
        return

    budget = compute_budget(len(rows))

    # --- APACHE-13 ---
    print("== APACHE-13 ==")
    evaluated, best_row = apache13(rows, perf_col, budget)
    print(f"Evaluated rows: {evaluated}")
    print(f"Best row: {best_row}")
    print(f"ANSWER APACHE-13 {json.dumps({'evaluated': evaluated, 'best': best_row}, separators=(',', ':'))}")

    # --- APACHE-14 ---
    print("\n== APACHE-14 ==")
    order = apache14(rows, perf_col, budget)
    print(f"Evaluation order: {order}")
    print(f"ANSWER APACHE-14 {json.dumps({'order': order}, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
