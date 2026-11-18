#!/usr/bin/env python3
"""
skill_07.py

Implements APACHE-13 and APACHE-14 for the Apache performance dataset.

Usage:
    python skill_07.py <csv_path>

Author: OpenAI ChatGPT
"""

import csv
import json
import sys

# --------------------------------------------------------------------------- #
# CSV utilities

def read_csv(path):
    """Return header list and rows (dict) from a CSV file."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def parse_int(val):
    """Parse an integer from a cell, return None for missing or bad values."""
    if val is None or val == '?':
        return None
    try:
        return int(val)
    except ValueError:
        return None

# --------------------------------------------------------------------------- #
# Helper to find first minimizing goal column

def find_performance_col(header):
    """Return the first column ending with '-'."""
    for c in header:
        if c.endswith('-'):
            return c
    return None

# --------------------------------------------------------------------------- #
# APACHE-13: Evaluate few rows to find a low performance configuration

def apache_13(header, rows, budget):
    perf_col = find_performance_col(header)
    if perf_col is None:
        raise RuntimeError("No minimizing performance column found")

    evaluated = list(range(1, budget + 1))  # 1‑based row numbers
    best_row = None
    best_val = None

    for rnum in evaluated:
        row = rows[rnum - 1]
        val = parse_int(row[perf_col])
        if val is None:
            continue
        if best_val is None or val < best_val:
            best_val = val
            best_row = rnum

    return evaluated, best_row

# --------------------------------------------------------------------------- #
# APACHE-14: Recommend evaluation order with limited budget

def apache_14(budget):
    # Simple strategy: evaluate rows in numeric order.
    order = list(range(1, budget + 1))
    return order

# --------------------------------------------------------------------------- #
# Main routine

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    n_rows = len(rows)
    budget = min(30, n_rows // 5)
    if budget == 0:
        budget = 1

    # --- APACHE-13 ---
    evaluated_rows, best_row = apache_13(header, rows, budget)
    print("== APACHE-13 ==")
    print(f"Evaluated rows: {evaluated_rows}")
    print(f"Best row found: {best_row}")
    print(f"Best performance value: {parse_int(rows[best_row - 1][find_performance_col(header)])}")
    ans13 = {"evaluated": evaluated_rows, "best": best_row}
    print("ANSWER APACHE-13", json.dumps(ans13))

    # --- APACHE-14 ---
    order = apache_14(budget)
    print("\n== APACHE-14 ==")
    print(f"Evaluation order (first to last): {order}")
    print(f"Number of rows to evaluate: {len(order)}")
    ans14 = {"order": order}
    print("ANSWER APACHE-14", json.dumps(ans14))

if __name__ == "__main__":
    main()
