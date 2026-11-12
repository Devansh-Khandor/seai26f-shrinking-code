#!/usr/bin/env python3
"""
Apache configuration monitoring and summarisation.

This script implements:

  * APACHE-07: Detects newly added benchmark rows that have unusually poor
    performance compared to the historical data.
  * APACHE-08: Keeps performance summaries and a recommended configuration
    shortlist up-to-date as new benchmark rows are added.

Usage:
    python skill_04.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------- #
# 1. CSV reading and numeric conversion
# --------------------------------------------------------------------------- #
def read_csv(csv_path: str):
    """
    Read CSV file and return list of rows as dictionaries with numeric values.
    Also returns the name of the performance column.
    """
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    if not rows:
        print("No data found in CSV.")
        sys.exit(0)

    # Identify the performance column (case-insensitive startswith "performance")
    perf_col = None
    for col in rows[0].keys():
        if col.lower().startswith("performance"):
            perf_col = col
            break

    if perf_col is None:
        print("Performance column not found.")
        sys.exit(1)

    # Convert values to numeric
    for r in rows:
        for k, v in r.items():
            if k == perf_col:
                r[k] = float(v)
            else:
                try:
                    r[k] = int(v)
                except ValueError:
                    # Fallback for unexpected non-integer values
                    r[k] = 0

    return rows, perf_col


# --------------------------------------------------------------------------- #
# 2. State handling (persistent history)
# --------------------------------------------------------------------------- #
STATE_FILENAME = "skill_04_state.json"

def load_state(state_path: str) -> dict:
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state_path: str, data: dict):
    with open(state_path, "w") as f:
        json.dump(data, f)


def config_key(row: dict, setting_cols: list[str]) -> str:
    """
    Return a deterministic key string for a configuration row.
    """
    parts = [f"{col}={row[col]}" for col in sorted(setting_cols)]
    return "|".join(parts)


# --------------------------------------------------------------------------- #
# 3. APACHE-07: Alert on unusually poor performance
# --------------------------------------------------------------------------- #
def alert_unusual_perf(rows, perf_col, state: dict):
    """
    Identify newly added configurations that are unusually poor compared to
    historical data. Returns a list of tuples (config_key, perf).
    """
    # Determine setting columns (everything except performance)
    setting_cols = [c for c in rows[0].keys() if c != perf_col]

    # Extract historical performance values
    historical_perfs = list(state.values())
    alerts = []

    if historical_perfs:
        hist_mean = np.mean(historical_perfs)
        hist_std = np.std(historical_perfs, ddof=0)  # population std
        threshold = hist_mean + 2 * hist_std
    else:
        # No history yet: no alerts possible
        threshold = None

    for r in rows:
        key = config_key(r, setting_cols)
        if key not in state:
            # New configuration
            perf = r[perf_col]
            if threshold is not None and perf > threshold:
                alerts.append((key, perf))

    return alerts


# --------------------------------------------------------------------------- #
# 4. APACHE-08: Performance summary & recommended shortlist
# --------------------------------------------------------------------------- #
def performance_summary(rows, perf_col):
    """
    Compute mean, median, min, max for all performance values in rows.
    """
    perf_vals = np.array([r[perf_col] for r in rows])
    return {
        "mean": float(np.mean(perf_vals)),
        "median": float(np.median(perf_vals)),
        "min": float(np.min(perf_vals)),
        "max": float(np.max(perf_vals)),
    }


def recommended_shortlist(rows, perf_col, top_n=5):
    """
    Return the top_n configurations with the lowest performance values.
    """
    setting_cols = [c for c in rows[0].keys() if c != perf_col]
    # Sort rows by performance ascending
    sorted_rows = sorted(rows, key=lambda r: r[perf_col])
    shortlist = []
    for r in sorted_rows[:top_n]:
        cfg = {col: r[col] for col in setting_cols}
        shortlist.append((cfg, r[perf_col]))
    return shortlist


# --------------------------------------------------------------------------- #
# 5. Main driver
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, perf_col = read_csv(csv_path)

    # Load historical state
    state_path = os.path.join(os.path.dirname(__file__), STATE_FILENAME)
    state = load_state(state_path)

    # -------------------------------------------------------------------- #
    # APACHE-07: Detect unusually poor performance
    # -------------------------------------------------------------------- #
    alerts = alert_unusual_perf(rows, perf_col, state)

    print("== APACHE-07 ==")
    if alerts:
        for cfg_key, perf in alerts:
            print(f"ALERT: Unusually poor performance detected.")
            print(f"Configuration: {cfg_key}")
            print(f"Measured performance: {perf:.2f}")
    else:
        print("No unusually poor configurations detected.")

    # -------------------------------------------------------------------- #
    # APACHE-08: Summary & shortlist
    # -------------------------------------------------------------------- #
    summary = performance_summary(rows, perf_col)
    shortlist = recommended_shortlist(rows, perf_col, top_n=5)

    print("\n== APACHE-08 ==")
    print(f"Mean performance: {summary['mean']:.2f}")
    print(f"Median performance: {summary['median']:.2f}")
    print(f"Best performance (min): {summary['min']:.2f}")
    print(f"Worst performance (max): {summary['max']:.2f}")
    print("\nRecommended configuration shortlist (best 5):")
    for cfg, perf in shortlist:
        cfg_parts = [f"{k}={v}" for k, v in cfg.items()]
        cfg_str = ", ".join(cfg_parts)
        print(f"{cfg_str} ; Performance={perf:.2f}")

    # -------------------------------------------------------------------- #
    # Update state with all current configurations
    # -------------------------------------------------------------------- #
    setting_cols = [c for c in rows[0].keys() if c != perf_col]
    for r in rows:
        key = config_key(r, setting_cols)
        state[key] = r[perf_col]

    save_state(state_path, state)


if __name__ == "__main__":
    main()
