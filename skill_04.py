#!/usr/bin/env python3
"""
Apache Server Configuration Analysis – Skill 04

APACHE-07  – Detect unusually poor performance as new records arrive.
APACHE-08  – Continuously update performance summaries and a
             recommended configuration shortlist without restarting the script.

The script processes the CSV line‑by‑line, updating statistics on the fly
and printing final results after all records have been read.

Usage:
    python skill_04.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import statistics
from typing import Dict, List, Tuple, Iterator


# --------------------------------------------------------------------------- #
# Helper: Streaming CSV iterator
# --------------------------------------------------------------------------- #
def csv_row_iterator(path: str) -> Iterator[Tuple[Dict[str, int], int, List[str]]]:
    """
    Yields for each CSV row:
        (config_dict, performance, flag_names)

    The iterator also returns the list of flag names on the first call
    via a special sentinel value: on the first iteration the flag_names
    list is returned, subsequent iterations return (config, perf, None).
    """
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        flag_names = [k.strip() for k in reader.fieldnames if k.strip() != 'Performance-']
        for idx, row in enumerate(reader):
            cfg = {k.strip(): int(v.strip()) for k, v in row.items() if k.strip() != 'Performance-'}
            perf = int(row['Performance-'].strip())
            yield cfg, perf, flag_names if idx == 0 else None


# --------------------------------------------------------------------------- #
# APACHE-07: Detect unusually poor performance
# --------------------------------------------------------------------------- #
def process_apache07(rows: Iterator[Tuple[Dict[str, int], int, List[str]]]) -> Tuple[int, List[Tuple[Dict[str, int], int]]]:
    """
    Process rows in streaming fashion and detect alerts.

    Returns:
        (total_configs, list_of_alerts)
    """
    alert_count = 0
    alerts: List[Tuple[Dict[str, int], int]] = []

    # Welford's online algorithm for mean and variance
    n = 0
    mean = 0.0
    M2 = 0.0

    # Threshold multiplier (higher = stricter)
    THRESH_MULT = 1.5

    for cfg, perf, flags in rows:
        n += 1
        if n == 1:
            mean = perf
            M2 = 0.0
            continue

        # Update mean & variance
        delta = perf - mean
        mean += delta / n
        delta2 = perf - mean
        M2 += delta * delta2

        variance = M2 / (n - 1) if n > 1 else 0.0
        std = variance ** 0.5

        # Determine alert
        if std > 0 and perf > mean + THRESH_MULT * std:
            alert_count += 1
            alerts.append((cfg, perf))

    return n, alerts


# --------------------------------------------------------------------------- #
# APACHE-08: Continuous summary & recommended shortlist
# --------------------------------------------------------------------------- #
def process_apache08(rows: Iterator[Tuple[Dict[str, int], int, List[str]]]) -> Tuple[Tuple[float, int, int], List[Tuple[Dict[str, int], int]]]:
    """
    As rows are processed, maintain running statistics and build a
    shortlist of top‑performing configurations.

    Returns:
        (mean, best_perf, worst_perf), recommended_configs
    """
    n = 0
    mean = 0.0
    M2 = 0.0

    best_perf = None
    worst_perf = None

    # Store all configurations for shortlist creation
    all_configs: List[Tuple[Dict[str, int], int]] = []

    for cfg, perf, _ in rows:
        n += 1
        all_configs.append((cfg, perf))

        # Update stats
        if n == 1:
            mean = perf
            M2 = 0.0
            best_perf = worst_perf = perf
        else:
            delta = perf - mean
            mean += delta / n
            delta2 = perf - mean
            M2 += delta * delta2

            if perf < best_perf:
                best_perf = perf
            if perf > worst_perf:
                worst_perf = perf

    # Compute mean from stored values
    mean_final = mean

    # Build recommended shortlist: top 5% best (lower perf is better)
    if not all_configs:
        return (mean_final, best_perf, worst_perf), []

    sorted_by_perf = sorted(all_configs, key=lambda x: x[1])
    top_n = max(1, len(sorted_by_perf) // 20)  # at least 1
    recommended = sorted_by_perf[:top_n]

    return (mean_final, best_perf, worst_perf), recommended


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # We need to iterate twice: once for APACHE-07, once for APACHE-08.
    # Since we want streaming, we will read the file twice.
    # For simplicity we can store rows in memory as a list of (cfg, perf, None).
    # This is acceptable for the small dataset size; if needed we can read twice
    # from file using a fresh iterator each time.

    # Read all rows into memory
    rows: List[Tuple[Dict[str, int], int, List[str]]] = []
    for cfg, perf, flags in csv_row_iterator(csv_path):
        rows.append((cfg, perf, flags))

    # ---------- APACHE-07 ----------
    # Process using the same rows but treat as streaming
    def rows_iterator() -> Iterator[Tuple[Dict[str, int], int, List[str]]]:
        for r in rows:
            yield r

    total_configs, alerts = process_apache07(rows_iterator())

    print("== APACHE-07 ==")
    print(f"Total configurations processed: {total_configs}")
    print(f"Number of unusually poor alerts: {len(alerts)}")
    if alerts:
        last_cfg, last_perf = alerts[-1]
        print(f"Last alert - Config: {last_cfg}, Performance: {last_perf} ms")
    else:
        print("No alerts were triggered during processing.")

    # ---------- APACHE-08 ----------
    summary_stats, recommended_configs = process_apache08(rows_iterator())

    mean_perf, best_perf, worst_perf = summary_stats
    print("\n== APACHE-08 ==")
    print(f"Typical (mean) performance: {mean_perf:.2f} ms")
    print(f"Best performance observed: {best_perf} ms")
    print(f"Worst performance observed: {worst_perf} ms")
    print("\nRecommended configuration shortlist (top 5% best):")
    for idx, (cfg, perf) in enumerate(recommended_configs, 1):
        print(f"  {idx}. Config: {cfg}, Performance: {perf} ms")

    # End of processing

if __name__ == "__main__":
    main()
