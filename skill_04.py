#!/usr/bin/env python3
"""
Skill 04: Incremental vehicle assessment and inventory upkeep

This script implements two new reports:

* CAR-07 – For every record in the CSV, immediately compute a desirability
            score and report whether the vehicle would belong to the
            current “most desirable” shortlist.

* CAR-08 – After processing the entire file, provide up‑to‑date
            inventory summaries and a recommended vehicle shortlist.
            The analysis is performed incrementally so that staff can
            run the script again without having to manually restart
            the entire pipeline.

Usage:
    python skill_04.py <csv_path>
"""

import csv
import sys
import math

# ----------------------------------------------------------------------
# Basic data parsing
# ----------------------------------------------------------------------
def read_csv(csv_path):
    """Read the Auto93 CSV and return a list of dicts with numeric values."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = {}
            for key in (
                "Clndrs",
                "Volume",
                "HpX",
                "Model",
                "origin",
                "Lbs-",
                "Acc+",
                "Mpg+",
            ):
                try:
                    parsed[key] = float(r[key])
                except (KeyError, ValueError, TypeError):
                    parsed[key] = None
            rows.append(parsed)
    return rows

# ----------------------------------------------------------------------
# Utility math helpers
# ----------------------------------------------------------------------
def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def stddev(lst, ddof=1):
    n = len(lst)
    if n <= ddof:
        return 0.0
    m = mean(lst)
    var = sum((x - m) ** 2 for x in lst) / (n - ddof)
    return math.sqrt(var)

# ----------------------------------------------------------------------
# Score calculation (same as CAR‑02)
# ----------------------------------------------------------------------
def normalize(val, low, high, reverse=False):
    if high == low:
        return 1.0
    norm = (val - low) / (high - low)
    return 1.0 - norm if reverse else norm

def score_for_row(row, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max):
    """Compute the combined desirability score for a single vehicle."""
    nw = normalize(row["Lbs-"], w_min, w_max, reverse=True)   # lighter better
    nm = normalize(row["Mpg+"], mpg_min, mpg_max)              # higher better
    na = normalize(row["Acc+"], acc_min, acc_max, reverse=True)  # lower better
    return (nw + nm + na) / 3.0

# ----------------------------------------------------------------------
# Summary statistics for a list of vehicles
# ----------------------------------------------------------------------
def inventory_summary(rows):
    weights = [r["Lbs-"] for r in rows]
    mpgs = [r["Mpg+"] for r in rows]
    accs = [r["Acc+"] for r in rows]
    return {
        "weight": (len(weights), mean(weights), min(weights), max(weights)),
        "mpg": (len(mpgs), mean(mpgs), min(mpgs), max(mpgs)),
        "acc": (len(accs), mean(accs), min(accs), max(accs)),
    }

# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    all_rows = read_csv(csv_path)

    # ------------------------------------------------------------------
    # CAR‑07: Incrementally assess each vehicle as it arrives
    # ------------------------------------------------------------------
    print("== CAR-07 ==")

    inventory = []          # processed vehicles
    top_n = 5               # size of shortlist
    # Initially no data, so we will skip score until we have at least one vehicle
    for idx, row in enumerate(all_rows, start=1):
        if None in (row["Lbs-"], row["Acc+"], row["Mpg+"]):
            continue  # skip incomplete records

        inventory.append(row)

        # Re‑compute min/max for the current inventory
        w_min, w_max = min(r["Lbs-"] for r in inventory), max(r["Lbs-"] for r in inventory)
        mpg_min, mpg_max = min(r["Mpg+"] for r in inventory), max(r["Mpg+"] for r in inventory)
        acc_min, acc_max = min(r["Acc+"] for r in inventory), max(r["Acc+"] for r in inventory)

        # Compute scores for all current vehicles
        scores = [score_for_row(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max) for r in inventory]
        # Get indices of top N by score
        top_indices = sorted(range(len(inventory)), key=lambda i: scores[i], reverse=True)[:top_n]
        # Determine if the current vehicle is in the top N
        desirable = idx - 1 in top_indices  # inventory index is idx-1

        # Output result for this vehicle
        cur_score = scores[-1]
        status = "Yes" if desirable else "No"
        print(f"Vehicle #{idx}: Score={cur_score:.3f}, Desirable={status}")

    # ------------------------------------------------------------------
    # CAR‑08: Final inventory summary and shortlist
    # ------------------------------------------------------------------
    print("\n== CAR-08 ==")

    # Re‑compute final summary statistics
    summary = inventory_summary(inventory)

    print("Inventory summary:")
    print(f"  Weight – Count={summary['weight'][0]:>4}  Mean={summary['weight'][1]:>7.2f}  "
          f"Min={summary['weight'][2]:>6.1f}  Max={summary['weight'][3]:>6.1f}")
    print(f"  MPG     – Count={summary['mpg'][0]:>4}  Mean={summary['mpg'][1]:>7.2f}  "
          f"Min={summary['mpg'][2]:>6.1f}  Max={summary['mpg'][3]:>6.1f}")
    print(f"  Acc     – Count={summary['acc'][0]:>4}  Mean={summary['acc'][1]:>7.2f}  "
          f"Min={summary['acc'][2]:>6.1f}  Max={summary['acc'][3]:>6.1f}")

    # Compute final scores
    w_min, w_max = min(r["Lbs-"] for r in inventory), max(r["Lbs-"] for r in inventory)
    mpg_min, mpg_max = min(r["Mpg+"] for r in inventory), max(r["Mpg+"] for r in inventory)
    acc_min, acc_max = min(r["Acc+"] for r in inventory), max(r["Acc+"] for r in inventory)
    scores = [score_for_row(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max) for r in inventory]
    # Sort to get final shortlist
    sorted_idx = sorted(range(len(inventory)), key=lambda i: scores[i], reverse=True)

    print(f"\nTop {top_n} vehicles by desirability score:")
    for rank, idx in enumerate(sorted_idx[:top_n], start=1):
        r = inventory[idx]
        s = scores[idx]
        print(f"  {rank}. Vehicle #{idx+1:>3} – "
              f"Weight={r['Lbs-']:.1f} lbs, MPG={r['Mpg+']:.1f}, Acc={r['Acc+']:.1f} s, "
              f"Score={s:.3f}")

if __name__ == "__main__":
    main()
