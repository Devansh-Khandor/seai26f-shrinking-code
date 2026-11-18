#!/usr/bin/env python3
"""
A small script to answer CAR‑01 and CAR‑02 queries for a used‑car
inventory CSV.

Usage:
    python skill_01.py <csv_path>
"""

import csv
import json
import sys
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def median(values: List[float]) -> float:
    """Return median of non‑empty list."""
    n = len(values)
    if n == 0:
        raise ValueError("median of empty list")
    values = sorted(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def score_rows(
    rows: List[Tuple[int, Optional[float], Optional[float], Optional[float]]]
) -> List[Tuple[int, float]]:
    """Return rows sorted by combined score (higher better)."""
    # Extract columns for normalization
    lbs_vals = [v for _, v, _, _ in rows if v is not None]
    acc_vals = [v for _, _, v, _ in rows if v is not None]
    mpg_vals = [v for _, _, _, v in rows if v is not None]

    # Pre‑compute min/max for each column
    lbs_min, lbs_max = min(lbs_vals), max(lbs_vals)
    acc_min, acc_max = min(acc_vals), max(acc_vals)
    mpg_min, mpg_max = min(mpg_vals), max(mpg_vals)

    denom_lbs = lbs_max - lbs_min
    denom_acc = acc_max - acc_min
    denom_mpg = mpg_max - mpg_min

    scores = []
    for rnum, lbs, acc, mpg in rows:
        # Skip rows with missing values for any of the three goals
        if lbs is None or acc is None or mpg is None:
            continue

        # Weight (lower better)
        if denom_lbs == 0:
            w_norm = 1.0
        else:
            w_norm = (lbs_max - lbs) / denom_lbs

        # Acceleration (higher better)
        if denom_acc == 0:
            a_norm = 1.0
        else:
            a_norm = (acc - acc_min) / denom_acc

        # Fuel efficiency (higher better)
        if denom_mpg == 0:
            m_norm = 1.0
        else:
            m_norm = (mpg - mpg_min) / denom_mpg

        score = (w_norm + a_norm + m_norm) / 3.0
        scores.append((rnum, score))

    # Sort by score desc, then row number asc
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores


def summarize_column(values: List[float], direction: str) -> Dict[str, float]:
    """Return statistical summary for a goal column."""
    mean = sum(values) / len(values)
    med = median(values)
    mn = min(values)
    mx = max(values)
    if direction == "min":
        best = mn
        worst = mx
    else:  # direction == "max"
        best = mx
        worst = mn
    return {
        "mean": round(mean, 6),
        "median": round(med, 6),
        "min": mn,
        "max": mx,
        "best": best,
        "worst": worst,
    }


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    # Column names as in the header
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]  # goal columns
    direction_map = {"Lbs-": "min", "Acc+": "max", "Mpg+": "max"}

    # Read CSV and collect data
    rows_data: List[Tuple[int, Optional[float], Optional[float], Optional[float]]] = []
    col_values: Dict[str, List[float]] = {c: [] for c in goal_cols}

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for rnum, row in enumerate(reader, start=1):
            lbs_raw = row["Lbs-"].strip()
            acc_raw = row["Acc+"].strip()
            mpg_raw = row["Mpg+"].strip()

            lbs = None if lbs_raw == "?" else float(lbs_raw)
            acc = None if acc_raw == "?" else float(acc_raw)
            mpg = None if mpg_raw == "?" else float(mpg_raw)

            rows_data.append((rnum, lbs, acc, mpg))

            # Store non‑missing values for summary
            if lbs is not None:
                col_values["Lbs-"].append(lbs)
            if acc is not None:
                col_values["Acc+"].append(acc)
            if mpg is not None:
                col_values["Mpg+"].append(mpg)

    # CAR‑01: summary of each goal column
    print("== CAR-01 ==")
    for col in goal_cols:
        summary = summarize_column(col_values[col], direction_map[col])
        print(
            f"{col}: mean={summary['mean']}, median={summary['median']}, "
            f"min={summary['min']}, max={summary['max']}, "
            f"best={summary['best']}, worst={summary['worst']}"
        )
    # JSON answer
    answer_dict = {c: summarize_column(col_values[c], direction_map[c]) for c in goal_cols}
    print(f"ANSWER CAR-01 {json.dumps(answer_dict, separators=(',', ':'))}")

    # CAR‑02: top 10 rows by combined score
    scored = score_rows(rows_data)
    top_rows = [rnum for rnum, _ in scored[:10]]
    print("\n== CAR-02 ==")
    print(f"Top rows: {top_rows}")
    print(f"ANSWER CAR-02 {json.dumps({'top': top_rows}, separators=(',', ':'))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
