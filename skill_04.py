#!/usr/bin/env python3
"""
Skill 04 – Incremental Inventory Assessment

This script performs two main tasks:

* CAR-07 – As new vehicle records arrive (via stdin), assess each one immediately
          and decide whether it belongs among the more desirable inventory
          options (top N by composite score).

* CAR-08 – After all new records are received, update inventory summaries
          and the recommended‑vehicle shortlist without restarting the analysis.

Usage:
    python skill_04.py <csv_path>

The script expects the CSV file to have the same columns as the existing
skill modules (Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+).  New
records must be supplied on stdin, one line per record, matching the header
order.  After processing, the script outputs the CAR‑07 assessment
followed by the CAR‑08 updated summary and shortlist.
"""

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV file into a list of row dictionaries."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def safe_float(val: str) -> float | None:
    """Return float or None if conversion fails."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ----------------------------------------------------------------------
# Composite score calculation
# ----------------------------------------------------------------------
def composite_score(row: Dict[str, str],
                    w_min: float, w_max: float,
                    mpg_min: float, mpg_max: float,
                    acc_min: float, acc_max: float) -> float | None:
    """Return the composite score for a single row using supplied bounds."""
    w = safe_float(row["Lbs-"])
    m = safe_float(row["Mpg+"])
    a = safe_float(row["Acc+"])
    if None in (w, m, a):
        return None

    def norm(val: float, low: float, high: float,
             higher_is_better: bool = True) -> float:
        if high == low:
            return 0.5
        return (val - low) / (high - low) if higher_is_better else (high - val) / (high - low)

    w_score = norm(w, w_min, w_max, higher_is_better=False)
    m_score = norm(m, mpg_min, mpg_max)
    a_score = norm(a, acc_min, acc_max)
    return (w_score + m_score + a_score) / 3


def compute_all_scores(rows: List[Dict[str, str]]) -> List[float]:
    """Return a list of composite scores for all rows."""
    weights = [safe_float(r["Lbs-"]) for r in rows]
    mpgs = [safe_float(r["Mpg+"]) for r in rows]
    accs = [safe_float(r["Acc+"]) for r in rows]

    weights = [v for v in weights if v is not None]
    mpgs = [v for v in mpgs if v is not None]
    accs = [v for v in accs if v is not None]

    w_min, w_max = min(weights), max(weights)
    mpg_min, mpg_max = min(mpgs), max(mpgs)
    acc_min, acc_max = min(accs), max(accs)

    return [composite_score(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max)
            for r in rows]


# ----------------------------------------------------------------------
# Summary statistics helpers
# ----------------------------------------------------------------------
def format_stat(name: str, mean: float, min_val: float, max_val: float) -> str:
    return f"{name:<6} Mean: {mean:>7.2f} | Min: {min_val:>7.2f} | Max: {max_val:>7.2f}"


def compute_summary(rows: List[Dict[str, str]]) -> Dict[str, Tuple[float, float, float]]:
    """Compute mean, min, max for weight, mpg, acceleration."""
    def collect(col: str) -> List[float]:
        return [v for v in (safe_float(r[col]) for r in rows) if v is not None]

    def stats(vals: List[float]) -> Tuple[float, float, float]:
        if not vals:
            return (0.0, 0.0, 0.0)
        m = sum(vals) / len(vals)
        return (m, min(vals), max(vals))

    weight_vals = collect("Lbs-")
    mpg_vals = collect("Mpg+")
    acc_vals = collect("Acc+")

    return {
        "Weight": stats(weight_vals),
        "MPG": stats(mpg_vals),
        "Acc": stats(acc_vals),
    }


# ----------------------------------------------------------------------
# CAR-07 – Assessment of new records
# ----------------------------------------------------------------------
def assess_new_records(existing_rows: List[Dict[str, str]],
                       new_rows: List[Dict[str, str]],
                       top_n: int) -> List[Tuple[int, float, bool]]:
    """
    Return a list of tuples: (index_of_new_record, score, recommended?)
    index is 1-based relative to all rows after insertion.
    """
    # Bounds based on existing data only
    weights = [safe_float(r["Lbs-"]) for r in existing_rows]
    mpgs = [safe_float(r["Mpg+"]) for r in existing_rows]
    accs = [safe_float(r["Acc+"]) for r in existing_rows]

    weights = [v for v in weights if v is not None]
    mpgs = [v for v in mpgs if v is not None]
    accs = [v for v in accs if v is not None]

    w_min, w_max = min(weights), max(weights)
    mpg_min, mpg_max = min(mpgs), max(mpgs)
    acc_min, acc_max = min(accs), max(accs)

    # Compute existing scores and threshold
    existing_scores = compute_all_scores(existing_rows)
    existing_scores_sorted = sorted(existing_scores, reverse=True)
    threshold = existing_scores_sorted[top_n - 1] if len(existing_scores_sorted) >= top_n else None

    results = []
    for idx, r in enumerate(new_rows, start=1):
        score = composite_score(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max)
        if score is None:
            rec = False
        else:
            rec = threshold is None or score > threshold
        results.append((idx, score, rec))
    return results


# ----------------------------------------------------------------------
# CAR-08 – Update summaries and shortlist
# ----------------------------------------------------------------------
def update_inventory(existing_rows: List[Dict[str, str]],
                     new_rows: List[Dict[str, str]],
                     top_n: int) -> Tuple[Dict[str, Tuple[float, float, float]],
                                          List[Tuple[int, float, Dict[str, str]]]]:
    """
    Return updated summary and top N shortlist.
    shortlist items: (row_index, score, row_dict)
    """
    all_rows = existing_rows + new_rows
    summary = compute_summary(all_rows)

    scores = compute_all_scores(all_rows)
    # Pair with row indices (1-based)
    indexed = [(i + 1, s, r) for i, (s, r) in enumerate(zip(scores, all_rows)) if s is not None]
    indexed_sorted = sorted(indexed, key=lambda x: x[1], reverse=True)
    shortlist = indexed_sorted[:top_n]
    return summary, shortlist


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File '{csv_path}' does not exist.")
        sys.exit(1)

    # Read existing dataset
    existing_rows = read_csv(csv_path)
    if not existing_rows:
        print("Error: CSV file is empty or has no data rows.")
        sys.exit(1)

    # Read new records from stdin (if any)
    new_rows_raw = [line.rstrip("\n") for line in sys.stdin]
    new_rows: List[Dict[str, str]] = []

    if new_rows_raw:
        header = existing_rows[0].keys()
        for line in new_rows_raw:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(header):
                print(f"Warning: Skipping malformed input line: {line}", file=sys.stderr)
                continue
            row = dict(zip(header, parts))
            new_rows.append(row)

    # --------------------------------------------
    # CAR-07
    # --------------------------------------------
    print("== CAR-07 ==")
    if not new_rows:
        print("No new records to assess.")
    else:
        assessment = assess_new_records(existing_rows, new_rows, top_n=5)
        for idx, score, rec in assessment:
            if score is None:
                score_str = "   N/A"
            else:
                score_str = f"{score:7.3f}"
            rec_str = "Yes" if rec else "No"
            print(f"{idx:3d}. Score: {score_str} | Recommended: {rec_str}")

    # --------------------------------------------
    # CAR-08
    # --------------------------------------------
    print("\n== CAR-08 ==")
    summary, shortlist = update_inventory(existing_rows, new_rows, top_n=5)

    print("Inventory Summary:")
    for name, (mean, min_val, max_val) in summary.items():
        print(format_stat(name, mean, min_val, max_val))

    print("\nTop 5 Recommended Vehicles (after updates):")
    for idx, score, row in shortlist:
        weight = safe_float(row["Lbs-"])
        mpg = safe_float(row["Mpg+"])
        acc = safe_float(row["Acc+"])
        print(
            f"{idx:3d}. Row {idx:<3d} | Score: {score:>5.3f} | "
            f"Weight: {weight:>7.1f} | MPG: {mpg:>4.1f} | Acc: {acc:>5.2f}"
        )


if __name__ == "__main__":
    main()
