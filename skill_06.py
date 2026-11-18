#!/usr/bin/env python3
"""
Skill 06 – NASA-11 & NASA-12
"""
import csv
import json
import math
import statistics
import sys
from typing import Any, Dict, List, Tuple

# Reuse helpers from earlier skills
from skill_01 import normalise_composite
from skill_02 import train_linreg, predict_linreg

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GOALS = ["Kloc+", "Effort-", "Defects-"]  # order matters for output

# ----------------------------------------------------------------------
# Helper to parse a CSV cell to float
# ----------------------------------------------------------------------
def parse_value(v: str) -> Any:
    """Parse a CSV cell to float, return None for missing."""
    if v == "" or v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return None

# ----------------------------------------------------------------------
# Load the CSV
# ----------------------------------------------------------------------
def load_data(csv_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns a list of rows (raw strings) and the column names.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row.copy())
    return rows, fieldnames

# ----------------------------------------------------------------------
# Build parsed rows (numerical goals parsed)
# ----------------------------------------------------------------------
def build_parsed_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parsed_rows = []
    for raw in raw_rows:
        parsed = {}
        for col, val in raw.items():
            if col in GOALS:
                parsed[col] = parse_value(val)
            else:
                parsed[col] = val
        parsed_rows.append(parsed)
    return parsed_rows

# ----------------------------------------------------------------------
# Distance between two rows (used for nearest‑neighbour explanation)
# ----------------------------------------------------------------------
def distance_between(row_a: Dict[str, Any], row_b: Dict[str, Any], cols: List[str]) -> float:
    """
    Compute a simple distance metric:
        - numeric columns: squared difference if both present
        - categorical columns: 1 if values differ (both present)
    Missing values are ignored.
    """
    dist_sq = 0.0
    for col in cols:
        if col in GOALS:
            continue  # ignore goal columns
        val_a_raw = row_a[col]
        val_b_raw = row_b[col]
        # Try numeric
        val_a = parse_value(val_a_raw)
        val_b = parse_value(val_b_raw)
        if val_a is not None and val_b is not None:
            dist_sq += (val_a - val_b) ** 2
        else:
            # treat as categorical if both present
            if val_a_raw not in ("", "?") and val_b_raw not in ("", "?") and val_a_raw != val_b_raw:
                dist_sq += 1.0
    return dist_sq

# ----------------------------------------------------------------------
# Output value conversion: numeric to float, missing to None
# ----------------------------------------------------------------------
def output_value(val_raw: str) -> Any:
    val = parse_value(val_raw)
    if val is not None:
        return val
    if val_raw in ("", "?"):
        return None
    return val_raw

# ----------------------------------------------------------------------
# NASA-11: Estimate Effort- for a held‑out proposal and explain with neighbors
# ----------------------------------------------------------------------
def nasa_11(
    raw_rows: List[Dict[str, Any]],
    parsed_rows: List[Dict[str, Any]],
    fieldnames: List[str],
) -> None:
    # Identify held‑out rows (divisible by 5)
    held_out_indices = [i for i in range(len(raw_rows)) if (i + 1) % 5 == 0]
    if not held_out_indices:
        print("== NASA-11 ==")
        print("No held‑out rows.")
        print("ANSWER NASA-11 {}")
        return

    # Train linear regression on all non‑held‑out rows
    training_years = []
    training_vals = []
    for i, r in enumerate(parsed_rows):
        if i in held_out_indices:
            continue
        year = parse_value(r["YearX"])
        effort = r["Effort-"]
        if year is not None and effort is not None:
            training_years.append(year)
            training_vals.append(effort)
    a_eff, b_eff = train_linreg(training_years, training_vals)

    # Take the first held‑out row for explanation
    h_idx = held_out_indices[0]
    h_row_raw = raw_rows[h_idx]
    h_row_parsed = parsed_rows[h_idx]
    year_h = parse_value(h_row_parsed["YearX"])
    estimate_effort = predict_linreg(a_eff, b_eff, year_h)

    # Find nearest neighbours among history
    history_indices = [i for i in range(len(raw_rows)) if i not in held_out_indices]
    distances = []
    for hi in history_indices:
        dist = distance_between(raw_rows[h_idx], raw_rows[hi], fieldnames)
        distances.append((hi, dist))
    distances.sort(key=lambda x: x[1])
    neighbor_indices = [idx for idx, _ in distances[:3]]

    # Prepare output
    print("== NASA-11 ==")
    print(f"Project (row {h_idx + 1}) estimate Effort- ≈ {estimate_effort:.3f}")
    for rank, ni in enumerate(neighbor_indices, start=1):
        neigh_vals = {col: output_value(raw_rows[ni][col]) for col in fieldnames}
        print(
            f"Top {rank} neighbor (row {ni + 1}): Effort- = {neigh_vals.get('Effort-', None)}"
        )

    # Build JSON answer
    neighbors = []
    for ni in neighbor_indices:
        vals = {col: output_value(raw_rows[ni][col]) for col in fieldnames}
        neighbors.append({"row": ni + 1, "values": vals})
    answer = {
        "project": h_idx + 1,
        "estimate": estimate_effort,
        "neighbors": neighbors,
    }
    print("ANSWER NASA-11 " + json.dumps(answer, separators=(",", ":")))

# ----------------------------------------------------------------------
# NASA-12: Compare two proposals with different risk assessments
# ----------------------------------------------------------------------
def nasa_12(
    raw_rows: List[Dict[str, Any]],
    parsed_rows: List[Dict[str, Any]],
    fieldnames: List[str],
) -> None:
    # Compute composite scores for all rows that have all goals
    composite_list = normalise_composite(parsed_rows)  # list of (row_num, score)
    composite_map: Dict[int, float] = {row_num: score for row_num, score in composite_list}

    if len(composite_map) < 2:
        print("== NASA-12 ==")
        print("Not enough rows with complete goal data.")
        print("ANSWER NASA-12 {}")
        return

    # Choose two rows with most extreme composite scores
    sorted_rows = sorted(composite_map.items(), key=lambda x: x[1])
    low_row, low_score = sorted_rows[0]
    high_row, high_score = sorted_rows[-1]

    # Determine preferred (higher composite)
    preferred = "a" if high_score >= low_score else "b"
    a_row, a_score = high_row, high_score
    b_row, b_score = low_row, low_score

    # Differences in features
    differences = []
    for col in fieldnames:
        val_a_raw = raw_rows[a_row - 1][col]
        val_b_raw = raw_rows[b_row - 1][col]
        val_a = output_value(val_a_raw)
        val_b = output_value(val_b_raw)
        if val_a != val_b:
            differences.append({"feature": col, "a": val_a, "b": val_b})
    if not differences:
        differences.append({"feature": "None", "a": None, "b": None})

    # Tradeoffs: goal columns where non‑preferred row is better
    tradeoffs = []
    for goal in GOALS:
        a_val = parsed_rows[a_row - 1][goal]
        b_val = parsed_rows[b_row - 1][goal]
        if a_val is None or b_val is None:
            continue
        # Determine which is better for the goal
        if goal.endswith("+"):  # maximize
            better_a = a_val >= b_val
        else:  # minimize
            better_a = a_val <= b_val
        if not better_a:  # non‑preferred better
            tradeoffs.append(goal)

    # Output
    print("== NASA-12 ==")
    print(f"Row {high_row} composite = {high_score:.3f}")
    print(f"Row {low_row} composite = {low_score:.3f}")
    print(f"Preferred row: {preferred.upper()} (row {high_row if preferred=='a' else low_row})")

    answer = {
        "a": high_row,
        "b": low_row,
        "preferred": preferred,
        "scores": {str(high_row): high_score, str(low_row): low_score},
        "differences": differences,
        "tradeoffs": tradeoffs,
    }
    print("ANSWER NASA-12 " + json.dumps(answer, separators=(",", ":")))

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    raw_rows, fieldnames = load_data(csv_path)
    parsed_rows = build_parsed_rows(raw_rows)

    nasa_11(raw_rows, parsed_rows, fieldnames)
    print()
    nasa_12(raw_rows, parsed_rows, fieldnames)

if __name__ == "__main__":
    main()
