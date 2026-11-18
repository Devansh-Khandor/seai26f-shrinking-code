#!/usr/bin/env python3
"""
Skill 08 – Advanced vehicle comparison.

Usage:
    python skill_08.py <csv_path>
"""

import csv
import json
import math
import sys
from typing import Dict, List, Tuple, Optional

# --------------------------------------------------------------------------- #
# Helpers – copy from earlier skills
# --------------------------------------------------------------------------- #
def parse_float(s: str) -> Optional[float]:
    """Parse a string into float, return None for '?' or empty."""
    s = s.strip()
    if s == "" or s == "?":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def normalize(value: float, vmin: float, vmax: float, maximize: bool) -> float:
    """Return 0-1 score for a value. If maximize is False, smaller is better."""
    if vmax == vmin:
        return 1.0
    if maximize:
        return (value - vmin) / (vmax - vmin)
    else:
        return (vmax - value) / (vmax - vmin)


def compute_overall_scores(
    data: List[Dict[str, Optional[float]]],
    goal_cols: List[str],
    goal_directions: Dict[str, bool],
) -> List[Tuple[int, float]]:
    """Return list of (row_number, score) for rows with all goals present."""
    # Pre‑compute min/max per goal
    min_vals: Dict[str, float] = {}
    max_vals: Dict[str, float] = {}
    for col in goal_cols:
        vals = [row[col] for row in data if row[col] is not None]
        if vals:
            min_vals[col] = min(vals)
            max_vals[col] = max(vals)
        else:
            min_vals[col] = max_vals[col] = 0.0

    scores: List[Tuple[int, float]] = []
    for idx, row in enumerate(data, start=1):
        if any(row[col] is None for col in goal_cols):
            continue
        vals = [
            normalize(row[col], min_vals[col], max_vals[col], goal_directions[col])
            for col in goal_cols
        ]
        score = sum(vals) / len(vals)
        scores.append((idx, round(score, 6)))
    return scores


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(csv_path: str) -> None:
    # Read data
    data: List[Dict[str, Optional[float]]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {k: parse_float(v) for k, v in row.items()}
            data.append(parsed_row)

    total_rows = len(data)
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]
    goal_directions = {"Lbs-": False, "Acc+": True, "Mpg+": True}
    input_cols = [
        c for c in data[0].keys() if not (c.endswith("+") or c.endswith("-") or c.endswith("X"))
    ]

    # Compute overall scores
    overall_scores = compute_overall_scores(data, goal_cols, goal_directions)
    if not overall_scores:
        # No complete rows to evaluate.
        print("== CAR-15 ==")
        print("No complete rows available.")
        print("ANSWER CAR-15 {}")
        print("\n== CAR-16 ==")
        print("No complete rows available.")
        print("ANSWER CAR-16 {}")
        return

    # Map row number to score
    score_map = {row_num: score for row_num, score in overall_scores}

    # ------------------------------------------------------------------ #
    # CAR-15: Poorly rated vehicle – find alternatives
    # ------------------------------------------------------------------ #
    # Poor row: lowest overall score
    poor_row_num, poor_score = min(overall_scores, key=lambda x: x[1])
    poor_row = data[poor_row_num - 1]

    # Determine arrivals (last round(0.2*rows))
    arrival_count = round(0.2 * total_rows)
    history_indices = list(range(total_rows - arrival_count))  # 0‑based

    # Normalized values for input columns across whole dataset
    min_input = {}
    max_input = {}
    for col in input_cols:
        vals = [row[col] for row in data if row[col] is not None]
        min_input[col] = min(vals) if vals else 0.0
        max_input[col] = max(vals) if vals else 0.0

    def norm_input(row: Dict[str, Optional[float]]) -> List[float]:
        return [
            normalize(row[col], min_input[col], max_input[col], True)
            for col in input_cols
        ]

    poor_vec = norm_input(poor_row)

    alternatives = []
    for idx in history_indices:
        row = data[idx]
        row_num = idx + 1
        # Must have all goal values
        if any(row[col] is None for col in goal_cols):
            continue
        row_score = score_map[row_num]
        if row_score <= poor_score:
            continue  # only better scores
        vec = norm_input(row)
        # Euclidean distance
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(poor_vec, vec)))
        alternatives.append((dist, row_num, row))

    # Pick nearest alternatives (up to 3)
    alternatives.sort(key=lambda x: x[0])
    selected = alternatives[:3]

    alt_json = []
    for _, alt_num, alt_row in selected:
        changes = {}
        for col in input_cols:
            poor_val = poor_row.get(col)
            alt_val = alt_row.get(col)
            if poor_val != alt_val:
                changes[col] = [poor_val, alt_val]
        alt_json.append({"row": alt_num, "changes": changes})

    # Print CAR-15
    print("== CAR-15 ==")
    print(f"Poorly rated vehicle: row {poor_row_num}")
    print("Alternatives:")
    for alt in alt_json:
        print(f"Row {alt['row']}: changes {alt['changes']}")

    ans_car_15 = json.dumps(
        {"row": poor_row_num, "alternatives": alt_json},
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-15 {ans_car_15}")

    # ------------------------------------------------------------------ #
    # CAR-16: Find vehicle missing targets and nearest better alternative
    # ------------------------------------------------------------------ #
    # Compute threshold for each goal as mean of that column
    thresholds = {}
    for col in goal_cols:
        vals = [row[col] for row in data if row[col] is not None]
        if vals:
            thresholds[col] = sum(vals) / len(vals)
        else:
            thresholds[col] = 0.0

    def meets_target(row: Dict[str, Optional[float]]) -> int:
        count = 0
        for col in goal_cols:
            val = row.get(col)
            if val is None:
                continue
            thr = thresholds[col]
            if goal_directions[col]:  # maximize
                if val >= thr:
                    count += 1
            else:  # minimize
                if val <= thr:
                    count += 1
        return count

    target_counts = []
    for idx, row in enumerate(data, start=1):
        if any(row[col] is None for col in goal_cols):
            continue
        target_counts.append((idx, meets_target(row)))

    if not target_counts:
        print("\n== CAR-16 ==")
        print("No complete rows available.")
        print("ANSWER CAR-16 {}")
        return

    # Find the row with fewest satisfied targets (at least one missing)
    target_counts.sort(key=lambda x: x[1])
    target_row_num, target_sat = target_counts[0]
    target_row = data[target_row_num - 1]

    # Candidates that meet more targets
    candidates = [
        (idx, sat, row) for idx, sat, row in zip([c[0] for c in target_counts], [c[1] for c in target_counts], [data[c[0] - 1] for c in target_counts])
        if sat > target_sat and not (idx == target_row_num)
    ]

    best_alt = None
    best_diff = None
    best_score = None

    for idx, sat, row in candidates:
        # Count differing input columns
        diff_cols = [
            col for col in input_cols if row.get(col) != target_row.get(col)
        ]
        diff_count = len(diff_cols)
        if best_alt is None or diff_count < best_diff or (diff_count == best_diff and score_map[idx] > best_score):
            best_alt = idx
            best_diff = diff_count
            best_score = score_map[idx]
            best_diff_cols = diff_cols

    if best_alt is None:
        # Fallback: use any row that meets more targets
        if candidates:
            best_alt = candidates[0][0]
            best_diff_cols = []
        else:
            best_alt = target_row_num
            best_diff_cols = []

    # Print CAR-16
    print("\n== CAR-16 ==")
    print(f"Target vehicle: row {target_row_num}")
    print("Thresholds:", thresholds)
    print(f"Alternative: row {best_alt}")
    print("Changed input columns:", best_diff_cols)

    ans_car_16 = json.dumps(
        {
            "row": target_row_num,
            "targets": thresholds,
            "alternative": best_alt,
            "changed": best_diff_cols,
        },
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-16 {ans_car_16}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
