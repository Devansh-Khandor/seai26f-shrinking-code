#!/usr/bin/env python3
"""
Skill 04: Real‑time assessment and dynamic inventory summaries.
"""

import sys
import csv
import json
import math
from typing import List, Dict, Tuple, Any, Optional

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def read_csv(path: str) -> Tuple[List[str], List[Tuple[int, Dict[str, str]]]]:
    """Read CSV file and return header and list of rows with their 1‑based row numbers."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [(idx + 1, row) for idx, row in enumerate(reader)]
    return header, rows


def parse_value(v: str) -> Optional[float]:
    """Parse numeric value or return None for missing data."""
    if v in ("", "?"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def is_goal(col: str) -> bool:
    return col.endswith('+') or col.endswith('-')


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #

def compute_minmax(dataset: List[Tuple[int, Dict[str, str]]],
                   goal_cols: List[str]) -> Dict[str, Tuple[float, float]]:
    """Return min and max for each goal column across the dataset."""
    mins = {}
    maxs = {}
    for col in goal_cols:
        values = [parse_value(row[col]) for _, row in dataset
                  if parse_value(row[col]) is not None]
        if not values:
            mins[col] = 0.0
            maxs[col] = 0.0
        else:
            mins[col] = min(values)
            maxs[col] = max(values)
    return mins, maxs


def compute_scores(dataset: List[Tuple[int, Dict[str, str]]],
                   goal_cols: List[str]) -> Dict[int, float]:
    """Return a dict mapping row number to its score."""
    mins, maxs = compute_minmax(dataset, goal_cols)
    scores = {}
    for rownum, row in dataset:
        norm_vals = []
        skip = False
        for col in goal_cols:
            v = parse_value(row[col])
            if v is None:
                skip = True
                break
            mn = mins[col]
            mx = maxs[col]
            if mn == mx:
                n_val = 1.0
            else:
                if col.endswith('-'):  # minimize
                    n_val = (mx - v) / (mx - mn)
                else:  # maximize
                    n_val = (v - mn) / (mx - mn)
            norm_vals.append(n_val)
        if skip:
            continue
        scores[rownum] = sum(norm_vals) / len(norm_vals)
    return scores


def get_top_k(dataset: List[Tuple[int, Dict[str, str]]],
              goal_cols: List[str],
              k: int = 5) -> List[int]:
    """Return the row numbers of the top k scored rows."""
    scores = compute_scores(dataset, goal_cols)
    # Sort by score desc, then rownum asc
    ranked = sorted(scores.items(),
                    key=lambda x: (-x[1], x[0]))
    return [rownum for rownum, _ in ranked[:k]]


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #

def compute_summary(dataset: List[Tuple[int, Dict[str, str]]],
                    goal_cols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Return summary stats for each goal column."""
    summary = {}
    for col in goal_cols:
        vals = [parse_value(row[col]) for _, row in dataset
                if parse_value(row[col]) is not None]
        n = len(vals)
        if n == 0:
            summary[col] = {"n": 0, "mean": None, "min": None, "max": None}
            continue
        mean_val = round(sum(vals) / n, 4)
        min_val = round(min(vals), 4)
        max_val = round(max(vals), 4)
        summary[col] = {"n": n, "mean": mean_val, "min": min_val, "max": max_val}
    return summary


# --------------------------------------------------------------------------- #
# Main processing
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, all_rows = read_csv(csv_path)

    goal_cols = [c for c in header if is_goal(c)]

    total_rows = len(all_rows)
    arrivals_count = int(round(0.2 * total_rows))
    history_count = total_rows - arrivals_count

    history_rows = all_rows[:history_count]
    arrival_rows = all_rows[history_count:]

    # ------------------------------------------------------------------
    # CAR-07: Arrival assessment
    # ------------------------------------------------------------------
    print("== CAR-07 ==")
    assessments = []
    current_dataset = history_rows.copy()
    for rownum, row in arrival_rows:
        current_dataset.append((rownum, row))
        top5 = get_top_k(current_dataset, goal_cols, k=5)
        desirable = rownum in top5
        assessments.append({"row": rownum, "desirable": desirable})
        # simple stats
    print(f"Total arrivals: {arrivals_count}")
    print(f"Assessments computed: {len(assessments)}")
    print(f"Example assessment: {assessments[0] if assessments else 'none'}")

    answer_car07 = {"assessments": assessments}
    print(f"ANSWER CAR-07 {json.dumps(answer_car07)}")

    # ------------------------------------------------------------------
    # CAR-08: Checkpoints after 0, 40, 80 arrivals (if available)
    # ------------------------------------------------------------------
    print("\n== CAR-08 ==")
    checkpoints = []
    for m in [0, 40, 80]:
        if m > arrivals_count:
            continue
        # build dataset up to this checkpoint
        checkpoint_dataset = history_rows + arrival_rows[:m]
        summary = compute_summary(checkpoint_dataset, goal_cols)
        shortlist = get_top_k(checkpoint_dataset, goal_cols, k=5)
        checkpoints.append({"after": m,
                            "summary": summary,
                            "shortlist": shortlist})
        # print a few lines
        print(f"Checkpoint after {m} arrivals: {len(checkpoint_dataset)} rows")
        print(f"  Shortlist: {shortlist}")
        print(f"  Summary keys: {list(summary.keys())}")

    answer_car08 = {"checkpoints": checkpoints}
    print(f"ANSWER CAR-08 {json.dumps(answer_car08)}")


if __name__ == "__main__":
    main()
