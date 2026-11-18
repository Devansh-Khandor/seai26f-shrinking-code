#!/usr/bin/env python3
"""
Skill 09: Detect anomalous vehicle records.

CAR-17: Flag vehicles whose combination of input characteristics is unusually different
        from the rest of the inventory.

CAR-18: Flag vehicles whose fuel efficiency (Mpg+) or weight (Lbs-) differs substantially
        from the expected value inferred from similar vehicles.

Usage:
    python skill_09.py <csv_path>
"""

import sys
import csv
import json
import math
from typing import List, Dict, Tuple

# --------------------------------------------------------------------------- #
# Reuse helpers from skill_01
# --------------------------------------------------------------------------- #

try:
    from skill_01 import parse_csv, is_missing, to_float, normalize
except Exception:
    def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames if reader.fieldnames else []
        return rows, header

    def is_missing(val: str) -> bool:
        return val == '?' or val is None

    def to_float(val: str) -> float:
        try:
            return float(val)
        except Exception:
            return math.nan

    def normalize(val: float, min_val: float, max_val: float, maximize: bool) -> float:
        if max_val == min_val:
            return 1.0
        if maximize:
            return (val - min_val) / (max_val - min_val)
        return (max_val - val) / (max_val - min_val)

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def identify_columns(rows: List[Dict[str, str]], header: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Return goal columns dict (col: dir) and list of input columns."""
    goal_columns = {col: col[-1] for col in header if col.endswith('-') or col.endswith('+')}
    input_columns = [col for col in header if col not in goal_columns and not col.endswith('X')]
    return goal_columns, input_columns

def compute_input_stats(rows: List[Dict[str, str]],
                        input_columns: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute mean and std for each input column (ignoring missing)."""
    means, stds = {}, {}
    for col in input_columns:
        vals = [to_float(r[col]) for r in rows if not is_missing(r[col])]
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            stds[col] = math.sqrt(var)
            means[col] = mean
        else:
            means[col] = 0.0
            stds[col] = 1.0  # avoid div‑by‑zero
    return means, stds

def standardized_row(row: Dict[str, str],
                     input_columns: List[str],
                     means: Dict[str, float],
                     stds: Dict[str, float]) -> List[float]:
    """Return list of standardized values for the input columns; skip rows with missing."""
    z = []
    for col in input_columns:
        if is_missing(row[col]):
            return None
        v = to_float(row[col])
        std = stds[col]
        if std == 0:
            z.append(0.0)
        else:
            z.append((v - means[col]) / std)
    return z

def euclidean_distance(z1: List[float], z2: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(z1, z2)))

# --------------------------------------------------------------------------- #
# CAR-17 logic
# --------------------------------------------------------------------------- #

def compute_unusualness_scores(rows: List[Dict[str, str]],
                               input_columns: List[str],
                               means: Dict[str, float],
                               stds: Dict[str, float]) -> Tuple[Dict[int, float], Dict[int, List[float]]]:
    """Return mapping from row index to unusualness score and z‑vectors."""
    scores = {}
    z_vectors = {}
    for idx, row in enumerate(rows, start=1):
        z = standardized_row(row, input_columns, means, stds)
        if z is None:
            continue
        # unusualness: sum of absolute z values
        score = sum(abs(v) for v in z)
        scores[idx] = score
        z_vectors[idx] = z
    return scores, z_vectors

def flag_unusual_rows(unusual_scores: Dict[int, float]) -> List[int]:
    """Return list of flagged row numbers (1–40) based on threshold."""
    if not unusual_scores:
        return []
    values = list(unusual_scores.values())
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    threshold = mean + 2 * std
    flagged = [idx for idx, score in unusual_scores.items() if score > threshold]
    if not flagged:
        # fallback: top 10 rows
        flagged = sorted(unusual_scores, key=unusual_scores.get, reverse=True)[:10]
    return flagged[:40]

# --------------------------------------------------------------------------- #
# CAR-18 logic
# --------------------------------------------------------------------------- #

def find_nearest_neighbors(z_vectors: Dict[int, List[float]],
                           target_idx: int,
                           k: int = 5) -> List[int]:
    """Return indices of k nearest neighbors (excluding target)."""
    target_z = z_vectors[target_idx]
    distances = [(euclidean_distance(target_z, z_vectors[idx]), idx)
                 for idx in z_vectors if idx != target_idx]
    distances.sort(key=lambda kv: kv[0])
    return [idx for _, idx in distances[:k]]

def expected_value_from_neighbors(rows: List[Dict[str, str]],
                                  neighbor_indices: List[int],
                                  column: str) -> Tuple[float, float]:
    """Return mean and std of column among neighbors; ignores missing values."""
    vals = [to_float(rows[idx - 1][column]) for idx in neighbor_indices
            if not is_missing(rows[idx - 1][column])]
    if not vals:
        return float('nan'), 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var)

def flag_discrepant_goals(rows: List[Dict[str, str]],
                         z_vectors: Dict[int, List[float]],
                         goal_columns: Dict[str, str]) -> List[Dict]:
    """Return list of flagged entries with {row, column, actual, expected}."""
    flagged = []
    for idx, row in enumerate(rows, start=1):
        # Must have both goal columns present
        for col in ('Mpg+', 'Lbs-'):
            if col not in goal_columns:
                continue
            if is_missing(row[col]):
                continue
            actual = to_float(row[col])
            # Find neighbors
            neighs = find_nearest_neighbors(z_vectors, idx, k=5)
            expected_mean, expected_std = expected_value_from_neighbors(rows, neighs, col)
            if math.isnan(expected_mean):
                continue
            diff = abs(actual - expected_mean)
            # Threshold: 1.5 * std; if std zero, any non‑zero diff flags
            if expected_std == 0.0:
                if diff > 1e-6:
                    flagged.append({"row": idx, "column": col, "actual": actual, "expected": expected_mean})
            else:
                if diff > 1.5 * expected_std:
                    flagged.append({"row": idx, "column": col, "actual": actual, "expected": expected_mean})
            if len(flagged) >= 40:
                return flagged[:40]
    return flagged[:40]

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows, header = parse_csv(csv_path)

    # Identify columns
    goal_columns, input_columns = identify_columns(rows, header)

    # Compute input stats
    means, stds = compute_input_stats(rows, input_columns)

    # CAR-17
    unusual_scores, z_vectors = compute_unusualness_scores(rows, input_columns, means, stds)
    flagged_rows_17 = flag_unusual_rows(unusual_scores)

    # Human‑readable CAR‑17 section
    print("== CAR-17 ==")
    print(f"Flagged {len(flagged_rows_17)} rows (max 40).")
    print(f"First 3 flagged rows: {flagged_rows_17[:3]}")
    print(f"Scores (sample): {{row:{unusual_scores[row]:.2f} for row in flagged_rows_17[:3]}}")
    print()

    # Machine‑checked answer for CAR‑17
    scores_json = {str(r): round(unusual_scores[r], 4) for r in flagged_rows_17}
    car17_answer = {"flagged": flagged_rows_17, "scores": scores_json}
    print(f"ANSWER CAR-17 {json.dumps(car17_answer, separators=(',', ':'))}")

    # CAR-18
    flagged_discrepancies = flag_discrepant_goals(rows, z_vectors, goal_columns)

    # Human‑readable CAR‑18 section
    print("== CAR-18 ==")
    print(f"Flagged {len(flagged_discrepancies)} rows with discrepant goals (max 40).")
    if flagged_discrepancies:
        print(f"Sample: {flagged_discrepancies[:3]}")
    print()

    # Machine‑checked answer for CAR‑18
    car18_answer = {"flagged": flagged_discrepancies}
    print(f"ANSWER CAR-18 {json.dumps(car18_answer, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
