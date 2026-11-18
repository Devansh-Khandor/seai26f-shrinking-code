#!/usr/bin/env python3
"""
skill_08.py

Usage: python skill_08.py <csv path>
"""

import json
import sys
from typing import Any, Dict, List, Tuple

import ezr

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def numeric(v: Any) -> float | None:
    """Return numeric value or None if missing."""
    return None if v == "?" else v


def median(values: List[float]) -> float:
    """Median of a list of numbers."""
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def get_goal_names(header: List[str]) -> List[str]:
    return [name for name in header if name[-1] in "+-"]


def get_input_names(header: List[str]) -> List[str]:
    """Input columns are those that are not X and not goals."""
    return [name for name in header if not name.endswith("X") and name[-1] not in "+-"]


def compute_goal_minmax(rows: List[List[Any]], goal_names: List[str]) -> Dict[str, Tuple[float, float]]:
    """Return (min, max) for each goal column."""
    mins_maxs: Dict[str, Tuple[float, float]] = {}
    for name in goal_names:
        vals = [numeric(r[header.index(name)]) for r in rows if numeric(r[header.index(name)]) is not None]
        if not vals:
            mins_maxs[name] = (0.0, 1.0)
        else:
            mn, mx = min(vals), max(vals)
            if mn == mx:
                mx = mn + 1.0
            mins_maxs[name] = (mn, mx)
    return mins_maxs


def compute_composite_score(row: List[Any], goal_names: List[str], goal_minmax: Dict[str, Tuple[float, float]]) -> float:
    """Mean of normalized goal values available in row."""
    norms: List[float] = []
    for name in goal_names:
        v = numeric(row[header.index(name)])
        if v is None:
            continue
        mn, mx = goal_minmax[name]
        norm = (v - mn) / (mx - mn) if mx > mn else 0.0
        if name.endswith("-"):
            norm = 1.0 - norm
        norms.append(norm)
    return sum(norms) / len(norms) if norms else 0.0


def compute_target_thresholds(rows: List[List[Any]], goal_names: List[str]) -> Dict[str, float]:
    """Thresholds are median values of each goal column."""
    thresholds: Dict[str, float] = {}
    for name in goal_names:
        vals = [numeric(r[header.index(name)]) for r in rows if numeric(r[header.index(name)]) is not None]
        thresholds[name] = median(vals)
    return thresholds


def count_targets_met(row: List[Any], goal_names: List[str], thresholds: Dict[str, float]) -> int:
    """Count how many goals meet the threshold."""
    cnt = 0
    for name in goal_names:
        v = numeric(row[header.index(name)])
        if v is None:
            continue
        th = thresholds[name]
        if name.endswith("+") and v >= th:
            cnt += 1
        elif name.endswith("-") and v <= th:
            cnt += 1
    return cnt


def input_differences(row1: List[Any], row2: List[Any], input_names: List[str]) -> Dict[str, Tuple[Any, Any]]:
    """Return dict of differing input columns with values from both rows."""
    diffs: Dict[str, Tuple[Any, Any]] = {}
    for name in input_names:
        v1 = row1[header.index(name)]
        v2 = row2[header.index(name)]
        if v1 != v2:
            diffs[name] = (v1, v2)
    return diffs


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Load data
    rows_iter = ezr.csv(csv_path)
    header = next(rows_iter)
    data_rows: List[List[Any]] = list(rows_iter)
    n_rows = len(data_rows)

    # Build table for distance calculations
    tbl = ezr.Tbl([header] + data_rows)

    goal_names = get_goal_names(header)
    input_names = get_input_names(header)

    goal_minmax = compute_goal_minmax(data_rows, goal_names)

    # Composite scores
    scores = [compute_composite_score(r, goal_names, goal_minmax) for r in data_rows]

    # ------------------------------------------------------------------
    # CAR-15: alternative vehicles for a poorly rated vehicle
    # ------------------------------------------------------------------
    print("== CAR-15 ==")

    poor_idx = min(range(n_rows), key=lambda i: scores[i])
    poor_row_num = poor_idx + 1
    poor_row = data_rows[poor_idx]
    poor_score = scores[poor_idx]

    # Compute distances to all others
    distances: List[float] = []
    for i, r in enumerate(data_rows):
        if i == poor_idx:
            distances.append(float('inf'))
        else:
            d = ezr.xdist(tbl, poor_row, r)
            distances.append(d)

    median_dist = median([d for d in distances if d != float('inf')])

    alternatives = [
        i for i, d in enumerate(distances)
        if i != poor_idx and d <= median_dist and scores[i] > poor_score
    ]

    # Pick top 3 by score
    alternatives.sort(key=lambda i: -scores[i])
    alternatives = alternatives[:3]

    alt_list = []
    for i in alternatives:
        alt_row = data_rows[i]
        diffs = input_differences(poor_row, alt_row, input_names)
        alt_entry = {"row": i + 1, "changes": diffs}
        alt_list.append(alt_entry)

    # Output
    print(f"Poor row: {poor_row_num}")
    print(f"Median distance to others: {median_dist:.3f}")
    print(f"Alternatives (top 3):")
    for alt in alt_list:
        print(f"  Row {alt['row']}: changes={alt['changes']}")

    print(f"ANSWER CAR-15 {json.dumps({'row': poor_row_num, 'alternatives': alt_list})}")

    # ------------------------------------------------------------------
    # CAR-16: closest alternative meeting more targets
    # ------------------------------------------------------------------
    print("\n== CAR-16 ==")

    thresholds = compute_target_thresholds(data_rows, goal_names)

    target_counts = [count_targets_met(r, goal_names, thresholds) for r in data_rows]
    min_met = min(target_counts)
    # choose first row with min_met (smallest index)
    candidate_indices = [i for i, c in enumerate(target_counts) if c == min_met]
    poor_idx_16 = candidate_indices[0]
    poor_row_num_16 = poor_idx_16 + 1
    poor_row_16 = data_rows[poor_idx_16]
    poor_met_16 = target_counts[poor_idx_16]

    # Find alternatives that meet more targets
    alternatives_16 = []
    for i, cnt in enumerate(target_counts):
        if i == poor_idx_16:
            continue
        if cnt > poor_met_16:
            diff_count = len(input_differences(poor_row_16, data_rows[i], input_names))
            alternatives_16.append((i, cnt, diff_count, scores[i]))

    if alternatives_16:
        # pick with fewest input changes; tie by higher score
        alternatives_16.sort(key=lambda x: (x[2], -x[3]))
        alt_idx = alternatives_16[0][0]
        alt_row_num = alt_idx + 1
        changed = list(input_differences(poor_row_16, data_rows[alt_idx], input_names).keys())
    else:
        alt_row_num = None
        changed = []

    # Output
    print(f"Poor row (fewest targets): {poor_row_num_16}")
    print(f"Targets met by poor row: {poor_met_16} out of {len(goal_names)}")
    print(f"Thresholds: {thresholds}")
    if alt_row_num:
        print(f"Alternative row: {alt_row_num}")
        print(f"Changed columns: {changed}")
    else:
        print("No alternative found meeting more targets.")

    answer_16 = {
        "row": poor_row_num_16,
        "targets": thresholds,
        "alternative": alt_row_num,
        "changed": changed
    }
    print(f"ANSWER CAR-16 {json.dumps(answer_16)}")


if __name__ == "__main__":
    main()
