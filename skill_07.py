#!/usr/bin/env python3
"""
skill_07.py

Usage: python skill_07.py <csv path>
"""

import json
import sys
from typing import Any, Dict, List, Tuple

import ezr

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def is_numeric(v: Any) -> bool:
    """Return True if v can be parsed as a number."""
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def numeric(v: Any) -> float | None:
    """Return a numeric value, or None if missing."""
    return None if v == "?" else v


def minmax(values: List[float]) -> Tuple[float, float]:
    """Return (min, max).  If all same, return (m, m+1)."""
    if not values:
        return 0.0, 1.0
    mn, mx = min(values), max(values)
    if mn == mx:
        mx = mn + 1.0
    return mn, mx


def normalize(v: float, mn: float, mx: float) -> float:
    """Map value to 0..1."""
    return 0.0 if mx == mn else (v - mn) / (mx - mn)


# ----------------------------------------------------------------------
# Main logic for CAR-13 & CAR-14
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Load all rows
    rows_iter = ezr.csv(csv_path)
    header = next(rows_iter)
    all_rows: List[List[Any]] = list(rows_iter)
    n_rows = len(all_rows)

    # Identify goal and feature columns
    goal_names = [name for name in header if name[-1] in "+-"]
    # Non-goal numeric columns used for heuristics
    feature_names = [
        name for name in header
        if name[-1] not in "+-X" and is_numeric(0)
    ]

    # Find indices
    goal_indices = {name: header.index(name) for name in goal_names}
    feature_indices = {name: header.index(name) for name in feature_names}

    # ------------------------------------------------------------------
    # Compute min/max for each feature (numeric) over all rows
    # ------------------------------------------------------------------
    feature_mins: Dict[str, float] = {}
    feature_maxs: Dict[str, float] = {}
    for name, idx in feature_indices.items():
        vals = [numeric(r[idx]) for r in all_rows if numeric(r[idx]) is not None]
        mn, mx = minmax(vals)
        feature_mins[name] = mn
        feature_maxs[name] = mx

    # ------------------------------------------------------------------
    # Heuristic score for each row
    # ------------------------------------------------------------------
    def row_score(row: List[Any]) -> float:
        """Simple score based on horsepower, cylinders and volume."""
        # weights (chosen arbitrarily)
        w_hp = 0.4
        w_vol = 0.3
        w_cyl = 0.3

        hp = numeric(row[feature_indices["HpX"]])
        vol = numeric(row[feature_indices["Volume"]])
        cyl = numeric(row[feature_indices["Clndrs"]])

        # Normalized values
        hp_norm = normalize(hp, feature_mins["HpX"], feature_maxs["HpX"]) if hp is not None else 0.0
        vol_norm = normalize(vol, feature_mins["Volume"], feature_maxs["Volume"]) if vol is not None else 0.0
        cyl_norm = normalize(cyl, feature_mins["Clndrs"], feature_maxs["Clndrs"]) if cyl is not None else 0.0

        # Prefer high horsepower, low volume, low cylinders
        return w_hp * hp_norm + w_vol * (1 - vol_norm) + w_cyl * (1 - cyl_norm)

    # Sort rows by heuristic score (descending) and take top 30
    scored_rows: List[Tuple[float, int]] = []
    for i, r in enumerate(all_rows, start=1):  # 1-based row number
        scored_rows.append((row_score(r), i))

    scored_rows.sort(key=lambda x: -x[0])
    evaluated_rows_idx = [idx for _, idx in scored_rows[:30]]  # indices of evaluated rows

    # ------------------------------------------------------------------
    # Evaluate those rows (read their goal values)
    # ------------------------------------------------------------------
    evaluated_rows: List[List[Any]] = [all_rows[i - 1] for i in evaluated_rows_idx]

    # Compute min/max for goals among evaluated rows
    goal_mins: Dict[str, float] = {}
    goal_maxs: Dict[str, float] = {}
    for name in goal_names:
        vals = [numeric(r[goal_indices[name]]) for r in evaluated_rows if numeric(r[goal_indices[name]]) is not None]
        mn, mx = minmax(vals)
        goal_mins[name] = mn
        goal_maxs[name] = mx

    # Composite score for each evaluated row
    def composite(row: List[Any]) -> float:
        norms = []
        for name in goal_names:
            val = numeric(row[goal_indices[name]])
            if val is None:
                continue
            mn, mx = goal_mins[name], goal_maxs[name]
            if name.endswith("+"):
                nval = normalize(val, mn, mx)
            else:  # "-"
                nval = 1 - normalize(val, mn, mx)
            norms.append(nval)
        if not norms:
            return 0.0
        return sum(norms) / len(norms)

    best_row = min(evaluated_rows, key=lambda r: -composite(r))
    best_row_num = evaluated_rows_idx[evaluated_rows.index(best_row)]

    # ------------------------------------------------------------------
    # CAR-13 output
    # ------------------------------------------------------------------
    print("== CAR-13 ==")
    print(f"Evaluated rows: {evaluated_rows_idx}")
    print(f"Best row: {best_row_num}")
    # Machine-checked answer
    ans_13 = {"evaluated": evaluated_rows_idx, "best": best_row_num}
    print(f"ANSWER CAR-13 {json.dumps(ans_13, separators=(',', ':'))}")

    # ------------------------------------------------------------------
    # CAR-14: ordering for evaluation
    # ------------------------------------------------------------------
    order = [idx for _, idx in scored_rows[:30]]  # same order used for CAR-13
    print("\n== CAR-14 ==")
    print(f"Evaluation order (first 30): {order}")
    ans_14 = {"order": order}
    print(f"ANSWER CAR-14 {json.dumps(ans_14, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
