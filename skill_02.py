#!/usr/bin/env python3
"""
skill_02.py

Usage: python skill_02.py <csv path>
"""

import json
import sys
import random
from typing import List, Tuple, Dict, Any

import ezr

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def numeric(v: Any) -> float | None:
    """Return a numeric value, or None if missing."""
    return None if v == "?" else v


def row_index(row: List[Any]) -> int:
    """Return the 1‑based index of a row (used for keys)."""
    return row[0]  # placeholder, actual index handled elsewhere


# ----------------------------------------------------------------------
# K‑nearest neighbour regression
# ----------------------------------------------------------------------
def knn_predict(
    target_name: str,
    all_rows: List[List[Any]],
    training_rows: List[Tuple[int, List[Any]]],
    heldout_indices: List[int],
    k: int = 3,
) -> Tuple[Dict[str, float], float]:
    """Return predictions and mae for a target column."""
    header = all_rows[0] if all_rows else []
    # Map column name to index
    col_to_i = {name: i for i, name in enumerate(header)}

    target_i = col_to_i[target_name]
    # X columns: all except target and those ending with 'X'
    x_cols = [i for i, name in enumerate(header) if not name.endswith("X") and i != target_i]
    # Build training set: (row_index, row, target_value)
    train_data = []
    for idx, row in training_rows:
        val = numeric(row[target_i])
        if val is not None:
            train_data.append((idx, row, val))

    predictions: Dict[str, float] = {}
    abs_errs: List[float] = []

    # For each held‑out row
    for hi in heldout_indices:
        row = all_rows[hi]  # 1‑based indexing: skip header
        # compute distance to each training row
        dists = []
        for _, tr_row, _ in train_data:
            d = ezr.xdist(
                ezr.Tbl([header] + [tr_row]),
                tr_row,
                {i: row[i] for i in x_cols},
            )
            dists.append((d, tr_row))
        # sort by distance
        dists.sort(key=lambda x: x[0])
        # take k nearest
        neighbors = dists[:k]
        # compute average target
        avg = sum(numeric(n[1][target_i]) for _, n in neighbors) / len(neighbors)
        predictions[str(hi)] = avg

        # actual value for mae
        actual_val = numeric(row[target_i])
        if actual_val is not None:
            abs_errs.append(abs(avg - actual_val))

    mae = sum(abs_errs) / len(abs_errs) if abs_errs else 0.0
    return predictions, mae


# ----------------------------------------------------------------------
# Uncertainty prediction (min/max) for Lbs-
# ----------------------------------------------------------------------
def predict_weight_with_range(
    all_rows: List[List[Any]],
    training_rows: List[Tuple[int, List[Any]]],
    heldout_indices: List[int],
    k: int = 3,
) -> Dict[str, Dict[str, float]]:
    header = all_rows[0]
    col_to_i = {name: i for i, name in enumerate(header)}
    target_name = "Lbs-"
    target_i = col_to_i[target_name]

    x_cols = [i for i, name in enumerate(header) if not name.endswith("X") and i != target_i]

    train_data = []
    for idx, row in training_rows:
        val = numeric(row[target_i])
        if val is not None:
            train_data.append((idx, row, val))

    predictions: Dict[str, Dict[str, float]] = {}

    for hi in heldout_indices:
        row = all_rows[hi]
        dists = []
        for _, tr_row, _ in train_data:
            d = ezr.xdist(
                ezr.Tbl([header] + [tr_row]),
                tr_row,
                {i: row[i] for i in x_cols},
            )
            dists.append((d, tr_row))
        dists.sort(key=lambda x: x[0])
        neighbors = dists[:k]
        vals = [numeric(n[1][target_i]) for _, n in neighbors]
        estimate = sum(vals) / len(vals)
        low = min(vals)
        high = max(vals)
        predictions[str(hi)] = {
            "estimate": estimate,
            "low": low,
            "high": high,
        }

    return predictions


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    # Load data
    rows_iter = ezr.csv(csv_path)
    header = next(rows_iter)
    data_rows = [row for row in rows_iter]

    # Convert to 1-based index list for easy lookup
    all_rows = [header] + data_rows

    total_rows = len(data_rows)
    # Held‑out rows: numbers divisible by 5
    heldout_indices = [i for i in range(1, total_rows + 1) if i % 5 == 0]

    # Budget: at most min(30, n//5)
    budget = min(30, total_rows // 5)

    # Build candidate training set (non‑held‑out rows with non‑missing target)
    candidates = [
        (i, row) for i, row in enumerate(data_rows, start=1) if i not in heldout_indices
    ]

    # Randomly pick up to budget rows for evaluation
    random.seed(ezr.the.Seed)
    selected = random.sample(candidates, min(budget, len(candidates)))

    # CAR‑03: predict Mpg+
    mpg_preds, mpg_mae = knn_predict(
        target_name="Mpg+",
        all_rows=all_rows,
        training_rows=selected,
        heldout_indices=heldout_indices,
        k=3,
    )
    print("== CAR-03 ==")
    # Print at least three lines of results
    # 1) number of predictions
    # 2) MAE
    # 3) sample prediction for first held‑out row
    print(f"predicted {len(mpg_preds)} rows")
    print(f"mae={mpg_mae:.3f}")
    first_row = sorted(mpg_preds.keys())[0]
    print(f"row {first_row} predicted={mpg_preds[first_row]:.3f}")

    # Machine‑checked answer
    car03_json = {"predictions": mpg_preds, "mae": mpg_mae}
    print(f"ANSWER CAR-03 {json.dumps(car03_json, separators=(',', ':'))}")

    # CAR‑04: predict Lbs- with uncertainty
    weight_preds = predict_weight_with_range(
        all_rows=all_rows,
        training_rows=selected,
        heldout_indices=heldout_indices,
        k=3,
    )
    print("\n== CAR-04 ==")
    print(f"predicted {len(weight_preds)} rows")
    # show first three predictions
    for row_key in sorted(weight_preds.keys())[:3]:
        pred = weight_preds[row_key]
        print(
            f"row {row_key}: estimate={pred['estimate']:.1f} "
            f"low={pred['low']:.1f} high={pred['high']:.1f}"
        )

    car04_json = {"predictions": weight_preds}
    print(f"ANSWER CAR-04 {json.dumps(car04_json, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
