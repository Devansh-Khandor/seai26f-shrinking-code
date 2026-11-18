#!/usr/bin/env python3
"""
skill_02.py: Forecast effort and defects for NASA data.
Usage: python skill_02.py <path-to-csv>
"""

import json
import sys
from typing import List, Dict

import ezr


def read_table(path: str) -> ezr.TBL:
    """Read CSV into an ezr table."""
    rows = list(ezr.csv(path))
    return ezr.Tbl(rows)


def column_at(tbl: ezr.TBL, name: str) -> int:
    """Return column index for the column with the given header name."""
    for at, col in tbl.cols.items():
        if col.txt == name:
            return at
    raise KeyError(f"Column {name!r} not found")


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    """Sample standard deviation (population if len==1)."""
    n = len(values)
    if n <= 1:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (n - 1)) ** 0.5


def ci_bounds(mean_val: float, sd_val: float, n: int, z: float = 1.96) -> (float, float):
    """Return lower and upper bounds of a 95% confidence interval for a mean."""
    if n == 0:
        return mean_val, mean_val
    margin = z * sd_val / (n ** 0.5)
    return mean_val - margin, mean_val + margin


def predict_effort(
    tbl: ezr.TBL, held_out: List[int], history: List[int]
) -> (Dict[str, float], float):
    """Predict Effort- for held‑out rows and compute MAE."""
    effort_at = column_at(tbl, "Effort-")
    # Gather training values, skip missing
    train_vals = [
        float(row[effort_at])
        for idx, row in enumerate(tbl.rows, start=1)
        if idx in history and row[effort_at] != "?"
    ]
    eff_mean = mean(train_vals)

    predictions = {}
    errors = []

    for idx in held_out:
        row = tbl.rows[idx - 1]  # 0‑based
        if row[effort_at] == "?":
            # cannot compute error; skip
            continue
        pred = eff_mean
        predictions[str(idx)] = pred
        errors.append(abs(float(row[effort_at]) - pred))

    mae = mean(errors) if errors else 0.0
    return predictions, mae


def predict_defects(
    tbl: ezr.TBL, held_out: List[int], history: List[int]
) -> Dict[str, Dict[str, float]]:
    """Predict Defects- and its 95% CI for held‑out rows."""
    defects_at = column_at(tbl, "Defects-")
    # Gather training values, skip missing
    train_vals = [
        float(row[defects_at])
        for idx, row in enumerate(tbl.rows, start=1)
        if idx in history and row[defects_at] != "?"
    ]
    n = len(train_vals)
    mean_val = mean(train_vals)
    sd_val = std(train_vals)

    low, high = ci_bounds(mean_val, sd_val, n)

    predictions = {}
    for idx in held_out:
        predictions[str(idx)] = {
            "estimate": mean_val,
            "low": low,
            "high": high,
        }
    return predictions


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv-file>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    tbl = read_table(path)

    # Identify held‑out rows: divisible by 5 (1‑based)
    all_indices = list(range(1, len(tbl.rows) + 1))
    held_out = [i for i in all_indices if i % 5 == 0]
    history = [i for i in all_indices if i % 5 != 0]

    # NASA‑03: Effort prediction
    print("== NASA-03 ==")
    eff_preds, mae = predict_effort(tbl, held_out, history)
    # Print a few example rows
    for idx in held_out[:3]:
        print(f"Row {idx}: predicted Effort- = {eff_preds[str(idx)]:.2f}")
    # JSON answer
    print("ANSWER NASA-03", json.dumps({"predictions": eff_preds, "mae": mae}, separators=(",", ":")))

    # NASA‑04: Defects prediction with uncertainty
    print("\n== NASA-04 ==")
    defect_preds = predict_defects(tbl, held_out, history)
    for idx in held_out[:3]:
        p = defect_preds[str(idx)]
        print(f"Row {idx}: estimate {p['estimate']:.2f} (95% CI {p['low']:.2f}–{p['high']:.2f})")
    print("ANSWER NASA-04", json.dumps({"predictions": defect_preds}, separators=(",", ":")))


if __name__ == "__main__":
    main()
