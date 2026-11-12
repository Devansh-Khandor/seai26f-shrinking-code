#!/usr/bin/env python3
"""
Skill 09 – Outlier Detection & Cross‑Check of Recorded Values

The script implements two new diagnostics for the used‑car inventory:

* CAR‑17 – Flag vehicles that are multivariate outliers (combinations of
  characteristics that differ unusually from the rest of the inventory).

* CAR‑18 – Flag vehicles whose weight or fuel‑efficiency (MPG) differs
  substantially from what would be expected based on similar cars,
  helping to spot potential data entry mistakes.

The script is self‑contained and uses only the Python standard library.
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
    """Read CSV file into a list of dictionaries."""
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
# Statistics helpers
# ----------------------------------------------------------------------
def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Return (mean, std) for a list of floats. std is population std."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return (mean, math.sqrt(var))


def compute_column_stats(rows: List[Dict[str, str]],
                        columns: List[str]) -> Dict[str, Tuple[float, float]]:
    """Compute mean and std for each numeric column."""
    stats: Dict[str, Tuple[float, float]] = {}
    for col in columns:
        vals = [safe_float(r[col]) for r in rows]
        vals = [v for v in vals if v is not None]
        mean, std = compute_mean_std(vals)
        if std == 0.0:
            std = 1.0  # avoid division by zero
        stats[col] = (mean, std)
    return stats


# ----------------------------------------------------------------------
# CAR‑17 – Multivariate outlier detection
# ----------------------------------------------------------------------
def compute_z_scores(rows: List[Dict[str, str]],
                     cols: List[str],
                     stats: Dict[str, Tuple[float, float]]) -> List[Dict[str, float]]:
    """Return list of z‑scores per row for the selected columns."""
    zlist: List[Dict[str, float]] = []
    for r in rows:
        zrow: Dict[str, float] = {}
        valid = True
        for col in cols:
            val = safe_float(r[col])
            if val is None:
                valid = False
                break
            mean, std = stats[col]
            zrow[col] = (val - mean) / std
        if valid:
            zlist.append(zrow)
        else:
            zlist.append({})  # placeholder for missing rows
    return zlist


def mahalanobis_like_distance(zrow_i: Dict[str, float],
                              zrow_j: Dict[str, float],
                              cols: List[str]) -> float:
    """Compute Euclidean distance between two rows in z‑space."""
    sum_sq = 0.0
    for col in cols:
        zi = zrow_i[col]
        zj = zrow_j[col]
        diff = zi - zj
        sum_sq += diff * diff
    return math.sqrt(sum_sq)


def detect_outliers(rows: List[Dict[str, str]],
                    numeric_cols: List[str]) -> List[Tuple[int, float]]:
    """Return list of (row_index, distance) sorted by decreasing distance."""
    stats = compute_column_stats(rows, numeric_cols)
    zrows = compute_z_scores(rows, numeric_cols, stats)

    outliers: List[Tuple[int, float]] = []
    n = len(rows)
    for i in range(n):
        zi = zrows[i]
        if not zi:
            continue  # skip rows with missing data
        # Find distance to mean (which is zero in z‑space)
        distance = mahalanobis_like_distance(zi, {col: 0.0 for col in numeric_cols}, numeric_cols)
        outliers.append((i, distance))

    outliers.sort(key=lambda x: x[1], reverse=True)
    return outliers


# ----------------------------------------------------------------------
# CAR‑18 – Cross‑check weight and MPG against similarity predictions
# ----------------------------------------------------------------------
def predict_value(rows: List[Dict[str, str]],
                  idx: int,
                  target_col: str,
                  similarity_cols: List[str],
                  stats: Dict[str, Tuple[float, float]],
                  k: int = 5) -> float | None:
    """
    Predict target value for row idx using weighted k‑nearest neighbours
    based on similarity_cols (z‑scaled). Return None if insufficient data.
    """
    zrows = compute_z_scores(rows, similarity_cols, stats)

    target_zs = [safe_float(r[target_col]) for r in rows]
    # Build distances
    distances: List[Tuple[int, float]] = []
    for j, zj in enumerate(zrows):
        if j == idx or not zj:
            continue
        dist = mahalanobis_like_distance(zrows[idx], zj, similarity_cols)
        distances.append((j, dist))
    if not distances:
        return None
    distances.sort(key=lambda x: x[1])
    # Weighted average of target values
    w_total = 0.0
    value_sum = 0.0
    for j, dist in distances[:k]:
        target_val = target_zs[j]
        if target_val is None:
            continue
        w = 1.0 / (dist + 1e-6)
        w_total += w
        value_sum += w * target_val
    if w_total == 0.0:
        return None
    return value_sum / w_total


def find_cross_check_issues(rows: List[Dict[str, str]],
                            numeric_cols: List[str]) -> List[Tuple[int, float, float, float, float]]:
    """
    Return list of vehicles where either weight or MPG deviates
    substantially from predictions based on similar cars.
    Each entry: (row_index, weight_residual, mpg_residual, predicted_weight, predicted_mpg)
    """
    # Stats for z‑scaling (use all numeric cols)
    stats = compute_column_stats(rows, numeric_cols)

    # Similarity features: all numeric columns except target
    weight_sim_cols = [c for c in numeric_cols if c != "Lbs-"]
    mpg_sim_cols = [c for c in numeric_cols if c != "Mpg+"]

    # Compute predictions for all rows
    preds_weight: List[float | None] = []
    preds_mpg: List[float | None] = []

    for i, r in enumerate(rows):
        w_pred = predict_value(rows, i, "Lbs-", weight_sim_cols, stats)
        m_pred = predict_value(rows, i, "Mpg+", mpg_sim_cols, stats)
        preds_weight.append(w_pred)
        preds_mpg.append(m_pred)

    # Compute residuals and their statistics
    weight_residuals: List[float] = []
    mpg_residuals: List[float] = []

    for i, r in enumerate(rows):
        w_actual = safe_float(r["Lbs-"])
        m_actual = safe_float(r["Mpg+"])
        w_pred = preds_weight[i]
        m_pred = preds_mpg[i]
        if w_actual is None or m_actual is None:
            continue
        if w_pred is not None:
            weight_residuals.append(abs(w_actual - w_pred))
        if m_pred is not None:
            mpg_residuals.append(abs(m_actual - m_pred))

    w_mean, w_std = compute_mean_std(weight_residuals)
    m_mean, m_std = compute_mean_std(mpg_residuals)

    threshold_w = w_mean + 2 * w_std if w_std > 0 else w_mean
    threshold_m = m_mean + 2 * m_std if m_std > 0 else m_mean

    issues: List[Tuple[int, float, float, float, float]] = []

    for i, r in enumerate(rows):
        w_actual = safe_float(r["Lbs-"])
        m_actual = safe_float(r["Mpg+"])
        w_pred = preds_weight[i]
        m_pred = preds_mpg[i]
        if w_actual is None or m_actual is None:
            continue
        w_res = abs(w_actual - w_pred) if w_pred is not None else None
        m_res = abs(m_actual - m_pred) if m_pred is not None else None
        if w_res is not None and w_res > threshold_w:
            issues.append((i, w_res, m_res, w_pred, m_pred))
        elif m_res is not None and m_res > threshold_m:
            issues.append((i, w_res, m_res, w_pred, m_pred))

    return issues


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File '{csv_path}' does not exist.")
        sys.exit(1)

    rows = read_csv(csv_path)
    if not rows:
        print("Error: CSV file is empty or has no data rows.")
        sys.exit(1)

    numeric_cols = ["Clndrs", "Volume", "HpX", "Model", "origin",
                    "Lbs-", "Acc+", "Mpg+"]

    # CAR‑17
    outliers = detect_outliers(rows, numeric_cols)
    print("== CAR-17 ==")
    if not outliers:
        print("No outlier vehicles detected.")
    else:
        for rank, (idx, dist) in enumerate(outliers[:5], 1):
            print(f"{rank:2d}. Row {idx + 1:<3d} | Distance: {dist:7.3f}")

    # CAR‑18
    issues = find_cross_check_issues(rows, numeric_cols)
    print("\n== CAR-18 ==")
    if not issues:
        print("No cross‑check anomalies found.")
    else:
        # Show at most five anomalies
        for rank, (idx, w_res, m_res, w_pred, m_pred) in enumerate(issues[:5], 1):
            w_actual = safe_float(rows[idx]["Lbs-"])
            m_actual = safe_float(rows[idx]["Mpg+"])
            print(f"{rank:2d}. Row {idx + 1:<3d}")
            if w_res is not None:
                print(f"   Weight: actual {w_actual:7.1f} lbs, predicted {w_pred:7.1f} lbs, residual {w_res:6.1f} lbs")
            if m_res is not None:
                print(f"   MPG   : actual {m_actual:7.1f} mpg, predicted {m_pred:7.1f} mpg, residual {m_res:6.1f} mpg")
    # End

if __name__ == "__main__":
    main()
