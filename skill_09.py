#!/usr/bin/env python3
"""
Skill 09: Identify anomalous historical records and flag possible data issues.

NASA-17: Flag historical projects whose combinations of project characteristics
are unusually different from the rest of our records so that we can review them.

NASA-18: Identify project records whose reported effort or defect counts differ
substantially from what comparable projects would lead us to expect, so we can
investigate possible measurement or recording issues.

The script reuses the data‑loading logic from skill_01 and the linear‑regression
helpers from skill_02.  It is designed to produce concise, well‑structured
output that can be parsed by downstream tooling if required.
"""

import sys
import csv
import numpy as np
from typing import List, Dict, Any, Tuple

try:
    import skill_01 as s01
except Exception:  # pragma: no cover
    # Minimal fallback for load_data
    def _load_data(csv_path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    r['Kloc+'] = float(r['Kloc+'])
                    r['Effort-'] = float(r['Effort-'])
                    r['Defects-'] = float(r['Defects-'])
                except (ValueError, KeyError):
                    continue
                rows.append(r)
        return rows

    class _S01:
        load_data = staticmethod(_load_data)

    s01 = _S01()  # type: ignore

try:
    import skill_02 as s02  # for model training helpers if needed
except Exception:  # pragma: no cover
    s02 = None


# ----------------------------------------------------------------------
# Helper: numeric vs categorical
# ----------------------------------------------------------------------
def _is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _numeric_columns(rows: List[Dict[str, Any]]) -> List[str]:
    """Return the list of numeric column names (excluding id and target)."""
    if not rows:
        return []
    sample = rows[0]
    cols = []
    for k, v in sample.items():
        if k in ("idX", "Kloc+", "Effort-", "Defects-", "MonthsX"):
            continue
        if _is_numeric(str(v)):
            cols.append(k)
    return cols


def _categorical_columns(rows: List[Dict[str, Any]]) -> List[str]:
    """Return the list of categorical column names (excluding id and target)."""
    if not rows:
        return []
    sample = rows[0]
    cols = []
    for k, v in sample.items():
        if k in ("idX", "Kloc+", "Effort-", "Defects-", "MonthsX"):
            continue
        if not _is_numeric(str(v)):
            cols.append(k)
    return cols


# ----------------------------------------------------------------------
# NASA-17: characteristic‑deviation outliers
# ----------------------------------------------------------------------
def _compute_characteristic_deviations(
    rows: List[Dict[str, Any]],
    rare_threshold: float = 0.05,
    rare_weight: float = 2.0,
) -> List[Tuple[int, float, Dict[str, Any]]:
    """Return a list of tuples (row_index, score, row) sorted by descending score."""
    if not rows:
        return []

    # Numeric features: compute z‑scores
    num_cols = _numeric_columns(rows)
    num_means = {}
    num_stds = {}
    for col in num_cols:
        vals = np.array([float(r[col]) for r in rows])
        num_means[col] = vals.mean()
        num_stds[col] = vals.std(ddof=1) or 1.0  # avoid division by zero

    # Categorical features: compute frequency of each value
    cat_cols = _categorical_columns(rows)
    cat_freq: Dict[str, Dict[str, int]] = {col: {} for col in cat_cols}
    for r in rows:
        for col in cat_cols:
            cat_freq[col][r[col]] = cat_freq[col].get(r[col], 0) + 1

    # Compute composite scores
    scores: List[Tuple[int, float, Dict[str, Any]]] = []
    for idx, r in enumerate(rows):
        score = 0.0
        # Numeric contribution (sum of squared z‑scores)
        for col in num_cols:
            z = (float(r[col]) - num_means[col]) / num_stds[col]
            score += z ** 2
        # Categorical contribution (rare categories)
        rare_count = 0
        for col in cat_cols:
            total = sum(cat_freq[col].values())
            freq = cat_freq[col][r[col]] / total if total else 0
            if freq <= rare_threshold:
                rare_count += 1
        score += rare_weight * rare_count
        scores.append((idx, score, r))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ----------------------------------------------------------------------
# NASA-18: prediction‑residual outliers
# ----------------------------------------------------------------------
def _train_linear_regression(rows: List[Dict[str, Any]], target_key: str):
    """Return a fitted sklearn LinearRegression model."""
    from sklearn.linear_model import LinearRegression

    X = np.array([float(r["Kloc+"]) for r in rows]).reshape(-1, 1)
    y = np.array([float(r[target_key]) for r in rows])
    model = LinearRegression().fit(X, y)
    return model


def _compute_prediction_outliers(
    rows: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return two lists: effort outliers, defect outliers."""
    effort_model = _train_linear_regression(rows, "Effort-")
    defect_model = _train_linear_regression(rows, "Defects-")

    # Predictions and residuals
    effort_residuals: List[float] = []
    defect_residuals: List[float] = []

    effort_predictions = []
    defect_predictions = []

    for r in rows:
        kloc = float(r["Kloc+"])
        effort_pred = effort_model.predict([[kloc]])[0]
        defect_pred = defect_model.predict([[kloc]])[0]
        effort_predictions.append(effort_pred)
        defect_predictions.append(defect_pred)
        effort_residuals.append(float(r["Effort-"]) - effort_pred)
        defect_residuals.append(float(r["Defects-"]) - defect_pred)

    effort_std = np.std(effort_residuals, ddof=1)
    defect_std = np.std(defect_residuals, ddof=1)

    effort_outliers = []
    defect_outliers = []

    for idx, r in enumerate(rows):
        eff_res = effort_residuals[idx]
        def_res = defect_residuals[idx]
        if abs(eff_res) > 2 * effort_std:
            effort_outliers.append(
                {
                    "idX": r["idX"],
                    "actual": float(r["Effort-"]),
                    "predicted": effort_predictions[idx],
                    "residual": eff_res,
                    "pct": eff_res / float(r["Effort-"]) * 100.0 if float(r["Effort-"]) != 0 else 0.0,
                }
            )
        if abs(def_res) > 2 * defect_std:
            defect_outliers.append(
                {
                    "idX": r["idX"],
                    "actual": float(r["Defects-"]),
                    "predicted": defect_predictions[idx],
                    "residual": def_res,
                    "pct": def_res / float(r["Defects-"]) * 100.0 if float(r["Defects-"]) != 0 else 0.0,
                }
            )

    return effort_outliers, defect_outliers


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------
def _print_nasa_17(scores: List[Tuple[int, float, Dict[str, Any]]]) -> None:
    print("== NASA-17 ==")
    if not scores:
        print("No projects identified as outliers.")
        return
    print("Top characteristic‑deviation outliers (by score):")
    header = f"{'Rank':>4} {'ID':>6} {'Score':>10}"
    print(header)
    for rank, (_, score, r) in enumerate(scores[:10], start=1):
        print(f"{rank:>4} {int(r['idX']):>6} {score:>10.2f}")


def _print_nasa_18(eff_outliers: List[Dict[str, Any]], def_outliers: List[Dict[str, Any]]) -> None:
    print("\n== NASA-18 ==")
    if not eff_outliers and not def_outliers:
        print("No projects exceed prediction thresholds.")
        return

    if eff_outliers:
        print("\nEffort prediction outliers (residual > 2×σ):")
        header = f"{'ID':>6} {'Actual':>10} {'Predicted':>10} {'Resid':>10} {'% Diff':>10}"
        print(header)
        for e in eff_outliers[:10]:
            print(
                f"{int(e['idX']):>6} {e['actual']:>10.2f} {e['predicted']:>10.2f} "
                f"{e['residual']:>10.2f} {e['pct']:>10.2f}"
            )

    if def_outliers:
        print("\nDefect prediction outliers (residual > 2×σ):")
        header = f"{'ID':>6} {'Actual':>10} {'Predicted':>10} {'Resid':>10} {'% Diff':>10}"
        print(header)
        for d in def_outliers[:10]:
            print(
                f"{int(d['idX']):>6} {d['actual']:>10.2f} {d['predicted']:>10.2f} "
                f"{d['residual']:>10.2f} {d['pct']:>10.2f}"
            )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = s01.load_data(csv_path)

    if not rows:
        print("No valid data found.")
        sys.exit(1)

    # NASA-17
    dev_scores = _compute_characteristic_deviations(rows)
    _print_nasa_17(dev_scores)

    # NASA-18
    effort_outliers, defect_outliers = _compute_prediction_outliers(rows)
    _print_nasa_18(effort_outliers, defect_outliers)


if __name__ == "__main__":
    main()
