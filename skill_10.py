#!/usr/bin/env python3
"""
Skill 10: Evaluate and compare estimation and selection models.

NASA-19: Compare our project‑effort estimation system against a simple
historical‑average estimate and show whether it makes meaningfully better
predictions on previously unseen projects.

NASA-20: Compare our project‑selection approach with randomly selecting
proposals under the same evaluation budget, and show whether our system
identifies more desirable project outcomes.

Usage:
    python skill_10.py <csv_path>
"""

import csv
import math
import random
import sys

# ---------- Basic helpers ----------
def parse_float(val: str | None) -> float | None:
    """Safely parse a string to float; return None on failure."""
    if val is None:
        return None
    try:
        return float(val.strip())
    except (AttributeError, ValueError):
        return None

def mean(values):
    return sum(values) / len(values) if values else 0.0

def stddev(values, mean_val):
    n = len(values)
    if n < 2:
        return 0.0
    return math.sqrt(sum((x - mean_val) ** 2 for x in values) / (n - 1))

def linear_regression(x, y):
    """Return slope and intercept for y = intercept + slope * x."""
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(v * v for v in x)
    sum_xy = sum(v * w for v, w in zip(x, y))
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 0.0, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept

def predict_with_confidence(slope, intercept, x0, x, y):
    """Return predicted y at x0 and its standard error of prediction."""
    y0 = intercept + slope * x0
    n = len(x)
    if n < 2:
        return y0, 0.0
    mean_x = sum(x) / n
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    sigma_sq = sum(r * r for r in residuals) / (n - 2) if n > 2 else 0.0
    if sum_sq_x == 0 or sigma_sq == 0:
        return y0, 0.0
    se = math.sqrt(
        sigma_sq * (1 + 1 / n + ((x0 - mean_x) ** 2) / sum_sq_x)
    )
    return y0, se

# ---------- Load data ----------
def load_data(csv_path: str):
    """Return list of rows with parsed numeric fields."""
    numeric_cols = ["Kloc+", "Effort-", "Defects-"]
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            r = raw.copy()
            for col in numeric_cols:
                r[col] = parse_float(r.get(col))
            rows.append(r)
    return rows

# ---------- NASA-19 ----------
def nasa_19(rows):
    print("== NASA-19 ==")

    # Filter rows that have all three numeric fields
    valid_rows = [r for r in rows if r["Kloc+"] is not None and r["Effort-"] is not None and r["Defects-"] is not None]
    if len(valid_rows) < 5:
        print("Not enough data to perform a split.")
        return

    # Use random split 70/30 (fixed seed for reproducibility)
    random.seed(0)
    indices = list(range(len(valid_rows)))
    random.shuffle(indices)
    split_at = int(0.7 * len(indices))
    train_idx = indices[:split_at]
    test_idx = indices[split_at:]

    # Training data
    kloc_train = [valid_rows[i]["Kloc+"] for i in train_idx]
    effort_train = [valid_rows[i]["Effort-"] for i in train_idx]

    # Test data
    kloc_test = [valid_rows[i]["Kloc+"] for i in test_idx]
    effort_test = [valid_rows[i]["Effort-"] for i in test_idx]

    # Regression model
    slope, intercept = linear_regression(kloc_train, effort_train)

    # Predictions and errors
    errors_reg = []
    errors_avg = []
    for k, e_true in zip(kloc_test, effort_test):
        # Regression prediction
        e_pred_reg, _ = predict_with_confidence(slope, intercept, k, kloc_train, effort_train)
        errors_reg.append(abs(e_true - e_pred_reg))

        # Simple average baseline
        e_pred_avg = mean(effort_train)
        errors_avg.append(abs(e_true - e_pred_avg))

    mae_reg = mean(errors_reg)
    mae_avg = mean(errors_avg)

    rmse_reg = math.sqrt(mean([err ** 2 for err in errors_reg]))
    rmse_avg = math.sqrt(mean([err ** 2 for err in errors_avg]))

    improvement = 100 * (mae_avg - mae_reg) / mae_avg if mae_avg else 0.0

    print(f"Training projects: {len(train_idx)}")
    print(f"Test projects: {len(test_idx)}")
    print(f"Regression MAE: {mae_reg:.2f} hours")
    print(f"Regression RMSE: {rmse_reg:.2f} hours")
    print(f"Average MAE: {mae_avg:.2f} hours")
    print(f"Average RMSE: {rmse_avg:.2f} hours")
    print(f"MAE improvement over average: {improvement:.1f}%")
    print()

# ---------- NASA-20 ----------
def z_score(value, mean_val, std_val):
    return (value - mean_val) / std_val if std_val > 0 else 0.0

def compute_score(row, stats):
    """Composite score = KLOC z - Effort z - Defects z."""
    k = row["Kloc+"]
    e = row["Effort-"]
    d = row["Defects-"]
    return (
        z_score(k, stats["mean_kloc"], stats["std_kloc"])
        - z_score(e, stats["mean_effort"], stats["std_effort"])
        - z_score(d, stats["mean_defects"], stats["std_defects"])
    )

def nasa_20(rows):
    print("== NASA-20 ==")

    # Filter rows with complete numeric data
    valid_rows = [r for r in rows if r["Kloc+"] is not None and r["Effort-"] is not None and r["Defects-"] is not None]
    if not valid_rows:
        print("No complete projects to evaluate.")
        return

    # Ask for evaluation budget
    try:
        inp = input("Enter number of proposals you can evaluate (budget): ").strip()
        budget = int(inp)
        if budget <= 0:
            raise ValueError
    except Exception:
        print("Invalid input; default budget = 3.")
        budget = 3

    # Compute overall stats for z‑scoring
    kloc_vals = [r["Kloc+"] for r in valid_rows]
    effort_vals = [r["Effort-"] for r in valid_rows]
    defect_vals = [r["Defects-"] for r in valid_rows]
    stats = {
        "mean_kloc": mean(kloc_vals),
        "std_kloc": stddev(kloc_vals, mean(kloc_vals)),
        "mean_effort": mean(effort_vals),
        "std_effort": stddev(effort_vals, mean(effort_vals)),
        "mean_defects": mean(defect_vals),
        "std_defects": stddev(defect_vals, mean(defect_vals)),
    }

    # Compute score for each proposal
    scored = [(compute_score(r, stats), r) for r in valid_rows]

    # Our selection: top‑budget proposals by score
    scored.sort(key=lambda t: t[0], reverse=True)
    selected = scored[:budget]
    avg_score_selected = mean([s for s, _ in selected])

    # Random selection for comparison
    random.seed(0)
    if len(valid_rows) <= budget:
        random_selected = scored
    else:
        indices = random.sample(range(len(valid_rows)), budget)
        random_selected = [scored[i] for i in indices]
    avg_score_random = mean([s for s, _ in random_selected])

    improvement = 100 * (avg_score_selected - avg_score_random) / abs(avg_score_random) if avg_score_random != 0 else 0.0

    print(f"Budget (number of proposals to evaluate): {budget}")
    print(f"Average score of our selected proposals: {avg_score_selected:.4f}")
    print(f"Average score of random selection: {avg_score_random:.4f}")
    print(f"Score improvement over random: {improvement:.1f}%")
    print("\nTop selected proposal IDs (by score):")
    for s, r in selected[:5]:
        print(f"  ID={r.get('id', 'N/A')} score={s:.4f}")
    print("\nRandomly selected proposal IDs (by score):")
    for s, r in random_selected[:5]:
        print(f"  ID={r.get('id', 'N/A')} score={s:.4f}")
    print()

# ---------- Entry point ----------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_data(csv_path)

    nasa_19(rows)
    nasa_20(rows)
