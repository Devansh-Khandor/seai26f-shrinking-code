#!/usr/bin/env python3
"""
skill_10.py

Implements the final NASA analyses:

- NASA-19: Compare our project‑effort estimation system (linear regression)
           against a simple historical‑average estimate and evaluate
           predictive performance on previously unseen projects.

- NASA-20: Compare our proposal‑selection approach with randomly selecting
           proposals under the same evaluation budget, showing whether
           our system identifies more desirable outcomes.

Usage:
    python skill_10.py path/to/nasa93dem.csv
"""

import csv
import sys
import random
import statistics
from pathlib import Path
import math

# --------------------------------------------------------------------------- #
# CSV reading and numeric conversion
# --------------------------------------------------------------------------- #

def _read_csv(path: Path):
    """
    Read a CSV file and convert all numeric values to float where possible.
    """
    data = []
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                if v is None:
                    cleaned[k] = None
                    continue
                val = v.strip()
                try:
                    cleaned[k] = float(val)
                except ValueError:
                    cleaned[k] = val
            data.append(cleaned)
    return data

# --------------------------------------------------------------------------- #
# Helper math utilities
# --------------------------------------------------------------------------- #

def _numeric_values(data, column):
    """Return a list of numeric values for the given column, excluding None."""
    return [row[column] for row in data if isinstance(row.get(column), (int, float))]

def _linear_regression(x_vals, y_vals):
    """
    Ordinary least squares linear regression.
    Returns (intercept, slope, SE_of_estimate, mean_x, ssx).
    """
    n = len(x_vals)
    if n < 2:
        raise ValueError("Need at least two data points for regression.")
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    ssx = sum((x - mean_x) ** 2 for x in x_vals)
    if ssx == 0:
        slope = 0.0
    else:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)) / ssx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(x_vals, y_vals)]
    sse = sum(r ** 2 for r in residuals)
    se = math.sqrt(sse / (n - 2)) if n > 2 else 0.0
    return intercept, slope, se, mean_x, ssx

def _predict(x, intercept, slope):
    return intercept + slope * x

# --------------------------------------------------------------------------- #
# NASA-19: Effort estimation evaluation
# --------------------------------------------------------------------------- #

def _nasa_19(data):
    print("== NASA-19 ==")
    # Completed projects
    completed = [r for r in data if isinstance(r.get("Effort-"), (int, float))
                 and isinstance(r.get("Kloc+"), (int, float))]
    if len(completed) < 2:
        print("Not enough completed projects to evaluate effort model.")
        return

    # 5‑fold cross‑validation (simple split)
    n_folds = 5
    indices = list(range(len(completed)))
    random.shuffle(indices)
    fold_size = max(1, len(completed) // n_folds)
    mae_reg = 0.0
    mae_baseline = 0.0
    for i in range(n_folds):
        test_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = [idx for idx in indices if idx not in test_idx]
        train = [completed[idx] for idx in train_idx]
        test = [completed[idx] for idx in test_idx]

        train_x = _numeric_values(train, "Kloc+")
        train_y = _numeric_values(train, "Effort-")
        try:
            intercept, slope, _, _, _ = _linear_regression(train_x, train_y)
        except Exception:
            # If regression fails, skip this fold
            continue
        test_x = _numeric_values(test, "Kloc+")
        test_y = _numeric_values(test, "Effort-")

        preds = [_predict(x, intercept, slope) for x in test_x]
        errors = [abs(p - a) for p, a in zip(preds, test_y)]
        mae_reg += sum(errors) / len(errors) if errors else 0

        # Baseline: mean effort of training set
        baseline_val = statistics.mean(train_y)
        baseline_errors = [abs(baseline_val - a) for a in test_y]
        mae_baseline += sum(baseline_errors) / len(baseline_errors) if baseline_errors else 0

    if n_folds > 0:
        mae_reg /= n_folds
        mae_baseline /= n_folds
        improvement = mae_baseline - mae_reg
        improvement_pct = 100 * improvement / mae_baseline if mae_baseline else 0
        print(f"Regression MAE: {mae_reg:.2f}")
        print(f"Historical‑average MAE: {mae_baseline:.2f}")
        print(f"Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")
    else:
        print("Could not compute cross‑validation results.")

    print()

# --------------------------------------------------------------------------- #
# NASA-20: Proposal selection comparison
# --------------------------------------------------------------------------- #

def _nasa_20(data):
    print("== NASA-20 ==")
    # Separate proposals (missing Effort- or Defects-)
    proposals = [r for r in data if not isinstance(r.get("Effort-"), (int, float))
                 or not isinstance(r.get("Defects-"), (int, float))]
    if not proposals:
        print("No proposals found.")
        return

    # Completed projects for training
    completed = [r for r in data if isinstance(r.get("Effort-"), (int, float))
                 and isinstance(r.get("Kloc+"), (int, float))]
    if len(completed) < 2:
        print("Not enough completed projects to build prediction models.")
        return

    # Build regression models
    train_x = _numeric_values(completed, "Kloc+")
    train_eff = _numeric_values(completed, "Effort-")
    try:
        e_inter, e_slope, _, _, _ = _linear_regression(train_x, train_eff)
    except Exception:
        print("Failed to build effort regression.")
        return

    # For defects, regress on Effort- using completed data
    eff_vals = train_eff
    defect_vals = _numeric_values(completed, "Defects-")
    try:
        d_inter, d_slope, _, _, _ = _linear_regression(eff_vals, defect_vals)
    except Exception:
        print("Failed to build defects regression.")
        return

    # Predict for each proposal and compute score (lower is better)
    proposal_scores = []
    for r in proposals:
        k = r.get("Kloc+")
        if not isinstance(k, (int, float)):
            continue
        est_eff = _predict(k, e_inter, e_slope)
        est_def = _predict(est_eff, d_inter, d_slope)
        score = est_eff + est_def
        proposal_scores.append((r.get("idX", "Unknown"), score))

    if not proposal_scores:
        print("No proposals with usable Kloc+ for prediction.")
        return

    # Determine evaluation budget: pick 3 proposals or fewer if not available
    budget = min(3, len(proposal_scores))

    # Our method: select proposals with lowest scores
    proposal_scores.sort(key=lambda kv: kv[1])
    selected = proposal_scores[:budget]
    avg_selected_score = statistics.mean([s for _, s in selected])

    # Random expectation: average score over all proposals
    all_scores = [s for _, s in proposal_scores]
    avg_random_score = statistics.mean(all_scores)

    print(f"Evaluation budget: {budget} proposals")
    print(f"Average score of selected proposals: {avg_selected_score:.2f}")
    print(f"Average score of random selection (all proposals): {avg_random_score:.2f}")
    improvement = avg_random_score - avg_selected_score
    improvement_pct = 100 * improvement / avg_random_score if avg_random_score else 0
    print(f"Improvement over random: {improvement:.2f} ({improvement_pct:.1f}%)")
    print()

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    data = _read_csv(csv_path)

    _nasa_19(data)
    _nasa_20(data)

    sys.exit(0)

if __name__ == "__main__":
    main()
