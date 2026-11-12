#!/usr/bin/env python3
"""
skill_10.py

Usage:
    python skill_10.py <csv_path>

This script implements the new client requests:
    NASA-19 – Compare our project-effort estimation system against a simple
              historical‑average estimate and show whether it makes
              meaningfully better predictions on previously unseen projects.
    NASA-20 – Compare our project‑selection approach with randomly selecting
              proposals under the same evaluation budget, and show whether
              our system identifies more desirable project outcomes.

Only the Python standard library is used.  The script prints the required
headers and at least three lines of results for each request.
"""

import csv
import math
import random
import sys
from statistics import mean

# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def _parse_float(value: str | None) -> float | None:
    """Parse a string into a float, or return None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean_std(values):
    """Return (mean, stddev) ignoring None."""
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return (None, None)
    m = sum(cleaned) / len(cleaned)
    var = sum((x - m) ** 2 for x in cleaned) / len(cleaned)
    return (m, math.sqrt(var))


# --------------------------------------------------------------------------- #
# Matrix utilities (copy of the ones in skill_04/06)
# --------------------------------------------------------------------------- #
def _matrix_mul(A, B):
    m, n = len(A), len(A[0])
    nB, p = len(B), len(B[0])
    assert n == nB, "Incompatible matrix sizes for multiplication."
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]


def _matrix_transpose(M):
    rows, cols = len(M), len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def _solve_linear(A, b):
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[pivot_row][k]) < 1e-12:
            raise ValueError("Singular or near‑singular matrix.")
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    x = [0.0] * n
    for i in reversed(range(n)):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


def _compute_regression(rows, predictors, target):
    """Perform ordinary least squares regression: target ~ 1 + predictors."""
    X = []
    y = []
    for r in rows:
        # Build a row with intercept
        row = [1.0]
        ok = True
        for p in predictors:
            val = r.get(p)
            if val is None:
                ok = False
                break
            row.append(val)
        if not ok:
            continue
        tgt = r.get(target)
        if tgt is None:
            continue
        X.append(row)
        y.append(tgt)

    if not X:
        raise ValueError("No complete rows for regression.")

    XT = _matrix_transpose(X)
    XT_X = _matrix_mul(XT, X)
    XT_y = _matrix_mul(XT, [[v] for v in y])
    XT_y = [row[0] for row in XT_y]

    coeff_vec = _solve_linear(XT_X, XT_y)

    coeff_dict = {"intercept": coeff_vec[0]}
    for idx, name in enumerate(predictors, start=1):
        coeff_dict[name] = coeff_vec[idx]
    return coeff_dict


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(csv_path):
    # Load CSV
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for r in reader:
            parsed = {
                "idX": r.get("idX"),
                "Kloc+": _parse_float(r.get("Kloc+")),
                "Effort-": _parse_float(r.get("Effort-")),
                "Defects-": _parse_float(r.get("Defects-")),
                "MonthsX": _parse_float(r.get("MonthsX")),
            }
            rows.append(parsed)

    # Separate completed projects (both Effort- and Defects- present)
    completed = [r for r in rows if r["Effort-"] is not None and r["Defects-"] is not None]
    # Proposals are those missing Effort-
    proposals = [r for r in rows if r["Effort-"] is None]

    # ------------------------------------------------------------------ NASA-19
    print("== NASA-19 ==")
    if len(completed) < 4:
        print("Insufficient completed projects for cross‑validation.")
        print("---")
    else:
        # Simple random split: 80% train, 20% test
        random.seed(42)
        indices = list(range(len(completed)))
        random.shuffle(indices)
        split = int(0.8 * len(completed))
        train_idx, test_idx = indices[:split], indices[split:]

        train_set = [completed[i] for i in train_idx]
        test_set = [completed[i] for i in test_idx]

        # Train regression on Effort- ~ 1 + Kloc+ + MonthsX
        try:
            coeffs = _compute_regression(train_set, ["Kloc+", "MonthsX"], "Effort-")
        except Exception:
            coeffs = None

        # Baseline: historical average Effort- from training set
        train_eff = [r["Effort-"] for r in train_set]
        base_avg = mean(train_eff)

        # Evaluate on test set
        preds = []
        actuals = []
        for r in test_set:
            if r["Kloc+"] is None or r["MonthsX"] is None:
                continue
            if coeffs is None:
                continue
            pred = coeffs["intercept"] + coeffs["Kloc+"] * r["Kloc+"] + coeffs["MonthsX"] * r["MonthsX"]
            preds.append(pred)
            actuals.append(r["Effort-"])

        if preds:
            mae_reg = mean(abs(p - a) for p, a in zip(preds, actuals))
            mae_base = mean(abs(base_avg - a) for a in actuals)
            improvement = (mae_base - mae_reg) / mae_base * 100 if mae_base else 0.0
            print(f"Regression MAE: {mae_reg:.2f} units")
            print(f"Baseline MAE:   {mae_base:.2f} units")
            print(f"Improvement:    {improvement:.1f}% lower error")
            if improvement > 5.0:
                print("Result: Regression model provides meaningfully better predictions.")
            else:
                print("Result: Regression model does not provide a significant improvement.")
        else:
            print("No test instances had sufficient predictors for evaluation.")
            print("---")

    # ------------------------------------------------------------------ NASA-20
    print("\n== NASA-20 ==")
    if not proposals:
        print("No proposals available for selection comparison.")
        print("---")
        return

    # Train regression on entire completed set for predictions
    try:
        coeffs_eff = _compute_regression(completed, ["Kloc+", "MonthsX"], "Effort-")
        coeffs_def = _compute_regression(completed, ["Kloc+", "Effort-"], "Defects-")
    except Exception:
        coeffs_eff = coeffs_def = None

    # Prepare proposal predictions
    proposal_data = []
    for r in proposals:
        if coeffs_eff and r["Kloc+"] is not None and r["MonthsX"] is not None:
            pred_eff = coeffs_eff["intercept"] + coeffs_eff["Kloc+"] * r["Kloc+"] + coeffs_eff["MonthsX"] * r["MonthsX"]
        else:
            pred_eff = None

        if coeffs_def and r["Kloc+"] is not None and pred_eff is not None:
            pred_def = coeffs_def["intercept"] + coeffs_def["Kloc+"] * r["Kloc+"] + coeffs_def["Effort-"] * pred_eff
        else:
            pred_def = None

        if pred_eff is not None and pred_def is not None:
            proposal_data.append({
                "id": r["idX"],
                "eff": pred_eff,
                "def": pred_def,
                "combined": pred_eff + pred_def,
            })

    if not proposal_data:
        print("No proposals could be predicted due to missing data.")
        print("---")
        return

    # Number of proposals we can evaluate
    eval_budget = min(5, len(proposal_data))
    print(f"Evaluation budget: {eval_budget} proposals")

    # Our selection: lowest combined metric
    selected = sorted(proposal_data, key=lambda x: x["combined"])[:eval_budget]
    selected_ids = [p["id"] for p in selected]
    selected_total = sum(p["eff"] + p["def"] for p in selected)

    # Random selection
    random.seed(123)
    random_selected = random.sample(proposal_data, eval_budget)
    random_ids = [p["id"] for p in random_selected]
    random_total = sum(p["eff"] + p["def"] for p in random_selected)

    # Output results
    print("\nOur selection (lowest predicted effort + defects):")
    for i, p in enumerate(selected, start=1):
        print(f"{i}. ID {p['id']} – Effort- ≈ {p['eff']:.1f}, Defects- ≈ {p['def']:.1f}, Combined ≈ {p['combined']:.1f}")
    print(f"Total predicted effort + defects: {selected_total:.1f}")

    print("\nRandom selection:")
    for i, p in enumerate(random_selected, start=1):
        print(f"{i}. ID {p['id']} – Effort- ≈ {p['eff']:.1f}, Defects- ≈ {p['def']:.1f}, Combined ≈ {p['combined']:.1f}")
    print(f"Total predicted effort + defects: {random_total:.1f}")

    # Compare
    if selected_total < random_total:
        diff = random_total - selected_total
        print("\nConclusion: Our systematic selection yields a lower total predicted cost "
              f"by {diff:.1f} units, indicating a more desirable portfolio.")
    else:
        diff = selected_total - random_total
        print("\nConclusion: Random selection performs better or equal by "
              f"{diff:.1f} units; our selection did not outperform random.")
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
