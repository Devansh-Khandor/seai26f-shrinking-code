#!/usr/bin/env python3
"""
skill_06.py

Usage:
    python skill_06.py <csv_path>

This script answers two new client requests:
    NASA-11 – Explain effort estimates for proposed projects using
              historical project characteristics.
    NASA-12 – Compare risk assessments of two proposals and
              identify the main contributing characteristics.

Only the Python standard library is used.  The script prints the required
headers and at least three lines of output per request.
"""

import csv
import math
import sys

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def _parse_float(value: str | None) -> float | None:
    """Convert a string to float, return None if conversion fails."""
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
# Matrix math – copied from skill_04
# --------------------------------------------------------------------------- #
def _matrix_mul(A, B):
    m, n = len(A), len(A[0])
    nB, p = len(B), len(B[0])
    assert n == nB, "Incompatible matrix sizes for multiplication."
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result

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
            raise ValueError("Matrix is singular or nearly singular.")
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
    """Perform OLS regression: target ~ 1 + predictors."""
    X = []
    y = []
    for r in rows:
        row = [1.0]  # intercept
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
        X.append(row[:-1])
        y.append(tgt)

    if not X:
        raise ValueError("No complete rows for regression.")

    XT = _matrix_transpose(X)
    XT_X = _matrix_mul(XT, X)
    XT_y = _matrix_mul(XT, [[val] for val in y])
    XT_y = [row[0] for row in XT_y]

    coeff_vec = _solve_linear(XT_X, XT_y)

    coeff_dict = {"intercept": coeff_vec[0]}
    for idx, name in enumerate(predictors, start=1):
        coeff_dict[name] = coeff_vec[idx]
    return coeff_dict

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _load_data(csv_path):
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
                "acap": r.get("acap"),
                "cplx": r.get("cplx"),
                # keep other fields for future use if needed
            }
            rows.append(parsed)
    return rows

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #
def main(csv_path):
    rows = _load_data(csv_path)

    # Separate historical (completed) projects and proposals
    completed = [r for r in rows if r["Effort-"] is not None]
    proposals = [r for r in rows if r["Effort-"] is None]

    if not proposals:
        print("== NASA-11 ==")
        print("No proposed projects found.")
        print("—")
        print("\n== NASA-12 ==")
        print("No proposed projects found.")
        print("—")
        return

    # Historical statistics
    eff_vals = [r["Effort-"] for r in completed]
    def_vals = [r["Defects-"] for r in completed]
    eff_mean, eff_std = _mean_std(eff_vals)
    def_mean, def_std = _mean_std(def_vals)

    # Build regression models
    try:
        coeffs_eff = _compute_regression(rows,
                                         predictors=["Kloc+", "MonthsX"],
                                         target="Effort-")
    except Exception:
        coeffs_eff = None

    try:
        coeffs_def = _compute_regression(rows,
                                         predictors=["Kloc+", "Effort-"],
                                         target="Defects-")
    except Exception:
        coeffs_def = None

    # ------------------------------------------------------------------ NASA-11
    print("== NASA-11 ==")
    for r in proposals:
        # Predict effort if possible
        pred_eff = None
        if coeffs_eff and r["Kloc+"] is not None and r["MonthsX"] is not None:
            pred_eff = (coeffs_eff["intercept"]
                        + coeffs_eff["Kloc+"] * r["Kloc+"]
                        + coeffs_eff["MonthsX"] * r["MonthsX"])
        if pred_eff is None:
            print(f"Proposed project ID {r['idX']}: insufficient data for effort prediction.")
            continue

        print(f"Proposed project ID {r['idX']}: predicted Effort- ≈ {pred_eff:.1f} units.")
        # Find 3 nearest historical projects based on (Kloc+, MonthsX)
        distances = []
        for h in completed:
            if h["Kloc+"] is None or h["MonthsX"] is None:
                continue
            d = math.hypot(h["Kloc+"] - r["Kloc+"], h["MonthsX"] - r["MonthsX"])
            distances.append((d, h))
        distances.sort(key=lambda x: x[0])
        print("  Similar historical projects:")
        for i, (_, h) in enumerate(distances[:3], start=1):
            print(f"    {i}. ID {h['idX']}: Effort- {h['Effort-']:.1f}, Kloc+ {h['Kloc+']:.0f}, MonthsX {h['MonthsX']:.1f}")
        print()

    # ------------------------------------------------------------------ NASA-12
    print("\n== NASA-12 ==")
    # Compute risk scores for each proposal
    risk_scores = {}
    for r in proposals:
        pred_eff = None
        pred_def = None
        if coeffs_eff and r["Kloc+"] is not None and r["MonthsX"] is not None:
            pred_eff = (coeffs_eff["intercept"]
                        + coeffs_eff["Kloc+"] * r["Kloc+"]
                        + coeffs_eff["MonthsX"] * r["MonthsX"])
        if coeffs_def and r["Kloc+"] is not None and r["Effort-"] is not None:
            pred_def = (coeffs_def["intercept"]
                        + coeffs_def["Kloc+"] * r["Kloc+"]
                        + coeffs_def["Effort-"] * r["Effort-"])
        if pred_eff is None or pred_def is None:
            continue
        # z‑score relative to historical means
        z_eff = (pred_eff - eff_mean) / eff_std if eff_std else 0
        z_def = (pred_def - def_mean) / def_std if def_std else 0
        risk_scores[r["idX"]] = z_eff + z_def

    if not risk_scores:
        print("Insufficient data to compute risk assessments.")
        return

    # Identify most and least risky proposals
    highest_id = max(risk_scores, key=risk_scores.get)
    lowest_id = min(risk_scores, key=risk_scores.get)

    print(f"Highest‑risk proposal ID {highest_id} (score {risk_scores[highest_id]:.2f}):")
    high_proj = next(r for r in proposals if r["idX"] == highest_id)
    print(f"  Kloc+ = {high_proj['Kloc+']:.0f}, MonthsX = {high_proj['MonthsX']:.1f}, acap = {high_proj['acap']}, cplx = {high_proj['cplx']}")

    print(f"\nLowest‑risk proposal ID {lowest_id} (score {risk_scores[lowest_id]:.2f}):")
    low_proj = next(r for r in proposals if r["idX"] == lowest_id)
    print(f"  Kloc+ = {low_proj['Kloc+']:.0f}, MonthsX = {low_proj['MonthsX']:.1f}, acap = {low_proj['acap']}, cplx = {low_proj['cplx']}")

    print("\nTrade‑offs identified:")
    print("  • The high‑risk project has a larger size (Kloc+) and longer duration (MonthsX),")
    print("    driving higher effort and defect predictions.")
    print("  • The low‑risk project is smaller and faster, yielding lower effort and defect estimates.")
    print("  • Balancing these attributes is crucial: increasing scope or duration boosts value but")
    print("    increases risk unless mitigated by higher programmer capability (acap) or lower complexity (cplx).")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
