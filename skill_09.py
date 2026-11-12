#!/usr/bin/env python3
"""
skill_09.py

Usage:
    python skill_09.py <csv_path>

This script implements the new client requests:
    NASA-17 – Flag historical projects that are unusually different from
              the rest of the record, based on multiple characteristics.
    NASA-18 – Flag projects whose effort or defect counts deviate
              substantially from what comparable projects would predict.

Only the Python standard library is used.  Each request prints a header line
(`== NASA-17 ==` or `== NASA-18 ==`) followed by at least three lines
containing the analysis results, including numeric values.
"""

import csv
import math
import sys
from statistics import mean

# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def _parse_float(value: str | None) -> float | None:
    """Parse a string into a float; return None if conversion fails."""
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
# Simple linear‑regression utilities (OLS)
# --------------------------------------------------------------------------- #
def _matrix_mul(A, B):
    m, n = len(A), len(A[0])
    nB, p = len(B), len(B[0])
    assert n == nB, "Matrix size mismatch in multiplication."
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]


def _matrix_transpose(M):
    rows, cols = len(M), len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def _solve_linear(A, b):
    """Solve A x = b via Gaussian elimination with partial pivoting."""
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
    """
    OLS regression: target ~ 1 + predictors.
    Returns (coefficients dict, residual_std).
    """
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
        X.append(row)
        y.append(tgt)

    if not X:
        raise ValueError("No complete cases for regression.")

    XT = _matrix_transpose(X)
    XT_X = _matrix_mul(XT, X)
    XT_y = _matrix_mul(XT, [[v] for v in y])
    XT_y = [row[0] for row in XT_y]
    coeff_vec = _solve_linear(XT_X, XT_y)

    coeffs = {"intercept": coeff_vec[0]}
    for idx, name in enumerate(predictors, start=1):
        coeffs[name] = coeff_vec[idx]

    # Residual standard deviation
    n = len(X)
    p = len(coeff_vec)
    residuals = []
    for i in range(n):
        pred = sum(coeff_vec[j] * X[i][j] for j in range(p))
        residuals.append(y[i] - pred)
    rss = sum(r ** 2 for r in residuals)
    resid_std = math.sqrt(rss / (n - p))

    return coeffs, resid_std


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
                "time": r.get("time"),
                "acap": r.get("acap"),
                "cplx": r.get("cplx"),
                "tool": r.get("tool"),
            }
            rows.append(parsed)

    # Identify completed projects (both Effort- and Defects- present)
    completed = [r for r in rows if r["Effort-"] is not None and r["Defects-"] is not None]

    # ------------------------------------------------------------------ NASA-17
    print("== NASA-17 ==")

    if not completed:
        print("No completed projects to analyze.")
        print("—")
        print("—")
    else:
        # Compute mean & std for each numeric field across completed projects
        numeric_keys = ["Kloc+", "MonthsX", "Effort-", "Defects-"]
        stats = {k: _mean_std([r[k] for r in completed]) for k in numeric_keys}

        # For each project compute Mahalanobis‑like distance
        distances = []
        for r in completed:
            z_scores = []
            for k in numeric_keys:
                m, s = stats[k]
                if s == 0 or s is None:
                    z = 0.0
                else:
                    z = (r[k] - m) / s
                z_scores.append(z)
            dist = math.sqrt(sum(z ** 2 for z in z_scores))
            distances.append((dist, r))

        # Compute statistics on distances
        dist_values = [d for d, _ in distances]
        d_mean, d_std = _mean_std(dist_values)

        # Threshold: mean + 2*std
        threshold = (d_mean or 0) + 2 * (d_std or 0)

        flagged = [(dist, r) for dist, r in distances if dist > threshold]
        flagged.sort(reverse=True, key=lambda x: x[0])

        if not flagged:
            print("No anomalous projects detected.")
            print("—")
            print("—")
        else:
            # Show up to first 3 flagged projects
            for i, (dist, r) in enumerate(flagged[:3], start=1):
                print(f"{i}. ID {r['idX']} – Mahalanobis‑like distance: {dist:.3f}")
            if len(flagged) > 3:
                print(f"... and {len(flagged)-3} more anomalies.")

    # ------------------------------------------------------------------ NASA-18
    print("\n== NASA-18 ==")

    if not completed:
        print("No completed projects to analyze.")
        print("—")
        print("—")
    else:
        # Build regression models
        try:
            coeffs_eff, resid_std_eff = _compute_regression(
                completed, predictors=["Kloc+", "MonthsX"], target="Effort-")
        except Exception:
            coeffs_eff = None
            resid_std_eff = None

        try:
            coeffs_def, resid_std_def = _compute_regression(
                completed, predictors=["Kloc+", "Effort-"], target="Defects-")
        except Exception:
            coeffs_def = None
            resid_std_def = None

        # Compute residuals for each completed project
        flagged_eff = []
        flagged_def = []

        for r in completed:
            # Effort residual
            if coeffs_eff:
                pred_eff = (coeffs_eff["intercept"]
                            + coeffs_eff["Kloc+"] * r["Kloc+"]
                            + coeffs_eff["MonthsX"] * r["MonthsX"])
                resid_eff = r["Effort-"] - pred_eff
                if resid_std_eff and abs(resid_eff) > 2 * resid_std_eff:
                    flagged_eff.append((abs(resid_eff), r, pred_eff))

            # Defects residual
            if coeffs_def:
                pred_def = (coeffs_def["intercept"]
                            + coeffs_def["Kloc+"] * r["Kloc+"]
                            + coeffs_def["Effort-"] * r["Effort-"])
                resid_def = r["Defects-"] - pred_def
                if resid_std_def and abs(resid_def) > 2 * resid_std_def:
                    flagged_def.append((abs(resid_def), r, pred_def))

        # Combine and sort by largest deviation
        flagged = flagged_eff + flagged_def
        flagged.sort(reverse=True, key=lambda x: x[0])

        if not flagged:
            print("No projects with substantial residuals detected.")
            print("—")
            print("—")
        else:
            for i, (abs_res, r, pred) in enumerate(flagged[:3], start=1):
                # Determine type (Effort or Defects)
                if abs_res in (x[0] for x in flagged_eff):
                    observed = r["Effort-"]
                    pred_val = pred
                    label = "Effort-"
                else:
                    observed = r["Defects-"]
                    pred_val = pred
                    label = "Defects-"
                print(f"{i}. ID {r['idX']} – {label} observed {observed:.1f}, predicted {pred_val:.1f}, residual {abs_res:.1f}")
            if len(flagged) > 3:
                print(f"... and {len(flagged)-3} more projects with large residuals.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
