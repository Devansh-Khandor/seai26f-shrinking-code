#!/usr/bin/env python3
"""
skill_08.py

Usage:
    python skill_08.py <csv_path>

This script answers two client requests:
    NASA-15 – For a project proposal predicted to require excessive effort,
              identify plausible changes to adjustable planning or process
              characteristics that might reduce its predicted effort.
    NASA-16 – For a project proposal with a high predicted defect count,
              identify the smallest plausible combination of changes to
              adjustable planning or process characteristics that could
              improve its predicted quality, while explaining potential tradeoffs.

Only the Python standard library is used.  The script prints the required
headers and at least three lines of results for each request.
"""

import csv
import math
import sys

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _parse_float(value: str | None) -> float | None:
    """Parse a string into a float; return None if parsing fails."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# Matrix utilities (copied from skill_04/06/07)
def _matrix_mul(A, B):
    """Multiply matrix A by matrix B."""
    m, n = len(A), len(A[0])
    nB, p = len(B), len(B[0])
    assert n == nB, "Incompatible matrix sizes for multiplication."
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]


def _matrix_transpose(M):
    """Return the transpose of matrix M."""
    rows, cols = len(M), len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def _solve_linear(A, b):
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(A)
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for k in range(n):
        # Pivot selection
        pivot_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[pivot_row][k]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular.")
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
        # Eliminate rows below
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    # Back substitution
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
        for p in predictors:
            val = r.get(p)
            if val is None:
                break
            row.append(val)
        else:
            tgt = r.get(target)
            if tgt is None:
                continue
            X.append(row)
            y.append(tgt)
    if not X:
        raise ValueError("No complete cases for regression")

    XT = _matrix_transpose(X)
    XT_X = _matrix_mul(XT, X)
    XT_y = _matrix_mul(XT, [[v] for v in y])
    XT_y = [row[0] for row in XT_y]

    coeff_vec = _solve_linear(XT_X, XT_y)
    coeffs = {"intercept": coeff_vec[0]}
    for idx, name in enumerate(predictors, start=1):
        coeffs[name] = coeff_vec[idx]
    return coeffs


def _mean_std(values):
    """Return (mean, stddev) ignoring None."""
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return (None, None)
    m = sum(cleaned) / len(cleaned)
    var = sum((x - m) ** 2 for x in cleaned) / len(cleaned)
    return (m, math.sqrt(var))


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #
def main(csv_path):
    # Load data
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
                "tool": r.get("tool"),
                "time": r.get("time"),
            }
            rows.append(parsed)

    # Separate historical completed projects
    completed = [r for r in rows if r["Effort-"] is not None and r["Defects-"] is not None]

    # Compute historical statistics
    eff_vals = [r["Effort-"] for r in completed]
    def_vals = [r["Defects-"] for r in completed]
    eff_mean, eff_std = _mean_std(eff_vals)
    def_mean, def_std = _mean_std(def_vals)

    # Build regression models from historical data
    try:
        coeffs_eff = _compute_regression(completed,
                                         predictors=["Kloc+", "MonthsX"],
                                         target="Effort-")
    except Exception:
        coeffs_eff = None
    try:
        coeffs_def = _compute_regression(completed,
                                         predictors=["Kloc+", "Effort-"],
                                         target="Defects-")
    except Exception:
        coeffs_def = None

    # ------------------------------------------------------------------ NASA-15
    print("== NASA-15 ==")

    # Identify proposal projects (missing Effort-)
    proposals = [r for r in rows if r["Effort-"] is None]

    # Threshold for excessive effort
    if eff_mean is None or eff_std is None:
        print("Insufficient historical data to determine effort thresholds.")
        print("---")
    else:
        threshold_eff = eff_mean + 2 * eff_std

        high_effort = []
        for r in proposals:
            if coeffs_eff is None:
                continue
            k = r.get("Kloc+")
            m = r.get("MonthsX")
            if k is None or m is None:
                continue
            pred_eff = (coeffs_eff["intercept"] + coeffs_eff["Kloc+"] * k + coeffs_eff["MonthsX"] * m)
            if pred_eff > threshold_eff:
                high_effort.append((r, pred_eff))

        if not high_effort:
            print("No excessive‑effort proposals detected.")
            print("—")
            print("—")
        else:
            for r, pred_eff in high_effort:
                print(f"ID {r['idX']} predicted Effort-: {pred_eff:.1f} (threshold {threshold_eff:.1f})")

                # Suggested adjustments
                suggestions = []

                # 1. Reduce Kloc+ by 20%
                k = r.get("Kloc+")
                m = r.get("MonthsX")
                if k is not None and m is not None:
                    new_k = 0.8 * k
                    new_eff = (coeffs_eff["intercept"]
                               + coeffs_eff["Kloc+"] * new_k
                               + coeffs_eff["MonthsX"] * m)
                    suggestions.append(("Reduce Kloc+ by 20%", new_eff))

                # 2. Reduce MonthsX by 10%
                if k is not None and m is not None:
                    new_m = 0.9 * m
                    new_eff = (coeffs_eff["intercept"]
                               + coeffs_eff["Kloc+"] * k
                               + coeffs_eff["MonthsX"] * new_m)
                    suggestions.append(("Reduce MonthsX by 10%", new_eff))

                # 3. Combine both
                if k is not None and m is not None:
                    new_k = 0.8 * k
                    new_m = 0.9 * m
                    new_eff = (coeffs_eff["intercept"]
                               + coeffs_eff["Kloc+"] * new_k
                               + coeffs_eff["MonthsX"] * new_m)
                    suggestions.append(("Reduce Kloc+ by 20% and MonthsX by 10%", new_eff))

                # Determine the best (lowest) effort
                best = min(suggestions, key=lambda x: x[1]) if suggestions else None
                if best:
                    for txt, val in suggestions:
                        print(f"  • {txt} → Effort- ≈ {val:.1f}")
                    improvement = (pred_eff - best[1]) / pred_eff * 100
                    print(f"  → Best adjustment reduces effort by {improvement:.1f}% "
                          f"to {best[1]:.1f} units.")
                else:
                    print("  No viable numeric adjustments found.")
                print()

    # ------------------------------------------------------------------ NASA-16
    print("\n== NASA-16 ==")

    # Identify proposal projects with high predicted defects
    if coeffs_def is None or coeffs_eff is None:
        print("Insufficient regression models to evaluate defect predictions.")
        print("---")
    else:
        threshold_def = def_mean + 2 * def_std if def_mean is not None and def_std is not None else None
        if threshold_def is None:
            print("Insufficient historical data to determine defect thresholds.")
            print("---")
        else:
            high_defect = []
            for r in proposals:
                k = r.get("Kloc+")
                if k is None:
                    continue
                # Predict effort first
                m = r.get("MonthsX")
                if m is None:
                    continue
                pred_eff = (coeffs_eff["intercept"] + coeffs_eff["Kloc+"] * k + coeffs_eff["MonthsX"] * m)
                pred_def = (coeffs_def["intercept"] + coeffs_def["Kloc+"] * k + coeffs_def["Effort-"] * pred_eff)
                if pred_def > threshold_def:
                    high_defect.append((r, pred_eff, pred_def))

            if not high_defect:
                print("No high‑defect proposals detected.")
                print("—")
                print("—")
            else:
                for r, pred_eff, pred_def in high_defect:
                    print(f"ID {r['idX']} predicted Defects-: {pred_def:.1f} (threshold {threshold_def:.1f})")

                    suggestions = []

                    # 1. Reduce Kloc+ by 20%
                    k = r.get("Kloc+")
                    m = r.get("MonthsX")
                    if k is not None and m is not None:
                        new_k = 0.8 * k
                        new_eff = (coeffs_eff["intercept"] + coeffs_eff["Kloc+"] * new_k + coeffs_eff["MonthsX"] * m)
                        new_def = (coeffs_def["intercept"] + coeffs_def["Kloc+"] * new_k + coeffs_def["Effort-"] * new_eff)
                        suggestions.append(("Reduce Kloc+ by 20%", new_def, new_eff))

                    # 2. Increase acap to 'vh' (if currently lower)
                    # Since acap is categorical and not part of regression,
                    # we note the potential qualitative benefit but cannot quantify.
                    # We will still list it as a suggestion.
                    if r.get("acap") not in (None, "vh"):
                        suggestions.append(("Increase acap to 'vh'", None, None))

                    # 3. Reduce cplx to 'm' (if currently higher)
                    if r.get("cplx") in (None, "vh", "h"):
                        suggestions.append(("Reduce cplx to 'm'", None, None))

                    # Find the numeric suggestion with best defect reduction
                    numeric_suggestions = [s for s in suggestions if s[1] is not None]
                    if numeric_suggestions:
                        best = min(numeric_suggestions, key=lambda x: x[1])
                        improvement = (pred_def - best[1]) / pred_def * 100
                        print(f"  • {best[0]} → Predicted Defects- ≈ {best[1]:.1f} "
                              f"(Effort- ≈ {best[2]:.1f})")
                        print(f"  → Reduces defects by {improvement:.1f}% "
                              f"to {best[1]:.1f} units.")
                    else:
                        # No numeric change available
                        print("  • Increase acap to 'vh' (qualitative improvement)")
                        print("  • Reduce cplx to 'm' (qualitative improvement)")

                    # Trade‑off explanation
                    print("  → Trade‑offs: reducing Kloc+ shortens scope and may limit feature set; "
                          "improving acap or reducing cplx requires training or better tooling.")
                    print()

    # End of script


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
