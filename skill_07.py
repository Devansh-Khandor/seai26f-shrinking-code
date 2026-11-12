#!/usr/bin/env python3
"""
skill_07.py

Usage:
    python skill_07.py <csv_path>

This script answers two new client requests:
    NASA-13 – Identify promising combinations of project characteristics that
              historically delivered more software with lower development effort
              and fewer defects.
    NASA-14 – Recommend which project proposals should be evaluated first to
              discover promising alternatives with a limited evaluation budget.

Only the Python standard library is used.  The script prints the required
headers and at least three lines of results for each request.
"""

import csv
import math
import sys
from collections import defaultdict, namedtuple

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _parse_float(value: str | None) -> float | None:
    """Parse a string to float, return None if parsing fails."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _matrix_mul(A, B):
    """Multiply two matrices (lists of lists)."""
    m, n = len(A), len(A[0])
    nB, p = len(B), len(B[0])
    assert n == nB, "Matrix size mismatch"
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]


def _matrix_transpose(M):
    """Transpose a matrix."""
    rows, cols = len(M), len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def _solve_linear(A, b):
    """Solve Ax = b for x using Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for k in range(n):
        # Pivot selection
        pivot_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[pivot_row][k]) < 1e-12:
            raise ValueError("Singular matrix")
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
    """
    Perform ordinary least squares regression: target ~ 1 + predictors.
    Returns a dict of coefficients {name: value}, with key 'intercept'.
    """
    X = []
    y = []
    for r in rows:
        # Build row with intercept
        row = [1.0]
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


# --------------------------------------------------------------------------- #
# Main
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
                "time": r.get("time"),
                "tool": r.get("tool"),
            }
            rows.append(parsed)

    # Separate historical projects (completed) and proposals (missing Effort- or Defects-)
    historical = [r for r in rows if r["Effort-"] is not None and r["Defects-"] is not None]
    proposals = [r for r in rows if r["Effort-"] is None or r["Defects-"] is None]

    # ------------------------------------------------------------------ NASA-13
    print("== NASA-13 ==")
    if not historical:
        print("No completed projects to analyze.")
    else:
        # Group by acap, cplx, time, tool (exclude rows with missing group keys)
        group_stats = defaultdict(lambda: {"kloc_sum": 0.0, "effort_sum": 0.0, "defect_sum": 0.0, "cnt": 0})
        for r in historical:
            key = (r["acap"], r["cplx"], r["time"], r["tool"])
            if None in key:
                continue
            group_stats[key]["kloc_sum"] += r["Kloc+"]
            group_stats[key]["effort_sum"] += r["Effort-"]
            group_stats[key]["defect_sum"] += r["Defects-"]
            group_stats[key]["cnt"] += 1

        # Compute mean stats per group and efficiency metrics
        GroupResult = namedtuple("GroupResult", ["key", "kloc", "effort", "defect",
                                                "eff_per_kloc", "def_per_kloc"])
        group_list = []
        for key, vals in group_stats.items():
            if vals["cnt"] == 0:
                continue
            kloc_mean = vals["kloc_sum"] / vals["cnt"]
            effort_mean = vals["effort_sum"] / vals["cnt"]
            defect_mean = vals["defect_sum"] / vals["cnt"]
            eff_per_kloc = effort_mean / kloc_mean if kloc_mean else float("inf")
            def_per_kloc = defect_mean / kloc_mean if kloc_mean else float("inf")
            group_list.append(GroupResult(key, kloc_mean, effort_mean, defect_mean,
                                          eff_per_kloc, def_per_kloc))

        # Rank groups by lowest combined efficiency metric
        group_list.sort(key=lambda g: (g.eff_per_kloc, g.def_per_kloc))

        # Print top 3 promising combinations (or fewer)
        header = f"{'Group (acap,cplx,time,tool)':>40} | {'Kloc+':>8} | {'Effort-':>8} | {'Defects-':>8} | {'Eff/Kloc+':>10} | {'Def/Kloc+':>10}"
        print(header)
        print("-" * len(header))
        for g in group_list[:3]:
            print(f"{str(g.key):>40} | {g.kloc:8.1f} | {g.effort:8.1f} | {g.defect:8.1f} | {g.eff_per_kloc:10.4f} | {g.def_per_kloc:10.4f}")

    # ------------------------------------------------------------------ NASA-14
    print("\n== NASA-14 ==")
    if not proposals:
        print("No project proposals to evaluate.")
    else:
        # Build regression models from historical data
        try:
            coeffs_eff = _compute_regression(historical,
                                             predictors=["Kloc+", "MonthsX"],
                                             target="Effort-")
        except Exception:
            coeffs_eff = None
        try:
            coeffs_def = _compute_regression(historical,
                                             predictors=["Kloc+", "Effort-"],
                                             target="Defects-")
        except Exception:
            coeffs_def = None

        if coeffs_eff is None or coeffs_def is None:
            print("Insufficient data to build regression models for predictions.")
        else:
            # Predict for each proposal and compute efficiency metrics
            ProposalResult = namedtuple("ProposalResult",
                                        ["id", "pred_eff", "pred_def",
                                         "eff_per_kloc", "def_per_kloc",
                                         "combined"])
            proposal_list = []

            for r in proposals:
                k = r["Kloc+"]
                m = r["MonthsX"]
                if k is None:
                    continue
                # Predict Effort-
                if m is None:
                    continue
                pred_eff = (coeffs_eff["intercept"] +
                            coeffs_eff["Kloc+"] * k +
                            coeffs_eff["MonthsX"] * m)
                # Predict Defects- (requires predicted effort)
                pred_def = (coeffs_def["intercept"] +
                            coeffs_def["Kloc+"] * k +
                            coeffs_def["Effort-"] * pred_eff)
                eff_per_kloc = pred_eff / k if k else float("inf")
                def_per_kloc = pred_def / k if k else float("inf")
                combined = eff_per_kloc + def_per_kloc
                proposal_list.append(ProposalResult(r["idX"],
                                                   pred_eff, pred_def,
                                                   eff_per_kloc, def_per_kloc,
                                                   combined))

            # Rank proposals by lowest combined metric
            proposal_list.sort(key=lambda p: p.combined)

            # Print top 5 proposals (or fewer)
            header = f"{'ID':>6} | {'Pred Effort-':>12} | {'Pred Defects-':>13} | {'Eff/Kloc+':>10} | {'Def/Kloc+':>10} | {'Combined':>9}"
            print(header)
            print("-" * len(header))
            for p in proposal_list[:5]:
                print(f"{p.id:>6} | {p.pred_eff:12.1f} | {p.pred_def:13.1f} | "
                      f"{p.eff_per_kloc:10.4f} | {p.def_per_kloc:10.4f} | {p.combined:9.4f}")

            if len(proposal_list) > 5:
                print(f"... and {len(proposal_list)-5} more proposals not shown.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
