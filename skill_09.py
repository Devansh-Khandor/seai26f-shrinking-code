#!/usr/bin/env python3
"""
skill_09.py

Usage:
    python skill_09.py <csv_path>

This script implements the NASA-17 and NASA-18 queries:

  NASA-17 – Flag historical projects whose combinations of project
            characteristics are unusually different from the rest of
            our records so that we can review them.

  NASA-18 – Identify project records whose reported effort or defect
            counts differ substantially from what comparable projects
            would lead us to expect, so we can investigate possible
            measurement or recording issues.

The script re‑uses parsing, regression, and statistical helpers from
earlier skills to keep the implementation concise.
"""

import statistics
from typing import List, Tuple

# --------------------------------------------------------------------
# Imports from prior skill modules
# --------------------------------------------------------------------
try:
    # Parsing and regression helpers (effort & defects vs KLOC)
    from skill_02 import read_csv, linear_regression, predict  # noqa: F401
    # Basic statistics helper
    from skill_04 import compute_mean_std  # noqa: F401
    # Row type and parser that include categorical ratings
    from skill_03 import Row as Row3, parse_row  # noqa: F401
except Exception as e:
    raise RuntimeError(f"Failed to import helpers from previous modules: {e}")

# --------------------------------------------------------------------
# Local utilities
# --------------------------------------------------------------------
# Encode categorical ratings to numeric values for distance calculations
PMAT_MAP = {"l": 1.0, "m": 2.0, "h": 3.0, "vh": 4.0}
CPLX_MAP = {"l": 1.0, "m": 2.0, "h": 3.0, "vh": 4.0}

def encode_pmat(p: str) -> float:
    return PMAT_MAP.get(p.lower(), 0.0)

def encode_cplx(c: str) -> float:
    return CPLX_MAP.get(c.lower(), 0.0)

# --------------------------------------------------------------------
# NASA-17 – Flag anomalous characteristic combinations
# --------------------------------------------------------------------
def nasa_17(rows: List[Row3]) -> None:
    """
    Identify projects that are outliers in the combined space of
    Kloc, effort, defects, programmer capability and complexity.
    """
    print("\n== NASA-17 ==")

    if not rows:
        print("  (No data available.)")
        return

    # Build feature vectors
    vectors: List[List[float]] = []
    for r in rows:
        vec = [
            r.kloc,
            r.effort,
            r.defects,
            encode_pmat(r.pmat),
            encode_cplx(r.cplx),
        ]
        vectors.append(vec)

    # Compute centroid
    dims = len(vectors[0])
    centroid = [sum(vec[i] for vec in vectors) / len(vectors) for i in range(dims)]

    # Compute Euclidean distance of each point to the centroid
    distances: List[float] = []
    for vec in vectors:
        dist = sum((vec[i] - centroid[i]) ** 2 for i in range(dims)) ** 0.5
        distances.append(dist)

    mean_dist, std_dist = compute_mean_std(distances)
    threshold = mean_dist + 2 * std_dist

    # Flag outliers
    outliers = [
        (row.id, row, dist)
        for row, dist in zip(rows, distances)
        if dist > threshold
    ]

    if not outliers:
        print("  No projects are flagged as anomalous.")
        return

    # Display results
    header = f"{'idX':>4} {'Size(KLOC)':>10} {'Effort':>8} {'Defects':>8} {'Dist':>8}"
    print(header)
    for id_, r, dist in outliers:
        print(
            f"{id_:4d} {r.kloc:10.1f} {r.effort:8.1f} {r.defects:8.1f} {dist:8.2f}"
        )


# --------------------------------------------------------------------
# NASA-18 – Flag projects with unexpected effort or defect counts
# --------------------------------------------------------------------
def nasa_18(rows: List[Row3]) -> None:
    """
    Using simple linear regressions of effort and defects against KLOC,
    flag projects whose residuals are > mean + 2*std for either metric.
    """
    print("\n== NASA-18 ==")

    if not rows:
        print("  (No data available.)")
        return

    klocs = [r.kloc for r in rows]
    efforts = [r.effort for r in rows]
    defects = [r.defects for r in rows]

    # Regression models
    try:
        a_eff, b_eff = linear_regression(klocs, efforts)
        a_def, b_def = linear_regression(klocs, defects)
    except ValueError:
        print("  (Insufficient variation for regression.)")
        return

    # Predict and compute residuals
    effort_resid = [e - predict(k, a_eff, b_eff) for k, e in zip(klocs, efforts)]
    defect_resid = [d - predict(k, a_def, b_def) for k, d in zip(klocs, defects)]

    # Compute thresholds
    eff_mean, eff_std = compute_mean_std(effort_resid)
    def_mean, def_std = compute_mean_std(defect_resid)
    eff_thresh = eff_mean + 2 * eff_std
    def_thresh = def_mean + 2 * def_std

    # Flag projects
    flagged = []
    for r, e_res, d_res in zip(rows, effort_resid, defect_resid):
        if abs(e_res) > eff_thresh or abs(d_res) > def_thresh:
            flagged.append((r.id, r, e_res, d_res))

    if not flagged:
        print("  No projects exhibit unexpected effort or defect counts.")
        return

    # Display
    header = f"{'idX':>4} {'Eff. Resid':>10} {'Def. Resid':>10}"
    print(header)
    for id_, r, e_res, d_res in flagged:
        print(f"{id_:4d} {e_res:10.1f} {d_res:10.1f}")


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------
def main(csv_path: str) -> None:
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    nasa_17(rows)
    nasa_18(rows)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
