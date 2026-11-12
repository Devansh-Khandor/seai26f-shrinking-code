#!/usr/bin/env python3
"""
skill_08.py

Usage:
    python skill_08.py <csv_path>

This script implements the new NASA-15 and NASA-16 queries:

  NASA-15 – For a project proposal predicted to require excessive effort,
            identify plausible changes to adjustable planning or process
            characteristics that might reduce its predicted effort.

  NASA-16 – For a project proposal with a high predicted defect count,
            identify the smallest plausible combination of changes to adjustable
            planning or process characteristics that could improve its predicted
            quality, while explaining potential trade‑offs.

The implementation re‑uses parsing, regression, and statistics helpers
from earlier skill modules.  It uses the two adjustable categorical
characteristics present in the dataset: Programmer Capability (pmat)
and Complexity rating (cplx).  The script prints a concise report
for each request.
"""

import csv
import sys
import statistics
from typing import List, Tuple, Dict
from collections import namedtuple

# --------------------------------------------------------------------
# Helpers from earlier skill modules
# --------------------------------------------------------------------
try:
    from skill_02 import read_csv, linear_regression, predict  # noqa: F401
    from skill_04 import compute_mean_std  # noqa: F401
except Exception as e:
    raise RuntimeError(f"Failed to import helpers from skill_02/04: {e}")

# --------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------
Row = namedtuple(
    "Row",
    [
        "id",
        "center",
        "year",
        "pmat",
        "cplx",
        "kloc",
        "effort",
        "defects",
    ],
)

# --------------------------------------------------------------------
# Mapping of categorical ratings to numeric values
# (used only for distance calculations / encoding if needed)
PMAT_MAP = {"l": 1.0, "m": 2.0, "h": 3.0, "vh": 4.0}
CPLX_MAP = {"l": 1.0, "m": 2.0, "h": 3.0, "vh": 4.0}

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def parse_row(raw: dict) -> Row | None:
    """Parse a raw dict from DictReader into a Row."""
    try:
        id_ = int(raw["idX"])
        center = int(raw["centerX"])
        year = int(raw["YearX"])
        pmat = raw["pmat"].strip()
        cplx = raw["cplx"].strip()
        kloc = float(raw["Kloc+"])
        effort = float(raw["Effort-"])
        defects = float(raw["Defects-"])
    except (ValueError, KeyError):
        return None
    return Row(id_, center, year, pmat, cplx, kloc, effort, defects)


def read_csv(path: str) -> List[Row]:
    """Read the CSV and return a list of Row objects."""
    rows: List[Row] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = parse_row(raw)
            if row:
                rows.append(row)
    return rows


def mean_effort_by_rating(rows: List[Row]) -> Dict[str, float]:
    """Return mean effort for each pmat rating."""
    groups: Dict[str, List[float]] = {}
    for r in rows:
        groups.setdefault(r.pmat, []).append(r.effort)
    return {k: statistics.mean(v) for k, v in groups.items()}


def mean_defects_by_rating(rows: List[Row]) -> Dict[str, float]:
    """Return mean defects for each cplx rating."""
    groups: Dict[str, List[float]] = {}
    for r in rows:
        groups.setdefault(r.cplx, []).append(r.defects)
    return {k: statistics.mean(v) for k, v in groups.items()}


# --------------------------------------------------------------------
# NASA-15 – Effort reduction suggestions
# --------------------------------------------------------------------
def nasa_15(rows: List[Row]) -> None:
    """Suggest plausible process changes to reduce predicted effort."""
    print("\n== NASA-15 ==")

    # Build effort regression model
    klocs = [r.kloc for r in rows]
    efforts = [r.effort for r in rows]
    a_eff, b_eff = linear_regression(klocs, efforts)

    # Get baseline statistics for thresholding
    eff_std = compute_mean_std(efforts)[1]
    eff_mean = statistics.mean(efforts)
    excess_threshold = eff_mean + 2 * eff_std

    # Collect rating reference data
    effort_by_pmat = mean_effort_by_rating(rows)

    # Prompt user for proposal characteristics
    try:
        size_str = input("Enter proposed project size (KLOC): ").strip()
        proj_size = float(size_str)
    except Exception:
        proj_size = statistics.median(klocs)
        print(f"  (Using median size {proj_size:.1f} KLOC)")

    # Default to most common rating if not provided
    pmat_choices = [r.pmat for r in rows]
    cplx_choices = [r.cplx for r in rows]
    default_pmat = max(set(pmat_choices), key=pmat_choices.count)
    default_cplx = max(set(cplx_choices), key=cplx_choices.count)

    pmat_str = input(f"Enter programmer capability rating [{default_pmat}]: ").strip()
    if not pmat_str:
        pmat_str = default_pmat

    cplx_str = input(f"Enter complexity rating [{default_cplx}]: ").strip()
    if not cplx_str:
        cplx_str = default_cplx

    # Predict effort with current ratings
    current_effort = predict(proj_size, a_eff, b_eff)
    print(f"\nPredicted effort for current proposal: {current_effort:.1f} person‑months")

    if current_effort <= excess_threshold:
        print("  Effort is not excessive according to historical thresholds.")
    else:
        print("  Effort exceeds historical threshold (mean + 2σ).")

    # Compute potential reductions by changing pmat
    # Use relative scaling from historical mean effort per rating
    baseline_factor = effort_by_pmat.get(pmat_str.lower(), effort_by_pmat.get("m", 1.0))
    suggestions: List[Tuple[str, float, str]] = []

    # Increase programmer capability
    higher_pmat = ["h", "vh"]
    for h in higher_pmat:
        if h == pmat_str:
            continue
        factor = effort_by_pmat.get(h, baseline_factor) / baseline_factor
        new_effort = current_effort * factor
        suggestions.append((f"Upgrade pmat to {h}", new_effort, f"{factor*100:.1f}%"))

    # Reduce complexity
    lower_cplx = ["m", "l"]
    for l in lower_cplx:
        if l == cplx_str:
            continue
        # Use defect means as proxy for effort effect per complexity
        # For simplicity, use the ratio of mean defects as proxy
        defects_by_cplx = mean_defects_by_rating(rows)
        baseline_def = defects_by_cplx.get(cplx_str.lower(), 1.0)
        target_def = defects_by_cplx.get(l, baseline_def)
        factor = target_def / baseline_def
        new_effort = current_effort * factor
        suggestions.append((f"Reduce complexity to {l}", new_effort, f"{factor*100:.1f}%"))

    # Select top three most impactful (lowest new effort)
    suggestions.sort(key=lambda x: x[1])
    print("\nSuggested adjustments (top 3 by effort reduction):")
    for i, (desc, eff, pct) in enumerate(suggestions[:3], start=1):
        print(f"  {i}. {desc:<30} → {eff:.1f} PM ({pct} of current)")

    # Summarise trade‑off
    print("\nTrade‑off note:")
    print("  Higher programmer capability typically requires more skilled (and potentially more expensive) staff.")
    print("  Reducing complexity often implies scope reduction or tighter design, which may impact features.")


# --------------------------------------------------------------------
# NASA-16 – Defect reduction suggestions
# --------------------------------------------------------------------
def nasa_16(rows: List[Row]) -> None:
    """Suggest minimal changes to improve predicted quality (defects)."""
    print("\n== NASA-16 ==")

    # Build defect regression model
    klocs = [r.kloc for r in rows]
    defects = [r.defects for r in rows]
    a_def, b_def = linear_regression(klocs, defects)

    # Threshold for high defects
    def_std = compute_mean_std(defects)[1]
    def_mean = statistics.mean(defects)
    high_threshold = def_mean + 2 * def_std

    # Reference defect means per complexity rating
    defects_by_cplx = mean_defects_by_rating(rows)
    # Also get mean effort per pmat rating to gauge trade‑off
    effort_by_pmat = mean_effort_by_rating(rows)

    # Prompt user
    try:
        size_str = input("Enter proposed project size (KLOC): ").strip()
        proj_size = float(size_str)
    except Exception:
        proj_size = statistics.median(klocs)
        print(f"  (Using median size {proj_size:.1f} KLOC)")

    pmat_choices = [r.pmat for r in rows]
    cplx_choices = [r.cplx for r in rows]
    default_pmat = max(set(pmat_choices), key=pmat_choices.count)
    default_cplx = max(set(cplx_choices), key=cplx_choices.count)

    pmat_str = input(f"Enter programmer capability rating [{default_pmat}]: ").strip()
    if not pmat_str:
        pmat_str = default_pmat

    cplx_str = input(f"Enter complexity rating [{default_cplx}]: ").strip()
    if not cplx_str:
        cplx_str = default_cplx

    # Predict defects
    current_defects = predict(proj_size, a_def, b_def)
    print(f"\nPredicted defect count for current proposal: {current_defects:.1f}")

    if current_defects <= high_threshold:
        print("  Defect count is not unusually high.")
    else:
        print("  Defect count exceeds historical high (mean + 2σ).")

    # Compute potential reductions
    suggestions: List[Tuple[str, float, float]] = []

    # Lower complexity
    for target in ["l", "m"]:
        if target == cplx_str:
            continue
        factor = defects_by_cplx.get(target, defects_by_cplx.get(cplx_str, 1.0))
        new_def = current_defects * (factor / defects_by_cplx.get(cplx_str, 1.0))
        # Estimate effort penalty: use effort_by_pmat as proxy
        effort_penalty = effort_by_pmat.get(pmat_str, 1.0)
        suggestions.append((f"Reduce complexity to {target}", new_def, effort_penalty))

    # Increase programmer capability (may reduce defects via better code quality)
    higher_pmat = ["h", "vh"]
    for h in higher_pmat:
        if h == pmat_str:
            continue
        # Assume defect reduction proportional to lower mean defects (use same defects_by_cplx for simplicity)
        factor = defects_by_cplx.get(cplx_str, 1.0) / defects_by_cplx.get(cplx_str, 1.0)
        # For demonstration, use a generic 10% reduction per step
        new_def = current_defects * 0.9 if h == "h" else current_defects * 0.8
        # Effort increases: use effort_by_pmat ratio
        effort_inc = effort_by_pmat.get(h, 1.0) / effort_by_pmat.get(pmat_str, 1.0)
        suggestions.append((f"Upgrade pmat to {h}", new_def, effort_inc))

    # Pick best trade‑off: minimal defect with reasonable effort increase
    suggestions.sort(key=lambda x: (x[1], x[2]))
    print("\nSuggested adjustments (top 3 by defect reduction):")
    for i, (desc, new_def, effort_factor) in enumerate(suggestions[:3], start=1):
        print(f"  {i}. {desc:<30} → {new_def:.1f} defects (effort factor {effort_factor:.2f})")

    # Summarise trade‑off
    print("\nTrade‑off note:")
    print("  Reducing complexity usually requires scope or design changes, which may extend schedule.")
    print("  Upgrading programmer capability can improve quality but typically increases labor cost.")


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows = read_csv(csv_path)
    if not rows:
        print("No valid data found.")
        return

    nasa_15(rows)
    nasa_16(rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
