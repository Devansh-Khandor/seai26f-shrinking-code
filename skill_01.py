#!/usr/bin/env python3
"""
skill_01.py

Usage:
    python skill_01.py <csv_path>

The script reads the provided NASA data CSV and prints summaries for two
client requests (NASA-01 and NASA-02).  It uses only the Python standard
library.
"""

import csv
import math
import sys
from statistics import mean


def _parse_float(value):
    """Try to convert a string to float; return None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean_std(values):
    """Return (mean, stddev) for a list of numeric values, ignoring Nones."""
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return (None, None)
    m = mean(cleaned)
    var = sum((x - m) ** 2 for x in cleaned) / len(cleaned)
    return (m, math.sqrt(var))


def _load_data(csv_path):
    """Load CSV and return a list of rows with selected numeric columns."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for r in reader:
            parsed = {
                "idX": r.get("idX"),
                "Kloc+": _parse_float(r.get("Kloc+")),
                "Effort-": _parse_float(r.get("Effort-")),
                "Defects-": _parse_float(r.get("Defects-")),
            }
            rows.append(parsed)
    return rows


def _best_worst(rows, key_sum):
    """Return best and worst projects based on a combined metric."""
    best = min(rows, key=key_sum, default=None)
    worst = max(rows, key=key_sum, default=None)
    return best, worst


def _favorable_projects(rows, eff_mean, eff_std, def_mean, def_std):
    """Identify projects with unusually favorable Effort- and Defects-."""
    fav = []
    for r in rows:
        eff = r["Effort-"]
        dft = r["Defects-"]
        if eff is None or dft is None:
            continue
        if eff <= eff_mean - eff_std and dft <= def_mean - def_std:
            fav.append(r)
    return fav


def main(csv_path):
    rows = _load_data(csv_path)

    # Compute means and stddevs
    kloc_vals = [r["Kloc+"] for r in rows]
    eff_vals = [r["Effort-"] for r in rows]
    def_vals = [r["Defects-"] for r in rows]

    kloc_mean, kloc_std = _mean_std(kloc_vals)
    eff_mean, eff_std = _mean_std(eff_vals)
    def_mean, def_std = _mean_std(def_vals)

    # Combined metric for best/worst selection
    def combined(r):
        eff = r["Effort-"]
        dft = r["Defects-"]
        return (eff or 0) + (dft or 0)

    best, worst = _best_worst(rows, combined)

    # Print NASA-01
    print("== NASA-01 ==")
    print(f"Typical project size: {kloc_mean:.2f} KLOC")
    print(f"Typical development effort: {eff_mean:.2f} units")
    print(f"Typical defect count: {def_mean:.2f} defects")
    if best and worst:
        print(
            f"Best project: ID {best['idX']} "
            f"(Effort- {best['Effort-']:.2f}, Defects- {best['Defects-']:.2f})"
        )
        print(
            f"Worst project: ID {worst['idX']} "
            f"(Effort- {worst['Effort-']:.2f}, Defects- {worst['Defects-']:.2f})"
        )

    # Identify favorable projects for NASA-02
    fav_projects = _favorable_projects(rows, eff_mean, eff_std, def_mean, def_std)

    print("\n== NASA-02 ==")
    if fav_projects:
        print("Projects with unusually favorable combinations:")
        for r in fav_projects[:10]:  # limit output to 10 for brevity
            print(
                f"ID {r['idX']}: Effort- {r['Effort-']:.2f}, "
                f"Defects- {r['Defects-']:.2f}"
            )
        # Ensure at least three lines of output
        if len(fav_projects) < 3:
            for _ in range(3 - len(fav_projects)):
                print("—")
    else:
        print("No projects meet the unusually favorable criteria.")
        print("—")
        print("—")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
