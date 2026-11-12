#!/usr/bin/env python3
"""
skill_08.py - NASA-15 and NASA-16: Recommended adjustments to reduce effort or defects.

NASA-15: For a project proposal predicted to require excessive effort, identify
plausible changes to adjustable planning or process characteristics that
might reduce its predicted effort.

NASA-16: For a project proposal with a high predicted defect count, identify
the smallest plausible combination of changes to adjustable characteristics
that could improve its predicted quality, while explaining potential trade‑offs.

The script re‑uses helpers from skill_01.py and skill_02.py and requires
only the Python standard library.
"""

import sys
import statistics
from pathlib import Path
from typing import Dict, Tuple, List

# Import helpers from the earlier modules
try:
    from skill_01 import load_data, compute_statistics, find_best_worst
    from skill_02 import linear_regression, estimate_for_median
except Exception as exc:
    raise RuntimeError("Could not import helpers from skill_01 or skill_02") from exc

# --------------------------------------------------------------------
# Parsing utilities
# --------------------------------------------------------------------
def read_proposal() -> Tuple[float, Dict[str, str]]:
    """
    Read proposal details from stdin.

    First line: numeric KLOC value.
    Subsequent lines (optional) may contain key=value pairs, e.g.
        acap=h, cplx=l, team=v, pmat=h
    """
    lines = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    if not lines:
        raise ValueError("No input provided. Expecting at least one line for KLOC.")
    try:
        kloc = float(lines[0].strip())
    except ValueError:
        raise ValueError("First line must be a numeric KLOC value.")
    attrs: Dict[str, str] = {}
    for line in lines[1:]:
        if "=" in line:
            key, val = line.split("=", 1)
            attrs[key.strip()] = val.strip()
    return kloc, attrs

# --------------------------------------------------------------------
# Helper functions for attribute statistics
# --------------------------------------------------------------------
def attribute_ratio_means(projects: List[Dict], field: str, metric: str) -> Dict[str, float]:
    """
    Compute mean ratio (effort/KLOC or defects/KLOC) for each value of a categorical field.
    metric must be 'eff' or 'def'.
    """
    values: Dict[str, List[float]] = {}
    for p in projects:
        val = p.get(field, "")
        if val == "":
            continue
        if metric == "eff":
            ratio = p["Effort-"] / p["Kloc+"]
        else:
            ratio = p["Defects-"] / p["Kloc+"]
        values.setdefault(val, []).append(ratio)
    # Calculate mean
    return {k: sum(vs) / len(vs) for k, vs in values.items() if vs}

def best_value(means: Dict[str, float]) -> Tuple[str, float]:
    """Return the value with the lowest mean ratio and the mean itself."""
    if not means:
        return "", float("inf")
    key = min(means, key=means.get)
    return key, means[key]

# --------------------------------------------------------------------
# NASA-15 implementation
# --------------------------------------------------------------------
def nasa_15(projects: List[Dict], kloc: float, attrs: Dict[str, str]) -> str:
    """
    Identify changes that could reduce predicted effort.
    """
    # Build regression for effort
    slope_eff, intercept_eff, _ = linear_regression(projects, "Kloc+", "Effort-")
    effort_pred = slope_eff * kloc + intercept_eff

    # Current attribute values
    adjustable_fields = ["acap", "cplx", "team", "pmat"]
    suggestions = []

    # Compute mean ratios per value
    eff_means = {f: attribute_ratio_means(projects, f, "eff") for f in adjustable_fields}

    for field in adjustable_fields:
        current_val = attrs.get(field, "")
        if not current_val:
            # Skip if the attribute not supplied
            continue
        best_val, best_ratio = best_value(eff_means[field])
        if not best_val or best_val == current_val:
            continue
        # Estimate potential reduction using the best ratio
        current_ratio = eff_means[field][current_val]
        improv_ratio = current_ratio - best_ratio
        improv_effort = kloc * improv_ratio
        suggestions.append(
            f"Change '{field}' from '{current_val}' to '{best_val}' "
            f"could lower effort by {improv_effort:.1f} person‑months (≈{improv_ratio:.3f} KLOC‑wise)."
        )

    lines = ["== NASA-15 =="]
    lines.append(f"Input KLOC: {kloc:.1f}")
    lines.append(f"Predicted effort: {effort_pred:.1f} person‑months")
    if suggestions:
        lines.append("Suggested adjustments to reduce effort:")
        lines.extend(suggestions)
    else:
        lines.append("No adjustable characteristics identified that improve effort.")
    return "\n".join(lines)

# --------------------------------------------------------------------
# NASA-16 implementation
# --------------------------------------------------------------------
def nasa_16(projects: List[Dict], kloc: float, attrs: Dict[str, str]) -> str:
    """
    Identify minimal combination of changes to reduce predicted defects.
    """
    # Build regression for defects
    slope_def, intercept_def, _ = linear_regression(projects, "Kloc+", "Defects-")
    defects_pred = slope_def * kloc + intercept_def

    adjustable_fields = ["acap", "cplx", "team", "pmat"]
    suggestions = []

    def_means = {f: attribute_ratio_means(projects, f, "def") for f in adjustable_fields}

    for field in adjustable_fields:
        current_val = attrs.get(field, "")
        if not current_val:
            continue
        best_val, best_ratio = best_value(def_means[field])
        if not best_val or best_val == current_val:
            continue
        current_ratio = def_means[field][current_val]
        improv_ratio = current_ratio - best_ratio
        improv_defects = kloc * improv_ratio
        suggestions.append(
            f"Change '{field}' from '{current_val}' to '{best_val}' "
            f"could lower defects by {improv_defects:.1f} (≈{improv_ratio:.3f} KLOC‑wise)."
        )

    lines = ["== NASA-16 =="]
    lines.append(f"Input KLOC: {kloc:.1f}")
    lines.append(f"Predicted defects: {defects_pred:.1f}")
    if suggestions:
        lines.append("Minimal changes that improve quality:")
        lines.extend(suggestions)
    else:
        lines.append("No adjustable characteristics identified that improve defect count.")
    return "\n".join(lines)

# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <path-to-nasa93dem.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    projects = load_data(csv_path)
    if not projects:
        print("Error: No valid project data found.")
        sys.exit(1)

    try:
        kloc, attrs = read_proposal()
    except Exception as exc:
        print(f"Error reading proposal: {exc}")
        sys.exit(1)

    # NASA-15
    print(nasa_15(projects, kloc, attrs))

    # NASA-16
    print("\n" + nasa_16(projects, kloc, attrs))

if __name__ == "__main__":
    main()
