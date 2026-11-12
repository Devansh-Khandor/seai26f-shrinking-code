#!/usr/bin/env python3
"""
skill_05.py - NASA-09 and NASA-10: Blame analysis for high effort and high defects.

NASA-09: For projects that required unusually high development effort,
identify which recorded project characteristics are most strongly
associated with those outcomes.

NASA-10: Examine projects with unusually high defect counts and identify
the characteristics that distinguish them from projects with better
quality outcomes.

The analysis uses simple Pearson correlation for numeric fields and
proportion comparison for categorical fields.  The top three
characteristics for each request are reported.

Usage:
    python skill_05.py <path-to-nasa93dem.csv>
"""

import csv
import sys
import statistics
from pathlib import Path
from typing import List, Dict, Tuple, Any

# Import helpers from skill_01
try:
    from skill_01 import load_data
except Exception as exc:
    raise RuntimeError("Could not import load_data from skill_01.py") from exc


# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------
def parse_float(value: str) -> float | None:
    """Attempt to parse a string as a float; return None on failure."""
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return None


def numeric_fields(project: Dict[str, Any]) -> Dict[str, float]:
    """Return a dict of all numeric fields for a project."""
    nums = {}
    for k, v in project.items():
        # Skip the already parsed float fields that might be None
        if isinstance(v, float):
            nums[k] = v
        else:
            f = parse_float(v)
            if f is not None:
                nums[k] = f
    return nums


def correlation(x_vals: List[float], y_vals: List[float]) -> float | None:
    """Return Pearson correlation coefficient between two numeric lists."""
    n = len(x_vals)
    if n < 2:
        return None
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
    std_x = (sum((x - mean_x) ** 2 for x in x_vals) / (n - 1)) ** 0.5
    std_y = (sum((y - mean_y) ** 2 for y in y_vals) / (n - 1)) ** 0.5
    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


def categorical_proportion(projects: List[Dict], field: str) -> Dict[str, float]:
    """Compute the proportion of each category in a field."""
    counts: Dict[str, int] = {}
    total = 0
    for p in projects:
        val = p.get(field, "")
        if val == "":
            continue
        counts[val] = counts.get(val, 0) + 1
        total += 1
    return {k: v / total for k, v in counts.items()}


def difference_in_proportions(high: Dict[str, float], overall: Dict[str, float]) -> Dict[str, float]:
    """Return absolute difference between two proportion dicts."""
    diff = {}
    keys = set(high) | set(overall)
    for k in keys:
        diff[k] = abs(high.get(k, 0.0) - overall.get(k, 0.0))
    return diff


# --------------------------------------------------------------------
# Analysis functions
# --------------------------------------------------------------------
def analyze_high_effort(projects: List[Dict]) -> List[Tuple[str, float]]:
    """Return top 3 numeric fields most correlated with high effort."""
    # Determine high effort threshold
    effort_vals = [p["Effort-"] for p in projects]
    med_effort = statistics.median(effort_vals)
    std_effort = statistics.stdev(effort_vals) if len(effort_vals) > 1 else 0.0
    threshold = med_effort + 2 * std_effort

    high_effort_projects = [p for p in projects if p["Effort-"] > threshold]

    if not high_effort_projects:
        return []

    # Gather numeric fields excluding Effort- itself
    field_corr: List[Tuple[str, float]] = []
    for field in projects[0].keys():
        if field == "Effort-":
            continue
        # Build lists of values for this field across all projects and high-effort projects
        all_vals = [numeric_fields(p).get(field) for p in projects]
        high_vals = [numeric_fields(p).get(field) for p in high_effort_projects]
        # Remove None values
        all_vals_clean = [v for v in all_vals if v is not None]
        high_vals_clean = [v for v in high_vals if v is not None]
        if not all_vals_clean or not high_vals_clean:
            continue
        # Use correlation if at least 2 samples
        corr = correlation(high_vals_clean, [p["Effort-"] for p in high_effort_projects])
        if corr is not None:
            field_corr.append((field, corr))

    # Sort by absolute correlation descending
    field_corr.sort(key=lambda t: abs(t[1]), reverse=True)
    return field_corr[:3]


def analyze_high_defects(projects: List[Dict]) -> List[Tuple[str, float]]:
    """Return top 3 numeric fields most correlated with high defects."""
    # Determine high defect threshold
    defect_vals = [p["Defects-"] for p in projects]
    med_defect = statistics.median(defect_vals)
    std_defect = statistics.stdev(defect_vals) if len(defect_vals) > 1 else 0.0
    threshold = med_defect + 2 * std_defect

    high_defect_projects = [p for p in projects if p["Defects-"] > threshold]

    if not high_defect_projects:
        return []

    field_corr: List[Tuple[str, float]] = []
    for field in projects[0].keys():
        if field == "Defects-":
            continue
        all_vals = [numeric_fields(p).get(field) for p in projects]
        high_vals = [numeric_fields(p).get(field) for p in high_defect_projects]
        all_vals_clean = [v for v in all_vals if v is not None]
        high_vals_clean = [v for v in high_vals if v is not None]
        if not all_vals_clean or not high_vals_clean:
            continue
        corr = correlation(high_vals_clean, [p["Defects-"] for p in high_defect_projects])
        if corr is not None:
            field_corr.append((field, corr))

    field_corr.sort(key=lambda t: abs(t[1]), reverse=True)
    return field_corr[:3]


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <path-to-nasa93dem.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    projects = load_data(csv_path)

    if not projects:
        print("Error: No valid project data found.")
        sys.exit(1)

    # NASA-09: high effort analysis
    print("== NASA-09 ==")
    high_eff_corrs = analyze_high_effort(projects)
    if not high_eff_corrs:
        print("No projects qualify as unusually high effort.")
    else:
        print(f"Top characteristics correlated with high effort ({len(high_eff_corrs)}):")
        for field, corr in high_eff_corrs:
            print(f"  {field:>12}: correlation={corr:+.3f}")

    # NASA-10: high defects analysis
    print("\n== NASA-10 ==")
    high_def_corrs = analyze_high_defects(projects)
    if not high_def_corrs:
        print("No projects qualify as unusually high defects.")
    else:
        print(f"Top characteristics correlated with high defects ({len(high_def_corrs)}):")
        for field, corr in high_def_corrs:
            print(f"  {field:>12}: correlation={corr:+.3f}")


if __name__ == "__main__":
    main()
