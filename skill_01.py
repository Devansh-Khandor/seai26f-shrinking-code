#!/usr/bin/env python3
"""
Skill 01: Summary and identification of favorable software projects.

Usage:
    python skill_01.py <path_to_nasa93dem.csv>
"""

import csv
import sys
import math
from collections import defaultdict

def read_data(csv_path):
    """Read CSV and return list of project dictionaries with numeric fields."""
    projects = []
    required_fields = ["idX", "Kloc+", "Effort-", "Defects-"]
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                proj = {
                    "id": row["idX"].strip(),
                    "Kloc": float(row["Kloc+"].strip()),
                    "Effort": float(row["Effort-"].strip()),
                    "Defects": float(row["Defects-"].strip())
                }
                # Skip rows with non-positive size
                if proj["Kloc"] <= 0:
                    continue
                projects.append(proj)
            except (ValueError, KeyError):
                # Skip rows with missing or bad numeric data
                continue
    return projects

def compute_summary(projects):
    """Compute average size, effort, defects and best/worst effort."""
    n = len(projects)
    if n == 0:
        return None

    total_size = sum(p["Kloc"] for p in projects)
    total_effort = sum(p["Effort"] for p in projects)
    total_defects = sum(p["Defects"] for p in projects)

    avg_size = total_size / n
    avg_effort = total_effort / n
    avg_defects = total_defects / n

    # Best and worst effort
    best = min(projects, key=lambda p: p["Effort"])
    worst = max(projects, key=lambda p: p["Effort"])

    return {
        "avg_size": avg_size,
        "avg_effort": avg_effort,
        "avg_defects": avg_defects,
        "best_effort": best,
        "worst_effort": worst
    }

def identify_favorable(projects):
    """Identify projects with favorable effort and defect densities."""
    if not projects:
        return []

    # Compute densities
    for p in projects:
        p["effort_density"] = p["Effort"] / p["Kloc"]
        p["defect_density"] = p["Defects"] / p["Kloc"]

    # Determine median thresholds
    densities_effort = sorted(p["effort_density"] for p in projects)
    densities_defect = sorted(p["defect_density"] for p in projects)
    med_effort = densities_effort[len(densities_effort)//2]
    med_defect = densities_defect[len(densities_defect)//2]

    # Select projects below or equal to medians
    favorable = [p for p in projects
                 if p["effort_density"] <= med_effort and p["defect_density"] <= med_defect]
    return favorable

def format_number(val, decimals=2):
    """Format number with specified decimal places, removing trailing zeros."""
    if math.isnan(val) or math.isinf(val):
        return "N/A"
    return f"{val:.{decimals}f}".rstrip('0').rstrip('.')

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    projects = read_data(csv_path)

    # NASA-01
    print("== NASA-01 ==")
    summary = compute_summary(projects)
    if summary:
        print(f"Average project size: {format_number(summary['avg_size'])} Kloc+")
        print(f"Average effort: {format_number(summary['avg_effort'])} man-months")
        print(f"Average defects: {format_number(summary['avg_defects'])}")
        print(f"Best effort: {format_number(summary['best_effort']['Effort'])} on project {summary['best_effort']['id']}")
        print(f"Worst effort: {format_number(summary['worst_effort']['Effort'])} on project {summary['worst_effort']['id']}")
    else:
        print("No valid project data found.")

    # NASA-02
    print("\n== NASA-02 ==")
    favorable = identify_favorable(projects)
    if favorable:
        ids = ', '.join(p["id"] for p in favorable)
        print(f"Favorable projects (low effort & defects per size): {ids}")
        print(f"Count: {len(favorable)}")
        # Show top 3 by lowest effort density
        top3 = sorted(favorable, key=lambda p: p["effort_density"])[:3]
        for p in top3:
            print(f"  - Project {p['id']}: Effort density {format_number(p['effort_density'])}, "
                  f"Defect density {format_number(p['defect_density'])}")
    else:
        print("No projects meet the favorable criteria.")

if __name__ == "__main__":
    main()
