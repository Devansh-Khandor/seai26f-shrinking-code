#!/usr/bin/env python3
"""
skill_07.py - NASA-13 and NASA-14: Promising project characteristic combinations and
                 proposal prioritisation under limited evaluation budgets.

NASA-13: Identify combinations of categorical project attributes that historically
        produced larger codebases (higher KLOC) while maintaining low effort per KLOC
        and low defects per KLOC.  The output lists the top 3 combinations
        with the best composite score.

NASA-14: Rank individual historical projects by a composite “promise” metric that
        rewards large KLOC with low effort and low defects per KLOC.  The
        top 5 projects are suggested as the first to evaluate in a limited
        evaluation budget.

The script re‑uses helpers from skill_01.py and performs all calculations
using the standard library only.

Usage:
    python skill_07.py <path-to-nasa93dem.csv>
"""

import sys
import statistics
from pathlib import Path
from typing import Dict, Tuple, List

# Import helpers from the earlier skill_01 module
try:
    from skill_01 import load_data, compute_statistics, find_best_worst
except Exception as exc:
    raise RuntimeError("Could not import helpers from skill_01.py") from exc


# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def project_ratios(project: Dict) -> Tuple[float, float]:
    """Return effort_per_KLOC and defect_per_KLOC for a project."""
    kloc = project["Kloc+"]
    effort = project["Effort-"]
    defects = project["Defects-"]
    if kloc <= 0:
        return (float("inf"), float("inf"))
    return (effort / kloc, defects / kloc)


def aggregate_combination(projects: List[Dict], key_fields: Tuple[str, ...]) -> List[Dict]:
    """
    Group projects by the specified categorical fields and compute
    mean KLOC, effort_per_kloc, defect_per_kloc for each group.
    Returns a list of dicts containing the group key and metrics.
    """
    groups: Dict[Tuple[str, ...], List[Dict]] = {}
    for p in projects:
        key = tuple(p.get(f, "") for f in key_fields)
        groups.setdefault(key, []).append(p)

    result = []
    for key, group in groups.items():
        total_kloc = sum(p["Kloc+"] for p in group)
        total_effort = sum(p["Effort-"] for p in group)
        total_defects = sum(p["Defects-"] for p in group)
        count = len(group)
        if total_kloc == 0:
            continue
        mean_kloc = total_kloc / count
        eff_per_kloc = total_effort / total_kloc
        def_per_kloc = total_defects / total_kloc
        # Composite score: higher KLOC, lower ratios
        score = mean_kloc / (eff_per_kloc + def_per_kloc + 1e-6)
        result.append(
            {
                "key": key,
                "mean_kloc": mean_kloc,
                "eff_per_kloc": eff_per_kloc,
                "def_per_kloc": def_per_kloc,
                "score": score,
                "count": count,
            }
        )
    return result


def project_promise_score(project: Dict) -> float:
    """Composite score for a single project."""
    kloc = project["Kloc+"]
    eff_per_kloc, def_per_kloc = project_ratios(project)
    if kloc <= 0:
        return -float("inf")
    # Higher KLOC, lower ratios
    return kloc / (eff_per_kloc + def_per_kloc + 1e-6)


# --------------------------------------------------------------------
# NASA-13 implementation
# --------------------------------------------------------------------
def nasa_13(projects: List[Dict]) -> None:
    """
    Prints the top 3 promising combinations of project characteristics.
    Combines the attributes 'acap', 'cplx', and 'team'.
    """
    key_fields = ("acap", "cplx", "team")
    groups = aggregate_combination(projects, key_fields)
    if not groups:
        print("== NASA-13 ==")
        print("No valid groups found.")
        return

    # Sort by composite score descending
    groups.sort(key=lambda g: g["score"], reverse=True)
    top = groups[:3]

    print("== NASA-13 ==")
    header = f"{'acap':>4} { 'cplx':>5} { 'team':>5}  {'#proj':>6}  {'KLOC':>6}  {'Eff/kloc':>8}  {'Def/kloc':>8}"
    print(header)
    for g in top:
        key = g["key"]
        print(
            f"{key[0]:>4} {key[1]:>5} {key[2]:>5}  {g['count']:>6}  {g['mean_kloc']:>6.1f}  {g['eff_per_kloc']:>8.2f}  {g['def_per_kloc']:>8.2f}"
        )


# --------------------------------------------------------------------
# NASA-14 implementation
# --------------------------------------------------------------------
def nasa_14(projects: List[Dict]) -> None:
    """
    Suggests the top 5 projects to evaluate first when the evaluation budget
    is limited.  Projects are ranked by a promise score that rewards large
    KLOC with low effort and low defects per KLOC.
    """
    scored = []
    for p in projects:
        score = project_promise_score(p)
        scored.append((score, p))

    # Sort by score descending
    scored.sort(key=lambda sp: sp[0], reverse=True)
    top_projects = [p for _, p in scored[:5]]

    print("== NASA-14 ==")
    header = f"{'idX':>4}  {'KLOC':>6}  {'Effort':>7}  {'Defects':>8}  {'Eff/kloc':>8}  {'Def/kloc':>8}"
    print(header)
    for p in top_projects:
        eff_per_kloc, def_per_kloc = project_ratios(p)
        print(
            f"{p['idX']:>4}  {p['Kloc+']:>6.1f}  {p['Effort-']:>7.1f}  {p['Defects-']:>8.1f}  {eff_per_kloc:>8.2f}  {def_per_kloc:>8.2f}"
        )


# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <path-to-nasa93dem.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    projects = load_data(csv_path)

    if not projects:
        print("Error: No valid project data found.")
        sys.exit(1)

    nasa_13(projects)
    print()
    nasa_14(projects)


if __name__ == "__main__":
    main()
