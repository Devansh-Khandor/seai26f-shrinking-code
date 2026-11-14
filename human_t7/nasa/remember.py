
"""NASA T7, Step 1: Remember (NASA-01 and NASA-02)."""

import csv
import math
import statistics
from pathlib import Path

DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "nasa93dem.csv"
)

GOALS = {
    "Kloc+": 1,       # Maximize
    "Effort-": -1,    # Minimize
    "Defects-": -1,   # Minimize
}


def load_projects(path=DATA_FILE):
    """Load projects, retaining their original CSV row numbers."""
    projects = []

    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("Missing CSV header")

        missing = set(GOALS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing goal columns: {missing}")

        for row_number, row in enumerate(reader, start=1):
            project = {
                key: value.strip()
                for key, value in row.items()
            }
            project["row_number"] = row_number

            for goal in GOALS:
                project[goal] = float(project[goal])

            projects.append(project)

    if not projects:
        raise ValueError("Dataset contains no projects")

    return projects


def summarize(projects):
    """Describe the distribution of each goal."""
    summary = {}

    for goal in GOALS:
        values = [project[goal] for project in projects]
        summary[goal] = {
            "minimum": min(values),
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "maximum": max(values),
        }

    return summary


def goal_bounds(projects):
    """Compute min/max for each goal."""
    return {
        goal: (
            min(project[goal] for project in projects),
            max(project[goal] for project in projects),
        )
        for goal in GOALS
    }


def distance_to_heaven(project, bounds):
    """Smaller scores mean more favorable combined goals."""
    squared_distances = []

    for goal, direction in GOALS.items():
        low, high = bounds[goal]

        if math.isclose(low, high):
            distance = 0.0
        else:
            normalized = (project[goal] - low) / (high - low)

            # For a maximized goal, heaven is 1.
            # For a minimized goal, heaven is 0.
            heaven = 1.0 if direction == 1 else 0.0
            distance = abs(heaven - normalized)

        squared_distances.append(distance ** 2)

    return math.sqrt(
        sum(squared_distances) / len(squared_distances)
    )


def rank_projects(projects):
    """Return the projects ranked from favorable to unfavorable."""
    bounds = goal_bounds(projects)

    ranked = []
    for project in projects:
        scored = project.copy()
        scored["d2h"] = distance_to_heaven(project, bounds)
        ranked.append(scored)

    return sorted(
        ranked,
        key=lambda project: (project["d2h"], project["row_number"]),
    )


def print_projects(projects):
    """Display a small ranked-project table."""
    print(
        f"{'Row':>5} {'KLOC':>10} {'Effort':>12}"
        f" {'Defects':>12} {'d2h':>10}"
    )

    for project in projects:
        print(
            f"{project['row_number']:>5}"
            f" {project['Kloc+']:>10.2f}"
            f" {project['Effort-']:>12.2f}"
            f" {project['Defects-']:>12.0f}"
            f" {project['d2h']:>10.4f}"
        )


def main():
    projects = load_projects()
    summary = summarize(projects)
    ranked = rank_projects(projects)

    print("NASA T7 — STEP 1: REMEMBER")
    print(f"\nLoaded {len(projects)} historical projects")

    print("\nNASA-01: HISTORICAL PROJECT SUMMARY")
    print(
        f"{'Goal':<12} {'Min':>12} {'Median':>12}"
        f" {'Mean':>12} {'Max':>12}"
    )

    for goal, stats in summary.items():
        print(
            f"{goal:<12}"
            f" {stats['minimum']:>12.2f}"
            f" {stats['median']:>12.2f}"
            f" {stats['mean']:>12.2f}"
            f" {stats['maximum']:>12.2f}"
        )

    print("\nNASA-02: FIVE MOST FAVORABLE PROJECTS")
    print_projects(ranked[:5])

    print("\nNASA-01: FIVE LEAST FAVORABLE PROJECTS")
    print_projects(ranked[-5:][::-1])

    print(
        "\nRanking uses normalized distance-to-heaven "
        "across all three goals."
    )


if __name__ == "__main__":
    main()
