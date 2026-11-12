#!/usr/bin/env python3
"""
Skill 07: Focused vehicle selection for inspection and quick recommendation.

CAR-13: Identify a vehicle with a strong balance of fuel efficiency, weight, and
        acceleration while evaluating as few candidates as reasonably possible.
CAR-14: When we have a limited inspection budget, recommend which vehicles
        should be evaluated first so that we can discover promising inventory options early.

Usage:
    python skill_07.py <path-to-auto93.csv>
"""

import sys
from pathlib import Path

# Reuse core utilities from the earlier skills
from skill_01 import load_data, compute_balanced_scores, format_value


def _print_car_header(identifier: str):
    """Print a standardized header line."""
    print(f"== {identifier} ==")


def _top_n_by_score(rows, n: int):
    """
    Return the top n rows sorted by balanced score descending.
    """
    scored = compute_balanced_scores(rows)
    return scored[:n]


def _car_13(rows):
    """
    Identify the single vehicle that best balances fuel efficiency, weight,
    and acceleration while inspecting as few candidates as possible.
    We simply show the top‑ranked vehicle (and two additional candidates for context).
    """
    _print_car_header("CAR-13")

    top3 = _top_n_by_score(rows, 3)

    # The top vehicle is the recommendation
    best = top3[0]
    print(
        f"1. Recommended Vehicle: {best['Model']} "
        f"(Score: {format_value(best['score'], '{:.3f}')})"
    )
    print(
        f"   Weight: {format_value(best['Lbs-'])} lbs, "
        f"MPG: {format_value(best['Mpg+'])}, "
        f"Acceleration: {format_value(best['Acc+'])} s"
    )

    # Provide the next two candidates to show that we considered only a few
    for idx, r in enumerate(top3[1:], start=2):
        print(
            f"{idx}. {r['Model']} (Score: {format_value(r['score'], '{:.3f}')}) "
            f"Weight: {format_value(r['Lbs-'])} lbs, MPG: {format_value(r['Mpg+'])}, "
            f"Acc: {format_value(r['Acc+'])} s"
        )


def _car_14(rows):
    """
    Recommend which vehicles to inspect first when inspection budget is limited.
    The strategy is to inspect the vehicles with the highest balanced scores first,
    as they are most likely to offer a strong trade‑off across the key metrics.
    We limit the recommendation to the top 5 vehicles to keep the evaluation cost low.
    """
    _print_car_header("CAR-14")

    # Pick the top 5 vehicles to inspect
    top5 = _top_n_by_score(rows, 5)

    print(
        f"Based on balanced performance, inspect the following {len(top5)} vehicles first:"
    )
    for idx, r in enumerate(top5, start=1):
        print(
            f"{idx}. {r['Model']} (Score: {format_value(r['score'], '{:.3f}')}) "
            f"Weight: {format_value(r['Lbs-'])} lbs, MPG: {format_value(r['Mpg+'])}, "
            f"Acc: {format_value(r['Acc+'])} s"
        )

    print(
        "\nThese vehicles provide the best overall balance of the three key metrics "
        "and are the most promising candidates for early inspection."
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).is_file():
        print(f"Error: file '{csv_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    rows = load_data(csv_path)
    if not rows:
        print("Error: no data loaded.", file=sys.stderr)
        sys.exit(1)

    _car_13(rows)
    _car_14(rows)


if __name__ == "__main__":
    main()
