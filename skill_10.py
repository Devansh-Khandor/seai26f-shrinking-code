#!/usr/bin/env python3
"""
skill_10.py - NASA-19 and NASA-20: Evaluation of estimation quality and selection strategy.

Usage:
    python skill_10.py <path-to-nasa93dem.csv>
"""

import csv
import sys
import statistics
import random
from pathlib import Path
from typing import List, Dict, Tuple

# --------------------------------------------------------------------
# Import helpers from earlier modules
# --------------------------------------------------------------------
try:
    from skill_01 import load_data, compute_statistics
    from skill_02 import linear_regression
    from skill_07 import project_promise_score
except Exception as exc:
    raise RuntimeError("Could not import helper functions") from exc

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def split_train_test(projects: List[Dict], train_ratio: float = 0.7, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Shuffle and split projects into training and testing sets.
    """
    rng = random.Random(seed)
    shuffled = projects.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]

def mae(predictions: List[float], actuals: List[float]) -> float:
    """Mean absolute error."""
    return sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(actuals)

# --------------------------------------------------------------------
# NASA-19 implementation
# --------------------------------------------------------------------
def nasa_19(projects: List[Dict]) -> None:
    """
    Compare regression‑based effort estimates against a simple historical‑average estimate.
    """
    if len(projects) < 2:
        print("== NASA-19 ==")
        print("Not enough data to perform a split.")
        return

    train, test = split_train_test(projects)
    # Train regression on training data
    slope, intercept, _ = linear_regression(train, "Kloc+", "Effort-")

    # Historical average effort from training set
    avg_effort_train = statistics.mean(p["Effort-"] for p in train)

    # Generate predictions for the test set
    reg_preds = [slope * p["Kloc+"] + intercept for p in test]
    avg_preds = [avg_effort_train for _ in test]
    actuals = [p["Effort-"] for p in test]

    mae_reg = mae(reg_preds, actuals)
    mae_avg = mae(avg_preds, actuals)
    improvement = ((mae_avg - mae_reg) / mae_avg) * 100 if mae_avg else 0.0

    print("== NASA-19 ==")
    print(f"Test set size: {len(test)} projects")
    print(f"Regression MAE: {mae_reg:.2f} person‑months")
    print(f"Historical average MAE: {mae_avg:.2f} person‑months")
    print(f"MAE improvement over average: {improvement:.1f}%")

# --------------------------------------------------------------------
# NASA-20 implementation
# --------------------------------------------------------------------
def nasa_20(projects: List[Dict], budget: int = 5) -> None:
    """
    Compare the system's proposal prioritisation (NASA-14) with random selection.
    """
    if len(projects) == 0:
        print("== NASA-20 ==")
        print("No projects available.")
        return

    rng = random.Random(42)

    # Our selection: top projects by promise score
    scored = sorted(projects, key=lambda p: project_promise_score(p), reverse=True)
    our_selection = scored[:min(budget, len(scored))]

    # Random selection
    random_selection = rng.sample(projects, k=min(budget, len(projects)))

    def combined_score(sel: List[Dict]) -> float:
        """Average of effort + defects for the selected set."""
        return statistics.mean(p["Effort-"] + p["Defects-"] for p in sel)

    our_score = combined_score(our_selection)
    random_score = combined_score(random_selection)

    print("== NASA-20 ==")
    print(f"Evaluation budget: {budget} projects")
    print(f"Our selection average (effort+defects): {our_score:.2f}")
    print(f"Random selection average (effort+defects): {random_score:.2f}")
    diff = random_score - our_score
    print(f"Difference (lower is better): {diff:+.2f}")

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <path-to-nasa93dem.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    projects = load_data(csv_path)
    if not projects:
        print("Error: No valid project data found.")
        sys.exit(1)

    nasa_19(projects)
    print()
    nasa_20(projects, budget=5)


if __name__ == "__main__":
    main()
