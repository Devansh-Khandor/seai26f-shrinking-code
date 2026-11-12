#!/usr/bin/env python3
"""
Skill 01: Inventory summary and best combination finder for the auto93 dataset.

Usage:
    python skill_01.py <path_to_auto93.csv>
"""

import sys
import csv
import math
from collections import namedtuple

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

Vehicle = namedtuple("Vehicle", ["Clndrs", "Volume", "HpX", "Model",
                                "origin", "Lbs", "Acc", "Mpg"])

# --------------------------------------------------------------------------- #
# Data loading utilities
# --------------------------------------------------------------------------- #

def load_vehicles(csv_path):
    """
    Load vehicles from the provided CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the inventory.

    Returns
    -------
    list[Vehicle]
        A list of Vehicle objects.
    """
    vehicles = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vehicle = Vehicle(
                    Clndrs=int(row["Clndrs"]),
                    Volume=int(row["Volume"]),
                    HpX=int(row["HpX"]),
                    Model=int(row["Model"]),
                    origin=int(row["origin"]),
                    Lbs=int(row["Lbs-"]),
                    Acc=float(row["Acc+"]),
                    Mpg=float(row["Mpg+"]),
                )
                vehicles.append(vehicle)
            except (KeyError, ValueError) as exc:
                # Skip malformed rows
                continue
    return vehicles

# --------------------------------------------------------------------------- #
# Summary statistics for CAR-01
# --------------------------------------------------------------------------- #

def compute_stats(vehicles):
    """
    Compute mean, min, max for weight, mpg, and acceleration.
    """
    if not vehicles:
        return {}
    weights = [v.Lbs for v in vehicles]
    mpgs = [v.Mpg for v in vehicles]
    accs = [v.Acc for v in vehicles]
    return {
        "weight_mean": sum(weights) / len(weights),
        "weight_min": min(weights),
        "weight_max": max(weights),
        "mpg_mean": sum(mpgs) / len(mpgs),
        "mpg_min": min(mpgs),
        "mpg_max": max(mpgs),
        "acc_mean": sum(accs) / len(accs),
        "acc_min": min(accs),
        "acc_max": max(accs),
    }

def format_stats(stats):
    """
    Return a list of formatted lines describing the stats.
    """
    lines = [
        f"Mean vehicle weight: {stats['weight_mean']:.1f} lbs",
        f"Weight range: {stats['weight_min']} – {stats['weight_max']} lbs",
        f"Mean fuel efficiency: {stats['mpg_mean']:.1f} mpg",
        f"Fuel efficiency range: {stats['mpg_min']} – {stats['mpg_max']} mpg",
        f"Mean acceleration: {stats['acc_mean']:.2f} s",
        f"Acceleration range: {stats['acc_min']} – {stats['acc_max']} s",
    ]
    return lines

# --------------------------------------------------------------------------- #
# CAR-02: best combination ranking
# --------------------------------------------------------------------------- #

def compute_z_scores(values):
    """
    Compute z-scores for a list of numeric values.
    """
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    if std == 0:
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]

def best_combination(vehicles, top_n=3):
    """
    Return the top_n vehicles ranked by a composite score that
    rewards low weight, high mpg, and low acceleration.
    """
    weights = [v.Lbs for v in vehicles]
    mpgs = [v.Mpg for v in vehicles]
    accs = [v.Acc for v in vehicles]

    # z-scores: weight & acceleration lower is better => negate
    weight_z = compute_z_scores(weights)
    mpg_z = compute_z_scores(mpgs)
    acc_z = compute_z_scores(accs)

    scores = []
    for i, v in enumerate(vehicles):
        # Composite score: lower weight (+), higher mpg (+), lower acceleration (+)
        score = -weight_z[i] + mpg_z[i] - acc_z[i]
        scores.append((score, i, v))

    # Sort descending by score
    scores.sort(reverse=True, key=lambda t: t[0])
    return scores[:top_n]

def format_top_vehicles(top_vehicles):
    """
    Return a list of formatted lines for the best vehicles.
    """
    lines = []
    for score, idx, v in top_vehicles:
        lines.append(
            f"Vehicle {idx + 1}: weight={v.Lbs} lbs, mpg={v.Mpg}, acc={v.Acc}s, "
            f"score={score:.4f}"
        )
    return lines

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def main(csv_path):
    vehicles = load_vehicles(csv_path)

    # CAR-01
    print("== CAR-01 ==")
    stats = compute_stats(vehicles)
    for line in format_stats(stats):
        print(line)

    # CAR-02
    print("\n== CAR-02 ==")
    top_vehicles = best_combination(vehicles, top_n=3)
    for line in format_top_vehicles(top_vehicles):
        print(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_01.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
