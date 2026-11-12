#!/usr/bin/env python3
"""
Skill 02: Fuel‑efficiency and weight estimation for the auto93 dataset.

Usage:
    python skill_02.py <path_to_auto93.csv>
"""

import sys
import csv
import math
import numpy as np
from sklearn.linear_model import LinearRegression

# Import all functions and the Vehicle namedtuple from skill_01
from skill_01 import (
    Vehicle,
    compute_stats,
    format_stats,
    best_combination,
    format_top_vehicles,
)

# --------------------------------------------------------------------------- #
# Data loading utilities – extended to allow missing mpg values
# --------------------------------------------------------------------------- #

def load_vehicles(csv_path):
    """
    Load vehicles from the provided CSV file. Missing mpg values
    are stored as ``None``.
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
                    Mpg=float(row["Mpg+"]) if row["Mpg+"] else None,
                )
                vehicles.append(vehicle)
            except (KeyError, ValueError):
                # Skip malformed rows
                continue
    return vehicles

# --------------------------------------------------------------------------- #
# CAR‑01: summary statistics – updated to ignore missing mpg values
# --------------------------------------------------------------------------- #

def compute_stats(vehicles):
    """
    Compute mean, min, max for weight, mpg (when available), and acceleration.
    """
    if not vehicles:
        return {}
    weights = [v.Lbs for v in vehicles]
    accs = [v.Acc for v in vehicles]
    mpgs = [v.Mpg for v in vehicles if v.Mpg is not None]

    stats = {
        "weight_mean": sum(weights) / len(weights),
        "weight_min": min(weights),
        "weight_max": max(weights),
        "acc_mean": sum(accs) / len(accs),
        "acc_min": min(accs),
        "acc_max": max(accs),
    }
    if mpgs:
        stats.update({
            "mpg_mean": sum(mpgs) / len(mpgs),
            "mpg_min": min(mpgs),
            "mpg_max": max(mpgs),
        })
    return stats

# --------------------------------------------------------------------------- #
# CAR‑03: estimate missing mpg values using linear regression
# --------------------------------------------------------------------------- #

def estimate_mpg(vehicles):
    """
    Return the updated vehicle list with estimated mpg for missing values
    and a list of predictions ``[(index, predicted_mpg), ...]``.
    """
    known = [(i, v) for i, v in enumerate(vehicles) if v.Mpg is not None]
    unknown = [(i, v) for i, v in enumerate(vehicles) if v.Mpg is None]

    if not unknown:
        return vehicles, []

    X = np.array([[v.Clndrs, v.Volume, v.HpX, v.Model, v.origin, v.Lbs, v.Acc] for _, v in known])
    y = np.array([v.Mpg for _, v in known])

    model = LinearRegression()
    model.fit(X, y)

    predictions = []
    for idx, v in unknown:
        x = np.array([[v.Clndrs, v.Volume, v.HpX, v.Model, v.origin, v.Lbs, v.Acc]])
        pred = model.predict(x)[0]
        predictions.append((idx, pred))
        # Replace vehicle with estimated mpg
        vehicles[idx] = v._replace(Mpg=pred)

    return vehicles, predictions

# --------------------------------------------------------------------------- #
# CAR‑04: estimate vehicle weight and uncertainty
# --------------------------------------------------------------------------- #

def predict_weights_with_uncertainty(vehicles):
    """
    For each vehicle with known mpg, predict weight and return
    a list ``[(index, vehicle, predicted_weight, std_residual), ...]``.
    """
    # Use only vehicles with known weight and mpg
    known = [v for v in vehicles if v.Lbs is not None and v.Mpg is not None]
    if not known:
        return []

    X = np.array([[v.Clndrs, v.Volume, v.HpX, v.Model, v.origin, v.Mpg, v.Acc] for v in known])
    y = np.array([v.Lbs for v in known])

    model = LinearRegression()
    model.fit(X, y)

    residuals = y - model.predict(X)
    std_residual = math.sqrt(np.mean(residuals ** 2))

    predictions = []
    for idx, v in enumerate(vehicles):
        if v.Mpg is not None:
            x = np.array([[v.Clndrs, v.Volume, v.HpX, v.Model, v.origin, v.Mpg, v.Acc]])
            pred = model.predict(x)[0]
            predictions.append((idx, v, pred, std_residual))

    return predictions

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def main(csv_path):
    vehicles = load_vehicles(csv_path)

    # CAR‑01
    print("== CAR-01 ==")
    stats = compute_stats(vehicles)
    for line in format_stats(stats):
        print(line)

    # CAR‑02
    print("\n== CAR-02 ==")
    top_vehicles = best_combination(vehicles, top_n=3)
    for line in format_top_vehicles(top_vehicles):
        print(line)

    # CAR‑03
    print("\n== CAR-03 ==")
    vehicles, mpg_predictions = estimate_mpg(vehicles)
    if mpg_predictions:
        for i, (idx, pred) in enumerate(mpg_predictions[:3], 1):
            print(f"Estimated mpg for vehicle {idx + 1}: {pred:.2f} mpg")
    else:
        print("No missing mpg values to estimate.")

    # CAR‑04
    print("\n== CAR-04 ==")
    weight_preds = predict_weights_with_uncertainty(vehicles)
    if weight_preds:
        for i, (idx, v, pred, std) in enumerate(weight_preds[:3], 1):
            print(f"Estimated weight for vehicle {idx + 1}: {pred:.0f} ± {std:.0f} lbs")
    else:
        print("Insufficient data to estimate weights.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_02.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
