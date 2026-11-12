#!/usr/bin/env python3
"""
Skill 09: Outlier detection and record consistency checks.

This script implements:
  CAR-17 – Flag vehicle records whose combination of characteristics
           is unusually different from the rest of our inventory.
  CAR-18 – Identify vehicles whose reported fuel efficiency or weight
           differs substantially from what similar vehicles would lead us
           to expect, enabling staff to verify potential recording errors.
"""

import sys
import math
import numpy as np
from collections import defaultdict

# Re‑use utilities from earlier skills
from skill_02 import load_vehicles
from skill_01 import compute_z_scores, Vehicle

# --------------------------------------------------------------------------- #
# Helper: z‑score outlier detection for a list of numeric features
# --------------------------------------------------------------------------- #
def detect_z_outliers(vehicles, threshold=2.0):
    """
    Return a list of tuples (idx, outlier_features, z_scores_dict) for vehicles
    where at least one feature has an absolute z‑score > threshold.
    """
    # Extract all numeric feature values
    features = ["Clndrs", "Volume", "HpX", "Model", "origin", "Lbs", "Acc", "Mpg"]
    values = defaultdict(list)
    for v in vehicles:
        for f in features:
            values[f].append(getattr(v, f))

    # Compute z‑scores per feature
    z_dict = {}
    for f in features:
        z_dict[f] = compute_z_scores(values[f])

    # Identify outliers
    outliers = []
    for idx, v in enumerate(vehicles):
        outlier_feats = []
        z_scores = {}
        for f in features:
            z = z_dict[f][idx]
            if abs(z) > threshold:
                outlier_feats.append(f)
                z_scores[f] = z
        if outlier_feats:
            outliers.append((idx, outlier_feats, z_scores))
    return outliers

# --------------------------------------------------------------------------- #
# Helper: regression prediction with residuals and standard deviation
# --------------------------------------------------------------------------- #
def _fit_predict_and_residual_std(vehicles, feature_names, target_name, include_missing_target=True):
    """
    Fit a linear regression model to predict target_name from feature_names.
    Returns predictions for all vehicles and the residual standard deviation
    computed on the training set.
    """
    # Build training data
    X_train = []
    y_train = []
    idx_train = []
    for i, v in enumerate(vehicles):
        if getattr(v, target_name) is None:
            continue
        X_train.append([getattr(v, f) for f in feature_names])
        y_train.append(getattr(v, target_name))
        idx_train.append(i)

    if not X_train:
        return [], None  # No training data

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions for all vehicles
    predictions = []
    for v in vehicles:
        x = np.array([getattr(v, f) for f in feature_names]).reshape(1, -1)
        predictions.append(model.predict(x)[0])

    # Residuals on training data
    train_pred = model.predict(X_train)
    residuals = y_train - train_pred
    std_resid = math.sqrt(np.mean(residuals ** 2))

    return predictions, std_resid

# --------------------------------------------------------------------------- #
# CAR-17 implementation
# --------------------------------------------------------------------------- #
def car_17_outlier_detection(vehicles, threshold=2.0, max_report=10):
    """
    Detect vehicles with unusually extreme combinations of characteristics.
    Returns a list of strings for printing.
    """
    outliers = detect_z_outliers(vehicles, threshold=threshold)
    lines = []
    if not outliers:
        lines.append("No significant outliers detected.")
        return lines

    lines.append(f"Detected {len(outliers)} vehicles with extreme characteristics (|z|>{threshold}).")
    for i, (idx, feats, zscores) in enumerate(outliers[:max_report], 1):
        feats_str = ", ".join(feats)
        z_str = ", ".join(f"{f}={zscores[f]:+.2f}" for f in feats)
        lines.append(f"{i}. Vehicle #{idx + 1} (idx={idx}): {feats_str} -> z-scores: {z_str}")
    if len(outliers) > max_report:
        lines.append(f"...and {len(outliers)-max_report} more.")
    return lines

# --------------------------------------------------------------------------- #
# CAR-18 implementation
# --------------------------------------------------------------------------- #
def car_18_consistency_check(vehicles, threshold=2.0, max_report=10):
    """
    Identify vehicles whose reported mpg or weight differs substantially
    from predictions based on other features.
    Returns a list of strings for printing.
    """
    lines = []

    # ---- Fuel‑efficiency consistency ----
    mpg_features = ["Clndrs", "Volume", "HpX", "Model", "origin", "Lbs", "Acc"]
    mpg_pred, mpg_std = _fit_predict_and_residual_std(vehicles, mpg_features, "Mpg")
    mpg_flags = []
    for idx, v in enumerate(vehicles):
        if v.Mpg is None:
            continue
        residual = v.Mpg - mpg_pred[idx]
        if abs(residual) > threshold * mpg_std:
            mpg_flags.append((idx, v.Mpg, mpg_pred[idx], residual, mpg_std))

    # ---- Weight consistency ----
    weight_features = ["Clndrs", "Volume", "HpX", "Model", "origin", "Mpg", "Acc"]
    weight_pred, weight_std = _fit_predict_and_residual_std(vehicles, weight_features, "Lbs")
    weight_flags = []
    for idx, v in enumerate(vehicles):
        if v.Lbs is None:
            continue
        residual = v.Lbs - weight_pred[idx]
        if abs(residual) > threshold * weight_std:
            weight_flags.append((idx, v.Lbs, weight_pred[idx], residual, weight_std))

    # ---- Reporting ----
    lines.append(f"Fuel‑efficiency consistency check: {len(mpg_flags)} out of {len(vehicles)} vehicles flagged.")
    for i, (idx, actual, pred, resid, std) in enumerate(mpg_flags[:max_report], 1):
        lines.append(
            f"{i}. Vehicle #{idx + 1}: Actual MPG={actual:.2f}, Predicted={pred:.2f}, "
            f"Residual={resid:.2f} ({threshold}×σ={threshold*std:.2f})"
        )
    if len(mpg_flags) > max_report:
        lines.append(f"...and {len(mpg_flags)-max_report} more.")

    lines.append("")  # blank line separator

    lines.append(f"Weight consistency check: {len(weight_flags)} vehicles flagged.")
    for i, (idx, actual, pred, resid, std) in enumerate(weight_flags[:max_report], 1):
        lines.append(
            f"{i}. Vehicle #{idx + 1}: Actual Lbs={actual}, Predicted={pred:.1f}, "
            f"Residual={resid:.1f} ({threshold}×σ={threshold*std:.1f})"
        )
    if len(weight_flags) > max_report:
        lines.append(f"...and {len(weight_flags)-max_report} more.")

    return lines

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def main(csv_path):
    vehicles = load_vehicles(csv_path)

    # CAR-17
    print("== CAR-17 ==")
    for line in car_17_outlier_detection(vehicles):
        print(line)

    # CAR-18
    print("\n== CAR-18 ==")
    for line in car_18_consistency_check(vehicles):
        print(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_09.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
