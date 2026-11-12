#!/usr/bin/env python3
"""
Extended data‑analysis script for the Auto93 dataset.

Adds two new client requests:
    CAR-03 – Estimate missing fuel‑efficiency (mpg) using regression.
    CAR-04 – Estimate vehicle weight and provide an uncertainty estimate.

Usage:
    python skill_02.py <path_to_auto93.csv>
"""

import sys
from typing import List, Dict, Any, Optional

# Import helper functions from skill_01
from skill_01 import (
    read_csv,
    compute_regression_parameters,
    predict,
    format_float
)

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # ----------------------------------------------------------------------- #
    # CAR-03: Estimate missing mpg
    # ----------------------------------------------------------------------- #
    print("== CAR-03 ==")

    # Define features to use for mpg prediction
    mpg_features = ['Lbs-', 'Acc+', 'Volume', 'HpX', 'Model', 'origin']

    # Training data: rows with known mpg and all features present
    training_rows_mpg = [r for r in rows if r['Mpg+'] is not None and
                         all(r.get(f) is not None for f in mpg_features)]

    if training_rows_mpg:
        beta_mpg, _ = compute_regression_parameters(
            training_rows_mpg, mpg_features, 'Mpg+')

        # Find rows with missing mpg but with available features
        missing_mpg_rows = [r for r in rows if r['Mpg+'] is None and
                            all(r.get(f) is not None for f in mpg_features)]

        if missing_mpg_rows:
            for i, row in enumerate(missing_mpg_rows[:3], start=1):
                est_mpg = predict(row, beta_mpg, mpg_features)
                print(f"{i}. Vehicle: Clndrs={int(row['Clndrs'])}, "
                      f"Estimated Mpg+={format_float(est_mpg) if est_mpg is not None else 'N/A'}")
        else:
            print("No vehicles with missing mpg found or insufficient data to predict.")
    else:
        print("Insufficient data to train mpg regression model.")

    # ----------------------------------------------------------------------- #
    # CAR-04: Estimate vehicle weight and uncertainty
    # ----------------------------------------------------------------------- #
    print("\n== CAR-04 ==")

    # Define features to use for weight prediction
    weight_features = ['Lbs-', 'Acc+', 'Volume', 'HpX', 'Model', 'origin']
    # Remove target from features list if present
    weight_features = [f for f in weight_features if f != 'Lbs-']

    # Training data: rows with known weight and all features present
    training_rows_weight = [r for r in rows if r['Lbs-'] is not None and
                            all(r.get(f) is not None for f in weight_features)]

    if training_rows_weight:
        beta_weight, rmse = compute_regression_parameters(
            training_rows_weight, weight_features, 'Lbs-')

        # Estimate weight for rows with missing weight
        missing_weight_rows = [r for r in rows if r['Lbs-'] is None and
                               all(r.get(f) is not None for f in weight_features)]

        if missing_weight_rows:
            for i, row in enumerate(missing_weight_rows[:3], start=1):
                est_weight = predict(row, beta_weight, weight_features)
                print(f"{i}. Vehicle: Clndrs={int(row['Clndrs'])}, "
                      f"Estimated Weight={format_float(est_weight) if est_weight is not None else 'N/A'} "
                      f"±{format_float(rmse)} lbs")
        else:
            print("No vehicles with missing weight found.")
    else:
        print("Insufficient data to train weight regression model.")
        return

if __name__ == "__main__":
    main()
