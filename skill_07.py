#!/usr/bin/env python3
"""
Skill 07: Apache performance analysis – targeted discovery & evaluation planning
"""

import sys
import numpy as np

# --------------------------------------------------------------------------- #
# Import helpers from earlier skills
# --------------------------------------------------------------------------- #
from skill_01 import (
    read_csv_to_numpy,
    summarize_performance,
    identify_favorable,
)
from skill_02 import (
    train_linear_model,
    predict_with_confidence,
    apache_03,
    apache_04,
    find_one_diff_pairs,
)
# --------------------------------------------------------------------------- #
# APACHE-13: Discover low‑performance configuration quickly
# --------------------------------------------------------------------------- #
def apache_13(header, data, model, XtX_inv, resid_var):
    """
    Find a configuration with a low performance metric (below the 5th percentile)
    while evaluating as few candidate configurations as possible.

    The algorithm uses the trained model to predict performance, sorts all
    configurations by ascending predicted performance, and evaluates them one
    by one until a real low‑performance configuration is found.
    """
    print("\n== APACHE-13 ==")

    perf_idx = len(header) - 1
    performance = data[:, perf_idx]

    # Determine low‑performance threshold (5th percentile)
    threshold = np.percentile(performance, 5)
    print(f"Low‑performance threshold: performance <= {threshold:.2f} (5th percentile)")

    # Predict performance for all configurations
    predictions = model.predict(data[:, :-1])

    # Sort indices by predicted performance ascending
    sorted_idx = np.argsort(predictions)

    evaluated = 0
    found = False

    for idx in sorted_idx:
        evaluated += 1
        actual_perf = performance[idx]
        if actual_perf <= threshold:
            found = True
            pred_perf = predictions[idx]
            # Get confidence interval for the configuration
            pred, low, high = predict_with_confidence(
                model, XtX_inv, resid_var, data[idx, :-1], confidence=0.95
            )
            print(f"Candidate #{evaluated}: Config {idx}")
            print(f"  Predicted performance: {pred_perf:.2f} ± {high - pred:.2f} (95% CI: [{low:.2f}, {high:.2f}])")
            print(f"  Actual performance: {actual_perf:.2f}")
            break

    if not found:
        print(f"No configuration below threshold after evaluating all {len(data)} candidates.")


# --------------------------------------------------------------------------- #
# APACHE-14: Prioritize configurations for limited benchmarking budget
# --------------------------------------------------------------------------- #
def apache_14(header, data, model, XtX_inv, resid_var, top_n=5):
    """
    Recommend the configurations that should be evaluated first when the
    benchmarking budget is limited. The recommendation is based on the model's
    predicted performance – the highest predicted values are the most promising.
    """
    print("\n== APACHE-14 ==")

    # Predict performance for all configurations
    predictions = model.predict(data[:, :-1])

    # Sort indices by predicted performance descending
    sorted_idx = np.argsort(predictions)[::-1]

    print(f"Recommend evaluating the top {top_n} configurations first:")
    for rank in range(min(top_n, len(sorted_idx))):
        idx = sorted_idx[rank]
        pred_perf = predictions[idx]
        print(f"  Rank {rank+1}: Config {idx} (predicted performance: {pred_perf:.2f})")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, data = read_csv_to_numpy(csv_path)

    # Basic summary and favorable detection
    summarize_performance(header, data)
    identify_favorable(header, data)

    # Train model
    model, XtX_inv, resid_var = train_linear_model(data, header)

    # Reference outputs from earlier skills
    apache_03(header, data, model, XtX_inv, resid_var)
    apache_04(header, data, model, XtX_inv, resid_var)

    # New requests
    apache_13(header, data, model, XtX_inv, resid_var)
    apache_14(header, data, model, XtX_inv, resid_var, top_n=5)


if __name__ == "__main__":
    main()
