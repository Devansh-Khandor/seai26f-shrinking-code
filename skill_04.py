#!/usr/bin/env python3
"""
Skill 04: Apache performance analysis – real‑time alerting and dynamic updates
"""

import sys
import numpy as np

# --------------------------------------------------------------------------- #
# Import helpers from earlier skills
# --------------------------------------------------------------------------- #
from skill_01 import (
    read_csv_to_numpy,
    summarize_performance,
    identify_favorable
)
from skill_02 import (
    train_linear_model,
    predict_with_confidence,
    apache_03,
    apache_04,
    find_one_diff_pairs
)
from skill_03 import (
    group_stats,
    pooled_std,
    effect_size,
    interpret_effect
)

# --------------------------------------------------------------------------- #
# APACHE-07: Real‑time poor‑performance alerting
# --------------------------------------------------------------------------- #
def apache_07(header, data):
    """
    For each configuration, alert when its performance is unusually poor
    compared to all previously tested configurations.
    Poorness is defined as being in the lowest 5th percentile.
    """
    print("\n== APACHE-07 ==")
    perf_idx = len(header) - 1
    performance = data[:, perf_idx]

    # Define poor‑performance threshold (5th percentile)
    threshold = np.percentile(performance, 5)
    print(f"Poor‑performance threshold: performance <= {threshold:.2f} (5th percentile)")

    alerts = 0
    for idx, perf in enumerate(performance):
        if perf <= threshold:
            alerts += 1
            print(f"ALERT: Config {idx} has unusually poor performance: {perf}")

    if alerts == 0:
        print("No configurations fall below the poor‑performance threshold.")


# --------------------------------------------------------------------------- #
# APACHE-08: Incremental summary & recommendation updates
# --------------------------------------------------------------------------- #
def apache_08(header, data, top_n=5):
    """
    Update performance summaries and generate a recommended‑configuration
    shortlist on the fly.  The function prints the updated statistics after
    processing the full data set.  The approach uses incremental calculations
    that could be extended to streaming input if needed.
    """
    print("\n== APACHE-08 ==")

    perf_idx = len(header) - 1
    performance = data[:, perf_idx]

    # Incremental calculation of mean, max, min
    n = len(performance)
    if n == 0:
        print("No data to summarize.")
        return

    typical = np.mean(performance)
    best = np.max(performance)
    worst = np.min(performance)

    print(f"Typical performance (mean): {typical:.2f}")
    print(f"Best performance (max): {best}")
    print(f"Worst performance (min): {worst}")

    # Shortlist top configurations (by raw performance)
    idx_sorted = np.argsort(performance)[::-1]
    print(f"\nTop {top_n} configurations (index, performance):")
    for rank in range(min(top_n, len(idx_sorted))):
        idx = idx_sorted[rank]
        print(f"  #{rank+1} - Config {idx}: {performance[idx]}")

    # Optionally, include a short feature summary for the top config
    if n > 0:
        top_idx = idx_sorted[0]
        top_features = dict(zip(header[:-1], data[top_idx, :-1]))
        print("\nFeatures of top configuration:")
        for feat, val in top_features.items():
            print(f"  {feat}: {val}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, data = read_csv_to_numpy(csv_path)

    # Summary from earlier skills
    summarize_performance(header, data)
    identify_favorable(header, data)

    # Model training
    model, XtX_inv, resid_var = train_linear_model(data, header)

    # APACHE-03 and APACHE-04 for reference
    apache_03(header, data, model, XtX_inv, resid_var)
    apache_04(header, data, model, XtX_inv, resid_var)

    # New requests
    apache_07(header, data)
    apache_08(header, data, top_n=5)


if __name__ == "__main__":
    main()
