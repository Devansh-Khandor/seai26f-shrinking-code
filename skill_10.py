#!/usr/bin/env python3
"""
skill_10.py

Usage:
    python skill_10.py <csv_path>

This script implements the new NASA-19 and NASA-20 queries:

  NASA-19 – Evaluate the regression‑based effort estimation against a simple
            historical‑average estimator on a held‑out test set, showing
            which approach has lower prediction error.

  NASA-20 – Compare our proposal‑ranking strategy (based on effort and defect
            per KLOC) with random ranking under a fixed evaluation budget,
            showing which yields a lower average “promise” score.

The script re‑uses parsing, regression, and basic‑statistics helpers
from the earlier skill modules.
"""

import csv
import sys
import random
import statistics
from typing import List, Tuple

# --------------------------------------------------------------------
# Import helpers from earlier skill modules
# --------------------------------------------------------------------
try:
    # Parsing and regression helpers
    from skill_02 import read_csv, linear_regression, predict  # noqa: F401
    # Basic statistics helper
    from skill_04 import compute_mean_std  # noqa: F401
except Exception as e:
    raise RuntimeError(f"Failed to import helpers from skill_02/04: {e}")

# --------------------------------------------------------------------
# NASA-19 – Effort estimation comparison
# --------------------------------------------------------------------
def nasa_19(rows: List[Tuple]) -> None:
    """Compare regression model vs simple historical average on a test set."""
    print("\n== NASA-19 ==")

    if len(rows) < 3:
        print("  (Not enough data for a meaningful train/test split.)")
        return

    # Shuffle and split 70% train / 30% test
    shuffled = rows[:]
    random.shuffle(shuffled)
    split = int(0.7 * len(shuffled))
    train = shuffled[:split]
    test = shuffled[split:]

    # Training data
    klocs_train = [r[5] for r in train]      # index 5 is kloc
    efforts_train = [r[6] for r in train]    # index 6 is effort

    # Regression model
    try:
        a_reg, b_reg = linear_regression(klocs_train, efforts_train)
    except ValueError:
        print("  (Regression failed – insufficient variance in training data.)")
        return

    # Simple average estimator
    avg_effort = statistics.mean(efforts_train)

    # Evaluate on test set
    def mae(preds, true):
        return statistics.mean(abs(p - t) for p, t in zip(preds, true))

    def rmse(preds, true):
        return (statistics.mean((p - t) ** 2 for p, t in zip(preds, true))) ** 0.5

    # Predictions
    pred_reg = [predict(k, a_reg, b_reg) for k in [r[5] for r in test]]
    pred_avg = [avg_effort] * len(test)

    mae_reg = mae(pred_reg, [r[6] for r in test])
    rmse_reg = rmse(pred_reg, [r[6] for r in test])
    mae_avg = mae(pred_avg, [r[6] for r in test])
    rmse_avg = rmse(pred_avg, [r[6] for r in test])

    # Output
    print(f"  Regression MAE : {mae_reg:.2f} PM, RMSE : {rmse_reg:.2f} PM")
    print(f"  Average  MAE : {mae_avg:.2f} PM, RMSE : {rmse_avg:.2f} PM")
    if mae_reg < mae_avg:
        print("  Regression model yields lower absolute error than the historical average.")
    else:
        print("  Historical average performs comparably or better for these data.")
    if rmse_reg < rmse_avg:
        print("  Regression model also yields lower squared error.")
    else:
        print("  Historical average performs comparably or better for squared error.")


# --------------------------------------------------------------------
# NASA-20 – Proposal ranking vs random selection
# --------------------------------------------------------------------
def nasa_20(rows: List[Tuple]) -> None:
    """Compare ranking by score vs random ranking under a budget."""
    print("\n== NASA-20 ==")

    if not rows:
        print("  No data available.")
        return

    # Ask user for evaluation budget (number of proposals)
    try:
        budget_str = input("\nEnter the number of proposals to evaluate: ").strip()
        budget = int(budget_str)
    except Exception:
        print("  (Invalid input – defaulting to 5 proposals.)")
        budget = 5

    # Collect proposal details
    proposals: List[Tuple[float, float, float]] = []  # (size, effort, defects)
    print("\nEnter proposal details:")
    for i in range(1, budget + 1):
        print(f"\nProposal #{i}")
        try:
            sz_str = input("  Enter size (KLOC): ").strip()
            size = float(sz_str)
        except Exception:
            size = statistics.median([r[5] for r in rows])
            print(f"  (Using median size {size:.1f} KLOC)")
        try:
            eff_str = input("  Enter estimated effort (PM): ").strip()
            effort = float(eff_str)
        except Exception:
            effort = statistics.mean([r[6] for r in rows])
            print(f"  (Using mean effort {effort:.1f} PM)")
        try:
            defe_str = input("  Enter estimated defects: ").strip()
            defects = float(defe_str)
        except Exception:
            defects = statistics.mean([r[7] for r in rows])
            print(f"  (Using mean defects {defects:.1f})")
        proposals.append((size, effort, defects))

    # Compute score for each proposal: lower is better
    def score(p):
        size, effort, defects = p
        return (effort / size) + (defects / size)

    # System ranking: sort by score ascending
    sys_sorted = sorted(proposals, key=score)
    sys_avg_score = statistics.mean(score(p) for p in sys_sorted[:budget])

    # Random ranking: shuffle proposals
    random.shuffle(proposals)
    rand_avg_score = statistics.mean(score(p) for p in proposals[:budget])

    # Output results
    print("\nSystem ranking (best first) – average score over selected proposals:")
    print(f"  {sys_avg_score:.3f}")
    print("\nRandom ranking – average score over selected proposals:")
    print(f"  {rand_avg_score:.3f}")

    if sys_avg_score < rand_avg_score:
        print("\n  Our system identifies more desirable outcomes (lower average score).")
    elif sys_avg_score > rand_avg_score:
        print("\n  Random selection performs better (lower average score).")
    else:
        print("\n  Both approaches yield identical average scores.")


# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows = read_csv(csv_path)
    if not rows:
        print("No valid data found.")
        return

    nasa_19(rows)
    nasa_20(rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
