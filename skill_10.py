#!/usr/bin/env python3
"""
skill_10.py

Implements the NASA-19 and NASA-20 requests:

NASA-19: Compare the project‑effort estimation system against a simple historical‑average
        estimate and show whether it makes meaningfully better predictions on previously
        unseen projects.

NASA-20: Compare the project‑selection approach with randomly selecting proposals under
        the same evaluation budget, and show whether our system identifies more desirable
        project outcomes.

The implementation reuses the training routine from skill_04, the data helpers from
skill_01 and the recommendation logic from skill_07.  It prints a header line
for each request followed by at least three lines of results.
"""

import sys
import numpy as np
from sklearn.metrics import mean_absolute_error

# Helpers from the earlier skills
from skill_01 import read_csv, extract_numeric_columns, compute_stats
from skill_04 import train_and_predict
from skill_07 import nasa_14  # use the recommendation logic


def _split_train_test(rows, test_size=0.2, random_state=42):
    """
    Split indices into training and test sets (stratified by presence
    of complete Effort- and Defects- data).  Only rows with complete
    effort/defect data are used for training/testing.
    """
    indices = np.arange(len(rows))
    # Keep only rows where we have complete effort and defects
    effort = extract_numeric_columns(rows, ["Effort-"])
    defects = extract_numeric_columns(rows, ["Defects-"])
    mask = ~np.isnan(effort) & ~np.isnan(defects)
    valid_idx = indices[mask]

    rng = np.random.default_rng(random_state)
    rng.shuffle(valid_idx)

    split_point = int(len(valid_idx) * (1 - test_size))
    train_idx = valid_idx[:split_point]
    test_idx = valid_idx[split_point:]

    return train_idx, test_idx


def nasa_19(csv_path: str):
    """
    Evaluate the trained linear models against a simple historical‑average
    baseline on a held‑out test set.  Reports mean absolute error (MAE)
    for both effort and defects, as well as percent improvement.
    """
    rows = read_csv(csv_path)

    train_idx, test_idx = _split_train_test(rows)

    # Train models on training set
    reg_effort, reg_defect, hist_effort_stats, hist_defect_stats = train_and_predict(rows)

    # Prepare test data
    size_test = extract_numeric_columns(rows, ["Kloc+"])[test_idx]
    effort_test = extract_numeric_columns(rows, ["Effort-"])[test_idx]
    defects_test = extract_numeric_columns(rows, ["Defects-"])[test_idx]

    # Predictions
    eff_pred = reg_effort.predict(size_test.reshape(-1, 1)).flatten()
    def_pred = reg_defect.predict(size_test.reshape(-1, 1)).flatten()

    # Baseline predictions (historical mean)
    eff_baseline = np.full_like(eff_pred, hist_effort_stats["mean"])
    def_baseline = np.full_like(def_pred, hist_defect_stats["mean"])

    # MAE
    eff_mae_model = mean_absolute_error(effort_test, eff_pred)
    eff_mae_base = mean_absolute_error(effort_test, eff_baseline)

    def_mae_model = mean_absolute_error(defects_test, def_pred)
    def_mae_base = mean_absolute_error(defects_test, def_baseline)

    # Percent improvement
    eff_impr = (eff_mae_base - eff_mae_model) / eff_mae_base * 100
    def_impr = (def_mae_base - def_mae_model) / def_mae_base * 100

    lines = [
        f"  Model MAE (effort): {eff_mae_model:.2f} months",
        f"  Baseline MAE (effort): {eff_mae_base:.2f} months",
        f"  Improvement: {eff_impr:.1f}% reduction in effort error",
        "",
        f"  Model MAE (defects): {def_mae_model:.2f}",
        f"  Baseline MAE (defects): {def_mae_base:.2f}",
        f"  Improvement: {def_impr:.1f}% reduction in defect error",
    ]
    return lines


def nasa_20(csv_path: str):
    """
    Compare the recommendation system (NASA‑14) with random proposal selection
    under the same evaluation budget.  Uses predicted effort+defects as the
    desirability metric.  Reports average predicted effort, defects, and
    combined score for both strategies and the percentage improvement.
    """
    rows = read_csv(csv_path)

    # Train models and get historical stats (used by nasa_14)
    reg_effort, reg_defect, hist_effort_stats, hist_defect_stats = train_and_predict(rows)

    # ---- Get NASA‑14 recommendations ----
    rec_lines = nasa_14(rows, reg_effort, reg_defect, hist_effort_stats, hist_defect_stats)

    # Parse selected proposal IDs from recommendation lines
    selected_ids = []
    for line in rec_lines:
        if line.startswith("  Proposal ID="):
            try:
                pid = int(line.split("ID=")[1].split(":")[0])
                selected_ids.append(pid)
            except Exception:
                continue
    if not selected_ids:
        return ["  No proposals were recommended by NASA‑14."]

    # All proposal indices (missing Effort- or Defects-)
    effort_vals = extract_numeric_columns(rows, ["Effort-"])
    defect_vals = extract_numeric_columns(rows, ["Defects-"])
    proposal_mask = np.isnan(effort_vals) | np.isnan(defect_vals)
    proposal_indices = np.where(proposal_mask)[0]
    proposal_ids = np.array([int(row["idX"]) for row in rows], dtype=int)[proposal_indices]

    # Size of recommendation set
    n_select = len(selected_ids)

    # ---- Compute predicted effort and defects for all proposals ----
    size_vals = extract_numeric_columns(rows, ["Kloc+"])
    size_proj = size_vals[proposal_mask]
    eff_pred_proj = reg_effort.predict(size_proj.reshape(-1, 1)).flatten()
    def_pred_proj = reg_defect.predict(size_proj.reshape(-1, 1)).flatten()

    # Map id to predicted values
    id_to_eff = dict(zip(proposal_ids, eff_pred_proj))
    id_to_def = dict(zip(proposal_ids, def_pred_proj))

    # ---- Recommendation strategy ----
    eff_rec = np.array([id_to_eff[pid] for pid in selected_ids])
    def_rec = np.array([id_to_def[pid] for pid in selected_ids])

    avg_eff_rec = np.mean(eff_rec)
    avg_def_rec = np.mean(def_rec)
    avg_comb_rec = np.mean(eff_rec + def_rec)

    # ---- Random strategy ----
    rng = np.random.default_rng(12345)
    random_ids = rng.choice(proposal_ids, size=n_select, replace=False)
    eff_rand = np.array([id_to_eff[pid] for pid in random_ids])
    def_rand = np.array([id_to_def[pid] for pid in random_ids])

    avg_eff_rand = np.mean(eff_rand)
    avg_def_rand = np.mean(def_rand)
    avg_comb_rand = np.mean(eff_rand + def_rand)

    # Percent improvement of our selection over random
    eff_impr = (avg_eff_rand - avg_eff_rec) / avg_eff_rand * 100
    def_impr = (avg_def_rand - avg_def_rec) / avg_def_rand * 100
    comb_impr = (avg_comb_rand - avg_comb_rec) / avg_comb_rand * 100

    lines = [
        f"  Number of proposals evaluated: {n_select}",
        "",
        "=== Our recommendation (NASA‑14) ===",
        f"  Avg predicted effort: {avg_eff_rec:.2f} months",
        f"  Avg predicted defects: {avg_def_rec:.2f}",
        f"  Avg combined score (effort + defects): {avg_comb_rec:.2f}",
        "",
        "=== Random selection ===",
        f"  Avg predicted effort: {avg_eff_rand:.2f} months",
        f"  Avg predicted defects: {avg_def_rand:.2f}",
        f"  Avg combined score (effort + defects): {avg_comb_rand:.2f}",
        "",
        f"  Effort improvement: {eff_impr:.1f}%",
        f"  Defect improvement: {def_impr:.1f}%",
        f"  Combined score improvement: {comb_impr:.1f}%",
    ]
    return lines


def main(csv_path: str):
    # NASA‑19
    print("== NASA-19 ==")
    for line in nasa_19(csv_path):
        print(line)

    # NASA‑20
    print("\n== NASA-20 ==")
    for line in nasa_20(csv_path):
        print(line)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
