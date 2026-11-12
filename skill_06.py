#!/usr/bin/env python3
"""
Apache performance diagnostics: recommendation explanation and config comparison.

This script extends the earlier skill modules with two new analysis points:

- APACHE-11: Explain each recommended configuration by comparing its settings
  and performance against the best and worst configurations seen so far.
- APACHE-12: Compare a favorable configuration with a non‑favorable one,
  highlighting the differing settings and the resulting performance difference,
  including a prediction from the regression model.

Usage:
    python skill_06.py <csv_path>
"""

import sys
import statistics
import skill_01
import skill_02


def load_csv(path: str):
    """Delegate to skill_01's load_csv."""
    return skill_01.load_csv(path)


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    if not rows:
        print("No data to analyze.")
        sys.exit(0)

    # --- Common preparation -------------------------------------------------
    # Feature names (exclude performance)
    feature_names = [k for k in rows[0] if k != 'Performance-']

    # Best and worst rows (for comparison in APACHE-11)
    best_row = max(rows, key=lambda r: r['Performance-'])
    worst_row = min(rows, key=lambda r: r['Performance-'])

    # Regression model (for APACHE-12)
    beta, feature_names_reg, predictions = skill_02.train_regression(rows)

    # -------------------------------------------------------------------------
    # APACHE-11: Recommendation explanation
    print("== APACHE-11 ==")
    favorable = skill_01.find_favorable_configs(rows)

    if not favorable:
        print("No unusually favorable configurations found. No recommendations to explain.")
    else:
        for cfg in favorable:
            perf = cfg['Performance-']
            diff_best = perf - best_row['Performance-']
            diff_worst = perf - worst_row['Performance-']

            # Features differing from best and worst
            diff_from_best = [f for f in feature_names if cfg[f] != best_row[f]]
            diff_from_worst = [f for f in feature_names if cfg[f] != worst_row[f]]

            print(f"- Config: {skill_01.format_config(cfg)}")
            print(f"  Performance: {perf}")
            print(f"  Difference from best ({best_row['Performance-']}): {diff_best}")
            print(f"  Difference from worst ({worst_row['Performance-']}): {diff_worst}")
            print(f"  Features differing from best: {diff_from_best}")
            print(f"  Features differing from worst: {diff_from_worst}")
            print()  # Blank line for readability

    # -------------------------------------------------------------------------
    # APACHE-12: Config comparison
    print("\n== APACHE-12 ==")

    # Build set of favorable rows for quick membership test
    fav_set = {tuple(sorted(row.items())) for row in favorable}
    non_favorable = [row for row in rows if tuple(sorted(row.items())) not in fav_set]

    if not favorable or not non_favorable:
        print("Insufficient data to perform a favorable vs. non‑favorable comparison.")
        return

    # Choose the first favorable and the first non‑favorable configuration
    fav_cfg = favorable[0]
    non_fav_cfg = non_favorable[0]

    # Settings that differ
    diff_feats = [f for f in feature_names if fav_cfg[f] != non_fav_cfg[f]]
    perf_diff = fav_cfg['Performance-'] - non_fav_cfg['Performance-']

    # Predicted performance for each
    idx_fav = rows.index(fav_cfg)
    idx_non_fav = rows.index(non_fav_cfg)
    pred_fav = predictions[idx_fav]
    pred_non_fav = predictions[idx_non_fav]
    pred_diff = pred_fav - pred_non_fav

    print(f"- Favorable configuration: {skill_01.format_config(fav_cfg)}")
    print(f"  Performance: {fav_cfg['Performance-']}")
    print(f"- Non‑favorable configuration: {skill_01.format_config(non_fav_cfg)}")
    print(f"  Performance: {non_fav_cfg['Performance-']}")
    print(f"  Settings that differ: {diff_feats}")
    print(f"  Observed performance difference: {perf_diff}")
    print(f"  Predicted performance difference (regression): {pred_diff:.1f}")
    print()

if __name__ == "__main__":
    main()
