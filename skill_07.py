#!/usr/bin/env python3
"""
Apache performance diagnostics: low-performance discovery and early-benchmark recommendation.

This script implements two new analysis points:

- APACHE-13: Discover a configuration with a low performance-metric value while
  evaluating as few candidate configurations as reasonably possible.
- APACHE-14: When the benchmarking budget is limited, recommend which
  configurations should be evaluated first so that promising server settings
  are found early.

The implementation reuses functions from the earlier skill modules:
  - skill_01.load_csv, skill_01.compute_summary, skill_01.format_config,
    skill_01.find_favorable_configs.
  - skill_02.train_regression, skill_02.regression_metrics.

Usage:
    python skill_07.py <csv_path>
"""

import sys
import statistics
import skill_01
import skill_02


def load_csv(path: str):
    """Delegate to skill_01's load_csv."""
    return skill_01.load_csv(path)


def find_unusually_poor(rows):
    """
    Identify rows that are unusually poor compared to the overall distribution.
    Returns a list of tuples: (row, mean, stddev).
    """
    if not rows:
        return []

    perf = [row['Performance-'] for row in rows]
    mean_all = statistics.mean(perf)
    std_all = statistics.stdev(perf) if len(perf) > 1 else 0.0
    threshold = mean_all - std_all

    # We only need to locate a single configuration that satisfies the
    # threshold.  To keep the evaluation cost low we stop at the first hit.
    for row in rows:
        if row['Performance-'] < threshold:
            return [(row, mean_all, std_all)]
    return []


def recommend_top_configs(rows, top_n=5):
    """
    Rank all configurations by their predicted performance (from regression)
    in descending order.  Return the top_n rows and their predictions.
    """
    if not rows:
        return []

    beta, feature_names, predictions = skill_02.train_regression(rows)

    # Attach predictions to rows for sorting
    scored = [(row, pred) for row, pred in zip(rows, predictions)]
    scored.sort(key=lambda rp: rp[1], reverse=True)  # highest predicted first

    return scored[:top_n]


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    if not rows:
        print("No data to analyze.")
        sys.exit(0)

    # ---------- APACHE-13 ----------
    print("== APACHE-13 ==")
    poor_rows = find_unusually_poor(rows)
    if not poor_rows:
        print("No unusually poor configurations found.")
    else:
        # Show the first unusually poor configuration
        row, mean_all, std_all = poor_rows[0]
        print(f"Unusually poor configuration discovered (performance < mean - stddev):")
        print(f"- Config: {skill_01.format_config(row)}")
        print(f"  Performance: {row['Performance-']}")
        print(f"  Mean performance: {mean_all:.2f}")
        print(f"  Standard deviation: {std_all:.2f}")

    # ---------- APACHE-14 ----------
    print("\n== APACHE-14 ==")
    top_configs = recommend_top_configs(rows, top_n=5)
    if not top_configs:
        print("Insufficient data to recommend configurations.")
    else:
        print("Recommended configurations to evaluate first (highest predicted performance):")
        for idx, (row, pred) in enumerate(top_configs, start=1):
            print(f"{idx}. {skill_01.format_config(row)}")
            print(f"   Predicted Performance: {pred:.2f}")
            # Show actual if available
            actual = row['Performance-']
            print(f"   Actual Performance: {actual}")

if __name__ == "__main__":
    main()
