#!/usr/bin/env python3
"""
Apache performance diagnostics: feature impact and combo analysis.

This script extends the earlier skill modules with two new analysis points:

- APACHE-09: For each poorly performing configuration, identify the settings
  (enabled/disabled) that are most strongly associated with unfavorable
  performance based on historical averages.
- APACHE-10: Identify combinations of server settings that frequently appear
  among poorly performing configurations and compare them with better‑performing
  combinations.

Usage:
    python skill_05.py <csv_path>
"""

import sys
import statistics
import skill_01

def load_csv(path: str):
    """Delegate to skill_01's load_csv."""
    return skill_01.load_csv(path)

def compute_global_stats(rows):
    """Return mean and stddev of performance over all rows."""
    if not rows:
        return 0.0, 0.0
    perf = [row['Performance-'] for row in rows]
    mean = statistics.mean(perf)
    stddev = statistics.stdev(perf) if len(perf) > 1 else 0.0
    return mean, stddev

def identify_poor_rows(rows, threshold):
    """Return rows whose performance is below the given threshold."""
    return [row for row in rows if row['Performance-'] < threshold]

def compute_feature_stats(rows):
    """
    For each feature compute mean performance when the feature is 1 and 0.
    Returns dict: feature -> (mean1, mean0, diff)
    diff = mean1 - mean0
    """
    if not rows:
        return {}
    feature_names = [k for k in rows[0] if k != 'Performance-']
    stats = {}
    for f in feature_names:
        perf1 = [row['Performance-'] for row in rows if row[f] == 1]
        perf0 = [row['Performance-'] for row in rows if row[f] == 0]
        mean1 = statistics.mean(perf1) if perf1 else 0.0
        mean0 = statistics.mean(perf0) if perf0 else 0.0
        diff = mean1 - mean0
        stats[f] = (mean1, mean0, diff)
    return stats

def top_features_for_row(row, feature_stats, top_n=3):
    """
    For a given row, find the features whose value (0/1) aligns with
    a historically unfavorable average.  Return a list of tuples:
    (feature, value, mean_value, diff), sorted by absolute diff descending.
    """
    results = []
    for f, (mean1, mean0, diff) in feature_stats.items():
        value = row[f]
        # Determine whether this value is associated with poorer performance
        # by comparing its mean to the other value.
        if value == 1:
            # If enabling is worse (mean1 < mean0), flag it
            if mean1 < mean0:
                results.append((f, value, mean1, diff))
        else:  # value == 0
            if mean0 < mean1:
                results.append((f, value, mean0, diff))
    # Sort by absolute diff descending
    results.sort(key=lambda x: abs(x[3]), reverse=True)
    return results[:top_n]

def combo_key(row):
    """Return a tuple that represents the configuration combo (excluding performance)."""
    return tuple(row[k] for k in row if k != 'Performance-')

def compute_combo_stats(rows):
    """
    Return a dict mapping combo tuples to:
        {'count': int, 'mean_perf': float}
    """
    combo_dict = {}
    for row in rows:
        key = combo_key(row)
        combo_dict.setdefault(key, {'count': 0, 'sum_perf': 0.0}).update({
            'count': combo_dict[key]['count'] + 1,
            'sum_perf': combo_dict[key]['sum_perf'] + row['Performance-']
        })
    # Finalize mean
    for key, data in combo_dict.items():
        data['mean_perf'] = data['sum_perf'] / data['count']
        del data['sum_perf']
    return combo_dict

def top_combo_by_frequency(combo_dict, top_n=3):
    """Return list of (key, count, mean_perf) sorted by count descending."""
    items = [(k, v['count'], v['mean_perf']) for k, v in combo_dict.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_n]

def top_combo_by_perf(combo_dict, top_n=3, best=True):
    """Return list of (key, count, mean_perf) sorted by mean_perf descending/ascending."""
    items = [(k, v['count'], v['mean_perf']) for k, v in combo_dict.items()]
    items.sort(key=lambda x: x[2], reverse=best)
    return items[:top_n]

def format_combo(key):
    """Convert combo tuple back to string representation."""
    return "{" + ", ".join(str(v) for v in key) + "}"

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    # -------- APACHE-09 --------
    print("== APACHE-09 ==")
    mean_all, stddev_all = compute_global_stats(rows)
    threshold = mean_all - stddev_all
    poor_rows = identify_poor_rows(rows, threshold)
    if not poor_rows:
        print("No poorly performing configurations detected.")
    else:
        # Compute feature statistics once
        feature_stats = compute_feature_stats(rows)
        # Show top 3 problematic features for the first 3 poor configs
        for row in poor_rows[:3]:
            print(f"- Config: {skill_01.format_config(row)}")
            top_feats = top_features_for_row(row, feature_stats)
            if not top_feats:
                print("  No clear feature associations found.")
            else:
                for f, val, mean_val, diff in top_feats:
                    sign = "enabling" if val == 1 else "disabling"
                    print(f"  - {sign} '{f}' historically averages {mean_val:.2f} "
                          f"permutation (diff {diff:+.2f})")

    # -------- APACHE-10 --------
    print("\n== APACHE-10 ==")
    combo_dict = compute_combo_stats(rows)
    # Identify poor and good combos based on overall mean +/- stddev
    poor_combo_thresh = mean_all - stddev_all
    good_combo_thresh = mean_all + stddev_all
    poor_combos = {k: v for k, v in combo_dict.items() if v['mean_perf'] < poor_combo_thresh}
    good_combos = {k: v for k, v in combo_dict.items() if v['mean_perf'] > good_combo_thresh}

    if not poor_combos:
        print("No combinations fall below the poor-performance threshold.")
    else:
        print("Top poorly performing combos by frequency:")
        for key, count, mean_perf in top_combo_by_frequency(poor_combos):
            print(f"- {format_combo(key)}: count={count}, mean_perf={mean_perf:.2f}")

    if not good_combos:
        print("\nNo combinations exceed the good-performance threshold.")
    else:
        print("\nTop well-performing combos by mean performance:")
        for key, count, mean_perf in top_combo_by_perf(good_combos):
            print(f"- {format_combo(key)}: count={count}, mean_perf={mean_perf:.2f}")

if __name__ == "__main__":
    main()
