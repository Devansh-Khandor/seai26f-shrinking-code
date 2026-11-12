#!/usr/bin/env python3
"""
Skill 03: Comparative analysis for used‑car dealership.

CAR-05: Compare the fuel efficiency of vehicles with different cylinder counts
and tell whether the observed differences are reliable enough to inform inventory decisions.

CAR-06: Compare the overall performance of two groups of vehicles from different
model‑year ranges and decide whether the evidence is strong enough to make a conclusion.
"""

import csv
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# Data handling helpers
# ----------------------------------------------------------------------
def read_data(csv_path):
    """
    Read CSV and return a list of rows with numeric values converted.
    Missing numeric values are stored as None.
    """
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric columns; leave None if conversion fails
            for key in ['Clndrs', 'Volume', 'HpX', 'Model', 'origin',
                        'Lbs-', 'Acc+', 'Mpg+']:
                try:
                    row[key] = float(row[key]) if row[key] != '' else None
                except (ValueError, KeyError):
                    row[key] = None
            rows.append(row)
    return rows

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def mean(values):
    return sum(values) / len(values) if values else 0.0

def stddev(values, ddof=1):
    n = len(values)
    if n <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (n - ddof))

def welch_t_test(values1, values2):
    """
    Return (t, df) for Welch's t‑test between two samples.
    If a sample has fewer than two values, returns (None, None).
    """
    n1, n2 = len(values1), len(values2)
    if n1 < 2 or n2 < 2:
        return None, None
    mean1, mean2 = mean(values1), mean(values2)
    s1, s2 = stddev(values1, ddof=1), stddev(values2, ddof=1)
    se = math.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
    t = (mean1 - mean2) / se if se != 0 else 0.0

    # Welch–Satterthwaite degrees of freedom
    df_num = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2
    df_den = ((s1 ** 4) / (n1 ** 2 * (n1 - 1))) + ((s2 ** 4) / (n2 ** 2 * (n2 - 1)))
    df = df_num / df_den if df_den != 0 else float('inf')
    return t, df

# ----------------------------------------------------------------------
# CAR‑05 helpers
# ----------------------------------------------------------------------
def group_by_cylinders(rows):
    """
    Return a dict mapping cylinder count to list of mpg values.
    """
    groups = defaultdict(list)
    for r in rows:
        if r['Clndrs'] is None or r['Mpg+'] is None:
            continue
        groups[int(r['Clndrs'])].append(r['Mpg+'])
    return groups

def car_05(rows):
    groups = group_by_cylinders(rows)

    print("== CAR-05 ==")
    if len(groups) < 2:
        print(f"Only {len(groups)} cylinder group(s) present; no comparison possible.")
        return

    # Summary table
    print("Cylinder | Count | Mean MPG | Std MPG")
    for cyl in sorted(groups):
        vals = groups[cyl]
        print(f"{cyl:>8} | {len(vals):>5} | {mean(vals):>8.2f} | {stddev(vals):>7.2f}")

    # Pairwise tests
    significant_pairs = []
    cyls = sorted(groups)
    for i in range(len(cyls)):
        for j in range(i + 1, len(cyls)):
            t, df = welch_t_test(groups[cyls[i]], groups[cyls[j]])
            if t is None:
                continue
            # Rough threshold: |t| > 2 indicates p < 0.05
            if abs(t) > 2.0:
                significant_pairs.append((cyls[i], cyls[j], t, df))

    if significant_pairs:
        print("\nSignificant differences found between the following cylinder groups:")
        for a, b, t, df in significant_pairs:
            print(f"  {a} vs {b}: t = {t:.3f} (df ≈ {df:.1f})")
        print("=> The observed MPG differences are statistically reliable enough to inform inventory decisions.")
    else:
        print("\nNo statistically significant MPG differences detected between cylinder groups.")
        print("=> The evidence is too weak to make a strong inventory decision based on cylinder count.")

# ----------------------------------------------------------------------
# CAR‑06 helpers
# ----------------------------------------------------------------------
def compute_overall_score(rows, stats):
    """
    Add 'overall_score' to each row based on weight, mpg and acceleration.
    Returns the updated list.
    """
    wmin, wmax = stats['weight_min'], stats['weight_max']
    mmin, mmax = stats['mpg_min'], stats['mpg_max']
    amin, amax = stats['acc_min'], stats['acc_max']

    for r in rows:
        # Skip rows lacking any of the components
        if r['Lbs-'] is None or r['Mpg+'] is None or r['Acc+'] is None:
            r['overall_score'] = None
            continue

        w_score = (wmax - r['Lbs-']) / (wmax - wmin) if wmax != wmin else 1.0
        m_score = (r['Mpg+'] - mmin) / (mmax - mmin) if mmax != mmin else 1.0
        a_score = (amax - r['Acc+']) / (amax - amin) if amax != amin else 1.0
        r['overall_score'] = w_score + m_score + a_score
    return rows

def compute_weight_stats(rows):
    weights = [r['Lbs-'] for r in rows if r['Lbs-'] is not None]
    mpgs   = [r['Mpg+'] for r in rows if r['Mpg+'] is not None]
    accs   = [r['Acc+'] for r in rows if r['Acc+'] is not None]
    return {
        'weight_min': min(weights) if weights else 0.0,
        'weight_max': max(weights) if weights else 0.0,
        'mpg_min': min(mpgs) if mpgs else 0.0,
        'mpg_max': max(mpgs) if mpgs else 0.0,
        'acc_min': min(accs) if accs else 0.0,
        'acc_max': max(accs) if accs else 0.0
    }

def group_by_year_range(rows, threshold=70):
    """
    Split rows into two groups:
    - group1: Model <= threshold
    - group2: Model > threshold
    """
    g1, g2 = [], []
    for r in rows:
        if r['Model'] is None:
            continue
        if r['Model'] <= threshold:
            g1.append(r)
        else:
            g2.append(r)
    return g1, g2

def car_06(rows, threshold=70):
    stats = compute_weight_stats(rows)
    compute_overall_score(rows, stats)

    g1, g2 = group_by_year_range(rows, threshold)

    print("\n== CAR-06 ==")
    # Summary stats for each group
    def group_stats(group, label):
        scores = [r['overall_score'] for r in group if r['overall_score'] is not None]
        if not scores:
            return None
        return {
            'label': label,
            'n': len(scores),
            'mean': mean(scores),
            'std': stddev(scores)
        }

    s1 = group_stats(g1, f"Model <= {threshold}")
    s2 = group_stats(g2, f"Model > {threshold}")

    if s1 is None and s2 is None:
        print("No rows with complete overall scores.")
        return
    if s1 is None or s2 is None:
        print("Insufficient complete data in one or both groups to compare.")
        return

    print(f"{s1['label']}: n={s1['n']}, mean={s1['mean']:.3f}, std={s1['std']:.3f}")
    print(f"{s2['label']}: n={s2['n']}, mean={s2['mean']:.3f}, std={s2['std']:.3f}")

    if s1['n'] < 2 or s2['n'] < 2:
        print("\nNot enough samples in one or both groups to perform a statistical test.")
        print("=> The evidence is too weak to conclude a meaningful difference in overall performance.")
        return

    # Perform Welch's t-test on overall scores
    t, df = welch_t_test([r['overall_score'] for r in g1], [r['overall_score'] for r in g2])
    if t is None:
        print("\nUnable to perform t-test due to insufficient data.")
        return

    # Rough significance check
    if abs(t) > 2.0:
        print("\nThe difference in overall performance is statistically significant (t ≈ {:.2f}, df ≈ {:.1f}).")
        print("=> The evidence suggests a meaningful difference between the two model‑year ranges.")
    else:
        print("\nThe difference in overall performance is not statistically significant (t ≈ {:.2f}, df ≈ {:.1f}).")
        print("=> The evidence is too weak to conclude a meaningful difference.")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_data(csv_path)
    if not rows:
        print("No data found in CSV.")
        return

    car_05(rows)
    car_06(rows)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
