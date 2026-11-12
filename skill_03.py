#!/usr/bin/env python3
"""
Skill 03: Statistical comparison of fuel efficiency and performance

This script reads the Auto93 dataset and produces two reports:

* CAR-05 – Compare fuel efficiency across cylinder counts and assess
            statistical significance of the differences.

* CAR-06 – Compare overall vehicle performance for two model‑year ranges
            and decide if the evidence is strong enough for inventory decisions.

Usage:
    python skill_03.py <csv_path>
"""

import csv
import sys
import math

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def read_csv(csv_path):
    """
    Read CSV and return a list of dictionaries with numeric values.
    Missing or non‑numeric entries become None.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = {}
            for key in ("Clndrs", "Volume", "HpX", "Model", "origin",
                        "Lbs-", "Acc+", "Mpg+"):
                try:
                    parsed[key] = float(r[key])
                except (KeyError, ValueError, TypeError):
                    parsed[key] = None
            rows.append(parsed)
    return rows


def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def stddev(lst, ddof=1):
    n = len(lst)
    if n <= ddof:
        return 0.0
    m = mean(lst)
    var = sum((x - m) ** 2 for x in lst) / (n - ddof)
    return math.sqrt(var)


def z_test(mean1, std1, n1, mean2, std2, n2):
    """
    Two‑sample z‑test for difference of means (normal approximation).
    Returns (z‑value, p‑value).
    """
    se = math.sqrt((std1 ** 2) / n1 + (std2 ** 2) / n2)
    if se == 0:
        return 0.0, 1.0
    z = (mean1 - mean2) / se
    p = math.erfc(abs(z) / math.sqrt(2))  # two‑tailed
    return z, p


# ----------------------------------------------------------------------
# CAR-05: Fuel efficiency by cylinder count
# ----------------------------------------------------------------------
def car_05(rows):
    # Group by cylinder count
    groups = {}
    for r in rows:
        c = r["Clndrs"]
        mpg = r["Mpg+"]
        if c is None or mpg is None:
            continue
        groups.setdefault(int(c), []).append(mpg)

    # Compute stats per group
    stats = {}
    for c, lst in sorted(groups.items()):
        n = len(lst)
        m = mean(lst)
        s = stddev(lst)
        stats[c] = (n, m, s)

    out = []
    out.append("Fuel efficiency by cylinder count (MPG):")
    out.append("{:<8} {:>5} {:>7} {:>7}".format("Cyl", "N", "Mean", "Std"))
    for c, (n, m, s) in stats.items():
        out.append("{:<8} {:>5} {:>7.2f} {:>7.2f}".format(c, n, m, s))

    # Pairwise comparisons
    out.append("\nPairwise comparisons (significant at p < 0.05):")
    cyl_keys = sorted(stats.keys())
    found_sig = False
    for i in range(len(cyl_keys)):
        for j in range(i + 1, len(cyl_keys)):
            c1, c2 = cyl_keys[i], cyl_keys[j]
            n1, m1, s1 = stats[c1]
            n2, m2, s2 = stats[c2]
            z, p = z_test(m1, s1, n1, m2, s2, n2)
            if p < 0.05:
                found_sig = True
                out.append(f"  {c1} vs {c2}: Δ={m1-m2: .2f} MPG, p={p:.3f}")

    if not found_sig:
        out.append("  No statistically significant differences found between cylinder groups.")
    return out


# ----------------------------------------------------------------------
# CAR-06: Performance comparison across model‑year ranges
# ----------------------------------------------------------------------
def car_06(rows):
    # Compute min/max for normalization
    lbs_vals = [r["Lbs-"] for r in rows if r["Lbs-"] is not None]
    mpg_vals = [r["Mpg+"] for r in rows if r["Mpg+"] is not None]
    acc_vals = [r["Acc+"] for r in rows if r["Acc+"] is not None]

    def norm(val, mn, mx, reverse=False):
        if mx == mn:
            return 0.5
        if reverse:
            return (mx - val) / (mx - mn)
        else:
            return (val - mn) / (mx - mn)

    lbs_min, lbs_max = min(lbs_vals), max(lbs_vals)
    mpg_min, mpg_max = min(mpg_vals), max(mpg_vals)
    acc_min, acc_max = min(acc_vals), max(acc_vals)

    # Build score for each vehicle
    scores = []
    models = []
    for r in rows:
        if None in (r["Lbs-"], r["Mpg+"], r["Acc+"], r["Model"]):
            continue
        lw = norm(r["Lbs-"], lbs_min, lbs_max, reverse=True)
        mg = norm(r["Mpg+"], mpg_min, mpg_max, reverse=False)
        ac = norm(r["Acc+"], acc_min, acc_max, reverse=True)
        score = (lw + mg + ac) / 3.0
        scores.append(score)
        models.append(int(r["Model"]))

    # Split into two model-year ranges: <=70 and >70
    group_a = [s for s, m in zip(scores, models) if m <= 70]
    group_b = [s for s, m in zip(scores, models) if m > 70]

    if not group_a or not group_b:
        return ["Insufficient data to compare the two model‑year ranges."]

    n_a, m_a, s_a = len(group_a), mean(group_a), stddev(group_a)
    n_b, m_b, s_b = len(group_b), mean(group_b), stddev(group_b)

    z, p = z_test(m_a, s_a, n_a, m_b, s_b, n_b)

    out = []
    out.append("Overall performance comparison by model year:")
    out.append(f"  Group A: Model <= 70  (n={n_a})  Mean score={m_a:.3f}  Std={s_a:.3f}")
    out.append(f"  Group B: Model > 70   (n={n_b})  Mean score={m_b:.3f}  Std={s_b:.3f}")
    out.append(f"Difference (A – B): {m_a - m_b:.3f}   z={z:.2f}   p={p:.3f}")

    if p < 0.05:
        out.append("Conclusion: The difference is statistically significant (p < 0.05).")
    else:
        out.append("Conclusion: The evidence is too weak to assert a meaningful difference (p >= 0.05).")

    return out


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    # CAR-05
    print("== CAR-05 ==")
    for line in car_05(rows):
        print(line)

    # CAR-06
    print("\n== CAR-06 ==")
    for line in car_06(rows):
        print(line)


if __name__ == "__main__":
    main()
