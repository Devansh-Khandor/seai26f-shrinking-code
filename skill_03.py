#!/usr/bin/env python3
"""
Skill 03 – Comparative Analysis of Fuel Efficiency and Performance

Usage:
    python skill_03.py <csv_path>
"""

import csv
import math
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read the CSV and return a list of rows as dictionaries."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def safe_float(val: str) -> float | None:
    """Return float or None if conversion fails."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

# ----------------------------------------------------------------------
# Composite score calculation (same as skill_01)
# ----------------------------------------------------------------------
def compute_composite_scores(rows: List[Dict[str, str]]) -> List[float]:
    """Return a list of composite scores for each row."""
    weights = [safe_float(r["Lbs-"]) for r in rows]
    mpgs = [safe_float(r["Mpg+"]) for r in rows]
    accs = [safe_float(r["Acc+"]) for r in rows]

    # Remove None values
    weights = [v for v in weights if v is not None]
    mpgs = [v for v in mpgs if v is not None]
    accs = [v for v in accs if v is not None]

    w_min, w_max = min(weights), max(weights)
    mpg_min, mpg_max = min(mpgs), max(mpgs)
    acc_min, acc_max = min(accs), max(accs)

    def norm(val: float, low: float, high: float, higher_is_better: bool = True) -> float:
        if high == low:
            return 0.5
        if higher_is_better:
            return (val - low) / (high - low)
        else:
            return (high - val) / (high - low)

    scores = []
    for r in rows:
        w = safe_float(r["Lbs-"])
        m = safe_float(r["Mpg+"])
        a = safe_float(r["Acc+"])
        if None in (w, m, a):
            scores.append(float("nan"))
            continue
        w_score = norm(w, w_min, w_max, higher_is_better=False)
        m_score = norm(m, mpg_min, mpg_max)
        a_score = norm(a, acc_min, acc_max)
        scores.append((w_score + m_score + a_score) / 3)
    return scores

# ----------------------------------------------------------------------
# Group statistics helpers
# ----------------------------------------------------------------------
def group_stats(values: List[float]) -> Tuple[float, float, int]:
    """Return (mean, std_dev, count). std_dev is population std dev."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    return (mean, std, n)

def welch_t_test(mean1: float, mean2: float,
                 var1: float, var2: float,
                 n1: int, n2: int) -> Tuple[float, float]:
    """Return (t_stat, df)."""
    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        return (0.0, 1)
    t = (mean1 - mean2) / se
    # Welch-Satterthwaite df approximation
    df_num = (var1 / n1 + var2 / n2) ** 2
    df_den = ((var1 ** 2) / (n1 ** 2 * (n1 - 1))) + ((var2 ** 2) / (n2 ** 2 * (n2 - 1)))
    df = df_num / df_den if df_den != 0 else 1
    return (t, df)

# ----------------------------------------------------------------------
# CAR-05 – Fuel efficiency by cylinder count
# ----------------------------------------------------------------------
def car_05(rows: List[Dict[str, str]]) -> None:
    # Group by cylinder count
    cyl_groups: Dict[str, List[float]] = {}
    for r in rows:
        cl = r.get("Clndrs")
        mpg = safe_float(r["Mpg+"])
        if mpg is None:
            continue
        cyl_groups.setdefault(cl, []).append(mpg)

    # Compute stats per group
    cyl_stats: Dict[str, Tuple[float, float, int]] = {}
    for cl, mpgs in cyl_groups.items():
        mean, std, n = group_stats(mpgs)
        cyl_stats[cl] = (mean, std, n)

    print("== CAR-05 ==")
    # Print group stats
    for cl, (mean, std, n) in sorted(cyl_stats.items()):
        print(f"Cylinders {cl:<2} – Mean MPG: {mean:7.2f}, Std Dev: {std:7.2f} (n={n})")

    # Pairwise comparisons
    pairs = []
    cyl_keys = sorted(cyl_stats.keys())
    for i in range(len(cyl_keys)):
        for j in range(i + 1, len(cyl_keys)):
            cl1, cl2 = cyl_keys[i], cyl_keys[j]
            m1, s1, n1 = cyl_stats[cl1]
            m2, s2, n2 = cyl_stats[cl2]
            var1, var2 = s1 ** 2, s2 ** 2
            t, df = welch_t_test(m1, m2, var1, var2, n1, n2)
            diff = m1 - m2
            pairs.append((cl1, cl2, diff, t))

    significant = [p for p in pairs if abs(p[3]) >= 2]  # threshold t >= 2
    if significant:
        for cl1, cl2, diff, t in significant:
            print(f"Significant difference: {cl1} vs {cl2} – ΔMPG = {diff:+7.2f} (t={t:6.2f})")
    else:
        print("No statistically significant differences between cylinder groups (t<2).")

# ----------------------------------------------------------------------
# CAR-06 – Performance by model year range
# ----------------------------------------------------------------------
def car_06(rows: List[Dict[str, str]], scores: List[float]) -> None:
    # Extract model years
    years = [safe_float(r["Model"]) for r in rows if safe_float(r["Model"]) is not None]
    if not years:
        print("== CAR-06 ==")
        print("No year information available.")
        return

    median_year = sorted(years)[len(years) // 2]

    group_indices = {"Early": [], "Late": []}
    for idx, year in enumerate(years):
        if year < median_year:
            group_indices["Early"].append(idx)
        else:
            group_indices["Late"].append(idx)

    def group_data(name: str) -> List[float]:
        return [scores[i] for i in group_indices[name] if not math.isnan(scores[i])]

    early_scores = group_data("Early")
    late_scores = group_data("Late")

    early_mean, early_std, early_n = group_stats(early_scores)
    late_mean, late_std, late_n = group_stats(late_scores)

    print("== CAR-06 ==")
    print(f"Early models (≤{median_year}) – Mean Score: {early_mean:7.4f} (n={early_n})")
    print(f"Late models (>{median_year})  – Mean Score: {late_mean:7.4f} (n={late_n})")

    # Compare means
    var_early = early_std ** 2
    var_late = late_std ** 2
    t, df = welch_t_test(early_mean, late_mean, var_early, var_late, early_n, late_n)
    diff = early_mean - late_mean
    print(f"Score difference Δ = {diff:+7.4f} (t={t:6.2f}, df={df:.1f})")
    if abs(t) >= 2:
        print("The difference is statistically significant (t≥2) – meaningful performance difference.")
    else:
        print("The evidence is weak (t<2); performance difference is not statistically supported.")

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File '{csv_path}' does not exist.")
        sys.exit(1)

    rows = read_csv(csv_path)
    if not rows:
        print("Error: CSV file is empty or has no data rows.")
        sys.exit(1)

    car_05(rows)

    scores = compute_composite_scores(rows)
    car_06(rows, scores)

if __name__ == "__main__":
    main()
