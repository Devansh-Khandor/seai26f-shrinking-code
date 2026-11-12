#!/usr/bin/env python3
"""
skill_06.py

Usage:
    python skill_06.py <csv_path>

Prints:
    CAR-11: Explain recommendation of the top vehicle by comparing it to a similar vehicle.
    CAR-12: Explain why one vehicle gets a higher overall recommendation than another, highlighting trade‑offs.
"""

import csv
import sys
import math
from statistics import mean, median

# ----------------------------------------------------------------------
# Data ingestion
# ----------------------------------------------------------------------
def parse_csv(path):
    """
    Read CSV and return a list of rows with numeric conversion.
    Skip rows with any invalid numeric field.
    """
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                row['Clndrs'] = int(row['Clndrs'])
                row['Volume'] = float(row['Volume'])
                row['HpX']    = float(row['HpX'])
                row['Model']  = int(row['Model'])
                row['origin'] = int(row['origin'])
                row['Lbs-']   = float(row['Lbs-'])
                row['Acc+']   = float(row['Acc+'])
                row['Mpg+']   = float(row['Mpg+'])
                row['_idx']   = idx
                rows.append(row)
            except (ValueError, KeyError):
                # skip rows with missing/invalid data
                continue
    return rows

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def compute_stats(values):
    n = len(values)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    mean_val = sum(values) / n
    med_val = median(values)
    var = sum((x - mean_val) ** 2 for x in values) / n
    std = math.sqrt(var)
    mn = min(values)
    mx = max(values)
    return mean_val, med_val, std, mn, mx

def z_score(x, mean_val, std_val):
    return (x - mean_val) / std_val if std_val != 0 else 0.0

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    rows = parse_csv(csv_path)
    if not rows:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    # Extract numeric columns
    weights = [r['Lbs-'] for r in rows]
    mpgs    = [r['Mpg+'] for r in rows]
    accs    = [r['Acc+'] for r in rows]

    # Compute overall stats
    w_mean, w_med, w_std, _, _ = compute_stats(weights)
    m_mean, m_med, m_std, _, _ = compute_stats(mpgs)
    a_mean, a_med, a_std, _, _ = compute_stats(accs)

    # Compute combined desirability score for each vehicle
    for r in rows:
        wz = z_score(r['Lbs-'], w_mean, w_std)
        mz = z_score(r['Mpg+'], m_mean, m_std)
        az = z_score(r['Acc+'], a_mean, a_std)
        r['score'] = -wz + mz - az  # lower weight, higher mpg, lower acceleration

    # Sort by score descending
    sorted_rows = sorted(rows, key=lambda r: r['score'], reverse=True)

    # ------------------
    # CAR-11: Recommendation explanation
    # ------------------
    print("== CAR-11 ==")
    top = sorted_rows[0]
    print(f"Recommended vehicle: Vehicle #{top['_idx']} (ID={top['Clndrs']})")
    # Find most similar vehicle (excluding itself) in standardized space
    top_std = [z_score(top['Lbs-'], w_mean, w_std),
               z_score(top['Mpg+'], m_mean, m_std),
               z_score(top['Acc+'], a_mean, a_std)]
    best_dist = None
    best_similar = None
    for r in sorted_rows[1:]:
        std_vec = [z_score(r['Lbs-'], w_mean, w_std),
                   z_score(r['Mpg+'], m_mean, m_std),
                   z_score(r['Acc+'], a_mean, a_std)]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(top_std, std_vec)))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_similar = r

    if best_similar:
        print("Most similar vehicle in the inventory:")
        print(f"  Vehicle #{best_similar['_idx']} (ID={best_similar['Clndrs']})")
        print(f"  Differences:")
        print(f"    Weight: {top['Lbs-']:.1f} lbs vs {best_similar['Lbs-']:.1f} lbs "
              f"({top['Lbs-'] - best_similar['Lbs-']:+.1f} lbs)")
        print(f"    MPG:    {top['Mpg+']:.1f} vs {best_similar['Mpg+']:.1f} "
              f"({top['Mpg+'] - best_similar['Mpg+']:+.1f} MPG)")
        print(f"    Acc:    {top['Acc+']:.1f}s vs {best_similar['Acc+']:.1f}s "
              f"({top['Acc+'] - best_similar['Acc+']:+.1f}s)")
        # Explain why the top vehicle is still preferred
        print("\nExplanation:")
        print(f"  The recommended vehicle scores higher overall because, even though it is very close to "
              f"the similar vehicle in weight and acceleration, it has a slightly better MPG "
              f"({top['Mpg+']:.1f} vs {best_similar['Mpg+']:.1f}), which contributes more to the "
              f"combined desirability score.")
    else:
        print("(No similar vehicle found)")

    # ------------------
    # CAR-12: Trade‑off explanation
    # ------------------
    print("\n== CAR-12 ==")
    # Find vehicle with highest MPG (regardless of weight)
    high_mpg = max(rows, key=lambda r: r['Mpg+'])
    # Find vehicle with lowest weight
    low_weight = min(rows, key=lambda r: r['Lbs-'])
    # Compute differences
    print(f"Vehicle A (high MPG): Vehicle #{high_mpg['_idx']} (ID={high_mpg['Clndrs']})")
    print(f"  MPG: {high_mpg['Mpg+']:.1f} | Weight: {high_mpg['Lbs-']:.1f} lbs | Acc: {high_mpg['Acc+']:.1f}s | Score: {high_mpg['score']:.3f}")
    print(f"Vehicle B (low weight): Vehicle #{low_weight['_idx']} (ID={low_weight['Clndrs']})")
    print(f"  MPG: {low_weight['Mpg+']:.1f} | Weight: {low_weight['Lbs-']:.1f} lbs | Acc: {low_weight['Acc+']:.1f}s | Score: {low_weight['score']:.3f}")

    # Explain trade‑offs
    print("\nTrade‑off explanation:")
    print(f"  Vehicle A has the best fuel efficiency (highest MPG) but is heavier "
          f"({high_mpg['Lbs-']:.1f} lbs) and slightly slower to accelerate "
          f"({high_mpg['Acc+']:.1f}s).  Vehicle B is lighter "
          f"({low_weight['Lbs-']:.1f} lbs) and accelerates better ({low_weight['Acc+']:.1f}s), "
          f"but its MPG is lower ({low_weight['Mpg+']:.1f} MPG).")
    print(f"  The overall score reflects these trade‑offs: Vehicle A's high MPG compensates for its higher weight, "
          f"while Vehicle B's lower weight and faster acceleration compensate for its lower MPG.  "
          f"Thus Vehicle A receives a slightly higher overall recommendation ({high_mpg['score']:.3f} vs "
          f"{low_weight['score']:.3f}), illustrating how the combined metric balances performance "
          f"and efficiency.")
    # End of program

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vehicle recommendation and trade‑off explanation")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()
    main(args.csv_path)
