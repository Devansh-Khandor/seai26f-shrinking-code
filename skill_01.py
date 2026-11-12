#!/usr/bin/env python3
"""
skill_01.py

Usage:
    python skill_01.py <csv_path>

Prints:
    CAR-01: Summary statistics of weight, fuel efficiency, and acceleration.
    CAR-02: Top vehicles based on a combined score of low weight, high MPG, and low acceleration.
"""

import csv
import sys
import math
from statistics import median
from argparse import ArgumentParser


def parse_csv(path):
    """
    Reads the CSV file and returns a list of rows as dicts with numeric conversions.
    """
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                row['Lbs-'] = float(row['Lbs-'])
                row['Mpg+'] = float(row['Mpg+'])
                row['Acc+'] = float(row['Acc+'])
                # Keep original index for display purposes
                row['_idx'] = i
                rows.append(row)
            except ValueError:
                # Skip rows with invalid numeric data
                continue
    return rows


def compute_stats(values):
    """
    Returns mean, median, std, min, max of a list of numbers.
    """
    n = len(values)
    if n == 0:
        return (float('nan'), float('nan'), float('nan'), float('nan'), float('nan'))

    mean = sum(values) / n
    med = median(values)
    # population std
    var = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(var)
    mn = min(values)
    mx = max(values)
    return mean, med, std, mn, mx


def main(csv_path):
    rows = parse_csv(csv_path)

    if not rows:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    # Extract columns
    weights = [r['Lbs-'] for r in rows]
    mpgs = [r['Mpg+'] for r in rows]
    accs = [r['Acc+'] for r in rows]

    # Compute stats for CAR-01
    weight_mean, weight_med, weight_std, weight_min, weight_max = compute_stats(weights)
    mpg_mean, mpg_med, mpg_std, mpg_min, mpg_max = compute_stats(mpgs)
    acc_mean, acc_med, acc_std, acc_min, acc_max = compute_stats(accs)

    # Print CAR-01
    print("== CAR-01 ==")
    print(f"Weight - mean: {weight_mean:.2f} lbs, median: {weight_med:.2f} lbs, std: {weight_std:.2f} lbs, min: {weight_min:.2f} lbs, max: {weight_max:.2f} lbs")
    print(f"Fuel Efficiency (MPG) - mean: {mpg_mean:.2f} MPG, median: {mpg_med:.2f} MPG, std: {mpg_std:.2f} MPG, min: {mpg_min:.2f} MPG, max: {mpg_max:.2f} MPG")
    print(f"Acceleration (0-60 mph in seconds) - mean: {acc_mean:.2f} s, median: {acc_med:.2f} s, std: {acc_std:.2f} s, min: {acc_min:.2f} s, max: {acc_max:.2f} s")

    # Compute z-scores
    def z(x, mean, std):
        return (x - mean) / std if std != 0 else 0.0

    # Compute combined score: lower weight => better, higher mpg => better, lower acceleration => better
    # Score = -weight_z + mpg_z - acc_z
    combined = []
    for r in rows:
        wz = z(r['Lbs-'], weight_mean, weight_std)
        mz = z(r['Mpg+'], mpg_mean, mpg_std)
        az = z(r['Acc+'], acc_mean, acc_std)
        score = -wz + mz - az
        combined.append({'row': r, 'score': score})

    # Sort descending by score
    combined.sort(key=lambda x: x['score'], reverse=True)

    # Print CAR-02
    print("\n== CAR-02 ==")
    print("Top vehicles offering best combination (low weight, high MPG, low acceleration):")
    top_n = min(5, len(combined))
    for i in range(top_n):
        r = combined[i]['row']
        s = combined[i]['score']
        print(f"{i+1}. Weight: {r['Lbs-']:.1f} lbs, MPG: {r['Mpg+']:.1f}, Acc: {r['Acc+']:.1f} s, Score: {s:.3f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Vehicle inventory analysis")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()
    main(args.csv_path)
