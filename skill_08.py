#!/usr/bin/env python3
"""
skill_08.py

Usage:
    python skill_08.py <csv_path>

The script prints:
    * CAR-15: For a poorly rated vehicle, show alternatives with similar
      characteristics but better performance and explain the differences.
    * CAR-16: For a vehicle that does not meet preferred targets, find the
      nearest alternative that satisfies more targets while changing as few
      characteristics as possible.
"""

import sys
import math
from pathlib import Path

# Import helpers from skill_01
from skill_01 import read_csv, compute_composite_scores, format_number, mean

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def euclidean_distance(v1, v2, keys=('Lbs-', 'Mpg+', 'Acc+')):
    """Return Euclidean distance between two vehicles on selected keys."""
    return math.sqrt(sum((float(v1[k]) - float(v2[k])) ** 2 for k in keys))

# --------------------------------------------------------------------------- #
# CAR-15: Alternatives for a poorly rated vehicle
# --------------------------------------------------------------------------- #

def print_car_15_alternatives(data):
    if not data:
        print("\n== CAR-15 ==")
        print("  No vehicle records to analyze.")
        return

    # Compute composite scores for all vehicles
    scored = compute_composite_scores(data)

    # Identify the worst vehicle (highest composite score)
    worst_score, worst_vehicle = max(scored, key=lambda x: x[0])

    # Compute distances to the worst vehicle
    distances = []
    for score, vehicle in scored:
        if vehicle is worst_vehicle:
            continue
        dist = euclidean_distance(vehicle, worst_vehicle)
        distances.append((dist, vehicle, score))

    # Sort by distance (closest first) and keep those with better scores
    distances.sort(key=lambda x: (x[0], x[2]))
    better_alternatives = [t for t in distances if t[2] < worst_score]

    # Take up to 3 best alternatives
    selected = better_alternatives[:3]

    print("\n== CAR-15 ==")
    print("Alternative vehicles to a poorly rated vehicle:")
    if not selected:
        print("  No better alternatives found.")
        return

    for i, (dist, vehicle, score) in enumerate(selected, start=1):
        # Compute differences
        w_diff = vehicle['Lbs-'] - worst_vehicle['Lbs-']
        mpg_diff = vehicle['Mpg+'] - worst_vehicle['Mpg+']
        acc_diff = vehicle['Acc+'] - worst_vehicle['Acc+']

        print(
            f"  {i}. Clndrs={vehicle['Clndrs']}, Model={vehicle['Model']}, "
            f"Weight={format_number(vehicle['Lbs-'])} kg, MPG={format_number(vehicle['Mpg+'])}, "
            f"Acc={format_number(vehicle['Acc+'])} s, Composite Score={format_number(score)}"
        )
        print(
            f"     ΔWeight={format_number(w_diff)} kg, ΔMPG={format_number(mpg_diff)} mpg, "
            f"ΔAcc={format_number(acc_diff)} s"
        )

        # Explain improvements
        improvements = []
        if w_diff < 0:
            improvements.append("lighter")
        if mpg_diff > 0:
            improvements.append("higher MPG")
        if acc_diff < 0:
            improvements.append("faster acceleration")
        if improvements:
            print(f"     These improvements likely account for better performance: {', '.join(improvements)}")
        else:
            print("     No clear improvement over the poor vehicle.")

# --------------------------------------------------------------------------- #
# CAR-16: Alternative meeting more preferred targets
# --------------------------------------------------------------------------- #

def print_car_16_alternative(data):
    if not data:
        print("\n== CAR-16 ==")
        print("  No vehicle records to analyze.")
        return

    # Define preferred performance targets (based on dataset means)
    weights = [v['Lbs-'] for v in data]
    mpgs = [v['Mpg+'] for v in data]
    accs = [v['Acc+'] for v in data]
    mean_w, mean_m, mean_a = mean(weights), mean(mpgs), mean(accs)

    def target_satisfied(v):
        """Return tuple of booleans for weight, mpg, acceleration."""
        return (
            v['Lbs-'] <= mean_w,   # weight below mean
            v['Mpg+'] >= mean_m,   # mpg above mean
            v['Acc+'] <= mean_a    # acceleration time below mean
        )

    # Find a vehicle that fails at least one target (worst by target count)
    worst_candidates = []
    for v in data:
        sat = target_satisfied(v)
        sat_count = sum(sat)
        if sat_count < 3:
            worst_candidates.append((sat_count, v))
    if not worst_candidates:
        print("\n== CAR-16 ==")
        print("  All vehicles meet preferred targets.")
        return

    # Choose the vehicle with the fewest satisfied targets
    _, poor_vehicle = min(worst_candidates, key=lambda x: x[0])

    # Search for the nearest alternative that satisfies more targets
    best_alt = None
    best_sat = sum(target_satisfied(poor_vehicle))
    best_diff_fields = None

    for v in data:
        if v is poor_vehicle:
            continue
        sat = target_satisfied(v)
        sat_count = sum(sat)
        if sat_count <= best_sat:
            continue
        # Count how many fields differ
        diff_fields = sum(1 for k in ['Clndrs', 'Volume', 'HpX', 'Model', 'origin',
                                     'Lbs-', 'Acc+', 'Mpg+'] if v[k] != poor_vehicle[k])
        if (best_alt is None or
            sat_count > best_sat or
            (sat_count == best_sat and diff_fields < best_diff_fields)):
            best_alt = v
            best_sat = sat_count
            best_diff_fields = diff_fields

    if best_alt is None:
        print("\n== CAR-16 ==")
        print("  No alternative improving targets found.")
        return

    # Prepare explanation
    poor_sat = target_satisfied(poor_vehicle)
    alt_sat = target_satisfied(best_alt)
    improved_targets = [name for name, p, a in zip(['Weight','MPG','Acceleration'],
                                                   poor_sat, alt_sat) if p != a]

    print("\n== CAR-16 ==")
    print("Alternative vehicle with more performance targets satisfied:")
    print(
        f"  Poor vehicle: Clndrs={poor_vehicle['Clndrs']}, Model={poor_vehicle['Model']}, "
        f"Weight={format_number(poor_vehicle['Lbs-'])} kg, MPG={format_number(poor_vehicle['Mpg+'])} mpg, "
        f"Acc={format_number(poor_vehicle['Acc+'])} s"
    )
    print(
        f"  Alternative: Clndrs={best_alt['Clndrs']}, Model={best_alt['Model']}, "
        f"Weight={format_number(best_alt['Lbs-'])} kg, MPG={format_number(best_alt['Mpg+'])} mpg, "
        f"Acc={format_number(best_alt['Acc+'])} s"
    )
    print("  Improved targets:", ', '.join(improved_targets))
    print("  Minimal changes required (fewest differing fields):")
    for k in ['Clndrs', 'Volume', 'HpX', 'Model', 'origin', 'Lbs-', 'Acc+', 'Mpg+']:
        if best_alt[k] != poor_vehicle[k]:
            print(f"    {k}: {poor_vehicle[k]} -> {best_alt[k]}")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    data = read_csv(csv_path)
    if not data:
        print("No data found in CSV.", file=sys.stderr)
        sys.exit(1)

    # CAR-15: Alternatives for a poorly rated vehicle
    print_car_15_alternatives(data)

    # CAR-16: Alternative meeting more preferred targets
    print_car_16_alternative(data)

if __name__ == "__main__":
    main()
