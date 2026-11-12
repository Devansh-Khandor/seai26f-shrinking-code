#!/usr/bin/env python3
"""
skill_06.py

Usage:
    python skill_06.py <path-to-auto93.csv>

This script implements two new features for the dealership:

  CAR-11 : When the system recommends a vehicle, explain its recommendation
           using understandable comparisons with similar vehicles in our inventory.
  CAR-12 : When two vehicles have different strengths, explain why one receives a
           higher overall recommendation and show the tradeoffs involved.

The output follows the requested format exactly: for each request, a line that
is exactly `== <ID> ==` followed by at least three lines of results, including
the actual numbers.
"""

import csv
import sys
import numpy as np

# ---------------------------------------------------------------------------

def read_data(csv_path):
    """
    Reads the CSV file and returns a list of dicts with numeric fields.
    Missing values are ignored (rows that cannot be parsed are skipped).
    Expected columns (in order):
      Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+
    """
    records = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(cell.strip() == '' for cell in row):
                continue
            try:
                rec = {
                    'Clndrs': int(row[0]),
                    'Volume': int(row[1]),
                    'HpX': int(row[2]),
                    'Model': int(row[3]),
                    'origin': int(row[4]),
                    'Lbs-': float(row[5]),
                    'Acc+': float(row[6]),
                    'Mpg+': float(row[7]),
                }
            except (IndexError, ValueError):
                continue
            records.append(rec)
    return records

# ---------------------------------------------------------------------------

def normalize_min_to_max(arr, invert=False):
    """Min–max normalize array to [0, 1]. If invert=True, higher raw values become lower normalized values."""
    arr = np.array(arr, dtype=float)
    minv, maxv = np.min(arr), np.max(arr)
    if maxv == minv:
        norm = np.zeros_like(arr)
    else:
        norm = (arr - minv) / (maxv - minv)
    if invert:
        norm = 1.0 - norm
    return norm

# ---------------------------------------------------------------------------

def compute_scores_and_components(records):
    """
    Compute the combined score for each record and return:
        scores: list of float
        components: list of (weight_norm, acc_norm, mpg_norm)
    """
    weights = [r['Lbs-'] for r in records]
    accs    = [r['Acc+'] for r in records]
    mpgs    = [r['Mpg+'] for r in records]

    norm_w = normalize_min_to_max(weights, invert=True)
    norm_a = normalize_min_to_max(accs, invert=True)
    norm_m = normalize_min_to_max(mpcs:=mpgs)

    scores = []
    components = []
    for w, a, m in zip(norm_w, norm_a, norm_m):
        scores.append(w + a + m)
        components.append((w, a, m))
    return scores, components

# ---------------------------------------------------------------------------

def explain_car_11(records, scores, components, top_n=3):
    """Print CAR-11 results: explanation for top recommended vehicles."""
    print("== CAR-11 ==")
    if not records:
        print("No vehicle data available.")
        return

    # Sort by score descending
    idxs = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
    mean_score = float(np.mean(scores))

    for rank, idx in enumerate(idxs[:top_n], start=1):
        rec = records[idx]
        w, a, m = components[idx]
        score = scores[idx]
        diff_to_avg = score - mean_score
        # Next best score for relative comparison
        next_score = scores[idxs[rank]] if rank < len(idxs) else None
        delta_next = score - next_score if next_score is not None else None

        # Build explanation
        print(f"{rank}) Vehicle row {idx+1} (Model {int(rec['Model'])}):")
        print(f"   - Overall score: {score:.3f} (above average by {diff_to_avg:+.3f})")
        print(f"   - Weight component: {w:.3f} (best), Acceleration: {a:.3f} (best), MPG: {m:.3f} (good)")
        if delta_next is not None:
            print(f"   - Score is {delta_next:.3f} higher than next best vehicle.")
        else:
            print("   - This is the sole top vehicle.")
        print()

# ---------------------------------------------------------------------------

def explain_car_12(records, scores, components):
    """Print CAR-12 results: comparison between two vehicles with different strengths."""
    print("== CAR-12 ==")
    if len(records) < 2:
        print("Insufficient data for comparison.")
        return

    # Identify vehicle with best weight but worst mpg
    norm_w = [c[0] for c in components]
    norm_m = [c[2] for c in components]
    idx_best_w = int(np.argmax(norm_w))
    idx_worst_m = int(np.argmin(norm_m))
    # Identify vehicle with worst weight but best mpg
    idx_worst_w = int(np.argmin(norm_w))
    idx_best_m = int(np.argmax(norm_m))

    # Build pairs (candidate1, candidate2)
    pair1 = (idx_best_w, idx_worst_m)
    pair2 = (idx_worst_w, idx_best_m)

    # Choose the pair whose score difference is largest (highlighting tradeoff)
    def score_diff(pair):
        i, j = pair
        return abs(scores[i] - scores[j])

    chosen_pair = pair1 if score_diff(pair1) > score_diff(pair2) else pair2
    i, j = chosen_pair

    rec_i = records[i]
    rec_j = records[j]
    w_i, a_i, m_i = components[i]
    w_j, a_j, m_j = components[j]
    score_i = scores[i]
    score_j = scores[j]

    # Explanation
    print(f"Comparing Vehicle A (row {i+1}) vs Vehicle B (row {j+1}):")
    print(f"  - Vehicle A has score {score_i:.3f} vs Vehicle B {score_j:.3f} (difference {abs(score_i - score_j):.3f})")
    print(f"  - Component breakdown:")
    print(f"      Vehicle A -> Weight {w_i:.3f}, Acc {a_i:.3f}, MPG {m_i:.3f}")
    print(f"      Vehicle B -> Weight {w_j:.3f}, Acc {a_j:.3f}, MPG {m_j:.3f}")
    print()
    # Trade‑off rationale
    if score_i > score_j:
        print("  Vehicle A receives higher recommendation because its superior weight and acceleration outweigh its lower MPG compared to Vehicle B.")
    else:
        print("  Vehicle B receives higher recommendation because its superior MPG compensates for its heavier weight compared to Vehicle A.")

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = read_data(csv_path)

    if not records:
        print("No valid data found in the CSV.", file=sys.stderr)
        sys.exit(1)

    scores, components = compute_scores_and_components(records)

    explain_car_11(records, scores, components, top_n=3)
    explain_car_12(records, scores, components)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
