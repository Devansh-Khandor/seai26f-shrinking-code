#!/usr/bin/env python3
"""
Skill 04: Real‑time assessment and dynamic inventory update for used‑car dealership.

CAR-07: As new vehicle records arrive, assess each one immediately and indicate whether it
belongs among our more desirable inventory options.

CAR-08: Keep inventory summaries (weight, mpg, acceleration) and a recommended‑vehicle
shortlist up to date as additional vehicle records are received, without requiring staff
to restart the analysis manually.
"""

import csv
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# Incremental statistics
# ----------------------------------------------------------------------
class RunningStat:
    """Maintain running count, mean, min, max for a numeric stream."""
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.min = None
        self.max = None

    def update(self, value):
        if value is None:
            return
        self.n += 1
        self.sum += value
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value > self.max:
            self.max = value

    @property
    def mean(self):
        return self.sum / self.n if self.n else 0.0

# ----------------------------------------------------------------------
# Overall score calculation
# ----------------------------------------------------------------------
def compute_overall_score(row, stats):
    """Return overall score (0–3) for a row, using current min/max stats."""
    w = row.get('Lbs-')
    m = row.get('Mpg+')
    a = row.get('Acc+')
    if w is None or m is None or a is None:
        return None

    wmin, wmax = stats['weight'].min, stats['weight'].max
    mmin, mmax = stats['mpg'].min, stats['mpg'].max
    amin, amax = stats['acc'].min, stats['acc'].max

    # Normalised components
    w_score = (wmax - w) / (wmax - wmin) if wmax != wmin else 1.0
    m_score = (m - mmin) / (mmax - mmin) if mmax != mmin else 1.0
    a_score = (amax - a) / (amax - amin) if amax != amin else 1.0

    return w_score + m_score + a_score

# ----------------------------------------------------------------------
# Shortlist handling
# ----------------------------------------------------------------------
def insert_into_shortlist(shortlist, entry, max_len=3):
    """Insert a (score, row) entry into the shortlist, keeping it sorted by descending score."""
    shortlist.append(entry)
    shortlist.sort(key=lambda x: x[0], reverse=True)
    if len(shortlist) > max_len:
        shortlist.pop()

# ----------------------------------------------------------------------
# CSV parsing
# ----------------------------------------------------------------------
def parse_row(row):
    """Convert numeric fields to float or None."""
    numeric_keys = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin',
                    'Lbs-', 'Acc+', 'Mpg+']
    for k in numeric_keys:
        try:
            val = row.get(k, '')
            row[k] = float(val) if val != '' else None
        except (ValueError, KeyError):
            row[k] = None
    return row

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    # Running statistics
    stats = {
        'weight': RunningStat(),
        'mpg': RunningStat(),
        'acc': RunningStat()
    }

    # Running stats for scores (to decide desirability)
    score_stat = RunningStat()

    # Shortlist of top vehicles by overall score
    shortlist = []

    record_counter = 0

    print("== CAR-07 ==")
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                row = parse_row(raw_row)
                record_counter += 1

                # Update basic stats
                stats['weight'].update(row.get('Lbs-'))
                stats['mpg'].update(row.get('Mpg+'))
                stats['acc'].update(row.get('Acc+'))

                # Compute overall score using current stats
                score = compute_overall_score(row, stats)
                if score is None:
                    # Cannot score due to missing data; skip classification
                    print(f"Record {record_counter}: Missing data -> Cannot assess desirability")
                    continue

                # Update score statistics
                score_stat.update(score)

                # Determine desirability relative to current average score
                current_mean = score_stat.mean
                desirable = score > current_mean

                # Insert into shortlist
                insert_into_shortlist(shortlist, (score, row))

                # Print assessment
                status = "DESIRABLE" if desirable else "Not desirable"
                print(f"Record {record_counter}: Lbs- {row['Lbs-']:.1f}, "
                      f"Mpg+ {row['Mpg+']:.1f}, Acc+ {row['Acc+']:.1f} -> "
                      f"Score {score:.3f} [{status}]")
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    # CAR-08: Final inventory summary and shortlist
    print("\n== CAR-08 ==")
    print(f"Total records processed: {record_counter}")

    def fmt_range(stat, name):
        min_val = stat.min if stat.min is not None else 0.0
        max_val = stat.max if stat.max is not None else 0.0
        mean_val = stat.mean
        print(f"{name:>12} | Mean: {mean_val:>8.2f} | Min: {min_val:>6.2f} | Max: {max_val:>6.2f}")

    fmt_range(stats['weight'], "Weight (lbs)")
    fmt_range(stats['mpg'], "Fuel Efficiency (mpg)")
    fmt_range(stats['acc'], "Acceleration (s)")

    # Shortlist output
    print("\nRecommended vehicles (top 3 by overall score):")
    if not shortlist:
        print("No vehicles with sufficient data to score.")
    else:
        for i, (sc, r) in enumerate(shortlist, start=1):
            print(f"{i}. Lbs- {r['Lbs-']:.1f}, Mpg+ {r['Mpg+']:.1f}, Acc+ {r['Acc+']:.1f}, Score {sc:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
