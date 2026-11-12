#!/usr/bin/env python3
"""
Skill 07: Identify promising project characteristic combinations and
prioritise proposal evaluations.

NASA-13: Find characteristic combinations that historically produced
large code sizes with low effort and low defects.

NASA-14: When evaluation resources are limited, recommend which
proposals to evaluate first in order to discover promising alternatives.
"""

import csv
import sys
import math
import numpy as np

# Reuse existing helpers
try:
    import skill_01 as s01
except ImportError:
    # Minimal fallback if skill_01 is unavailable
    import csv
    def load_data(csv_path: str):
        rows = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    r['Kloc+'] = float(r['Kloc+'])
                    r['Effort-'] = float(r['Effort-'])
                    r['Defects-'] = float(r['Defects-'])
                except (ValueError, KeyError):
                    continue
                rows.append(r)
        return rows
    s01 = type('s01', (), {'load_data': staticmethod(load_data)})

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def _is_numeric(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def get_categorical_keys(rows):
    """Return list of keys that are not numeric and not the id column."""
    if not rows:
        return []
    sample = rows[0]
    cat_keys = []
    for k, v in sample.items():
        if k in ('idX', 'Kloc+', 'Effort-', 'Defects-', 'MonthsX'):
            continue
        if not _is_numeric(v):
            cat_keys.append(k)
    return cat_keys

def parse_rows(csv_path):
    """
    Split the CSV into historical rows (complete numeric fields) and proposals
    (rows missing Effort- or Defects-).
    """
    historical = []
    proposals = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r_copy = r.copy()
            is_historical = True
            for key in ('Kloc+', 'Effort-', 'Defects-'):
                try:
                    r_copy[key] = float(r_copy[key])
                except (ValueError, KeyError):
                    is_historical = False
            if is_historical:
                historical.append(r_copy)
            else:
                proposals.append(r_copy)
    return historical, proposals

def group_combinations(rows, cat_keys):
    """
    Group historical rows by the categorical keys and compute aggregated metrics.
    Returns a list of dicts containing the group key and statistics.
    """
    groups = {}
    for r in rows:
        key = tuple((k, r[k]) for k in cat_keys)
        groups.setdefault(key, []).append(r)

    combo_stats = []
    for key, group_rows in groups.items():
        sizes = np.array([rr['Kloc+'] for rr in group_rows])
        efforts = np.array([rr['Effort-'] for rr in group_rows])
        defects = np.array([rr['Defects-'] for rr in group_rows])

        # Avoid division by zero
        mean_size = sizes.mean()
        mean_effort = efforts.mean()
        mean_defects = defects.mean()

        eff_per_kloc = mean_effort / mean_size if mean_size else float('inf')
        def_per_kloc = mean_defects / mean_size if mean_size else float('inf')

        # Composite score: larger size + smaller per‑KLOC effort and defects
        # Weight per‑KLOC metrics more heavily to favour efficiency
        score = mean_size - 3 * (eff_per_kloc + def_per_kloc)

        combo_stats.append({
            'key': key,
            'mean_size': mean_size,
            'mean_effort': mean_effort,
            'mean_defects': mean_defects,
            'eff_per_kloc': eff_per_kloc,
            'def_per_kloc': def_per_kloc,
            'score': score
        })
    return combo_stats

def rank_proposals(proposals, cat_keys, promising_combos):
    """
    Rank proposals for evaluation priority.
    Promising combos provide a bonus if proposal matches the combo.
    """
    ranked = []
    for p in proposals:
        # Base score: size (larger is better)
        size = float(p.get('Kloc+', 0))
        score = size

        # Bonus if matches any promising combo exactly
        bonus = 0
        for combo in promising_combos:
            match = True
            for k, v in combo['key']:
                if p.get(k) != v:
                    match = False
                    break
            if match:
                bonus = 100
                break

        # Partial match bonus (at least half the categorical keys match)
        if bonus == 0:
            match_count = 0
            for k, v in combo['key']:
                if p.get(k) == v:
                    match_count += 1
            # Use the best partial match across combos
            partial_bonus = 0
            if cat_keys:
                partial_bonus = 50 * (match_count / len(cat_keys))
            if partial_bonus > bonus:
                bonus = partial_bonus

        ranked.append((p, score + bonus))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    historical, proposals = parse_rows(csv_path)

    if not historical:
        print("No historical data found. Cannot proceed.")
        sys.exit(1)

    # Determine categorical keys
    cat_keys = get_categorical_keys(historical)

    # ------------------------------------------------------------------
    # NASA-13: Identify promising combinations
    # ------------------------------------------------------------------
    combo_stats = group_combinations(historical, cat_keys)
    # Sort by composite score descending
    combo_stats.sort(key=lambda x: x['score'], reverse=True)

    print("== NASA-13 ==")
    if not combo_stats:
        print("No categorical combinations found.")
    else:
        # Take top 3 combinations
        for i, combo in enumerate(combo_stats[:3], 1):
            key_desc = ", ".join(f"{k}={v}" for k, v in combo['key'])
            print(f"{i}. Combination: {key_desc}")
            print(f"   Size: {combo['mean_size']:.2f} KLOC")
            print(f"   Effort: {combo['mean_effort']:.2f} PM")
            print(f"   Defects: {combo['mean_defects']:.2f}")
            print(f"   Effort per KLOC: {combo['eff_per_kloc']:.3f}")
            print(f"   Defects per KLOC: {combo['def_per_kloc']:.3f}")

    # ------------------------------------------------------------------
    # NASA-14: Prioritise proposal evaluations
    # ------------------------------------------------------------------
    print("\n== NASA-14 ==")
    if not proposals:
        print("No proposals to evaluate.")
        sys.exit(0)

    # Use top 5 combos as "promising" (tune as needed)
    promising = combo_stats[:5]
    ranked_proposals = rank_proposals(proposals, cat_keys, promising)

    for i, (p, score) in enumerate(ranked_proposals[:3], 1):
        pid = p.get('idX', 'N/A')
        size = p.get('Kloc+', 0)
        # Determine if it matches a promising combo
        matched = any(all(p.get(k) == v for k, v in combo['key']) for combo in promising)
        print(f"{i}. Proposal ID {pid}: Size {size} KLOC, {'matches' if matched else 'partial/none'} promising combo, Score {score:.2f}")

if __name__ == "__main__":
    main()
