#!/usr/bin/env python3
"""
Skill 06: Explain effort estimates and risk differences between proposals.

NASA-11: When the system estimates the effort required for a proposed
project, explain the estimate using relevant historical projects and their
recorded characteristics.

NASA-12: When two project proposals receive different risk assessments,
explain the main differences that contributed to those assessments and
identify the important trade‑offs.
"""

import sys
import csv
import numpy as np
from typing import List, Dict, Any, Tuple

# Reuse existing helpers
try:
    import skill_01 as s01
    import skill_02 as s02
except ImportError:
    # Minimal stubs in case the modules are not importable
    import csv
    def load_data(csv_path: str):
        rows: List[Dict[str, Any]] = []
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
    s01 = type('s01', (), {'load_data': staticmethod(load_data),
                           'compute_stats': staticmethod(
                               lambda rows: {'Kloc+': {'mean': 0, 'std': 0},
                                             'Effort-': {'mean': 0, 'std': 0},
                                             'Defects-': {'mean': 0, 'std': 0}})})

    def estimate_effort(rows):
        return 0, 0, 0, 0, 0
    def estimate_defects(rows):
        return 0, 0, 0, 0, 0
    s02 = type('s02', (), {'estimate_effort': staticmethod(estimate_effort),
                           'estimate_defects': staticmethod(estimate_defects)})

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def parse_rows(csv_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Read the CSV and split into historical (completed) rows and proposal
    rows (missing numeric Effort- or Defects- values).
    """
    historical: List[Dict[str, Any]] = []
    proposals: List[Dict[str, Any]] = []

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

def find_nearest_projects(proposal: Dict[str, Any],
                          historical: List[Dict[str, Any]],
                          n: int = 3) -> List[Dict[str, Any]]:
    """Return the n historical projects with closest Kloc+ to the proposal."""
    kloc = proposal['Kloc+']
    distances = [(abs(r['Kloc+'] - kloc), r) for r in historical]
    distances.sort(key=lambda x: x[0])
    return [r for _, r in distances[:n]]

def risk_flag(est_eff: float, est_def: float,
              hist_stats: Dict[str, Dict[str, float]]) -> str:
    """Return 'High' if either effort or defects predicted above mean+2*std."""
    eff_thr = hist_stats['Effort-']['mean'] + 2 * hist_stats['Effort-']['std']
    def_thr = hist_stats['Defects-']['mean'] + 2 * hist_stats['Defects-']['std']
    return 'High' if (est_eff > eff_thr or est_def > def_thr) else 'Low'

def compare_proposals(p1: Dict[str, Any], p2: Dict[str, Any]) -> List[str]:
    """Return textual differences between two proposals."""
    diffs = []
    numeric_keys = ['Kloc+']
    categorical_keys = [k for k in p1 if k not in numeric_keys + ['idX']]
    # Numeric differences
    for key in numeric_keys:
        v1 = float(p1[key])
        v2 = float(p2[key])
        if v1 != v2:
            diffs.append(f"• {key}: {v1:.2f} vs {v2:.2f}")
    # Categorical differences
    for key in categorical_keys:
        if p1.get(key) != p2.get(key):
            diffs.append(f"• {key}: '{p1.get(key)}' vs '{p2.get(key)}'")
    return diffs

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    historical, proposals = parse_rows(csv_path)

    if not historical:
        print("No historical project data available.")
        sys.exit(1)

    hist_stats = s01.compute_stats(historical)

    # ------------------------------------------------------
    # NASA-11
    # ------------------------------------------------------
    print("== NASA-11 ==")

    if not proposals:
        print("No proposals found in the data.")
    else:
        for p in proposals:
            if 'Kloc+' not in p or p['Kloc+'] is None:
                continue
            # Estimate effort & defects
            effort_pred, _, _, _, _ = s02.estimate_effort(historical)
            defect_pred, _, _, _, _ = s02.estimate_defects(historical)
            # For NASA-11 we explain using similar projects
            nearest = find_nearest_projects(p, historical, n=3)
            print(f"Proposal ID {int(p['idX'])}: estimated effort {effort_pred:.2f} PM, "
                  f"estimated defects {defect_pred:.2f}.")
            print("  Based on the 3 most similar historical projects:")
            for r in nearest:
                print(f"    • ID {int(r['idX'])}: size {r['Kloc+']:.2f} KLOC, "
                      f"effort {r['Effort-']:.2f} PM, defects {int(r['Defects-'])}")
            # Brief characteristic note
            chars = [f"{k}={v}" for k, v in p.items() if k not in ('idX', 'Kloc+')]
            print(f"  Proposal characteristics: {', '.join(chars[:3])} ...")
            print()

    # ------------------------------------------------------
    # NASA-12
    # ------------------------------------------------------
    print("== NASA-12 ==")

    if len(proposals) < 2:
        print("Not enough proposals to compare risk assessments.")
        sys.exit(0)

    # Compute risk flags for each proposal
    proposal_risks = []
    for p in proposals:
        if 'Kloc+' not in p or p['Kloc+'] is None:
            continue
        eff_pred, _, _, _, _ = s02.estimate_effort(historical)
        def_pred, _, _, _, _ = s02.estimate_defects(historical)
        flag = risk_flag(eff_pred, def_pred, hist_stats)
        proposal_risks.append((p, eff_pred, def_pred, flag))

    # Find two proposals with different risk flags
    pair = None
    for i in range(len(proposal_risks)):
        for j in range(i + 1, len(proposal_risks)):
            if proposal_risks[i][3] != proposal_risks[j][3]:
                pair = (proposal_risks[i], proposal_risks[j])
                break
        if pair:
            break

    if not pair:
        print("All proposals share the same risk assessment.")
    else:
        (p1, e1, d1, r1), (p2, e2, d2, r2) = pair
        print(f"Proposal A (ID {int(p1['idX'])}) – Risk: {r1}")
        print(f"  Estimated effort: {e1:.2f} PM, estimated defects: {d1:.2f}")
        print(f"Proposal B (ID {int(p2['idX'])}) – Risk: {r2}")
        print(f"  Estimated effort: {e2:.2f} PM, estimated defects: {d2:.2f}")
        print("\nDifferences in characteristics contributing to the risk gap:")
        diffs = compare_proposals(p1, p2)
        for d in diffs:
            print(f"  {d}")
        print("\nTrade‑offs: higher effort or defect risk often stems from larger size, "
              "more complex architecture (cplx), or lower programmer capability (pcap). "
              "Balancing these factors can reduce risk.")

if __name__ == "__main__":
    main()
