#!/usr/bin/env python3
"""
skill_07.py

This script implements the NASA-13 and NASA-14 requests.

NASA-13: Identify historically promising combinations of project characteristics
         that delivered more software with lower development effort and fewer defects.

NASA-14: When detailed project evaluations are expensive, recommend which proposals
         should be evaluated first to discover promising alternatives within a limited
         evaluation budget.

The implementation re‑uses helper functions from earlier skills:
* skill_01.read_csv, extract_numeric_columns, compute_stats
* skill_03.compute_group_stats
* skill_04.train_and_predict

The script prints a header line for each request (== NASA-13 ==, == NASA-14 ==)
followed by at least three lines of results, including actual numeric values.
"""

import sys
import numpy as np

# Import helper functions
from skill_01 import read_csv, extract_numeric_columns, compute_stats
from skill_03 import compute_group_stats
from skill_04 import train_and_predict


def nasa_13(rows):
    """
    Identify the most promising combinations of 'apex' (programmer capability)
    and 'cplx' (project complexity) that historically had low effort and
    low defect counts.  Returns a list of tuples:
        (apex, cplx, mean_effort, mean_defects, mean_size, count)
    sorted by combined score (mean_effort + mean_defects) ascending.
    """
    # Filter out rows missing effort or defects
    effort = extract_numeric_columns(rows, ['Effort-'])
    defects = extract_numeric_columns(rows, ['Defects-'])
    size = extract_numeric_columns(rows, ['Kloc+'])

    mask = ~np.isnan(effort) & ~np.isnan(defects) & ~np.isnan(size)
    effort = effort[mask]
    defects = defects[mask]
    size = size[mask]
    apex = np.array([row['apex'] for row in np.array(rows)[mask]], dtype=str)
    cplx = np.array([row['cplx'] for row in np.array(rows)[mask]], dtype=str)

    # Group by (apex, cplx)
    group_stats = compute_group_stats(effort, np.core.defchararray.add(apex, '|', cplx))
    # compute also size stats and count
    group_stats_extended = {}
    for key, (m_effort, s_effort, n_effort) in group_stats.items():
        # Extract indices for this group
        ap, cp = key.split('|')
        idx = (apex == ap) & (cplx == cp)
        m_defects = np.nanmean(defects[idx]) if np.sum(idx) > 0 else np.nan
        m_size = np.nanmean(size[idx]) if np.sum(idx) > 0 else np.nan
        group_stats_extended[key] = (m_effort, m_defects, m_size, np.sum(idx))

    # Compute overall medians to filter promising combos
    median_effort = np.nanmedian(effort)
    median_defects = np.nanmedian(defects)

    # Build list of promising combos
    combos = []
    for key, (m_eff, m_def, m_sz, n) in group_stats_extended.items():
        if n < 2:
            continue
        if m_eff <= median_effort and m_def <= median_defects:
            score = m_eff + m_def
            combos.append((key, m_eff, m_def, m_sz, n, score))

    # Sort by combined score ascending
    combos.sort(key=lambda x: x[5])

    # Keep top 3
    top = combos[:3]

    # Prepare output lines
    lines = []
    for key, m_eff, m_def, m_sz, n, _ in top:
        ap, cp = key.split('|')
        lines.append(f"  Apex={ap}, Complexity={cp}: "
                     f"mean effort={m_eff:.1f} mo, mean defects={m_def:.1f}, "
                     f"mean size={m_sz:.1f} KLOC, projects={n}")
    if not lines:
        lines.append("  No promising combinations found.")
    return lines


def nasa_14(rows, reg_effort, reg_defect, hist_effort_stats, hist_defect_stats):
    """
    Recommend proposals to evaluate first (with limited budget) based on predicted
    effort and defects.  Returns a list of strings with proposal ID and predicted
    values, sorted by a combined 'promising' score (lower is better).
    """
    # Identify proposals (missing Effort- or Defects-)
    effort = extract_numeric_columns(rows, ['Effort-'])
    defects = extract_numeric_columns(rows, ['Defects-'])
    size = extract_numeric_columns(rows, ['Kloc+'])
    ids = np.array([int(row['idX']) for row in rows], dtype=int)

    proposal_mask = np.isnan(effort) | np.isnan(defects)
    if not np.any(proposal_mask):
        return ["  No proposals found that require evaluation."]

    # For each proposal with size available, predict effort and defects
    proposal_ids = ids[proposal_mask]
    proposal_sizes = size[proposal_mask]
    proposal_effort_pred = np.array([reg_effort.predict([[s]])[0] if not np.isnan(s) else np.nan
                                    for s in proposal_sizes])
    proposal_defect_pred = np.array([reg_defect.predict([[s]])[0] if not np.isnan(s) else np.nan
                                    for s in proposal_sizes])

    # Compute a simple promising score: (pred_effort + pred_defects) / size
    # Avoid division by zero
    mask_valid = (~np.isnan(proposal_sizes) &
                  ~np.isnan(proposal_effort_pred) &
                  ~np.isnan(proposal_defect_pred) &
                  (proposal_sizes > 0))
    if not np.any(mask_valid):
        return ["  No proposals with sufficient size information to evaluate."]

    promising_score = (proposal_effort_pred[mask_valid] + proposal_defect_pred[mask_valid]) / proposal_sizes[mask_valid]

    # Sort ascending
    sorted_idx = np.argsort(promising_score)
    top_indices = sorted_idx[:3]  # recommend top 3

    lines = []
    for i in top_indices:
        idx = np.where(mask_valid)[0][i]
        pid = proposal_ids[idx]
        eff = proposal_effort_pred[idx]
        defc = proposal_defect_pred[idx]
        sz = proposal_sizes[idx]
        score = promising_score[idx]
        lines.append(f"  Proposal ID={pid}: Pred Effort={eff:.1f} mo, Pred Defects={defc:.1f}, "
                     f"Size={sz:.1f} KLOC, Score={score:.3f}")
    return lines


def main(csv_path: str):
    rows = read_csv(csv_path)

    # Train models once (used for NASA-14)
    reg_effort, reg_defect, hist_effort_stats, hist_defect_stats = train_and_predict(rows)

    # ---------------------------------------------------
    # NASA-13
    # ---------------------------------------------------
    print("== NASA-13 ==")
    for line in nasa_13(rows):
        print(line)

    # ---------------------------------------------------
    # NASA-14
    # ---------------------------------------------------
    print("\n== NASA-14 ==")
    for line in nasa_14(rows, reg_effort, reg_defect,
                       hist_effort_stats, hist_defect_stats):
        print(line)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
