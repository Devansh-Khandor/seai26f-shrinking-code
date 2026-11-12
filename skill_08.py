#!/usr/bin/env python3
"""
skill_08.py

This script implements the NASA-15 and NASA-16 requests.

NASA-15: For a project proposal predicted to require excessive effort,
         identify plausible changes to adjustable planning or process
         characteristics that might reduce its predicted effort.

NASA-16: For a project proposal with a high predicted defect count,
         identify the smallest plausible combination of changes to
         adjustable planning or process characteristics that could
         improve its predicted quality, while explaining potential trade‑offs.

The implementation reuses helper functions from the earlier modules:
* skill_01.read_csv, extract_numeric_columns, compute_stats
* skill_04.train_and_predict

The script prints a header line for each request (== NASA-15 ==,
== NASA-16 ==) followed by at least three lines of results, including
the actual numbers.
"""

import sys
import numpy as np

# Import helpers
from skill_01 import read_csv, extract_numeric_columns, compute_stats
from skill_04 import train_and_predict


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def next_level(current, levels):
    """
    Return the next higher level in 'levels' if possible, otherwise
    return the same level.
    """
    if current in levels:
        idx = levels.index(current)
        if idx < len(levels) - 1:
            return levels[idx + 1]
    return current


def previous_level(current, levels):
    """
    Return the next lower level in 'levels' if possible, otherwise
    return the same level.
    """
    if current in levels:
        idx = levels.index(current)
        if idx > 0:
            return levels[idx - 1]
    return current


# ----------------------------------------------------------------------
# NASA-15: Effort‑reduction suggestions
# ----------------------------------------------------------------------
def nasa_15(rows, reg_effort, hist_effort_stats):
    """
    Return a list of strings describing effort‑reduction suggestions
    for proposals that are predicted to require excessive effort.
    """
    # Find proposals (missing Effort- or Defects-)
    effort_vals = extract_numeric_columns(rows, ["Effort-"])
    defect_vals = extract_numeric_columns(rows, ["Defects-"])
    size_vals   = extract_numeric_columns(rows, ["Kloc+"])
    ids         = np.array([int(r["idX"]) for r in rows], dtype=int)

    proposal_mask = np.isnan(effort_vals) | np.isnan(defect_vals)
    # We only need proposals with a known size
    proposal_mask &= ~np.isnan(size_vals)

    lines = []
    threshold = hist_effort_stats["mean"] + hist_effort_stats["std"]

    for idx in np.where(proposal_mask)[0]:
        pid      = ids[idx]
        size_val = size_vals[idx]
        # Predict effort using the model
        pred_effort = reg_effort.predict([[size_val]])[0]

        # Only consider proposals that exceed the threshold
        if pred_effort <= threshold:
            continue

        # Basic line
        lines.append(f"  Proposal ID={pid}: Predicted effort={pred_effort:.1f} months")

        # Suggest size reductions
        for pct in (10, 20, 30):
            new_size = size_val * (1 - pct / 100.0)
            new_eff = reg_effort.predict([[new_size]])[0]
            lines.append(
                f"    Reduce size by {pct}% ({new_size:.1f} KLOC) => Effort={new_eff:.1f} months"
            )

        # Suggest apex upgrade (programmer capability)
        apex = rows[idx]["apex"]
        apex_levels = ["l", "h", "vh"]
        new_apex = next_level(apex, apex_levels)
        if new_apex != apex:
            # We can't adjust the model but we note the potential impact
            lines.append(
                f"    Upgrade programmer capability from '{apex}' to '{new_apex}' (potentially reduces effort)"
            )
        else:
            lines.append(
                f"    Programmer capability already at highest level '{apex}'"
            )

        # Suggest complexity reduction
        cplx = rows[idx]["cplx"]
        cplx_levels = ["l", "h", "vh"]
        new_cplx = previous_level(cplx, cplx_levels)
        if new_cplx != cplx:
            lines.append(
                f"    Reduce project complexity from '{cplx}' to '{new_cplx}' (potentially reduces effort)"
            )
        else:
            lines.append(
                f"    Complexity already at lowest level '{cplx}'"
            )

        # Stop after three proposals to satisfy the requirement
        if len(lines) >= 9:  # 3 proposals * 3 lines each (approx)
            break

    if not lines:
        lines.append("  No proposals exceed the effort threshold.")
        lines.append("  Consider reducing project size or improving capability levels.")
    return lines


# ----------------------------------------------------------------------
# NASA-16: Defect‑reduction suggestions
# ----------------------------------------------------------------------
def nasa_16(rows, reg_defect, hist_defect_stats):
    """
    Return a list of strings describing defect‑reduction suggestions
    for proposals that are predicted to have high defect counts.
    """
    # Find proposals (missing Effort- or Defects-)
    effort_vals = extract_numeric_columns(rows, ["Effort-"])
    defect_vals = extract_numeric_columns(rows, ["Defects-"])
    size_vals   = extract_numeric_columns(rows, ["Kloc+"])
    ids         = np.array([int(r["idX"]) for r in rows], dtype=int)

    proposal_mask = np.isnan(effort_vals) | np.isnan(defect_vals)
    proposal_mask &= ~np.isnan(size_vals)

    lines = []
    threshold = hist_defect_stats["mean"] + hist_defect_stats["std"]

    for idx in np.where(proposal_mask)[0]:
        pid      = ids[idx]
        size_val = size_vals[idx]
        pred_def = reg_defect.predict([[size_val]])[0]

        if pred_def <= threshold:
            continue

        lines.append(f"  Proposal ID={pid}: Predicted defects={pred_def:.1f}")

        # Suggest size reductions
        for pct in (10, 20, 30):
            new_size = size_val * (1 - pct / 100.0)
            new_def = reg_defect.predict([[new_size]])[0]
            lines.append(
                f"    Reduce size by {pct}% ({new_size:.1f} KLOC) => Defects={new_def:.1f}"
            )

        # Suggest complexity reduction
        cplx = rows[idx]["cplx"]
        cplx_levels = ["l", "h", "vh"]
        new_cplx = previous_level(cplx, cplx_levels)
        if new_cplx != cplx:
            lines.append(
                f"    Reduce project complexity from '{cplx}' to '{new_cplx}' (potentially reduces defects)"
            )
        else:
            lines.append(
                f"    Complexity already at lowest level '{cplx}'"
            )

        # Suggest tool improvement
        tool = rows[idx]["tool"]
        tool_levels = ["l", "h", "vh"]
        new_tool = next_level(tool, tool_levels)
        if new_tool != tool:
            lines.append(
                f"    Upgrade tool support from '{tool}' to '{new_tool}' (potentially improves quality)"
            )
        else:
            lines.append(
                f"    Tool support already at highest level '{tool}'"
            )

        # Stop after three proposals
        if len(lines) >= 9:
            break

    if not lines:
        lines.append("  No proposals exceed the defect threshold.")
        lines.append("  Consider reducing size, complexity, or improving tooling.")
    return lines


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path: str):
    rows = read_csv(csv_path)

    # Train models and get historical statistics
    reg_effort, reg_defect, hist_effort_stats, hist_defect_stats = train_and_predict(rows)

    # ------------------------------------------------------------------
    # NASA-15
    # ------------------------------------------------------------------
    print("== NASA-15 ==")
    for line in nasa_15(rows, reg_effort, hist_effort_stats):
        print(line)

    # ------------------------------------------------------------------
    # NASA-16
    # ------------------------------------------------------------------
    print("\n== NASA-16 ==")
    for line in nasa_16(rows, reg_defect, hist_defect_stats):
        print(line)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
