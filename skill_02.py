#!/usr/bin/env python3
"""
Skill 02 – NASA effort and defect estimation

This script reads a NASA project CSV file and prints two analyses:

  * NASA-03 – Predict Effort- for all held‑out rows (those whose
    1‑based index is divisible by 5) using a tree built from the
    other rows.  It also prints the mean absolute error of the
    predictions.

  * NASA-04 – Predict Defects- for the same held‑out rows and give a
    simple uncertainty range (mean ± standard deviation).  It
    prints the estimates and the uncertainty for each row.

The output follows the required format, with a machine‑checked
JSON line after each section.

The script uses only the existing `ezr` library and the helper
functions from `skill_01.py`.  It is fully generic and works on
any CSV file with the same column names as in the specification.
"""

import json
import sys
import math
import ezr

# ----------------------------------------------------------------------
# Helpers – augment tree leaves with mean effort and defect stats

def _augment_stats(node: ezr.NODE, eff_col: ezr.NUM, def_col: ezr.NUM) -> None:
    """Recursively store average Effort- and Defect- statistics in leaf nodes."""
    if node.kids:
        for kid in node.kids:
            _augment_stats(kid, eff_col, def_col)
    else:
        # Effort statistics
        effs = [r[eff_col.at] for r in node.rows if r[eff_col.at] != "?"]
        node.avg_effort = sum(effs) / len(effs) if effs else None

        # Defect statistics
        defs = [r[def_col.at] for r in node.rows if r[def_col.at] != "?"]
        if defs:
            mean_def = sum(defs) / len(defs)
            # Use unbiased sample variance
            if len(defs) > 1:
                var = sum((d - mean_def) ** 2 for d in defs) / (len(defs) - 1)
            else:
                var = 0.0
            sd = math.sqrt(var)
            node.avg_defect = mean_def
            node.sd_defect = sd
        else:
            node.avg_defect = node.sd_defect = None

# ----------------------------------------------------------------------
# Prediction helpers

def _predict_effort(node: ezr.NODE, row: ezr.ROW) -> float:
    leaf = ezr.leaf(node, row)
    return leaf.avg_effort if leaf.avg_effort is not None else 0.0

def _predict_defect(node: ezr.NODE, row: ezr.ROW) -> tuple[float, float, float]:
    leaf = ezr.leaf(node, row)
    mean = leaf.avg_defect if leaf.avg_defect is not None else 0.0
    sd   = leaf.sd_defect   if leaf.sd_defect   is not None else 0.0
    # Use mean ± sd as the uncertainty interval
    low = max(0.0, mean - sd)
    high = mean + sd
    return mean, low, high

# ----------------------------------------------------------------------
# Main analysis

def main(csv_path: str) -> None:
    # Load table
    tbl = ezr.Tbl(ezr.csv(csv_path))

    # Identify goal columns
    eff_col = next(col for col in tbl.cols.values() if col.txt == "Effort-")
    def_col = next(col for col in tbl.cols.values() if col.txt == "Defects-")

    # Determine held‑out rows (index divisible by 5, 1‑based)
    held_out_rows = [row for i, row in enumerate(tbl.rows, start=1) if i % 5 == 0]
    history_rows = [row for i, row in enumerate(tbl.rows, start=1) if i % 5 != 0]

    # ------------------------------------------------------------------
    # NASA‑03 – Effort prediction

    # Build training set: history rows with non‑missing Effort-
    training_eff = [row for row in history_rows if row[eff_col.at] != "?"]
    lab_eff = ezr.clone(tbl, training_eff)

    # Build tree and augment leaf stats
    tree_eff = ezr.tree(lab_eff, lab_eff.rows)
    _augment_stats(tree_eff, eff_col, def_col)

    # Predict on held‑out rows and compute MAE
    predictions_eff = {}
    abs_errs = []
    for row in held_out_rows:
        pred = _predict_effort(tree_eff, row)
        predictions_eff[str(tbl.rows.index(row) + 1)] = pred
        # If true value missing, ignore for MAE
        true_val = row[eff_col.at]
        if true_val != "?":
            abs_errs.append(abs(pred - true_val))

    mae = sum(abs_errs) / len(abs_errs) if abs_errs else 0.0

    # Output NASA‑03 section
    print("== NASA-03 ==")
    for rid, val in predictions_eff.items():
        print(f"{rid}: {val:.4f}")
    print(f"MAE: {mae:.4f}")

    ans_03 = {
        "predictions": predictions_eff,
        "mae": round(mae, 4)
    }
    print(f"ANSWER NASA-03 {json.dumps(ans_03, separators=(',', ':'))}")

    # ------------------------------------------------------------------
    # NASA‑04 – Defect prediction with uncertainty

    # Build training set: history rows with non‑missing Defects-
    training_def = [row for row in history_rows if row[def_col.at] != "?"]
    lab_def = ezr.clone(tbl, training_def)

    # Build tree and augment leaf stats
    tree_def = ezr.tree(lab_def, lab_def.rows)
    _augment_stats(tree_def, eff_col, def_col)

    # Predict on held‑out rows
    predictions_def = {}
    for row in held_out_rows:
        mean, low, high = _predict_defect(tree_def, row)
        predictions_def[str(tbl.rows.index(row) + 1)] = {
            "estimate": round(mean, 4),
            "low": round(low, 4),
            "high": round(high, 4)
        }

    # Output NASA‑04 section
    print("== NASA-04 ==")
    for rid, rec in predictions_def.items():
        print(f"{rid}: estimate={rec['estimate']:.4f} low={rec['low']:.4f} high={rec['high']:.4f}")

    ans_04 = {"predictions": predictions_def}
    print(f"ANSWER NASA-04 {json.dumps(ans_04, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Entry point

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_02.py <csv_path>")
    main(sys.argv[1])
