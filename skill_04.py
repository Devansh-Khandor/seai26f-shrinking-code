#!/usr/bin/env python3
"""
Skill 04 – NASA assessment and incremental updates

This script reads a NASA project CSV file and prints two analyses:
  * NASA-07 – For each new proposal (arrival), predict Effort- and
    Defects- from the history and earlier arrivals.  Alert when either
    prediction is unusually high (more than 2 standard deviations
    above the historical mean).
  * NASA-08 – After 0, 9 and 19 arrivals, print updated goal
    summaries and a 5‑row shortlist of the best projects seen so far.

The output follows the required format, with a machine‑checked JSON line
after each section.  All calculations are derived from the CSV file
supplied on the command line.
"""

import json
import math
import sys
import ezr
import skill_01 as s01  # provides compute_goal_stats & compute_top_projects


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _augment_stats(node, eff_col, def_col):
    """
    Recursively compute and store average Effort- and Defect- values
    in each leaf node of the tree.
    """
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
            if len(defs) > 1:
                var = sum((d - mean_def) ** 2 for d in defs) / (len(defs) - 1)
            else:
                var = 0.0
            node.avg_defect = mean_def
            node.sd_defect = math.sqrt(var)
        else:
            node.avg_defect = node.sd_defect = None


def _predict_effort(node, row):
    leaf = ezr.leaf(node, row)
    return leaf.avg_effort if leaf.avg_effort is not None else 0.0


def _predict_defect(node, row):
    leaf = ezr.leaf(node, row)
    mean = leaf.avg_defect if leaf.avg_defect is not None else 0.0
    sd   = leaf.sd_defect   if leaf.sd_defect   is not None else 0.0
    return mean, max(0.0, mean - sd), mean + sd


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    # Load the full table
    full_tbl = ezr.Tbl(ezr.csv(csv_path))
    n_rows = len(full_tbl.rows)

    # Determine arrivals (last round(0.2 * n_rows) rows)
    arrivals_count = round(0.2 * n_rows)
    arrivals_count = min(arrivals_count, n_rows)
    history_len = n_rows - arrivals_count

    # History rows
    history_rows = full_tbl.rows[:history_len]

    # Compute historical statistics for alert thresholds
    eff_col = next(c for c in full_tbl.cols.values() if c.txt == "Effort-")
    def_col = next(c for c in full_tbl.cols.values() if c.txt == "Defects-")

    hist_eff_vals = [r[eff_col.at] for r in history_rows if r[eff_col.at] != "?"]
    hist_def_vals = [r[def_col.at] for r in history_rows if r[def_col.at] != "?"]

    def mean_and_sd(vals):
        if not vals:
            return 0.0, 0.0
        m = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        else:
            var = 0.0
        return m, math.sqrt(var)

    mean_eff, sd_eff = mean_and_sd(hist_eff_vals)
    mean_def, sd_def = mean_and_sd(hist_def_vals)

    # ------------------------------------------------------------------
    # NASA-07: predictions for each arrival
    # ------------------------------------------------------------------
    assessments = []

    for i in range(1, arrivals_count + 1):
        # Subset of rows up to this arrival (history + earlier arrivals)
        subset_rows = full_tbl.rows[:history_len + i]
        lab_tbl = ezr.clone(full_tbl, subset_rows)

        # Build tree and augment leaf statistics
        tree_root = ezr.tree(lab_tbl, lab_tbl.rows)
        _augment_stats(tree_root, eff_col, def_col)

        # Current arrival row
        arrival_row = subset_rows[history_len]  # first arrival in this subset
        pred_eff = _predict_effort(tree_root, arrival_row)
        pred_def_mean, _, _ = _predict_defect(tree_root, arrival_row)

        # Alert if unusually high
        alert = (pred_eff > mean_eff + 2 * sd_eff) or (pred_def_mean > mean_def + 2 * sd_def)

        assessments.append({
            "row": history_len + i,  # global row number (1‑based)
            "predicted_effort": pred_eff,
            "predicted_defects": pred_def_mean,
            "alert": alert
        })

    # ------------------------------------------------------------------
    # NASA-08: checkpoints after 0, 9 and 19 arrivals
    # ------------------------------------------------------------------
    checkpoints = []
    for m in (0, 9, 19):
        if m > arrivals_count:
            continue
        # Rows up to this checkpoint
        ckpt_rows = full_tbl.rows[:history_len + m]
        ckpt_tbl = ezr.clone(full_tbl, ckpt_rows)

        # Goal summaries
        goal_cols = [c for c in ckpt_tbl.cols.values() if c.txt.endswith(('+', '-'))]
        summary = {}
        for c in goal_cols:
            stats = s01.compute_goal_stats(ckpt_tbl)[c.txt]
            summary[c.txt] = {
                "n": c.n,
                "mean": stats["mean"],
                "min": stats["min"],
                "max": stats["max"]
            }

        # Shortlist of 5 best rows seen so far
        top_rows = s01.compute_top_projects(ckpt_tbl, s01.compute_goal_stats(ckpt_tbl), top_n=5)
        checkpoints.append({
            "after": m,
            "summary": summary,
            "shortlist": top_rows
        })

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    # NASA-07
    print("== NASA-07 ==")
    for a in assessments:
        print(f"row: {a['row']} predicted_effort: {a['predicted_effort']:.4f} "
              f"predicted_defects: {a['predicted_defects']:.4f} alert: {str(a['alert']).lower()}")
    print(f"ANSWER NASA-07 {json.dumps({'assessments': assessments}, separators=(',', ':'))}")

    # NASA-08
    print("== NASA-08 ==")
    for ckpt in checkpoints:
        print(f"after: {ckpt['after']} summary: {json.dumps(ckpt['summary'], separators=(',', ':'))}"
              f" shortlist: {json.dumps(ckpt['shortlist'])}")
    print(f"ANSWER NASA-08 {json.dumps({'checkpoints': checkpoints}, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_04.py <csv_path>")
    main(sys.argv[1])
