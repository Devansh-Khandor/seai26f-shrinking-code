#!/usr/bin/env python3
"""
skill_08.py – NASA-15 & NASA-16

This script analyses a CSV file containing software project data and
produces two reports:

* NASA-15 – For a project proposal predicted to require excessive effort,
  identify plausible changes to adjustable planning or process characteristics
  that might reduce its predicted effort.

* NASA-16 – For a project proposal with a high predicted defect count,
  identify the smallest plausible combination of changes to adjustable
  planning or process characteristics that could improve its predicted quality,
  while explaining potential tradeoffs.

The output format is fixed: for each NASA ID the script prints a header line
"== <ID> ==" followed by at least three human‑readable lines and a machine‑checked
JSON answer line.  The JSON objects are derived purely from the supplied CSV file
and are not hard‑coded.

The script accepts a single command‑line argument: the path to the CSV file.
"""

import csv
import json
import math
import sys

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(val):
    if val is None:
        return None
    v = val.strip()
    if v == "" or v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def linear_regression(x_vals, y_vals):
    """Least‑squares linear regression: returns slope, intercept."""
    n = len(x_vals)
    if n == 0:
        return 0.0, 0.0
    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
    var_x = sum((x - mean_x) ** 2 for x in x_vals)
    if var_x == 0.0:
        return 0.0, mean_y
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def is_numeric(series):
    """Return True if all non‑None, non‑? values can be parsed to float."""
    for v in series:
        if v is None or v == "?":
            continue
        try:
            float(v)
        except ValueError:
            return False
    return True


def get_adjustable_columns(header):
    """Non‑goal, not Kloc+, not ending with X."""
    return [
        c for c in header
        if not c.endswith(("+", "-")) and not c.endswith("X") and c != "Kloc+"
    ]


def build_regressions(train_rows, target_col, all_columns):
    """Return dict of regressions per column for target_col."""
    regressions = {}
    for col in all_columns:
        # collect X (column value) and Y (target_col value)
        x_vals = []
        y_vals = []
        for r in train_rows:
            xv = r.get(col)
            yv = r.get(target_col)
            if xv is None or yv is None:
                continue
            if xv.strip() == "?" or yv.strip() == "?":
                continue
            # For numeric columns
            try:
                x = float(xv)
                y = float(yv)
                x_vals.append(x)
                y_vals.append(y)
            except ValueError:
                # categorical
                x_vals.append(xv.strip())
                y_vals.append(float(yv))
        if not x_vals:
            continue
        # Decide numeric vs categorical
        if all(isinstance(x, float) for x in x_vals):
            # numeric
            slope, intercept = linear_regression(x_vals, y_vals)
            regressions[col] = {
                "type": "numeric",
                "slope": slope,
                "intercept": intercept,
                "values": sorted(set(x_vals)),
            }
        else:
            # categorical
            cat_stats = {}
            for x, y in zip(x_vals, y_vals):
                cat_stats.setdefault(x, []).append(y)
            means = {c: sum(vals) / len(vals) for c, vals in cat_stats.items()}
            regressions[col] = {
                "type": "categorical",
                "means": means,
                "categories": sorted(cat_stats.keys()),
            }
    return regressions


def predict_value(reg, val):
    """Return predicted value using regression object and given column value."""
    if reg["type"] == "numeric":
        return reg["slope"] * val + reg["intercept"]
    else:
        return reg["means"].get(val, None)


def find_best_value(reg):
    """Return value that yields minimal predicted value."""
    if reg["type"] == "numeric":
        # evaluate all distinct values
        best_val = None
        best_pred = None
        for v in reg["values"]:
            pred = reg["slope"] * v + reg["intercept"]
            if best_pred is None or pred < best_pred:
                best_pred = pred
                best_val = v
        return best_val, best_pred
    else:
        # categorical: choose category with smallest mean
        best_cat = None
        best_mean = None
        for c, m in reg["means"].items():
            if best_mean is None or m < best_mean:
                best_mean = m
                best_cat = c
        return best_cat, best_mean


def round_num(v):
    """Round to two decimals if v is not None."""
    return None if v is None else round(v, 2)


# ----------------------------------------------------------------------
# Arrival predictions
# ----------------------------------------------------------------------
def predict_arrivals(rows, history_rows, arrivals):
    """Return lists of predicted effort and defects for each arrival."""
    pred_eff = []
    pred_def = []
    for i, arr_row in enumerate(arrivals):
        train_rows = history_rows + arrivals[:i]
        # regression Kloc+ vs Effort-
        slope_eff, intercept_eff = linear_regression(
            [to_float(r["Kloc+"]) for r in train_rows if to_float(r["Kloc+"]) is not None],
            [to_float(r["Effort-"]) for r in train_rows if to_float(r["Effort-"]) is not None],
        )
        # regression Kloc+ vs Defects-
        slope_def, intercept_def = linear_regression(
            [to_float(r["Kloc+"]) for r in train_rows if to_float(r["Kloc+"]) is not None],
            [to_float(r["Defects-"]) for r in train_rows if to_float(r["Defects-"]) is not None],
        )
        kloc = to_float(arr_row["Kloc+"])
        if kloc is None:
            # impute with mean Kloc+ from training data
            kloc_vals = [to_float(r["Kloc+"]) for r in train_rows]
            kloc_vals = [v for v in kloc_vals if v is not None]
            kloc = sum(kloc_vals) / len(kloc_vals) if kloc_vals else 0.0
        pred_eff.append(slope_eff * kloc + intercept_eff)
        pred_def.append(slope_def * kloc + intercept_def)
    return pred_eff, pred_def


# ----------------------------------------------------------------------
# NASA-15
# ----------------------------------------------------------------------
def nasa15(rows, history_rows, arrivals):
    """Find arrival with highest predicted effort and suggest one change."""
    pred_eff, _ = predict_arrivals(rows, history_rows, arrivals)
    idx = max(range(len(pred_eff)), key=lambda i: pred_eff[i])
    row_no = len(history_rows) + idx + 1
    target_row = arrivals[idx]

    # Build regressions for each adjustable column against Effort-
    header = rows[0].keys()
    adj_cols = get_adjustable_columns(header)
    regressions = build_regressions(history_rows + arrivals[:idx], "Effort-", adj_cols)

    best_col = None
    best_before = None
    best_after = None
    best_improvement = -math.inf

    for col, reg in regressions.items():
        cur_val_raw = target_row.get(col)
        if cur_val_raw is None:
            continue
        cur_val_str = cur_val_raw.strip()
        if cur_val_str == "?":
            continue
        if reg["type"] == "numeric":
            try:
                cur_val = float(cur_val_str)
            except ValueError:
                continue
            pred_before = predict_value(reg, cur_val)
            best_val, pred_after = find_best_value(reg)
            improvement = pred_before - pred_after
        else:
            cur_val = cur_val_str
            pred_before = predict_value(reg, cur_val)
            best_val, pred_after = find_best_value(reg)
            improvement = pred_before - pred_after
        if improvement > best_improvement:
            best_improvement = improvement
            best_col = col
            best_before = pred_before
            best_after = pred_after

    # Prepare output
    before_eff = round_num(best_before)
    after_eff = round_num(best_after)
    changes = {best_col: [target_row[best_col], str(best_val)]}
    answer = {
        "project": row_no,
        "changes": changes,
        "predicted_effort": {"before": before_eff, "after": after_eff},
    }

    # Human‑readable
    print(f"Target row: {row_no} predicted effort before {before_eff}, after {after_eff}")
    print(f"Suggested change: {best_col} from {target_row[best_col]} to {best_val}")
    print(f"Effect: effort reduction from {before_eff} to {after_eff}")

    print("ANSWER NASA-15", json.dumps(answer, separators=(",", ":")))
    return


# ----------------------------------------------------------------------
# NASA-16
# ----------------------------------------------------------------------
def nasa16(rows, history_rows, arrivals):
    """Find arrival with highest predicted defects and suggest one change."""
    _, pred_def = predict_arrivals(rows, history_rows, arrivals)
    idx = max(range(len(pred_def)), key=lambda i: pred_def[i])
    row_no = len(history_rows) + idx + 1
    target_row = arrivals[idx]

    header = rows[0].keys()
    adj_cols = get_adjustable_columns(header)

    # Build regressions against Defects- and Effort-
    regress_def = build_regressions(history_rows + arrivals[:idx], "Defects-", adj_cols)
    regress_eff = build_regressions(history_rows + arrivals[:idx], "Effort-", adj_cols)

    best_col = None
    best_before_def = None
    best_after_def = None
    best_improvement_def = -math.inf
    best_eff_before = None
    best_eff_after = None

    for col in adj_cols:
        if col not in regress_def or col not in regress_eff:
            continue
        cur_val_raw = target_row.get(col)
        if cur_val_raw is None:
            continue
        cur_val_str = cur_val_raw.strip()
        if cur_val_str == "?":
            continue

        reg_def = regress_def[col]
        reg_eff = regress_eff[col]

        if reg_def["type"] == "numeric":
            try:
                cur_val = float(cur_val_str)
            except ValueError:
                continue
            pred_def_before = predict_value(reg_def, cur_val)
            best_val, pred_def_after = find_best_value(reg_def)
        else:
            cur_val = cur_val_str
            pred_def_before = predict_value(reg_def, cur_val)
            best_val, pred_def_after = find_best_value(reg_def)

        # Effort predictions
        if reg_eff["type"] == "numeric":
            try:
                cur_val_eff = float(cur_val_str)
            except ValueError:
                continue
            pred_eff_before = predict_value(reg_eff, cur_val_eff)
            _, pred_eff_after = find_best_value(reg_eff)
        else:
            cur_val_eff = cur_val_str
            pred_eff_before = predict_value(reg_eff, cur_val_eff)
            _, pred_eff_after = find_best_value(reg_eff)

        improvement_def = pred_def_before - pred_def_after
        if improvement_def > best_improvement_def:
            best_improvement_def = improvement_def
            best_col = col
            best_before_def = pred_def_before
            best_after_def = pred_def_after
            best_eff_before = pred_eff_before
            best_eff_after = pred_eff_after

    before_def = round_num(best_before_def)
    after_def = round_num(best_after_def)
    before_eff = round_num(best_eff_before)
    after_eff = round_num(best_eff_after)
    changes = {best_col: [target_row[best_col], str(best_val)]}
    answer = {
        "project": row_no,
        "changes": changes,
        "predicted_defects": {"before": before_def, "after": after_def},
        "predicted_effort": {"before": before_eff, "after": after_eff},
    }

    # Human‑readable
    print(f"Target row: {row_no} predicted defects before {before_def}, after {after_def}")
    print(f"Suggested change: {best_col} from {target_row[best_col]} to {best_val}")
    print(f"Effect: defects reduction from {before_def} to {after_def}")
    print(f"Effort trade‑off: before {before_eff}, after {after_eff}")

    print("ANSWER NASA-16", json.dumps(answer, separators=(",", ":")))
    return


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        sys.exit(1)

    n_rows = len(rows)
    arrivals_count = round(0.2 * n_rows)
    history_rows = rows[: n_rows - arrivals_count]
    arrivals = rows[n_rows - arrivals_count :]

    print("== NASA-15 ==")
    nasa15(rows, history_rows, arrivals)

    print("\n== NASA-16 ==")
    nasa16(rows, history_rows, arrivals)


if __name__ == "__main__":
    main()
