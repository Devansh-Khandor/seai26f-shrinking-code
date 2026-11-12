#!/usr/bin/env python3
"""
Skill 08: Guidance on adjusting proposals to reduce effort or defects.

NASA-15: For a proposal predicted to require excessive effort, suggest the
plausible changes to adjustable planning or process characteristics that
might reduce its predicted effort.

NASA-16: For a proposal with a high predicted defect count, identify the
smallest plausible combination of changes to adjustable characteristics that
could improve its predicted quality, and explain the trade‑offs.

Usage:
    python skill_08.py <path_to_nasa93dem.csv>
"""

import csv
import sys
import math
import itertools
from collections import defaultdict

from sklearn.linear_model import LinearRegression
import numpy as np

# ---------------------------------------------------------------------------

def read_csv(csv_path):
    """
    Parse the NASA data set.
    Returns two lists of dictionaries:
    - completed: projects with Kloc+, Effort-, Defects- (all present)
    - proposals: projects with Kloc+ but missing Effort- or Defects-
    """
    completed = []
    proposals = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # strip all values
            clean = {k: v.strip() for k, v in row.items()}
            try:
                k = float(clean.get("Kloc+", ""))
            except (ValueError, TypeError):
                continue
            if k <= 0:
                continue

            id_ = clean.get("idX", "")
            eff_str = clean.get("Effort-", "")
            def_str = clean.get("Defects-", "")

            # all other columns (categorical)
            cat = {k: v for k, v in clean.items()
                   if k not in ("Kloc+", "Effort-", "Defects-", "idX")}

            if eff_str and def_str:
                try:
                    eff = float(eff_str)
                    defs = float(def_str)
                except ValueError:
                    continue
                completed.append(
                    {"id": id_, "Kloc": k, "Effort": eff, "Defects": defs, **cat}
                )
            else:
                proposals.append(
                    {"id": id_, "Kloc": k, "Effort": None, "Defects": None, **cat}
                )
    return completed, proposals

# ---------------------------------------------------------------------------

def categorical_columns(all_rows):
    """
    Return list of column names that are categorical
    (i.e. not Kloc, Effort, Defects or idX).
    """
    if not all_rows:
        return []

    cols = set()
    for r in all_rows:
        cols.update(r.keys())
    # Remove known numeric columns
    cols -= {"Kloc", "Effort", "Defects", "id", "idX", "Kloc+", "Effort-", "Defects-"}
    # Sort for deterministic order
    return sorted(cols)

# ---------------------------------------------------------------------------

def build_category_maps(all_rows, cat_cols):
    """
    Build mapping from category string -> numeric index for each categorical column.
    Also return reverse mapping (index -> category) and minimal category (index 0).
    """
    maps = {}
    reverse_maps = {}
    min_cat = {}
    for col in cat_cols:
        # Gather all observed values
        values = set(r.get(col, "") for r in all_rows)
        sorted_vals = sorted(values)  # deterministic
        cat_to_idx = {v: i for i, v in enumerate(sorted_vals)}
        idx_to_cat = {i: v for v, i in cat_to_idx.items()}
        maps[col] = cat_to_idx
        reverse_maps[col] = idx_to_cat
        min_cat[col] = sorted_vals[0] if sorted_vals else ""
    return maps, reverse_maps, min_cat

# ---------------------------------------------------------------------------

def build_feature_matrix(rows, cat_maps, cat_cols):
    """
    Build X matrix for given rows.
    X = [Kloc, categorical indices ...]
    """
    X = []
    for r in rows:
        row = [r["Kloc"]]
        for col in cat_cols:
            val = r.get(col, "")
            idx = cat_maps[col].get(val, 0)  # default to 0 if unseen
            row.append(idx)
        X.append(row)
    return np.array(X, dtype=float)

# ---------------------------------------------------------------------------

def fit_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# ---------------------------------------------------------------------------

def predict(model, X):
    return model.predict(X)

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    completed, proposals = read_csv(csv_path)

    if not proposals:
        print("No proposal data to analyze.")
        sys.exit(0)

    # Combine all rows to gather categories
    all_rows = completed + proposals
    cat_cols = categorical_columns(all_rows)
    cat_maps, rev_maps, min_cat = build_category_maps(all_rows, cat_cols)

    # Build feature matrices
    X_comp = build_feature_matrix(completed, cat_maps, cat_cols)
    y_eff = np.array([r["Effort"] for r in completed], dtype=float)
    y_def = np.array([r["Defects"] for r in completed], dtype=float)

    # Fit models
    eff_model = fit_model(X_comp, y_eff)
    def_model = fit_model(X_comp, y_def)

    # Predict for proposals
    X_prop = build_feature_matrix(proposals, cat_maps, cat_cols)
    eff_pred = predict(eff_model, X_prop)
    def_pred = predict(def_model, X_prop)

    # Store predictions in proposal dict
    for p, eff, defe in zip(proposals, eff_pred, def_pred):
        p["_eff_pred"] = eff
        p["_def_pred"] = defe

    # Identify target proposals
    idx_max_eff = int(np.argmax(eff_pred))
    idx_max_def = int(np.argmax(def_pred))
    prop_eff = proposals[idx_max_eff]
    prop_def = proposals[idx_max_def]

    # --------------------------------------------------------------------
    # NASA-15: reduce effort
    print("== NASA-15 ==")
    print(f"Proposal with highest predicted effort: ID {prop_eff['id']} (predicted {prop_eff['_eff_pred']:.2f} man-months)")
    # Compute effect of changing each categorical variable to its minimal value
    effort_changes = []
    for col in cat_cols:
        orig = prop_eff.get(col, "")
        # Build new proposal copy
        new_vals = prop_eff.copy()
        new_vals[col] = min_cat[col]
        # Build feature vector
        row = [new_vals["Kloc"]]
        for c in cat_cols:
            v = new_vals.get(c, "")
            row.append(cat_maps[c].get(v, 0))
        new_X = np.array([row], dtype=float)
        new_eff = eff_model.predict(new_X)[0]
        delta = new_eff - prop_eff["_eff_pred"]
        effort_changes.append((col, orig, min_cat[col], delta, new_eff))
    # Sort by greatest reduction (most negative delta)
    effort_changes.sort(key=lambda x: x[3])
    # Show top 3
    for col, orig, new, delta, new_eff in effort_changes[:3]:
        print(f"  Change '{col}': {orig} → {new} reduces effort by {abs(delta):.2f} "
              f"to {new_eff:.2f} man-months")

    # --------------------------------------------------------------------
    # NASA-16: reduce defects with smallest changes
    print("\n== NASA-16 ==")
    print(f"Proposal with highest predicted defects: ID {prop_def['id']} (predicted {prop_def['_def_pred']:.2f})")
    # Compute single attribute changes
    defect_changes = []
    for col in cat_cols:
        orig = prop_def.get(col, "")
        new_vals = prop_def.copy()
        new_vals[col] = min_cat[col]
        row = [new_vals["Kloc"]]
        for c in cat_cols:
            v = new_vals.get(c, "")
            row.append(cat_maps[c].get(v, 0))
        new_X = np.array([row], dtype=float)
        new_def = def_model.predict(new_X)[0]
        delta = new_def - prop_def["_def_pred"]
        defect_changes.append((col, orig, min_cat[col], delta, new_def))
    # Evaluate pairwise combinations
    pair_changes = []
    for (c1, orig1, new1, d1, n1), (c2, orig2, new2, d2, n2) in itertools.combinations(defect_changes, 2):
        # Build new proposal with both changes
        new_vals = prop_def.copy()
        new_vals[c1] = new1
        new_vals[c2] = new2
        row = [new_vals["Kloc"]]
        for c in cat_cols:
            v = new_vals.get(c, "")
            row.append(cat_maps[c].get(v, 0))
        new_X = np.array([row], dtype=float)
        new_def = def_model.predict(new_X)[0]
        delta = new_def - prop_def["_def_pred"]
        pair_changes.append((c1, new1, c2, new2, delta, new_def))
    # Choose best single and best pair
    defect_changes.sort(key=lambda x: x[3])
    pair_changes.sort(key=lambda x: x[4])

    # Show best single change
    c1, orig1, new1, delta1, new_def1 = defect_changes[0]
    print(f"\nBest single attribute change:")
    print(f"  {c1}: {orig1} → {new1} reduces defects by {abs(delta1):.2f} to {new_def1:.2f}")

    # Show best pair change
    c1, new1, c2, new2, delta_pair, new_def_pair = pair_changes[0]
    print(f"\nBest pair of attribute changes:")
    print(f"  {c1}: {prop_def.get(c1, '')} → {new1}")
    print(f"  {c2}: {prop_def.get(c2, '')} → {new2}")
    print(f"  Reduces defects by {abs(delta_pair):.2f} to {new_def_pair:.2f}")

    # --------------------------------------------------------------------
    # Trade‑off analysis: effort impact of defect‑reducing changes
    # For best single change
    def_eff1_row = [prop_def["Kloc"]]
    for c in cat_cols:
        v = prop_def[c] if c != c1 else new1
        def_eff1_row.append(cat_maps[c].get(v, 0))
    new_eff1 = eff_model.predict(np.array([def_eff1_row], dtype=float))[0]
    eff_delta1 = new_eff1 - prop_def["_eff_pred"]
    print(f"\nEffect on effort of single change:")
    if eff_delta1 >= 0:
        print(f"  Effort increases by {eff_delta1:.2f} to {new_eff1:.2f}")
    else:
        print(f"  Effort decreases by {abs(eff_delta1):.2f} to {new_eff1:.2f}")

    # For best pair change
    def_eff2_row = [prop_def["Kloc"]]
    for c in cat_cols:
        if c == c1:
            v = new1
        elif c == c2:
            v = new2
        else:
            v = prop_def[c]
        def_eff2_row.append(cat_maps[c].get(v, 0))
    new_eff2 = eff_model.predict(np.array([def_eff2_row], dtype=float))[0]
    eff_delta2 = new_eff2 - prop_def["_eff_pred"]
    print(f"\nEffect on effort of pair changes:")
    if eff_delta2 >= 0:
        print(f"  Effort increases by {eff_delta2:.2f} to {new_eff2:.2f}")
    else:
        print(f"  Effort decreases by {abs(eff_delta2):.2f} to {new_eff2:.2f}")

if __name__ == "__main__":
    main()
