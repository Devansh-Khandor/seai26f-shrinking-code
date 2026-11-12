#!/usr/bin/env python3
"""
Skill 07: Identify promising project combinations and recommend proposals for evaluation.

NASA-13: Find historically successful characteristic combinations that produced
          higher software volume with lower effort and fewer defects.
NASA-14: Rank project proposals by predicted efficiency and risk to guide a
          limited evaluation budget.

Usage:
    python skill_07.py <path_to_nasa93dem.csv>
"""

import csv
import sys
import math
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------

def read_csv(csv_path):
    """
    Parse the NASA data set.

    Returns two lists of dictionaries:
    - completed: rows with Kloc+, Effort-, Defects- (all present)
    - proposals: rows with Kloc+ but missing Effort- or Defects-
    Each dictionary contains:
        * id (str) – project id or empty string
        * Kloc (float)
        * Effort (float, optional)
        * Defects (float, optional)
        * all other fields as strings (categorical)
    """
    completed = []
    proposals = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = float(row.get("Kloc+", "").strip())
            except (ValueError, TypeError):
                continue
            if k <= 0:
                continue

            id_ = row.get("idX", "").strip()
            eff_str = row.get("Effort-", "").strip()
            def_str = row.get("Defects-", "").strip()

            # All other columns (categorical)
            cat = {k: v.strip() for k, v in row.items()
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

def numeric_columns(rows):
    """Return list of column names that are numeric (floatable) across all rows."""
    if not rows:
        return []

    # Known numeric columns we care about
    base = {"Kloc", "Effort", "Defects"}
    # Add any other columns that can be parsed as float
    for key in list(rows[0].keys()):
        if key in base:
            continue
        try:
            float(rows[0][key])
            base.add(key)
        except (ValueError, TypeError):
            continue
    return list(base)

# ---------------------------------------------------------------------------

def categorical_columns(rows, numeric_cols):
    """Return list of columns that are categorical (non-numeric)."""
    if not rows:
        return []

    all_keys = set(rows[0].keys())
    return sorted(all_keys - set(numeric_cols) - {"id"})

# ---------------------------------------------------------------------------

def overall_densities(rows):
    """Return overall effort density and defect density (per Kloc)."""
    if not rows:
        return None, None

    total_kloc = sum(r["Kloc"] for r in rows)
    total_effort = sum(r["Effort"] for r in rows)
    total_defects = sum(r["Defects"] for r in rows)
    if total_kloc == 0:
        return None, None
    return total_effort / total_kloc, total_defects / total_kloc

# ---------------------------------------------------------------------------

def compute_combination_stats(rows, cat1, cat2):
    """Group rows by (cat1, cat2) and compute densities per group."""
    groups = defaultdict(list)
    for r in rows:
        key = (r.get(cat1, ""), r.get(cat2, ""))
        groups[key].append(r)

    stats = {}
    for key, recs in groups.items():
        kloc_sum = sum(r["Kloc"] for r in recs)
        eff_sum = sum(r["Effort"] for r in recs)
        def_sum = sum(r["Defects"] for r in recs)
        if kloc_sum > 0:
            stats[key] = {
                "Kloc": kloc_sum,
                "Effort": eff_sum,
                "Defects": def_sum,
                "EffDensity": eff_sum / kloc_sum,
                "DefDensity": def_sum / kloc_sum,
                "Count": len(recs),
            }
    return stats

# ---------------------------------------------------------------------------

def top_combinations(stats, overall_eff, overall_def, top_n=3):
    """Return best combinations sorted by improvement over overall densities."""
    results = []
    for key, s in stats.items():
        # Only consider groups with densities lower than overall
        if s["EffDensity"] <= overall_eff and s["DefDensity"] <= overall_def:
            # Improvement score: product of ratios (higher is better)
            eff_ratio = overall_eff / s["EffDensity"] if s["EffDensity"] else 0
            def_ratio = overall_def / s["DefDensity"] if s["DefDensity"] else 0
            score = eff_ratio * def_ratio
            results.append((score, key, s))
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top_n]

# ---------------------------------------------------------------------------

def build_linear_model(rows, feature_name, target_name):
    """Fit linear regression of target on feature (1D)."""
    X = np.array([r[feature_name] for r in rows]).reshape(-1, 1)
    y = np.array([r[target_name] for r in rows])
    model = LinearRegression()
    model.fit(X, y)
    return model

# ---------------------------------------------------------------------------

def predict_densities(proposals, eff_model, def_model, overall_eff, overall_def):
    """Return list of proposals enriched with predicted densities."""
    enriched = []
    for p in proposals:
        k = p["Kloc"]
        eff_pred = eff_model.predict(np.array([[k]]))[0]
        def_pred = def_model.predict(np.array([[k]]))[0]
        eff_dens = eff_pred / k
        def_dens = def_pred / k
        delta = (eff_dens - overall_eff) + (def_dens - overall_def)
        enriched.append(
            {
                "id": p["id"] or f"PROPOSAL-{p['Kloc']}",
                "Kloc": k,
                "Effort": eff_pred,
                "Defects": def_pred,
                "EffDensity": eff_dens,
                "DefDensity": def_dens,
                "Delta": delta,
            }
        )
    return enriched

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    completed, proposals = read_csv(csv_path)

    if not completed:
        print("No completed project data available.")
        sys.exit(1)

    # --------------------------------------------------------------------
    # NASA-13
    print("== NASA-13 ==")

    # Identify categorical columns
    numeric_cols = numeric_columns(completed)
    cat_cols = categorical_columns(completed, numeric_cols)

    if len(cat_cols) < 2:
        print("Insufficient categorical data for combination analysis.")
    else:
        # Pick top 2 categorical columns by number of distinct values
        col_counts = {c: len({r.get(c, "") for r in completed}) for c in cat_cols}
        top_two = sorted(col_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        cat1, cat2 = top_two[0][0], top_two[1][0]

        overall_eff, overall_def = overall_densities(completed)
        combo_stats = compute_combination_stats(completed, cat1, cat2)
        best_combos = top_combinations(combo_stats, overall_eff, overall_def, top_n=3)

        if not best_combos:
            print("No combination found that improves on overall effort/defect densities.")
        else:
            print(f"Selected categorical columns for analysis: {cat1}, {cat2}")
            print("Top promising combinations:")
            for score, key, s in best_combos:
                cat_vals = f"{cat1}={key[0]}, {cat2}={key[1]}"
                print(f"  {cat_vals}")
                print(f"    Projects: {s['Count']}")
                print(f"    Avg Kloc: {s['Kloc']:.1f}")
                print(f"    Effort density: {s['EffDensity']:.4f}")
                print(f"    Defect density: {s['DefDensity']:.4f}")

    # --------------------------------------------------------------------
    # NASA-14
    print("\n== NASA-14 ==")

    if not proposals:
        print("No proposals to evaluate.")
        sys.exit(0)

    # Build models from completed data
    eff_model = build_linear_model(completed, "Kloc", "Effort")
    def_model = build_linear_model(completed, "Kloc", "Defects")
    overall_eff, overall_def = overall_densities(completed)

    enriched = predict_densities(proposals, eff_model, def_model, overall_eff, overall_def)

    # Sort proposals by Delta ascending (most negative is best improvement)
    enriched.sort(key=lambda x: x["Delta"])

    top_budget = min(5, len(enriched))
    if top_budget == 0:
        print("No proposals with evaluable predictions.")
    else:
        print(f"Top {top_budget} proposals to evaluate first (most promising improvements):")
        for i in range(top_budget):
            p = enriched[i]
            print(f"  Proposal ID: {p['id']}")
            print(f"    Kloc: {p['Kloc']:.1f}")
            print(f"    Predicted Effort: {p['Effort']:.2f} man-months")
            print(f"    Predicted Defects: {p['Defects']:.2f}")
            print(f"    Effort density: {p['EffDensity']:.4f}")
            print(f"    Defect density: {p['DefDensity']:.4f}")
            print(f"    Improvement score (Δ): {p['Delta']:.4f}")

if __name__ == "__main__":
    main()
