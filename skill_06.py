#!/usr/bin/env python3
"""
Skill 06: Explain effort estimates and compare risk between proposals.

NASA-11: For each proposed project, explain the effort estimate by referencing
the three most similar completed projects and their recorded characteristics.

NASA-12: For any two proposals with different risk assessments, explain the
main differences that contributed to those assessments and identify the
important trade‑offs.

Usage:
    python skill_06.py <path_to_nasa93dem.csv>
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
    Read the NASA data set.
    Returns:
        completed: list of dicts with id, Kloc, Effort, Defects
        proposals: list of dicts with id, Kloc, and all other fields
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
            except ValueError:
                continue
            if k <= 0:
                continue

            id_ = clean.get("idX", "")
            # check if effort and defects exist
            eff_str = clean.get("Effort-", "")
            def_str = clean.get("Defects-", "")

            if eff_str and def_str:
                try:
                    eff = float(eff_str)
                    defs = float(def_str)
                except ValueError:
                    continue
                completed.append({"id": id_, "Kloc": k, "Effort": eff, "Defects": defs})
            else:
                # proposal – keep all fields for later comparison
                proposals.append({"id": id_, "Kloc": k, **clean})
    return completed, proposals

# ---------------------------------------------------------------------------

def fit_linear(x, y):
    """Fit a simple linear regression y = a*x + b."""
    X = np.array(x).reshape(-1, 1)
    Y = np.array(y)
    model = LinearRegression()
    model.fit(X, Y)
    return model

# ---------------------------------------------------------------------------

def compute_stats(values):
    """Return mean and std of a numeric list (population)."""
    if not values:
        return None, None
    arr = np.array(values)
    return np.mean(arr), np.std(arr, ddof=0)

# ---------------------------------------------------------------------------

def predict_and_risk(completed, proposals):
    """
    For each proposal compute effort/defect predictions and risk classification.
    Returns:
        predictions: list of dicts with proposal id, predicted Effort/Defects, risk string
    """
    # Build training sets
    k_values = [p["Kloc"] for p in completed]
    eff_values = [p["Effort"] for p in completed]
    def_values = [p["Defects"] for p in completed]

    eff_model = fit_linear(k_values, eff_values)
    def_model = fit_linear(k_values, def_values)

    # Train‑set predictions to build thresholds
    eff_train_pred = eff_model.predict(np.array(k_values).reshape(-1, 1))
    def_train_pred = def_model.predict(np.array(k_values).reshape(-1, 1))

    eff_mean, eff_std = compute_stats(eff_train_pred)
    def_mean, def_std = compute_stats(def_train_pred)

    eff_thresh = eff_mean + 2 * eff_std if eff_std is not None else None
    def_thresh = def_mean + 2 * def_std if def_std is not None else None

    predictions = []
    for p in proposals:
        k = p["Kloc"]
        eff_pred = eff_model.predict([[k]])[0]
        def_pred = def_model.predict([[k]])[0]

        risk = []
        if eff_thresh is not None and eff_pred > eff_thresh:
            risk.append("High effort")
        if def_thresh is not None and def_pred > def_thresh:
            risk.append("High defects")
        risk_str = " & ".join(risk) if risk else "Low risk"

        predictions.append({
            "id": p["id"],
            "Kloc": k,
            "Effort": eff_pred,
            "Defects": def_pred,
            "risk": risk_str,
            "raw": p  # keep original for diff
        })
    return predictions, eff_thresh, def_thresh

# ---------------------------------------------------------------------------

def nearest_neighbors(completed, k_loc, n=3):
    """
    Find n completed projects nearest to k_loc (by absolute Kloc difference).
    """
    sorted_proj = sorted(completed, key=lambda p: abs(p["Kloc"] - k_loc))
    return sorted_proj[:n]

# ---------------------------------------------------------------------------

def numeric_features(proposal):
    """Return dict of numeric attributes (float) for a proposal."""
    attrs = {}
    for k, v in proposal.items():
        if k in ("id", "Kloc"):  # skip id and Kloc
            continue
        try:
            attrs[k] = float(v)
        except (ValueError, TypeError):
            pass
    return attrs

def compare_proposals(p1, p2):
    """Return list of top numeric differences between two proposals."""
    f1 = numeric_features(p1)
    f2 = numeric_features(p2)

    diffs = []
    for key in set(f1.keys()).intersection(f2.keys()):
        diff = abs(f1[key] - f2[key])
        diffs.append((key, diff, f1[key], f2[key]))
    diffs.sort(key=lambda x: x[1], reverse=True)
    return diffs[:2]  # top two differences

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    completed, proposals = read_csv(csv_path)

    if not completed:
        print("No completed projects found – cannot build models.")
        sys.exit(1)

    if not proposals:
        print("No proposals to analyze.")
        sys.exit(1)

    predictions, eff_thresh, def_thresh = predict_and_risk(completed, proposals)

    # --------------------------------------------------------------------
    # NASA-11: Explain effort estimates
    print("== NASA-11 ==")
    for pred in predictions[:3]:  # explain first three proposals
        print(f"\nProposal ID: {pred['id']}")
        print(f"  Predicted effort: {format_number(pred['Effort'])} man-months")
        print(f"  Predicted defects: {format_number(pred['Defects'])}")
        similar = nearest_neighbors(completed, pred['Kloc'], n=3)
        print("  3 most similar completed projects:")
        for s in similar:
            print(f"    ID {s['id']}: Kloc {format_number(s['Kloc'])}, "
                  f"Effort {format_number(s['Effort'])}, "
                  f"Defects {format_number(s['Defects'])}")

    # --------------------------------------------------------------------
    # NASA-12: Compare risk between two different proposals
    print("\n== NASA-12 ==")
    # Find first pair with differing risk
    pair = None
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            if predictions[i]["risk"] != predictions[j]["risk"]:
                pair = (predictions[i], predictions[j])
                break
        if pair:
            break

    if not pair:
        print("  No two proposals with differing risk assessments found.")
    else:
        p1, p2 = pair
        print(f"\nProposal {p1['id']} risk: {p1['risk']}")
        print(f"Proposal {p2['id']} risk: {p2['risk']}")
        diffs = compare_proposals(p1["raw"], p2["raw"])
        print("\n  Key differences contributing to risk disparity:")
        for key, diff, v1, v2 in diffs:
            print(f"    {key} : {format_number(v1)} vs {format_number(v2)} "
                  f"(Δ {format_number(diff)})")
        # Simple trade‑off statement
        if "Kloc" in diffs[0]:
            print("\n  Trade‑off: Higher Kloc generally increases effort and defect risk.")
        else:
            print("\n  Trade‑off: Feature differences (e.g., project size, complexity) "
                  "drive risk assessments.")

# ---------------------------------------------------------------------------

def format_number(val, decimals=2):
    if math.isnan(val) or math.isinf(val):
        return "N/A"
    return f"{val:.{decimals}f}".rstrip('0').rstrip('.')

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
