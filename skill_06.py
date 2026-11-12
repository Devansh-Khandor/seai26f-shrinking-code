#!/usr/bin/env python3
"""
skill_06.py

Implements NASA-11 and NASA-12 analyses on the NASA software project dataset.

NASA-11: Explain an effort estimate for a proposed project using similar historical projects.
NASA-12: Compare two proposals with different risk assessments and highlight key differences.

Usage:
    python skill_06.py <csv_path>
"""

import sys
import csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict

# ---------------------------------------------------------------------------

def parse_csv(csv_path: Path):
    """Read CSV and return rows and numeric column names."""
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        numeric_cols = []
        for col in reader.fieldnames:
            try:
                for r in rows:
                    val = r[col]
                    if val != '':
                        float(val.replace(',', ''))
                        numeric_cols.append(col)
                        break
            except (ValueError, TypeError):
                continue
    return rows, numeric_cols


def to_numpy(rows, cols):
    """Return (n_samples, n_features) array of floats, NaN for missing."""
    arr = []
    for r in rows:
        row = []
        for c in cols:
            val = r[c]
            if val == '':
                row.append(np.nan)
            else:
                try:
                    row.append(float(val.replace(',', '')))
                except ValueError:
                    row.append(np.nan)
        arr.append(row)
    return np.array(arr, dtype=float)


def train_models(X, y_eff, y_def):
    """Train linear regression models for effort and defects."""
    # Fit models
    model_eff = LinearRegression(fit_intercept=True)
    model_eff.fit(X, y_eff)
    model_def = LinearRegression(fit_intercept=True)
    model_def.fit(X, y_def)
    return model_eff, model_def


def predict_and_classify(model, X, y_pred, thresholds, feature_name):
    """Predict values and assign risk category based on thresholds."""
    preds = model.predict(X)
    # Compute percentiles thresholds
    lower, upper = thresholds
    # Category: 'Low' if <= lower, 'High' if >= upper, else 'Medium'
    categories = []
    for p in preds:
        if p <= lower:
            categories.append('Low')
        elif p >= upper:
            categories.append('High')
        else:
            categories.append('Medium')
    return preds, categories


def find_proposals(rows, numeric_cols, feature_cols):
    """Return proposals (rows with missing Effort- or Defects-) that have full features."""
    proposals = []
    for r in rows:
        if r['Effort-'] == '' or r['Defects-'] == '':
            # Check that all numeric features are present
            missing = False
            for c in feature_cols:
                if r[c] == '':
                    missing = True
                    break
            if not missing:
                proposals.append(r)
    return proposals


def find_k_nearest(proposal_vec, X_hist, ids_hist, k=3):
    """Return indices of k nearest historical projects (Euclidean)."""
    dists = np.linalg.norm(X_hist - proposal_vec, axis=1)
    idx = np.argsort(dists)[:k]
    return idx


def explain_estimate(proposal, model_eff, model_def, X_hist, y_eff_hist, y_def_hist, ids_hist, feature_cols):
    """Generate explanation for a single proposal."""
    # Build feature vector
    vec = np.array([float(proposal[c]) for c in feature_cols], dtype=float)

    # Predict effort with interval (approximate 95% PI)
    pred_eff, unc_eff = predict_with_interval(model_eff, X_hist, y_eff_hist, vec)
    pred_def, unc_def = predict_with_interval(model_def, X_hist, y_def_hist, vec)

    # Find k nearest historical projects
    idx = find_k_nearest(vec, X_hist, ids_hist, k=3)

    lines = []
    lines.append(f"== NASA-11 ==")
    lines.append(f"Proposal ID {proposal['idX']}:")
    lines.append(f"  Estimated effort: {pred_eff:.2f} ± {unc_eff:.2f} person‑months (95% PI)")
    lines.append(f"  Estimated defects: {pred_def:.2f} ± {unc_def:.2f} (95% PI)")
    lines.append("  Similar historical projects (by Euclidean distance):")
    for rank, i in enumerate(idx, start=1):
        lines.append(f"    {rank}) ID {ids_hist[i]} – Effort {y_eff_hist[i]:.2f}, Defects {y_def_hist[i]:.2f}")
    return "\n".join(lines)


def predict_with_interval(model, X_all, y_all, x0):
    """
    Compute prediction for x0 and approximate 95% prediction interval.
    Uses formula with inverse X^T X.
    """
    n, p = X_all.shape
    df = n - p - 1
    residuals = y_all - model.predict(X_all)
    s = np.sqrt(np.sum(residuals ** 2) / df)

    Xfull = np.hstack([np.ones((n, 1)), X_all])
    inv_xtx = np.linalg.pinv(Xfull.T @ Xfull)

    x0_full = np.hstack([1.0, x0])
    se_pred = s * np.sqrt(1 + x0_full @ inv_xtx @ x0_full.T)

    # 95% t critical approx 2
    t_crit = 2.0
    pred = model.predict(x0_full.reshape(1, -1))[0]
    return pred, t_crit * se_pred


def compare_proposals(proj_a, proj_b, feature_cols):
    """Generate comparison of two proposals with different risk."""
    lines = []
    lines.append("== NASA-12 ==")
    lines.append(f"Proposal {proj_a['idX']} risk: {proj_a['risk']}")
    lines.append(f"Proposal {proj_b['idX']} risk: {proj_b['risk']}")
    lines.append("Key differences in numeric features:")
    diff_list = []
    for c in feature_cols:
        val_a = float(proj_a[c])
        val_b = float(proj_b[c])
        diff = abs(val_a - val_b)
        diff_list.append((c, val_a, val_b, diff))
    # Take top 3 differences
    diff_list.sort(key=lambda t: t[3], reverse=True)
    top3 = diff_list[:3]
    for c, va, vb, d in top3:
        lines.append(f"  {c}: {va:.2f} vs {vb:.2f} (Δ={d:.2f})")
    # Trade‑offs explanation
    lines.append("Trade‑offs:")
    # Simplistic rule: higher values in features that typically increase effort
    effort_related = ['Kloc+', 'centerX', 'YearX', 'time', 'acap', 'pcap', 'apex', 'plex', 'ltex', 'tool', 'site', 'sced']
    # We only use numeric features present in the data
    effort_increase = []
    for c, va, vb, d in top3:
        if c in effort_related:
            if va > vb:
                effort_increase.append((c, va, vb))
            else:
                effort_increase.append((c, vb, va))
    if effort_increase:
        for c, higher, lower in effort_increase:
            lines.append(f"  Feature '{c}' is higher in the higher‑risk proposal, contributing to increased effort.")
    else:
        lines.append("  No clear effort‑driving feature differences detected.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: {csv_path} does not exist or is not a file.")
        sys.exit(1)

    rows, numeric_cols = parse_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    # Identify required columns
    eff_col = 'Effort-'
    defect_col = 'Defects-'
    if eff_col not in numeric_cols or defect_col not in numeric_cols:
        print("Required numeric columns 'Effort-' and 'Defects-' not found.")
        sys.exit(1)

    # Feature columns: all numeric except the two targets
    feature_cols = [c for c in numeric_cols if c not in (eff_col, defect_col)]
    if not feature_cols:
        print("No feature columns available for analysis.")
        sys.exit(1)

    # Build data matrix for completed projects
    completed = [r for r in rows if r[eff_col] != '' and r[defect_col] != '']
    if not completed:
        print("No completed projects available for modeling.")
        sys.exit(1)

    X_all = to_numpy(completed, feature_cols)
    y_eff = to_numpy(completed, [eff_col]).flatten()
    y_def = to_numpy(completed, [defect_col]).flatten()
    ids_all = [r['idX'] for r in completed]

    # Train models
    model_eff, model_def = train_models(X_all, y_eff, y_def)

    # Predict on completed to get thresholds
    eff_preds, eff_cats = predict_and_classify(
        model_eff, X_all, y_eff, (np.percentile(y_eff, 25), np.percentile(y_eff, 75)), 'Effort')
    def_preds, def_cats = predict_and_classify(
        model_def, X_all, y_def, (np.percentile(y_def, 25), np.percentile(y_def, 75)), 'Defects')

    # Assign risk for completed (for threshold use only)
    # Not needed for proposals

    # Find proposals
    proposals = find_proposals(rows, numeric_cols, feature_cols)

    # If no proposals, nothing to explain
    if not proposals:
        print("== NASA-11 ==")
        print("No proposals with missing effort/defects found.")
        print()
        print("== NASA-12 ==")
        print("No proposals available for risk comparison.")
        sys.exit(0)

    # NASA-11: explain first proposal (or all)
    # We'll explain the first proposal for brevity
    proposal = proposals[0]
    print(explain_estimate(proposal, model_eff, model_def, X_all, y_eff, y_def, ids_all, feature_cols))
    print()

    # NASA-12: find two proposals with different risk categories
    # Compute risk for proposals using same thresholds
    def get_risk(pred, low, high):
        if pred <= low:
            return 'Low'
        elif pred >= high:
            return 'High'
        else:
            return 'Medium'

    eff_low, eff_high = np.percentile(y_eff, 25), np.percentile(y_eff, 75)
    def_low, def_high = np.percentile(y_def, 25), np.percentile(y_def, 75)

    # Assign risk to proposals
    for p in proposals:
        vec = np.array([float(p[c]) for c in feature_cols], dtype=float)
        eff_pred, _ = predict_with_interval(model_eff, X_all, y_eff, vec)
        def_pred, _ = predict_with_interval(model_def, X_all, y_def, vec)
        p['eff_risk'] = get_risk(eff_pred, eff_low, eff_high)
        p['def_risk'] = get_risk(def_pred, def_low, def_high)
        # Overall risk: consider high if either aspect high, low if both low, else medium
        if p['eff_risk'] == 'High' or p['def_risk'] == 'High':
            p['risk'] = 'High'
        elif p['eff_risk'] == 'Low' and p['def_risk'] == 'Low':
            p['risk'] = 'Low'
        else:
            p['risk'] = 'Medium'

    # Find a pair with different overall risk
    pair = None
    for i in range(len(proposals)):
        for j in range(i+1, len(proposals)):
            if proposals[i]['risk'] != proposals[j]['risk']:
                pair = (proposals[i], proposals[j])
                break
        if pair:
            break

    if pair:
        proj_a, proj_b = pair
        print(compare_proposals(proj_a, proj_b, feature_cols))
    else:
        # Fallback: compare first two proposals
        print("== NASA-12 ==")
        print("Only proposals with identical risk levels found; comparing first two.")
        print(compare_proposals(proposals[0], proposals[1], feature_cols))


if __name__ == "__main__":
    main()
