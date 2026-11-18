#!/usr/bin/env python3
import sys
import csv
import json
import math

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(row)
    return rows, reader.fieldnames

def to_float(val):
    if val is None or val == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def compute_statistics(rows, goal_col, settings):
    # Compute means of goal and each setting
    goal_vals = []
    setting_means = {}
    setting_counts = {}
    covariances = {}
    for s in settings:
        covariances[s] = []
        setting_means[s] = 0.0
        setting_counts[s] = 0

    for row in rows:
        g = to_float(row[goal_col])
        if g is None:
            continue
        goal_vals.append(g)

        for s in settings:
            v = to_float(row[s])
            if v is not None:
                setting_means[s] += v
                setting_counts[s] += 1

    # Finalize means
    for s in settings:
        if setting_counts[s]:
            setting_means[s] /= setting_counts[s]
        else:
            setting_means[s] = 0.0

    if not goal_vals:
        mean_goal = 0.0
    else:
        mean_goal = sum(goal_vals) / len(goal_vals)

    # Compute variance and covariance for each setting
    var_s = {}
    cov_s = {}
    for s in settings:
        var = 0.0
        cov = 0.0
        count = 0
        for row in rows:
            g = to_float(row[goal_col])
            v = to_float(row[s])
            if g is None or v is None:
                continue
            var += (v - setting_means[s]) ** 2
            cov += (v - setting_means[s]) * (g - mean_goal)
            count += 1
        var_s[s] = var / count if count else 0.0
        cov_s[s] = cov / count if count else 0.0

    # Compute regression coefficients
    beta = {}
    for s in settings:
        if var_s[s] == 0:
            beta[s] = 0.0
        else:
            beta[s] = cov_s[s] / var_s[s]
    # Intercept
    intercept = mean_goal
    for s in settings:
        intercept -= beta[s] * setting_means[s]
    return intercept, beta, setting_means, mean_goal

def predict_row(row, goal_col, settings, intercept, beta, setting_means):
    pred = intercept
    for s in settings:
        v = to_float(row[s])
        if v is None:
            v = setting_means[s]
        pred += beta[s] * v
    return pred

def compute_mae(predictions, actuals):
    n = len(predictions)
    if n == 0:
        return None
    total = sum(abs(p - a) for p, a in zip(predictions, actuals))
    return total / n

def main(csv_path):
    rows, columns = read_csv(csv_path)

    goal_cols = [c for c in columns if c.endswith('+') or c.endswith('-')]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    goal_col = goal_cols[0]

    settings = [c for c in columns if not (c.endswith('+') or c.endswith('-') or c.endswith('X'))]

    intercept, beta, setting_means, mean_goal = compute_statistics(rows, goal_col, settings)

    # Predict on held-out rows (row number divisible by 5)
    predictions_dict = {}
    heldout_actuals = []
    heldout_preds = []
    for idx, row in enumerate(rows, 1):
        if idx % 5 == 0:
            pred = predict_row(row, goal_col, settings, intercept, beta, setting_means)
            predictions_dict[str(idx)] = round(pred, 4)
            actual = to_float(row[goal_col])
            if actual is not None:
                heldout_actuals.append(actual)
                heldout_preds.append(pred)

    mae = compute_mae(heldout_preds, heldout_actuals)
    if mae is not None:
        mae = round(mae, 4)
    else:
        mae = None

    # Compute residuals and std dev
    residuals = []
    for idx, row in enumerate(rows, 1):
        g = to_float(row[goal_col])
        if g is None:
            continue
        pred = predict_row(row, goal_col, settings, intercept, beta, setting_means)
        residuals.append(g - pred)
    if residuals:
        mean_res = sum(residuals) / len(residuals)
        std_res = math.sqrt(sum((r - mean_res) ** 2 for r in residuals) / len(residuals))
    else:
        std_res = 0.0

    # Find pairs differing in exactly one setting
    pairs = []
    nrows = len(rows)
    for i in range(nrows):
        row_i = rows[i]
        for j in range(i+1, nrows):
            row_j = rows[j]
            diff_settings = []
            for s in settings:
                vi = row_i[s]
                vj = row_j[s]
                if vi == '?' or vj == '?' or vi == vj:
                    continue
                diff_settings.append(s)
            if len(diff_settings) == 1:
                diff_col = diff_settings[0]
                pred_i = predict_row(row_i, goal_col, settings, intercept, beta, setting_means)
                pred_j = predict_row(row_j, goal_col, settings, intercept, beta, setting_means)
                diff = pred_j - pred_i
                low = round(diff - 2 * std_res, 4)
                high = round(diff + 2 * std_res, 4)
                pairs.append({
                    "a": i+1,
                    "b": j+1,
                    "setting": diff_col,
                    "difference": round(diff, 4),
                    "low": low,
                    "high": high
                })
    # Sort by absolute difference descending and take first 5
    pairs.sort(key=lambda x: abs(x["difference"]), reverse=True)
    selected_pairs = pairs[:max(5, len(pairs))]

    # ---- APACHE-03 ----
    print("== APACHE-03 ==")
    # print first 3 predictions as demonstration
    for idx in sorted(predictions_dict.keys())[:3]:
        print(f"{idx}: {predictions_dict[idx]}")
    print("ANSWER APACHE-03 " + json.dumps({
        "predictions": predictions_dict,
        "mae": mae
    }))

    # ---- APACHE-04 ----
    print("\n== APACHE-04 ==")
    for p in selected_pairs[:5]:
        print(f"Pair: a={p['a']} b={p['b']} setting={p['setting']} diff={p['difference']:.4f} low={p['low']:.4f} high={p['high']:.4f}")
    print("ANSWER APACHE-04 " + json.dumps({
        "pairs": selected_pairs
    }))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
