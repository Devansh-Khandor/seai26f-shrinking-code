#!/usr/bin/env python3
import json
import sys
import math
import ezr


def build_tree_and_stats(tbl, rows):
    """
    Build a decision tree on the given rows and compute for every leaf
    the mean and standard deviation of the goal (Performance-) values.
    Returns the root node.
    """
    node = ezr.tree(tbl, rows)

    goal_col = next(col for col in tbl.cols.values() if col.txt.endswith("-"))

    def leaf_stats(n):
        if n.kids:
            for kid in n.kids:
                leaf_stats(kid)
        else:
            # compute mean & std of goal values in this leaf
            vals = [r[goal_col.at] for r in n.rows if r[goal_col.at] != "?"]
            n.mu_perf = sum(vals) / len(vals) if vals else 0.0
            # standard deviation
            if len(vals) > 1:
                mean = n.mu_perf
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                n.sd_perf = math.sqrt(var)
            else:
                n.sd_perf = 0.0

    leaf_stats(node)
    return node


def predict_leaf(node, row):
    """Walk the tree and return the leaf node."""
    n = node
    while n.kids:
        n = n.kids[0 if n.go(row) else 1]
    return n


def apache_03(tbl, heldout_idx, history_idx):
    """Estimate performance for held‑out rows."""
    # build tree on history rows
    hist_rows = [tbl.rows[i - 1] for i in history_idx]
    tree_root = build_tree_and_stats(tbl, hist_rows)

    # predictions
    predictions = {}
    mae_sum = 0.0
    count = 0

    goal_col = next(col for col in tbl.cols.values() if col.txt.endswith("-"))
    for idx in heldout_idx:
        row = tbl.rows[idx - 1]
        leaf = predict_leaf(tree_root, row)
        pred = leaf.mu_perf
        predictions[str(idx)] = pred
        mae_sum += abs(pred - row[goal_col.at])
        count += 1

    mae = mae_sum / count if count else 0.0
    # Output human readable results
    print(f"== APACHE-03 ==")
    for idx in sorted(heldout_idx):
        print(f"Row {idx} predicted: {predictions[str(idx)]:.2f}")
    print(f"MAE: {mae:.2f}")

    # JSON answer
    ans = {"predictions": predictions, "mae": round(mae, 2)}
    print("ANSWER APACHE-03", json.dumps(ans, separators=(",", ":")))


def apache_04(tbl, history_idx):
    """Estimate difference between pairs that differ in exactly one setting."""
    # build tree on history rows
    hist_rows = [tbl.rows[i - 1] for i in history_idx]
    tree_root = build_tree_and_stats(tbl, hist_rows)

    # collect all setting columns (exclude goal and X)
    setting_cols = [c for c in tbl.cols.values()
                    if not c.txt.endswith(("X", "+", "-"))]
    goal_col = next(c for c in tbl.cols.values() if c.txt.endswith("-"))

    pairs = []

    # helper to get leaf stats quickly
    leaf_cache = {}

    def get_leaf(row):
        l = predict_leaf(tree_root, row)
        if l not in leaf_cache:
            leaf_cache[l] = l
        return leaf_cache[l]

    # iterate over all pairs
    for i in range(len(history_idx)):
        idx_i = history_idx[i]
        row_i = tbl.rows[idx_i - 1]
        for j in range(i + 1, len(history_idx)):
            idx_j = history_idx[j]
            row_j = tbl.rows[idx_j - 1]
            # count differences in setting columns
            diff_cols = []
            for col in setting_cols:
                if row_i[col.at] != row_j[col.at]:
                    diff_cols.append(col)
            if len(diff_cols) != 1:
                continue
            col = diff_cols[0]
            leaf_i = get_leaf(row_i)
            leaf_j = get_leaf(row_j)
            diff_est = leaf_j.mu_perf - leaf_i.mu_perf
            # uncertainty: use sum of std devs of the two leaves
            uncert = leaf_i.sd_perf + leaf_j.sd_perf
            low = diff_est - uncert
            high = diff_est + uncert
            pairs.append({
                "a": idx_i,
                "b": idx_j,
                "setting": col.txt,
                "difference": round(diff_est, 2),
                "low": round(low, 2),
                "high": round(high, 2),
            })
            if len(pairs) >= 5:
                break
        if len(pairs) >= 5:
            break

    # Output human readable results
    print(f"== APACHE-04 ==")
    for p in pairs:
        print(f"Rows {p['a']} & {p['b']} ({p['setting']}): diff={p['difference']:.2f} low={p['low']:.2f} high={p['high']:.2f}")
    print(f"Total pairs reported: {len(pairs)}")

    # JSON answer
    ans = {"pairs": pairs}
    print("ANSWER APACHE-04", json.dumps(ans, separators=(",", ":")))


def main(csv_path):
    rows = list(ezr.csv(csv_path))
    tbl = ezr.Tbl(rows)

    # determine indices
    num_rows = len(rows)
    heldout_idx = [i + 1 for i in range(num_rows) if (i + 1) % 5 == 0]
    history_idx = [i + 1 for i in range(num_rows) if (i + 1) % 5 != 0]

    # Run the two requested skills
    apache_03(tbl, heldout_idx, history_idx)
    apache_04(tbl, history_idx)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
