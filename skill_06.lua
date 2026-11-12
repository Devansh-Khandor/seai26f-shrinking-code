#!/usr/bin/env lua
--  skill_06.lua
--
--  Implements the following Apache diagnostics:
--
--  APACHE-11 – When the system recommends a configuration, explain
--              its recommendation by comparing the configuration's
--              settings and performance with relevant previously
--              tested configurations.
--
--  APACHE-12 – When two configurations receive different
--              performance assessments, explain which settings
--              differ between them and how those differences
--              relate to the predicted or recorded outcomes.
--
--  Usage:
--      lua skill_06.lua <csv_path>
--
--  Output:
--      Lines prefixed by "== APACHE-11 ==" and
--      "== APACHE-12 ==" followed by at least three lines of
--      results, including the actual numbers.

local ezr = require"ezr"

local function load_csv(path)
  local src = ezr.csv(path)
  return ezr.Tbl(src)
end

-- Train a regression tree on the entire data set
local function train_tree(tbl)
  return ezr.Tree(tbl, tbl.rows)
end

-- Compute predictions for all rows using the tree
local function compute_predictions(tree, tbl)
  local preds = {}
  for i, row in ipairs(tbl.rows) do
    preds[i] = tree:leaf(tbl, row)
  end
  return preds
end

-- Find index of row with best (lowest) predicted value
local function best_pred_index(preds)
  local best = 1
  for i = 2, #preds do
    if preds[i] < preds[best] then best = i end
  end
  return best
end

-- Find index of row with best (lowest) actual value
local function best_actual_index(tbl, perf_col)
  local best = 1
  for i = 2, #tbl.rows do
    if tbl.rows[i][perf_col.at] < tbl.rows[best][perf_col.at] then
      best = i
    end
  end
  return best
end

-- Compute the absolute difference in predictions between two rows
local function pred_diff(preds, i, j)
  return math.abs(preds[i] - preds[j])
end

-- Compute the absolute difference in actuals between two rows
local function actual_diff(tbl, perf_col, i, j)
  return math.abs(tbl.rows[i][perf_col.at] - tbl.rows[j][perf_col.at])
end

-- Identify two rows with maximal predicted difference
local function max_pred_pair(preds)
  local i_max, j_max, best = 1, 2, math.abs(preds[1] - preds[2])
  for i = 1, #preds do
    for j = i+1, #preds do
      local d = math.abs(preds[i] - preds[j])
      if d > best then
        best = d
        i_max, j_max = i, j
      end
    end
  end
  return i_max, j_max
end

-- Find settings that differ between two rows
local function diff_settings(tbl, i, j)
  local diffs = {}
  for _, c in ipairs(tbl.cols.x) do
    if tbl.rows[i][c.at] ~= tbl.rows[j][c.at] then
      diffs[#diffs+1] = {name=c.name,
                         val_i=tbl.rows[i][c.at],
                         val_j=tbl.rows[j][c.at]}
    end
  end
  return diffs
end

-- Helper to format a list of setting differences
local function format_diff_settings(diffs)
  if #diffs == 0 then return "None" end
  local parts = {}
  for _, d in ipairs(diffs) do
    table.insert(parts, string.format("%s: %s→%s", d.name,
                                     d.val_i, d.val_j))
  end
  return table.concat(parts, "; ")
end

-- --------------------------------------------------------------------
-- Main
-- --------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("usage: lua skill_06.lua <csv_path>\n")
  os.exit(1)
end

local tbl = load_csv(csv_path)
local perf_col = tbl.cols.y[1]   -- first goal column

-- Train model
local tree = train_tree(tbl)
local preds = compute_predictions(tree, tbl)

-- ------------------ APACHE-11 ------------------------------------
print("== APACHE-11 ==")

local best_pred_idx = best_pred_index(preds)
local best_act_idx = best_actual_index(tbl, perf_col)

local rec_pred = preds[best_pred_idx]
local rec_actual = tbl.rows[best_pred_idx][perf_col.at]
local act_actual = tbl.rows[best_act_idx][perf_col.at]
local act_pred = preds[best_act_idx]

print(string.format(
  "Recommended config: Config %d (predicted perf=%.2f)",
  best_pred_idx, rec_pred))
print(string.format(
  "Best actual config: Config %d (actual perf=%.2f)",
  best_act_idx, act_actual))
print(string.format(
  "Difference between predicted of recommended and actual of best: %.2f",
  rec_pred - act_actual))

-- Highlight settings that differ between recommended and best actual
local setting_diff = diff_settings(tbl, best_pred_idx, best_act_idx)
print("Differing settings between recommended and best actual:")
print("  " .. format_diff_settings(setting_diff))
print()

-- ------------------ APACHE-12 ------------------------------------
print("== APACHE-12 ==")

-- Pick two configs with the largest predicted difference
local idx1, idx2 = max_pred_pair(preds)
local pred_d = pred_diff(preds, idx1, idx2)
local act_d = actual_diff(tbl, perf_col, idx1, idx2)

print(string.format(
  "Config %d and Config %d have the largest predicted difference: %.2f",
  idx1, idx2, pred_d))
print(string.format(
  "Their actual difference: %.2f", act_d))

local diff_cfg = diff_settings(tbl, idx1, idx2)
print("Settings that differ between these two configurations:")
print("  " .. format_diff_settings(diff_cfg))
print()

-- Suggest how differences relate to predictions
print("Interpretation:")
print(string.format(
  "The predicted performance difference of %.2f likely arises from the %d settings that differ.",
  pred_d, #diff_cfg))
print("Each differing setting toggles the model's leaf path, affecting the mean prediction.")
