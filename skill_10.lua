#!/usr/bin/env lua
-- Skill 10 – Comparative analysis of vehicle selection and performance estimation
-- Usage:  lua skill_10.lua <csv path>

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Helper functions
-- ------------------------------------------------------------------

-- Compute the average of a list of numbers
local function average(vs)
  if #vs == 0 then return nil end
  local sum = 0
  for _, v in ipairs(vs) do sum = sum + v end
  return sum / #vs
end

-- Compute the mean absolute error between a set of predictions and actuals
local function mae(preds, act)
  local sum = 0
  for i = 1, #preds do sum = sum + math.abs(preds[i] - act[i]) end
  return sum / #preds
end

-- Return a list of rows sorted by distance to heaven (ascending)
local function sorted_by_disty(tbl, rows)
  local d = {}
  for _, r in ipairs(rows) do
    table.insert(d, {row = r, dist = tbl:disty(r)})
  end
  table.sort(d, function(a, b) return a.dist < b.dist end)
  local sorted = {}
  for _, v in ipairs(d) do table.insert(sorted, v.row) end
  return sorted, d
end

-- Pick the best rows based on lowest distance to heaven
local function best_dist_selection(tbl, cap)
  local sorted, _ = sorted_by_disty(tbl, tbl.rows)
  local sel = {}
  for i = 1, math.min(cap, #sorted) do
    table.insert(sel, sorted[i])
  end
  return sel
end

-- Random selection of rows
local function random_selection(tbl, cap)
  local shuffled = ezr.shuffle(tbl.rows)
  local sel = {}
  for i = 1, math.min(cap, #shuffled) do
    table.insert(sel, shuffled[i])
  end
  return sel
end

-- Compute mean distance for a selection
local function mean_dist(tbl, selection)
  local sum = 0
  for _, r in ipairs(selection) do sum = sum + tbl:disty(r) end
  return sum / #selection
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------

local csv_path = arg[1]
if not csv_path then
  io.stderr:write("Error: CSV file path required\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))
local target_col = tbl.cols.all["Mpg+"]

-- --------------------------------------------------------------
-- CAR-19 : Compare selection strategy vs random
-- --------------------------------------------------------------
print("== CAR-19 ==")

local cap = math.min(10, #tbl.rows)   -- evaluation budget

-- Best-distance strategy
local best_sel = best_dist_selection(tbl, cap)
local best_mean_dist = mean_dist(tbl, best_sel)

-- Random strategy
local rand_sel = random_selection(tbl, cap)
local rand_mean_dist = mean_dist(tbl, rand_sel)

print(string.format("Evaluation budget: %d rows", cap))
print(string.format("Best‑dist strategy mean distance to heaven: %.4f", best_mean_dist))
print(string.format("Random strategy mean distance to heaven: %.4f", rand_mean_dist))
print("\nTop 5 vehicles chosen by best‑dist strategy:")
for i = 1, math.min(5, #best_sel) do
  local r = best_sel[i]
  print(string.format("%2d) dist=%.4f  %s",
          i, tbl:disty(r), ezr.show(r)))
end
print("\nTop 5 vehicles chosen by random strategy:")
for i = 1, math.min(5, #rand_sel) do
  local r = rand_sel[i]
  print(string.format("%2d) dist=%.4f  %s",
          i, tbl:disty(r), ezr.show(r)))
end

-- --------------------------------------------------------------
-- CAR-20 : Evaluate predictive performance vs historical average
-- --------------------------------------------------------------
print("\n== CAR-20 ==")

-- Historical average for target
local hist_vals = {}
for _, r in ipairs(tbl.rows) do
  local v = r[target_col.at]
  if v ~= "?" then table.insert(hist_vals, v) end
end
local hist_avg = average(hist_vals)

-- Split data into training (half) and testing (half)
local shuffled = ezr.shuffle(tbl.rows)
local n = math.floor(#shuffled / 2)
local train = ezr.slice(shuffled, 1, n)
local test = ezr.slice(shuffled, n + 1)

-- Build a tree on training data
local tree = ezr.Tree(tbl, train)

-- Predict on test set using the tree and compute errors
local pred_vals = {}
local act_vals = {}
for _, r in ipairs(test) do
  local pred = tree:leaf(tbl, r)
  table.insert(pred_vals, pred)
  table.insert(act_vals, r[target_col.at])
end

-- Errors
local tree_mae = mae(pred_vals, act_vals)
local hist_mae = 0
for _, v in ipairs(act_vals) do hist_mae = hist_mae + math.abs(v - hist_avg) end
hist_mae = hist_mae / #act_vals

print(string.format("Number of test rows: %d", #test))
print(string.format("Historical‑average MAE: %.4f", hist_mae))
print(string.format("Tree‑based MAE:         %.4f", tree_mae))
print(string.format("Improvement:           %.2f%%",
        100 * (hist_mae - tree_mae) / hist_mae))
