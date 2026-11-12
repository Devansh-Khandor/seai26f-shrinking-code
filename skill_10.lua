#!/usr/bin/env lua
--  skill_10.lua
--
--  Implements the following Apache diagnostics:
--
--  APACHE-19 – Compare our configuration-selection system against random
--              configuration selection under the same benchmarking budget,
--              and show which approach discovers better-performing
--              configurations.
--
--  APACHE-20 – Compare our performance predictions against a simple
--              historical-average estimate and show whether our system
--              provides a meaningful improvement on previously unseen
--              configurations.
--
--  Usage:
--      lua skill_10.lua <csv_path>
--
--  The script prints a line with the exact heading
--  "== APACHE-19 ==" and "== APACHE-20 ==" followed by at least
--  three lines of results each.

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- Helper: load CSV into a table and return the table object
-- ------------------------------------------------------------------
local function load_csv(path)
  local src = ezr.csv(path)
  return ezr.Tbl(src)
end

-- ------------------------------------------------------------------
-- Helper: compute best (lowest) actual performance in a subset
-- and return its value, index, and row reference
-- ------------------------------------------------------------------
local function best_actual(subset, idx_of_row)
  local best_val = nil
  local best_idx = nil
  local best_row = nil
  for _, r in ipairs(subset) do
    local v = r[perf_col.at]
    local i = idx_of_row[r]
    if best_val == nil or v < best_val then
      best_val = v
      best_idx = i
      best_row = r
    end
  end
  return best_val, best_idx, best_row
end

-- ------------------------------------------------------------------
-- Helper: compute mean of actual performance in a subset
-- ------------------------------------------------------------------
local function mean_actual(subset)
  local sum = 0
  for _, r in ipairs(subset) do
    sum = sum + r[perf_col.at]
  end
  return sum / #subset
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("usage: lua skill_10.lua <csv_path>\n")
  os.exit(1)
end

-- Load data
local tbl = load_csv(csv_path)
local perf_col = tbl.cols.y[1]   -- first goal column
local n = #tbl.rows

-- Map rows to their original indices for reporting
local idx_of_row = {}
for i, r in ipairs(tbl.rows) do idx_of_row[r] = i end

-- ------------------------------------------------------------------
-- Define a budget for benchmarking
-- ------------------------------------------------------------------
local budget = math.min(20, math.floor(n * 0.10))
if budget < 1 then budget = 1 end

-- ------------------------------------------------------------------
-- APACHE-19 : Compare active learner vs random selection
-- ------------------------------------------------------------------
print("== APACHE-19 ==")

-- 1. Active learner – use built‑in acquirer
local active_subset = tbl:acquirer(budget)

-- 2. Random subset – shuffle indices and pick first budget
local indices = {}
for i = 1, n do indices[i] = i end
indices = ezr.shuffle(indices)
local random_subset = {}
for i = 1, budget do
  table.insert(random_subset, tbl.rows[indices[i]])
end

-- 3. Find best actual performance in each subset
local best_act_val, best_act_idx, _ = best_actual(active_subset, idx_of_row)
local best_rand_val, best_rand_idx, _ = best_actual(random_subset, idx_of_row)

-- 4. Compute means of subsets (just for illustration)
local mean_act = mean_actual(active_subset)
local mean_rand = mean_actual(random_subset)

-- 5. Report
print(string.format(
  "  Active learner selected %d configs, best actual=%.2f (Config %d)",
  #active_subset, best_act_val, best_act_idx))
print(string.format(
  "  Random selection selected %d configs, best actual=%.2f (Config %d)",
  #random_subset, best_rand_val, best_rand_idx))
print(string.format(
  "  Mean performance in active subset: %.2f, in random subset: %.2f",
  mean_act, mean_rand))
if best_act_val < best_rand_val then
  print("  Active learner discovers a better configuration.")
else
  print("  Random selection discovers a better (or equal) configuration.")
end
print()

-- ------------------------------------------------------------------
-- APACHE-20 : Compare tree predictions vs historical average
-- ------------------------------------------------------------------
print("== APACHE-20 ==")

-- 1. Train full regression tree
local tree = ezr.Tree(tbl, tbl.rows)

-- 2. Compute predictions and residuals
local preds = {}
local residuals = {}
for i, r in ipairs(tbl.rows) do
  local pred = tree:leaf(tbl, r)
  local act = r[perf_col.at]
  preds[i] = pred
  residuals[i] = act - pred
end

-- 3. Compute RMSE for tree
local sumsq = 0
for _, r in ipairs(residuals) do
  sumsq = sumsq + r * r
end
local rmse_tree = math.sqrt(sumsq / n)

-- 4. Historical average baseline
local sum_all = 0
for _, r in ipairs(tbl.rows) do
  sum_all = sum_all + r[perf_col.at]
end
local mean_all = sum_all / n

local residuals_base = {}
for i, r in ipairs(tbl.rows) do
  local act = r[perf_col.at]
  residuals_base[i] = act - mean_all
end
sumsq = 0
for _, r in ipairs(residuals_base) do
  sumsq = sumsq + r * r
end
local rmse_base = math.sqrt(sumsq / n)

-- 5. Report
print(string.format(
  "  RMSE of tree predictions: %.4f", rmse_tree))
print(string.format(
  "  RMSE of historical‑average predictions: %.4f", rmse_base))
local improvement = (rmse_base - rmse_tree) / rmse_base * 100
print(string.format(
  "  Improvement over baseline: %.2f%%", improvement))
