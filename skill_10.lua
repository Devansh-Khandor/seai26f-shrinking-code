#!/usr/bin/env lua
-- skill_10.lua
-- Implements the requested APACHE-19 and APACHE-20 analyses.
--
-- Usage: lua skill_10.lua <csv path>
--
-- Requires the ezr library from the provided codebase.

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- Utility helpers

-- Round numbers using ezr's formatting
local function display(v)
  if type(v) == "number" then
    return ezr.round(v)
  else
    return tostring(v)
  end
end

-- Convert a row table to a CSV string
local function row_to_csv(row)
  local parts = {}
  for _,v in ipairs(row) do
    if v == "?" then
      table.insert(parts, "?")
    else
      table.insert(parts, tostring(v))
    end
  end
  return table.concat(parts, ",")
end

-- ------------------------------------------------------------------
-- Main logic

local csvfile = arg[1]
assert(csvfile, "Usage: lua skill_10.lua <csv path>")

-- Load data
local tbl = ezr.Tbl(ezr.csv(csvfile))
local perf_col = tbl.cols.y[1]   -- the only goal column

--------------------------------------------------------------------
-- APACHE-19: Compare active learning vs random sampling

print("== APACHE-19 ==")

-- Benchmark budget: use the same as in the library
local cap = ezr.the.budget

-- Active learning selection
local active_lab = tbl:acquirer(cap)

local best_active = math.huge
for _,r in ipairs(active_lab) do
  local v = r[perf_col.at]
  if v < best_active then best_active = v end
end

-- Random selection
local shuffled = ezr.shuffle(tbl.rows)
local random_lab = {}
for i=1,cap do
  table.insert(random_lab, shuffled[i])
end

local best_random = math.huge
for _,r in ipairs(random_lab) do
  local v = r[perf_col.at]
  if v < best_random then best_random = v end
end

print(string.format("Benchmark budget = %d rows", cap))
print(string.format("Best performance found by active learning: %s",
      display(best_active)))
print(string.format("Best performance found by random sampling: %s",
      display(best_random)))
if best_active < best_random then
  print("Active learning discovers better-performing configurations.")
else
  print("Random sampling discovers better or equal performing configurations.")
end
print()

--------------------------------------------------------------------
-- APACHE-20: Compare prediction accuracy of our model vs historical average

print("== APACHE-20 ==")

-- Split data: 80% training, 20% test
local shuffled_all = ezr.shuffle(tbl.rows)
local ntrain = math.floor(#shuffled_all * 0.8)
local train_rows = {}
for i=1,ntrain do table.insert(train_rows, shuffled_all[i]) end
local test_rows  = {}
for i=ntrain+1,#shuffled_all do table.insert(test_rows, shuffled_all[i]) end

-- Build tree model on training data
local train_tbl = tbl:clone(train_rows)
local model_tree = ezr.Tree(train_tbl, train_rows)

-- Mean performance on training set (historical average)
local mean_train = perf_col:mid()

-- Helper to compute MAE
local function mae(preds, actuals)
  local sum = 0
  for i=1,#preds do
    sum = sum + math.abs(preds[i] - actuals[i])
  end
  return sum / #preds
end

-- Predict on test set using tree
local preds_tree = {}
local actuals = {}
for _,r in ipairs(test_rows) do
  table.insert(preds_tree, model_tree:leaf(train_tbl, r))
  table.insert(actuals, r[perf_col.at])
end

local mae_tree = mae(preds_tree, actuals)

-- Predict on test set using baseline (historical average)
local preds_baseline = {}
for _,v in ipairs(actuals) do
  table.insert(preds_baseline, mean_train)
end

local mae_baseline = mae(preds_baseline, actuals)

print(string.format("Test set size = %d rows", #test_rows))
print(string.format("MAE of tree model: %s", display(mae_tree)))
print(string.format("MAE of historical average: %s", display(mae_baseline)))
if mae_tree < mae_baseline then
  local rel = (mae_baseline - mae_tree) / mae_baseline * 100
  print(string.format("Tree model improves accuracy by %.1f%% over the baseline.",
        rel))
else
  print("Tree model does not improve accuracy compared to the historical average.")
end
