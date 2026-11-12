#!/usr/bin/env lua
--  skill_10.lua – performance comparison (CAR‑19 & CAR‑20)
--  Usage: lua skill_10.lua <csv_path>
--
--  The script compares the dealer’s active‑learning vehicle‑selection
--  system against a random baseline (CAR‑19) and evaluates the
--  prediction accuracy of the tree‑based estimates versus a simple
--  historical‑average baseline (CAR‑20).  Only the core ezr library
--  and standard Lua functions are used.

local ezr = require("ezr")

-- ------------------------------------------------------------------
--  Helpers – copied from earlier skills
-- ------------------------------------------------------------------
-- Normalised score for a column.
local function norm_score(col, r, maximize)
  local v = r[col.at]
  if v == "?" then return 0.5 end          -- neutral if unknown
  local n = col:norm(v)
  return maximize and n or (1 - n)
end

-- Composite desirability score: higher is better.
local function composite_score(r, cols)
  return (norm_score(cols.Lbs, r, false) +
          norm_score(cols.Acc, r, true) +
          norm_score(cols.Mpg, r, true)) / 3
end

-- Mean of a list of numbers, ignores missing values.
local function mean_of(vals)
  local s, n = 0, 0
  for _, v in ipairs(vals) do
    if v ~= "?" then
      s, n = s + v, n + 1
    end
  end
  return n > 0 and s / n or nil
end

-- Mean absolute error between two lists of numbers
local function mae(pred, actual)
  local sum, n = 0, 0
  for i = 1, #actual do
    local a = actual[i]
    local p = pred[i]
    if a ~= "?" and p ~= nil then
      sum, n = sum + math.abs(a - p), n + 1
    end
  end
  return n > 0 and sum / n or nil
end

-- ------------------------------------------------------------------
--  Load data
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("usage: lua skill_10.lua <csv_path>\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

-- Reference columns
local cols = {
  Lbs = tbl.cols.all[5],   -- "Lbs-"
  Acc = tbl.cols.all[7],   -- "Acc+"
  Mpg = tbl.cols.all[8]    -- "Mpg+"
}

-- ------------------------------------------------------------------
--  CAR‑19 : Active‑learning vs random selection
-- ------------------------------------------------------------------
print("== CAR-19 ==")

local budget = ezr.the.budget or 10
budget = math.min(budget, #tbl.rows)

-- Active‑learning list
local active_list = tbl:acquirer(budget)

-- Random list – same length
local random_list = table.sub(tbl.rows, 1)   -- copy
shuffle(random_list)                       -- defined in ezr-lib
random_list = table.sub(random_list, 1, budget)

-- Helper to compute average composite score
local function avg_score(list)
  local sum, n = 0, 0
  for _, r in ipairs(list) do
    local s = composite_score(r, cols)
    sum, n = sum + s, n + 1
  end
  return n > 0 and sum / n or 0
end

local avg_active = avg_score(active_list)
local avg_random = avg_score(random_list)

print(string.format("Average composite score – active‑learning: %.3f", avg_active))
print(string.format("Average composite score – random:      %.3f", avg_random))
print(string.format("Improvement: %.3f (%% of random)", (avg_active-avg_random)/avg_random * 100))
print()

-- ------------------------------------------------------------------
--  CAR‑20 : Tree‑based estimate vs historical mean
-- ------------------------------------------------------------------
print("== CAR-20 ==")

-- Split data into train/test (two‑thirds train)
local shuffled = table.sub(tbl.rows, 1)
shuffle(shuffled)
local n = #shuffled
local train_n = math.floor(2*n/3)
local train_rows = {}
local test_rows  = {}
for i = 1, train_n do train_rows[i] = shuffled[i] end
for i = train_n+1, n do test_rows[i-train_n] = shuffled[i] end

-- Build a regression tree for weight (Lbs-)
local weight_tree = ezr.Tree(tbl, train_rows, function(r) return r[cols.Lbs.at] end)

-- Predict on test set
local pred_weights = {}
local actual_weights = {}
for _, r in ipairs(test_rows) do
  table.insert(pred_weights, weight_tree:leaf(tbl, r))
  table.insert(actual_weights, r[cols.Lbs.at])
end

local mae_tree = mae(pred_weights, actual_weights)

-- Historical mean from training set
local train_weights = {}
for _, r in ipairs(train_rows) do
  if r[cols.Lbs.at] ~= "?" then
    table.insert(train_weights, r[cols.Lbs.at])
  end
end
local mean_weight = mean_of(train_weights)

-- Predict mean for all test rows
local pred_mean_weights = {}
for _, _ in ipairs(test_rows) do
  table.insert(pred_mean_weights, mean_weight)
end

local mae_mean = mae(pred_mean_weights, actual_weights)

print(string.format("MAE – tree estimate: %.2f", mae_tree))
print(string.format("MAE – historical‑average: %.2f", mae_mean))
print(string.format("Tree improves by %.2f%% over mean", (mae_mean-mae_tree)/mae_mean*100))
