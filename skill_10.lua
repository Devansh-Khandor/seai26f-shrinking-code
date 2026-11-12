#!/usr/bin/env lua
-- skill_10.lua
-- Run: lua skill_10.lua <csv_path>
--
-- Provides two NASA requests:
--   NASA-19 – Compare project‑effort estimation system against simple historical‑average estimate
--             and show whether it makes meaningfully better predictions on previously unseen projects.
--   NASA-20 – Compare our project‑selection approach with randomly selecting proposals
--             under the same evaluation budget, and show whether our system identifies more desirable project outcomes.

local ezr = require "ezr"

-- Helpers from ezr
local csv     = ezr.csv
local Tbl     = ezr.Tbl
local push    = ezr.push
local sorted  = ezr.sorted
local slice   = ezr.slice
local shuffle = ezr.shuffle
local sqrt    = math.sqrt
local the     = ezr.the

-- ------------------------------------------------------------------
-- Read data
-- ------------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_10.lua <csv_path>\n")
  os.exit(1)
end

local tbl = Tbl(csv(path))
local rows = tbl.rows

-- ------------------------------------------------------------------
-- Utility functions
-- ------------------------------------------------------------------
local function mean_err(preds, actuals)
  local n = #preds
  local sum = 0
  for i = 1, n do
    sum = sum + math.abs(preds[i] - actuals[i])
  end
  return sum / n
end

local function rmse_err(preds, actuals)
  local n = #preds
  local sum = 0
  for i = 1, n do
    local d = preds[i] - actuals[i]
    sum = sum + d * d
  end
  return sqrt(sum / n)
end

local function compute_mean(rows, col)
  local n, sum = 0, 0
  for _, r in ipairs(rows) do
    local v = r[col.at]
    if v ~= "?" then
      sum = sum + v
      n = n + 1
    end
  end
  return n > 0 and sum / n or nil
end

-- ------------------------------------------------------------------
-- NASA-19: Compare regression tree vs historical average
-- ------------------------------------------------------------------
print("== NASA-19 ==")

-- 50/50 split
local shuffled = shuffle(rows)
local n = #shuffled
local half = math.floor(n / 2)

local train_rows = slice(shuffled, 1, half)
local test_rows  = slice(shuffled, half + 1)

local train_tbl = tbl:clone(train_rows)
local test_tbl  = tbl:clone(test_rows)

-- Build regression tree on Effort (first goal column)
local effort_col = train_tbl.cols.y[1]  -- Effort-
local tree = ezr.Tree(train_tbl, train_rows)

-- Predictions
local preds_tree  = {}
local actuals     = {}
for _, r in ipairs(test_rows) do
  local pred = tree:leaf(train_tbl, r)
  push(preds_tree, pred)
  push(actuals, r[effort_col.at])
end

-- Historical‑average predictions
local mean_effort = train_tbl.cols.y[1]:mid()
local preds_mean = {}
for _ in ipairs(test_rows) do push(preds_mean, mean_effort) end

-- Errors
local mae_tree = mean_err(preds_tree, actuals)
local mae_mean = mean_err(preds_mean, actuals)
local rmse_tree = rmse_err(preds_tree, actuals)
local rmse_mean = rmse_err(preds_mean, actuals)

local improvement_mae = (mae_mean - mae_tree) / mae_mean * 100
local improvement_rmse = (rmse_mean - rmse_tree) / rmse_mean * 100

print(string.format("Training size: %d, Testing size: %d", half, #test_rows))
print(string.format("Mean Absolute Error (MAE):")
      .."\n  Tree model: %.3f" .."\n  Historical mean: %.3f",
      mae_tree, mae_mean))
print(string.format("Root‑Mean‑Square Error (RMSE):"
      .."\n  Tree model: %.3f" .."\n  Historical mean: %.3f",
      rmse_tree, rmse_mean))
print(string.format("Improvement of Tree over Mean: "
      .."MAE: %.1f%%, RMSE: %.1f%%",
      improvement_mae, improvement_rmse))

-- ------------------------------------------------------------------
-- NASA-20: Selection vs Random
-- ------------------------------------------------------------------
print("\n== NASA-20 ==")

local budget = the.budget or 50
if budget > #rows then budget = #rows end

-- Scoring function (same as NASA-14)
local function score_row(r)
  local sz = r[26]  -- Kloc+
  local ef = r[27]  -- Effort-
  local df = r[28]  -- Defects-
  if sz == "?" or ef == "?" or df == "?" then
    return -1e9
  end
  local denom = ef + df
  if denom == 0 then return 1e9 end
  return sz / denom
end

-- Select top by score
local scored = {}
for _, r in ipairs(rows) do
  push(scored, {row=r, score=score_row(r)})
end
scored = sorted(scored, function(a,b) return a.score > b.score end)
local selected = {}
for i = 1, budget do
  push(selected, scored[i].row)
end

-- Random selection
local rand_rows = shuffle(rows)
local random_selected = slice(rand_rows, 1, budget)

-- Compute metrics for both sets
local function metrics(rows)
  local eff_sum, def_sum, size_sum, n = 0, 0, 0, 0
  for _, r in ipairs(rows) do
    local ef = r[27]
    local df = r[28]
    local sz = r[26]
    if ef ~= "?" then eff_sum = eff_sum + ef; n = n + 1 end
    if df ~= "?" then def_sum = def_sum + df end
    if sz ~= "?" then size_sum = size_sum + sz end
  end
  return {
    avg_eff = n > 0 and eff_sum / n or nil,
    avg_def = n > 0 and def_sum / n or nil,
    avg_sz  = n > 0 and size_sum / n or nil,
    count   = n
  }
end

local sel_metrics   = metrics(selected)
local rand_metrics  = metrics(random_selected)

print(string.format("Evaluation budget: %d projects", budget))
print("\nSelected by NASA‑14 scoring:")
print(string.format("  Avg Effort  : %.2f", sel_metrics.avg_eff))
print(string.format("  Avg Defects : %.2f", sel_metrics.avg_def))
print(string.format("  Avg Size    : %.2f", sel_metrics.avg_sz))

print("\nRandom selection:")
print(string.format("  Avg Effort  : %.2f", rand_metrics.avg_eff))
print(string.format("  Avg Defects : %.2f", rand_metrics.avg_def))
print(string.format("  Avg Size    : %.2f", rand_metrics.avg_sz))

print(string.format("\nComparison:")
      .."\n  Effort reduction: %.1f%%" ..
      "\n  Defect reduction: %.1f%%",
      100*(rand_metrics.avg_eff - sel_metrics.avg_eff)/rand_metrics.avg_eff,
      100*(rand_metrics.avg_def - sel_metrics.avg_def)/rand_metrics.avg_def))
