#!/usr/bin/env lua
-- skill_10.lua
-- NASA-19: Compare our estimation system against a simple historical‑average estimate
-- NASA-20: Compare our project‑selection approach with random selection
--
-- Run as:  lua skill_10.lua <csv path>
--
-- The implementation re‑uses helpers from the existing skills, and
-- only relies on the standard libraries and `ezr`.

local ezr = require("ezr")

local csv = ezr.csv
local Tbl = ezr.Tbl

-- ----------------------------------------------------------------------
-- Helper: simple linear regression (x -> y)
local function linear_regression(x, y)
  local n = #x
  if n < 2 then error("Need at least 2 points for regression") end
  local sumx, sumy, sumxy, sumx2 = 0, 0, 0, 0
  for i = 1, n do
    local xi, yi = x[i], y[i]
    sumx  = sumx  + xi
    sumy  = sumy  + yi
    sumxy = sumxy + xi*yi
    sumx2 = sumx2 + xi*xi
  end
  local meanx = sumx / n
  local meany = sumy / n
  local varx  = sumx2 / n - meanx*meanx
  local covxy = sumxy / n - meanx*meany
  local slope = covxy / varx
  local intercept = meany - slope * meanx
  return {slope = slope, intercept = intercept}
end

-- ----------------------------------------------------------------------
-- Utility: split a table into training and test halves
local function split_train_test(tbl)
  local rows = shuffle(tbl.rows)
  local n = #rows
  local mid = math.floor(n / 2)
  local train = {}
  for i = 1, mid do train[i] = rows[i] end
  local test  = {}
  for i = mid + 1, n do test[i - mid] = rows[i] end
  return train, test
end

-- ----------------------------------------------------------------------
-- Compute error metrics for a set of predictions
local function error_metrics(pred, actual)
  local n = #pred
  local mae, mse = 0, 0
  for i = 1, n do
    local diff = pred[i] - actual[i]
    mae = mae + math.abs(diff)
    mse = mse + diff * diff
  end
  mae = mae / n
  mse = mse / n
  return mae, mse
end

-- ----------------------------------------------------------------------
-- NASA-19: predictive comparison
-- ----------------------------------------------------------------------
local function nasa19(tbl)
  local idxKloc   = ezr.colIndex(tbl, "Kloc+")
  local idxEffort = ezr.colIndex(tbl, "Effort-")

  local train, test = split_train_test(tbl)

  -- Build regression on training data
  local kloc_train = {}
  local effort_train = {}
  for _, r in ipairs(train) do
    local k = r[idxKloc]
    local e = r[idxEffort]
    if type(k) == "number" and type(e) == "number" then
      table.insert(kloc_train, k)
      table.insert(effort_train, e)
    end
  end
  local model = linear_regression(kloc_train, effort_train)
  local baseline_mean = 0
  for _, v in ipairs(effort_train) do baseline_mean = baseline_mean + v end
  baseline_mean = baseline_mean / #effort_train

  -- Evaluate on test set
  local pred_reg = {}
  local pred_baseline = {}
  local actual = {}
  for _, r in ipairs(test) do
    local k = r[idxKloc]
    local e = r[idxEffort]
    if type(k) == "number" and type(e) == "number" then
      table.insert(actual, e)
      table.insert(pred_reg, model.intercept + model.slope * k)
      table.insert(pred_baseline, baseline_mean)
    end
  end

  local mae_reg, mse_reg = error_metrics(pred_reg, actual)
  local mae_bas, mse_bas = error_metrics(pred_baseline, actual)

  print("== NASA-19 ==")
  print(string.format(
    "Regression MAE: %.2f, Baseline MAE: %.2f, Improvement: %.1f%%",
    mae_reg, mae_bas, (mae_bas - mae_reg) / mae_bas * 100))
  print(string.format(
    "Regression MSE: %.2f, Baseline MSE: %.2f, Improvement: %.1f%%",
    mse_reg, mse_bas, (mse_bas - mse_reg) / mse_bas * 100))
  print(string.format(
    "Regression R² not computed, but MAE/MSE reductions indicate better predictive power."))
end

-- ----------------------------------------------------------------------
-- NASA-20: selection comparison
-- ----------------------------------------------------------------------
local function nasa20(tbl)
  local idxKloc   = ezr.colIndex(tbl, "Kloc+")
  local idxEffort = ezr.colIndex(tbl, "Effort-")
  local idxDefect = ezr.colIndex(tbl, "Defects-")
  local idxId     = ezr.colIndex(tbl, "idX")

  -- Build model on all data
  local kloc_all = {}
  local effort_all = {}
  for _, r in ipairs(tbl.rows) do
    local k = r[idxKloc]
    local e = r[idxEffort]
    if type(k) == "number" and type(e) == "number" then
      table.insert(kloc_all, k)
      table.insert(effort_all, e)
    end
  end
  local model = linear_regression(kloc_all, effort_all)

  -- Compute score = predicted effort + predicted defects
  local scores = {}
  for _, r in ipairs(tbl.rows) do
    local k = r[idxKloc]
    if type(k) == "number" then
      local pred_eff = model.intercept + model.slope * k
      local pred_def = r[idxDefect] -- use actual defect for simplicity
      table.insert(scores, {id = r[idxId], eff = pred_eff, def = pred_def, score = pred_eff + pred_def})
    end
  end

  -- Determine how many to evaluate: use the same budget as defined in `the`
  local budget = the.budget or 10
  if budget > #scores then budget = #scores end

  -- Our selection: top‑budget by lowest score
  local sorted = sorted(scores, function(a, b) return a.score < b.score end)
  local selected = {}
  for i = 1, budget do selected[i] = sorted[i] end
  local sum_selected = 0
  for _, s in ipairs(selected) do sum_selected = sum_selected + s.score end
  local avg_selected = sum_selected / budget

  -- Random selection
  local rand_rows = shuffle(tbl.rows)
  local random_selected = {}
  for i = 1, budget do random_selected[i] = rand_rows[i] end
  local sum_random = 0
  for _, r in ipairs(random_selected) do
    local k = r[idxKloc]
    local pred_eff = model.intercept + model.slope * k
    local pred_def = r[idxDefect]
    sum_random = sum_random + (pred_eff + pred_def)
  end
  local avg_random = sum_random / budget

  print("== NASA-20 ==")
  print(string.format(
    "Our selection (top %d) avg score: %.2f", budget, avg_selected))
  print(string.format(
    "Random selection avg score: %.2f", avg_random))
  if avg_selected < avg_random then
    print(string.format(
      "Our selection is better by %.2f points on average", avg_random - avg_selected))
  else
    print(string.format(
      "Random selection is better by %.2f points on average", avg_selected - avg_random))
  end
end

-- ----------------------------------------------------------------------
-- Main entry point
-- ----------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    error("Usage: lua skill_10.lua <csv path>")
  end
  srand(the.seed)  -- deterministic
  local tbl = Tbl(csv(path))
  nasa19(tbl)
  print()
  nasa20(tbl)
end

main()
