#!/usr/bin/env lua
--[[
  skill_10.lua
  NASA-93 data analysis – evaluation of effort estimation and proposal selection.
  NASA-19: compare regression‑based effort predictor with a simple historical‑average estimate.
  NASA-20: compare our proposal selection strategy with random selection under the same budget.
  Usage:  lua skill_10.lua <csv-path>
  Prints:
    NASA-19: MAE of regression vs historical average and % improvement.
    NASA-20: average (eff+def)/size for our selection vs random, with totals.
]]--

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- helper functions
-- ------------------------------------------------------------------
local function abort(msg)
  io.stderr:write(msg .. "\n")
  os.exit(1)
end

local function round(v)
  return ezr.round(v)
end

-- simple linear regression (least squares)
local function linreg(x, y)
  local n = #x
  if n < 2 then abort("Not enough points for regression") end
  local sx, sx2, sy, sy2, sxy = 0,0,0,0,0
  for i = 1, n do
    local xi, yi = x[i], y[i]
    sx  = sx + xi
    sx2 = sx2 + xi*xi
    sy  = sy + yi
    sy2 = sy2 + yi*yi
    sxy = sxy + xi*yi
  end
  local mean_x = sx / n
  local mean_y = sy / n
  local var_x  = sx2 / n - mean_x*mean_x
  local b = (sxy / n - mean_x*mean_y) / var_x
  local a = mean_y - b * mean_x
  return {a=a, b=b}
end

local function mean(values)
  local sum, n = 0, #values
  for _,v in ipairs(values) do sum = sum + v end
  return n>0 and sum/n or 0
end

local function absmean(values)
  local sum, n = 0, #values
  for _,v in ipairs(values) do sum = sum + math.abs(v) end
  return n>0 and sum/n or 0
end

local function shuffle(tbl)
  local t = table.clone(tbl)
  local n = #t
  for i=n,2,-1 do
    local j = ezr.rand(1,i)
    t[i],t[j] = t[j],t[i]
  end
  return t
end

-- ------------------------------------------------------------------
-- Load data
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then abort("Usage: lua skill_10.lua <csv-path>") end

local rows = {}
for r in ezr.csv(csv_path) do
  table.insert(rows, r)
end
if #rows == 0 then abort("No data found in " .. csv_path) end

-- Map column names to indices
local col_map = {}
for i,c in ipairs(ezr.Tbl(rows).cols.all) do
  col_map[ezr.Tbl(rows).cols.names[i]] = c
end

local sz_col   = col_map["Kloc+"]
local eff_col  = col_map["Effort-"]
local def_col  = col_map["Defects-"]

-- ------------------------------------------------------------------
-- NASA-19: effort estimation comparison
-- ------------------------------------------------------------------
print("== NASA-19 ==")

-- 70/30 split
local shuffled = shuffle(rows)
local n = #shuffled
local train_n = math.floor(0.7 * n)
local train_rows = {}
for i=1,train_n do table.insert(train_rows, shuffled[i]) end
local test_rows = {}
for i=train_n+1,n do table.insert(test_rows, shuffled[i]) end

-- Build training data for regression
local train_k, train_e = {}, {}
for _,r in ipairs(train_rows) do
  local k = r[sz_col.at]
  local e = r[eff_col.at]
  if k ~= "?" and e ~= "?" then
    table.insert(train_k, k)
    table.insert(train_e, e)
  end
end

local model = linreg(train_k, train_e)

-- Historical average from training set
local avg_eff_train = mean(train_e)

-- Evaluate on test set
local pred_mae = {}
local avg_mae = {}
for _,r in ipairs(test_rows) do
  local k = r[sz_col.at]
  local e = r[eff_col.at]
  if k ~= "?" and e ~= "?" then
    local pred = model.a + model.b * k
    table.insert(pred_mae, pred - e)
    table.insert(avg_mae, avg_eff_train - e)
  end
end

local mae_regr = absmean(pred_mae)
local mae_avg  = absmean(avg_mae)

-- Percentage improvement of regression over average
local improvement = (mae_avg - mae_regr) / mae_avg * 100

print(string.format("Regression MAE      : %.2f", round(mae_regr)))
print(string.format("Historical Avg MAE  : %.2f", round(mae_avg)))
print(string.format("MAE improvement     : %.2f%%", round(improvement)))

-- ------------------------------------------------------------------
-- NASA-20: proposal selection comparison
-- ------------------------------------------------------------------
print("\n== NASA-20 ==")

local budget = math.max(5, math.floor(#rows * 0.05))

-- --- our selection ------------------------------------------------
local function compute_score(r)
  local sz = r[sz_col.at]
  local eff = r[eff_col.at]
  local def = r[def_col.at]
  if sz ~= "?" and eff ~= "?" and def ~= "?" then
    return (eff + def) / sz
  end
  return nil
end

local candidates = {}
for _,r in ipairs(rows) do
  local s = compute_score(r)
  if s then
    table.insert(candidates, {row=r, score=s})
  end
end

table.sort(candidates, function(a,b) return a.score < b.score end)
local chosen = {}
for i=1,budget do
  table.insert(chosen, candidates[i].row)
end

local function metrics(sel)
  local total_eff, total_def, total_sz = 0,0,0
  local scores = {}
  for _,r in ipairs(sel) do
    local sz = r[sz_col.at]
    local eff = r[eff_col.at]
    local def = r[def_col.at]
    total_eff = total_eff + eff
    total_def = total_def + def
    total_sz  = total_sz  + sz
    table.insert(scores, (eff+def)/sz)
  end
  local avg_score = mean(scores)
  return {
    avg_score = avg_score,
    total_eff = total_eff,
    total_def = total_def,
    total_sz  = total_sz
  }
end

local chosen_metrics = metrics(chosen)

-- --- random selection ---------------------------------------------
local shuffled_rows = shuffle(rows)
local random_sel = {}
for i=1,budget do
  table.insert(random_sel, shuffled_rows[i])
end
local random_metrics = metrics(random_sel)

-- --- output ---------------------------------------------------------
print(string.format("Budget (rows selected): %d", budget))
print("\nOur selection:")
print(string.format("  Avg (Eff+Def)/Size : %.3f", round(chosen_metrics.avg_score)))
print(string.format("  Total Effort       : %.1f", round(chosen_metrics.total_eff)))
print(string.format("  Total Defects      : %.1f", round(chosen_metrics.total_def)))
print("\nRandom selection:")
print(string.format("  Avg (Eff+Def)/Size : %.3f", round(random_metrics.avg_score)))
print(string.format("  Total Effort       : %.1f", round(random_metrics.total_eff)))
print(string.format("  Total Defects      : %.1f", round(random_metrics.total_def)))

local score_improvement = (random_metrics.avg_score - chosen_metrics.avg_score) / random_metrics.avg_score * 100
print("\nImprovement of our strategy over random (lower avg score better):")
print(string.format("  %.2f%% improvement in average (Eff+Def)/Size", round(score_improvement)))
