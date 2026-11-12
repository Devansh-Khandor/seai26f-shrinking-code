#!/usr/bin/env lua
--[[
  skill_04.lua
  NASA-93 data analysis – continuous assessment and dynamic update.
  Usage:  lua skill_04.lua <csv-path>
  Prints:
    NASA-07: flag proposals whose predicted effort or defects
             are unusually high compared to the current historical
             distribution.
    NASA-08: updated regression models and summary statistics
             that incorporate the latest data set.
]]--

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Helpers
-- ------------------------------------------------------------------
local function abort(msg)
  io.stderr:write(msg .. "\n")
  os.exit(1)
end

local function round(v)
  return ezr.round(v)
end

-- Simple linear regression (least squares)
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
  local var_y  = sy2 / n - mean_y*mean_y
  local cov_xy = sxy / n - mean_x*mean_y
  local b = cov_xy / var_x
  local a = mean_y - b * mean_x
  -- residual standard deviation
  local ss_res = 0
  for i = 1, n do
    local yi_hat = a + b * x[i]
    ss_res = ss_res + (y[i] - yi_hat)^2
  end
  local residual_std = math.sqrt(ss_res / (n - 2))
  return {a=a,b=b,mean_x=mean_x,mean_y=mean_y,
          var_x=var_x,var_y=var_y, residual_std=residual_std}
end

-- mean & std for numeric array
local function mean_std(vals)
  local n = #vals
  if n == 0 then return 0,0 end
  local sum, sumsq = 0,0
  for _,v in ipairs(vals) do sum = sum + v; sumsq = sumsq + v*v end
  local mean = sum / n
  local var = (sumsq - sum*mean)/(n-1) or 0
  local std = math.sqrt(var)
  return mean,std
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then abort("Usage: lua skill_04.lua <csv-path>") end

-- Load rows
local rows_iter = ezr.csv(csv_path)
local rows = {}
for r in rows_iter do table.insert(rows,r) end
if #rows == 0 then abort("No data found in " .. csv_path) end

-- Build table structure
local tbl = ezr.Tbl(rows)

-- Map column names to indices
local col_map = {}
for i,c in ipairs(tbl.cols.all) do
  col_map[tbl.cols.names[i]] = c
end

local idx_kloc = col_map["Kloc+"].at
local idx_eff  = col_map["Effort-"].at
local idx_def  = col_map["Defects-"].at

-- Prepare data for regressions
local x_eff, y_eff = {}, {}
local x_def, y_def = {}, {}

for _,r in ipairs(rows) do
  local k = r[idx_kloc]
  local e = r[idx_eff]
  local d = r[idx_def]
  if k ~= "?" and e ~= "?" then
    table.insert(x_eff,k); table.insert(y_eff,e)
  end
  if k ~= "?" and d ~= "?" then
    table.insert(x_def,k); table.insert(y_def,d)
  end
end

-- Fit models
local eff_model = linreg(x_eff, y_eff)
local def_model = linreg(x_def, y_def)

-- Predict for every row (ignore rows with missing size)
local preds_eff, preds_def = {}, {}
for _,r in ipairs(rows) do
  local k = r[idx_kloc]
  if k ~= "?" then
    local peff = eff_model.a + eff_model.b * k
    local pdef = def_model.a + def_model.b * k
    table.insert(preds_eff, peff)
    table.insert(preds_def, pdef)
  end
end

-- Summary of predictions
local mean_pred_eff, std_pred_eff = mean_std(preds_eff)
local mean_pred_def, std_pred_def = mean_std(preds_def)

local thresh_eff = mean_pred_eff + 1.96 * std_pred_eff
local thresh_def = mean_pred_def + 1.96 * std_pred_def

-- ------------------------------------------------------------------
-- NASA-07: flag unusually high predictions
-- ------------------------------------------------------------------
print("== NASA-07 ==")
print("ID Size PredEff PredDef ActEff ActDef AlertEff AlertDef")
local any_alert = false
for _,r in ipairs(rows) do
  local id   = r[1]
  local k    = r[idx_kloc]
  local eff  = r[idx_eff]
  local def  = r[idx_def]
  if k ~= "?" then
    local peff = eff_model.a + eff_model.b * k
    local pdef = def_model.a + def_model.b * k
    local alert_eff = (peff > thresh_eff)
    local alert_def = (pdef > thresh_def)
    if alert_eff or alert_def then
      any_alert = true
      print(string.format("%-5s %5s %6s %6s %5s %5s %8s %8s",
             id, round(k), round(peff), round(pdef),
             eff and round(eff) or "?", def and round(def) or "?",
             alert_eff and "Y" or "N",
             alert_def and "Y" or "N"))
    end
  end
end
if not any_alert then
  print("No proposals have unusually high predicted effort or defects.")
end

-- ------------------------------------------------------------------
-- NASA-08: updated models & summary statistics
-- ------------------------------------------------------------------
print("\n== NASA-08 ==")
print(string.format("Effort- model: a = %s, b = %s", round(eff_model.a), round(eff_model.b)))
print(string.format("Defects- model: a = %s, b = %s", round(def_model.a), round(def_model.b)))
print()
print(string.format("Mean predicted Effort: %s, Std: %s", round(mean_pred_eff), round(std_pred_eff)))
print(string.format("Mean predicted Defects: %s, Std: %s", round(mean_pred_def), round(std_pred_def)))
print()

-- Also give updated raw data summaries
local function stats(col)
  local vals = {}
  for _,r in ipairs(rows) do
    local v = r[col.at]
    if v ~= "?" then table.insert(vals,v) end
  end
  local n = #vals
  local sum = 0
  for _,v in ipairs(vals) do sum = sum + v end
  local mean = n>0 and sum/n or 0
  local var = 0
  for _,v in ipairs(vals) do var = var + (v-mean)^2 end
  local std = n>1 and math.sqrt(var/(n-1)) or 0
  return n, mean, std
end

local n_kloc, m_kloc, s_kloc = stats(col_map["Kloc+"])
local n_eff,  m_eff,  s_eff  = stats(col_map["Effort-"])
local n_def,  m_def,  s_def  = stats(col_map["Defects-"])

print(string.format("Kloc+: n=%d mean=%s std=%s", n_kloc, round(m_kloc), round(s_kloc)))
print(string.format("Effort-: n=%d mean=%s std=%s", n_eff, round(m_eff), round(s_eff)))
print(string.format("Defects-: n=%d mean=%s std=%s", n_def, round(m_def), round(s_def)))
