#!/usr/bin/env lua
--[[
  skill_02.lua
  NASA-93 data analysis – effort and defect estimation.
  Usage:  lua skill_02.lua <csv-path>
  Prints:
    NASA-03: regression model for Effort- vs. Kloc+
    NASA-04: regression model for Defects- vs. Kloc+
  Each section shows the fitted line, R², residual standard deviation
  and a 95 % prediction interval width.
]]--

local ezr = require "ezr"

-- ---- Helper -----------------------------------------------------------------
local function abort(msg)
  io.stderr:write(msg .. "\n")
  os.exit(1)
end

local function round(v)
  return ezr.round(v)
end

local function linreg(x, y)
  local n = #x
  if n < 2 then abort("Not enough data points for regression") end

  local sx, sx2, sy, sy2, sxy = 0, 0, 0, 0, 0
  for i = 1, n do
    local xi, yi = x[i], y[i]
    sx  = sx + xi
    sx2 = sx2 + xi * xi
    sy  = sy + yi
    sy2 = sy2 + yi * yi
    sxy = sxy + xi * yi
  end

  local mean_x = sx / n
  local mean_y = sy / n
  local var_x  = sx2 / n - mean_x * mean_x
  local var_y  = sy2 / n - mean_y * mean_y
  local cov_xy = sxy / n - mean_x * mean_y

  local b = cov_xy / var_x
  local a = mean_y - b * mean_x
  local r2 = (cov_xy * cov_xy) / (var_x * var_y)

  -- residual standard deviation
  local ss_res = 0
  for i = 1, n do
    local yi_hat = a + b * x[i]
    ss_res = ss_res + (y[i] - yi_hat) * (y[i] - yi_hat)
  end
  local residual_std = math.sqrt(ss_res / (n - 2))

  return {
    a = a, b = b, r2 = r2, residual_std = residual_std,
    mean_x = mean_x, mean_y = mean_y, var_x = var_x
  }
end

-- ---- Main --------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then abort("Usage: lua skill_02.lua <csv-path>") end

-- Load all rows
local rows_iter = ezr.csv(csv_path)
local rows = {}
for r in rows_iter do table.insert(rows, r) end
if #rows == 0 then abort("No data found in " .. csv_path) end

-- Build table structure and map column names to indices
local tbl = ezr.Tbl(rows)
local col_map = {}
for i, c in ipairs(tbl.cols.all) do
  col_map[tbl.cols.names[i]] = c
end

-- Indices for the columns we need
local idx_kloc = col_map["Kloc+"].at
local idx_eff  = col_map["Effort-"].at
local idx_def  = col_map["Defects-"].at

-- Collect data for the two regressions
local x_eff, y_eff = {}, {}
local x_def, y_def = {}, {}

for _, r in ipairs(rows) do
  local k = r[idx_kloc]
  local e = r[idx_eff]
  local d = r[idx_def]

  if k ~= "?" and e ~= "?" then
    table.insert(x_eff, k)
    table.insert(y_eff, e)
  end

  if k ~= "?" and d ~= "?" then
    table.insert(x_def, k)
    table.insert(y_def, d)
  end
end

-- Fit regressions
local eff_model = linreg(x_eff, y_eff)
local def_model = linreg(x_def, y_def)

-- ---- Output NASA‑03 ---------------------------------------------------------
print("== NASA-03 ==")
print(string.format(
  "Effort-  = %.4f + %.4f * Kloc+",
  round(eff_model.a), round(eff_model.b)
))
print(string.format(
  "R²       = %.4f",
  round(eff_model.r2)
))
print(string.format(
  "Residual SD   = %.4f",
  round(eff_model.residual_std)
))
print(string.format(
  "95%% Prediction SD = %.4f",
  round(1.96 * eff_model.residual_std)
))
print()

-- ---- Output NASA‑04 ---------------------------------------------------------
print("== NASA-04 ==")
print(string.format(
  "Defects- = %.4f + %.4f * Kloc+",
  round(def_model.a), round(def_model.b)
))
print(string.format(
  "R²       = %.4f",
  round(def_model.r2)
))
print(string.format(
  "Residual SD   = %.4f",
  round(def_model.residual_std)
))
print(string.format(
  "95%% Prediction SD = %.4f",
  round(1.96 * def_model.residual_std)
))
