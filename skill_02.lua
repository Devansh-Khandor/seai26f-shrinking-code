#!/usr/bin/env lua
-- skill_02.lua
-- NASA-03: Predict effort from size
-- NASA-04: Predict defect count from size
-- Uses a simple linear‑regression on the historic data

local ezr = require("ezr")

local csv = ezr.csv
local Tbl = ezr.Tbl

-- ------------------------------------------------------------
-- Helper: find column index by name
local function colIndex(tbl, name)
  for at, c in ipairs(tbl.cols.all) do
    if c.name == name then
      return at
    end
  end
  error("Column '" .. name .. "' not found")
end

-- ------------------------------------------------------------
-- Simple linear regression (x -> y)
-- Returns slope, intercept, standard error, mean absolute error
local function linear_regression(x, y)
  local n = #x
  local sumx, sumy, sumxy, sumx2, sumy2 = 0, 0, 0, 0, 0
  for i = 1, n do
    local xi, yi = x[i], y[i]
    sumx  = sumx  + xi
    sumy  = sumy  + yi
    sumxy = sumxy + xi*yi
    sumx2 = sumx2 + xi*xi
    sumy2 = sumy2 + yi*yi
  end
  local meanx = sumx / n
  local meany = sumy / n
  local covxy = sumxy/n - meanx*meany
  local varx  = sumx2/n - meanx*meanx
  local vary  = sumy2/n - meany*meany

  local slope     = covxy / varx
  local intercept = meany - slope * meanx

  local se_sum, mae_sum = 0, 0
  for i = 1, n do
    local yhat = intercept + slope * x[i]
    local err  = y[i] - yhat
    se_sum    = se_sum + err*err
    mae_sum   = mae_sum + math.abs(err)
  end
  local se  = math.sqrt(se_sum / (n-2))
  local mae = mae_sum / n

  local corr = covxy / math.sqrt(varx * vary)

  return {
    slope   = slope,
    intercept = intercept,
    se     = se,
    mae    = mae,
    corr   = corr,
    n      = n
  }
end

-- ------------------------------------------------------------
local path = arg[1]
if not path then
  error("Usage: lua skill_02.lua <csv path>")
end

local tbl = Tbl(csv(path))

-- Column indices
local idxKloc   = colIndex(tbl, "Kloc+")
local idxEffort = colIndex(tbl, "Effort-")
local idxDefect = colIndex(tbl, "Defects-")

-- Extract vectors
local kloc = {}
local effort = {}
local defects = {}
for _, r in ipairs(tbl.rows) do
  local k = r[idxKloc]
  local e = r[idxEffort]
  local d = r[idxDefect]
  if type(k) == "number" and type(e) == "number" and type(d) == "number" then
    table.insert(kloc,    k)
    table.insert(effort,  e)
    table.insert(defects, d)
  end
end

-- Build regressions
local effort_mod = linear_regression(kloc, effort)
local defect_mod = linear_regression(kloc, defects)

-- ------------------------------------------------------------
print("== NASA-03 ==")
print(string.format(
  "Effort model: Effort = %.4f + %.4f * Kloc",
  effort_mod.intercept, effort_mod.slope))
print(string.format(
  "Correlation: %.3f   Standard Error: %.3f   MAE: %.3f",
  effort_mod.corr, effort_mod.se, effort_mod.mae))
print(string.format(
  "Model fits %d rows (n-2 degrees of freedom = %d)",
  effort_mod.n, effort_mod.n-2))
print()

print("== NASA-04 ==")
print(string.format(
  "Defect model: Defects = %.4f + %.4f * Kloc",
  defect_mod.intercept, defect_mod.slope))
print(string.format(
  "Correlation: %.3f   Standard Error: %.3f   MAE: %.3f",
  defect_mod.corr, defect_mod.se, defect_mod.mae))
print(string.format(
  "Model fits %d rows (n-2 degrees of freedom = %d)",
  defect_mod.n, defect_mod.n-2))
print()
