#!/usr/bin/env lua
-- skill_04.lua
-- NASA-07: Assess new proposals, flag unusually high predictions
-- NASA-08: Update historical summaries & models with new records
--
-- Run as:  lua skill_04.lua <csv path>
--
-- The script reads the CSV, builds linear models for effort and defects
-- (as in skill_02), then applies these models to each record.  Any
-- prediction that is unusually high (median + 1.5 * IQR) is marked as an
-- alert.  After the assessment the script prints updated statistics
-- and model summaries so that future assessments will use the most
-- recent data.

local ezr = require("ezr")
local csv = ezr.csv
local Tbl = ezr.Tbl

-- ----------------------------------------------------------------------
-- Utility: find column index by name
local function colIndex(tbl, name)
  for at, c in ipairs(tbl.cols.all) do
    if c.name == name then return at end
  end
  error("Column '" .. name .. "' not found")
end

-- ----------------------------------------------------------------------
-- Simple linear regression (x -> y)
-- Returns slope, intercept, standard error, mean absolute error, corr, n
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
    slope      = slope,
    intercept  = intercept,
    se         = se,
    mae        = mae,
    corr       = corr,
    n          = n
  }
end

-- ----------------------------------------------------------------------
-- Basic statistics: mean, median, min, max
local function stats(vals)
  local n = #vals
  if n == 0 then return {mean = 0, median = 0, min = 0, max = 0} end
  local sum = 0
  local minv, maxv = vals[1], vals[1]
  for _, v in ipairs(vals) do
    sum = sum + v
    if v < minv then minv = v end
    if v > maxv then maxv = v end
  end
  local mean = sum / n
  local sorted = {}
  for i = 1, n do sorted[i] = vals[i] end
  table.sort(sorted)
  local median = sorted[(n + 1) // 2]
  return {mean = mean, median = median, min = minv, max = maxv}
end

-- ----------------------------------------------------------------------
-- ----------------------------------------------------------------------
-- MAIN
-- ----------------------------------------------------------------------
local path = arg[1]
if not path then
  error("Usage: lua skill_04.lua <csv path>")
end

-- Load historical data
local tbl = Tbl(csv(path))

-- Column indices
local idxKloc   = colIndex(tbl, "Kloc+")
local idxEffort = colIndex(tbl, "Effort-")
local idxDefect = colIndex(tbl, "Defects-")
local idxId     = colIndex(tbl, "idX")
local idxYear   = colIndex(tbl, "YearX")   -- may be used for filtering

-- Extract numeric vectors
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

-- Build models
local effort_mod = linear_regression(kloc, effort)
local defect_mod = linear_regression(kloc, defects)

-- ----------------------------------------------------------------------
-- Compute thresholds for unusual predictions
-- Median + 1.5 * IQR of predictions
local function threshold(preds)
  local n = #preds
  if n == 0 then return 0 end
  table.sort(preds)
  local q1 = preds[(n + 3) // 4]
  local q3 = preds[(3*n + 1) // 4]
  local iqr = q3 - q1
  return preds[(n + 1) // 2] + 1.5 * iqr
end

-- Generate predictions
local eff_preds = {}
local def_preds = {}
for _, r in ipairs(tbl.rows) do
  local k = r[idxKloc]
  table.insert(eff_preds, effort_mod.intercept + effort_mod.slope * k)
  table.insert(def_preds, defect_mod.intercept + defect_mod.slope * k)
end

local eff_thresh = threshold(eff_preds)
local def_thresh = threshold(def_preds)

-- ----------------------------------------------------------------------
-- NASA-07: Assess each proposal
print("== NASA-07 ==")
print(string.format(
  "Effort threshold (median + 1.5 IQR): %.1f hrs",
  eff_thresh))
print(string.format(
  "Defects threshold (median + 1.5 IQR): %.1f",
  def_thresh))
print("")

for i, r in ipairs(tbl.rows) do
  local id = r[idxId]
  local k = r[idxKloc]
  local eff_actual = r[idxEffort]
  local def_actual = r[idxDefect]

  local eff_pred = effort_mod.intercept + effort_mod.slope * k
  local def_pred = defect_mod.intercept + defect_mod.slope * k

  local alerts = {}
  if eff_pred > eff_thresh then table.insert(alerts, "EFFORT") end
  if def_pred > def_thresh then table.insert(alerts, "DEFS") end
  local alert_text = alerts[1] and ("[ALERT: " .. table.concat(alerts, ", ") .. "]") or ""

  print(string.format(
    "Project %s: Kloc=%.1f | Effort: predicted %.1f (actual %.1f) | Defects: predicted %.1f (actual %d) %s",
    id, k, eff_pred, eff_actual, def_pred, def_actual, alert_text))
end

-- ----------------------------------------------------------------------
-- NASA-08: Update historical summaries & models
print("\n== NASA-08 ==")

-- Updated stats
local statsK = stats(kloc)
local statsE = stats(effort)
local statsD = stats(defects)

print(string.format(
  "Size (KLOC)   mean: %.2f  median: %.2f  min: %.0f  max: %.0f",
  statsK.mean, statsK.median, statsK.min, statsK.max))
print(string.format(
  "Effort (hrs)  mean: %.2f  median: %.2f  min: %.1f  max: %.1f",
  statsE.mean, statsE.median, statsE.min, statsE.max))
print(string.format(
  "Defects       mean: %.2f  median: %.2f  min: %d  max: %d",
  statsD.mean, statsD.median, statsD.min, statsD.max))

print("\nEffort model:")
print(string.format(
  "Effort = %.4f + %.4f * Kloc",
  effort_mod.intercept, effort_mod.slope))
print(string.format(
  "Correlation: %.3f   Standard Error: %.3f   MAE: %.3f",
  effort_mod.corr, effort_mod.se, effort_mod.mae))
print(string.format(
  "Model fits %d rows (n-2 degrees of freedom = %d)",
  effort_mod.n, effort_mod.n-2))

print("\nDefect model:")
print(string.format(
  "Defects = %.4f + %.4f * Kloc",
  defect_mod.intercept, defect_mod.slope))
print(string.format(
  "Correlation: %.3f   Standard Error: %.3f   MAE: %.3f",
  defect_mod.corr, defect_mod.se, defect_mod.mae))
print(string.format(
  "Model fits %d rows (n-2 degrees of freedom = %d)",
  defect_mod.n, defect_mod.n-2))
