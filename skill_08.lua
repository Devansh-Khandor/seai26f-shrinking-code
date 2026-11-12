#!/usr/bin/env lua
-- skill_08.lua
-- NASA-15: Suggest process changes to reduce predicted effort
-- NASA-16: Suggest process changes to reduce predicted defect count
--
-- Run as:  lua skill_08.lua <csv path>
--
-- Reuse the linear‑regression and helper code from earlier skills.
-- The suggestions are based on the most common values of each
-- adjustable characteristic among projects that lie in the
-- lower‑effort / lower‑defect part of the data.

local ezr = require("ezr")
local csv = ezr.csv
local Tbl = ezr.Tbl

-- ----------------------------------------------------------------------
-- Helpers -------------------------------------------------------------
-- ----------------------------------------------------------------------
-- Find column index by name
local function colIndex(tbl, name)
  for at, c in ipairs(tbl.cols.all) do
    if c.name == name then return at end
  end
  error("Column '" .. name .. "' not found")
end

-- Simple linear regression (x -> y) – copy from skill_02
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
  return {slope = slope, intercept = intercept}
end

-- Median + 1.5 * IQR threshold (used for “high” groups)
local function threshold(vals)
  local n = #vals
  if n == 0 then return 0 end
  table.sort(vals)
  local q1 = vals[(n + 3) // 4]
  local q3 = vals[(3*n + 1) // 4]
  local iqr = q3 - q1
  return vals[(n + 1) // 2] + 1.5 * iqr
end

-- Mode for a list of values (ties: first seen)
local function mode(vals)
  local freq = {}
  local best, bestcnt = nil, -1
  for _, v in ipairs(vals) do
    freq[v] = (freq[v] or 0) + 1
    if freq[v] > bestcnt then
      bestcnt = freq[v]
      best = v
    end
  end
  return best
end

-- ----------------------------------------------------------------------
-- Load data ------------------------------------------------------------
-- ----------------------------------------------------------------------
local path = arg[1]
if not path then
  error("Usage: lua skill_08.lua <csv path>")
end

local tbl = Tbl(csv(path))

-- ----------------------------------------------------------------------
-- Identify target project (maximum predicted effort / defects) ----------
-- ----------------------------------------------------------------------
local idxKloc   = colIndex(tbl, "Kloc+")
local idxEffort = colIndex(tbl, "Effort-")
local idxDefect = colIndex(tbl, "Defects-")
local idxId     = colIndex(tbl, "idX")

-- Build regression models (as in skill_02)
local kloc = {}
local effort = {}
local defects = {}
for _, r in ipairs(tbl.rows) do
  local k = r[idxKloc]
  local e = r[idxEffort]
  local d = r[idxDefect]
  if type(k) == "number" and type(e) == "number" and type(d) == "number" then
    table.insert(kloc, k)
    table.insert(effort, e)
    table.insert(defects, d)
  end
end

local effort_mod = linear_regression(kloc, effort)
local defect_mod = linear_regression(kloc, defects)

-- Predictions for each row
local predEff = {}
local predDef = {}
for _, r in ipairs(tbl.rows) do
  local k = r[idxKloc]
  predEff[#predEff+1] = effort_mod.intercept + effort_mod.slope * k
  predDef[#predDef+1] = defect_mod.intercept + defect_mod.slope * k
end

-- Project with highest predicted effort
local maxEffIdx = 1
for i = 2, #predEff do
  if predEff[i] > predEff[maxEffIdx] then
    maxEffIdx = i
  end
end
local maxEffRow = tbl.rows[maxEffIdx]
local maxEffPred = predEff[maxEffIdx]

-- Project with highest predicted defects
local maxDefIdx = 1
for i = 2, #predDef do
  if predDef[i] > predDef[maxDefIdx] then
    maxDefIdx = i
  end
end
local maxDefRow = tbl.rows[maxDefIdx]
local maxDefPred = predDef[maxDefIdx]

-- ----------------------------------------------------------------------
-- Define low‑effort and low‑defect groups --------------------------------
-- ----------------------------------------------------------------------
local lowEffThreshold = threshold(predEff)
local lowDefThreshold = threshold(predDef)

local lowEffRows, lowDefRows = {}, {}
for i, r in ipairs(tbl.rows) do
  if predEff[i] <= lowEffThreshold then table.insert(lowEffRows, r) end
  if predDef[i] <= lowDefThreshold then table.insert(lowDefRows, r) end
end

-- ----------------------------------------------------------------------
-- Helper to compute target value for a column ---------------------------
-- ----------------------------------------------------------------------
local function targetValue(rows, colIdx)
  local vals = {}
  for _, r in ipairs(rows) do
    table.insert(vals, r[colIdx])
  end
  if vals[1] == nil then return nil end
  local t = type(vals[1])
  if t == "number" then
    local sum = 0
    for _, v in ipairs(vals) do sum = sum + v end
    return math.floor((sum / #vals) * 10 + 0.5) / 10  -- round 1 dp
  else
    return mode(vals)
  end
end

-- ----------------------------------------------------------------------
-- Build suggestions -----------------------------------------------------
-- ----------------------------------------------------------------------
-- Adjustable columns: all except idX, centerX, YearX, Kloc+, Effort-, Defects-, MonthsX
local adjustable = {}
for at, c in ipairs(tbl.cols.all) do
  if not (
        c.name == "idX" or c.name == "centerX" or c.name == "YearX" or
        c.name == "Kloc+" or c.name == "Effort-" or
        c.name == "Defects-" or c.name == "MonthsX"
     ) then
    table.insert(adjustable, c)
  end
end

-- Suggestions for effort (NASA‑15)
local effortSuggs = {}
for _, c in ipairs(adjustable) do
  local val = targetValue(lowEffRows, c.at)
  if val ~= nil then
    table.insert(effortSuggs, {col=c, target=val})
  end
end

-- Suggestions for defects (NASA‑16)
local defectSuggs = {}
for _, c in ipairs(adjustable) do
  local val = targetValue(lowDefRows, c.at)
  if val ~= nil then
    table.insert(defectSuggs, {col=c, target=val})
  end
end

-- Keep only first three suggestions (at least three required)
if #effortSuggs > 3 then effortSuggs = {effortSuggs[1], effortSuggs[2], effortSuggs[3]} end
if #defectSuggs > 3 then defectSuggs = {defectSuggs[1], defectSuggs[2], defectSuggs[3]} end

-- ----------------------------------------------------------------------
-- Output ---------------------------------------------------------------
-- ----------------------------------------------------------------------
print("== NASA-15 ==")
print(string.format(
  "Project %s predicted effort: %.2f hrs (max among all projects)",
  maxEffRow[idxId], maxEffPred))
print("Suggested process adjustments to reduce effort:")

for _, s in ipairs(effortSuggs) do
  local curr = maxEffRow[s.col.at]
  print(string.format(
    "• Change %s from %s to %s",
    s.col.name, tostring(curr), tostring(s.target)))
end

print("\n== NASA-16 ==")
print(string.format(
  "Project %s predicted defects: %d (max among all projects)",
  maxDefRow[idxId], math.floor(maxDefPred + 0.5)))
print("Suggested process adjustments to improve quality:")

for _, s in ipairs(defectSuggs) do
  local curr = maxDefRow[s.col.at]
  print(string.format(
    "• Change %s from %s to %s",
    s.col.name, tostring(curr), tostring(s.target)))
end
