#!/usr/bin/env lua
-- skill_09.lua
-- NASA-17: Flag projects with unusually distinctive feature combinations
-- NASA-18: Flag projects whose effort or defect counts are anomalous
--
-- Run as:  lua skill_09.lua <csv path>

local ezr = require("ezr")
local csv = ezr.csv
local Tbl = ezr.Tbl

-- ----------------------------------------------------------------------
-- Helpers (copied from earlier skills for reuse)
-- ----------------------------------------------------------------------
local function colIndex(tbl, name)
  for at, c in ipairs(tbl.cols.all) do
    if c.name == name then return at end
  end
  error("Column '" .. name .. "' not found")
end

local function threshold(vals)
  local n = #vals
  if n == 0 then return 0 end
  table.sort(vals)
  local q1 = vals[(n + 3) // 4]
  local q3 = vals[(3*n + 1) // 4]
  local iqr = q3 - q1
  return vals[(n + 1) // 2] + 1.5 * iqr
end

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

-- ----------------------------------------------------------------------
-- Load data
-- ----------------------------------------------------------------------
local path = arg[1]
if not path then
  error("Usage: lua skill_09.lua <csv path>")
end

local tbl = Tbl(csv(path))

-- Column indices
local idxId      = colIndex(tbl, "idX")
local idxEffort  = colIndex(tbl, "Effort-")
local idxDefect  = colIndex(tbl, "Defects-")

-- ----------------------------------------------------------------------
-- Build regression models for effort and defects
-- ----------------------------------------------------------------------
local kloc = {}
local effort = {}
local defects = {}
for _, r in ipairs(tbl.rows) do
  local k = r[idxEffort]   -- Kloc+ is the first numeric feature
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

-- ----------------------------------------------------------------------
-- NASA-18: Residual outliers for effort and defects
-- ----------------------------------------------------------------------
local effort_residuals = {}
local defect_residuals = {}
for _, r in ipairs(tbl.rows) do
  local k = r[idxEffort]
  local e = r[idxEffort]
  local d = r[idxDefect]
  if type(k) == "number" and type(e) == "number" and type(d) == "number" then
    local pred_e = effort_mod.intercept + effort_mod.slope * k
    local pred_d = defect_mod.intercept + defect_mod.slope * k
    table.insert(effort_residuals, e - pred_e)
    table.insert(defect_residuals, d - pred_d)
  end
end

local effortThresh = threshold(effort_residuals)
local defectThresh = threshold(defect_residuals)

local flaggedEffort = {}
local flaggedDefect = {}
for _, r in ipairs(tbl.rows) do
  local k = r[idxEffort]
  local e = r[idxEffort]
  local d = r[idxDefect]
  if type(k) == "number" and type(e) == "number" and type(d) == "number" then
    local pred_e = effort_mod.intercept + effort_mod.slope * k
    local pred_d = defect_mod.intercept + defect_mod.slope * k
    local diff_e = e - pred_e
    local diff_d = d - pred_d
    if math.abs(diff_e) > effortThresh then
      table.insert(flaggedEffort, {
        id = r[idxId],
        eff = e,
        pred = pred_e,
        diff = diff_e
      })
    end
    if math.abs(diff_d) > defectThresh then
      table.insert(flaggedDefect, {
        id = r[idxId],
        def = d,
        pred = pred_d,
        diff = diff_d
      })
    end
  end
end

-- ----------------------------------------------------------------------
-- NASA-17: Feature‑space outliers (average distance to others)
-- ----------------------------------------------------------------------
-- Precompute all rows for distance calculations
local function avgDist(row, others)
  local sum, cnt = 0, 0
  for _, o in ipairs(others) do
    if o ~= row then
      sum = sum + tbl:distx(row, o)
      cnt = cnt + 1
    end
  end
  return cnt > 0 and sum / cnt or 0
end

local avgDists = {}
for _, r in ipairs(tbl.rows) do
  avgDists[#avgDists + 1] = avgDist(r, tbl.rows)
end

-- Find mean & SD of avgDists
local function mean(arr)
  local sum, n = 0, #arr
  for _, v in ipairs(arr) do sum = sum + v end
  return sum / n
end

local function std(arr, mu)
  local sum, n = 0, #arr
  for _, v in ipairs(arr) do sum = sum + (v - mu)^2 end
  return math.sqrt(sum / (n - 1))
end

local muAvg = mean(avgDists)
local sdAvg = std(avgDists, muAvg)
local distThresh = muAvg + 2 * sdAvg

local flaggedDist = {}
for i, r in ipairs(tbl.rows) do
  if avgDists[i] > distThresh then
    table.insert(flaggedDist, {
      id = r[idxId],
      dist = avgDists[i]
    })
  end
end

-- ----------------------------------------------------------------------
-- Output results
-- ----------------------------------------------------------------------
print("== NASA-17 ==")
if #flaggedDist == 0 then
  print("No unusually distinctive projects found.")
else
  for i, f in ipairs(flaggedDist) do
    print(string.format("%d. id=%s: avg. distance %.3f", i, f.id, f.dist))
    if i >= 3 then break end
  end
end

print("\n== NASA-18 ==")
print("Effort outliers:")
if #flaggedEffort == 0 then
  print("  None.")
else
  for i, f in ipairs(flaggedEffort) do
    print(string.format("%d. id=%s: actual %.2f, pred %.2f, diff %.2f",
          i, f.id, f.eff, f.pred, f.diff))
    if i >= 3 then break end
  end
end

print("\nDefect outliers:")
if #flaggedDefect == 0 then
  print("  None.")
else
  for i, f in ipairs(flaggedDefect) do
    print(string.format("%d. id=%s: actual %d, pred %.2f, diff %.2f",
          i, f.id, f.def, f.pred, f.diff))
    if i >= 3 then break end
  end
end
