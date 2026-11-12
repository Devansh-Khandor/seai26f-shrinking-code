#!/usr/bin/env lua
-- skill_01.lua
-- Summarises historical NASA software projects and finds
-- projects with unusually favourable size/effort/defects.

local ezr = require("ezr")

local csv = ezr.csv
local Tbl = ezr.Tbl

local path = arg[1]
if not path then
  error("Usage: lua skill_01.lua <csv path>")
end

local tbl = Tbl(csv(path))

-- Find column indices by name
local function colIndex(tbl, name)
  for at, c in ipairs(tbl.cols.all) do
    if c.name == name then
      return at
    end
  end
  error("Column '" .. name .. "' not found")
end

local idxKloc  = colIndex(tbl, "Kloc+")
local idxEffort = colIndex(tbl, "Effort-")
local idxDefects = colIndex(tbl, "Defects-")
local idxId = colIndex(tbl, "idX")

-- Extract numeric values from a column
local function colValues(tbl, idx)
  local vals = {}
  for _, r in ipairs(tbl.rows) do
    local v = r[idx]
    if type(v) == "number" then
      table.insert(vals, v)
    end
  end
  return vals
end

-- Basic statistics: mean, median, min, max
local function stats(vals)
  local n = #vals
  local sum = 0
  local minv, maxv = vals[1], vals[1]
  for _, v in ipairs(vals) do
    sum = sum + v
    if v < minv then minv = v end
    if v > maxv then maxv = v end
  end
  local mean = sum / n
  local sorted = clone(vals)
  table.sort(sorted)
  local median = sorted[(n + 1) // 2]   -- integer division
  return {mean = mean, median = median, min = minv, max = maxv}
end

local valuesK = colValues(tbl, idxKloc)
local valuesE = colValues(tbl, idxEffort)
local valuesD = colValues(tbl, idxDefects)

local statsK = stats(valuesK)
local statsE = stats(valuesE)
local statsD = stats(valuesD)

-- Best/worst rows by effort
local function bestWorst(tbl, idx)
  local sorted = clone(tbl.rows)
  table.sort(sorted, function(a, b)
    return a[idx] < b[idx]
  end)
  return sorted[1], sorted[#sorted]
end

local bestRow, worstRow = bestWorst(tbl, idxEffort)

-- Identify projects with effort <= median and defects <= median
local favEffortThreshold = statsE.median
local favDefectThreshold = statsD.median
local favorable = {}
for _, r in ipairs(tbl.rows) do
  if r[idxEffort] <= favEffortThreshold and r[idxDefects] <= favDefectThreshold then
    table.insert(favorable, r)
  end
end

-- ---------- Output ----------
print("== NASA-01 ==")
print(string.format(
  "Size (KLOC)   mean: %.2f  median: %.2f  min: %d  max: %d",
  statsK.mean, statsK.median, math.floor(statsK.min), math.ceil(statsK.max)))
print(string.format(
  "Effort (hrs)  mean: %.2f  median: %.2f  min: %.2f  max: %.2f",
  statsE.mean, statsE.median, statsE.min, statsE.max))
print(string.format(
  "Defects       mean: %.2f  median: %.2f  min: %d  max: %d",
  statsD.mean, statsD.median, statsD.min, statsD.max))
print(string.format(
  "Best outcome (lowest effort) id: %s (effort %.2f)",
  bestRow[idxId], bestRow[idxEffort]))
print(string.format(
  "Worst outcome (highest effort) id: %s (effort %.2f)",
  worstRow[idxId], worstRow[idxEffort]))

print()
print("== NASA-02 ==")
print("Projects with low effort and low defects (<= median):")
for _, r in ipairs(favorable) do
  print(string.format(
    "id %s: size %.1f KLOC, effort %.1f hrs, defects %d",
    r[idxId], r[idxKloc], r[idxEffort], r[idxDefects]))
end
