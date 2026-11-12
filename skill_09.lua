#!/usr/bin/env lua
-- skill_09.lua
-- APACHE-17 & APACHE-18: Outlier analysis
--
-- Usage:  lua skill_09.lua <path-to-Apache_AllMeasurements.csv>
--
-- Requires only the ezr library (ezr.lua & ezr-lib.lua) in the same directory.

local ezr = require("ezr")
local csv  = ezr.csv
local Tbl   = ezr.Tbl
local sorted = ezr.sorted
local keysort = ezr.keysort
local map = ezr.map
local copy = ezr.copy
local rand = ezr.rand

-- ------------------------------------------------------------------
-- Load CSV into a TBL
-- ------------------------------------------------------------------
local function load_tbl(path)
  return Tbl(csv(path))
end

-- ------------------------------------------------------------------
-- k‑nearest‑neighbour prediction (k=5)
-- ------------------------------------------------------------------
local function estimate_knn(tbl, data, row, k)
  k = k or 5
  local dists = {}
  for _, r in ipairs(data) do
    if r ~= row then
      local d = tbl:distx(row, r)
      table.insert(dists, {d=d, perf=r[tbl.cols.y[1].at]})
    end
  end
  table.sort(dists, function(a,b) return a.d < b.d end)
  local n = math.min(k, #dists)
  if n == 0 then return nil, nil end
  local sum, sq = 0,0
  for i=1,n do
    local v = dists[i].perf
    sum = sum + v
    sq = sq + v*v
  end
  local mean = sum / n
  local var = (sq / n) - mean*mean
  local sd = math.sqrt(math.max(var, 0))
  return mean, sd
end

-- ------------------------------------------------------------------
-- Compute residuals between actual and kNN predicted performance
-- ------------------------------------------------------------------
local function compute_residuals(tbl, data)
  local res = {}
  for _, r in ipairs(data) do
    local act = r[tbl.cols.y[1].at]
    if act ~= "?" then
      local pred, _ = estimate_knn(tbl, data, r)
      if pred then
        local diff = act - pred
        table.insert(res, {row=r, act=act, pred=pred, diff=diff})
      end
    end
  end
  return res
end

-- ------------------------------------------------------------------
-- Find nearest‑neighbour distance for every row
-- ------------------------------------------------------------------
local function compute_nearest_distances(tbl, data)
  local dists = {}
  for i, r in ipairs(data) do
    local min = math.huge
    for j, s in ipairs(data) do
      if i ~= j then
        local d = tbl:distx(r, s)
        if d < min then min = d end
      end
    end
    table.insert(dists, {row=r, dist=min, idx=i})
  end
  return dists
end

-- ------------------------------------------------------------------
-- Statistical helpers
-- ------------------------------------------------------------------
local function mean_and_sd(values)
  local n = #values
  if n == 0 then return 0,0 end
  local sum, sq = 0,0
  for _, v in ipairs(values) do
    sum = sum + v
    sq = sq + v*v
  end
  local mu = sum / n
  local var = (sq / n) - mu*mu
  return mu, math.sqrt(math.max(var, 0))
end

-- ------------------------------------------------------------------
-- Print row as comma‑separated string
-- ------------------------------------------------------------------
local function row_to_string(row)
  local parts = {}
  for _, v in ipairs(row) do
    table.insert(parts, tostring(v))
  end
  return table.concat(parts, ", ")
end

-- ------------------------------------------------------------------
-- APACHE‑17: Performance outliers
-- ------------------------------------------------------------------
local function apache17(tbl, data)
  print("== APACHE-17 ==")
  local res = compute_residuals(tbl, data)
  local diffs = map(res, function(r) return math.abs(r.diff) end)
  local mean, sd = mean_and_sd(diffs)
  local threshold = mean + 1.5 * sd

  -- Select rows exceeding threshold
  local outliers = {}
  for _, r in ipairs(res) do
    if math.abs(r.diff) > threshold then
      table.insert(outliers, r)
    end
  end
  table.sort(outliers, function(a,b) return math.abs(a.diff) > math.abs(b.diff) end)

  local limit = math.min(#outliers, 10)
  if limit == 0 then
    print("  No performance outliers detected.")
    return
  end
  for i=1,limit do
    local r = outliers[i]
    local idx = table.find(tbl.rows, r.row)
    if not idx then idx = "?"; end
    print(string.format("  %d: idx=%3d  actual=%4d  pred=%6.2f  diff=%6.2f",
                        i, idx, r.act, r.pred, r.diff))
  end
  print()
end

-- ------------------------------------------------------------------
-- APACHE‑18: Configuration dissimilarity outliers
-- ------------------------------------------------------------------
local function apache18(tbl, data)
  print("== APACHE-18 ==")
  local dists = compute_nearest_distances(tbl, data)
  local dist_vals = map(dists, "dist")
  local mean, sd = mean_and_sd(dist_vals)
  local threshold = mean + 1.5 * sd

  local outliers = {}
  for _, r in ipairs(dists) do
    if r.dist > threshold then
      table.insert(outliers, r)
    end
  end
  table.sort(outliers, function(a,b) return a.dist > b.dist end)

  local limit = math.min(#outliers, 10)
  if limit == 0 then
    print("  No highly dissimilar configurations detected.")
    return
  end
  for i=1,limit do
    local r = outliers[i]
    local idx = r.idx
    print(string.format("  %d: idx=%3d  dist=%6.2f  row=%s",
                        i, idx, r.dist, row_to_string(r.row)))
  end
  print()
end

-- ------------------------------------------------------------------
-- Main entry point
-- ------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Error: CSV file path required\n")
    os.exit(1)
  end

  local tbl = load_tbl(path)
  local data = tbl.rows

  apache17(tbl, data)
  apache18(tbl, data)
end

main()
