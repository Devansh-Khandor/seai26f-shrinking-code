#!/usr/bin/env lua
-- skill_02.lua
-- Estimate performance of an Apache configuration and compare
-- two configurations that differ by a single setting.
--
-- Requires the ezr library (ezr.lua & ezr-lib.lua) in the same directory.
--
-- Run: lua skill_02.lua <path-to-Apache_AllMeasurements.csv>
--
local ezr = require("ezr")
local csv  = ezr.csv
local TBL  = ezr.Tbl
local sorted = ezr.sorted
local keysort = ezr.keysort
local map = ezr.map
local rand = ezr.rand
local table = table

-- ------------------------------------------------------------------
-- Load data
-- ------------------------------------------------------------------
local function load_data(path)
  local tbl = TBL(csv(path))
  -- Keep a list of rows that have a numeric performance value
  local data = {}
  for _, r in ipairs(tbl.rows) do
    if r[tbl.cols.y[1].at] ~= "?" then
      table.insert(data, r)
    end
  end
  return tbl, data
end

-- ------------------------------------------------------------------
-- Distance between two rows (ignoring the performance column)
-- ------------------------------------------------------------------
local function dist(tbl, r1, r2)
  return tbl:distx(r1, r2)
end

-- ------------------------------------------------------------------
-- Estimate performance of a row using k-nearest neighbours
-- ------------------------------------------------------------------
local function estimate_perf(tbl, data, row, k)
  local dists = {}
  for _, r in ipairs(data) do
    -- skip identical row objects
    if r ~= row then
      table.insert(dists, {d = dist(tbl, row, r), perf = r[tbl.cols.y[1].at]})
    end
  end
  table.sort(dists, function(a,b) return a.d < b.d end)
  local neighbours = {}
  for i = 1, math.min(k, #dists) do
    table.insert(neighbours, dists[i].perf)
  end
  if #neighbours == 0 then return nil, nil end
  local n = #neighbours
  local sum, sumsq = 0, 0
  for _, v in ipairs(neighbours) do
    sum = sum + v
    sumsq = sumsq + v*v
  end
  local mean = sum / n
  local var = (sumsq / n) - mean*mean
  local sd = math.sqrt(math.max(var, 0))
  return mean, sd
end

-- ------------------------------------------------------------------
-- Find a pair of rows that differ in exactly one setting
-- ------------------------------------------------------------------
local function find_one_diff_pair(tbl, data)
  local cols = tbl.cols.x
  local function diff_count(r1, r2)
    local c = 0
    for _, col in ipairs(cols) do
      if r1[col.at] ~= r2[col.at] then
        c = c + 1
        if c > 1 then return c end
      end
    end
    return c
  end

  for i = 1, #data do
    for j = i+1, #data do
      if diff_count(data[i], data[j]) == 1 then
        return data[i], data[j]
      end
    end
  end
  return nil, nil
end

-- ------------------------------------------------------------------
-- Pretty print a row (excluding performance)
-- ------------------------------------------------------------------
local function row_to_string(tbl, row)
  local parts = {}
  for _, col in ipairs(tbl.cols.x) do
    table.insert(parts, tostring(row[col.at]))
  end
  return table.concat(parts, ", ")
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Error: CSV file path required\n")
    os.exit(1)
  end

  local tbl, data = load_data(path)
  if #data == 0 then
    print("No rows with known performance found.")
    os.exit(0)
  end

  -- ------------------------------------------------------------------
  -- APACHE-03: estimate performance of the first row in the file
  -- ------------------------------------------------------------------
  print("== APACHE-03 ==")
  local target = data[1]
  local est, sd = estimate_perf(tbl, data, target, 5)
  if est then
    print(string.format("Row 1 settings: %s", row_to_string(tbl, target)))
    print(string.format("Estimated performance: %.0f (±%.1f)", est, sd))
  else
    print("Unable to estimate performance.")
  end

  -- ------------------------------------------------------------------
  -- APACHE-04: compare two configurations that differ by one setting
  -- ------------------------------------------------------------------
  print("\n== APACHE-04 ==")
  local r1, r2 = find_one_diff_pair(tbl, data)
  if r1 and r2 then
    local est1, sd1 = estimate_perf(tbl, data, r1, 5)
    local est2, sd2 = estimate_perf(tbl, data, r2, 5)
    if est1 and est2 then
      local diff = est1 - est2
      local uncert = math.sqrt(sd1*sd1 + sd2*sd2)
      print(string.format("Configurations differ in one setting:"))
      print(string.format("  Row A: %s", row_to_string(tbl, r1)))
      print(string.format("  Row B: %s", row_to_string(tbl, r2)))
      print(string.format("Estimated performance difference: %.0f ± %.1f",
                          diff, uncert))
    else
      print("Could not estimate performance for both configurations.")
    end
  else
    print("No pair of configurations differing by exactly one setting found.")
  end
end

main()
