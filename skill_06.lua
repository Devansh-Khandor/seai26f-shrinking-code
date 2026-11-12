#!/usr/bin/env lua
-- skill_06.lua
-- APACHE-11 & APACHE-12: explanation of recommendations and
-- differences between configurations.
-- Usage: lua skill_06.lua <path-to-Apache_AllMeasurements.csv>
--
-- Requires only the ezr library (ezr.lua & ezr-lib.lua) in the same
-- directory.

local ezr = require("ezr")
local csv  = ezr.csv
local sorted = ezr.sorted
local keysort = ezr.keysort
local map = ezr.map
local rand = ezr.rand
local keys = ezr.keys

-- ------------------------------------------------------------------
-- Load data
-- ------------------------------------------------------------------
local function load_tbl(path)
  return ezr.Tbl(csv(path))
end

-- ------------------------------------------------------------------
-- Estimate performance of a row using k‑nearest neighbours
-- ------------------------------------------------------------------
local function estimate_perf(tbl, data, row, k)
  local dists = {}
  for _, r in ipairs(data) do
    if r ~= row then
      table.insert(dists, {d = tbl:distx(row, r),
                           perf = r[tbl.cols.y[1].at]})
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
-- Find first pair of rows that differ in exactly one setting
-- ------------------------------------------------------------------
local function one_diff_pair(tbl, data)
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
-- Pretty print a row (all columns)
-- ------------------------------------------------------------------
local function row_to_string(row)
  local parts = {}
  for _, v in ipairs(row) do
    table.insert(parts, tostring(v))
  end
  return table.concat(parts, ", ")
end

-- ------------------------------------------------------------------
-- Identify differences between two rows
-- ------------------------------------------------------------------
local function diff_settings(tbl, r1, r2)
  local diffs = {}
  for at=1, #tbl.cols.all do
    local c = tbl.cols.all[at]
    if r1[at] ~= r2[at] then
      table.insert(diffs, c.name .. "=" .. tostring(r1[at]) ..
                   " -> " .. tostring(r2[at]))
    end
  end
  return diffs
end

-- ------------------------------------------------------------------
-- APACHE-11: Recommendation explanation
-- ------------------------------------------------------------------
local function apache11(tbl)
  print("== APACHE-11 ==")

  -- Gather all rows with a known performance value
  local perf_col = tbl.cols.y[1].at
  local data = {}
  for _, r in ipairs(tbl.rows) do
    if r[perf_col] ~= "?" then
      table.insert(data, r)
    end
  end
  if #data == 0 then
    print("  No performance data available.")
    return
  end

  -- Sort by performance (lowest first)
  local best_rows = sorted(data, function(a,b) return a[perf_col] < b[perf_col] end)

  local best = best_rows[1]
  local second = best_rows[2]

  print(string.format("  Recommended configuration (lowest performance): %s",
                      row_to_string(best)))
  if second then
    local diff = second[perf_col] - best[perf_col]
    print(string.format("  Performance gap vs second best: %.1f (%s -> %s)",
                        diff,
                        tostring(best[perf_col]),
                        tostring(second[perf_col])))
    local diffs = diff_settings(tbl, best, second)
    if #diffs > 0 then
      print("  Settings that differ from the second best:")
      for _, d in ipairs(diffs) do print("    "..d) end
    else
      print("  No settings differ from the second best.")
    end
  else
    print("  Only one configuration with known performance.")
  end

  print()
end

-- ------------------------------------------------------------------
-- APACHE-12: Explain differences between two configs
-- ------------------------------------------------------------------
local function apache12(tbl)
  print("== APACHE-12 ==")

  local perf_col = tbl.cols.y[1].at
  local data = {}
  for _, r in ipairs(tbl.rows) do
    if r[perf_col] ~= "?" then
      table.insert(data, r)
    end
  end
  if #data < 2 then
    print("  Insufficient data to compare configurations.")
    return
  end

  local r1, r2 = one_diff_pair(tbl, data)
  if not r1 or not r2 then
    print("  No pair of configurations that differ by exactly one setting found.")
    return
  end

  local est1, sd1 = estimate_perf(tbl, data, r1, 5)
  local est2, sd2 = estimate_perf(tbl, data, r2, 5)
  local diff = est1 - est2
  local uncert = math.sqrt((sd1 or 0)^2 + (sd2 or 0)^2)

  print(string.format("  Configuration A: %s", row_to_string(r1)))
  print(string.format("  Configuration B: %s", row_to_string(r2)))
  print(string.format("  Estimated performance A: %.1f ±%.1f", est1, sd1))
  print(string.format("  Estimated performance B: %.1f ±%.1f", est2, sd2))
  print(string.format("  Performance difference (A - B): %.1f ±%.1f", diff, uncert))
  local diffs = diff_settings(tbl, r1, r2)
  if #diffs > 0 then
    print("  Settings that differ:")
    for _, d in ipairs(diffs) do print("    "..d) end
  else
    print("  No differing settings (should not happen).")
  end

  print()
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

  local tbl = load_tbl(path)

  apache11(tbl)
  apache12(tbl)
end

main()
