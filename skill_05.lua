#!/usr/bin/env lua
-- skill_05.lua
-- Identify settings most associated with poor performance
-- and frequent combinations among poorly performing configs.
--
-- Usage:  lua skill_05.lua <path-to-Apache_AllMeasurements.csv>
--
-- Requires only the ezr library (ezr.lua & ezr-lib.lua) in the same directory.

local ezr = require("ezr")
local csv  = ezr.csv
local sorted = ezr.sorted
local map = ezr.map
local sum = ezr.sum
local floor = math.floor
local min = math.min
local max = math.max

-- ------------------------------------------------------------------
-- Load data and compute overall performance stats
-- ------------------------------------------------------------------
local function load_and_stats(path)
  local tbl = ezr.Tbl(csv(path))
  local perf_col = tbl.cols.y[1].at
  local perf_vals = {}
  for _, r in ipairs(tbl.rows) do
    local v = r[perf_col]
    if v ~= "?" then table.insert(perf_vals, v) end
  end
  local n = #perf_vals
  if n == 0 then error("No performance data") end
  local sumv = 0
  local sumsq = 0
  for _, v in ipairs(perf_vals) do
    sumv = sumv + v
    sumsq = sumsq + v * v
  end
  local mean = sumv / n
  local variance = (sumsq / n) - (mean * mean)
  local sd = math.sqrt(math.max(variance, 0))
  return tbl, {mean=mean, sd=sd, n=n}
end

-- ------------------------------------------------------------------
-- Utility: mean of a numeric list
-- ------------------------------------------------------------------
local function mean_of(vals)
  if #vals == 0 then return nil end
  local sum = 0
  for _, v in ipairs(vals) do sum = sum + v end
  return sum / #vals
end

-- ------------------------------------------------------------------
-- APACHE-09: Most settings linked to poor performance
-- ------------------------------------------------------------------
local function apache09(tbl, perf_col, threshold)
  print("== APACHE-09 ==")
  local results = {}
  for _, col in ipairs(tbl.cols.x) do
    local vals0, vals1 = {}, {}
    for _, r in ipairs(tbl.rows) do
      local perf = r[perf_col]
      if perf ~= "?" then
        if r[col.at] == 0 then
          table.insert(vals0, perf)
        else
          table.insert(vals1, perf)
        end
      end
    end
    local mean0 = mean_of(vals0)
    local mean1 = mean_of(vals1)
    if mean0 and mean1 then
      local diff = mean1 - mean0   -- positive => enabled is worse
      table.insert(results, {col=col.name, diff=diff,
                            mean0=mean0, mean1=mean1})
    end
  end
  table.sort(results, function(a,b) return a.diff > b.diff end)
  local header = string.format("%-20s %-10s %-10s %-10s %-10s",
                               "Setting", "Mean(0)", "Mean(1)", "Diff", "Δ")
  print(header)
  for i=1,min(3,#results) do
    local r = results[i]
    print(string.format("%-20s %10.2f %10.2f %10.2f %10.2f",
                        r.col, r.mean0, r.mean1, r.diff, r.diff))
  end
  print()
end

-- ------------------------------------------------------------------
-- APACHE-10: Frequent combos among poor and best configs
-- ------------------------------------------------------------------
local function combo_key(row, cols)
  local parts = {}
  for _, c in ipairs(cols) do
    table.insert(parts, tostring(row[c.at]))
  end
  return table.concat(parts, ",")
end

local function compute_combo_stats(rows, cols, perf_col)
  local combos = {}
  for _, r in ipairs(rows) do
    local key = combo_key(r, cols)
    local v = r[perf_col]
    if v ~= "?" then
      local e = combos[key]
      if not e then
        e = {count=0, sum=0, sumsq=0}
        combos[key] = e
      end
      e.count = e.count + 1
      e.sum = e.sum + v
      e.sumsq = e.sumsq + v * v
    end
  end
  -- Convert to list with mean
  local list = {}
  for k, e in pairs(combos) do
    local mean = e.sum / e.count
    local variance = (e.sumsq / e.count) - (mean * mean)
    local sd = math.sqrt(math.max(variance, 0))
    table.insert(list, {key=k, count=e.count, mean=mean, sd=sd})
  end
  return list
end

local function apache10(tbl, perf_col, poor_rows, all_rows)
  print("== APACHE-10 ==")
  local xcols = tbl.cols.x

  -- Poor combos
  local poor_combos = compute_combo_stats(poor_rows, xcols, perf_col)
  table.sort(poor_combos, function(a,b) return a.count > b.count end)
  print("Top 3 frequent poor-performance combinations:")
  for i=1,min(3,#poor_combos) do
    local c = poor_combos[i]
    print(string.format("  #%d: %s  (count=%d, mean=%.2f)",
                        i, c.key, c.count, c.mean))
  end
  print()

  -- Best combos (overall lowest mean)
  local best_combos = compute_combo_stats(all_rows, xcols, perf_col)
  table.sort(best_combos, function(a,b) return a.mean < b.mean end)
  print("Top 3 best-performance combinations:")
  for i=1,min(3,#best_combos) do
    local c = best_combos[i]
    print(string.format("  #%d: %s  (count=%d, mean=%.2f)",
                        i, c.key, c.count, c.mean))
  end
  print()

  -- Comparison
  print("Comparison (mean difference = poor - best):")
  for i=1,min(3,#poor_combos) do
    local pc = poor_combos[i]
    local bc = best_combos[i]
    local diff = pc.mean - bc.mean
    print(string.format("  Poor #%d vs Best #%d: %.2f",
                        i, i, diff))
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

  local tbl, stats = load_and_stats(path)
  local perf_col = tbl.cols.y[1].at
  local threshold = stats.mean + 1.5 * stats.sd

  -- Identify poor rows
  local poor_rows = {}
  for _, r in ipairs(tbl.rows) do
    if r[perf_col] ~= "?" and r[perf_col] > threshold then
      table.insert(poor_rows, r)
    end
  end

  apache09(tbl, perf_col, threshold)
  apache10(tbl, perf_col, poor_rows, tbl.rows)
end

-- Run
main()
