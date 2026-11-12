#!/usr/bin/env lua
-- skill_03.lua
-- Compare Apache configurations for KeepAlive and AccessLog settings
-- and determine whether the observed performance differences are
-- statistically meaningful using the ezr library.
--
-- Usage: lua skill_03.lua <path-to-Apache_AllMeasurements.csv>
--
-- Requires the ezr library (ezr.lua & ezr-lib.lua) in the same directory.

local ezr = require("ezr")
local csv = ezr.csv
local sorted = ezr.sorted
local same = ezr.same

-- ------------------------------------------------------------------
-- Load data
-- ------------------------------------------------------------------
local function load_table(path)
  return ezr.Tbl(csv(path))
end

-- ------------------------------------------------------------------
-- Gather performance values for rows matching a predicate
-- ------------------------------------------------------------------
local function group_perf(tbl, pred)
  local vals = {}
  for _, row in ipairs(tbl.rows) do
    local perf = row[tbl.cols.y[1].at]
    if perf ~= "?" and pred(row) then
      table.insert(vals, perf)
    end
  end
  return vals
end

-- ------------------------------------------------------------------
-- Basic statistics
-- ------------------------------------------------------------------
local function stats(arr)
  local n = #arr
  if n == 0 then return nil end
  table.sort(arr)
  local sum, sumsq = 0, 0
  for _, v in ipairs(arr) do
    sum = sum + v
    sumsq = sumsq + v * v
  end
  local mean = sum / n
  local variance = (sumsq / n) - (mean * mean)
  local sd = math.sqrt(math.max(variance, 0))
  local median = arr[math.floor((n + 1) / 2)]
  return {
    count = n,
    min   = arr[1],
    max   = arr[n],
    mean  = mean,
    median = median,
    sd    = sd
  }
end

-- ------------------------------------------------------------------
-- Pretty print stats
-- ------------------------------------------------------------------
local function fmt_stats(s)
  return string.format(
    "  count: %d\n  mean: %.2f\n  median: %.2f\n  sd: %.2f\n  min: %d\n  max: %d",
    s.count, s.mean, s.median, s.sd, s.min, s.max)
end

-- ------------------------------------------------------------------
-- Compare two groups and report significance
-- ------------------------------------------------------------------
local function compare_groups(name, vals1, vals2)
  local s1 = stats(vals1)
  local s2 = stats(vals2)
  print("== " .. name .. " ==")
  if not s1 or not s2 then
    print("  Insufficient data for comparison.")
    return
  end
  print("  " .. name .. " - group 1:")
  print(fmt_stats(s1))
  print("  " .. name .. " - group 2:")
  print(fmt_stats(s2))

  -- Assess if distributions differ
  local diff_mean = s1.mean - s2.mean
  local diff_med  = s1.median - s2.median
  print(string.format("  Mean difference (group1 - group2): %.2f", diff_mean))
  print(string.format("  Median difference (group1 - group2): %.2f", diff_med))

  if same(sorted(vals1), sorted(vals2)) then
    print("  Observed differences are inconclusive.")
  else
    print("  Observed differences are meaningful enough to guide decisions.")
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

  local tbl = load_table(path)

  -- KeepAlive is column 3
  local keepalive_on  = group_perf(tbl, function(row) return row[3] == 1 end)
  local keepalive_off = group_perf(tbl, function(row) return row[3] == 0 end)

  compare_groups("KeepAlive ON vs OFF", keepalive_on, keepalive_off)

  -- AccessLog is column 6
  local accesslog_on  = group_perf(tbl, function(row) return row[6] == 1 end)
  local accesslog_off = group_perf(tbl, function(row) return row[6] == 0 end)

  compare_groups("AccessLog ON vs OFF", accesslog_on, accesslog_off)
end

main()
