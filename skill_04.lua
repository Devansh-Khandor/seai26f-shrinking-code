#!/usr/bin/env lua
-- skill_04.lua
-- Immediate assessment of new Apache benchmark records and
-- incremental update of performance summaries and a top‑3 list.
--
-- Usage:  lua skill_04.lua <path-to-Apache_AllMeasurements.csv>
--
-- The script uses only the ezr library (ezr.lua & ezr-lib.lua) in the same directory.

local ezr = require("ezr")
local csv  = ezr.csv
local sorted = ezr.sorted

-- ------------------------------------------------------------------
-- Utilities
-- ------------------------------------------------------------------
local function row_to_string(row)
  local parts = {}
  for _, v in ipairs(row) do
    table.insert(parts, tostring(v))
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

  -- Load all rows
  local tbl = ezr.Tbl(csv(path))

  -- Identify the index of the performance column (the goal)
  local perf_col = tbl.cols.y[1].at

  -- Statistics that will be updated incrementally
  local n = 0
  local mean = 0
  local M2 = 0          -- sum of squared deviations
  local min_perf = nil
  local max_perf = nil
  local perf_vals = {}  -- for median
  local all_rows = {}   -- keep rows for best‑3 list
  local alerts = {}     -- messages for APACHE‑07

  for rownum, row in ipairs(tbl.rows) do
    local perf = row[perf_col]
    if perf ~= "?" then
      -- Update online mean & variance (Welford)
      n = n + 1
      local delta = perf - mean
      mean = mean + delta / n
      local delta2 = perf - mean
      M2 = M2 + delta * delta2

      -- Update min/max
      if not min_perf or perf < min_perf then min_perf = perf end
      if not max_perf or perf > max_perf then max_perf = perf end

      table.insert(perf_vals, perf)
      table.insert(all_rows, row)

      -- After at least 3 samples, flag unusually poor results
      if n >= 3 then
        local sd = math.sqrt(M2 / (n - 1))
        local threshold = mean + 1.5 * sd
        if perf > threshold then
          table.insert(alerts, string.format(
            "Row %d (%s) has unusually poor performance: %d (mean=%.1f, sd=%.1f)",
            rownum, row_to_string(row), perf, mean, sd))
        end
      end
    end
  end

  -- ------------------------------------------------------------------
  -- Output APACHE‑07 alerts
  -- ------------------------------------------------------------------
  print("== APACHE-07 ==")
  if #alerts == 0 then
    print("No unusually poor configurations found.")
  else
    for _, msg in ipairs(alerts) do
      print(msg)
    end
  end

  -- ------------------------------------------------------------------
  -- Prepare summary statistics for APACHE‑08
  -- ------------------------------------------------------------------
  local sd = n > 1 and math.sqrt(M2 / (n - 1)) or 0
  local median
  if n > 0 then
    local sorted_perfs = sorted(perf_vals)
    median = sorted_perfs[math.floor((n + 1) / 2)]
  else
    median = 0
  end

  -- Best‑3 (lowest performance) list
  local best_rows = sorted(all_rows, function(a, b)
    return a[perf_col] < b[perf_col]
  end)

  -- ------------------------------------------------------------------
  -- Output APACHE‑08 summary
  -- ------------------------------------------------------------------
  print("\n== APACHE-08 ==")
  print(string.format(
    "Performance summary: count=%d  min=%d  max=%d  mean=%.1f  sd=%.1f  median=%d",
    n, min_perf, max_perf, mean, sd, median))
  print("Top 3 configurations (lowest performance):")
  for i = 1, 3 do
    local r = best_rows[i]
    if r then
      print(string.format("%d: %s", i, row_to_string(r)))
    end
  end
end

-- Run the program
main()
