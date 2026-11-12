#!/usr/bin/env lua
-- Skill 04: Real‑time alerting and incremental summaries
--
-- This script reads the benchmark CSV, then:
--   * APACHE-07  – prints an alert for any configuration that
--                  performs unusually poorly compared to all
--                  previously seen rows.
--   * APACHE-08  – prints updated performance statistics and
--                  a short list of recommended configurations
--                  as new data arrives.
--
-- The code re‑uses the helper logic from skill_01.lua but
-- incorporates an incremental pass over the rows.
--

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Pretty‑print helper
-- ------------------------------------------------------------------
local function pretty(v) return ezr.show(v) end

-- ------------------------------------------------------------------
-- Compute overall statistics for the performance column
-- ------------------------------------------------------------------
local function performance_stats(tbl)
  local perf_at = tbl.cols.y[1].at
  local mids = tbl:mids()
  local mean   = mids[perf_at]
  local sd     = tbl.cols.all[perf_at]:div()

  local sorted_rows = ezr.sorted(tbl.rows,
    function(r) return r[perf_at] end)

  local n = #sorted_rows
  local min_perf = sorted_rows[1][perf_at]
  local max_perf = sorted_rows[n][perf_at]
  local median_perf = sorted_rows[math.floor(n/2)+1][perf_at]

  return {
    mean   = mean,
    median = median_perf,
    sd     = sd,
    best   = min_perf,
    worst  = max_perf,
    sorted = sorted_rows,
    at     = perf_at,
    mids   = mids
  }
end

-- ------------------------------------------------------------------
-- Identify distinguishing settings for the top configurations
-- ------------------------------------------------------------------
local function best_rows_details(stats, top_n)
  local details = {}
  local sorted = stats.sorted
  local n = #sorted
  local top = math.min(n, top_n)
  local mids = stats.mids

  for i = 1, top do
    local r = sorted[i]
    local diffs = {}
    for _, c in ipairs(stats.tbl.cols.all) do
      local v   = r[c.at]
      local mid = mids[c.at]
      local diff = false
      if c.__index == ezr.NUM then
        if mid == 0 then
          diff = v ~= 0
        else
          diff = math.abs(v - mid) / math.abs(mid) > 0.1
        end
      else
        diff = v ~= mid
      end
      if diff then
        diffs[c.name] = v
      end
    end
    table.insert(details, {perf = r[stats.at], row = r, diffs = diffs})
  end
  return details
end

-- ------------------------------------------------------------------
-- Main routine
-- ------------------------------------------------------------------
local function main()
  local csv_path = arg[1]
  assert(csv_path, "Usage: lua skill_04.lua <csv path>")

  -- Load data
  local tbl = ezr.Tbl(ezr.csv(csv_path))
  local stats = performance_stats(tbl)
  stats.tbl = tbl  -- for best_rows_details

  -- APACHE-07 : alert on unusually poor performances
  print("== APACHE-07 ==")
  local perf_at = tbl.cols.y[1].at
  local mean = stats.mean
  local sd   = stats.sd
  local threshold = mean + 2 * sd   -- 2‑sigma cutoff

  for i, r in ipairs(tbl.rows) do
    local perf = r[perf_at]
    if perf > threshold then
      local z = (perf - mean) / sd
      print(string.format(
        "Alert: Row %d has unusually poor performance: %s (z=%.2f)",
        i, pretty(perf), z))
    end
  end

  -- APACHE-08 : updated summary & recommended shortlist
  print("\n== APACHE-08 ==")
  print(string.format("Mean Performance: %s", pretty(stats.mean)))
  print(string.format("Median Performance: %s", pretty(stats.median)))
  print(string.format("Std Dev: %s", pretty(stats.sd)))
  print(string.format("Best Performance: %s", pretty(stats.best)))
  print(string.format("Worst Performance: %s", pretty(stats.worst)))

  local best_details = best_rows_details(stats, 5)
  print("\nTop 5 recommended configurations:")
  for _, d in ipairs(best_details) do
    print(string.format("  Performance: %s", pretty(d.perf)))
    if next(d.diffs) then
      print("    Distinguishing settings:")
      for k, v in pairs(d.diffs) do
        print(string.format("      %s: %s", k, pretty(v)))
      end
    else
      print("    No distinguishing settings.")
    end
  end
end

-- Execute if run directly
if not debug.getinfo(1, "S").short_src:match("[/\\]skill_04.lua$") then
  main()
end
