#!/usr/bin/env lua
-- Skill 06: Explain recommendations and differences
--
--  APACHE-11  – When recommending a configuration, compare it
--                to the best‑performing configuration already seen.
--  APACHE-12  – When two configurations differ in performance,
--                list the differing settings and how they relate
--                to the performance gap.
--
--  This script re‑uses helpers from the previous skills and
--  only adds the explanation logic.

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Pretty printer
-- ------------------------------------------------------------------
local function pretty(v) return ezr.show(v) end

-- ------------------------------------------------------------------
-- Common statistics helpers – copied from skill_04
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
-- Find the top N configurations (by lowest performance)
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
-- Compare two rows – list differing settings and the performance gap
-- ------------------------------------------------------------------
local function compare_rows(tbl, rowA, rowB)
  local perf_at = tbl.cols.y[1].at
  local diffs = {}
  for _, c in ipairs(tbl.cols.all) do
    local a = rowA[c.at]
    local b = rowB[c.at]
    if a ~= b then
      diffs[#diffs+1] = {name = c.name, a = a, b = b}
    end
  end
  local gap = rowB[perf_at] - rowA[perf_at]
  return diffs, gap
end

-- ------------------------------------------------------------------
-- Main routine
-- ------------------------------------------------------------------
local function main()
  local csv_path = arg[1]
  assert(csv_path, "Usage: lua skill_06.lua <csv path>")

  -- Load data
  local tbl = ezr.Tbl(ezr.csv(csv_path))
  local stats = performance_stats(tbl)
  stats.tbl = tbl  -- for best_rows_details

  -- ----------------------------------------------------------------
  -- APACHE-11 – recommendation explanation
  -- ----------------------------------------------------------------
  print("== APACHE-11 ==")
  local best_cfg = stats.sorted[1]
  local top_details = best_rows_details(stats, 3)  -- show top 3

  for _, d in ipairs(top_details) do
    print(string.format("  Performance: %s", pretty(d.perf)))
    if next(d.diffs) then
      print("  Differing settings from the best config:")
      for k, v in pairs(d.diffs) do
        print(string.format("    %s: %s", k, pretty(v)))
      end
    else
      print("  Identical to the best configuration.")
    end
    -- Show performance difference to the best
    local diff_perf = d.perf - best_cfg[stats.at]
    print(string.format("  Performance gap vs best: %s", pretty(diff_perf)))
    print()
  end

  -- ----------------------------------------------------------------
  -- APACHE-12 – compare two extreme configurations
  -- ----------------------------------------------------------------
  print("== APACHE-12 ==")
  local worst_cfg = stats.sorted[#stats.sorted]
  local diffs, gap = compare_rows(tbl, best_cfg, worst_cfg)

  print(string.format("Best config performance: %s", pretty(best_cfg[stats.at])))
  print(string.format("Worst config performance: %s", pretty(worst_cfg[stats.at])))
  print(string.format("Performance difference (worst - best): %s", pretty(gap)))
  print("Differing settings:")
  for _, d in ipairs(diffs) do
    print(string.format("  %s: best=%s, worst=%s",
      d.name, pretty(d.a), pretty(d.b)))
  end
  print()
end

-- Execute if run directly
if not debug.getinfo(1, "S").short_src:match("[/\\]skill_06.lua$") then
  main()
end
