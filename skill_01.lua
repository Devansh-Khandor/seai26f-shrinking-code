#!/usr/bin/env lua
local ezr = require "ezr"

-- helper: round numbers for pretty printing
local function pretty(v)
  return ezr.show(v)
end

-- compute statistics for the performance column
local function performance_stats(tbl)
  local perf_at = tbl.cols.y[1].at

  local mids = tbl:mids()
  local mean   = mids[perf_at]
  local sd     = tbl.cols.all[perf_at]:div()

  -- sorted rows by performance (ascending, lower is better)
  local sorted_rows = ezr.sorted(tbl.rows,
    function(r) return r[perf_at] end)

  local n = #sorted_rows
  local min_perf = sorted_rows[1][perf_at]
  local max_perf = sorted_rows[n][perf_at]
  local median_perf = sorted_rows[math.floor(n/2)+1][perf_at]

  return {
    mean = mean,
    median = median_perf,
    sd = sd,
    best = min_perf,
    worst = max_perf,
    sorted = sorted_rows,
    at = perf_at,
    mids = mids
  }
end

-- find distinguishing settings for best rows
local function best_rows_details(stats, top_n)
  local details = {}
  local sorted = stats.sorted
  local n = #sorted
  local top = math.max(1, math.floor(n * 0.1)) -- top 10%
  top = math.min(top, top_n)
  local mids = stats.mids
  local tbl = sorted[1].tbl -- we don't have tbl here; will pass it separately
  -- Since we need column definitions, we will rebuild using stats.tbl
  for i = 1, top do
    local r = sorted[i]
    local diffs = {}
    for _, c in ipairs(stats.tbl.cols.all) do
      local v = r[c.at]
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

-- main
local function main()
  local csv_path = arg[1]
  assert(csv_path, "Usage: lua skill_01.lua <csv path>")

  -- read the data
  local tbl = ezr.Tbl(ezr.csv(csv_path))

  -- collect stats
  local stats = performance_stats(tbl)
  stats.tbl = tbl  -- pass tbl for later use

  -- APACHE-01
  print("== APACHE-01 ==")
  print("Mean Performance:", pretty(stats.mean))
  print("Median Performance:", pretty(stats.median))
  print("Std Dev:", pretty(stats.sd))
  print("Best Performance:", pretty(stats.best))
  print("Worst Performance:", pretty(stats.worst))

  -- APACHE-02
  print("\n== APACHE-02 ==")
  local best_details = best_rows_details(stats, 10) -- show up to 10 best configs
  for _, d in ipairs(best_details) do
    print(string.format("Performance: %s", pretty(d.perf)))
    if next(d.diffs) then
      print("  Distinguishing settings:")
      for k, v in pairs(d.diffs) do
        print(string.format("    %s: %s", k, pretty(v)))
      end
    else
      print("  No distinguishing settings.")
    end
  end
end

if not debug.getinfo(1,"S").short_src:match("[/\\]skill_01.lua$") then
  main()
end
