#!/usr/bin/env lua
-- Skill 03: Compare KeepAlive and AccessLog effects on performance
-- Reuses the ezr library and the helper functions from skill_01 and skill_02

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Utility: pretty‑print numbers/values using ezr.show
-- ------------------------------------------------------------------
local function pretty(v) return ezr.show(v) end

-- ------------------------------------------------------------------
-- Compute statistics for a subset of rows matching a filter
-- ------------------------------------------------------------------
local function subset_stats(tbl, filter)
  local perf_at = tbl.cols.y[1].at
  local rows = {}
  for _, r in ipairs(tbl.rows) do
    if filter(r) then table.insert(rows, r) end
  end
  local n = #rows
  if n == 0 then return {n = 0} end

  local sum, sumsq = 0, 0
  local min_perf, max_perf = math.huge, -math.huge
  for _, r in ipairs(rows) do
    local p = r[perf_at]
    sum = sum + p
    sumsq = sumsq + p * p
    if p < min_perf then min_perf = p end
    if p > max_perf then max_perf = p end
  end
  local mean = sum / n
  local sd = math.sqrt(math.max(sumsq - sum * sum / n, 0) / n)
  return {
    n = n,
    mean = mean,
    sd = sd,
    min = min_perf,
    max = max_perf,
    rows = rows
  }
end

-- ------------------------------------------------------------------
-- Compute difference between two groups and 95% confidence interval
-- ------------------------------------------------------------------
local function diff_ci(stats1, stats2)
  local mean_diff = stats1.mean - stats2.mean
  local se = math.sqrt(stats1.sd * stats1.sd / stats1.n + stats2.sd * stats2.sd / stats2.n)
  local ci_low = mean_diff - 1.96 * se
  local ci_high = mean_diff + 1.96 * se
  return {diff = mean_diff, ci_low = ci_low, ci_high = ci_high, se = se}
end

-- ------------------------------------------------------------------
-- Print results for a given comparison
-- ------------------------------------------------------------------
local function print_comparison(id, label, stats_true, stats_false)
  print("== " .. id .. " ==")
  print(string.format("  %s Enabled:  n=%d  mean=%s  sd=%s",
        label, stats_true.n, pretty(stats_true.mean), pretty(stats_true.sd)))
  print(string.format("  %s Disabled: n=%d  mean=%s  sd=%s",
        label, stats_false.n, pretty(stats_false.mean), pretty(stats_false.sd)))
  local c = diff_ci(stats_true, stats_false)
  print(string.format("  Mean difference (Enabled - Disabled) = %s",
        pretty(c.diff)))
  print(string.format("  95%% CI: [%s , %s]",
        pretty(c.ci_low), pretty(c.ci_high)))
  if c.ci_low > 0 or c.ci_high < 0 then
    print("  => Difference is statistically significant (p<0.05).")
  else
    print("  => Difference is not statistically significant; evidence inconclusive.")
  end
  print()
end

-- ------------------------------------------------------------------
-- Main routine
-- ------------------------------------------------------------------
local function main()
  local csv_path = arg[1]
  assert(csv_path, "Usage: lua skill_03.lua <csv path>")

  -- Load the data
  local tbl = ezr.Tbl(ezr.csv(csv_path))

  -- Group stats for KeepAlive
  local keep_true = subset_stats(tbl, function(r) return r[3] == 1 end)  -- keepAlive column is third
  local keep_false = subset_stats(tbl, function(r) return r[3] == 0 end)

  -- Group stats for AccessLog
  local log_true = subset_stats(tbl, function(r) return r[6] == 1 end)  -- accessLog column is sixth
  local log_false = subset_stats(tbl, function(r) return r[6] == 0 end)

  -- Print comparisons
  print_comparison("APACHE-05", "KeepAlive", keep_true, keep_false)
  print_comparison("APACHE-06", "Access Log", log_true, log_false)
end

-- Execute if run directly
if not debug.getinfo(1,"S").short_src:match("[/\\]skill_03.lua$") then
  main()
end
