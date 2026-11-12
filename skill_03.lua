#!/usr/bin/env lua
-- skill_03.lua
-- Implements the requested APACHE-05 and APACHE-06 analyses.
--
-- Usage: lua skill_03.lua <csv path>
--
-- Requires the ezr library from the provided codebase.

local ezr = require"ezr"

-- Utility: round numbers for display, matching ezr's format.
local function display(v)
  if type(v) == "number" then
    return ezr.round(v)
  else
    return tostring(v)
  end
end

-- Utility: find a column object by its name.
local function col_by_name(tbl, name)
  for _,c in ipairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("Column '" .. name .. "' not found")
end

-- Compute statistics (n, mean, std) for all rows whose
-- value in column `c` equals `val`.
local function stats_by_col(tbl, c, val, perf_col)
  local vals = {}
  for _,r in ipairs(tbl.rows) do
    if r[c.at] == val then
      table.insert(vals, r[perf_col.at])
    end
  end
  local n = #vals
  if n == 0 then
    return {n=0, mean=0, std=0, vals={}}
  end
  local sum, sumsq = 0, 0
  for _,v in ipairs(vals) do
    sum   = sum + v
    sumsq = sumsq + v*v
  end
  local mean = sum / n
  local var  = (sumsq - n*mean*mean) / (n-1)
  local std  = math.sqrt(var)
  return {n=n, mean=mean, std=std, vals=vals}
end

-- Compute 95% confidence interval for difference of two means.
local function ci_diff(s1, s2)
  local diff = s2.mean - s1.mean
  local se   = math.sqrt(s1.std*s1.std / s1.n + s2.std*s2.std / s2.n)
  local margin = 1.96 * se
  return diff - margin, diff + margin
end

-- Main -------------------------------------------------------------------

local csvfile = arg[1]
assert(csvfile, "Usage: lua skill_03.lua <csv path>")

local tbl = ezr.Tbl(ezr.csv(csvfile))
local perf_col = tbl.cols.y[1]  -- the single goal column

-- APACHE-05: KeepAlive comparison ------------------------------------------------
local keep_col = col_by_name(tbl, "keepAlive")

local stats0 = stats_by_col(tbl, keep_col, 0, perf_col)
local stats1 = stats_by_col(tbl, keep_col, 1, perf_col)

local ci_low, ci_high = ci_diff(stats0, stats1)

print("== APACHE-05 ==")
print(string.format("KeepAlive=0: n=%d, mean=%s, std=%s",
      stats0.n, display(stats0.mean), display(stats0.std)))
print(string.format("KeepAlive=1: n=%d, mean=%s, std=%s",
      stats1.n, display(stats1.mean), display(stats1.std)))
print(string.format("Mean difference (1 - 0): %s",
      display(stats1.mean - stats0.mean)))
print(string.format("95%% CI for mean difference: [%s, %s]",
      display(ci_low), display(ci_high)))
if ci_low > 0 or ci_high < 0 then
  print("The difference is statistically significant (p < 0.05).")
else
  print("The difference is not statistically significant; evidence inconclusive.")
end
print()

-- APACHE-06: AccessLog comparison ------------------------------------------------
local access_col = col_by_name(tbl, "accessLog")

local stats0_log = stats_by_col(tbl, access_col, 0, perf_col)
local stats1_log = stats_by_col(tbl, access_col, 1, perf_col)

local ci_low_log, ci_high_log = ci_diff(stats0_log, stats1_log)

print("== APACHE-06 ==")
print(string.format("AccessLog=0: n=%d, mean=%s, std=%s",
      stats0_log.n, display(stats0_log.mean), display(stats0_log.std)))
print(string.format("AccessLog=1: n=%d, mean=%s, std=%s",
      stats1_log.n, display(stats1_log.mean), display(stats1_log.std)))
print(string.format("Mean difference (1 - 0): %s",
      display(stats1_log.mean - stats0_log.mean)))
print(string.format("95%% CI for mean difference: [%s, %s]",
      display(ci_low_log), display(ci_high_log)))
if ci_low_log > 0 or ci_high_log < 0 then
  print("The difference is statistically significant (p < 0.05).")
else
  print("The difference is not statistically significant; evidence inconclusive.")
end
