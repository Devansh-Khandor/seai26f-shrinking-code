#!/usr/bin/env lua
-- skill_03.lua
-- Run: lua skill_03.lua <csv_path>
--
-- Provides two NASA requests:
--   NASA-05 – Compare projects with different programmer
--             capability ratings and assess if effort
--             differences are statistically meaningful.
--   NASA-06 – Compare projects with different complexity
--             ratings and assess if defect outcomes differ
--             reliably or evidence is insufficient.

local ezr = require "ezr"

-- Helpers from ezr
local csv    = ezr.csv
local Tbl    = ezr.Tbl
local push   = ezr.push
local sorted = ezr.sorted
local map    = ezr.map
local kap    = ezr.kap
local floor  = math.floor
local sqrt   = math.sqrt

-- ------------------------------------------------------------------
-- Read data
-- ------------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_03.lua <csv_path>\n")
  os.exit(1)
end

local tbl = Tbl(csv(path))

-- ------------------------------------------------------------------
-- Utility functions
-- ------------------------------------------------------------------
-- Column statistics (mean, std, min, max)
local function col_stats(tbl, at)
  local col = tbl.cols.all[at]
  local minv, maxv = nil, nil
  for _, r in ipairs(tbl.rows) do
    local v = r[at]
    if not minv or v < minv then minv = v end
    if not maxv or v > maxv then maxv = v end
  end
  return {
    min  = minv,
    max  = maxv,
    mean = col:mid(),
    std  = col:div()
  }
end

-- Median of a column (numeric)
local function median_from_col(tbl, at)
  local vals = {}
  for _, r in ipairs(tbl.rows) do push(vals, r[at]) end
  local s = sorted(vals)
  local n = #s
  if n == 0 then return nil end
  if n % 2 == 1 then
    return s[(n + 1) / 2]
  else
    return (s[n / 2] + s[n / 2 + 1]) / 2
  end
end

-- Group statistics for a target column, grouped by a grouping column.
-- Returns a table {key = {n, mean, std}}
local function group_stats(tbl, group_idx, target_idx)
  local groups = {}
  for _, r in ipairs(tbl.rows) do
    local g = r[group_idx]
    if not groups[g] then
      groups[g] = {sum = 0, sumsq = 0, n = 0}
    end
    local val = r[target_idx]
    local gstat = groups[g]
    gstat.n = gstat.n + 1
    gstat.sum = gstat.sum + val
    gstat.sumsq = gstat.sumsq + val * val
  end
  for g, s in pairs(groups) do
    local n = s.n
    local mean = s.sum / n
    local var = (s.sumsq - s.sum * s.sum / n) / (n - 1)
    local std = sqrt(math.max(var, 0))
    groups[g] = {n = n, mean = mean, std = std}
  end
  return groups
end

-- 95% confidence interval for the difference of two means
local function diff_ci(mean1, std1, n1, mean2, std2, n2)
  local diff = mean1 - mean2
  local se = sqrt(std1 * std1 / n1 + std2 * std2 / n2)
  local low = diff - 1.96 * se
  local high = diff + 1.96 * se
  return diff, low, high, se
end

-- ------------------------------------------------------------------
-- NASA-05: Programmer capability vs effort
-- ------------------------------------------------------------------
local acap_at   = 17   -- acap column
local effort_at = 27   -- Effort- column

-- Compute median acap to split into low/high groups
local acap_median = median_from_col(tbl, acap_at)

-- Low group: acap <= median
-- High group: acap > median
local low_stats  = {n = 0, sum = 0, sumsq = 0}
local high_stats = {n = 0, sum = 0, sumsq = 0}

for _, r in ipairs(tbl.rows) do
  local acap = r[acap_at]
  local eff  = r[effort_at]
  if acap <= acap_median then
    low_stats.n = low_stats.n + 1
    low_stats.sum = low_stats.sum + eff
    low_stats.sumsq = low_stats.sumsq + eff * eff
  else
    high_stats.n = high_stats.n + 1
    high_stats.sum = high_stats.sum + eff
    high_stats.sumsq = high_stats.sumsq + eff * eff
  end
end

local function stats_from(s)
  local n = s.n
  local mean = s.sum / n
  local var = (s.sumsq - s.sum * s.sum / n) / (n - 1)
  local std = sqrt(math.max(var, 0))
  return {n = n, mean = mean, std = std}
end

local low = stats_from(low_stats)
local high = stats_from(high_stats)

local diff, ci_low, ci_high, se = diff_ci(high.mean, high.std, high.n,
                                         low.mean, low.std, low.n)

print("== NASA-05 ==")
print(string.format(
  "Programmer capability: median=%.2f -> low (n=%d), high (n=%d)",
  acap_median, low.n, high.n))
print(string.format(
  "Low group mean effort=%.2f, std=%.2f",
  low.mean, low.std))
print(string.format(
  "High group mean effort=%.2f, std=%.2f",
  high.mean, high.std))
print(string.format(
  "Difference in mean effort: %.2f ± %.2f (95%% CI: [%.2f, %.2f])",
  diff, se, ci_low, ci_high))
if (ci_low > 0 or ci_high < 0) then
  print("Conclusion: The effort difference between low and high programmer capability groups is statistically meaningful.")
else
  print("Conclusion: The effort difference between low and high programmer capability groups is not statistically meaningful; evidence is insufficient.")
end

-- ------------------------------------------------------------------
-- NASA-06: Complexity rating vs defects
-- ------------------------------------------------------------------
local cplx_at   = 11   -- cplx column (symbolic)
local defect_at = 28   -- Defects- column

local groups = group_stats(tbl, cplx_at, defect_at)

-- Find the group with lowest and highest mean defects
local low_def_group, high_def_group
local lowest = 1/0  -- +inf
local highest = -1/0 -- -inf
for g, s in pairs(groups) do
  if s.mean < lowest then
    lowest = s.mean
    low_def_group = g
  end
  if s.mean > highest then
    highest = s.mean
    high_def_group = g
  end
end

local lowg = groups[low_def_group]
local highg = groups[high_def_group]

local diff_def, ci_low_def, ci_high_def, se_def =
  diff_ci(highg.mean, highg.std, highg.n,
          lowg.mean, lowg.std, lowg.n)

print("\n== NASA-06 ==")
print("Complexity rating groups (cplx):")
for g, s in kap(groups, function(k,v) return k.." (n="..v.n..")" end) do
  print(string.format(
    "  %s: mean defects=%.2f, std=%.2f",
    g, s.mean, s.std))
end
print(string.format(
  "Difference in mean defects between '%s' and '%s': %.2f ± %.2f (95%% CI: [%.2f, %.2f])",
  high_def_group, low_def_group,
  diff_def, se_def, ci_low_def, ci_high_def))
if (ci_low_def > 0 or ci_high_def < 0) then
  print("Conclusion: Defect outcomes differ reliably across complexity levels.")
else
  print("Conclusion: Evidence is insufficient to conclude a reliable difference in defect outcomes across complexity levels.")
end
