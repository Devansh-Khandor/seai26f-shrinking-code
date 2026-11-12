#!/usr/bin/env lua
-- Skill 03 – Apache performance comparison by configuration flags
--
-- Usage:  lua skill_03.lua <csv path>
--
-- Requires the ezr library (ezr.lua, ezr-lib.lua) in the same directory.

local ezr = require("ezr")

-- Find column index by exact name
local function find_col_index(tbl, name)
  for _, c in ipairs(tbl.cols.all) do
    if c.name == name then
      return c.at
    end
  end
  return nil
end

-- Compute mean, std dev and count for a subset of rows
local function stats_for_value(rows, col_index, val, goal_at)
  local sum, sumsq, n = 0, 0, 0
  for _, r in ipairs(rows) do
    if r[col_index] == val then
      local perf = r[goal_at]
      sum   = sum + perf
      sumsq = sumsq + perf * perf
      n     = n + 1
    end
  end
  local mean = n > 0 and sum / n or 0
  local var
  if n > 1 then
    var = (sumsq - n * mean * mean) / (n - 1)
  else
    var = 0
  end
  local std = math.sqrt(math.max(var, 0))
  return {mean = mean, std = std, n = n}
end

-- Produce the statistical report for a binary flag
local function report_flag(rows, flag_name, flag_index, goal_at)
  local stats0 = stats_for_value(rows, flag_index, 0, goal_at)
  local stats1 = stats_for_value(rows, flag_index, 1, goal_at)

  print(string.format("-- %s enabled (1) vs disabled (0) --", flag_name))
  print(string.format("Enabled (1):   mean = %.2f, std = %.2f, n = %d",
          stats1.mean, stats1.std, stats1.n))
  print(string.format("Disabled (0):  mean = %.2f, std = %.2f, n = %d",
          stats0.mean, stats0.std, stats0.n))

  local diff = stats1.mean - stats0.mean
  local se = math.sqrt((stats1.std * stats1.std) / stats1.n +
                       (stats0.std * stats0.std) / stats0.n)
  local ci = 1.96 * se
  local signif = math.abs(diff) > ci

  print(string.format("Difference (1 - 0): %.2f", diff))
  print(string.format("95%% CI: %.2f ± %.2f", diff, ci))
  print("Significant at 95% confidence: " .. (signif and "Yes" or "No"))
  print()
end

-- Main --------------------------------------------------------------------
local function main(csv_path)
  if not csv_path then
    print("Usage: lua skill_03.lua <csv path>")
    os.exit(1)
  end

  -- Load data
  local tbl = ezr.Tbl(ezr.csv(csv_path))

  -- Goal column index
  local perf_col = tbl.cols.y[1]
  local perf_at  = perf_col.at

  -- Indices for flags we are interested in
  local keep_idx   = find_col_index(tbl, "keepAlive")
  local access_idx = find_col_index(tbl, "accessLog")

  if not keep_idx then
    print("Column 'keepAlive' not found in data")
    os.exit(1)
  end
  if not access_idx then
    print("Column 'accessLog' not found in data")
    os.exit(1)
  end

  print("== APACHE-05 ==")
  report_flag(tbl.rows, "keepAlive", keep_idx, perf_at)

  print("== APACHE-06 ==")
  report_flag(tbl.rows, "accessLog", access_idx, perf_at)
end

main(arg[1])
