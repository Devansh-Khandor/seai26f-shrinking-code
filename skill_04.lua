#!/usr/bin/env lua
-- skill_04.lua
--
-- Implements the following Apache diagnostics:
--
--  APACHE-07 – As new benchmark results arrive, assess each
--              configuration immediately and alert when its
--              measured performance is unusually poor
--              compared with previously tested configurations.
--
--  APACHE-08 – Keep performance summaries and a recommended
--              configuration shortlist updated as
--              additional benchmark records become available,
--              without requiring a complete manual restart.
--
-- Usage:
--      lua skill_04.lua <csv_path>
--
-- Output:
--      Lines prefixed by "== APACHE-07 ==" and
--      "== APACHE-08 ==" followed by at least three lines of
--      results, including the actual numbers.

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- 1. Load CSV and build table
-- ------------------------------------------------------------------
local csvpath = arg[1]
if not csvpath then
  io.stderr:write("usage: lua skill_04.lua <csv_path>\n")
  os.exit(1)
end

local src = ezr.csv(csvpath)
local tbl = ezr.Tbl(src)

-- The performance column (first goal)
local perf_col = tbl.cols.y[1]
local n = #tbl.rows

-- ------------------------------------------------------------------
-- 2. Compute overall statistics (mean, std dev)
-- ------------------------------------------------------------------
local sum, sumsq = 0, 0
for _, row in ipairs(tbl.rows) do
  local v = row[perf_col.at]
  sum = sum + v
  sumsq = sumsq + v * v
end
local mean = sum / n
local var = sumsq / n - mean * mean
var = var < 0 and 0 or var
local sd = math.sqrt(var)

-- ------------------------------------------------------------------
-- 3. APACHE-07 – Detect unusually poor configurations
-- ------------------------------------------------------------------
local alerts = {}
local running_sum, running_sumsq, running_cnt = 0, 0, 0

for i, row in ipairs(tbl.rows) do
  local val = row[perf_col.at]
  if running_cnt >= 2 then
    local m = running_sum / running_cnt
    local v = running_sumsq / running_cnt - m * m
    if v < 0 then v = 0 end
    local s = math.sqrt(v)
    if val < m - 2 * s then
      table.insert(alerts, {i = i, val = val, mean = m, sd = s})
    end
  end
  running_sum = running_sum + val
  running_sumsq = running_sumsq + val * val
  running_cnt = running_cnt + 1
end

print("== APACHE-07 ==")
print(string.format("Processed %d configurations.", n))
print(string.format("Unusually poor configurations detected: %d", #alerts))
for _, a in ipairs(alerts) do
  print(string.format("  Config %d: performance=%.2f (< mean %.2f - 2*sd %.2f)",
        a.i, a.val, a.mean, 2 * a.sd))
end
if #alerts == 0 then
  print("  No unusually poor configurations found.")
end

-- ------------------------------------------------------------------
-- 4. APACHE-08 – Update summaries and recommended shortlist
-- ------------------------------------------------------------------
-- Train a regression tree on the full data
local tree = ezr.Tree(tbl, tbl.rows)

-- Compute predictions for all rows
local preds = {}
for i, row in ipairs(tbl.rows) do
  preds[i] = tree:leaf(tbl, row)
end

-- Determine top configurations by actual performance (lowest first)
local topN = math.min(5, n)
local indices = {}
for i = 1, n do
  indices[i] = i
end
table.sort(indices, function(a, b)
  return tbl.rows[a][perf_col.at] < tbl.rows[b][perf_col.at]
end)

print("\n== APACHE-08 ==")
print(string.format("Top %d configurations by actual performance:", topN))
for rank = 1, topN do
  local i = indices[rank]
  print(string.format("  %d) Config %d: actual=%.2f  predicted=%.2f",
        rank, i, tbl.rows[i][perf_col.at], preds[i]))
end
print(string.format("\n  Total configurations: %d", n))
print(string.format("  Mean performance: %.2f", mean))
print(string.format("  Std deviation: %.2f", sd))
