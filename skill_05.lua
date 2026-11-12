#!/usr/bin/env lua
--  skill_05.lua
--
--  Implements the following Apache diagnostics:
--
--  APACHE-09 – Identify settings most strongly associated
--              with poor performance.
--  APACHE-10 – Find frequent combinations of settings that
--              appear among poorly performing configurations
--              and compare their mean performance with the
--              overall mean.
--
--  Usage:
--      lua skill_05.lua <csv_path>
--
--  The script prints a line with the exact heading
--  "== APACHE-09 ==" and "== APACHE-10 ==" followed by
--  at least three lines of results each.

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- 1. Load CSV and build table
-- ------------------------------------------------------------------
local csvpath = arg[1]
if not csvpath then
  io.stderr:write("usage: lua skill_05.lua <csv_path>\n")
  os.exit(1)
end

local src = ezr.csv(csvpath)
local tbl = ezr.Tbl(src)

local perf_col = tbl.cols.y[1]   -- first goal column
local n = #tbl.rows

-- ------------------------------------------------------------------
-- 2. Overall statistics (mean & sd) for defining “poor”
-- ------------------------------------------------------------------
local sum, sumsq = 0, 0
for _, row in ipairs(tbl.rows) do
  local v = row[perf_col.at]
  sum   = sum + v
  sumsq = sumsq + v * v
end
local mean = sum / n
local var  = sumsq / n - mean * mean
var = var < 0 and 0 or var
local sd = math.sqrt(var)

local poor_thresh = mean + 2 * sd   -- unusually poor

-- ------------------------------------------------------------------
-- 3. Identify poorly performing rows
-- ------------------------------------------------------------------
local poor_idx = {}
for i, row in ipairs(tbl.rows) do
  if row[perf_col.at] > poor_thresh then
    table.insert(poor_idx, i)
  end
end

-- ------------------------------------------------------------------
-- 4. APACHE-09 – Feature importance for poor performance
-- ------------------------------------------------------------------
local feature_stats = {}
for _, c in ipairs(tbl.cols.x) do
  local sum0, sum1 = 0, 0
  local n0, n1 = 0, 0
  for _, row in ipairs(tbl.rows) do
    local val = row[c.at]
    if val == 1 or val == "1" then
      sum1 = sum1 + row[perf_col.at]
      n1   = n1 + 1
    else
      sum0 = sum0 + row[perf_col.at]
      n0   = n0 + 1
    end
  end
  local mean0 = n0 > 0 and sum0 / n0 or 0
  local mean1 = n1 > 0 and sum1 / n1 or 0
  local diff  = mean1 - mean0   -- positive => enabled worse
  table.insert(feature_stats, {
    name  = c.name,
    mean0 = mean0,
    mean1 = mean1,
    diff  = diff
  })
end

table.sort(feature_stats, function(a, b)
  return a.diff > b.diff
end)

print("== APACHE-09 ==")
print(string.format(
  "Setting     enabled mean   disabled mean   diff (enabled - disabled)"))
for i = 1, math.min(3, #feature_stats) do
  local f = feature_stats[i]
  print(string.format(
    "%-12s %.2f      %.2f       %.2f",
    f.name, f.mean1, f.mean0, f.diff))
end
print()

-- ------------------------------------------------------------------
-- 5. APACHE-10 – Frequent combinations in poor configs
-- ------------------------------------------------------------------
local combos = {}   -- key -> {count, sum}
local features = tbl.cols.x

for _, idx in ipairs(poor_idx) do
  local row = tbl.rows[idx]
  for i = 1, #features - 1 do
    for j = i + 1, #features do
      local c1 = features[i]
      local c2 = features[j]
      local key = string.format("%s=%s,%s=%s",
        c1.name, tostring(row[c1.at]),
        c2.name, tostring(row[c2.at]))
      local rec = combos[key]
      if not rec then
        rec = {count = 0, sum = 0}
        combos[key] = rec
      end
      rec.count = rec.count + 1
      rec.sum   = rec.sum + row[perf_col.at]
    end
  end
end

-- Convert map to sorted list
local combo_list = {}
for key, rec in pairs(combos) do
  table.insert(combo_list, {
    key  = key,
    count = rec.count,
    mean  = rec.sum / rec.count
  })
end

table.sort(combo_list, function(a, b)
  return a.count > b.count
end)

print("== APACHE-10 ==")
print(string.format(
  "Combination                     Count in poor  Mean perf  Mean perf – overall"))
for i = 1, math.min(3, #combo_list) do
  local c = combo_list[i]
  print(string.format(
    "%-30s %3d           %.2f      %.2f",
    c.key, c.count, c.mean, c.mean - mean))
end
