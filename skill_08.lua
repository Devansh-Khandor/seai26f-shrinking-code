#!/usr/bin/env lua
-- skill_08.lua
-- Implements the requested APACHE-15 and APACHE-16 analyses.
--
-- Usage: lua skill_08.lua <csv path>
--
-- Requires the ezr library from the provided codebase.

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- Utility helpers (copied from earlier skills)

-- Round numbers using ezr's formatting
local function display(v)
  if type(v) == "number" then
    return ezr.round(v)
  else
    return tostring(v)
  end
end

-- Convert a row table to a CSV string
local function row_to_csv(row)
  local parts = {}
  for _,v in ipairs(row) do
    if v == "?" then
      table.insert(parts, "?")
    else
      table.insert(parts, tostring(v))
    end
  end
  return table.concat(parts, ",")
end

-- Count number of setting differences between two rows
local function diff_count(r1, r2, exclude_col)
  local d = 0
  for _,c in ipairs(r1.cols.all) do
    if c ~= exclude_col then
      local v1, v2 = r1[c.at], r2[c.at]
      if v1 ~= v2 and v1 ~= "?" and v2 ~= "?" then
        d = d + 1
      end
    end
  end
  return d
end

-- ------------------------------------------------------------------
-- Main logic

local csvfile = arg[1]
assert(csvfile, "Usage: lua skill_08.lua <csv path>")

-- Build table
local tbl = ezr.Tbl(ezr.csv(csvfile))
local perf_col = tbl.cols.y[1]   -- only goal column

-- Gather all performance values.
local perf_vals = ezr.map(tbl.rows, function(r) return r[perf_col.at] end)

-- Basic statistics
local mean_perf = perf_col:mid()

-- Determine threshold for poorly performing configurations
local function high_threshold(vals)
  local q1, q3 = ezr.quartiles(vals)
  local iqr = q3 - q1
  return q3 + 1.5 * iqr
end

local threshold = high_threshold(perf_vals)

-- Find a poor configuration (first one)
local poor_cfg = nil
for _,r in ipairs(tbl.rows) do
  if r[perf_col.at] > threshold then
    poor_cfg = r
    break
  end
end

if not poor_cfg then
  print("== APACHE-15 ==")
  print("No poorly performing configuration detected.")
  print()
  print("== APACHE-16 ==")
  print("No poorly performing configuration detected.")
  os.exit()
end

-- ------------------------------------------------------------------
-- APACHE-15: Suggest a small change to a poorly performing configuration

print("== APACHE-15 ==")
print(string.format("Poorly performing configuration found (performance=%s): %s",
      display(poor_cfg[perf_col.at]), row_to_csv(poor_cfg)))

-- Search for alternatives differing by at most 1 setting
local best_alt = nil
local best_diff = math.huge
local best_perf = math.huge

for _,r in ipairs(tbl.rows) do
  if r ~= poor_cfg then
    local d = diff_count(poor_cfg, r, perf_col)
    if d <= 1 and r[perf_col.at] < best_perf then
      best_alt, best_diff, best_perf = r, d, r[perf_col.at]
    end
  end
end

-- If no 1‑change alternative, look for 2‑change alternatives
if not best_alt then
  for _,r in ipairs(tbl.rows) do
    if r ~= poor_cfg then
      local d = diff_count(poor_cfg, r, perf_col)
      if d <= 2 and r[perf_col.at] < best_perf then
        best_alt, best_diff, best_perf = r, d, r[perf_col.at]
      end
    end
  end
end

if best_alt then
  print(string.format("Recommended alternative configuration (%d setting changes, performance=%s): %s",
        best_diff, display(best_perf), row_to_csv(best_alt)))
else
  print("No alternative configuration with <=2 setting changes found that improves performance.")
end
print()

-- ------------------------------------------------------------------
-- APACHE-16: Minimal changes to reach a target performance

print("== APACHE-16 ==")
-- Target performance: use the mean performance as a conservative goal.
local target_perf = mean_perf
print(string.format("Target performance (mean): %s", display(target_perf)))

-- Find all configurations meeting or better than target
local candidates = {}
for _,r in ipairs(tbl.rows) do
  if r[perf_col.at] <= target_perf then
    table.insert(candidates, r)
  end
end

if #candidates == 0 then
  print("No configuration meets the target performance.")
else
  -- Find the one requiring the fewest setting changes
  local best_cfg = nil
  local min_diff = math.huge
  local best_perf = math.huge
  for _,r in ipairs(candidates) do
    local d = diff_count(poor_cfg, r, perf_col)
    if d < min_diff or (d == min_diff and r[perf_col.at] < best_perf) then
      best_cfg, min_diff, best_perf = r, d, r[perf_col.at]
    end
  end

  print(string.format("Best alternative with minimal changes (%d changes, performance=%s): %s",
        min_diff, display(best_perf), row_to_csv(best_cfg)))
  print(string.format("Original poor configuration (performance=%s) differs by %d settings.",
        display(poor_cfg[perf_col.at]), min_diff))
end
