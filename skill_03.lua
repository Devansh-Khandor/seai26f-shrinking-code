#!/usr/bin/env lua
--  skill_03.lua – inventory statistical analysis (CAR‑05 & CAR‑06)
--  Usage: lua skill_03.lua <csv_path>
--
--  Depends on the ezr library (ezr.lua & ezr-lib.lua) in the
--  same directory or in the Lua package path.
--
--  CAR‑05: Compare fuel‑efficiency (Mpg+) across cylinder
--          counts (Clndrs column).  Use the `same()` test from
--          ezr to decide if differences are statistically
--          reliable.
--
--  CAR‑06: Compare overall performance across model‑year
--          ranges (Model column).  Again we use `same()` to
--          judge the evidence strength.

local ezr = require("ezr")

--------------------------------------------------------------------
-- Helpers
--------------------------------------------------------------------

-- Group rows by a column value
local function group_by(rows, col)
  local groups = {}
  for _, r in ipairs(rows) do
    local key = r[col.at]
    groups[key] = groups[key] or {}
    table.insert(groups[key], r)
  end
  return groups
end

-- Compute mean of a numeric list
local function mean(vals)
  local n = #vals
  if n == 0 then return 0 end
  local sum = 0
  for _, v in ipairs(vals) do sum = sum + v end
  return sum / n
end

-- Extract mpg values from a group of rows
local function mpg_values(rows)
  local mpgs = {}
  for _, r in ipairs(rows) do
    table.insert(mpgs, r[8])   -- Mpg+ is at index 8
  end
  return mpgs
end

-- Print a comparison summary
local function print_comparison(name1, mean1, name2, mean2, reliable)
  local status = reliable and "RELIABLE" or "UNRELIABLE"
  print(string.format(
    "%s vs %s: %s difference.  Means: %s = %s, %s = %s",
    name1, name2, status, name1, ezr.round(mean1),
    name2, ezr.round(mean2)))
end

--------------------------------------------------------------------
-- Load data
--------------------------------------------------------------------

local csv_path = arg[1]
if not csv_path then
  io.stderr:write("usage: lua skill_03.lua <csv_path>\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

--------------------------------------------------------------------
-- Identify relevant columns
--------------------------------------------------------------------

local Clndrs_col = tbl.cols.all[1]   -- cylinder count (Clndrs)
local Model_col  = tbl.cols.all[4]   -- model year (Model)
local rows       = tbl.rows

--------------------------------------------------------------------
-- CAR-05: Cylinder‑count vs MPG
--------------------------------------------------------------------

print("== CAR-05 ==")

-- Group rows by cylinder count
local cyl_groups = group_by(rows, Clndrs_col)

-- Prepare sorted mpg lists per group
local cyl_keys = {}
for k in pairs(cyl_groups) do table.insert(cyl_keys, k) end
table.sort(cyl_keys, function(a, b) return a < b end)

-- Pre‑compute sorted mpg lists and means
local cyl_mpg_sorted = {}
local cyl_mean       = {}
for _, key in ipairs(cyl_keys) do
  local mpgs = mpg_values(cyl_groups[key])
  cyl_mpg_sorted[key] = ezr.sorted(mpgs)
  cyl_mean[key] = mean(mpgs)
end

-- Compare every pair of cylinder groups
for i = 1, #cyl_keys do
  for j = i+1, #cyl_keys do
    local k1, k2 = cyl_keys[i], cyl_keys[j]
    local reliable = ezr.same(cyl_mpg_sorted[k1], cyl_mpg_sorted[k2])
    print_comparison(
      "Cyl " .. k1, cyl_mean[k1],
      "Cyl " .. k2, cyl_mean[k2],
      reliable)
  end
end

--------------------------------------------------------------------
-- CAR-06: Model‑year ranges vs performance
--------------------------------------------------------------------

print("\n== CAR-06 ==")

-- Group rows by model year
local yr_groups = group_by(rows, Model_col)

-- Prepare sorted mpg lists per year
local yr_keys = {}
for k in pairs(yr_groups) do table.insert(yr_keys, k) end
table.sort(yr_keys, function(a, b) return a < b end)

local yr_mpg_sorted = {}
local yr_mean       = {}
for _, key in ipairs(yr_keys) do
  local mpgs = mpg_values(yr_groups[key])
  yr_mpg_sorted[key] = ezr.sorted(mpgs)
  yr_mean[key] = mean(mpgs)
end

-- Compare every pair of year groups
for i = 1, #yr_keys do
  for j = i+1, #yr_keys do
    local k1, k2 = yr_keys[i], yr_keys[j]
    local reliable = ezr.same(yr_mpg_sorted[k1], yr_mpg_sorted[k2])
    print_comparison(
      "Year " .. k1, yr_mean[k1],
      "Year " .. k2, yr_mean[k2],
      reliable)
  end
end
