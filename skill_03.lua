#!/usr/bin/env lua
-- Skill 03 – Statistical comparison of fuel efficiency and performance
-- Usage:  lua skill_03.lua <csv_path>

local ezr = require("ezr")
local round = ezr.round
local show  = ezr.show

-- ------------------------------------------------------------------
-- Utility: find a column object by name
local function get_col(tbl, name)
  for _, c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. tostring(name))
end

-- ------------------------------------------------------------------
-- Read data
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("Error: missing CSV path\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- CAR-05: Compare MPG across cylinder counts
print("== CAR-05 ==")

local cyl_col = get_col(tbl, "Clndrs")
local mpg_col = get_col(tbl, "Mpg+")

-- Build mapping cylinder -> list of mpg values
local cyl_mpg = {}
for _, row in ipairs(tbl.rows) do
  local cyl = row[cyl_col.at]
  local mpg = row[mpg_col.at]
  if mpg ~= "?" then
    cyl_mpg[cyl] = cyl_mpg[cyl] or {}
    table.insert(cyl_mpg[cyl], mpg)
  end
end

-- Keep only groups with at least 2 observations
local good_keys = {}
for k,_ in pairs(cyl_mpg) do
  if #cyl_mpg[k] >= 2 then
    table.insert(good_keys, k)
  end
end
table.sort(good_keys)

local function mean(arr)
  local s = 0
  for _,v in ipairs(arr) do s = s + v end
  return s / #arr
end

local function stddev(arr, m)
  if #arr < 2 then return 0 end
  local s = 0
  for _,v in ipairs(arr) do s = s + (v - m)^2 end
  return math.sqrt(s / (#arr - 1))
end

local lines_printed = 0
for i = 1, #good_keys do
  for j = i+1, #good_keys do
    local a = good_keys[i]
    local b = good_keys[j]
    local list_a = ezr.sorted(cyl_mpg[a])
    local list_b = ezr.sorted(cyl_mpg[b])
    local same = ezr.same(list_a, list_b)
    local reliable = not same
    local mean_a = mean(cyl_mpg[a])
    local mean_b = mean(cyl_mpg[b])
    print(string.format(
      "Cylinders %d vs %d: mean MPG = %.2f vs %.2f, reliable difference: %s",
      a, b, round(mean_a,2), round(mean_b,2), reliable and "yes" or "no"))
    lines_printed = lines_printed + 1
    if lines_printed >= 3 then break end
  end
  if lines_printed >= 3 then break end
end
if lines_printed == 0 then
  print("No cylinder groups with at least 2 MPG records.")
end

-- ------------------------------------------------------------------
-- CAR-06: Compare performance between two model‑year ranges
print("\n== CAR-06 ==")

local model_col = get_col(tbl, "Model")

-- Gather all years (numeric) and sort
local years = {}
for _, row in ipairs(tbl.rows) do
  local yr = row[model_col.at]
  if yr ~= "?" then
    table.insert(years, yr)
  end
end
table.sort(years)

-- Determine a split point: median year
local n_years = #years
local split
if n_years > 0 then
  split = years[math.floor(n_years / 2) + 1]
else
  split = nil
end

-- Build mpg lists for two groups
local mpg_low  = {}
local mpg_high = {}
for _, row in ipairs(tbl.rows) do
  local yr = row[model_col.at]
  local mpg = row[mpg_col.at]
  if yr ~= "?" and mpg ~= "?" then
    if split and yr <= split then
      table.insert(mpg_low, mpg)
    else
      table.insert(mpg_high, mpg)
    end
  end
end

-- Helper to compute stats
local function stats(arr)
  if #arr == 0 then return nil, nil, nil end
  local m = mean(arr)
  local s = stddev(arr, m)
  return m, s, #arr
end

local low_mean, low_std, low_n = stats(mpg_low)
local high_mean, high_std, high_n = stats(mpg_high)

if low_n == 0 or high_n == 0 then
  print("Insufficient data for one or both year groups.")
else
  local same = ezr.same(ezr.sorted(mpg_low), ezr.sorted(mpg_high))
  local reliable = not same

  print(string.format("Year range 1 (%d–%d): count=%d, mean MPG=%.2f, std=%.2f",
    years[1], split, low_n, round(low_mean,2), round(low_std,2)))
  print(string.format("Year range 2 (%d–%d): count=%d, mean MPG=%.2f, std=%.2f",
    split+1, years[#years], high_n, round(high_mean,2), round(high_std,2)))
  print(string.format("Reliable difference between ranges: %s",
    reliable and "yes" or "no"))
end
