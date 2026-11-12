#!/usr/bin/env lua
-- skill_03.lua
-- CAR-05 & CAR-06: Statistical comparison of fuel efficiency
-- by cylinder count and by model‑year ranges.
--
-- Usage:  lua skill_03.lua <path-to-auto93.csv>
--
-- Requires: ezr (bundled in this repository)

local ezr = require("ezr")
local csv = ezr.csv

-- Helper: compute mean of numeric list
local function mean(vals)
  local s = 0
  for _, v in ipairs(vals) do s = s + v end
  return s / #vals
end

local function main(csv_path)
  -- Load data into a table
  local tbl = ezr.Tbl(csv(csv_path))

  -- Column positions (1‑based)
  local cyl_col  = tbl.cols.all[1].at   -- Clndrs
  local mpg_col  = tbl.cols.all[8].at   -- Mpg+

  ------------------------------------------------------------
  -- CAR‑05: Compare fuel efficiency by cylinder count
  ------------------------------------------------------------
  local cyl_groups = {}  -- cylinder -> {vals, mean}
  for _,row in ipairs(tbl.rows) do
    local mpg = row[mpg_col]
    if mpg ~= "?" and mpg ~= nil then
      local cyl = row[cyl_col]
      local g = cyl_groups[cyl]
      if not g then
        g = {vals = {}}
        cyl_groups[cyl] = g
      end
      table.insert(g.vals, mpg)
    end
  end

  -- Compute means
  for _,g in pairs(cyl_groups) do
    g.mean = mean(g.vals)
  end

  print("== CAR-05 ==")
  local cyl_keys = ezr.keys(cyl_groups)   -- sorted
  for i = 1, #cyl_keys - 1 do
    for j = i + 1, #cyl_keys do
      local a = cyl_groups[cyl_keys[i]]
      local b = cyl_groups[cyl_keys[j]]
      local diff = a.mean - b.mean
      local effect = ezr.cohen(a.vals, b.vals)
      local ksval = ezr.ks(ezr.sorted(a.vals), ezr.sorted(b.vals))
      local cliffsval = ezr.cliffs(ezr.sorted(a.vals), ezr.sorted(b.vals))
      local same_flag = ezr.same(ezr.sorted(a.vals), ezr.sorted(b.vals))
      local reliable = not same_flag
      print(string.format(
        "Cyl %s vs %s: mean %.2f vs %.2f, diff %.2f, effect %.2f, ks %.2f, cliffs %.2f, reliable %s",
        cyl_keys[i], cyl_keys[j], a.mean, b.mean, diff,
        effect, ksval, cliffsval, reliable and "YES" or "NO"))
    end
  end

  ------------------------------------------------------------
  -- CAR‑06: Compare fuel efficiency by model‑year ranges
  ------------------------------------------------------------
  local model_col = tbl.cols.all[4].at   -- Model
  local model_rows = {}
  for _,row in ipairs(tbl.rows) do
    local m = row[model_col]
    if type(m) == "number" then
      table.insert(model_rows, row)
    end
  end

  table.sort(model_rows, function(r1, r2)
    return r1[model_col] < r2[model_col]
  end)

  local mid = math.floor(#model_rows / 2)
  local g1, g2 = {vals = {}}, {vals = {}}

  for i = 1, mid do
    local mpg = model_rows[i][mpg_col]
    if mpg ~= "?" and mpg ~= nil then table.insert(g1.vals, mpg) end
  end
  for i = mid + 1, #model_rows do
    local mpg = model_rows[i][mpg_col]
    if mpg ~= "?" and mpg ~= nil then table.insert(g2.vals, mpg) end
  end

  if #g1.vals == 0 or #g2.vals == 0 then
    print("\n== CAR-06 ==")
    print("Insufficient data for model‑year comparison")
    return
  end

  g1.mean = mean(g1.vals)
  g2.mean = mean(g2.vals)
  local diff = g1.mean - g2.mean
  local effect = ezr.cohen(g1.vals, g2.vals)
  local ksval = ezr.ks(ezr.sorted(g1.vals), ezr.sorted(g2.vals))
  local cliffsval = ezr.cliffs(ezr.sorted(g1.vals), ezr.sorted(g2.vals))
  local same_flag = ezr.same(ezr.sorted(g1.vals), ezr.sorted(g2.vals))
  local reliable = not same_flag

  print("\n== CAR-06 ==")
  print(string.format(
    "Group 1 (lower %d models) mean mpg %.2f, n=%d",
    mid, g1.mean, #g1.vals))
  print(string.format(
    "Group 2 (higher %d models) mean mpg %.2f, n=%d",
    #model_rows - mid, g2.mean, #g2.vals))
  print(string.format(
    "Diff %.2f, effect %.2f, ks %.2f, cliffs %.2f, reliable %s",
    diff, effect, ksval, cliffsval,
    reliable and "YES" or "NO"))
end

if arg[1] then
  main(arg[1])
else
  io.stderr:write("Usage: lua skill_03.lua <csv_path>\n")
end
