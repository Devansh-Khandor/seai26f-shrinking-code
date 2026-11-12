#!/usr/bin/env lua
-- Skill 03: Certify comparisons of fuel efficiency and performance
-- Usage: lua skill_03.lua <csv path>

local ezr = require("ezr")
local show, round, keysort, cohen, ks, cliffs, same, ranks =
      ezr.show, ezr.round, ezr.keysort, ezr.cohe, ezr.ks, ezr.cliff, ezr.same, ezr.ranks

-- Helper: find a column object by name
local function col_by_name(tbl, name)
  for _, col in ipairs(tbl.cols.all) do
    if col.name == name then return col end
  end
  error("column " .. name .. " not found")
end

-- Helper: group rows by a column's value
local function group_by_col(tbl, colname)
  local col = col_by_name(tbl, colname)
  local groups = {}
  for _, row in ipairs(tbl.rows) do
    local key = row[col.at]
    if key ~= "?" and key ~= nil then
      groups[key] = groups[key] or {}
      table.insert(groups[key], row)
    end
  end
  return groups
end

-- Helper: compute mean, sd and count for a numeric column in a group
local function stats(values)
  local n = #values
  if n == 0 then return {n=0, mean=0, sd=0} end
  local sum, sumsq = 0, 0
  for _, v in ipairs(values) do
    sum  = sum + v
    sumsq = sumsq + v*v
  end
  local mean = sum / n
  local var = (sumsq - n*mean*mean) / (n-1)
  local sd = var > 0 and math.sqrt(var) or 0
  return {n=n, mean=mean, sd=sd}
end

-- Helper: extract numeric values from a column for all rows in a group
local function get_col_values(rows, col)
  local vals = {}
  for _, row in ipairs(rows) do
    local v = row[col.at]
    if v ~= "?" and v ~= nil then
      table.insert(vals, v)
    end
  end
  return vals
end

-- Main
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_03.lua <csv path>\n")
  os.exit(1)
end
local tbl = ezr.Tbl(ezr.csv(path))

--------------------------------------------------------------------
-- CAR-05: Fuel efficiency vs. cylinder count
--------------------------------------------------------------------
print("== CAR-05 ==")

local cyl_col   = col_by_name(tbl, "Clndrs")
local mpg_col   = col_by_name(tbl, "Mpg+")
local cyl_groups = group_by_col(tbl, cyl_col.name)

-- Collect statistics per cylinder
local cyl_stats = {}
print(string.format("%-10s %-6s %-9s %-6s", "Cyl", "Count", "Mean MPG", "SD"))
for cyl, rows in pairs(cyl_groups) do
  local vals = get_col_values(rows, mpg_col)
  local s = stats(vals)
  cyl_stats[cyl] = {vals=vals, stats=s}
  print(string.format("%-10s %-6d %-9.2f %-6.2f",
        tostring(cyl), s.n, round(s.mean,2), round(s.sd,2)))
end

-- Pairwise comparison using `same`
print("\nPairwise cylinder comparison:")
local cyl_keys = keysort(keys(cyl_groups, ""), tonumber)
for i=1,#cyl_keys do
  for j=i+1,#cyl_keys do
    local a = cyl_keys[i]
    local b = cyl_keys[j]
    local sa = cyl_stats[a].vals
    local sb = cyl_stats[b].vals
    local reliable = not same(sa, sb)
    print(string.format("  %s vs %s: %s",
          a, b, reliable and "Reliable" or "Not reliable"))
  end
end

--------------------------------------------------------------------
-- CAR-06: Performance vs. model-year range
--------------------------------------------------------------------
print("\n== CAR-06 ==")

local model_col = col_by_name(tbl, "Model")
local model_groups = group_by_col(tbl, model_col.name)

-- Collect statistics per model (treated as year)
print(string.format("%-10s %-6s %-9s %-6s", "Model", "Count", "Mean MPG", "SD"))
local model_stats = {}
for model, rows in pairs(model_groups) do
  local vals = get_col_values(rows, mpg_col)
  local s = stats(vals)
  model_stats[model] = {vals=vals, stats=s}
  print(string.format("%-10s %-6d %-9.2f %-6.2f",
        tostring(model), s.n, round(s.mean,2), round(s.sd,2)))
end

-- Pairwise comparison
print("\nPairwise model-year comparison:")
local model_keys = keysort(keys(model_groups, ""), tonumber)
for i=1,#model_keys do
  for j=i+1,#model_keys do
    local a = model_keys[i]
    local b = model_keys[j]
    local sa = model_stats[a].vals
    local sb = model_stats[b].vals
    local reliable = not same(sa, sb)
    print(string.format("  %s vs %s: %s",
          a, b, reliable and "Reliable" or "Not reliable"))
  end
end
