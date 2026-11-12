#!/usr/bin/env lua
--[[
  skill_05.lua
  NASA-93 data analysis – feature importance for high effort & high defect projects.
  Usage:  lua skill_05.lua <csv-path>
  Prints:
    NASA-09: characteristics most associated with unusually high development effort.
    NASA-10: characteristics most associated with unusually high defect counts.
]]--

local ezr = require "ezr"

local function abort(msg)
  io.stderr:write(msg .. "\n")
  os.exit(1)
end

-- Helper to compute Pearson correlation
local function pearson(xs, ys)
  local n = #xs
  if n < 2 then return 0 end
  local sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
  for i = 1, n do
    local x = xs[i]
    local y = ys[i]
    sx  = sx + x
    sy  = sy + y
    sxx = sxx + x * x
    syy = syy + y * y
    sxy = sxy + x * y
  end
  local denom = math.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))
  if denom == 0 then return 0 end
  return (sxy - sx * sy / n) / denom
end

-- Build a list of numeric column names and indices
local function numeric_columns(tbl)
  local cols = {}
  for _, c in ipairs(tbl.cols.all) do
    local first = tbl.rows[1][c.at]
    if type(first) == "number" then
      table.insert(cols, {name=c.name, at=c.at, col=c})
    end
  end
  return cols
end

-- Compute average value for each category of a symbolic column
local function category_averages(tbl, cat_col, target_col)
  local sums, counts = {}, {}
  for _, row in ipairs(tbl.rows) do
    local cat = row[cat_col.at]
    local val = row[target_col.at]
    if cat ~= "?" and val ~= "?" then
      sums[cat] = (sums[cat] or 0) + val
      counts[cat] = (counts[cat] or 0) + 1
    end
  end
  local avgs = {}
  for cat, sum in pairs(sums) do
    avgs[cat] = sum / counts[cat]
  end
  return avgs
end

local csv_path = arg[1]
if not csv_path then abort("Usage: lua skill_05.lua <csv-path>") end

-- Load all rows
local rows = {}
for r in ezr.csv(csv_path) do table.insert(rows, r) end
if #rows == 0 then abort("No data found in " .. csv_path) end

-- Build table structure
local tbl = ezr.Tbl(rows)

-- Map columns by name
local col_map = {}
for i, c in ipairs(tbl.cols.all) do
  col_map[tbl.cols.names[i]] = c
end

-- Indices for target columns
local eff_col = col_map["Effort-"]
local def_col = col_map["Defects-"]

-- Prepare numeric columns list
local num_cols = numeric_columns(tbl)

-- Gather numeric data for effort and defects (exclude rows with missing)
local eff_xs, eff_ys = {}, {}
local def_xs, def_ys = {}, {}
for _, row in ipairs(tbl.rows) do
  local eff = row[eff_col.at]
  local def = row[def_col.at]
  if eff ~= "?" then
    table.insert(eff_ys, eff)
  end
  if def ~= "?" then
    table.insert(def_ys, def)
  end
end

-- Compute correlation of each numeric column with effort
local eff_corrs = {}
for _, col in ipairs(num_cols) do
  local xs = {}
  for _, row in ipairs(tbl.rows) do
    local val = row[col.at]
    local eff = row[eff_col.at]
    if val ~= "?" and eff ~= "?" then
      table.insert(xs, val)
    end
  end
  local corr = #xs > 0 and pearson(xs, eff_ys) or 0
  table.insert(eff_corrs, {name=col.name, corr=corr})
end
table.sort(eff_corrs, function(a,b) return math.abs(b.corr) < math.abs(a.corr) end)

-- Compute correlation of each numeric column with defects
local def_corrs = {}
for _, col in ipairs(num_cols) do
  local xs = {}
  for _, row in ipairs(tbl.rows) do
    local val = row[col.at]
    local def = row[def_col.at]
    if val ~= "?" and def ~= "?" then
      table.insert(xs, val)
    end
  end
  local corr = #xs > 0 and pearson(xs, def_ys) or 0
  table.insert(def_corrs, {name=col.name, corr=corr})
end
table.sort(def_corrs, function(a,b) return math.abs(b.corr) < math.abs(a.corr) end)

-- Find symbolic columns
local sym_cols = {}
for _, c in ipairs(tbl.cols.all) do
  if type(tbl.rows[1][c.at]) ~= "number" then
    table.insert(sym_cols, c)
  end
end

-- Compute average effort per category for symbolic cols
local eff_cat_avgs = {}
for _, c in ipairs(sym_cols) do
  eff_cat_avgs[c.name] = category_averages(tbl, c, eff_col)
end

-- Compute average defects per category
local def_cat_avgs = {}
for _, c in ipairs(sym_cols) do
  def_cat_avgs[c.name] = category_averages(tbl, c, def_col)
end

-- Output NASA-09
print("== NASA-09 ==")
print("Numeric features most strongly correlated with high Effort-:")
for i=1,math.min(3,#eff_corrs) do
  local r = eff_corrs[i]
  print(string.format("  %-12s : correlation %.3f", r.name, r.corr))
end
print("Symbolic features with highest average Effort-:")
for _, c in ipairs(sym_cols) do
  local avgs = eff_cat_avgs[c.name]
  local best_cat, best_avg = nil, -1
  for cat, avg in pairs(avgs) do
    if avg > best_avg then best_avg, best_cat = avg, cat end
  end
  if best_cat then
    print(string.format("  %-12s : %s = %.2f", c.name, best_cat, best_avg))
  end
end

-- Output NASA-10
print("\n== NASA-10 ==")
print("Numeric features most strongly correlated with high Defects-:")
for i=1,math.min(3,#def_corrs) do
  local r = def_corrs[i]
  print(string.format("  %-12s : correlation %.3f", r.name, r.corr))
end
print("Symbolic features with highest average Defects-:")
for _, c in ipairs(sym_cols) do
  local avgs = def_cat_avgs[c.name]
  local best_cat, best_avg = nil, -1
  for cat, avg in pairs(avgs) do
    if avg > best_avg then best_avg, best_cat = avg, cat end
  end
  if best_cat then
    print(string.format("  %-12s : %s = %.2f", c.name, best_cat, best_avg))
  end
end
