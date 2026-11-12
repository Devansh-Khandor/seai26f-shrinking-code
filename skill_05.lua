#!/usr/bin/env lua
-- Skill 05: Diagnostic and Comparative Analysis
-- Usage: lua skill_05.lua <csv path>

local ezr = require("ezr")
local show, round, keysort = ezr.show, ezr.round, ezr.keysort

-- -----------------------------------------------------------------
-- Helper functions
-- -----------------------------------------------------------------
-- Find a column by its header name
local function col_by_name(tbl, name)
  for _,col in ipairs(tbl.cols.all) do
    if col.name == name then return col end
  end
  error("column " .. name .. " not found")
end

-- Normalisation helper (same as in skill_04)
local function norm(v, minv, maxv, reverse)
  if maxv == minv then return 0.5 end
  local n = (v - minv) / (maxv - minv)
  return reverse and 1 - n or n
end

-- Return all numeric columns (upper‑case first letter)
local function numeric_cols(tbl)
  local ncols = {}
  for _,col in ipairs(tbl.cols.x) do
    if col.name:find"^%u" and col.name ~= "Mpg+" then
      table.insert(ncols, col) end
  end
  for _,col in ipairs(tbl.cols.y) do
    if col.name:find"^%u" and col.name ~= "Mpg+" then
      table.insert(ncols, col) end
  end
  return ncols
end

-- Compute Pearson correlation of two numeric columns
local function corr(colx, coly, rows)
  local n, sx, sy, sxy, sx2, sy2 = 0,0,0,0,0,0
  for _,row in ipairs(rows) do
    local x = row[colx.at]
    local y = row[coly.at]
    if x~="?" and y~"?" then
      n = n + 1
      sx = sx + x
      sy = sy + y
      sxy = sxy + x*y
      sx2 = sx2 + x*x
      sy2 = sy2 + y*y
    end
  end
  if n < 2 then return 0 end
  local num = sxy - sx*sy/n
  local den = math.sqrt((sx2 - sx*sx/n) * (sy2 - sy*sy/n))
  return num/den
end

-- Compute mean mpg for each category of a symbolic column
local function cat_mean(col, mpg_col, rows)
  local groups = {}
  for _,row in ipairs(rows) do
    local key = row[col.at]
    local val = row[mpg_col.at]
    if key~="?" and val~"?" then
      if not groups[key] then groups[key] = {sum=0, n=0} end
      groups[key].sum = groups[key].sum + val
      groups[key].n = groups[key].n + 1
    end
  end
  for k,v in pairs(groups) do
    v.mean = v.sum / v.n
  end
  return groups
end

-- Compute simple mean of a numeric column over a list of rows
local function col_mean(col, rows)
  local sum, n = 0,0
  for _,row in ipairs(rows) do
    local v = row[col.at]
    if v~="?" then
      sum = sum + v
      n = n + 1
    end
  end
  return n>0 and sum/n or nil
end

-- -----------------------------------------------------------------
-- Load CSV and build table
-- -----------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_05.lua <csv path>\n")
  os.exit(1)
end
local tbl = ezr.Tbl(ezr.csv(path))

-- -----------------------------------------------------------------
-- CAR-09 : Features most associated with poor mpg
-- -----------------------------------------------------------------
print("== CAR-09 ==")

local mpg_col = col_by_name(tbl, "Mpg+")
local ncols = numeric_cols(tbl)

print("\nCorrelation of each numeric characteristic with MPG:")
for _,c in ipairs(ncols) do
  local r = corr(c, mpg_col, tbl.rows)
  print(string.format("  %-10s  corr = %+0.2f", c.name, round(r,2)))
end

print("\nCategories that deviate most from overall MPG mean:")
-- overall mean
local overall_mean = col_mean(mpg_col, tbl.rows)
local sym_cols = {}
for _,c in ipairs(tbl.cols.x) do
  if c.name:find"^%l" then table.insert(sym_cols, c) end
end
for _,c in ipairs(sym_cols) do
  local groups = cat_mean(c, mpg_col, tbl.rows)
  local best, worst = nil, nil
  local maxdiff, mindiff = -1, 1e9
  for k,v in pairs(groups) do
    local diff = v.mean - overall_mean
    if diff > maxdiff then maxdiff, best = diff, k end
    if diff < mindiff then mindiff, worst = diff, k end
  end
  print(string.format(
    "  %-10s  best=%s (+%0.2f), worst=%s (%+0.2f)",
    c.name, best, round(maxdiff,2), worst, round(mindiff,2)))
end

-- -----------------------------------------------------------------
-- CAR-10 : Common traits of top vs bottom performers
-- -----------------------------------------------------------------
print("\n== CAR-10 ==")

-- Score components: weight, mpg, acceleration
local wcol = col_by_name(tbl, "Lbs-")
local mcol = col_by_name(tbl, "Mpg+")
local acol = col_by_name(tbl, "Acc+")

-- Compute global min/max for normalization
local wmin,wmax = col_minmax(wcol, tbl.rows) or 0,0
local mmin,mmax = col_minmax(mcol, tbl.rows) or 0,0
local amin,amax = col_minmax(acol, tbl.rows) or 0,0

-- Helper to compute score for a single row
local function score(row)
  local w = row[wcol.at]
  local m = row[mcol.at]
  local a = row[acol.at]
  if w=="?" or m=="?" or a=="?" then return nil end
  local ws = 1 - norm(w,wmin,wmax)
  local ms = norm(m,mmin,mmax)
  local as = 1 - norm(a,amin,amax)
  return ws + ms + as
end

-- Build list of scored rows
local scored = {}
for _,row in ipairs(tbl.rows) do
  local s = score(row)
  if s then table.insert(scored, {row=row, score=s}) end
end

-- Sort by score descending
table.sort(scored, function(a,b) return a.score > b.score end)

local n = #scored
local group_size = math.max(1, math.floor(n*0.2))  -- 20%

local top_group = {}
for i=1,group_size do table.insert(top_group, scored[i].row) end
local bot_group = {}
for i=n-group_size+1,n do table.insert(bot_group, scored[i].row) end

-- Compute means of numeric columns for each group
local function group_means(group)
  local m = {}
  for _,col in ipairs(numeric_cols(tbl)) do
    m[col.name] = col_mean(col, group)
  end
  return m
end

local top_mean = group_means(top_group)
local bot_mean = group_means(bot_group)

print("\nTop 20% performers (by composite score):")
for name, val in pairs(top_mean) do
  print(string.format("  %-10s  mean = %0.2f", name, val))
end

print("\nBottom 20% performers:")
for name, val in pairs(bot_mean) do
  print(string.format("  %-10s  mean = %0.2f", name, val))
end

print("\nDifferences (top - bottom):")
for name, val in pairs(top_mean) do
  local diff = val - (bot_mean[name] or 0)
  print(string.format("  %-10s  diff = %0.2f", name, diff))
end

-- Helper to compute min/max for numeric columns
function col_minmax(col, rows)
  local minv, maxv = math.huge, -math.huge
  for _,row in ipairs(rows) do
    local v = row[col.at]
    if v~="?" then
      if v < minv then minv = v end
      if v > maxv then maxv = v end
    end
  end
  if minv == math.huge then return nil,nil end
  return minv, maxv
end
